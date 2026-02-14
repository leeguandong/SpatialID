"""
SpatialID IBench Test Script
基于 test_ibench.py 模板，使用 SpatialID 的空间自适应 ID 注入

核心改进（相比 AttrID v1）：
1. 对所有 prompt 生效（不依赖关键词匹配，覆盖率 100%）
2. 不修改 ArcFace 嵌入（避免 OOD 问题）
3. 通过空间掩码限制 ID 注入到人脸区域，非人脸区域自由跟随文本

测试流程：
1. 加载 chineseid (100 IDs) × human_longer_template (41 prompts) = 4100 张图
2. 使用 SpatialID 的空间自适应注入 + 时序调度
3. 保存结果 JSON 供 IBench 评估

Author: leeguandong
"""

import os
import re
import uuid
import json
import math
import time
from PIL import Image
from tqdm import tqdm
from pathlib import Path

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
import numpy as np
import cv2

from spatialid.models.flux.util import (
    SamplingOptions,
    load_ae,
    load_clip,
    load_flow_model,
    load_t5,
)
from spatialid.models.pulid.pipeline_flux import PuLIDPipeline
from spatialid.models.pulid.utils import resize_numpy_image_long

from spatialid import (
    SpatialPerceiverAttentionCA,
    TemporalSpatialScheduler,
    replace_pulid_ca_modules,
    spatialid_denoise,
)


# ============================================================================
# Sampling utilities (same as PuLID/flux/sampling.py)
# ============================================================================

def get_noise(num_samples, height, width, device, dtype, seed):
    return torch.randn(
        num_samples, 16,
        2 * math.ceil(height / 16), 2 * math.ceil(width / 16),
        device=device, dtype=dtype,
        generator=torch.Generator(device=device).manual_seed(seed),
    )

def prepare(t5, clip, img, prompt):
    bs, c, h, w = img.shape
    if bs == 1 and not isinstance(prompt, str):
        bs = len(prompt)
    img = rearrange(img, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
    if img.shape[0] == 1 and bs > 1:
        img = repeat(img, "1 ... -> bs ...", bs=bs)
    img_ids = torch.zeros(h // 2, w // 2, 3)
    img_ids[..., 1] = img_ids[..., 1] + torch.arange(h // 2)[:, None]
    img_ids[..., 2] = img_ids[..., 2] + torch.arange(w // 2)[None, :]
    img_ids = repeat(img_ids, "h w c -> b (h w) c", b=bs)
    if isinstance(prompt, str):
        prompt = [prompt]
    txt = t5(prompt)
    if txt.shape[0] == 1 and bs > 1:
        txt = repeat(txt, "1 ... -> bs ...", bs=bs)
    txt_ids = torch.zeros(bs, txt.shape[1], 3)
    vec = clip(prompt)
    if vec.shape[0] == 1 and bs > 1:
        vec = repeat(vec, "1 ... -> bs ...", bs=bs)
    return {
        "img": img, "img_ids": img_ids.to(img.device),
        "txt": txt.to(img.device), "txt_ids": txt_ids.to(img.device),
        "vec": vec.to(img.device),
    }

def time_shift(mu, sigma, t):
    return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)

def get_lin_function(x1=256, y1=0.5, x2=4096, y2=1.15):
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    return lambda x: m * x + b

def get_schedule(num_steps, image_seq_len, base_shift=0.5, max_shift=1.15, shift=True):
    timesteps = torch.linspace(1, 0, num_steps + 1)
    if shift:
        mu = get_lin_function(y1=base_shift, y2=max_shift)(image_seq_len)
        timesteps = time_shift(mu, 1.0, timesteps)
    return timesteps.tolist()

def unpack(x, height, width):
    return rearrange(
        x, "b (h w) (c ph pw) -> b c (h ph) (w pw)",
        h=math.ceil(height / 16), w=math.ceil(width / 16), ph=2, pw=2,
    )


# ============================================================================
# Utilities
# ============================================================================

def load_existing_results(path):
    if os.path.exists(path):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return {"datas": []}
    return {"datas": []}

def save_results(path, results):
    try:
        temp = path + ".tmp"
        with open(temp, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
        os.replace(temp, path)
    except Exception as e:
        print(f"Save error: {e}")

def process_prompts(prompt, gender):
    prompt = re.sub(r'\[.*?\]', '', prompt)
    mapping = {"female": "woman", "male": "man"}
    prompt = prompt.replace("{person}", mapping.get(gender, "person"))
    return prompt.strip()

def get_models(name, device, offload, fp8):
    t5 = load_t5(device, max_length=128)
    clip = load_clip(device)
    model = load_flow_model(name, device="cpu" if offload else device)
    model.eval()
    ae = load_ae(name, device="cpu" if offload else device)
    return model, ae, t5, clip


# ============================================================================
# SpatialID Generator for IBench
# ============================================================================

class SpatialIDIBenchGenerator:
    def __init__(self, config):
        self.device = torch.device(config["device"])
        self.offload = config["offload"]
        self.model_name = config["model_name"]

        print("Loading FLUX models...")
        self.model, self.ae, self.t5, self.clip = get_models(
            self.model_name, self.device, self.offload, config.get("fp8", False)
        )

        print("Loading PuLID pipeline...")
        self.pulid_model = PuLIDPipeline(
            self.model, device="cpu" if self.offload else self.device,
            weight_dtype=torch.bfloat16,
            onnx_provider=config.get("onnx_provider", "gpu"),
        )
        if self.offload:
            self.pulid_model.face_helper.face_det.mean_tensor = \
                self.pulid_model.face_helper.face_det.mean_tensor.to(torch.device("cuda"))
            self.pulid_model.face_helper.device = torch.device("cuda")

        self.pulid_model.load_pretrain(
            config["pretrained_model"],
            version=config.get("pulid_version", "v0.9.1"),
        )

        # SpatialID: replace pulid_ca modules with spatial-aware wrappers
        print("Initializing SpatialID modules...")
        replace_pulid_ca_modules(self.model)

        # Temporal-spatial scheduler
        self.spatial_scheduler = TemporalSpatialScheduler(
            early_threshold=config.get("early_threshold", 0.7),
            late_threshold=config.get("late_threshold", 0.3),
            late_floor=config.get("late_floor", 0.5),
            center_sigma=config.get("center_sigma", 0.3),
            global_floor=config.get("global_floor", 0.0),
        )
        print("SpatialID ready. Spatial masking active for ALL prompts.")

    @torch.no_grad()
    def generate_image(self, width, height, num_steps, start_step, guidance,
                       seed, prompt, id_image=None, id_weight=1.0,
                       neg_prompt="", true_cfg=1.0, timestep_to_start_cfg=1,
                       max_sequence_length=128):

        self.t5.max_length = max_sequence_length
        seed = int(seed) if seed != -1 else torch.seed() & 0xFFFFFFFF
        opts = SamplingOptions(
            prompt=prompt, width=width, height=height,
            num_steps=num_steps, guidance=guidance, seed=seed,
        )

        x = get_noise(1, opts.height, opts.width, self.device, torch.bfloat16, opts.seed)
        timesteps = get_schedule(opts.num_steps, x.shape[-1] * x.shape[-2] // 4, shift=True)

        # Compute patch grid dimensions for spatial masking
        h_patches = math.ceil(opts.height / 16)
        w_patches = math.ceil(opts.width / 16)

        if self.offload:
            self.t5, self.clip = self.t5.to(self.device), self.clip.to(self.device)
        inp = prepare(self.t5, self.clip, x, opts.prompt)
        use_true_cfg = abs(true_cfg - 1.0) > 1e-2
        inp_neg = prepare(self.t5, self.clip, x, neg_prompt) if use_true_cfg else None
        if self.offload:
            self.t5, self.clip = self.t5.cpu(), self.clip.cpu()
            torch.cuda.empty_cache()

        id_embeddings, uncond_id_embeddings = None, None

        if id_image is not None:
            if not isinstance(id_image, Image.Image):
                id_image = Image.fromarray(id_image) if isinstance(id_image, np.ndarray) else Image.open(id_image).convert("RGB")

            if self.offload:
                self.pulid_model.components_to_device(torch.device("cuda"))

            id_img_np = np.array(id_image)
            id_img_np = resize_numpy_image_long(id_img_np, 1024)

            # Standard PuLID ID embedding extraction (NO modification)
            id_embeddings, uncond_id_embeddings = self.pulid_model.get_id_embedding(
                id_img_np, cal_uncond=use_true_cfg
            )

            if self.offload:
                self.pulid_model.components_to_device(torch.device("cpu"))
                torch.cuda.empty_cache()

        # Denoise with SpatialID
        if self.offload:
            self.model = self.model.to(self.device)

        x = spatialid_denoise(
            self.model, **inp, timesteps=timesteps, guidance=opts.guidance,
            id_weight=id_weight, id=id_embeddings, start_step=start_step,
            uncond_id=uncond_id_embeddings, true_cfg=true_cfg,
            timestep_to_start_cfg=timestep_to_start_cfg,
            neg_txt=inp_neg["txt"] if use_true_cfg else None,
            neg_txt_ids=inp_neg["txt_ids"] if use_true_cfg else None,
            neg_vec=inp_neg["vec"] if use_true_cfg else None,
            aggressive_offload=config.get("aggressive_offload", False),
            # SpatialID params
            spatial_scheduler=self.spatial_scheduler,
            h_patches=h_patches,
            w_patches=w_patches,
        )

        if self.offload:
            self.model.cpu(); torch.cuda.empty_cache()
            self.ae.decoder.to(self.device)

        x = unpack(x.float(), opts.height, opts.width)
        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
            x = self.ae.decode(x)

        x = x.clamp(-1, 1)
        x = rearrange(x[0], "c h w -> h w c")
        img = Image.fromarray((127.5 * (x + 1.0)).cpu().byte().numpy())
        return img, str(opts.seed)


# ============================================================================
# Batch Generation
# ============================================================================

def generate_batch(config):
    with open(config["image_file"], 'r') as f:
        image_data = json.load(f)
    with open(config["prompt_file"], 'r') as f:
        prompt_data = json.load(f)

    save_path = config["save_path_template"].format(version=config["version"])
    Path(save_path).mkdir(parents=True, exist_ok=True)
    result_json_path = config["result_json_path"].format(version=config["version"])
    results = load_existing_results(result_json_path)
    # Ensure IBench-compatible format
    if "tags" not in results:
        results["tags"] = {"category": "imageid"}

    # Build set of already-done (id, prompt) pairs for resume
    done_set = set()
    for entry in results.get("datas", []):
        done_set.add((entry.get("id", ""), entry.get("prompt", "")))

    generator = SpatialIDIBenchGenerator(config)

    total_ids = len(image_data["datas"])
    total_prompts = len(prompt_data["datas"])
    print(f"Starting SpatialID IBench: {total_ids} IDs × {total_prompts} prompts = {total_ids * total_prompts} images")
    print(f"Already done: {len(done_set)}, remaining: {total_ids * total_prompts - len(done_set)}")

    global_num = len(results.get("datas", []))

    for img_entry in tqdm(image_data["datas"], desc="IDs"):
        try:
            id_path = img_entry["id"]
            if not os.path.exists(id_path):
                id_path = id_path.replace("/home/gdli7/", "/dev_share/gdli7/")
            if not os.path.exists(id_path):
                print(f"Image not found: {img_entry['id']}, skipping")
                continue

            id_img = Image.open(id_path).convert("RGB")
            gender = img_entry["gender"]

            for prompt_entry in prompt_data["datas"]:
                prompt = process_prompts(prompt_entry["prompt"], gender)

                if (img_entry["id"], prompt) in done_set:
                    continue

                img, seed = generator.generate_image(
                    width=config["width"], height=config["height"],
                    num_steps=config["num_steps"], start_step=config["start_step"],
                    guidance=config["guidance"], seed=config["seed"],
                    prompt=prompt, id_image=id_img, id_weight=config["id_weight"],
                    neg_prompt=config["neg_prompt"], true_cfg=config["true_cfg"],
                )

                save_name = f'{prompt.replace(" ", "_").replace(",","")[:40]}_{uuid.uuid4().hex[:6]}.png'
                full_path = os.path.join(save_path, save_name)
                Path(save_path).mkdir(parents=True, exist_ok=True)
                img.save(full_path)

                # IBench-compatible entry format
                results["datas"].append({
                    "num": global_num,
                    "id": img_entry["id"],
                    "prompt": prompt,
                    "prompt_attr": prompt_entry.get("prompt_attr", ["long"]),
                    "prompt_style": prompt_entry.get("prompt_style", ["human_longer"]),
                    "imagewithid": full_path,
                })
                global_num += 1
                save_results(result_json_path, results)

        except Exception as e:
            print(f"Error processing {img_entry.get('id', 'unknown')}: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    config = {
        # Data
        "image_file": "/dev_share/gdli7/IBench/data/images/chineseid_images.json",
        "prompt_file": "/dev_share/gdli7/IBench/data/prompts/human_longer_template.json",
        # Output
        "version": "v2",
        "save_path_template": "./results/spatialid_{version}",
        "result_json_path": "./results/spatialid_{version}.json",
        # Model
        "pretrained_model": "/dev_share/gdli7/models/pulid/pulid_flux_v0.9.1.safetensors",
        "model_name": "flux-dev",
        "device": "cuda:2",
        "offload": False,
        "onnx_provider": "gpu",
        # SpatialID scheduler params (Config F - best trade-off)
        "early_threshold": 0.7,
        "late_threshold": 0.3,
        "late_floor": 0.8,       # 从 0.5 提升到 0.8
        "center_sigma": 0.7,     # 从 0.3 提升到 0.7
        "global_floor": 0.3,     # 新增全局下限
        # Generation
        "seed": 42,
        "width": 896,
        "height": 1152,
        "num_steps": 25,
        "start_step": 0,
        "guidance": 4.0,
        "id_weight": 1.0,
        "true_cfg": 1.0,
        "neg_prompt": "bad quality",
    }

    try:
        generate_batch(config)
        print("SpatialID IBench generation finished.")
    except Exception as e:
        print(f"Critical Error: {e}")
        import traceback
        traceback.print_exc()
