"""
SpatialID 快速验证脚本
5 IDs × 3 prompts = 15 张图
"""

import os
import json
import math
import time
from PIL import Image
from pathlib import Path

import torch
from einops import rearrange, repeat
import numpy as np

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
    TemporalSpatialScheduler,
    replace_pulid_ca_modules,
    spatialid_denoise,
)

# Sampling utilities
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
    img_ids[..., 1] += torch.arange(h // 2)[:, None]
    img_ids[..., 2] += torch.arange(w // 2)[None, :]
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


def main():
    device = torch.device("cuda:2")
    print(f"Using device: {device}")

    # ---- 5 IDs ----
    with open("/dev_share/gdli7/IBench/data/images/chineseid_images.json") as f:
        all_ids = json.load(f)["datas"]
    id_entries = all_ids[:5]

    # ---- 3 prompts (选不同类型: 场景/动作/属性) ----
    with open("/dev_share/gdli7/IBench/data/prompts/human_longer_template.json") as f:
        all_prompts = json.load(f)["datas"]
    prompt_entries = [all_prompts[0], all_prompts[1], all_prompts[5]]

    # ---- Load models ----
    print("Loading T5 & CLIP...")
    t5 = load_t5(device, max_length=128)
    clip = load_clip(device)

    print("Loading FLUX model...")
    model = load_flow_model("flux-dev", device=device)
    model.eval()

    print("Loading VAE...")
    ae = load_ae("flux-dev", device=device)

    print("Loading PuLID pipeline...")
    pulid_model = PuLIDPipeline(model, device=device, weight_dtype=torch.bfloat16, onnx_provider='gpu')
    pulid_model.load_pretrain(
        "/dev_share/gdli7/models/pulid/pulid_flux_v0.9.1.safetensors",
        version="v0.9.1",
    )

    # ---- SpatialID setup ----
    print("Setting up SpatialID...")
    replace_pulid_ca_modules(model)
    scheduler = TemporalSpatialScheduler(
        early_threshold=0.7, late_threshold=0.3,
        late_floor=0.5, center_sigma=0.3,
    )

    # ---- Output dir ----
    out_dir = Path("./results/spatialid_quick_test")
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Generation params ----
    width, height = 896, 1152
    num_steps = 25
    seed = 42
    guidance = 4.0
    id_weight = 1.0

    h_patches = math.ceil(height / 16)
    w_patches = math.ceil(width / 16)

    total = len(id_entries) * len(prompt_entries)
    count = 0

    for id_entry in id_entries:
        id_path = id_entry["id"].replace("/home/gdli7/", "/dev_share/gdli7/")
        gender = id_entry["gender"]

        if not os.path.exists(id_path):
            print(f"  [SKIP] Image not found: {id_path}")
            continue

        id_img = Image.open(id_path).convert("RGB")
        id_img_np = np.array(id_img)
        id_img_np = resize_numpy_image_long(id_img_np, 1024)

        print(f"\n{'='*60}")
        print(f"ID: {os.path.basename(id_path)} ({gender})")

        # Extract ID embedding (standard PuLID, no modification)
        with torch.no_grad():
            id_embeddings, _ = pulid_model.get_id_embedding(id_img_np, cal_uncond=False)

        for p_entry in prompt_entries:
            raw_prompt = p_entry["prompt"]
            # Replace {person} with gender
            import re
            prompt = re.sub(r'\[.*?\]', '', raw_prompt)
            mapping = {"female": "woman", "male": "man"}
            prompt = prompt.replace("{person}", mapping.get(gender, "person"))
            prompt = prompt.strip()

            count += 1
            print(f"\n  [{count}/{total}] Prompt: {prompt[:80]}...")

            t0 = time.time()

            with torch.no_grad():
                x = get_noise(1, height, width, device, torch.bfloat16, seed)
                timesteps = get_schedule(num_steps, x.shape[-1] * x.shape[-2] // 4, shift=True)

                t5.max_length = 128
                inp = prepare(t5, clip, x, prompt)

                x = spatialid_denoise(
                    model, **inp, timesteps=timesteps, guidance=guidance,
                    id_weight=id_weight, id=id_embeddings, start_step=0,
                    uncond_id=None, true_cfg=1.0,
                    timestep_to_start_cfg=1,
                    neg_txt=None, neg_txt_ids=None, neg_vec=None,
                    aggressive_offload=False,
                    spatial_scheduler=scheduler,
                    h_patches=h_patches, w_patches=w_patches,
                )

                x = unpack(x.float(), height, width)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    x = ae.decode(x)

            x = x.clamp(-1, 1)
            x = rearrange(x[0], "c h w -> h w c")
            img = Image.fromarray((127.5 * (x + 1.0)).cpu().byte().numpy())

            elapsed = time.time() - t0
            fname = f"id{id_entry['num']}_{gender}_p{p_entry['num']}.png"
            img.save(out_dir / fname)
            print(f"  Saved: {fname} ({elapsed:.1f}s)")

    print(f"\n{'='*60}")
    print(f"Done! {count} images saved to {out_dir}")


if __name__ == "__main__":
    main()
