"""
SpatialID 消融实验脚本
测试不同参数配置对 FaceSim / ClipT 的影响

5 IDs × 5 prompts × N configs
内联计算 FaceSim 和 ClipT，无需 IBench
"""

import os
import re
import json
import math
import time
from PIL import Image
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
from einops import rearrange, repeat
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
    TemporalSpatialScheduler,
    replace_pulid_ca_modules,
    spatialid_denoise,
)


# ============================================================================
# Sampling utilities
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


# ============================================================================
# Inline metrics
# ============================================================================

class InlineMetrics:
    """Compute FaceSim and ClipT without IBench."""

    def __init__(self, device):
        self.device = device
        self._face_app = None
        self._clip_model = None
        self._clip_preprocess = None
        self._clip_tokenizer = None

    def _init_face(self):
        if self._face_app is not None:
            return
        from insightface.app import FaceAnalysis
        self._face_app = FaceAnalysis(
            name="buffalo_l",
            root="/dev_share/gdli7/models/insightface",
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        self._face_app.prepare(ctx_id=0, det_size=(640, 640))

    def _init_clip(self):
        if self._clip_model is not None:
            return
        from transformers import CLIPProcessor, CLIPModel
        clip_path = "/dev_share/gdli7/models/clip/clip-vit-large-patch14"
        self._clip_model = CLIPModel.from_pretrained(clip_path).to(self.device)
        self._clip_processor = CLIPProcessor.from_pretrained(clip_path)
        self._clip_model.eval()

    def face_embedding(self, img_path_or_np):
        """Extract ArcFace embedding from image."""
        self._init_face()
        if isinstance(img_path_or_np, str):
            img = cv2.imread(img_path_or_np)
        else:
            img = img_path_or_np
        if img.shape[2] == 4:
            img = img[:, :, :3]
        # BGR for insightface
        if len(img.shape) == 3 and img.shape[2] == 3:
            faces = self._face_app.get(img)
            if len(faces) == 0:
                return None
            # Pick largest face
            face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]))
            return face.normed_embedding
        return None

    def face_sim(self, emb1, emb2):
        """Cosine similarity between two face embeddings."""
        if emb1 is None or emb2 is None:
            return float('nan')
        return float(np.dot(emb1, emb2))

    def clip_text_score(self, img_pil, text):
        """CLIP text-image similarity."""
        self._init_clip()
        inputs = self._clip_processor(text=[text], images=img_pil, return_tensors="pt", padding=True, truncation=True, max_length=77)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self._clip_model(**inputs)
            img_feat = outputs.image_embeds
            txt_feat = outputs.text_embeds
            img_feat = F.normalize(img_feat, dim=-1)
            txt_feat = F.normalize(txt_feat, dim=-1)
            sim = (img_feat @ txt_feat.T).item()
        return sim


# ============================================================================
# Ablation configs
# ============================================================================

ABLATION_CONFIGS = {
    "A_baseline_pulid": {
        "desc": "PuLID baseline (no spatial mask)",
        "use_spatial": False,
    },
    "B_current_v1": {
        "desc": "Current SpatialID v1",
        "use_spatial": True,
        "early_threshold": 0.7, "late_threshold": 0.3,
        "late_floor": 0.5, "center_sigma": 0.3, "global_floor": 0.0,
    },
    "C_higher_floor": {
        "desc": "late_floor=0.7, center_sigma=0.5",
        "use_spatial": True,
        "early_threshold": 0.7, "late_threshold": 0.3,
        "late_floor": 0.7, "center_sigma": 0.5, "global_floor": 0.0,
    },
    "D_global_floor_03": {
        "desc": "global_floor=0.3",
        "use_spatial": True,
        "early_threshold": 0.7, "late_threshold": 0.3,
        "late_floor": 0.5, "center_sigma": 0.3, "global_floor": 0.3,
    },
    "E_global_floor_05": {
        "desc": "global_floor=0.5",
        "use_spatial": True,
        "early_threshold": 0.7, "late_threshold": 0.3,
        "late_floor": 0.5, "center_sigma": 0.3, "global_floor": 0.5,
    },
    "F_relaxed": {
        "desc": "late_floor=0.8, center_sigma=0.7, global_floor=0.3",
        "use_spatial": True,
        "early_threshold": 0.7, "late_threshold": 0.3,
        "late_floor": 0.8, "center_sigma": 0.7, "global_floor": 0.3,
    },
}


# ============================================================================
# Main
# ============================================================================

def main():
    device = torch.device("cuda:2")
    print(f"Using device: {device}")

    # ---- Data ----
    with open("/dev_share/gdli7/IBench/data/images/chineseid_images.json") as f:
        all_ids = json.load(f)["datas"]
    id_entries = all_ids[:5]

    with open("/dev_share/gdli7/IBench/data/prompts/human_longer_template.json") as f:
        all_prompts = json.load(f)["datas"]
    # Pick diverse prompts: scene, action, attribute, profession, sport
    prompt_indices = [0, 1, 5, 10, 20]
    prompt_entries = [all_prompts[i] for i in prompt_indices if i < len(all_prompts)]

    # ---- Load models ----
    print("Loading T5 & CLIP...")
    t5 = load_t5(device, max_length=128)
    clip_enc = load_clip(device)

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

    # Replace CA modules (needed for spatial configs)
    replace_pulid_ca_modules(model)

    # ---- Metrics ----
    metrics = InlineMetrics(device)

    # ---- Generation params ----
    width, height = 896, 1152
    num_steps = 25
    seed = 42
    guidance = 4.0
    id_weight = 1.0
    h_patches = math.ceil(height / 16)
    w_patches = math.ceil(width / 16)

    # ---- Pre-extract ID embeddings and face embeddings ----
    print("\nExtracting ID embeddings...")
    id_data = []
    for entry in id_entries:
        id_path = entry["id"].replace("/home/gdli7/", "/dev_share/gdli7/")
        if not os.path.exists(id_path):
            print(f"  [SKIP] {id_path}")
            continue
        id_img = Image.open(id_path).convert("RGB")
        id_img_np = np.array(id_img)
        id_img_np_resized = resize_numpy_image_long(id_img_np, 1024)

        with torch.no_grad():
            id_embeddings, _ = pulid_model.get_id_embedding(id_img_np_resized, cal_uncond=False)

        # Face embedding for FaceSim
        id_face_emb = metrics.face_embedding(cv2.cvtColor(id_img_np, cv2.COLOR_RGB2BGR))

        id_data.append({
            "entry": entry,
            "path": id_path,
            "embeddings": id_embeddings,
            "face_emb": id_face_emb,
        })
        print(f"  ID {entry['num']}: face={'OK' if id_face_emb is not None else 'FAIL'}")

    # ---- Prepare prompts ----
    print(f"\nPrompts ({len(prompt_entries)}):")
    for p in prompt_entries:
        print(f"  [{p['num']}] {p['prompt'][:80]}...")

    # ---- Run ablation ----
    out_dir = Path("./results/ablation")
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    for cfg_name, cfg in ABLATION_CONFIGS.items():
        print(f"\n{'='*70}")
        print(f"Config: {cfg_name} — {cfg['desc']}")
        print(f"{'='*70}")

        if cfg["use_spatial"]:
            scheduler = TemporalSpatialScheduler(
                early_threshold=cfg["early_threshold"],
                late_threshold=cfg["late_threshold"],
                late_floor=cfg["late_floor"],
                center_sigma=cfg["center_sigma"],
                global_floor=cfg.get("global_floor", 0.0),
            )
        else:
            scheduler = None

        face_sims = []
        clip_ts = []
        cfg_dir = out_dir / cfg_name
        cfg_dir.mkdir(parents=True, exist_ok=True)

        total = len(id_data) * len(prompt_entries)
        count = 0

        for idd in id_data:
            entry = idd["entry"]
            gender = entry["gender"]

            for p_entry in prompt_entries:
                raw_prompt = p_entry["prompt"]
                prompt = re.sub(r'\[.*?\]', '', raw_prompt)
                mapping = {"female": "woman", "male": "man"}
                prompt = prompt.replace("{person}", mapping.get(gender, "person")).strip()

                count += 1
                t0 = time.time()

                with torch.no_grad():
                    x = get_noise(1, height, width, device, torch.bfloat16, seed)
                    timesteps = get_schedule(num_steps, x.shape[-1] * x.shape[-2] // 4, shift=True)
                    t5.max_length = 128
                    inp = prepare(t5, clip_enc, x, prompt)

                    if scheduler is not None:
                        x = spatialid_denoise(
                            model, **inp, timesteps=timesteps, guidance=guidance,
                            id_weight=id_weight, id=idd["embeddings"], start_step=0,
                            uncond_id=None, true_cfg=1.0,
                            timestep_to_start_cfg=1,
                            neg_txt=None, neg_txt_ids=None, neg_vec=None,
                            aggressive_offload=False,
                            spatial_scheduler=scheduler,
                            h_patches=h_patches, w_patches=w_patches,
                        )
                    else:
                        # PuLID baseline: use spatialid_denoise without scheduler
                        x = spatialid_denoise(
                            model, **inp, timesteps=timesteps, guidance=guidance,
                            id_weight=id_weight, id=idd["embeddings"], start_step=0,
                            uncond_id=None, true_cfg=1.0,
                            timestep_to_start_cfg=1,
                            neg_txt=None, neg_txt_ids=None, neg_vec=None,
                            aggressive_offload=False,
                            spatial_scheduler=None,
                            h_patches=h_patches, w_patches=w_patches,
                        )

                    x = unpack(x.float(), height, width)
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        x = ae.decode(x)

                x = x.clamp(-1, 1)
                x = rearrange(x[0], "c h w -> h w c")
                img_pil = Image.fromarray((127.5 * (x + 1.0)).cpu().byte().numpy())

                # Save
                fname = f"id{entry['num']}_p{p_entry['num']}.png"
                img_pil.save(cfg_dir / fname)

                # Compute metrics
                gen_np = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
                gen_face_emb = metrics.face_embedding(gen_np)
                fsim = metrics.face_sim(idd["face_emb"], gen_face_emb)
                ct = metrics.clip_text_score(img_pil, prompt)

                face_sims.append(fsim)
                clip_ts.append(ct)

                elapsed = time.time() - t0
                print(f"  [{count}/{total}] FaceSim={fsim:.3f} ClipT={ct:.3f} ({elapsed:.1f}s) | {prompt[:50]}...")

        # Aggregate
        valid_fsim = [x for x in face_sims if not math.isnan(x)]
        avg_fsim = np.mean(valid_fsim) if valid_fsim else float('nan')
        avg_ct = np.mean(clip_ts)

        results[cfg_name] = {
            "desc": cfg["desc"],
            "avg_facesim": float(avg_fsim),
            "avg_clipt": float(avg_ct),
            "n_valid_faces": len(valid_fsim),
            "n_total": len(face_sims),
        }

        print(f"\n  >> {cfg_name}: FaceSim={avg_fsim:.4f} ClipT={avg_ct:.4f} (faces: {len(valid_fsim)}/{len(face_sims)})")

    # ---- Summary ----
    print(f"\n\n{'='*70}")
    print("ABLATION RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"{'Config':<25} {'FaceSim':>10} {'ClipT':>10} {'Faces':>8}")
    print("-" * 55)
    for name, r in results.items():
        print(f"{name:<25} {r['avg_facesim']:>10.4f} {r['avg_clipt']:>10.4f} {r['n_valid_faces']:>5}/{r['n_total']}")

    # Save results
    with open(out_dir / "ablation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_dir / 'ablation_results.json'}")


if __name__ == "__main__":
    main()
