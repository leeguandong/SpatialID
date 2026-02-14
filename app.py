"""SpatialID Gradio Demo: Spatially-Adaptive Identity Injection.

Launch with:
    python app.py --device cuda:0
"""

import os
import argparse
import math
import logging
from datetime import datetime

import gradio as gr
import numpy as np
from PIL import Image
import torch
from einops import rearrange, repeat

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

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


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
# Global state
# ============================================================================

class SpatialIDGenerator:
    def __init__(self):
        self.model = None
        self.ae = None
        self.t5 = None
        self.clip = None
        self.pulid_model = None
        self.spatial_scheduler = None
        self.device = None
        self.initialized = False

    def initialize(self, device="cuda:0", pretrained_model=None):
        if self.initialized:
            return

        self.device = torch.device(device)
        logger.info(f"Initializing SpatialID on {self.device}...")

        logger.info("Loading T5 & CLIP...")
        self.t5 = load_t5(self.device, max_length=128)
        self.clip = load_clip(self.device)

        logger.info("Loading FLUX model...")
        self.model = load_flow_model("flux-dev", device=self.device)
        self.model.eval()

        logger.info("Loading VAE...")
        self.ae = load_ae("flux-dev", device=self.device)

        logger.info("Loading PuLID pipeline...")
        self.pulid_model = PuLIDPipeline(
            self.model, device=self.device,
            weight_dtype=torch.bfloat16,
            onnx_provider='gpu'
        )

        if pretrained_model is None:
            pretrained_model = "/dev_share/gdli7/models/pulid/pulid_flux_v0.9.1.safetensors"
        self.pulid_model.load_pretrain(pretrained_model, version="v0.9.1")

        logger.info("Setting up SpatialID...")
        replace_pulid_ca_modules(self.model)
        self.spatial_scheduler = TemporalSpatialScheduler(
            early_threshold=0.7,
            late_threshold=0.3,
            late_floor=0.5,
            center_sigma=0.3,
        )

        self.initialized = True
        logger.info("SpatialID ready!")

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        id_image: Image.Image,
        width: int = 896,
        height: int = 1152,
        num_steps: int = 25,
        guidance: float = 4.0,
        id_weight: float = 1.0,
        seed: int = -1,
        batch_size: int = 1,
    ):
        if not self.initialized:
            raise RuntimeError("Generator not initialized. Call initialize() first.")

        base_seed = int(seed) if seed != -1 else torch.seed() & 0xFFFFFFFF

        h_patches = math.ceil(height / 16)
        w_patches = math.ceil(width / 16)

        # Extract ID embedding once (shared across batch)
        id_embeddings = None
        if id_image is not None:
            id_img_np = np.array(id_image)
            id_img_np = resize_numpy_image_long(id_img_np, 1024)
            id_embeddings, _ = self.pulid_model.get_id_embedding(id_img_np, cal_uncond=False)

        results = []
        seeds_used = []
        for i in range(batch_size):
            cur_seed = (base_seed + i) & 0xFFFFFFFF

            x = get_noise(1, height, width, self.device, torch.bfloat16, cur_seed)
            timesteps = get_schedule(num_steps, x.shape[-1] * x.shape[-2] // 4, shift=True)

            self.t5.max_length = 128
            inp = prepare(self.t5, self.clip, x, prompt)

            x = spatialid_denoise(
                self.model, **inp, timesteps=timesteps, guidance=guidance,
                id_weight=id_weight, id=id_embeddings, start_step=0,
                uncond_id=None, true_cfg=1.0,
                timestep_to_start_cfg=1,
                neg_txt=None, neg_txt_ids=None, neg_vec=None,
                aggressive_offload=False,
                spatial_scheduler=self.spatial_scheduler,
                h_patches=h_patches, w_patches=w_patches,
            )

            x = unpack(x.float(), height, width)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                x = self.ae.decode(x)

            x = x.clamp(-1, 1)
            x = rearrange(x[0], "c h w -> h w c")
            img = Image.fromarray((127.5 * (x + 1.0)).cpu().byte().numpy())
            results.append(img)
            seeds_used.append(cur_seed)

        return results, seeds_used


# Global generator instance
generator = SpatialIDGenerator()


# ============================================================================
# Gradio interface
# ============================================================================

def generate_image(
    id_image,
    prompt,
    width,
    height,
    num_steps,
    guidance,
    id_weight,
    seed,
    batch_size,
):
    if id_image is None:
        return None, "Please upload an ID image."

    if not prompt.strip():
        return None, "Please enter a prompt."

    try:
        id_pil = Image.fromarray(id_image).convert("RGB")

        results, seeds_used = generator.generate(
            prompt=prompt,
            id_image=id_pil,
            width=int(width),
            height=int(height),
            num_steps=int(num_steps),
            guidance=float(guidance),
            id_weight=float(id_weight),
            seed=int(seed),
            batch_size=int(batch_size),
        )

        # Save as PNG
        output_dir = os.path.join(os.path.dirname(__file__), "outputs")
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        saved_paths = []
        for i, (img, s) in enumerate(zip(results, seeds_used)):
            fname = f"{timestamp}_seed{s}.png"
            fpath = os.path.join(output_dir, fname)
            img.save(fpath, format="PNG")
            saved_paths.append(fpath)

        seeds_str = ", ".join(str(s) for s in seeds_used)
        return results, f"Generated {len(results)} image(s) | Seeds: {seeds_str} | Saved to outputs/"

    except Exception as e:
        logger.exception("Generation failed")
        return None, f"Error: {str(e)}"


def create_demo():
    with gr.Blocks(title="SpatialID Demo", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # SpatialID: Spatially-Adaptive Identity Injection

        Upload a reference face image and enter a text prompt. SpatialID will generate an image
        that preserves the identity in face regions while allowing non-face regions (background,
        clothing, scene) to freely follow the text prompt.
        """)

        with gr.Row():
            with gr.Column(scale=1):
                id_image = gr.Image(label="Reference ID Image", type="numpy")
                prompt = gr.Textbox(
                    label="Prompt",
                    placeholder="A portrait of this person as an astronaut on Mars...",
                    lines=3,
                )

                with gr.Accordion("Advanced Settings", open=False):
                    with gr.Row():
                        width = gr.Slider(512, 1536, value=896, step=64, label="Width")
                        height = gr.Slider(512, 1536, value=1152, step=64, label="Height")
                    with gr.Row():
                        num_steps = gr.Slider(10, 50, value=25, step=1, label="Steps")
                        guidance = gr.Slider(1.0, 10.0, value=4.0, step=0.5, label="Guidance")
                    with gr.Row():
                        id_weight = gr.Slider(0.0, 2.0, value=1.0, step=0.1, label="ID Weight")
                        seed = gr.Number(value=-1, label="Seed (-1 for random)")
                    batch_size = gr.Slider(1, 8, value=1, step=1, label="Batch Size")

                generate_btn = gr.Button("Generate", variant="primary")

            with gr.Column(scale=1):
                output_gallery = gr.Gallery(label="Generated Images", columns=2, height="auto")
                status = gr.Textbox(label="Status", interactive=False)

        # Examples
        gr.Examples(
            examples=[
                ["A portrait of this person as an astronaut floating in space, Earth visible in background"],
                ["This person as a medieval knight in shining armor, castle in background"],
                ["A professional headshot of this person in a modern office setting"],
                ["This person enjoying coffee at a cozy Parisian café, Eiffel Tower visible through window"],
                ["An oil painting portrait of this person in Renaissance style"],
            ],
            inputs=[prompt],
            label="Example Prompts",
        )

        generate_btn.click(
            fn=generate_image,
            inputs=[id_image, prompt, width, height, num_steps, guidance, id_weight, seed, batch_size],
            outputs=[output_gallery, status],
        )

    return demo


def main():
    parser = argparse.ArgumentParser(description="SpatialID Gradio Demo")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use")
    parser.add_argument("--pretrained_model", type=str, default=None, help="Path to PuLID checkpoint")
    parser.add_argument("--port", type=int, default=49019, help="Port to run on")
    parser.add_argument("--share", action="store_true", help="Create public link")
    args = parser.parse_args()

    # Initialize generator
    generator.initialize(device=args.device, pretrained_model=args.pretrained_model)

    # Create and launch demo
    demo = create_demo()
    demo.launch(server_name="0.0.0.0",server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
