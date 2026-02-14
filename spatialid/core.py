"""SpatialID: Training-Free Spatially-Adaptive Identity Injection.

Core idea: PuLID injects ID features uniformly across all image patches,
including background, clothing, and scene regions. SpatialID computes a
spatial relevance mask from the cross-attention output and restricts ID
injection to face-relevant patches only. Non-face regions are free to
follow the text prompt.

Modules:
    SpatialPerceiverAttentionCA — wrapper that returns both output and attention weights
    SpatialMaskExtractor — extracts spatial relevance mask from CA output/weights
    MaskRefiner — Gaussian smoothing + thresholding + dilation
    TemporalSpatialScheduler — time-step adaptive mask strategy
    spatialid_denoise() — modified denoise loop with spatial masking
"""

import math
import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ============================================================================
# 1. SpatialPerceiverAttentionCA — wrapper returning attention weights
# ============================================================================

def reshape_tensor(x, heads):
    bs, length, width = x.shape
    x = x.view(bs, length, heads, -1)
    x = x.transpose(1, 2)
    x = x.reshape(bs, heads, length, -1)
    return x


class SpatialPerceiverAttentionCA(nn.Module):
    """Drop-in replacement for PerceiverAttentionCA that also returns
    the cross-attention output *before* to_out projection, enabling
    spatial mask extraction.

    The original module is not modified — we copy its weights at init time.
    """

    def __init__(self, original_ca: nn.Module):
        super().__init__()
        # Share the same parameters (no copy, just reference)
        self.norm1 = original_ca.norm1
        self.norm2 = original_ca.norm2
        self.to_q = original_ca.to_q
        self.to_kv = original_ca.to_kv
        self.to_out = original_ca.to_out
        self.heads = original_ca.heads
        self.dim_head = original_ca.dim_head
        self.scale = original_ca.scale

    def forward(self, x, latents, return_spatial_info=False):
        """
        Args:
            x: ID features [B, n_id, D_kv] (key/value)
            latents: image features [B, H*W, D] (query)
            return_spatial_info: if True, return (output, ca_raw) tuple
                where ca_raw is the output *before* to_out, reshaped.
        """
        x = self.norm1(x)
        latents = self.norm2(latents)

        b, seq_len, _ = latents.shape

        q = self.to_q(latents)
        k, v = self.to_kv(x).chunk(2, dim=-1)

        q = reshape_tensor(q, self.heads)
        k = reshape_tensor(k, self.heads)
        v = reshape_tensor(v, self.heads)

        scale = 1 / math.sqrt(math.sqrt(self.dim_head))
        weight = (q * scale) @ (k * scale).transpose(-2, -1)
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        out = weight @ v  # [B, heads, H*W, dim_head]

        out_perm = out.permute(0, 2, 1, 3).reshape(b, seq_len, -1)
        result = self.to_out(out_perm)

        if return_spatial_info:
            return result, out_perm  # out_perm: [B, H*W, heads*dim_head]
        return result


# ============================================================================
# 2. SpatialMaskExtractor
# ============================================================================

class SpatialMaskExtractor:
    """Extracts a spatial relevance mask [B, H*W, 1] indicating which
    image patches are face-relevant based on the CA output magnitude.

    Method: L2 norm of the CA output per patch. High norm = strong ID
    response = face region. Normalized to [0, 1].
    """

    @staticmethod
    def from_ca_output(ca_output: torch.Tensor) -> torch.Tensor:
        """Compute spatial mask from CA output (before or after to_out).

        Args:
            ca_output: [B, H*W, D]

        Returns:
            mask: [B, H*W, 1] in range [0, 1]
        """
        # L2 norm per patch
        norm = ca_output.norm(dim=-1, keepdim=True)  # [B, H*W, 1]
        # Normalize to [0, 1] per sample
        norm_min = norm.amin(dim=1, keepdim=True)
        norm_max = norm.amax(dim=1, keepdim=True)
        denom = (norm_max - norm_min).clamp(min=1e-6)
        mask = (norm - norm_min) / denom
        return mask

    @staticmethod
    def from_attention_entropy(
        attn_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Compute spatial mask from attention entropy.

        Low entropy = concentrated attention = face patch.
        High entropy = dispersed attention = background patch.

        Args:
            attn_weights: [B, heads, H*W, n_id]

        Returns:
            mask: [B, H*W, 1] in range [0, 1]
        """
        # Entropy per patch: -sum(p * log(p)) over ID tokens
        eps = 1e-8
        entropy = -(attn_weights * (attn_weights + eps).log()).sum(dim=-1)  # [B, heads, H*W]
        entropy = entropy.mean(dim=1)  # [B, H*W]

        # Invert: low entropy -> high mask value
        e_min = entropy.amin(dim=1, keepdim=True)
        e_max = entropy.amax(dim=1, keepdim=True)
        denom = (e_max - e_min).clamp(min=1e-6)
        mask = 1.0 - (entropy - e_min) / denom
        return mask.unsqueeze(-1)  # [B, H*W, 1]


# ============================================================================
# 3. MaskRefiner
# ============================================================================

class MaskRefiner:
    """Refines a raw spatial mask via Gaussian smoothing, thresholding,
    and optional dilation to ensure face edges are covered."""

    @staticmethod
    def refine(
        mask: torch.Tensor,
        h_patches: int,
        w_patches: int,
        kernel_size: int = 5,
        sigma: float = 1.5,
        soft_weight: float = 0.7,
        hard_weight: float = 0.3,
        dilate_kernel: int = 3,
    ) -> torch.Tensor:
        """Refine mask with Gaussian blur + soft-hard combination.

        Args:
            mask: [B, H*W, 1]
            h_patches, w_patches: spatial dimensions of the patch grid
            kernel_size: Gaussian kernel size
            sigma: Gaussian sigma
            soft_weight: weight for soft (blurred) mask
            hard_weight: weight for hard (thresholded) mask
            dilate_kernel: dilation kernel size for hard mask

        Returns:
            refined mask: [B, H*W, 1]
        """
        B = mask.shape[0]
        device = mask.device
        dtype = mask.dtype

        # Reshape to 2D spatial
        mask_2d = mask.squeeze(-1).reshape(B, 1, h_patches, w_patches).float()

        # Gaussian blur
        blurred = MaskRefiner._gaussian_blur_2d(mask_2d, kernel_size, sigma)

        # Hard mask: threshold + dilate
        threshold = blurred.mean(dim=(2, 3), keepdim=True) + 0.5 * blurred.std(dim=(2, 3), keepdim=True)
        hard = (blurred > threshold).float()

        # Dilate hard mask
        if dilate_kernel > 0:
            pad = dilate_kernel // 2
            hard = F.max_pool2d(hard, kernel_size=dilate_kernel, stride=1, padding=pad)

        # Combine soft + hard
        combined = soft_weight * blurred + hard_weight * hard
        combined = combined.clamp(0.0, 1.0)

        # Reshape back
        return combined.reshape(B, h_patches * w_patches, 1).to(dtype)

    @staticmethod
    def _gaussian_blur_2d(
        x: torch.Tensor, kernel_size: int, sigma: float
    ) -> torch.Tensor:
        """Apply 2D Gaussian blur to a [B, 1, H, W] tensor."""
        # Create 1D Gaussian kernel
        coords = torch.arange(kernel_size, dtype=torch.float32, device=x.device) - kernel_size // 2
        kernel_1d = torch.exp(-0.5 * (coords / sigma) ** 2)
        kernel_1d = kernel_1d / kernel_1d.sum()

        # Separable 2D convolution
        kernel_h = kernel_1d.view(1, 1, -1, 1)
        kernel_w = kernel_1d.view(1, 1, 1, -1)

        pad_h = kernel_size // 2
        pad_w = kernel_size // 2

        x = F.pad(x, (0, 0, pad_h, pad_h), mode='reflect')
        x = F.conv2d(x, kernel_h)
        x = F.pad(x, (pad_w, pad_w, 0, 0), mode='reflect')
        x = F.conv2d(x, kernel_w)
        return x


# ============================================================================
# 4. TemporalSpatialScheduler
# ============================================================================

class TemporalSpatialScheduler:
    """Adapts the spatial mask strategy based on the denoising timestep.

    - Early (t > 0.7): Image is mostly noise, attention is unreliable.
      Use a center-biased Gaussian prior as the spatial mask.
    - Mid (0.3 < t <= 0.7): Face structure is forming, attention is
      meaningful. Use the attention-derived mask.
    - Late (t <= 0.3): Face details are being refined. Relax the mask
      to allow ID texture to bleed slightly into surrounding context.

    Args:
        early_threshold: timestep above which to use center prior (default 0.7)
        late_threshold: timestep below which to relax mask (default 0.3)
        late_floor: minimum mask value in late stage (default 0.5)
        center_sigma: sigma for center Gaussian prior (default 0.3)
        global_floor: minimum mask value for ALL stages (default 0.0, disabled)
    """

    def __init__(
        self,
        early_threshold: float = 0.7,
        late_threshold: float = 0.3,
        late_floor: float = 0.5,
        center_sigma: float = 0.3,
        global_floor: float = 0.0,
    ):
        self.early_threshold = early_threshold
        self.late_threshold = late_threshold
        self.late_floor = late_floor
        self.center_sigma = center_sigma
        self.global_floor = global_floor
        # Cache for center prior
        self._center_prior_cache: Dict[Tuple[int, int], torch.Tensor] = {}

    def get_mask(
        self,
        t: float,
        ca_output: torch.Tensor,
        h_patches: int,
        w_patches: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Compute the spatial mask for the current timestep.

        Args:
            t: current normalized timestep (1.0 = noise, 0.0 = clean)
            ca_output: raw CA output [B, H*W, D] (used for mid-stage)
            h_patches, w_patches: patch grid dimensions
            device, dtype: target device and dtype

        Returns:
            mask: [B, H*W, 1]
        """
        B = ca_output.shape[0]

        if t > self.early_threshold:
            # Early stage: use center Gaussian prior
            mask = self._get_center_prior(h_patches, w_patches, device, dtype)
            mask = mask.unsqueeze(0).expand(B, -1, -1)  # [B, H*W, 1]
            if self.global_floor > 0:
                mask = torch.clamp(mask, min=self.global_floor)
            return mask

        # Extract attention-based mask
        raw_mask = SpatialMaskExtractor.from_ca_output(ca_output)

        # Refine
        refined = MaskRefiner.refine(
            raw_mask, h_patches, w_patches,
            kernel_size=5, sigma=1.5,
        )

        if t <= self.late_threshold:
            # Late stage: relax mask (raise floor)
            refined = self.late_floor + (1.0 - self.late_floor) * refined

        # Apply global floor if set
        if self.global_floor > 0:
            refined = torch.clamp(refined, min=self.global_floor)

        return refined.to(dtype)

    def _get_center_prior(
        self, h: int, w: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        """Generate a center-biased 2D Gaussian mask [H*W, 1]."""
        key = (h, w)
        if key not in self._center_prior_cache:
            y = torch.linspace(-1, 1, h, device=device)
            x = torch.linspace(-1, 1, w, device=device)
            yy, xx = torch.meshgrid(y, x, indexing='ij')
            dist_sq = xx ** 2 + yy ** 2
            prior = torch.exp(-dist_sq / (2 * self.center_sigma ** 2))
            # Normalize to [0, 1]
            prior = prior / prior.max()
            self._center_prior_cache[key] = prior.reshape(-1, 1)
        return self._center_prior_cache[key].to(device=device, dtype=dtype)


# ============================================================================
# 5. Runtime module replacement
# ============================================================================

def replace_pulid_ca_modules(model: nn.Module) -> List[SpatialPerceiverAttentionCA]:
    """Replace all PerceiverAttentionCA modules in model.pulid_ca with
    SpatialPerceiverAttentionCA wrappers. Returns the new module list.

    This is non-destructive: the original weights are shared (not copied).
    """
    if model.pulid_ca is None:
        raise ValueError("model.pulid_ca is None — PuLID pipeline not initialized")

    new_ca_list = nn.ModuleList()
    for i, ca in enumerate(model.pulid_ca):
        wrapper = SpatialPerceiverAttentionCA(ca)
        new_ca_list.append(wrapper)

    # Replace on the model
    model.pulid_ca = new_ca_list
    logger.info(f"Replaced {len(new_ca_list)} PerceiverAttentionCA modules with SpatialPerceiverAttentionCA")
    return new_ca_list


# ============================================================================
# 6. SpatialID forward hook (alternative to model.py modification)
# ============================================================================

def spatialid_forward(
    model,
    img, img_ids, txt, txt_ids, timesteps, y,
    guidance=None, id=None, id_weight=1.0,
    aggressive_offload=False,
    # SpatialID params
    spatial_scheduler: Optional[TemporalSpatialScheduler] = None,
    t_normalized: float = 1.0,
    h_patches: int = 0,
    w_patches: int = 0,
):
    """Modified Flux forward that applies spatial masking to ID injection.

    Instead of modifying model.py, we replicate the forward pass with
    spatial mask computation at each injection point.
    """
    from spatialid.models.flux.modules.layers import timestep_embedding

    if img.ndim != 3 or txt.ndim != 3:
        raise ValueError("Input img and txt tensors must have 3 dimensions.")

    DEVICE = img.device

    img_emb = model.img_in(img)
    vec = model.time_in(timestep_embedding(timesteps, 256))
    if model.params.guidance_embed:
        if guidance is None:
            raise ValueError("Didn't get guidance strength for guidance distilled model.")
        vec = vec + model.guidance_in(timestep_embedding(guidance, 256))
    vec = vec + model.vector_in(y)
    txt_emb = model.txt_in(txt)

    ids = torch.cat((txt_ids, img_ids), dim=1)
    pe = model.pe_embedder(ids)

    ca_idx = 0

    if aggressive_offload:
        model.double_blocks = model.double_blocks.to(DEVICE)

    for i, block in enumerate(model.double_blocks):
        img_emb, txt_emb = block(img=img_emb, txt=txt_emb, vec=vec, pe=pe)

        if i % model.pulid_double_interval == 0 and id is not None:
            ca_module = model.pulid_ca[ca_idx]
            if spatial_scheduler is not None and isinstance(ca_module, SpatialPerceiverAttentionCA):
                ca_out, ca_raw = ca_module(id, img_emb, return_spatial_info=True)
                mask = spatial_scheduler.get_mask(
                    t_normalized, ca_raw, h_patches, w_patches,
                    device=img_emb.device, dtype=img_emb.dtype,
                )
                img_emb = img_emb + id_weight * mask * ca_out
            else:
                img_emb = img_emb + id_weight * ca_module(id, img_emb)
            ca_idx += 1

    if aggressive_offload:
        model.double_blocks.cpu()

    img_emb = torch.cat((txt_emb, img_emb), 1)

    if aggressive_offload:
        for i in range(len(model.single_blocks) // 2):
            model.single_blocks[i] = model.single_blocks[i].to(DEVICE)

    for i, block in enumerate(model.single_blocks):
        if aggressive_offload and i == len(model.single_blocks) // 2:
            for j in range(len(model.single_blocks) // 2):
                model.single_blocks[j].cpu()
            for j in range(len(model.single_blocks) // 2, len(model.single_blocks)):
                model.single_blocks[j] = model.single_blocks[j].to(DEVICE)

        x = block(img_emb, vec=vec, pe=pe)
        real_img, txt_emb = x[:, txt_emb.shape[1]:, ...], x[:, :txt_emb.shape[1], ...]

        if i % model.pulid_single_interval == 0 and id is not None:
            ca_module = model.pulid_ca[ca_idx]
            if spatial_scheduler is not None and isinstance(ca_module, SpatialPerceiverAttentionCA):
                ca_out, ca_raw = ca_module(id, real_img, return_spatial_info=True)
                mask = spatial_scheduler.get_mask(
                    t_normalized, ca_raw, h_patches, w_patches,
                    device=real_img.device, dtype=real_img.dtype,
                )
                real_img = real_img + id_weight * mask * ca_out
            else:
                real_img = real_img + id_weight * ca_module(id, real_img)
            ca_idx += 1

        img_emb = torch.cat((txt_emb, real_img), 1)

    if aggressive_offload:
        model.single_blocks.cpu()

    img_emb = img_emb[:, txt_emb.shape[1]:, ...]
    img_emb = model.final_layer(img_emb, vec)
    return img_emb


# ============================================================================
# 7. SpatialID Denoise Loop
# ============================================================================

def spatialid_denoise(
    model,
    img, img_ids, txt, txt_ids, vec,
    timesteps,
    guidance=4.0,
    id_weight=1.0,
    id=None,
    start_step=0,
    uncond_id=None,
    true_cfg=1.0,
    timestep_to_start_cfg=1,
    neg_txt=None, neg_txt_ids=None, neg_vec=None,
    aggressive_offload=False,
    # SpatialID params
    spatial_scheduler: Optional[TemporalSpatialScheduler] = None,
    h_patches: int = 0,
    w_patches: int = 0,
):
    """Denoise loop with spatially-adaptive ID injection.

    At each denoising step, the spatial mask is computed from the CA
    output and applied to restrict ID injection to face-relevant patches.

    Args:
        model: Flux model with SpatialPerceiverAttentionCA modules
        spatial_scheduler: TemporalSpatialScheduler instance
        h_patches, w_patches: patch grid dimensions (height//16, width//16)
        (all other args same as PuLID denoise)
    """
    guidance_vec = torch.full(
        (img.shape[0],), guidance, device=img.device, dtype=img.dtype
    )
    use_true_cfg = abs(true_cfg - 1.0) > 1e-2

    for i, (t_curr, t_prev) in enumerate(zip(timesteps[:-1], timesteps[1:])):
        t_vec = torch.full(
            (img.shape[0],), t_curr, dtype=img.dtype, device=img.device
        )

        pred = spatialid_forward(
            model,
            img=img, img_ids=img_ids, txt=txt, txt_ids=txt_ids,
            timesteps=t_vec, y=vec, guidance=guidance_vec,
            id=id if i >= start_step else None,
            id_weight=id_weight,
            aggressive_offload=aggressive_offload,
            spatial_scheduler=spatial_scheduler if i >= start_step else None,
            t_normalized=t_curr,
            h_patches=h_patches,
            w_patches=w_patches,
        )

        if use_true_cfg and i >= timestep_to_start_cfg:
            neg_pred = spatialid_forward(
                model,
                img=img, img_ids=img_ids, txt=neg_txt, txt_ids=neg_txt_ids,
                timesteps=t_vec, y=neg_vec, guidance=guidance_vec,
                id=uncond_id if i >= start_step else None,
                id_weight=id_weight,
                aggressive_offload=aggressive_offload,
                spatial_scheduler=spatial_scheduler if i >= start_step else None,
                t_normalized=t_curr,
                h_patches=h_patches,
                w_patches=w_patches,
            )
            pred = neg_pred + true_cfg * (pred - neg_pred)

        img = img + (t_prev - t_curr) * pred

    return img
