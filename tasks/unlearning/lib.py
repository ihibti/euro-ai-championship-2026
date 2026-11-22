"""Shared helpers for the unlearning experiments: views, soft targets, profiles and proxy scores.

Every script runs from the JURECA task folder, which holds model.pt and forget_images.npz.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent / "upstream"))

import task_template as tt  # noqa: E402

TEMPERATURE = 2.0
DEVICE = tt.DEVICE


def model_input(pixels: torch.Tensor) -> torch.Tensor:
    """Float pixels [B, 3, H, W] in 0..255 to the server's model input in [-1, 1]."""
    return (pixels / 255.0 - 0.5) / 0.5


def chw(images: torch.Tensor) -> torch.Tensor:
    """uint8 [B, H, W, 3] to float pixels [B, 3, H, W] on the device."""
    return images.to(DEVICE, non_blocking=True).permute(0, 3, 1, 2).float()


def quantize(pixels: torch.Tensor) -> torch.Tensor:
    return pixels.round().clamp(0, 255)


@torch.no_grad()
def logits_of(model: torch.nn.Module, images: torch.Tensor, batch: int = 250) -> torch.Tensor:
    """Logits like the server (eval mode, float16 autocast) for uint8 images [N, H, W, 3]."""
    model.eval()
    out = []
    for start in range(0, len(images), batch):
        pixels = chw(images[start : start + batch])
        with torch.autocast("cuda", dtype=torch.float16, enabled=DEVICE == "cuda"):
            out.append(model(model_input(pixels)).float().cpu())
    return torch.cat(out)


def _gaussian_kernel(sigma: float, device: torch.device) -> torch.Tensor:
    radius = max(1, math.ceil(2.5 * sigma))
    grid = torch.arange(-radius, radius + 1, device=device, dtype=torch.float32)
    kernel = torch.exp(-(grid**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    return kernel


def blur(pixels: torch.Tensor, sigma: float) -> torch.Tensor:
    kernel = _gaussian_kernel(sigma, pixels.device)
    pad = len(kernel) // 2
    channels = pixels.shape[1]
    horizontal = kernel.view(1, 1, 1, -1).repeat(channels, 1, 1, 1)
    vertical = kernel.view(1, 1, -1, 1).repeat(channels, 1, 1, 1)
    out = F.conv2d(F.pad(pixels, (pad, pad, 0, 0), mode="reflect"), horizontal, groups=channels)
    return F.conv2d(F.pad(out, (0, 0, pad, pad), mode="reflect"), vertical, groups=channels)


def affine(
    pixels: torch.Tensor,
    scale: torch.Tensor,
    angle_deg: torch.Tensor,
    shift_x: torch.Tensor,
    shift_y: torch.Tensor,
    flip: torch.Tensor,
) -> torch.Tensor:
    """Per-image zoom (scale < 1 crops), rotation, shift (fraction of the half-width) and flip."""
    angle = angle_deg * math.pi / 180
    cos, sin = torch.cos(angle) * scale, torch.sin(angle) * scale
    sign = 1 - 2 * flip.float()
    theta = torch.stack(
        [
            torch.stack([cos * sign, -sin, shift_x], dim=1),
            torch.stack([sin * sign, cos, shift_y], dim=1),
        ],
        dim=1,
    )
    grid = F.affine_grid(theta, list(pixels.shape), align_corners=False)
    return F.grid_sample(
        pixels, grid, mode="bilinear", padding_mode="reflection", align_corners=False
    )


def color(
    pixels: torch.Tensor, brightness: torch.Tensor, contrast: torch.Tensor, saturation: torch.Tensor
) -> torch.Tensor:
    shape = (-1, 1, 1, 1)
    out = pixels * brightness.view(shape)
    mean = out.mean(dim=(1, 2, 3), keepdim=True)
    out = (out - mean) * contrast.view(shape) + mean
    gray = (0.299 * out[:, 0:1] + 0.587 * out[:, 1:2] + 0.114 * out[:, 2:3]).expand_as(out)
    return gray + (out - gray) * saturation.view(shape)


def random_views(
    pixels: torch.Tensor, generator: torch.Generator, strength: float = 1.0
) -> torch.Tensor:
    """Random slight alterations of float pixels [B, 3, H, W]: geometry, color, blur, noise."""
    count = pixels.shape[0]
    device = pixels.device

    def uniform(low: float, high: float) -> torch.Tensor:
        return low + (high - low) * torch.rand(count, generator=generator, device=device)

    out = affine(
        pixels,
        scale=uniform(1 - 0.35 * strength, 1.0),
        angle_deg=uniform(-12 * strength, 12 * strength),
        shift_x=uniform(-0.12 * strength, 0.12 * strength),
        shift_y=uniform(-0.12 * strength, 0.12 * strength),
        flip=torch.rand(count, generator=generator, device=device) < 0.5,
    )
    out = color(
        out,
        brightness=uniform(1 - 0.25 * strength, 1 + 0.25 * strength),
        contrast=uniform(1 - 0.25 * strength, 1 + 0.25 * strength),
        saturation=uniform(1 - 0.3 * strength, 1 + 0.3 * strength),
    )
    blurred = torch.rand(count, generator=generator, device=device) < 0.3 * strength
    if blurred.any():
        out[blurred] = blur(out[blurred], sigma=1.0)
    noise = uniform(0, 8 * strength).view(-1, 1, 1, 1)
    out = out + noise * torch.randn(out.shape, generator=generator, device=device)
    return quantize(out)


def jpeg(pixels: torch.Tensor, quality: int) -> torch.Tensor:
    from torchvision.io import decode_jpeg, encode_jpeg

    images = quantize(pixels).to("cpu", torch.uint8)
    decoded = [decode_jpeg(encode_jpeg(image, quality=quality)) for image in images]
    return torch.stack(decoded).to(pixels.device).float()


def fixed_view(name: str, pixels: torch.Tensor, seed: int) -> torch.Tensor:
    """Deterministic alterations used only for evaluation."""
    count = pixels.shape[0]
    ones = torch.ones(count, device=pixels.device)
    zeros = torch.zeros(count, device=pixels.device)
    if name == "exact":
        return pixels
    if name == "flip":
        return pixels.flip(3)
    if name == "crop85":
        return quantize(affine(pixels, 0.85 * ones, zeros, zeros, zeros, zeros.bool()))
    if name == "rot8":
        return quantize(affine(pixels, ones, 8 * ones, zeros, zeros, zeros.bool()))
    if name == "shift10":
        return quantize(affine(pixels, ones, zeros, 0.1 * ones, -0.1 * ones, zeros.bool()))
    if name == "blur1":
        return quantize(blur(pixels, 1.0))
    if name == "jpeg50":
        return jpeg(pixels, 50)
    if name == "jitter":
        return quantize(color(pixels, 1.15 * ones, 0.85 * ones, 1.2 * ones))
    if name == "noise8":
        generator = torch.Generator(device=pixels.device).manual_seed(seed)
        noise = torch.randn(pixels.shape, generator=generator, device=pixels.device)
        return quantize(pixels + 8 * noise)
    raise ValueError(f"unknown view {name}")


EVAL_VIEWS = ("exact", "flip", "crop85", "rot8", "shift10", "blur1", "jpeg50", "jitter", "noise8")


@torch.no_grad()
def view_logits(
    model: torch.nn.Module, images: torch.Tensor, views: tuple[str, ...], batch: int = 250
) -> dict[str, torch.Tensor]:
    """Logits of every fixed view of uint8 images [N, H, W, 3]."""
    model.eval()
    out: dict[str, list[torch.Tensor]] = {name: [] for name in views}
    for start in range(0, len(images), batch):
        pixels = chw(images[start : start + batch])
        for name in views:
            view = fixed_view(name, pixels, seed=start)
            with torch.autocast("cuda", dtype=torch.float16, enabled=DEVICE == "cuda"):
                out[name].append(model(model_input(view)).float().cpu())
    return {name: torch.cat(parts) for name, parts in out.items()}


def class_plan(
    labels: torch.Tensor, full: float, hedge: float, full_count: int = 129, class_size: int = 130
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-class amounts: a logit shift for whole-forgotten classes, a label hedge for the others.

    A class with at least full_count forget images is lowered by `full` on every image. An image
    of a thinned class c has its own label lowered by hedge * n_c / class_size.
    """
    counts = torch.bincount(labels, minlength=tt.NUM_CLASSES).float()
    is_full = counts >= full_count
    shift = torch.where(is_full, torch.tensor(float(full)), torch.tensor(0.0))
    beta = torch.where(is_full | (counts == 0), torch.tensor(0.0), hedge * counts / class_size)
    return shift, beta


def soft_targets(
    logits: torch.Tensor,
    labels: torch.Tensor,
    shift: torch.Tensor,
    beta: torch.Tensor,
    tau: float = 1.0,
) -> torch.Tensor:
    """Target log-probabilities: class shifts on every image, plus the image's own label hedge."""
    shift = shift.to(logits.device, logits.dtype)
    own = beta.to(logits.device, logits.dtype)[labels].view(-1, 1)
    shifted = logits - shift.view(1, -1) - own * F.one_hot(labels, logits.shape[1]).to(logits.dtype)
    return F.log_softmax(shifted / tau, dim=1)


def mixture_targets(
    logits: torch.Tensor,
    steps: torch.Tensor,
    labels: torch.Tensor,
    shift: torch.Tensor,
    beta: torch.Tensor,
    tau: float = 1.0,
) -> torch.Tensor:
    """Log of the mean, over held-out steps [J, B, C], of the soft targets each step gives."""
    parts = torch.stack([soft_targets(logits + step, labels, shift, beta, tau) for step in steps])
    return torch.logsumexp(parts, dim=0) - math.log(len(steps))


def heldout_delta(orig: torch.Tensor, proxy: torch.Tensor, columns: torch.Tensor) -> torch.Tensor:
    """Per-image logit step from the original model to its held-out prediction, on the columns."""
    delta = torch.zeros_like(orig)
    delta[:, columns] = proxy[:, columns].to(orig.dtype) - orig[:, columns]
    return delta


def mix_with_proxy(
    log_probs: torch.Tensor, proxy: torch.Tensor, columns: torch.Tensor, mix: float
) -> torch.Tensor:
    """Blend targets with held-out predictions, at the scoring temperature, on the given columns.

    Columns left out (classes the held-out predictor never trained on) keep their offset to the
    rest, so the loss level set by the class dial is untouched.
    """
    allowed = log_probs[:, columns]
    mass = torch.logsumexp(allowed, dim=1, keepdim=True)
    own = F.softmax(allowed / TEMPERATURE, dim=1)
    other = F.softmax(proxy[:, columns] / TEMPERATURE, dim=1)
    blended = TEMPERATURE * torch.log(((1 - mix) * own + mix * other).clamp_min(1e-30))
    out = log_probs.clone()
    out[:, columns] = blended - torch.logsumexp(blended, dim=1, keepdim=True) + mass
    return F.log_softmax(out, dim=1)


def label_loss(log_probs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return -log_probs.gather(1, labels.view(-1, 1)).squeeze(1)


def label_prob_t2(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return F.softmax(logits / TEMPERATURE, dim=1).gather(1, labels.view(-1, 1)).squeeze(1)


def kl_t2(reference_log_probs_t1: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
    """Per-image KL(p_ref || p) with both distributions taken at the scoring temperature 2."""
    ref = F.log_softmax(reference_log_probs_t1 / TEMPERATURE, dim=1)
    log_p = F.log_softmax(logits / TEMPERATURE, dim=1)
    return (ref.exp() * (ref - log_p)).sum(1)


def profile(logits: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
    log_probs = F.log_softmax(logits, dim=1)
    loss = label_loss(log_probs, labels)
    p2 = F.softmax(logits / TEMPERATURE, dim=1)
    return {
        "loss_mean": float(loss.mean()),
        "loss_std": float(loss.std()),
        "acc": float((logits.argmax(1) == labels).float().mean()),
        "label_prob_t1": float(log_probs.exp().gather(1, labels.view(-1, 1)).mean()),
        "max_prob_t2": float(p2.max(1).values.mean()),
    }


def state_dict_float32(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    state = {key: value.detach().cpu().float() for key, value in model.state_dict().items()}
    bad = [key for key, value in state.items() if not torch.isfinite(value).all()]
    if bad:
        raise ValueError(f"non-finite values in {bad[:5]}")
    return state
