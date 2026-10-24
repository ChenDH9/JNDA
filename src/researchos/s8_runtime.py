"""PhenoBench native representation and transformer construction."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image
from researchos.s8_config import SEEDS
from researchos.s8_native_training import sha256_file
ROOT = Path.cwd()
MODEL_ROOT = Path("models/FLUX.2-klein-base-4B")

PROMPT = (
    "Create a spatially aligned three-class semantic segmentation mask of the input "
    "field image: background black, crop plants green, weeds magenta. Flat solid "
    "colors only; no text, legend, shading, outlines, or photograph."
)


EXPECTED_ADAPTER_SHA256 = {
    20260723: "fb6f0f5bac3543ba848af44df12b4b5fc1d5c58bf274a3dfca2a694227c03924",
    20260724: "feebcce77b90f14fa0a458a05e4e8ca93da618f4c5a1625a84d2e30fee46a431",
    20260725: "b54b1eebcf7f5e50dc56882ce9656b438f5af806c73759e4a5ea5ec0404fed45",
}


PALETTE = np.asarray(((0, 0, 0), (34, 139, 34), (255, 0, 255)), dtype=np.uint8)


LORA_TARGET_MODULES_SHA256 = (
    "5d6980c8a99e528164ab61a1aeb04de50cf1374ac1c96731d4e6f22f4d9571f6"
)


EXPECTED_ATTENTION_PROCESSORS = {
    "diffusers.models.transformers.transformer_flux2.Flux2AttnProcessor",
    "diffusers.models.transformers.transformer_flux2.Flux2ParallelSelfAttnProcessor",
}


def _resize_rgb_and_tensor(image: Image.Image):
    import torch

    resized = image.convert("RGB").resize(
        (512, 512), resample=Image.Resampling.BILINEAR
    )
    array = np.asarray(resized, dtype=np.uint8).copy()
    tensor = torch.from_numpy(array).permute(2, 0, 1).float().div_(127.5).sub_(1.0)
    return resized, tensor.unsqueeze(0)


def _label_from_rgb(image: Image.Image):
    import torch

    resized = image.convert("RGB").resize(
        (512, 512), resample=Image.Resampling.NEAREST
    )
    pixels = np.asarray(resized, dtype=np.uint8)
    matches = np.all(pixels[:, :, None, :] == PALETTE[None, None, :, :], axis=3)
    valid = np.any(matches, axis=2)
    if not bool(valid.all()):
        invalid = np.unique(pixels[~valid], axis=0)
        raise RuntimeError(f"training target has non-canonical RGB values: {invalid[:5].tolist()}")
    return torch.from_numpy(np.argmax(matches, axis=2).astype(np.uint8, copy=False).copy())


def exact_vae_packed_tokens(pipeline, image_tensor, *, device: str):
    import torch

    source = image_tensor.to(device=device, dtype=torch.bfloat16)
    latent = pipeline.vae.encode(source).latent_dist.mode()
    if tuple(latent.shape[1:]) != (32, 64, 64):
        raise RuntimeError(f"VAE latent contract failed: {tuple(latent.shape)}")
    patched = pipeline._patchify_latents(latent)
    if tuple(patched.shape[1:]) != (128, 32, 32):
        raise RuntimeError("patchified latent contract failed")
    mean = pipeline.vae.bn.running_mean.view(1, -1, 1, 1).float().to(device)
    variance = pipeline.vae.bn.running_var.view(1, -1, 1, 1).float().to(device)
    normalized = (
        (patched.float() - mean)
        / torch.sqrt(variance + float(pipeline.vae.config.batch_norm_eps))
    ).to(torch.bfloat16)
    packed = pipeline._pack_latents(normalized)
    if tuple(packed.shape[1:]) != (1024, 128) or packed.dtype != torch.bfloat16:
        raise RuntimeError("packed VAE token contract failed")
    return packed


def _load_transformer_pipeline(
    *,
    seed: int,
    arm: str,
    device: str,
    include_vae: bool,
    gradient_checkpointing: bool = False,
):
    import torch
    from diffusers import Flux2KleinPipeline

    if seed not in SEEDS or arm not in {"A", "B"}:
        raise ValueError("unknown S8 arm or seed")
    pipeline = Flux2KleinPipeline.from_pretrained(
        str(MODEL_ROOT),
        vae=None if not include_vae else None,
        text_encoder=None,
        tokenizer=None,
        scheduler=None,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    )
    # Passing vae=None suppresses loading; inference attaches a separately loaded VAE.
    transformer = pipeline.transformer
    processors = {
        f"{type(processor).__module__}.{type(processor).__name__}"
        for processor in transformer.attn_processors.values()
    }
    if processors != EXPECTED_ATTENTION_PROCESSORS or len(transformer.attn_processors) != 25:
        raise RuntimeError("transformer attention backend differs from the implementation lock")
    for parameter in transformer.parameters():
        parameter.requires_grad_(False)
    if arm == "A":
        adapter_path = ROOT / (
            "runs/S7_Ex01_phenobench_rgb_lora/fraction_100/"
            f"seed_{seed}/attempt_001/pytorch_lora_weights.safetensors"
        )
        expected = EXPECTED_ADAPTER_SHA256[seed]
        if sha256_file(adapter_path) != expected:
            raise RuntimeError("S7 adapter SHA256 mismatch")
        pipeline.load_lora_weights(
            str(adapter_path.parent),
            weight_name=adapter_path.name,
            adapter_name="s8_frozen_s7",
            local_files_only=True,
        )
        pipeline.set_adapters("s8_frozen_s7", adapter_weights=1.0)
        for parameter in transformer.parameters():
            parameter.requires_grad_(False)
        loaded_lora = [
            (name, parameter)
            for name, parameter in transformer.named_parameters()
            if "lora_" in name.casefold()
        ]
        if (
            len(loaded_lora) != 120
            or sum(parameter.numel() for _name, parameter in loaded_lora) != 8355840
            or any(parameter.dtype != torch.bfloat16 for _name, parameter in loaded_lora)
            or any(parameter.requires_grad for _name, parameter in loaded_lora)
        ):
            raise RuntimeError("frozen S7 LoRA in-memory tensor contract failed")
        transformer.eval()
    else:
        from peft import LoraConfig

        from researchos.flux2_klein_i2i.real_trainer import lora_target_modules

        target_modules = lora_target_modules(int(transformer.config.num_single_layers))
        if (
            len(target_modules) != 25
            or hashlib.sha256(
                json.dumps(
                    target_modules, sort_keys=True, separators=(",", ":")
                ).encode()
            ).hexdigest()
            != LORA_TARGET_MODULES_SHA256
        ):
            raise RuntimeError("LoRA target module contract failed")
        transformer.add_adapter(
            LoraConfig(
                r=8,
                lora_alpha=8,
                lora_dropout=0.0,
                init_lora_weights="gaussian",
                target_modules=target_modules,
            )
        )
        if gradient_checkpointing:
            transformer.enable_gradient_checkpointing()
        trainable = [
            (name, parameter)
            for name, parameter in transformer.named_parameters()
            if parameter.requires_grad
        ]
        if (
            len(trainable) != 120
            or sum(parameter.numel() for _name, parameter in trainable) != 8355840
            or any("lora_" not in name.casefold() for name, _parameter in trainable)
            or any(parameter.dtype != torch.bfloat16 for _name, parameter in trainable)
        ):
            raise RuntimeError("fresh LoRA trainable tensor contract failed")
        peft_config = transformer.peft_config.get("default")
        if (
            peft_config is None
            or int(peft_config.r) != 8
            or int(peft_config.lora_alpha) != 8
            or float(peft_config.lora_dropout) != 0.0
            or set(peft_config.target_modules) != set(target_modules)
        ):
            raise RuntimeError("fresh LoRA PEFT configuration differs")
        if gradient_checkpointing and not transformer.is_gradient_checkpointing:
            raise RuntimeError("B gradient checkpointing failed to enable")
        transformer.train()
    pipeline.to(device)
    return pipeline, transformer
