"""Original four-system training and inference.

All optimizer batches pass through ``SplitAccessGuard`` before model forward.
Development uses inference only; primary input manifests contain no label path.
``portable_original`` supplies user-configured asset and manifest bindings.
"""

from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import random
import time
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from researchos.s8r_ex01_execution import (
    DEVELOPMENT_SPLIT,
    REPLICATION_SPLIT,
    TRAIN_SPLIT,
    S8RContractError,
    SplitAccessGuard,
    capture_rng_state,
    restore_rng_state,
    seed_all,
    sha256_file,
    stable_sample_order,
)

MODEL_REVISION = "8c44a2fbef88fae175da65df054db0f901aa9747"
SYSTEMS_WITH_FLUX = frozenset(
    {"Fresh_S7", "Fresh_Arm_A", "Fresh_Arm_B"}
)
PALETTE = np.asarray(
    ((0, 0, 0), (34, 139, 34), (255, 0, 255)), dtype=np.uint8
)
CLASS_WEIGHTS = (0.25, 1.3431242219633113, 1.4751680850839344)
PROMPT = (
    "Create a spatially aligned three-class semantic segmentation mask of "
    "the input agricultural field image: soil/background black, crop plants "
    "green, weeds magenta. Flat solid colors only; no text, legend, shading, "
    "outlines, or photograph."
)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(
                dict(value),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_torch(path: Path, value: Mapping[str, Any]) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        torch.save(dict(value), temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(
            json.dumps(dict(value), ensure_ascii=False, sort_keys=True) + "\n"
        )
        stream.flush()
        os.fsync(stream.fileno())


def _manifest(
    root: Path,
    *,
    split: str,
    labels_required: bool,
) -> list[dict[str, Any]]:
    names = {
        TRAIN_SPLIT: "formal_train_manifest.json",
        DEVELOPMENT_SPLIT: "formal_development_manifest.json",
        REPLICATION_SPLIT: "formal_confirmation_input_only_manifest.json",
    }
    path = (
        root
        / "audit/S8R_Ex01_preregistration_preflight_v1_1"
        / names[split]
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("rows")
    if (
        payload.get("split") != split
        or not isinstance(rows, list)
        or bool(payload.get("labels_present")) is not labels_required
    ):
        raise S8RContractError(f"{split} manifest envelope differs")
    expected = {
        TRAIN_SPLIT: 4821,
        DEVELOPMENT_SPLIT: 1206,
        REPLICATION_SPLIT: 844,
    }[split]
    if len(rows) != expected:
        raise S8RContractError(f"{split} manifest count differs")
    for row in rows:
        if row.get("split") != split:
            raise S8RContractError(f"{split} manifest row differs")
        if labels_required and (
            "label_path" not in row or "label_sha256" not in row
        ):
            raise S8RContractError(f"{split} label binding is absent")
        if not labels_required and any(
            "label" in key.casefold() for key in row
        ):
            raise S8RContractError(
                "confirmation input-only row contains a label field"
            )
    return [dict(row) for row in rows]


def _taxonomy(root: Path) -> np.ndarray:
    path = (
        root
        / "audit/S8R_Ex01_preregistration_hardening/"
        "complete_taxonomy_mapping.json"
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("mapping_rows")
    if not isinstance(rows, list) or len(rows) != 101:
        raise S8RContractError("taxonomy is not 101/101")
    lookup = np.full(256, 255, dtype=np.uint8)
    observed = set()
    for row in rows:
        source = int(row["source_class_id"])
        target = int(row["target_class_id"])
        if source in observed or target not in {0, 1, 2, 255}:
            raise S8RContractError("taxonomy row is invalid")
        observed.add(source)
        lookup[source] = target
    if observed != {*range(100), 255}:
        raise S8RContractError("taxonomy coverage differs")
    return lookup


def _verified_path(root: Path, row: Mapping[str, Any], kind: str) -> Path:
    path = (root / str(row[f"{kind}_path"])).resolve(strict=True)
    path.relative_to(root)
    if sha256_file(path) != row[f"{kind}_sha256"]:
        raise S8RContractError(f"{kind} SHA256 differs")
    return path


def _load_rgb(root: Path, row: Mapping[str, Any]) -> np.ndarray:
    with Image.open(_verified_path(root, row, "image")) as image:
        return np.asarray(
            image.convert("RGB").resize(
                (512, 512), Image.Resampling.BILINEAR
            ),
            dtype=np.uint8,
        ).copy()


def _load_label(
    root: Path, row: Mapping[str, Any], lookup: np.ndarray
) -> np.ndarray:
    with Image.open(_verified_path(root, row, "label")) as label:
        raw = np.asarray(
            label.resize((512, 512), Image.Resampling.NEAREST),
            dtype=np.uint8,
        )
    return lookup[raw].copy()


def _rgb_tensor(array: np.ndarray, *, minus_one_to_one: bool):
    import torch

    value = torch.from_numpy(array.copy()).permute(2, 0, 1).float() / 255.0
    return value.mul(2.0).sub(1.0) if minus_one_to_one else value


def _segformer_normalize_tensor(tensor):
    """Apply the ImageNet normalization frozen for SegFormer-B2 inputs."""

    from torchvision.transforms import functional as vision_functional

    return vision_functional.normalize(
        tensor,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )


def _target_rgb_and_valid(labels: np.ndarray):
    import torch

    safe = labels.copy()
    valid = safe != 255
    safe[~valid] = 0
    rgb = PALETTE[safe]
    return (
        _rgb_tensor(rgb, minus_one_to_one=True),
        torch.from_numpy(valid.copy()).unsqueeze(0).float(),
    )


def _label_tensor(labels: np.ndarray):
    import torch

    return torch.from_numpy(labels.copy()).long()


def _checkpoint_file(output: Path, epoch: int) -> Path:
    return output / f"checkpoint_epoch_{epoch:02d}" / "training_state.pt"


def _save_checkpoint(
    *,
    output: Path,
    epoch: int,
    system: str,
    seed: int,
    model_state: Mapping[str, Any],
    optimizer: Any,
    scheduler: Any,
    generator: Any,
    global_step: int,
    producer_context: Mapping[str, Any],
) -> Path:
    path = _checkpoint_file(output, epoch)
    _atomic_torch(
        path,
        {
            "system": system,
            "seed": seed,
            "completed_epochs": epoch,
            "global_step": global_step,
            "producer_pid": os.getpid(),
            "model_state": dict(model_state),
            "optimizer": optimizer.state_dict(),
            "scheduler": (
                scheduler.state_dict() if scheduler is not None else None
            ),
            "rng": capture_rng_state(generator),
            "producer_context": dict(producer_context),
        },
    )
    return path


def _resume_checkpoint(
    *,
    output: Path,
    system: str,
    seed: int,
    model_loader: Any,
    optimizer: Any,
    scheduler: Any,
    generator: Any,
) -> dict[str, Any]:
    import torch

    path = _checkpoint_file(output, 5)
    state = torch.load(path, map_location="cpu", weights_only=False)
    checks = {
        "system": state.get("system") == system,
        "seed": state.get("seed") == seed,
        "epoch": state.get("completed_epochs") == 5,
        "fresh_pid": int(state.get("producer_pid", -1)) != os.getpid(),
    }
    if not all(checks.values()):
        raise S8RContractError(
            f"fresh-process midpoint checkpoint differs: {checks}"
        )
    model_loader(state["model_state"])
    optimizer.load_state_dict(state["optimizer"])
    if scheduler is not None:
        scheduler.load_state_dict(state["scheduler"])
    restore_rng_state(state["rng"], generator)
    return {
        "path": path,
        "sha256": sha256_file(path),
        "producer_pid": state["producer_pid"],
        "global_step": int(state["global_step"]),
    }


def _training_bounds(process: str) -> tuple[int, int]:
    if process == "process_a":
        return 0, 5
    if process == "process_b":
        return 5, 10
    raise S8RContractError("training process must be process_a or process_b")


def _model_root(root: Path) -> Path:
    return (
        root
        / "models/verified/black-forest-labs/FLUX.2-klein-base-4B"
        / MODEL_REVISION
    ).resolve(strict=True)


def _prompt_embeddings(root: Path, *, device: str):
    import torch
    from diffusers import Flux2KleinPipeline

    text_pipeline = Flux2KleinPipeline.from_pretrained(
        str(_model_root(root)),
        vae=None,
        transformer=None,
        scheduler=None,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    ).to(device)
    with torch.inference_mode():
        prompt, text_ids = text_pipeline.encode_prompt(
            prompt=PROMPT,
            device=device,
            max_sequence_length=512,
        )
    prompt = prompt.detach()
    text_ids = text_ids.detach()
    text_pipeline.to("cpu")
    del text_pipeline
    gc.collect()
    torch.cuda.empty_cache()
    return prompt, text_ids


def _flux_components(
    root: Path,
    *,
    seed: int,
    device: str,
    adapter_path: Path | None = None,
    train_lora: bool,
):
    import torch
    from diffusers import AutoencoderKLFlux2, Flux2KleinPipeline
    from peft import LoraConfig

    from researchos.flux2_klein_i2i.real_trainer import lora_target_modules

    seed_all(seed)
    prompt, text_ids = _prompt_embeddings(root, device=device)
    pipeline = Flux2KleinPipeline.from_pretrained(
        str(_model_root(root)),
        vae=None,
        text_encoder=None,
        tokenizer=None,
        scheduler=None,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    )
    transformer = pipeline.transformer
    for parameter in transformer.parameters():
        parameter.requires_grad_(False)
    if adapter_path is None:
        targets = lora_target_modules(
            int(transformer.config.num_single_layers)
        )
        transformer.add_adapter(
            LoraConfig(
                r=8,
                lora_alpha=8,
                lora_dropout=0.0,
                init_lora_weights="gaussian",
                target_modules=targets,
            )
        )
    else:
        pipeline.load_lora_weights(
            str(adapter_path.parent),
            weight_name=adapter_path.name,
            adapter_name="paired_s7",
            local_files_only=True,
        )
        pipeline.set_adapters("paired_s7", adapter_weights=1.0)
    for name, parameter in transformer.named_parameters():
        parameter.requires_grad_(train_lora and "lora_" in name.casefold())
    if train_lora:
        transformer.enable_gradient_checkpointing()
        if not transformer.is_gradient_checkpointing:
            raise S8RContractError(
                "FLUX gradient checkpointing failed to enable"
            )
    trainable = [
        (name, value)
        for name, value in transformer.named_parameters()
        if value.requires_grad
    ]
    if train_lora and (
        not trainable
        or any("lora_" not in name.casefold() for name, _value in trainable)
    ):
        raise S8RContractError("FLUX trainable whitelist differs")
    if not train_lora and trainable:
        raise S8RContractError("frozen FLUX unexpectedly has trainables")
    pipeline.to(device)
    vae = AutoencoderKLFlux2.from_pretrained(
        str(_model_root(root)),
        subfolder="vae",
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    ).to(device)
    vae.eval()
    for parameter in vae.parameters():
        parameter.requires_grad_(False)
    return pipeline, transformer, vae, prompt, text_ids


def _lora_state(transformer: Any) -> dict[str, Any]:
    from peft.utils import get_peft_model_state_dict

    return {
        key: value.detach().cpu()
        for key, value in get_peft_model_state_dict(transformer).items()
    }


def _load_lora_state(transformer: Any, state: Mapping[str, Any]) -> None:
    from peft.utils import set_peft_model_state_dict

    result = set_peft_model_state_dict(transformer, dict(state))
    unexpected = getattr(result, "unexpected_keys", ())
    if unexpected:
        raise S8RContractError(
            f"LoRA checkpoint has unexpected keys: {unexpected}"
        )


def _save_lora_adapter(directory: Path, transformer: Any) -> Path:
    from diffusers import Flux2KleinPipeline

    directory.mkdir(parents=True, exist_ok=True)
    Flux2KleinPipeline.save_lora_weights(
        save_directory=directory,
        transformer_lora_layers=_lora_state(transformer),
    )
    path = directory / "pytorch_lora_weights.safetensors"
    if not path.is_file():
        raise S8RContractError("LoRA adapter save failed")
    return path


def train_flux_segment(
    *,
    project_root: Path,
    config: Mapping[str, Any],
    config_path: Path,
    config_sha256: str,
    system: str,
    seed: int,
    output_dir: Path,
    gpu: Mapping[str, Any],
    s7_checkpoint: Path | None,
    process: str,
) -> dict[str, Any]:
    """Train the registered Fresh S7 LoRA with masked latent flow loss."""

    del config_path, s7_checkpoint
    if system != "Fresh_S7":
        raise S8RContractError("FLUX RGB trainer accepts Fresh S7 only")
    import torch
    from diffusers import FlowMatchEulerDiscreteScheduler, Flux2KleinPipeline
    from diffusers.training_utils import (
        compute_density_for_timestep_sampling,
        compute_loss_weighting_for_sd3,
    )
    from torch.nn import functional as functional

    rows = _manifest(
        project_root, split=TRAIN_SPLIT, labels_required=True
    )
    lookup = _taxonomy(project_root)
    start_epoch, stop_epoch = _training_bounds(process)
    seeded = seed_all(seed)
    generator = seeded["dataloader_generator"]
    output_dir.mkdir(parents=True, exist_ok=True)
    pipeline, transformer, vae, prompt, text_ids = _flux_components(
        project_root,
        seed=seed,
        device="cuda:0",
        adapter_path=None,
        train_lora=True,
    )
    optimizer = torch.optim.AdamW(
        [value for value in transformer.parameters() if value.requires_grad],
        lr=float(config["training"]["Fresh_S7"]["learning_rate"]),
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda _step: 1.0
    )
    resume = None
    global_step = start_epoch * len(rows)
    if process == "process_b":
        resume = _resume_checkpoint(
            output=output_dir,
            system=system,
            seed=seed,
            model_loader=lambda state: _load_lora_state(
                transformer, state["lora"]
            ),
            optimizer=optimizer,
            scheduler=scheduler,
            generator=generator,
        )
        if resume["global_step"] != global_step:
            raise S8RContractError("S7 midpoint step differs")
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        str(_model_root(project_root)),
        subfolder="scheduler",
        local_files_only=True,
    )
    latents_mean = vae.bn.running_mean.view(1, -1, 1, 1).to("cuda:0")
    latents_std = torch.sqrt(
        vae.bn.running_var.view(1, -1, 1, 1)
        + vae.config.batch_norm_eps
    ).to("cuda:0")
    guard = SplitAccessGuard()
    journal = output_dir / f"metrics_{process}.jsonl"
    if journal.exists():
        raise FileExistsError(journal)
    started = time.monotonic()
    transformer.train()
    for epoch in range(start_epoch, stop_epoch):
        ordered = stable_sample_order(
            [str(row["sample_id"]) for row in rows],
            seed=seed,
            epoch=epoch,
        )
        by_id = {str(row["sample_id"]): row for row in rows}
        for sample_id in ordered:
            row = by_id[sample_id]
            record = [{"sample_id": sample_id, "split": TRAIN_SPLIT}]
            guard.before_optimizer_forward(record)
            rgb = _load_rgb(project_root, row)
            labels = _load_label(project_root, row, lookup)
            target_rgb, valid = _target_rgb_and_valid(labels)
            condition = _rgb_tensor(
                rgb, minus_one_to_one=True
            ).unsqueeze(0).to("cuda:0", dtype=torch.bfloat16)
            target_rgb = target_rgb.unsqueeze(0).to(
                "cuda:0", dtype=torch.bfloat16
            )
            valid = valid.unsqueeze(0).to("cuda:0")
            with torch.no_grad():
                model_input = vae.encode(target_rgb).latent_dist.mode()
                cond_input = vae.encode(condition).latent_dist.mode()
                model_input = Flux2KleinPipeline._patchify_latents(
                    model_input
                )
                cond_input = Flux2KleinPipeline._patchify_latents(cond_input)
                model_input = (model_input - latents_mean) / latents_std
                cond_input = (cond_input - latents_mean) / latents_std
            input_ids = Flux2KleinPipeline._prepare_latent_ids(
                model_input
            ).to(model_input.device)
            cond_ids = Flux2KleinPipeline._prepare_image_ids(
                [cond_input[0:1]]
            ).to(cond_input.device)
            noise = torch.randn_like(model_input)
            density = compute_density_for_timestep_sampling(
                weighting_scheme="none",
                batch_size=1,
                logit_mean=0.0,
                logit_std=1.0,
                mode_scale=1.29,
            )
            indices = (
                density * noise_scheduler.config.num_train_timesteps
            ).long()
            timesteps = noise_scheduler.timesteps[indices].to(
                model_input.device
            )
            schedule_timesteps = noise_scheduler.timesteps.to(
                model_input.device
            )
            sigma = noise_scheduler.sigmas[
                [(schedule_timesteps == value).nonzero().item()
                 for value in timesteps]
            ].to(device=model_input.device, dtype=model_input.dtype)
            while sigma.ndim < model_input.ndim:
                sigma = sigma.unsqueeze(-1)
            noisy = (1.0 - sigma) * model_input + sigma * noise
            packed_noisy = Flux2KleinPipeline._pack_latents(noisy)
            packed_cond = Flux2KleinPipeline._pack_latents(cond_input)
            original_shape = packed_noisy.shape
            original_ids_shape = input_ids.shape
            packed_noisy = torch.cat([packed_noisy, packed_cond], dim=1)
            all_ids = torch.cat([input_ids, cond_ids], dim=1)
            guidance = torch.full(
                (1,), 4.0, device="cuda:0", dtype=torch.float32
            )
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type="cuda", dtype=torch.bfloat16
            ):
                prediction = transformer(
                    hidden_states=packed_noisy,
                    timestep=timesteps / 1000,
                    guidance=guidance,
                    encoder_hidden_states=prompt,
                    txt_ids=text_ids,
                    img_ids=all_ids,
                    return_dict=False,
                )[0]
                prediction = prediction[:, : original_shape[1], :]
                prediction = (
                    Flux2KleinPipeline._unpack_latents_with_ids(
                        prediction,
                        all_ids[:, : original_ids_shape[1], :],
                    )
                )
                target = noise - model_input
                weighting = compute_loss_weighting_for_sd3(
                    weighting_scheme="none", sigmas=sigma
                )
                latent_valid = functional.interpolate(
                    valid,
                    size=target.shape[-2:],
                    mode="nearest",
                )
                squared = (
                    weighting.float()
                    * (prediction.float() - target.float()).square()
                    * latent_valid
                )
                denominator = (
                    latent_valid.sum()
                    * target.shape[1]
                ).clamp_min(1.0)
                loss = squared.sum() / denominator
            if not bool(torch.isfinite(loss).item()):
                raise FloatingPointError("Fresh S7 loss is non-finite")
            guard.record_backward(record)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [value for value in transformer.parameters()
                 if value.requires_grad],
                1.0,
            )
            guard.record_optimizer_step(record)
            optimizer.step()
            scheduler.step()
            global_step += 1
            _append_jsonl(
                journal,
                {
                    "system": system,
                    "seed": seed,
                    "epoch": epoch,
                    "sample_id": sample_id,
                    "global_step": global_step,
                    "loss": float(loss.detach().cpu()),
                    "split": TRAIN_SPLIT,
                },
            )
    checkpoint = _save_checkpoint(
        output=output_dir,
        epoch=stop_epoch,
        system=system,
        seed=seed,
        model_state={"lora": _lora_state(transformer)},
        optimizer=optimizer,
        scheduler=scheduler,
        generator=generator,
        global_step=global_step,
        producer_context={
            "config_sha256": config_sha256,
            "journal_sha256": sha256_file(journal),
            "gpu_uuid": gpu["visible_gpu_uuid"],
        },
    )
    adapter = None
    if stop_epoch == 10:
        adapter = _save_lora_adapter(
            output_dir / "checkpoint_epoch_10", transformer
        )
        receipt = {
            "status": "PASS",
            "system": system,
            "seed": seed,
            "completed_epochs": 10,
            "global_step": global_step,
            "scientific_checkpoint": True,
            "checkpoint_sha256": sha256_file(checkpoint),
            "adapter_path": adapter.relative_to(project_root).as_posix(),
            "adapter_sha256": sha256_file(adapter),
            "same_seed_arm_a_eligible": True,
            "config_sha256": config_sha256,
        }
        _atomic_json(
            output_dir
            / "checkpoint_epoch_10/formal_checkpoint_receipt.json",
            receipt,
        )
    pipeline.to("cpu")
    vae.to("cpu")
    pipeline = transformer = vae = None
    gc.collect()
    torch.cuda.empty_cache()
    return {
        "status": "PASS",
        "system": system,
        "seed": seed,
        "phase": process,
        "completed_epochs": stop_epoch,
        "global_step": global_step,
        "checkpoint_sha256": sha256_file(checkpoint),
        "adapter_sha256": sha256_file(adapter) if adapter else None,
        "fresh_process_resume": process == "process_a" or resume is not None,
        "guard": guard.receipt(),
        "elapsed_seconds": time.monotonic() - started,
        "replication_confirmation_label_open_count": 0,
        "final_method_test_label_open_count": 0,
    }


def _native_head(seed: int, *, device: str):
    import torch

    seed_all(seed)
    head = torch.nn.Linear(128, 16 * 16 * 3, bias=True)
    if sum(value.numel() for value in head.parameters()) != 99072:
        raise S8RContractError("native head parameter count differs")
    return head.to(device)


def _head_logits(head: Any, native: Any):
    projected = head(native)
    if tuple(projected.shape[1:]) != (1024, 768):
        raise S8RContractError("native projected shape differs")
    batch = projected.shape[0]
    return (
        projected.reshape(batch, 32, 32, 16, 16, 3)
        .permute(0, 1, 3, 2, 4, 5)
        .reshape(batch, 512, 512, 3)
        .permute(0, 3, 1, 2)
        .contiguous()
    )


def _native_forward(
    transformer: Any,
    packed: Any,
    *,
    prompt: Any,
    text_ids: Any,
):
    import torch

    batch = packed.shape[0]
    h = torch.arange(32, dtype=torch.int64)
    w = torch.arange(32, dtype=torch.int64)
    ids = torch.cartesian_prod(
        torch.arange(1, dtype=torch.int64), h, w, torch.arange(1)
    ).unsqueeze(0).expand(batch, -1, -1).to(packed.device)
    result = transformer(
        hidden_states=packed,
        encoder_hidden_states=prompt.expand(batch, -1, -1),
        timestep=torch.full(
            (batch,), 0.999, dtype=torch.float32, device=packed.device
        ),
        img_ids=ids,
        txt_ids=text_ids.expand(batch, -1, -1),
        guidance=torch.zeros(
            batch, dtype=torch.float32, device=packed.device
        ),
        return_dict=False,
    )[0]
    if tuple(result.shape) != (batch, 1024, 128):
        raise S8RContractError("native transformer output shape differs")
    return result


def _packed_input(pipeline: Any, vae: Any, rgb: np.ndarray):
    import torch

    tensor = _rgb_tensor(rgb, minus_one_to_one=True).unsqueeze(0).to(
        "cuda:0", dtype=torch.bfloat16
    )
    with torch.no_grad():
        latent = vae.encode(tensor).latent_dist.mode()
        latent = pipeline._patchify_latents(latent)
        latent = (latent - vae.bn.running_mean.view(1, -1, 1, 1)) / torch.sqrt(
            vae.bn.running_var.view(1, -1, 1, 1)
            + vae.config.batch_norm_eps
        )
        packed = pipeline._pack_latents(latent)
    if tuple(packed.shape) != (1, 1024, 128):
        raise S8RContractError("packed VAE token shape differs")
    return packed


def train_native_segment(
    *,
    project_root: Path,
    config: Mapping[str, Any],
    config_path: Path,
    config_sha256: str,
    system: str,
    seed: int,
    output_dir: Path,
    gpu: Mapping[str, Any],
    s7_checkpoint: Path | None,
    process: str,
) -> dict[str, Any]:
    """Train paired frozen-adapter A or fresh joint-native B."""

    del config_path
    if system not in {"Fresh_Arm_A", "Fresh_Arm_B"}:
        raise S8RContractError("native trainer accepts only Arm A/B")
    import torch

    rows = _manifest(
        project_root, split=TRAIN_SPLIT, labels_required=True
    )
    lookup = _taxonomy(project_root)
    start_epoch, stop_epoch = _training_bounds(process)
    seeded = seed_all(seed)
    generator = seeded["dataloader_generator"]
    adapter = (
        s7_checkpoint / "pytorch_lora_weights.safetensors"
        if s7_checkpoint is not None
        else None
    )
    if system == "Fresh_Arm_A" and (
        adapter is None or not adapter.is_file()
    ):
        raise S8RContractError("paired S7 adapter is absent")
    pipeline, transformer, vae, prompt, text_ids = _flux_components(
        project_root,
        seed=seed,
        device="cuda:0",
        adapter_path=adapter,
        train_lora=system == "Fresh_Arm_B",
    )
    head = _native_head(seed, device="cuda:0")
    parameters = [
        {"params": [head.weight], "lr": 1e-3, "weight_decay": 0.01},
        {"params": [head.bias], "lr": 1e-3, "weight_decay": 0.0},
    ]
    if system == "Fresh_Arm_B":
        parameters.append(
            {
                "params": [
                    value
                    for value in transformer.parameters()
                    if value.requires_grad
                ],
                "lr": 1e-4,
                "weight_decay": 0.01,
            }
        )
    optimizer = torch.optim.AdamW(
        parameters, betas=(0.9, 0.999), eps=1e-8
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda _step: 1.0
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    resume = None
    global_step = start_epoch * len(rows)

    def load_state(state: Mapping[str, Any]) -> None:
        head.load_state_dict(state["head"], strict=True)
        if system == "Fresh_Arm_B":
            _load_lora_state(transformer, state["lora"])

    if process == "process_b":
        resume = _resume_checkpoint(
            output=output_dir,
            system=system,
            seed=seed,
            model_loader=load_state,
            optimizer=optimizer,
            scheduler=scheduler,
            generator=generator,
        )
        if resume["global_step"] != global_step:
            raise S8RContractError("native midpoint step differs")
    class_weights = torch.tensor(
        CLASS_WEIGHTS, dtype=torch.float32, device="cuda:0"
    )
    guard = SplitAccessGuard()
    journal = output_dir / f"metrics_{process}.jsonl"
    if journal.exists():
        raise FileExistsError(journal)
    started = time.monotonic()
    transformer.train(system == "Fresh_Arm_B")
    head.train()
    for epoch in range(start_epoch, stop_epoch):
        ordered = stable_sample_order(
            [str(row["sample_id"]) for row in rows],
            seed=seed,
            epoch=epoch,
        )
        by_id = {str(row["sample_id"]): row for row in rows}
        for sample_id in ordered:
            row = by_id[sample_id]
            record = [{"sample_id": sample_id, "split": TRAIN_SPLIT}]
            guard.before_optimizer_forward(record)
            rgb = _load_rgb(project_root, row)
            labels = _label_tensor(
                _load_label(project_root, row, lookup)
            ).unsqueeze(0).to("cuda:0")
            packed = _packed_input(pipeline, vae, rgb)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type="cuda", dtype=torch.bfloat16
            ):
                if system == "Fresh_Arm_A":
                    with torch.no_grad():
                        native = _native_forward(
                            transformer,
                            packed,
                            prompt=prompt,
                            text_ids=text_ids,
                        )
                else:
                    native = _native_forward(
                        transformer,
                        packed,
                        prompt=prompt,
                        text_ids=text_ids,
                    )
                logits = _head_logits(head, native)
                loss = torch.nn.functional.cross_entropy(
                    logits.float(),
                    labels,
                    weight=class_weights,
                    ignore_index=255,
                )
            if not bool(torch.isfinite(loss).item()):
                raise FloatingPointError("native loss is non-finite")
            guard.record_backward(record)
            loss.backward()
            trainables = [
                value
                for value in [*head.parameters(), *transformer.parameters()]
                if value.requires_grad
            ]
            torch.nn.utils.clip_grad_norm_(trainables, 1.0)
            guard.record_optimizer_step(record)
            optimizer.step()
            scheduler.step()
            global_step += 1
            _append_jsonl(
                journal,
                {
                    "system": system,
                    "seed": seed,
                    "epoch": epoch,
                    "sample_id": sample_id,
                    "global_step": global_step,
                    "loss": float(loss.detach().cpu()),
                    "split": TRAIN_SPLIT,
                },
            )
    model_state = {"head": head.state_dict()}
    if system == "Fresh_Arm_B":
        model_state["lora"] = _lora_state(transformer)
    checkpoint = _save_checkpoint(
        output=output_dir,
        epoch=stop_epoch,
        system=system,
        seed=seed,
        model_state=model_state,
        optimizer=optimizer,
        scheduler=scheduler,
        generator=generator,
        global_step=global_step,
        producer_context={
            "config_sha256": config_sha256,
            "journal_sha256": sha256_file(journal),
            "gpu_uuid": gpu["visible_gpu_uuid"],
            "paired_s7_adapter_sha256": (
                sha256_file(adapter) if adapter else None
            ),
        },
    )
    if stop_epoch == 10:
        adapter_output = (
            _save_lora_adapter(
                output_dir / "checkpoint_epoch_10", transformer
            )
            if system == "Fresh_Arm_B"
            else None
        )
        receipt = {
            "status": "PASS",
            "system": system,
            "seed": seed,
            "completed_epochs": 10,
            "global_step": global_step,
            "scientific_checkpoint": True,
            "checkpoint_sha256": sha256_file(checkpoint),
            "adapter_sha256": (
                sha256_file(adapter_output) if adapter_output else None
            ),
            "paired_s7_adapter_sha256": (
                sha256_file(adapter) if adapter else None
            ),
            "config_sha256": config_sha256,
        }
        _atomic_json(
            output_dir
            / "checkpoint_epoch_10/formal_checkpoint_receipt.json",
            receipt,
        )
    pipeline.to("cpu")
    vae.to("cpu")
    pipeline = transformer = vae = head = None
    gc.collect()
    torch.cuda.empty_cache()
    return {
        "status": "PASS",
        "system": system,
        "seed": seed,
        "phase": process,
        "completed_epochs": stop_epoch,
        "global_step": global_step,
        "checkpoint_sha256": sha256_file(checkpoint),
        "fresh_process_resume": process == "process_a" or resume is not None,
        "guard": guard.receipt(),
        "elapsed_seconds": time.monotonic() - started,
        "replication_confirmation_label_open_count": 0,
        "final_method_test_label_open_count": 0,
    }


def _segformer_model(root: Path):
    from researchos.s6_supervised import (
        build_openmmlab_segformer_b2,
        load_openmmlab_segformer_b2_checkpoint,
    )

    checkpoint = (
        root
        / "models/verified/openmmlab/"
        "segformer_mit-b2_512x512_160k_ade20k/cbd414ac/"
        "segformer_mit-b2_512x512_160k_ade20k_20210726_112103-cbd414ac.pth"
    )
    model = build_openmmlab_segformer_b2(num_classes=3)
    report = load_openmmlab_segformer_b2_checkpoint(model, checkpoint)
    return model, report


def _segformer_optimizer(model: Any):
    import torch

    groups: dict[tuple[float, float], list[Any]] = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        decoder = (
            "decode" in name.casefold()
            or "classifier" in name.casefold()
            or "head" in name.casefold()
        )
        lr = 6e-4 if decoder else 6e-5
        no_decay = name.endswith(".bias") or parameter.ndim == 1
        groups.setdefault((lr, 0.0 if no_decay else 0.01), []).append(
            parameter
        )
    return torch.optim.AdamW(
        [
            {"params": values, "lr": lr, "weight_decay": decay}
            for (lr, decay), values in groups.items()
        ],
        betas=(0.9, 0.999),
    )


def _segformer_transform(
    rgb: np.ndarray,
    labels: np.ndarray,
    *,
    seed: int,
    epoch: int,
    sample_id: str,
) -> tuple[Any, Any]:
    import torch
    from torchvision.transforms import functional as functional

    local_seed = int.from_bytes(
        hashlib.sha256(
            f"{seed}\0{epoch}\0{sample_id}".encode()
        ).digest()[:8],
        "big",
    )
    rng = random.Random(local_seed)
    image = Image.fromarray(rgb, mode="RGB")
    label = Image.fromarray(labels, mode="L")
    if rng.random() < 0.5:
        image = functional.hflip(image)
        label = functional.hflip(label)
    if rng.random() < 0.5:
        image = functional.vflip(image)
        label = functional.vflip(label)
    image = functional.adjust_brightness(image, rng.uniform(0.6, 1.4))
    image = functional.adjust_contrast(image, rng.uniform(0.6, 1.4))
    image = functional.adjust_saturation(image, rng.uniform(0.8, 1.2))
    image = functional.adjust_hue(image, rng.uniform(-0.0125, 0.0125))
    tensor = functional.to_tensor(image)
    tensor = functional.normalize(
        tensor,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )
    return tensor, torch.from_numpy(np.asarray(label).copy()).long()


def train_segformer_segment(
    *,
    project_root: Path,
    config: Mapping[str, Any],
    config_path: Path,
    config_sha256: str,
    system: str,
    seed: int,
    output_dir: Path,
    gpu: Mapping[str, Any],
    s7_checkpoint: Path | None,
    process: str,
) -> dict[str, Any]:
    """Train Fresh SegFormer-B2 for an exact five-epoch process half."""

    del config_path, s7_checkpoint
    if system != "Fresh_SegFormer_B2":
        raise S8RContractError("SegFormer backend system differs")
    import torch
    from torch.nn import functional as functional

    rows = _manifest(
        project_root, split=TRAIN_SPLIT, labels_required=True
    )
    lookup = _taxonomy(project_root)
    start_epoch, stop_epoch = _training_bounds(process)
    seeded = seed_all(seed)
    generator = seeded["dataloader_generator"]
    model, load_report = _segformer_model(project_root)
    model.to("cuda:0")
    optimizer = _segformer_optimizer(model)
    total_steps = 6030
    warmup = math.ceil(total_steps * 0.05)

    def factor(step: int) -> float:
        if step < warmup:
            return 1e-6 + (1.0 - 1e-6) * step / max(1, warmup)
        return max(
            0.0,
            1.0 - (step - warmup) / max(1, total_steps - warmup),
        )

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, factor)
    output_dir.mkdir(parents=True, exist_ok=True)
    resume = None
    global_step = start_epoch * 603
    if process == "process_b":
        resume = _resume_checkpoint(
            output=output_dir,
            system=system,
            seed=seed,
            model_loader=lambda state: model.load_state_dict(
                state["model"], strict=True
            ),
            optimizer=optimizer,
            scheduler=scheduler,
            generator=generator,
        )
        if resume["global_step"] != global_step:
            raise S8RContractError("SegFormer midpoint step differs")
    weights = torch.tensor(
        CLASS_WEIGHTS, dtype=torch.float32, device="cuda:0"
    )
    guard = SplitAccessGuard()
    journal = output_dir / f"metrics_{process}.jsonl"
    if journal.exists():
        raise FileExistsError(journal)
    started = time.monotonic()
    model.train()
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(start_epoch, stop_epoch):
        ordered = stable_sample_order(
            [str(row["sample_id"]) for row in rows],
            seed=seed,
            epoch=epoch,
        )
        by_id = {str(row["sample_id"]): row for row in rows}
        for batch_start in range(0, len(ordered), 4):
            ids = ordered[batch_start : batch_start + 4]
            records = [
                {"sample_id": sample_id, "split": TRAIN_SPLIT}
                for sample_id in ids
            ]
            guard.before_optimizer_forward(records)
            tensors, targets = [], []
            for sample_id in ids:
                row = by_id[sample_id]
                rgb = _load_rgb(project_root, row)
                labels = _load_label(project_root, row, lookup)
                tensor, target = _segformer_transform(
                    rgb,
                    labels,
                    seed=seed,
                    epoch=epoch,
                    sample_id=sample_id,
                )
                tensors.append(tensor)
                targets.append(target)
            images = torch.stack(tensors).to("cuda:0")
            labels = torch.stack(targets).to("cuda:0")
            with torch.autocast(
                device_type="cuda", dtype=torch.bfloat16
            ):
                logits = model(images)
                logits = functional.interpolate(
                    logits,
                    size=(512, 512),
                    mode="bilinear",
                    align_corners=False,
                )
                loss = functional.cross_entropy(
                    logits.float(),
                    labels,
                    weight=weights,
                    ignore_index=255,
                )
            if not bool(torch.isfinite(loss).item()):
                raise FloatingPointError("SegFormer loss is non-finite")
            guard.record_backward(records)
            batch_index = batch_start // 4
            boundary = (
                (batch_index + 1) % 2 == 0
                or batch_start + 4 >= len(ordered)
            )
            accumulation_group_size = (
                1
                if batch_start + 4 >= len(ordered)
                and (batch_index + 1) % 2 != 0
                else 2
            )
            (loss / accumulation_group_size).backward()
            if boundary:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                guard.record_optimizer_step(records)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
            _append_jsonl(
                journal,
                {
                    "system": system,
                    "seed": seed,
                    "epoch": epoch,
                    "sample_ids": ids,
                    "global_step": global_step,
                    "optimizer_boundary": boundary,
                    "loss": float(loss.detach().cpu()),
                    "split": TRAIN_SPLIT,
                },
            )
    checkpoint = _save_checkpoint(
        output=output_dir,
        epoch=stop_epoch,
        system=system,
        seed=seed,
        model_state={"model": model.state_dict()},
        optimizer=optimizer,
        scheduler=scheduler,
        generator=generator,
        global_step=global_step,
        producer_context={
            "config_sha256": config_sha256,
            "journal_sha256": sha256_file(journal),
            "gpu_uuid": gpu["visible_gpu_uuid"],
            "load_report": load_report,
        },
    )
    if stop_epoch == 10:
        _atomic_json(
            output_dir
            / "checkpoint_epoch_10/formal_checkpoint_receipt.json",
            {
                "status": "PASS",
                "system": system,
                "seed": seed,
                "completed_epochs": 10,
                "global_step": global_step,
                "scientific_checkpoint": True,
                "checkpoint_sha256": sha256_file(checkpoint),
                "adapter_sha256": None,
                "config_sha256": config_sha256,
            },
        )
    model.to("cpu")
    model = None
    gc.collect()
    torch.cuda.empty_cache()
    return {
        "status": "PASS",
        "system": system,
        "seed": seed,
        "phase": process,
        "completed_epochs": stop_epoch,
        "global_step": global_step,
        "checkpoint_sha256": sha256_file(checkpoint),
        "fresh_process_resume": process == "process_a" or resume is not None,
        "guard": guard.receipt(),
        "elapsed_seconds": time.monotonic() - started,
        "replication_confirmation_label_open_count": 0,
        "final_method_test_label_open_count": 0,
    }


def _load_final_state(output: Path, *, system: str, seed: int):
    import torch

    path = _checkpoint_file(output, 10)
    state = torch.load(path, map_location="cpu", weights_only=False)
    if (
        state.get("system") != system
        or state.get("seed") != seed
        or state.get("completed_epochs") != 10
    ):
        raise S8RContractError("final checkpoint binding differs")
    return state, path


class _Predictor:
    def __init__(
        self,
        *,
        root: Path,
        output: Path,
        system: str,
        seed: int,
        s7_checkpoint: Path | None,
    ) -> None:
        import torch

        self.root = root
        self.system = system
        self.seed = seed
        self.device = "cuda:0"
        self.state, self.checkpoint = _load_final_state(
            output, system=system, seed=seed
        )
        self.pipeline = None
        self.transformer = None
        self.vae = None
        self.prompt = None
        self.text_ids = None
        self.head = None
        self.model = None
        self.s7_positive = None
        self.s7_negative = None
        if system == "Fresh_SegFormer_B2":
            self.model, _report = _segformer_model(root)
            self.model.load_state_dict(
                self.state["model_state"]["model"], strict=True
            )
            self.model.to(self.device).eval()
        elif system == "Fresh_S7":
            from diffusers import Flux2KleinPipeline

            adapter = (
                output
                / "checkpoint_epoch_10/pytorch_lora_weights.safetensors"
            )
            text_pipeline = Flux2KleinPipeline.from_pretrained(
                str(_model_root(root)),
                vae=None,
                transformer=None,
                scheduler=None,
                torch_dtype=torch.bfloat16,
                local_files_only=True,
            )
            text_pipeline.to(self.device)
            with torch.inference_mode():
                encoded, _ids = text_pipeline.encode_prompt(
                    prompt=[PROMPT, ""],
                    device=self.device,
                    max_sequence_length=512,
                )
            self.s7_positive = encoded[0:1].detach()
            self.s7_negative = encoded[1:2].detach()
            text_pipeline.to("cpu")
            del text_pipeline
            gc.collect()
            torch.cuda.empty_cache()
            self.pipeline = Flux2KleinPipeline.from_pretrained(
                str(_model_root(root)),
                text_encoder=None,
                tokenizer=None,
                torch_dtype=torch.bfloat16,
                local_files_only=True,
            )
            self.pipeline.load_lora_weights(
                str(adapter.parent),
                weight_name=adapter.name,
                adapter_name="fresh_s7",
                local_files_only=True,
            )
            self.pipeline.set_adapters(
                "fresh_s7", adapter_weights=1.0
            )
            progress = getattr(
                self.pipeline, "set_progress_bar_config", None
            )
            if callable(progress):
                progress(disable=True)
            self.pipeline.to(self.device)
        else:
            adapter = (
                s7_checkpoint / "pytorch_lora_weights.safetensors"
                if system == "Fresh_Arm_A" and s7_checkpoint is not None
                else None
            )
            self.pipeline, self.transformer, self.vae, self.prompt, self.text_ids = (
                _flux_components(
                    root,
                    seed=seed,
                    device=self.device,
                    adapter_path=adapter,
                    train_lora=False,
                )
            )
            self.head = _native_head(seed, device=self.device)
            self.head.load_state_dict(
                self.state["model_state"]["head"], strict=True
            )
            if system == "Fresh_Arm_B":
                _load_lora_state(
                    self.transformer,
                    self.state["model_state"]["lora"],
                )
            self.transformer.eval()
            self.head.eval()
        torch.cuda.synchronize(self.device)

    def predict(self, rgb: np.ndarray, *, sample_seed: int):
        import torch
        from torch.nn import functional as nn_functional

        if self.system == "Fresh_SegFormer_B2":
            tensor = _rgb_tensor(rgb, minus_one_to_one=False)
            tensor = _segformer_normalize_tensor(tensor)
            with torch.inference_mode(), torch.autocast(
                device_type="cuda", dtype=torch.bfloat16
            ):
                logits = self.model(tensor.unsqueeze(0).to(self.device))
                logits = nn_functional.interpolate(
                    logits,
                    size=(512, 512),
                    mode="bilinear",
                    align_corners=False,
                )
                return logits.argmax(1).squeeze(0).byte().cpu().numpy()
        if self.system == "Fresh_S7":
            generator = torch.Generator(device=self.device).manual_seed(
                sample_seed
            )
            with torch.inference_mode(), torch.autocast(
                device_type="cuda", dtype=torch.bfloat16
            ):
                output = self.pipeline(
                    image=Image.fromarray(rgb),
                    prompt_embeds=self.s7_positive,
                    negative_prompt_embeds=self.s7_negative,
                    height=512,
                    width=512,
                    num_inference_steps=10,
                    guidance_scale=4.0,
                    generator=generator,
                    output_type="pil",
                ).images[0]
            pixels = np.asarray(output.convert("RGB"), dtype=np.int32)
            distances = (
                pixels[:, :, None, :] - PALETTE.astype(np.int32)[None, None]
            )
            return np.square(distances).sum(axis=3).argmin(axis=2).astype(
                np.uint8
            )
        with torch.inference_mode(), torch.autocast(
            device_type="cuda", dtype=torch.bfloat16
        ):
            packed = _packed_input(self.pipeline, self.vae, rgb)
            native = _native_forward(
                self.transformer,
                packed,
                prompt=self.prompt,
                text_ids=self.text_ids,
            )
            return (
                _head_logits(self.head, native)
                .argmax(1)
                .squeeze(0)
                .byte()
                .cpu()
                .numpy()
            )

    def close(self) -> None:
        import torch

        for value in (self.model, self.pipeline, self.vae):
            if value is not None and hasattr(value, "to"):
                value.to("cpu")
        self.model = self.pipeline = self.vae = self.transformer = None
        self.head = self.prompt = self.text_ids = None
        self.s7_positive = self.s7_negative = None
        gc.collect()
        torch.cuda.empty_cache()


def _sample_seed(seed: int, sample_id: str) -> int:
    return int.from_bytes(
        hashlib.sha256(f"{seed}\0{sample_id}".encode()).digest()[:8],
        "big",
    ) % (2**63 - 1)


def run_development(
    *,
    project_root: Path,
    config: Mapping[str, Any],
    config_path: Path,
    config_sha256: str,
    system: str,
    seed: int,
    output_dir: Path,
    gpu: Mapping[str, Any],
    s7_checkpoint: Path | None,
) -> dict[str, Any]:
    """Run all development images under inference mode, without selection."""

    del config, config_path, gpu
    import torch

    rows = _manifest(
        project_root, split=DEVELOPMENT_SPLIT, labels_required=True
    )
    predictor = _Predictor(
        root=project_root,
        output=output_dir,
        system=system,
        seed=seed,
        s7_checkpoint=s7_checkpoint,
    )
    guard = SplitAccessGuard()
    valid = 0
    started = time.monotonic()
    with torch.inference_mode():
        for row in rows:
            record = [
                {
                    "sample_id": row["sample_id"],
                    "split": DEVELOPMENT_SPLIT,
                }
            ]
            guard.before_development_forward(record)
            prediction = predictor.predict(
                _load_rgb(project_root, row),
                sample_seed=_sample_seed(seed, str(row["sample_id"])),
            )
            if (
                prediction.shape == (512, 512)
                and prediction.dtype == np.uint8
                and set(np.unique(prediction)) <= {0, 1, 2}
            ):
                valid += 1
    predictor.close()
    receipt = {
        "status": "PASS" if valid == len(rows) else "FAIL",
        "system": system,
        "seed": seed,
        "split": DEVELOPMENT_SPLIT,
        "prediction_count": len(rows),
        "valid_prediction_count": valid,
        "valid_output_rate": valid / len(rows),
        "descriptive_metrics_used_for_selection": False,
        "retry_triggered": False,
        "checkpoint_selected": False,
        "development_backward_count": guard.development_backward_count,
        "development_optimizer_step_count": (
            guard.development_optimizer_step_count
        ),
        "guard": guard.receipt(),
        "config_sha256": config_sha256,
        "elapsed_seconds": time.monotonic() - started,
        "replication_confirmation_label_open_count": 0,
        "final_method_test_label_open_count": 0,
    }
    _atomic_json(output_dir / "development_receipt.json", receipt)
    return receipt


def _save_prediction(path: Path, prediction: np.ndarray) -> None:
    if prediction.shape != (512, 512) or prediction.dtype != np.uint8:
        raise S8RContractError("prediction shape/dtype differs")
    if not set(np.unique(prediction)) <= {0, 1, 2}:
        raise S8RContractError("prediction contains an invalid class ID")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        Image.fromarray(prediction, mode="L").save(temporary, format="PNG")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_confirmation_predictions(
    *,
    project_root: Path,
    config: Mapping[str, Any],
    config_path: Path,
    config_sha256: str,
    system: str,
    seed: int,
    output_dir: Path,
    gpu: Mapping[str, Any],
    s7_checkpoint: Path | None,
) -> dict[str, Any]:
    """Write exactly 844 label-free predictions for one formal cell."""

    del config, config_path, gpu
    rows = _manifest(
        project_root,
        split=REPLICATION_SPLIT,
        labels_required=False,
    )
    predictor = _Predictor(
        root=project_root,
        output=output_dir,
        system=system,
        seed=seed,
        s7_checkpoint=s7_checkpoint,
    )
    prediction_root = output_dir / "confirmation_predictions"
    inventory = []
    started = time.monotonic()
    for row in rows:
        sample_id = str(row["sample_id"])
        prediction = predictor.predict(
            _load_rgb(project_root, row),
            sample_seed=_sample_seed(seed, sample_id),
        )
        path = prediction_root / f"{int(row['index']):04d}.png"
        if path.exists():
            raise FileExistsError(path)
        _save_prediction(path, prediction)
        inventory.append(
            {
                "system": system,
                "seed": seed,
                "confirmation_index": int(row["index"]),
                "sample_id": sample_id,
                "prediction_path": path.relative_to(
                    project_root
                ).as_posix(),
                "prediction_sha256": sha256_file(path),
            }
        )
    predictor.close()
    receipt = {
        "status": "PASS",
        "system": system,
        "seed": seed,
        "prediction_count": len(inventory),
        "inventory": inventory,
        "inventory_sha256": hashlib.sha256(
            json.dumps(
                inventory,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest(),
        "config_sha256": config_sha256,
        "elapsed_seconds": time.monotonic() - started,
        "replication_confirmation_label_open_count": 0,
        "final_method_test_label_open_count": 0,
    }
    _atomic_json(output_dir / "confirmation_prediction_receipt.json", receipt)
    return receipt


def run_cost40_timing(
    *,
    project_root: Path,
    config: Mapping[str, Any],
    config_path: Path,
    config_sha256: str,
    system: str,
    seed: int,
    output_dir: Path,
    gpu: Mapping[str, Any],
    s7_checkpoint: Path | None,
) -> dict[str, Any]:
    """Run the frozen resident-core cost40 timing campaign."""

    del config_path
    import torch

    manifest_path = (
        project_root
        / config["efficiency"]["confirmatory_bootstrap"]["input_manifest"]
    )
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = payload.get("rows", payload.get("samples"))
    if not isinstance(rows, list) or len(rows) != 40:
        raise S8RContractError("cost40 manifest count differs")
    predictor = _Predictor(
        root=project_root,
        output=output_dir,
        system=system,
        seed=seed,
        s7_checkpoint=s7_checkpoint,
    )
    prepared = [
        (row, _load_rgb(project_root, row))
        for row in rows
    ]
    for warmup in range(10):
        row, rgb = prepared[warmup % len(prepared)]
        predictor.predict(
            rgb,
            sample_seed=_sample_seed(seed, str(row["sample_id"])),
        )
    timing_rows = []
    torch.cuda.reset_peak_memory_stats("cuda:0")
    for row, rgb in prepared:
        for repeat in (1, 2, 3):
            torch.cuda.synchronize("cuda:0")
            started = time.perf_counter()
            predictor.predict(
                rgb,
                sample_seed=_sample_seed(seed, str(row["sample_id"])),
            )
            torch.cuda.synchronize("cuda:0")
            latency_ms = (time.perf_counter() - started) * 1000.0
            if not math.isfinite(latency_ms) or latency_ms <= 0:
                raise S8RContractError("timing latency is invalid")
            timing_rows.append(
                {
                    "system": system,
                    "seed": seed,
                    "sample_id": row["sample_id"],
                    "recording_session_key": row[
                        "recording_session_key"
                    ],
                    "source_set": row["source_set"],
                    "repeat": repeat,
                    "latency_ms": latency_ms,
                }
            )
    predictor.close()
    receipt = {
        "status": "PASS",
        "system": system,
        "seed": seed,
        "gpu_uuid": gpu["visible_gpu_uuid"],
        "warmups": 10,
        "timing_rows": timing_rows,
        "observation_count": len(timing_rows),
        "peak_allocated_bytes": int(
            torch.cuda.max_memory_allocated("cuda:0")
        ),
        "peak_reserved_bytes": int(
            torch.cuda.max_memory_reserved("cuda:0")
        ),
        "config_sha256": config_sha256,
        "replication_confirmation_label_open_count": 0,
        "final_method_test_label_open_count": 0,
    }
    _atomic_json(output_dir / "cost40_timing_receipt.json", receipt)
    return receipt
