"""Deterministic two-process training for the frozen S8 native-field heads."""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import time
import uuid
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from researchos.s8_config import FINAL_STEPS, MIDPOINT_STEPS, SEEDS

CLASS_WEIGHTS = (0.25, 0.467545936319718, 2.282454063680282)
HEAD_IN_FEATURES = 128
HEAD_OUT_FEATURES = 16 * 16 * 3
HEAD_PARAMETER_COUNT = 99072
MAX_GPU_HOURS_PER_CELL = 200.0
MAX_GPU_HOURS_ALL_ATTEMPTS = 1150.0


class ResourceStop(RuntimeError):
    """A preregistered engineering budget stop, not a scientific invalidation."""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode()
    ).hexdigest()


def atomic_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(dict(payload), indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def append_jsonl_fsync(path: str | Path, payload: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(dict(payload), sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def stable_sample_order(sample_ids: Sequence[str], seed: int) -> list[int]:
    if seed not in SEEDS:
        raise ValueError("seed is not preregistered")
    if not sample_ids or len(set(sample_ids)) != len(sample_ids):
        raise ValueError("sample IDs must be non-empty and unique")
    return sorted(
        range(len(sample_ids)),
        key=lambda index: hashlib.sha256(
            f"{seed}\0{sample_ids[index]}".encode()
        ).hexdigest(),
    )


def sample_order_sha256(sample_ids: Sequence[str], seed: int) -> str:
    order = stable_sample_order(sample_ids, seed)
    return canonical_json_sha256([sample_ids[index] for index in order])


def verify_training_journal(
    path: str | Path,
    *,
    sample_ids: Sequence[str],
    seed: int,
    start_step: int,
    target_step: int,
) -> dict[str, Any]:
    """Read back a complete segment journal and prove its step/sample order."""
    source = Path(path)
    rows: list[dict[str, Any]] = []
    with source.open("r", encoding="utf-8") as stream:
        for line_number, raw in enumerate(stream, start=1):
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"training journal has invalid JSON at line {line_number}"
                ) from exc
            if not isinstance(row, dict):
                raise RuntimeError("training journal rows must be JSON objects")
            rows.append(row)
    expected_count = target_step - start_step
    if len(rows) != expected_count:
        raise RuntimeError(
            f"training journal row count mismatch: {len(rows)} != {expected_count}"
        )
    order = stable_sample_order(sample_ids, seed)
    for offset, row in enumerate(rows):
        step = start_step + offset + 1
        sample_index = order[(step - 1) % len(sample_ids)]
        if (
            row.get("step") != step
            or row.get("sample_index") != sample_index
            or row.get("sample_id") != sample_ids[sample_index]
        ):
            raise RuntimeError(f"training journal order mismatch at step {step}")
    return {
        "path": str(source),
        "sha256": sha256_file(source),
        "row_count": len(rows),
        "start_step": start_step + 1,
        "end_step": target_step,
        "sample_order_sha256": sample_order_sha256(sample_ids, seed),
        "verified": True,
    }


def seed_everything(seed: int) -> None:
    if seed not in SEEDS:
        raise ValueError("seed is not preregistered")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed % (2**32))
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def create_head(seed: int, *, device: str | None = None):
    """Use the frozen torch Linear.reset_parameters initialization after an explicit seed."""
    seed_everything(seed)
    import torch

    head = torch.nn.Linear(HEAD_IN_FEATURES, HEAD_OUT_FEATURES, bias=True)
    if sum(parameter.numel() for parameter in head.parameters()) != HEAD_PARAMETER_COUNT:
        raise RuntimeError("head parameter count differs from the preregistration")
    return head.to(device) if device is not None else head


def unpatchify_head_logits(projected):
    """Exact frozen q/h/w/dy/dx/class mapping."""
    if tuple(projected.shape[1:]) != (1024, 768):
        raise ValueError(f"head output must be [B,1024,768], observed {tuple(projected.shape)}")
    batch = projected.shape[0]
    return (
        projected.reshape(batch, 32, 32, 16, 16, 3)
        .permute(0, 1, 3, 2, 4, 5)
        .reshape(batch, 512, 512, 3)
        .permute(0, 3, 1, 2)
        .contiguous()
    )


def head_logits(head, native_tokens):
    if tuple(native_tokens.shape[1:]) != (1024, 128):
        raise ValueError("native tokens must be [B,1024,128]")
    return unpatchify_head_logits(head(native_tokens))


def latent_image_ids(batch_size: int, *, device: Any):
    import torch

    if batch_size <= 0:
        raise ValueError("batch size must be positive")
    h = torch.arange(32, dtype=torch.int64)
    w = torch.arange(32, dtype=torch.int64)
    zero = torch.arange(1, dtype=torch.int64)
    ids = torch.cartesian_prod(zero, h, w, zero)
    return ids.unsqueeze(0).expand(batch_size, -1, -1).to(device=device)


def native_transformer_forward(
    transformer,
    packed_tokens,
    *,
    prompt_embeds,
    txt_ids,
    img_ids=None,
):
    import torch

    if packed_tokens.ndim != 3 or tuple(packed_tokens.shape[1:]) != (1024, 128):
        raise ValueError("packed image tokens must be [B,1024,128]")
    batch = packed_tokens.shape[0]
    device = packed_tokens.device
    ids = latent_image_ids(batch, device=device) if img_ids is None else img_ids.to(device)
    text = prompt_embeds
    text_ids = txt_ids
    if text.shape[0] == 1 and batch != 1:
        text = text.expand(batch, -1, -1)
    if text_ids.shape[0] == 1 and batch != 1:
        text_ids = text_ids.expand(batch, -1, -1)
    text = text.to(device=device, dtype=packed_tokens.dtype)
    text_ids = text_ids.to(device=device, dtype=torch.int64)
    timestep = torch.full((batch,), 0.999, dtype=torch.float32, device=device)
    guidance = torch.zeros(batch, dtype=torch.float32, device=device)
    output = transformer(
        hidden_states=packed_tokens,
        encoder_hidden_states=text,
        timestep=timestep,
        img_ids=ids,
        txt_ids=text_ids,
        guidance=guidance,
        return_dict=False,
    )[0]
    if tuple(output.shape) != (batch, 1024, 128):
        raise RuntimeError(f"native transformer output contract failed: {tuple(output.shape)}")
    return output


def canonical_tensor_digest(
    named_tensors: Iterable[tuple[str, Any]],
    *,
    include_per_tensor: bool = True,
) -> dict[str, Any]:
    """Hash tensors without converting bfloat16 through unsupported NumPy dtypes."""
    import torch

    combined = hashlib.sha256()
    rows: list[dict[str, Any]] = []
    tensor_count = 0
    total_numel = 0
    for name, tensor in sorted(named_tensors, key=lambda item: item[0]):
        value = tensor.detach().contiguous()
        raw = value.reshape(-1).view(torch.uint8).cpu().numpy().tobytes()
        tensor_digest = hashlib.sha256(raw).hexdigest()
        metadata = json.dumps(
            {
                "name": name,
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "sha256": tensor_digest,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        combined.update(len(metadata).to_bytes(8, "big"))
        combined.update(metadata)
        tensor_count += 1
        total_numel += value.numel()
        if include_per_tensor:
            rows.append(
                {
                    "name": name,
                    "shape": list(value.shape),
                    "dtype": str(value.dtype),
                    "sha256": tensor_digest,
                }
            )
    return {
        "algorithm": "sha256(metadata(name,shape,dtype,tensor_sha256))",
        "sha256": combined.hexdigest(),
        "tensor_count": tensor_count,
        "total_numel": total_numel,
        "tensors": rows if include_per_tensor else None,
    }


def frozen_transformer_tensors(transformer) -> Iterable[tuple[str, Any]]:
    """Canonical state-dict tensors excluding PEFT LoRA entries."""
    for name, value in transformer.state_dict(keep_vars=True).items():
        if "lora_" not in name.casefold():
            yield f"state_dict:{name}", value


def module_state_tensor_digest(module) -> dict[str, Any]:
    """Canonical digest over the module's exact persistent state_dict contract."""
    return canonical_tensor_digest(
        [
            (f"state_dict:{name}", value)
            for name, value in module.state_dict(keep_vars=True).items()
        ]
    )


def trainable_tensor_digest(transformer, head, arm: str) -> dict[str, Any]:
    tensors: list[tuple[str, Any]] = [
        (f"head.{name}", parameter) for name, parameter in head.named_parameters()
    ]
    if arm == "B":
        tensors.extend(
            (f"transformer.{name}", parameter)
            for name, parameter in transformer.named_parameters()
            if parameter.requires_grad
        )
    return canonical_tensor_digest(tensors)


def assert_trainable_whitelist(transformer, head, arm: str) -> dict[str, Any]:
    if arm not in {"A", "B"}:
        raise ValueError("arm must be A or B")
    head_names = [f"head.{name}" for name, parameter in head.named_parameters() if parameter.requires_grad]
    if head_names != ["head.weight", "head.bias"]:
        raise RuntimeError(f"head whitelist mismatch: {head_names}")
    transformer_trainable: list[str] = []
    if transformer is not None:
        transformer_trainable = [
            name for name, parameter in transformer.named_parameters() if parameter.requires_grad
        ]
    if arm == "A" and transformer_trainable:
        raise RuntimeError("A transformer must be fully frozen")
    if arm == "B" and (
        not transformer_trainable
        or any("lora_" not in name.casefold() for name in transformer_trainable)
    ):
        raise RuntimeError("B may train only non-empty LoRA parameters")
    return {
        "arm": arm,
        "head_parameter_names": head_names,
        "transformer_parameter_names": transformer_trainable,
        "transformer_trainable_count": sum(
            parameter.numel()
            for _name, parameter in (
                transformer.named_parameters() if transformer is not None else []
            )
            if parameter.requires_grad
        ),
    }


def build_optimizer(transformer, head, arm: str):
    import torch

    whitelist = assert_trainable_whitelist(transformer, head, arm)
    groups: list[dict[str, Any]] = [
        {
            "params": [head.weight],
            "lr": 1e-3,
            "weight_decay": 0.01,
            "group_name": "head_weight",
            "parameter_names": ["head.weight"],
        },
        {
            "params": [head.bias],
            "lr": 1e-3,
            "weight_decay": 0.01,
            "group_name": "head_bias",
            "parameter_names": ["head.bias"],
        },
    ]
    if arm == "B":
        lora_named = [
            (name, parameter)
            for name, parameter in transformer.named_parameters()
            if parameter.requires_grad
        ]
        groups.append(
            {
                "params": [parameter for _name, parameter in lora_named],
                "lr": 1e-4,
                "weight_decay": 0.01,
                "group_name": "transformer_lora",
                "parameter_names": [name for name, _parameter in lora_named],
            }
        )
    optimizer = torch.optim.AdamW(
        groups,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _step: 1.0)
    optimizer_parameter_ids = {
        id(parameter) for group in optimizer.param_groups for parameter in group["params"]
    }
    allowed_parameter_ids = {id(parameter) for parameter in head.parameters()}
    if arm == "B":
        allowed_parameter_ids.update(
            id(parameter)
            for parameter in transformer.parameters()
            if parameter.requires_grad
        )
    if optimizer_parameter_ids != allowed_parameter_ids:
        raise RuntimeError("optimizer parameter IDs differ from the trainable whitelist")
    return optimizer, scheduler, whitelist


def _atomic_torch_save(path: Path, payload: Any) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _trainable_state(transformer, head, arm: str) -> dict[str, Any]:
    state = {
        f"head.{name}": parameter.detach().cpu().contiguous()
        for name, parameter in head.named_parameters()
    }
    if arm == "B":
        state.update(
            {
                f"transformer.{name}": parameter.detach().cpu().contiguous()
                for name, parameter in transformer.named_parameters()
                if parameter.requires_grad
            }
        )
    return state


def _save_trainable_safetensors(path: Path, transformer, head, arm: str) -> None:
    from safetensors.torch import save_file

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        save_file(_trainable_state(transformer, head, arm), str(temporary))
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_trainable_safetensors(path: str | Path, transformer, head, arm: str) -> None:
    from safetensors.torch import load_file

    expected = _trainable_state(transformer, head, arm)
    observed = load_file(str(path), device="cpu")
    if set(observed) != set(expected):
        raise RuntimeError("checkpoint trainable tensor names differ from the current model")
    destinations = {f"head.{name}": parameter for name, parameter in head.named_parameters()}
    if arm == "B":
        destinations.update(
            {
                f"transformer.{name}": parameter
                for name, parameter in transformer.named_parameters()
                if parameter.requires_grad
            }
        )
    import torch

    with torch.no_grad():
        for name, value in observed.items():
            destination = destinations[name]
            if tuple(value.shape) != tuple(destination.shape):
                raise RuntimeError(f"checkpoint tensor shape mismatch: {name}")
            destination.copy_(value.to(device=destination.device, dtype=destination.dtype))


def _rng_state() -> dict[str, Any]:
    import torch

    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def _restore_rng_state(payload: Mapping[str, Any]) -> None:
    import torch

    random.setstate(payload["python"])
    np.random.set_state(payload["numpy"])
    torch.set_rng_state(payload["torch_cpu"])
    if torch.cuda.is_available():
        torch.cuda.set_rng_state_all(payload["torch_cuda"])


def checkpoint_inventory(
    path: str | Path, *, exclude_receipt: bool = True
) -> list[dict[str, Any]]:
    root = Path(path)
    return [
        {
            "path": item.relative_to(root).as_posix(),
            "size_bytes": item.stat().st_size,
            "sha256": sha256_file(item),
        }
        for item in sorted(root.rglob("*"))
        if item.is_file() and (not exclude_receipt or item.name != "receipt.json")
    ]


def save_checkpoint(
    output_dir: str | Path,
    *,
    step: int,
    arm: str,
    seed: int,
    transformer,
    head,
    optimizer,
    scheduler,
    binding: Mapping[str, Any],
    producer_context: Mapping[str, Any],
    sample_order_sha256_value: str,
) -> dict[str, Any]:
    if step not in {MIDPOINT_STEPS, FINAL_STEPS} and not binding.get("development_smoke"):
        raise ValueError("formal checkpoint step is not preregistered")
    output = Path(output_dir)
    target = output / f"checkpoint-{step}"
    if target.exists():
        raise FileExistsError(f"refusing to replace checkpoint: {target}")
    temporary = output / f".checkpoint-{step}.{uuid.uuid4().hex}.tmp"
    temporary.mkdir(parents=True)
    try:
        _save_trainable_safetensors(
            temporary / "trainable.safetensors", transformer, head, arm
        )
        _atomic_torch_save(temporary / "optimizer.pt", optimizer.state_dict())
        _atomic_torch_save(temporary / "scheduler.pt", scheduler.state_dict())
        _atomic_torch_save(temporary / "rng_state.pt", _rng_state())
        state = {
            "schema_version": 1,
            "arm": arm,
            "seed": seed,
            "global_step": step,
            "producer_process_id": os.getpid(),
            "sample_cursor": step % int(binding["train_count"]),
            "binding": dict(binding),
            "binding_sha256": canonical_json_sha256(dict(binding)),
            "sample_order_sha256": sample_order_sha256_value,
            "producer_context": dict(producer_context),
            "trainable_digest": trainable_tensor_digest(transformer, head, arm),
        }
        atomic_json(temporary / "state.json", state)
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            raise RuntimeError(f"incomplete checkpoint staging remains: {temporary}")
    inventory = checkpoint_inventory(target)
    receipt_unsigned = {
        "schema_version": 1,
        "task_id": "S8_Ex01_native_field_readout",
        "status": "COMPLETE",
        "path": str(target),
        "arm": arm,
        "seed": seed,
        "global_step": step,
        "binding_sha256": state["binding_sha256"],
        "sample_order_sha256": sample_order_sha256_value,
        "state_sha256": sha256_file(target / "state.json"),
        "inventory": inventory,
        "inventory_sha256": canonical_json_sha256(inventory),
    }
    receipt = {
        **receipt_unsigned,
        "payload_sha256": canonical_json_sha256(receipt_unsigned),
    }
    atomic_json(target / "receipt.json", receipt)
    return receipt


def verify_checkpoint_receipt(
    checkpoint: str | Path,
    *,
    expected_arm: str,
    expected_seed: int,
    expected_step: int,
    expected_binding: Mapping[str, Any],
    expected_sample_order_sha256: str | None = None,
) -> dict[str, Any]:
    root = Path(checkpoint).resolve(strict=True)
    if not root.is_dir() or root.is_symlink():
        raise RuntimeError("checkpoint must be a non-symlink directory")
    receipt_path = root / "receipt.json"
    state_path = root / "state.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    state = json.loads(state_path.read_text(encoding="utf-8"))
    unsigned = {key: value for key, value in receipt.items() if key != "payload_sha256"}
    inventory = checkpoint_inventory(root)
    expected_inventory_paths = {
        "optimizer.pt",
        "rng_state.pt",
        "scheduler.pt",
        "state.json",
        "trainable.safetensors",
    }
    observed_inventory_paths = {row.get("path") for row in inventory}
    binding_sha = canonical_json_sha256(dict(expected_binding))
    checks = {
        "receipt_schema": receipt.get("schema_version") == 1,
        "receipt_task": receipt.get("task_id") == "S8_Ex01_native_field_readout",
        "receipt_payload": receipt.get("payload_sha256") == canonical_json_sha256(unsigned),
        "receipt_status": receipt.get("status") == "COMPLETE",
        "receipt_arm": receipt.get("arm") == expected_arm,
        "receipt_seed": receipt.get("seed") == expected_seed,
        "receipt_step": receipt.get("global_step") == expected_step,
        "receipt_binding": receipt.get("binding_sha256") == binding_sha,
        "receipt_state": receipt.get("state_sha256") == sha256_file(state_path),
        "receipt_inventory": receipt.get("inventory") == inventory,
        "receipt_inventory_sha": receipt.get("inventory_sha256")
        == canonical_json_sha256(inventory),
        "receipt_inventory_paths": observed_inventory_paths
        == expected_inventory_paths,
        "receipt_inventory_nonempty": all(
            isinstance(row.get("size_bytes"), int) and row["size_bytes"] > 0
            for row in inventory
        ),
        "state_schema": state.get("schema_version") == 1,
        "state_arm": state.get("arm") == expected_arm,
        "state_seed": state.get("seed") == expected_seed,
        "state_step": state.get("global_step") == expected_step,
        "state_binding": state.get("binding") == dict(expected_binding),
        "state_binding_sha": state.get("binding_sha256") == binding_sha,
        "state_producer_context": isinstance(state.get("producer_context"), dict),
        "state_trainable_digest": isinstance(state.get("trainable_digest"), dict)
        and isinstance(state["trainable_digest"].get("sha256"), str),
    }
    if expected_sample_order_sha256 is not None:
        checks.update(
            {
                "receipt_sample_order": receipt.get("sample_order_sha256")
                == expected_sample_order_sha256,
                "state_sample_order": state.get("sample_order_sha256")
                == expected_sample_order_sha256,
            }
        )
    if not all(checks.values()):
        raise RuntimeError(f"checkpoint receipt verification failed: {checks}")
    return {"receipt": receipt, "state": state, "checks": checks}


def load_checkpoint(
    checkpoint: str | Path,
    *,
    expected_step: int,
    arm: str,
    seed: int,
    transformer,
    head,
    optimizer,
    scheduler,
    expected_binding: Mapping[str, Any],
    expected_sample_order_sha256: str,
) -> dict[str, Any]:
    import torch

    root = Path(checkpoint)
    verified = verify_checkpoint_receipt(
        root,
        expected_arm=arm,
        expected_seed=seed,
        expected_step=expected_step,
        expected_binding=expected_binding,
        expected_sample_order_sha256=expected_sample_order_sha256,
    )
    state = verified["state"]
    checks = {
        "arm": state.get("arm") == arm,
        "seed": state.get("seed") == seed,
        "global_step": state.get("global_step") == expected_step,
        "binding": state.get("binding") == dict(expected_binding),
        "sample_cursor": state.get("sample_cursor")
        == expected_step % int(expected_binding["train_count"]),
        "sample_order_sha256": state.get("sample_order_sha256")
        == expected_sample_order_sha256,
    }
    if not all(checks.values()):
        raise RuntimeError(f"checkpoint producer binding failed: {checks}")
    load_trainable_safetensors(root / "trainable.safetensors", transformer, head, arm)
    optimizer.load_state_dict(torch.load(root / "optimizer.pt", map_location="cpu", weights_only=True))
    scheduler.load_state_dict(torch.load(root / "scheduler.pt", map_location="cpu", weights_only=True))
    rng = torch.load(root / "rng_state.pt", map_location="cpu", weights_only=False)
    _restore_rng_state(rng)
    observed = trainable_tensor_digest(transformer, head, arm)
    if observed["sha256"] != state["trainable_digest"]["sha256"]:
        raise RuntimeError("loaded trainable tensor digest differs from checkpoint")
    return {
        "checks": {**verified["checks"], **checks},
        "state": state,
        "fresh_process_resume": True,
    }


def load_cache_tensors(cache_dir: str | Path) -> dict[str, Any]:
    """Read an immutable S8 cache emitted by s8_native_data."""
    from safetensors.torch import load_file

    root = Path(cache_dir)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    tensor_file = root / "tensors.safetensors"
    if sha256_file(tensor_file) != manifest.get("tensor_file_sha256"):
        raise RuntimeError("cache tensor file SHA256 mismatch")
    tensors = load_file(str(tensor_file), device="cpu")
    if set(tensors) != {"labels", "tokens"}:
        raise RuntimeError("cache must contain exactly labels and tokens")
    sample_ids = manifest.get("sample_ids")
    if not isinstance(sample_ids, list) or len(sample_ids) != tensors["tokens"].shape[0]:
        raise RuntimeError("cache sample inventory mismatch")
    if tuple(tensors["tokens"].shape[1:]) != (1024, 128):
        raise RuntimeError("cache tokens have an invalid shape")
    if tuple(tensors["labels"].shape[1:]) != (512, 512):
        raise RuntimeError("cache labels have an invalid shape")
    return {"manifest": manifest, "sample_ids": sample_ids, **tensors}


def load_prompt_cache(path: str | Path) -> dict[str, Any]:
    from safetensors.torch import load_file

    source = Path(path)
    tensors = load_file(str(source), device="cpu")
    if set(tensors) != {"prompt_embeds", "txt_ids"}:
        raise RuntimeError("prompt cache has an unexpected schema")
    if tensors["prompt_embeds"].ndim != 3 or tensors["prompt_embeds"].shape[0] != 1:
        raise RuntimeError("prompt embedding shape mismatch")
    if tensors["txt_ids"].ndim != 3 or tensors["txt_ids"].shape[0] != 1:
        raise RuntimeError("text ID shape mismatch")
    if tensors["txt_ids"].shape[-1] != 4:
        raise RuntimeError("text IDs must end in four coordinates")
    return tensors


def run_training_segment(
    *,
    arm: str,
    seed: int,
    transformer,
    head,
    cache: Mapping[str, Any],
    prompt_cache: Mapping[str, Any] | None,
    output_dir: str | Path,
    mode: str,
    binding: Mapping[str, Any],
    device: str = "cuda:0",
    smoke_steps: int | None = None,
    all_attempts_gpu_hours_before: float = 0.0,
) -> dict[str, Any]:
    """Run one exact process half; formal process_b must resume checkpoint-7035."""
    import torch
    import torch.nn.functional as functional

    if arm not in {"A", "B"} or seed not in SEEDS:
        raise ValueError("unknown arm or seed")
    if mode not in {"smoke", "process_a", "process_b"}:
        raise ValueError("mode must be smoke, process_a, or process_b")
    sample_ids = list(cache["sample_ids"])
    train_count = len(sample_ids)
    if mode == "smoke" and train_count != 20:
        raise RuntimeError("engineering smoke requires the exact 20-sample train-dev cache")
    if mode != "smoke" and train_count != 1407:
        raise RuntimeError("formal training requires exactly 1407 samples")
    if int(binding["train_count"]) != train_count:
        raise RuntimeError("training binding count differs from cache")
    order = stable_sample_order(sample_ids, seed)
    order_sha256 = sample_order_sha256(sample_ids, seed)
    if binding.get("sample_order_sha256") != order_sha256:
        raise RuntimeError("training binding sample order differs from the cache")
    if arm == "A" and cache["manifest"].get("cache_kind") != "native_tokens_bf16":
        raise RuntimeError("A requires a frozen BF16 native-token cache")
    if arm == "B" and cache["manifest"].get("cache_kind") != "vae_packed_tokens_bf16":
        raise RuntimeError("B requires frozen VAE packed tokens and online transformer")
    if arm == "A" and (
        cache["manifest"].get("arm") != "A"
        or cache["manifest"].get("seed") != seed
        or cache["manifest"].get("s7_adapter_sha256")
        != binding.get("s7_adapter_sha256")
    ):
        raise RuntimeError("A cache seed/arm/S7-adapter binding mismatch")
    if arm == "B" and (
        cache["manifest"].get("arm") is not None
        or cache["manifest"].get("seed") is not None
        or cache["manifest"].get("s7_adapter_sha256") is not None
    ):
        raise RuntimeError("B requires the seed-independent shared VAE cache")
    if cache["manifest"].get("formal_cache") is (mode == "smoke"):
        raise RuntimeError("cache formal/development status differs from training mode")
    if cache["tokens"].dtype != torch.bfloat16:
        raise RuntimeError("formal token cache must be BF16")
    if prompt_cache is None:
        raise RuntimeError("both arms require the frozen prompt-cache binding")

    if mode == "smoke":
        if smoke_steps is None or not 1 <= smoke_steps <= 5:
            raise ValueError("smoke steps must be in [1,5]")
        start_step, target_step = 0, smoke_steps
    elif mode == "process_a":
        start_step, target_step = 0, MIDPOINT_STEPS
    else:
        start_step, target_step = MIDPOINT_STEPS, FINAL_STEPS

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    evidence_path = output / f"evidence_{mode}.json"
    journal_path = output / f"metrics_{mode}.jsonl"
    if journal_path.exists():
        raise FileExistsError(f"refusing to replace training journal: {journal_path}")

    optimizer, scheduler, whitelist = build_optimizer(transformer, head, arm)
    head_initialization_digest = canonical_tensor_digest(
        [(f"head.{name}", parameter) for name, parameter in head.named_parameters()]
    )
    fresh_trainable_initialization_digest = trainable_tensor_digest(
        transformer, head, arm
    )
    resume = None
    producer_evidence = None
    producer_pid = None
    producer_elapsed_seconds = 0.0
    if mode == "process_b":
        producer = output / f"checkpoint-{MIDPOINT_STEPS}"
        producer_evidence_path = output / "evidence_process_a.json"
        producer_evidence = json.loads(
            producer_evidence_path.read_text(encoding="utf-8")
        )
        producer_state = json.loads(
            (producer / "state.json").read_text(encoding="utf-8")
        )
        producer_pid = int(producer_state["producer_process_id"])
        producer_elapsed_seconds = float(producer_evidence.get("elapsed_seconds", -1.0))
        if producer_elapsed_seconds < 0:
            raise RuntimeError("process A elapsed time is missing")
        if producer_pid == os.getpid():
            raise RuntimeError("midpoint resume must occur in a fresh process")
        producer_checks = {
            "status": producer_evidence.get("status") == "PASS_PROCESS_A",
            "arm": producer_evidence.get("arm") == arm,
            "seed": producer_evidence.get("seed") == seed,
            "binding": producer_evidence.get("binding") == dict(binding),
            "final_step": producer_evidence.get("final_global_step") == MIDPOINT_STEPS,
            "process_id": producer_evidence.get("process_id") == producer_pid,
            "journal": producer_evidence.get("journal_sha256")
            == sha256_file(output / "metrics_process_a.jsonl"),
            "checkpoint_receipt": producer_evidence.get("checkpoint", {}).get(
                "payload_sha256"
            )
            == json.loads((producer / "receipt.json").read_text(encoding="utf-8")).get(
                "payload_sha256"
            ),
            "head_initialization": producer_evidence.get(
                "head_initialization_digest", {}
            ).get("sha256")
            == head_initialization_digest["sha256"],
            "fresh_trainable_initialization": producer_evidence.get(
                "fresh_trainable_initialization_digest", {}
            ).get("sha256")
            == fresh_trainable_initialization_digest["sha256"],
            "sample_order": producer_state.get("sample_order_sha256")
            == order_sha256,
            "state_journal": producer_state.get("producer_context", {}).get(
                "journal_sha256"
            )
            == producer_evidence.get("journal_sha256"),
            "state_head_initialization": producer_state.get(
                "producer_context", {}
            ).get("head_initialization_sha256")
            == producer_evidence.get("head_initialization_digest", {}).get("sha256"),
            "state_trainable_initialization": producer_state.get(
                "producer_context", {}
            ).get("fresh_trainable_initialization_sha256")
            == producer_evidence.get(
                "fresh_trainable_initialization_digest", {}
            ).get("sha256"),
            "state_frozen_after": producer_state.get("producer_context", {}).get(
                "frozen_transformer_after_sha256"
            )
            == (
                producer_evidence.get("frozen_transformer_after", {}).get("sha256")
                if arm == "B"
                else None
            ),
        }
        if not all(producer_checks.values()):
            raise RuntimeError(
                f"fresh-process producer evidence invalid: {producer_checks}"
            )
        verify_training_journal(
            output / "metrics_process_a.jsonl",
            sample_ids=sample_ids,
            seed=seed,
            start_step=0,
            target_step=MIDPOINT_STEPS,
        )
        resume = load_checkpoint(
            producer,
            expected_step=MIDPOINT_STEPS,
            arm=arm,
            seed=seed,
            transformer=transformer,
            head=head,
            optimizer=optimizer,
            scheduler=scheduler,
            expected_binding=binding,
            expected_sample_order_sha256=order_sha256,
        )
    else:
        seed_everything(seed)

    head.train()
    if transformer is not None:
        transformer.train() if arm == "B" else transformer.eval()
    class_weights = torch.tensor(CLASS_WEIGHTS, dtype=torch.float32, device=device)
    parameters = [parameter for group in optimizer.param_groups for parameter in group["params"]]
    init_digest = trainable_tensor_digest(transformer, head, arm)
    frozen_before = (
        canonical_tensor_digest(frozen_transformer_tensors(transformer))
        if arm == "B"
        else None
    )
    if arm == "B":
        if any(
            parameter.requires_grad
            for name, parameter in transformer.named_parameters()
            if "lora_" not in name.casefold()
        ):
            raise RuntimeError("a frozen base transformer parameter requires grad")
        if mode == "process_b" and producer_evidence.get(
            "frozen_transformer_after", {}
        ).get("sha256") != frozen_before["sha256"]:
            raise RuntimeError("B frozen base digest changed across the fresh process")
        if mode == "process_b" and producer_evidence.get(
            "final_trainable_digest", {}
        ).get("sha256") != init_digest["sha256"]:
            raise RuntimeError("resumed trainable digest differs from process A final")
    transformer_forward_count = 0
    forward_hook = None
    if arm == "B":

        def count_transformer_forward(_module, _inputs, _output):
            nonlocal transformer_forward_count
            transformer_forward_count += 1

        forward_hook = transformer.register_forward_hook(count_transformer_forward)
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    baseline_allocated = int(torch.cuda.memory_allocated(device))
    baseline_reserved = int(torch.cuda.memory_reserved(device))
    started = time.monotonic()
    base_evidence = {
        "schema_version": 1,
        "task_id": "S8_Ex01_native_field_readout",
        "arm": arm,
        "seed": seed,
        "mode": mode,
        "attempt": int(binding["attempt"]),
        "process_id": os.getpid(),
        "producer_process_id": producer_pid,
        "start_global_step": start_step,
        "target_global_step": target_step,
        "train_count": train_count,
        "cache_manifest_sha256": cache["manifest"].get("manifest_sha256"),
        "sample_order_sha256": order_sha256,
        "binding": dict(binding),
        "whitelist": whitelist,
        "head_initialization_digest": head_initialization_digest,
        "fresh_trainable_initialization_digest": fresh_trainable_initialization_digest,
        "segment_initial_trainable_digest": init_digest,
        "frozen_transformer_before": frozen_before,
        "resume": resume,
        "gpu_baseline_allocated_bytes": baseline_allocated,
        "gpu_baseline_reserved_bytes": baseline_reserved,
        "producer_elapsed_seconds": producer_elapsed_seconds,
        "all_attempts_gpu_hours_before": all_attempts_gpu_hours_before,
        "final_test_access_count": 0,
        "validation_access_count": 0,
    }
    atomic_json(evidence_path, {**base_evidence, "status": "RUNNING"})
    losses: list[float] = []
    try:
        for global_step in range(start_step + 1, target_step + 1):
            sample_index = order[(global_step - 1) % train_count]
            token = cache["tokens"][sample_index : sample_index + 1].to(
                device=device, dtype=torch.bfloat16, non_blocking=False
            )
            label = cache["labels"][sample_index : sample_index + 1].to(
                device=device, dtype=torch.int64, non_blocking=False
            )
            unique = set(int(value) for value in torch.unique(label).tolist())
            if not unique <= {0, 1, 2}:
                raise RuntimeError(f"unexpected label IDs: {sorted(unique)}")
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                if arm == "A":
                    native = token
                else:
                    native = native_transformer_forward(
                        transformer,
                        token,
                        prompt_embeds=prompt_cache["prompt_embeds"],
                        txt_ids=prompt_cache["txt_ids"],
                    )
                logits = head_logits(head, native)
            loss = functional.cross_entropy(
                logits.float(), label, weight=class_weights, reduction="mean"
            )
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("training loss is non-finite")
            loss.backward()
            for name, parameter in head.named_parameters():
                if parameter.grad is None or not bool(torch.isfinite(parameter.grad).all()):
                    raise RuntimeError(f"head gradient missing or non-finite: {name}")
            if arm == "B" and not any(
                parameter.grad is not None
                and bool(torch.isfinite(parameter.grad).all())
                and bool(torch.count_nonzero(parameter.grad))
                for name, parameter in transformer.named_parameters()
                if parameter.requires_grad and "lora_" in name.casefold()
            ):
                raise RuntimeError("B has no finite non-zero LoRA gradient")
            if arm == "B" and any(
                parameter.grad is not None
                for name, parameter in transformer.named_parameters()
                if "lora_" not in name.casefold()
            ):
                raise RuntimeError("a frozen base transformer parameter acquired a grad")
            grad_norm = torch.nn.utils.clip_grad_norm_(parameters, max_norm=1.0)
            if not bool(torch.isfinite(grad_norm)):
                raise FloatingPointError("gradient norm is non-finite")
            optimizer.step()
            scheduler.step()
            numeric_loss = float(loss.detach().cpu())
            losses.append(numeric_loss)
            append_jsonl_fsync(
                journal_path,
                {
                    "step": global_step,
                    "sample_id": sample_ids[sample_index],
                    "sample_index": sample_index,
                    "loss": numeric_loss,
                    "grad_norm": float(grad_norm.detach().cpu()),
                    "head_lr": float(optimizer.param_groups[0]["lr"]),
                    "lora_lr": (
                        float(optimizer.param_groups[2]["lr"]) if arm == "B" else None
                    ),
                },
            )
            combined_elapsed_seconds = (
                producer_elapsed_seconds + time.monotonic() - started
            )
            if combined_elapsed_seconds > MAX_GPU_HOURS_PER_CELL * 3600:
                raise ResourceStop("RESOURCE_STOP_PER_CELL_200_GPU_HOURS")
            if (
                all_attempts_gpu_hours_before
                + (time.monotonic() - started) / 3600.0
                > MAX_GPU_HOURS_ALL_ATTEMPTS
            ):
                raise ResourceStop("RESOURCE_STOP_ALL_ATTEMPTS_1150_GPU_HOURS")

        torch.cuda.synchronize(device)
        elapsed_seconds = time.monotonic() - started
        expected_transformer_forward_count = (
            target_step - start_step if arm == "B" else 0
        )
        if transformer_forward_count != expected_transformer_forward_count:
            raise RuntimeError(
                "observed transformer forward count differs from training steps"
            )
        frozen_after = (
            canonical_tensor_digest(frozen_transformer_tensors(transformer))
            if arm == "B"
            else None
        )
        if arm == "B" and frozen_after["sha256"] != frozen_before["sha256"]:
            raise RuntimeError("frozen base transformer changed during B training")
        final_trainable = trainable_tensor_digest(transformer, head, arm)
        producer_context = {
            "process_id": os.getpid(),
            "mode": mode,
            "journal_path": str(journal_path),
            "journal_sha256": sha256_file(journal_path),
            "head_initialization_sha256": head_initialization_digest["sha256"],
            "fresh_trainable_initialization_sha256": (
                fresh_trainable_initialization_digest["sha256"]
            ),
            "frozen_transformer_before_sha256": (
                frozen_before["sha256"] if frozen_before is not None else None
            ),
            "frozen_transformer_after_sha256": (
                frozen_after["sha256"] if frozen_after is not None else None
            ),
            "sample_order_sha256": order_sha256,
        }
        journal_verification = verify_training_journal(
            journal_path,
            sample_ids=sample_ids,
            seed=seed,
            start_step=start_step,
            target_step=target_step,
        )
        checkpoint = save_checkpoint(
            output,
            step=target_step,
            arm=arm,
            seed=seed,
            transformer=transformer,
            head=head,
            optimizer=optimizer,
            scheduler=scheduler,
            binding=binding,
            producer_context=producer_context,
            sample_order_sha256_value=order_sha256,
        )
        result = {
            **base_evidence,
            "status": (
                "PASS_SMOKE"
                if mode == "smoke"
                else "PASS_PROCESS_A"
                if mode == "process_a"
                else "TRAINING_COMPLETE_PENDING_JOINT_SEAL"
            ),
            "steps_completed_this_process": target_step - start_step,
            "final_global_step": target_step,
            "elapsed_seconds": elapsed_seconds,
            "gpu_hours": elapsed_seconds / 3600.0,
            "combined_cell_elapsed_seconds": (
                producer_elapsed_seconds + elapsed_seconds
            ),
            "combined_cell_gpu_hours": (
                producer_elapsed_seconds + elapsed_seconds
            )
            / 3600.0,
            "loss_first": losses[0],
            "loss_last": losses[-1],
            "loss_min": min(losses),
            "loss_max": max(losses),
            "all_losses_finite": all(math.isfinite(value) for value in losses),
            "transformer_forward_count": transformer_forward_count,
            "expected_transformer_forward_count": (
                expected_transformer_forward_count
            ),
            "vae_decoder_call_count": 0,
            "head_initialization_digest": head_initialization_digest,
            "fresh_trainable_initialization_digest": (
                fresh_trainable_initialization_digest
            ),
            "final_trainable_digest": final_trainable,
            "frozen_transformer_after": frozen_after,
            "frozen_transformer_unchanged": (
                frozen_after["sha256"] == frozen_before["sha256"] if arm == "B" else None
            ),
            "all_frozen_requires_grad_false": (
                all(
                    not parameter.requires_grad
                    for name, parameter in transformer.named_parameters()
                    if "lora_" not in name.casefold()
                )
                if arm == "B"
                else True
            ),
            "all_frozen_grads_none": (
                all(
                    parameter.grad is None
                    for name, parameter in transformer.named_parameters()
                    if "lora_" not in name.casefold()
                )
                if arm == "B"
                else True
            ),
            "gpu_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "gpu_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
            "checkpoint": checkpoint,
            "journal_path": str(journal_path),
            "journal_sha256": sha256_file(journal_path),
            "journal_verification": journal_verification,
        }
        if forward_hook is not None:
            forward_hook.remove()
        atomic_json(evidence_path, result)
        return result
    except ResourceStop as exc:
        if forward_hook is not None:
            forward_hook.remove()
        elapsed_seconds = time.monotonic() - started
        atomic_json(
            evidence_path,
            {
                **base_evidence,
                "status": "RESOURCE_STOP",
                "scientific_validity": "NOT_EVALUATED",
                "requires_user_direction": True,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "elapsed_seconds": elapsed_seconds,
                "combined_cell_elapsed_seconds": (
                    producer_elapsed_seconds + elapsed_seconds
                ),
                "combined_cell_gpu_hours": (
                    producer_elapsed_seconds + elapsed_seconds
                )
                / 3600.0,
            },
        )
        raise
    except Exception as exc:
        if forward_hook is not None:
            forward_hook.remove()
        atomic_json(
            evidence_path,
            {
                **base_evidence,
                "status": "INVALID",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "elapsed_seconds": time.monotonic() - started,
            },
        )
        raise
