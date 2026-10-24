"""Strict S7-to-native LoRA initialization helpers.

This module deliberately contains no training, inference, or dataset access.
It validates an S7 safetensors adapter against the already-instantiated Fresh
JNDA LoRA structure, then overwrites only those LoRA tensors.  Every contract
check is fail-closed and failures carry a machine-readable audit payload.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

_LORA_A_PATTERN = re.compile(r"^(?P<module>.+)\.lora_A(?:\.[^.]+)?\.weight$")
_LORA_B_PATTERN = re.compile(r"^(?P<module>.+)\.lora_B(?:\.[^.]+)?\.weight$")
_HEX_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class S7InitLoraContractError(RuntimeError):
    """A fail-closed S7-init LoRA contract violation.

    ``audit`` is JSON-serializable and is suitable for writing into the
    per-seed step-0 parameter manifest by the caller.
    """

    def __init__(self, message: str, audit: Mapping[str, Any]) -> None:
        super().__init__(message)
        self.audit = dict(audit)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_dtype(value: Any) -> str:
    return str(getattr(value, "dtype", type(value).__name__))


def _json_shape(value: Any) -> list[int]:
    shape = getattr(value, "shape", None)
    if shape is None:
        return []
    return [int(dimension) for dimension in shape]


def _fail(
    message: str,
    audit: dict[str, Any],
    *,
    check: str,
) -> None:
    audit["status"] = "FAIL"
    audit["failed_check"] = check
    audit["error"] = message
    raise S7InitLoraContractError(message, audit)


def normalize_transformer_safetensors_state(
    raw_state: Mapping[str, Any],
    *,
    audit: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Strip exactly one required ``transformer.`` prefix from every key.

    Mixed prefixes, absent prefixes, duplicate normalized keys, empty suffixes,
    and double prefixes are rejected.  This avoids permissive key rewriting
    that could silently bind an S7 tensor to the wrong Fresh LoRA parameter.
    """

    working_audit = audit if audit is not None else {}
    raw_keys = sorted(str(key) for key in raw_state)
    working_audit["raw_keys"] = raw_keys
    working_audit["raw_key_count"] = len(raw_keys)
    if not raw_state:
        _fail(
            "S7 safetensors state is empty",
            working_audit,
            check="transformer_prefix",
        )

    non_string_keys = [repr(key) for key in raw_state if not isinstance(key, str)]
    if non_string_keys:
        working_audit["non_string_keys"] = non_string_keys
        _fail(
            "S7 safetensors contains non-string keys",
            working_audit,
            check="transformer_prefix",
        )

    missing_prefix = sorted(key for key in raw_state if not key.startswith("transformer."))
    if missing_prefix:
        working_audit["keys_missing_transformer_prefix"] = missing_prefix
        _fail(
            "every S7 safetensors key must have exactly one transformer. prefix",
            working_audit,
            check="transformer_prefix",
        )

    normalized: dict[str, Any] = {}
    double_prefix: list[str] = []
    empty_suffix: list[str] = []
    collisions: list[str] = []
    for raw_key, value in raw_state.items():
        key = raw_key.removeprefix("transformer.")
        if not key:
            empty_suffix.append(raw_key)
            continue
        if key.startswith("transformer."):
            double_prefix.append(raw_key)
            continue
        if key in normalized:
            collisions.append(key)
            continue
        normalized[key] = value

    working_audit["double_prefixed_keys"] = sorted(double_prefix)
    working_audit["empty_normalized_keys"] = sorted(empty_suffix)
    working_audit["normalization_collisions"] = sorted(collisions)
    if double_prefix or empty_suffix or collisions:
        _fail(
            "S7 safetensors transformer-prefix normalization is ambiguous",
            working_audit,
            check="transformer_prefix",
        )

    working_audit["normalized_keys"] = sorted(normalized)
    working_audit["normalized_key_count"] = len(normalized)
    working_audit["transformer_prefix_status"] = "PASS"
    return normalized


def _single_peft_config(transformer: Any, audit: dict[str, Any]) -> tuple[str, Any]:
    configs = getattr(transformer, "peft_config", None)
    if not isinstance(configs, Mapping) or len(configs) != 1:
        audit["peft_config_names"] = (
            sorted(str(key) for key in configs) if isinstance(configs, Mapping) else []
        )
        _fail(
            "Fresh LoRA transformer must expose exactly one PEFT adapter config",
            audit,
            check="lora_config",
        )
    adapter_name, config = next(iter(configs.items()))
    audit["peft_adapter_name"] = str(adapter_name)
    return str(adapter_name), config


def _target_list(value: Any, audit: dict[str, Any]) -> list[str]:
    if isinstance(value, str) or not isinstance(value, Sequence | set | frozenset):
        _fail(
            "PEFT target_modules must be an explicit finite collection",
            audit,
            check="targets",
        )
    targets = [str(item) for item in value]
    if len(targets) != len(set(targets)):
        _fail(
            "PEFT target_modules contains duplicates",
            audit,
            check="targets",
        )
    return sorted(targets)


def _scaling_values(
    transformer: Any,
    *,
    adapter_name: str,
    audit: dict[str, Any],
) -> list[dict[str, Any]]:
    named_modules = getattr(transformer, "named_modules", None)
    if not callable(named_modules):
        _fail(
            "Fresh LoRA transformer does not expose named_modules",
            audit,
            check="scaling",
        )

    records: list[dict[str, Any]] = []
    for module_name, module in named_modules():
        if not hasattr(module, "lora_A") or not hasattr(module, "lora_B"):
            continue
        scaling = getattr(module, "scaling", None)
        if scaling is None:
            _fail(
                f"LoRA module {module_name!r} has no scaling value",
                audit,
                check="scaling",
            )
        if isinstance(scaling, Mapping) or (
            hasattr(scaling, "keys") and hasattr(scaling, "__getitem__")
        ):
            keys = [str(key) for key in scaling]
            if adapter_name not in keys:
                audit["scaling_adapter_keys"] = sorted(keys)
                _fail(
                    f"LoRA module {module_name!r} lacks scaling for adapter {adapter_name!r}",
                    audit,
                    check="scaling",
                )
            value = scaling[adapter_name]
        else:
            value = scaling
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            _fail(
                f"LoRA module {module_name!r} scaling is not numeric",
                audit,
                check="scaling",
            )
        if not math.isfinite(numeric):
            _fail(
                f"LoRA module {module_name!r} scaling is not finite",
                audit,
                check="scaling",
            )
        records.append({"module": str(module_name), "value": numeric})

    if not records:
        _fail(
            "Fresh LoRA transformer exposes no auditable LoRA modules",
            audit,
            check="scaling",
        )
    return sorted(records, key=lambda record: record["module"])


def _audit_lora_structure(
    *,
    transformer: Any,
    state: Mapping[str, Any],
    expected_targets: Sequence[str],
    expected_rank: int,
    expected_alpha: int | float,
    expected_scaling: float,
    audit: dict[str, Any],
) -> None:
    if expected_rank <= 0:
        _fail("expected_rank must be positive", audit, check="rank")
    adapter_name, config = _single_peft_config(transformer, audit)

    actual_rank = getattr(config, "r", None)
    audit["rank"] = {
        "expected": int(expected_rank),
        "config_actual": actual_rank,
        "tensor_checks": [],
    }
    if actual_rank != expected_rank:
        _fail(
            f"Fresh LoRA rank differs: expected {expected_rank}, got {actual_rank}",
            audit,
            check="rank",
        )

    actual_alpha = getattr(config, "lora_alpha", None)
    audit["alpha"] = {
        "expected": float(expected_alpha),
        "actual": actual_alpha,
    }
    try:
        alpha_matches = float(actual_alpha) == float(expected_alpha)
    except (TypeError, ValueError):
        alpha_matches = False
    if not alpha_matches:
        _fail(
            f"Fresh LoRA alpha differs: expected {expected_alpha}, got {actual_alpha}",
            audit,
            check="alpha",
        )

    expected_target_list = sorted(str(item) for item in expected_targets)
    if len(expected_target_list) != len(set(expected_target_list)):
        _fail(
            "expected_targets contains duplicates",
            audit,
            check="targets",
        )
    actual_targets = _target_list(getattr(config, "target_modules", None), audit)
    audit["targets"] = {
        "expected": expected_target_list,
        "actual": actual_targets,
        "missing": sorted(set(expected_target_list) - set(actual_targets)),
        "unexpected": sorted(set(actual_targets) - set(expected_target_list)),
    }
    if actual_targets != expected_target_list:
        _fail(
            "Fresh LoRA target module list differs",
            audit,
            check="targets",
        )

    tensor_modules: dict[str, set[str]] = {}
    unrecognized_keys: list[str] = []
    tensor_checks: list[dict[str, Any]] = []
    for key, tensor in sorted(state.items()):
        match_a = _LORA_A_PATTERN.match(key)
        match_b = _LORA_B_PATTERN.match(key)
        if match_a is not None:
            module = match_a.group("module")
            factor = "A"
            rank_dimension = _json_shape(tensor)[0] if len(_json_shape(tensor)) >= 2 else None
        elif match_b is not None:
            module = match_b.group("module")
            factor = "B"
            rank_dimension = _json_shape(tensor)[1] if len(_json_shape(tensor)) >= 2 else None
        else:
            unrecognized_keys.append(key)
            continue
        tensor_modules.setdefault(module, set()).add(factor)
        record = {
            "key": key,
            "factor": factor,
            "shape": _json_shape(tensor),
            "rank_dimension": rank_dimension,
            "matches": rank_dimension == expected_rank,
        }
        tensor_checks.append(record)

    audit["rank"]["tensor_checks"] = tensor_checks
    audit["rank"]["unrecognized_lora_keys"] = unrecognized_keys
    audit["rank"]["factor_pairs"] = {
        module: sorted(factors) for module, factors in sorted(tensor_modules.items())
    }
    bad_pairs = sorted(
        module for module, factors in tensor_modules.items() if factors != {"A", "B"}
    )
    bad_rank = [record["key"] for record in tensor_checks if not record["matches"]]
    if unrecognized_keys or not tensor_checks or bad_pairs or bad_rank:
        audit["rank"]["bad_factor_pairs"] = bad_pairs
        audit["rank"]["bad_rank_keys"] = bad_rank
        _fail(
            "LoRA tensor rank/factor structure differs",
            audit,
            check="rank",
        )

    scaling_records = _scaling_values(
        transformer,
        adapter_name=adapter_name,
        audit=audit,
    )
    audit["scaling"] = {
        "expected": float(expected_scaling),
        "module_values": scaling_records,
    }
    bad_scaling = [
        record
        for record in scaling_records
        if not math.isclose(
            record["value"],
            float(expected_scaling),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ]
    if bad_scaling:
        audit["scaling"]["mismatches"] = bad_scaling
        _fail(
            "Fresh LoRA effective scaling differs",
            audit,
            check="scaling",
        )
    audit["rank"]["status"] = "PASS"
    audit["alpha"]["status"] = "PASS"
    audit["targets"]["status"] = "PASS"
    audit["scaling"]["status"] = "PASS"


def load_s7_lora_strict(
    *,
    transformer: Any,
    adapter_path: Path,
    get_state: Callable[[Any], Mapping[str, Any]],
    set_state: Callable[[Any, Mapping[str, Any]], Any],
    expected_targets: Sequence[str],
    expected_rank: int = 8,
    expected_alpha: int | float = 8,
    expected_scaling: float = 1.0,
) -> dict[str, Any]:
    """Strictly overwrite a Fresh LoRA with an S7 safetensors adapter.

    The caller must instantiate the full Fresh JNDA path, including the native
    head, before invoking this function.  This function touches only tensors
    returned by ``get_state(transformer)`` and never loads optimizer state.
    """

    audit: dict[str, Any] = {
        "schema_version": "s7init_lora_load_audit_v1",
        "status": "IN_PROGRESS",
        "source": {"adapter_path": str(Path(adapter_path).resolve())},
        "missing_keys": [],
        "unexpected_keys": [],
        "shape_checks": [],
        "dtype_checks": [],
        "post_load_equality": {},
    }
    path = Path(adapter_path)
    if not path.is_file():
        _fail(
            f"S7 adapter file is absent: {path}",
            audit,
            check="source_file",
        )
    source_hash = _sha256_file(path)
    audit["source"]["sha256"] = source_hash
    audit["source"]["size_bytes"] = path.stat().st_size

    try:
        from safetensors.torch import load_file
    except ImportError as error:
        _fail(
            f"safetensors is required for strict S7 loading: {error}",
            audit,
            check="source_file",
        )
    try:
        raw_state = load_file(str(path), device="cpu")
    except Exception as error:
        _fail(
            f"S7 adapter is not a readable safetensors file: {error}",
            audit,
            check="source_file",
        )

    normalized = normalize_transformer_safetensors_state(raw_state, audit=audit)
    fresh_state = dict(get_state(transformer))
    if not fresh_state:
        _fail(
            "Fresh _lora_state is empty",
            audit,
            check="fresh_state",
        )
    non_tensor_fresh = sorted(
        key for key, value in fresh_state.items() if not hasattr(value, "shape")
    )
    non_tensor_source = sorted(
        key for key, value in normalized.items() if not hasattr(value, "shape")
    )
    if non_tensor_fresh or non_tensor_source:
        audit["non_tensor_fresh_keys"] = non_tensor_fresh
        audit["non_tensor_source_keys"] = non_tensor_source
        _fail(
            "LoRA states must contain tensors only",
            audit,
            check="tensor_types",
        )

    expected_keys = set(fresh_state)
    supplied_keys = set(normalized)
    missing = sorted(expected_keys - supplied_keys)
    unexpected = sorted(supplied_keys - expected_keys)
    audit["expected_fresh_keys"] = sorted(expected_keys)
    audit["missing_keys"] = missing
    audit["unexpected_keys"] = unexpected
    if missing or unexpected:
        _fail(
            "S7 and Fresh LoRA key sets differ",
            audit,
            check="exact_keys",
        )

    shape_mismatches: list[dict[str, Any]] = []
    dtype_mismatches: list[dict[str, Any]] = []
    for key in sorted(expected_keys):
        expected_shape = _json_shape(fresh_state[key])
        source_shape = _json_shape(normalized[key])
        shape_record = {
            "key": key,
            "fresh": expected_shape,
            "source": source_shape,
            "matches": expected_shape == source_shape,
        }
        audit["shape_checks"].append(shape_record)
        if not shape_record["matches"]:
            shape_mismatches.append(shape_record)

        expected_dtype = _json_dtype(fresh_state[key])
        source_dtype = _json_dtype(normalized[key])
        dtype_record = {
            "key": key,
            "fresh": expected_dtype,
            "source": source_dtype,
            "matches": expected_dtype == source_dtype,
        }
        audit["dtype_checks"].append(dtype_record)
        if not dtype_record["matches"]:
            dtype_mismatches.append(dtype_record)

    audit["shape_mismatches"] = shape_mismatches
    audit["dtype_mismatches"] = dtype_mismatches
    if shape_mismatches:
        _fail(
            "S7 and Fresh LoRA tensor shapes differ",
            audit,
            check="exact_shapes",
        )
    if dtype_mismatches:
        _fail(
            "S7 and Fresh LoRA tensor dtypes differ",
            audit,
            check="exact_dtypes",
        )

    _audit_lora_structure(
        transformer=transformer,
        state=normalized,
        expected_targets=expected_targets,
        expected_rank=expected_rank,
        expected_alpha=expected_alpha,
        expected_scaling=expected_scaling,
        audit=audit,
    )

    setter_result = set_state(transformer, normalized)
    setter_unexpected = sorted(
        str(key) for key in (getattr(setter_result, "unexpected_keys", ()) or ())
    )
    audit["setter_unexpected_keys"] = setter_unexpected
    if setter_unexpected:
        _fail(
            "LoRA setter reported unexpected keys",
            audit,
            check="setter_result",
        )

    loaded_state = dict(get_state(transformer))
    loaded_missing = sorted(expected_keys - set(loaded_state))
    loaded_unexpected = sorted(set(loaded_state) - expected_keys)
    audit["post_load_missing_keys"] = loaded_missing
    audit["post_load_unexpected_keys"] = loaded_unexpected
    if loaded_missing or loaded_unexpected:
        _fail(
            "post-load Fresh LoRA key set differs",
            audit,
            check="post_load_equality",
        )

    try:
        import torch
    except ImportError as error:
        _fail(
            f"PyTorch is required for post-load equality: {error}",
            audit,
            check="post_load_equality",
        )
    equality: dict[str, bool] = {}
    for key in sorted(expected_keys):
        source_tensor = normalized[key].detach().cpu()
        loaded_tensor = loaded_state[key].detach().cpu()
        equality[key] = bool(torch.equal(source_tensor, loaded_tensor))
    audit["post_load_equality"] = equality
    unequal = sorted(key for key, matches in equality.items() if not matches)
    audit["post_load_unequal_keys"] = unequal
    audit["post_load_all_equal"] = not unequal
    if unequal:
        _fail(
            "post-load LoRA tensor equality failed",
            audit,
            check="post_load_equality",
        )

    audit["status"] = "PASS"
    audit["failed_check"] = None
    return audit


def validate_s7_checkpoint_provenance(
    *,
    requested_seed: int,
    receipt: Mapping[str, Any],
    adapter_path: Path | None = None,
    allowed_seeds: Sequence[int] = (1701, 1702, 1703, 1704, 1705),
    expected_system: str = "Fresh_S7",
    expected_epoch: int = 10,
) -> dict[str, Any]:
    """Validate exact same-seed S7 epoch-10 adapter provenance.

    If ``adapter_path`` is supplied, its SHA256 must match the receipt.  A seed
    mismatch is always fatal; no nearest-checkpoint or cross-seed fallback is
    represented by this interface.
    """

    audit: dict[str, Any] = {
        "schema_version": "s7init_checkpoint_provenance_audit_v1",
        "status": "IN_PROGRESS",
        "requested_seed": int(requested_seed),
        "allowed_seeds": [int(seed) for seed in allowed_seeds],
        "expected_system": expected_system,
        "expected_epoch": int(expected_epoch),
    }
    if requested_seed not in allowed_seeds:
        _fail(
            f"requested seed {requested_seed} is outside the frozen seed set",
            audit,
            check="requested_seed",
        )

    receipt_hash = receipt.get("adapter_sha256")
    checks = {
        "system": receipt.get("system") == expected_system,
        "same_seed": receipt.get("seed") == requested_seed,
        "completed_epochs": receipt.get("completed_epochs") == expected_epoch,
        "scientific_checkpoint": receipt.get("scientific_checkpoint") is True,
        "adapter_sha256": isinstance(receipt_hash, str)
        and _HEX_SHA256_PATTERN.fullmatch(receipt_hash) is not None,
    }
    audit["receipt"] = {
        "system": receipt.get("system"),
        "seed": receipt.get("seed"),
        "completed_epochs": receipt.get("completed_epochs"),
        "scientific_checkpoint": receipt.get("scientific_checkpoint"),
        "adapter_sha256": receipt_hash,
    }
    audit["checks"] = checks
    if not all(checks.values()):
        _fail(
            f"S7 checkpoint provenance differs: {checks}",
            audit,
            check="receipt_provenance",
        )

    if adapter_path is not None:
        path = Path(adapter_path)
        audit["adapter_path"] = str(path.resolve())
        if not path.is_file():
            _fail(
                f"S7 provenance adapter is absent: {path}",
                audit,
                check="adapter_hash",
            )
        actual_hash = _sha256_file(path)
        audit["actual_adapter_sha256"] = actual_hash
        if actual_hash != receipt_hash:
            _fail(
                "S7 adapter SHA256 differs from its formal receipt",
                audit,
                check="adapter_hash",
            )
        audit["checks"]["adapter_file_hash"] = True

    audit["status"] = "PASS"
    audit["failed_check"] = None
    return audit


__all__ = [
    "S7InitLoraContractError",
    "load_s7_lora_strict",
    "normalize_transformer_safetensors_state",
    "validate_s7_checkpoint_provenance",
]
