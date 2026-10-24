"""SegFormer-B2 architecture and strict OpenMMLab checkpoint mapping."""
from __future__ import annotations
import hashlib
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any
_SEGFORMER_CLASS = None

OPENMMLAB_SEGFORMER_B2_SHA256 = "cbd414ac055c8436207914ac3a8e08c69879e8ae9e70f1a57cf0f7e17f19025a"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _segformer_class() -> Any:
    """Create the MiT-B2 plus OpenMMLab SegFormerHead module lazily."""

    global _SEGFORMER_CLASS
    if _SEGFORMER_CLASS is not None:
        return _SEGFORMER_CLASS
    import torch
    from torch import nn
    from torch.nn import functional as functional
    from transformers import SegformerConfig, SegformerModel

    class ConvBNReLU(nn.Module):
        def __init__(self, in_channels: int, out_channels: int) -> None:
            super().__init__()
            self.conv = nn.Conv2d(in_channels, out_channels, 1, bias=False)
            self.bn = nn.BatchNorm2d(out_channels)
            self.activate = nn.ReLU(inplace=True)

        def forward(self, value: Any) -> Any:
            return self.activate(self.bn(self.conv(value)))

    class OpenMMLabSegFormerB2(nn.Module):
        def __init__(self, num_classes: int = 3) -> None:
            super().__init__()
            encoder_config = SegformerConfig(
                num_channels=3,
                num_encoder_blocks=4,
                depths=[3, 4, 6, 3],
                sr_ratios=[8, 4, 2, 1],
                hidden_sizes=[64, 128, 320, 512],
                patch_sizes=[7, 3, 3, 3],
                strides=[4, 2, 2, 2],
                num_attention_heads=[1, 2, 5, 8],
                mlp_ratios=[4, 4, 4, 4],
                hidden_act="gelu",
                hidden_dropout_prob=0.0,
                attention_probs_dropout_prob=0.0,
                drop_path_rate=0.1,
                reshape_last_stage=True,
                output_hidden_states=True,
            )
            self.encoder = SegformerModel(encoder_config)
            self.decode_head = nn.Module()
            self.decode_head.convs = nn.ModuleList(
                [ConvBNReLU(channels, 256) for channels in (64, 128, 320, 512)]
            )
            self.decode_head.fusion_conv = ConvBNReLU(1024, 256)
            self.decode_head.dropout = nn.Dropout2d(0.1)
            self.classifier = nn.Conv2d(256, num_classes, 1)
            nn.init.normal_(self.classifier.weight, mean=0.0, std=0.01)
            nn.init.zeros_(self.classifier.bias)

        def forward(self, pixel_values: Any) -> Any:
            output = self.encoder(
                pixel_values=pixel_values,
                output_hidden_states=True,
                return_dict=True,
            )
            features = output.hidden_states
            if features is None or len(features) != 4:
                raise RuntimeError("MiT-B2 encoder did not return four feature stages")
            output_size = features[0].shape[-2:]
            projected = []
            for feature, conv in zip(features, self.decode_head.convs, strict=True):
                value = conv(feature)
                if value.shape[-2:] != output_size:
                    value = functional.interpolate(
                        value,
                        size=output_size,
                        mode="bilinear",
                        align_corners=False,
                    )
                projected.append(value)
            fused = self.decode_head.fusion_conv(torch.cat(projected, dim=1))
            return self.classifier(self.decode_head.dropout(fused))

    _SEGFORMER_CLASS = OpenMMLabSegFormerB2
    return _SEGFORMER_CLASS


def build_openmmlab_segformer_b2(*, num_classes: int = 3) -> Any:
    """Build the frozen architecture without fetching or loading an asset."""

    return _segformer_class()(num_classes=num_classes)


def _translate_openmmlab_encoder_key(source_key: str, value: Any) -> list[tuple[str, Any]]:
    patch = re.fullmatch(
        r"backbone\.layers\.(\d+)\.0\.(projection|norm)\.(weight|bias)",
        source_key,
    )
    if patch:
        stage, component, parameter = patch.groups()
        target_component = "proj" if component == "projection" else "layer_norm"
        return [
            (
                f"encoder.stages.{stage}.patch_embeddings.{target_component}.{parameter}",
                value,
            )
        ]
    final_norm = re.fullmatch(r"backbone\.layers\.(\d+)\.2\.(weight|bias)", source_key)
    if final_norm:
        stage, parameter = final_norm.groups()
        return [(f"encoder.stages.{stage}.layer_norm.{parameter}", value)]
    block = re.fullmatch(r"backbone\.layers\.(\d+)\.1\.(\d+)\.(.+)", source_key)
    if not block:
        return []
    stage, block_index, suffix = block.groups()
    prefix = f"encoder.stages.{stage}.blocks.{block_index}"
    direct_suffixes = {
        "norm1.weight": "layernorm_before.weight",
        "norm1.bias": "layernorm_before.bias",
        "attn.attn.out_proj.weight": "attention.o_proj.weight",
        "attn.attn.out_proj.bias": "attention.o_proj.bias",
        "attn.sr.weight": "attention.sequence_reduction.sequence_reduction.weight",
        "attn.sr.bias": "attention.sequence_reduction.sequence_reduction.bias",
        "attn.norm.weight": "attention.sequence_reduction.layer_norm.weight",
        "attn.norm.bias": "attention.sequence_reduction.layer_norm.bias",
        "norm2.weight": "layernorm_after.weight",
        "norm2.bias": "layernorm_after.bias",
        "ffn.layers.1.weight": "mlp.dwconv.dwconv.weight",
        "ffn.layers.1.bias": "mlp.dwconv.dwconv.bias",
        "ffn.layers.0.bias": "mlp.fc1.bias",
        "ffn.layers.4.bias": "mlp.fc2.bias",
    }
    if suffix in direct_suffixes:
        return [(f"{prefix}.{direct_suffixes[suffix]}", value)]
    if suffix in {"ffn.layers.0.weight", "ffn.layers.4.weight"}:
        if value.ndim != 4 or tuple(value.shape[-2:]) != (1, 1):
            raise ValueError(f"OpenMMLab FFN pointwise weight has wrong shape: {source_key}")
        target = "mlp.fc1.weight" if ".0." in suffix else "mlp.fc2.weight"
        return [(f"{prefix}.{target}", value[:, :, 0, 0])]
    if suffix in {"attn.attn.in_proj_weight", "attn.attn.in_proj_bias"}:
        if int(value.shape[0]) % 3:
            raise ValueError(f"OpenMMLab fused QKV has indivisible shape: {source_key}")
        query, key, projection_value = value.chunk(3, dim=0)
        parameter = "weight" if suffix.endswith("weight") else "bias"
        return [
            (f"{prefix}.attention.q_proj.{parameter}", query),
            (f"{prefix}.attention.k_proj.{parameter}", key),
            (f"{prefix}.attention.v_proj.{parameter}", projection_value),
        ]
    return []


def map_openmmlab_segformer_b2_state(
    source_state: Mapping[str, Any], target_state: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Strictly map all non-classifier OpenMMLab weights into the frozen module.

    Exactly two source tensors (the ADE150 classifier) and exactly two target
    tensors (the new three-class classifier) are whitelisted.  Everything else
    must be consumed one-to-one, including SyncBN running statistics.
    """

    source_classifier = {
        "decode_head.conv_seg.weight",
        "decode_head.conv_seg.bias",
    }
    target_classifier = {"classifier.weight", "classifier.bias"}
    mapped: dict[str, Any] = {}
    consumed: set[str] = set()
    for source_key, value in source_state.items():
        if source_key in source_classifier:
            consumed.add(source_key)
            continue
        translations: list[tuple[str, Any]]
        if source_key.startswith("backbone."):
            translations = _translate_openmmlab_encoder_key(source_key, value)
        elif source_key.startswith("decode_head.convs.") or source_key.startswith(
            "decode_head.fusion_conv."
        ):
            translations = [(source_key, value)]
        else:
            translations = []
        if not translations:
            raise ValueError(f"unmapped OpenMMLab SegFormer key: {source_key}")
        for target_key, target_value in translations:
            if target_key in mapped:
                raise ValueError(f"duplicate mapped SegFormer target key: {target_key}")
            if target_key not in target_state:
                raise ValueError(f"mapped SegFormer target key is absent: {target_key}")
            if tuple(target_value.shape) != tuple(target_state[target_key].shape):
                raise ValueError(
                    f"SegFormer mapped tensor shape mismatch: {source_key} -> {target_key}"
                )
            mapped[target_key] = target_value
        consumed.add(source_key)
    expected_source = set(source_state)
    if consumed != expected_source:
        raise ValueError(
            f"OpenMMLab SegFormer source coverage mismatch: {sorted(expected_source - consumed)}"
        )
    expected_target = set(target_state) - target_classifier
    if set(mapped) != expected_target:
        missing = sorted(expected_target - set(mapped))
        extra = sorted(set(mapped) - expected_target)
        raise ValueError(
            f"OpenMMLab SegFormer target coverage mismatch; missing={missing}, extra={extra}"
        )
    report = {
        "source_key_count": len(source_state),
        "source_loaded_key_count": len(source_state) - len(source_classifier),
        "source_classifier_reset_keys": sorted(source_classifier),
        "target_key_count": len(target_state),
        "target_loaded_key_count": len(mapped),
        "target_random_classifier_keys": sorted(target_classifier),
        "missing_keys_whitelist": sorted(target_classifier),
        "unexpected_keys_whitelist": [],
        "strict_nonclassifier_coverage": True,
    }
    return mapped, report


def load_openmmlab_segformer_b2_checkpoint(
    model: Any,
    checkpoint_path: str | Path,
    *,
    expected_sha256: str = OPENMMLAB_SEGFORMER_B2_SHA256,
) -> dict[str, Any]:
    """Load the verified MMseg 0.15 checkpoint with a closed mapping whitelist."""

    import torch

    path = Path(checkpoint_path)
    observed_sha = sha256_file(path)
    if observed_sha != expected_sha256:
        raise ValueError("OpenMMLab SegFormer-B2 checkpoint SHA256 mismatch")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping) or not isinstance(payload.get("state_dict"), Mapping):
        raise ValueError("OpenMMLab checkpoint has no state_dict")
    source_state = _normalize_state_dict(payload["state_dict"])
    mapped, report = map_openmmlab_segformer_b2_state(source_state, model.state_dict())
    incompatible = model.load_state_dict(mapped, strict=False)
    if set(incompatible.missing_keys) != {"classifier.weight", "classifier.bias"}:
        raise ValueError(
            f"SegFormer load missing-key whitelist mismatch: {incompatible.missing_keys}"
        )
    if incompatible.unexpected_keys:
        raise ValueError(f"SegFormer load has unexpected keys: {incompatible.unexpected_keys}")
    return {
        **report,
        "checkpoint_sha256": observed_sha,
        "checkpoint_size_bytes": path.stat().st_size,
    }


def _normalize_state_dict(payload: Any) -> dict[str, Any]:
    if isinstance(payload, Mapping) and isinstance(payload.get("state_dict"), Mapping):
        payload = payload["state_dict"]
    if not isinstance(payload, Mapping):
        raise ValueError("model weight payload is not a state dict")
    result: dict[str, Any] = {}
    for raw_key, value in payload.items():
        key = str(raw_key)
        if key.startswith("module."):
            key = key.removeprefix("module.")
        result[key] = value
    return result
