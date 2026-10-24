"""UPerNet with ConvNeXt-Tiny, pretrained initialization and optimizer."""
import hashlib
import math
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

REVISION = "876ffc56b819a829448f7e81e9f8606deef6fb65"

WEIGHT_SHA256 = "cd55b1d36c6602d34f8cd300d7f788ea0fa5c3ed928e9ac98f353ecb31c7fa1d"

CLASSIFIER_KEYS = frozenset({
    "decode_head.classifier.weight", "decode_head.classifier.bias",
    "auxiliary_head.classifier.weight", "auxiliary_head.classifier.bias",
})

CLASS_WEIGHTS = (0.25, 1.3431242219633113, 1.4751680850839344)

def tensor_hash(t):
    t = t.detach().cpu().contiguous()
    return hashlib.sha256(t.numpy().tobytes()).hexdigest()

def state_hash(state):
    h = hashlib.sha256()
    for k, t in sorted(state.items()):
        h.update(k.encode() + b"\0" + str(tuple(t.shape)).encode() + b"\0")
        h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def build_model(asset_dir, seed=2701):
    from safetensors.torch import load_file
    from transformers import UperNetConfig, UperNetForSemanticSegmentation

    p = Path(asset_dir)
    assert (p / "config.json").is_file(), "B1 config must arrive before loading"
    raw_hash = hashlib.sha256((p / "model.safetensors").read_bytes()).hexdigest()
    assert raw_hash == WEIGHT_SHA256, raw_hash
    torch.manual_seed(seed)
    config = UperNetConfig.from_pretrained(p, local_files_only=True)
    assert config.backbone_config.model_type == "convnext", config.backbone_config.model_type
    original_config = config.to_dict()
    config.num_labels = 3
    config.id2label = {0: "soil", 1: "crop", 2: "weed"}
    config.label2id = {v: k for k, v in config.id2label.items()}
    model = UperNetForSemanticSegmentation(config)
    source = load_file(str(p / "model.safetensors"), device="cpu")
    target = model.state_dict()
    unexpected = sorted(set(source) - set(target))
    missing_in_source = sorted(set(target) - set(source))
    mismatched = sorted(k for k in source if k in target and source[k].shape != target[k].shape)
    assert not unexpected and not missing_in_source, (unexpected, missing_in_source)
    assert set(mismatched) == CLASSIFIER_KEYS, mismatched
    compatible = {k: v for k, v in source.items() if k not in CLASSIFIER_KEYS}
    result = model.load_state_dict(compatible, strict=False)
    assert set(result.missing_keys) == CLASSIFIER_KEYS and not result.unexpected_keys
    after = model.state_dict()
    key_report = []
    for key in sorted(source):
        reset = key in CLASSIFIER_KEYS
        exact = None if reset else torch.equal(source[key], after[key])
        assert reset or exact, key
        key_report.append({"key": key, "pretrained_shape": list(source[key].shape),
                           "model_shape": list(after[key].shape),
                           "action": "RESET_150_TO_3_CLASSIFIER" if reset else "LOADED_EXACT",
                           "pretrained_sha256": tensor_hash(source[key]),
                           "actual_sha256": tensor_hash(after[key]), "exact_equal": exact})
    report = {"repo": "openmmlab/upernet-convnext-tiny", "revision": REVISION,
              "weight_sha256": raw_hash, "reset_keys": sorted(CLASSIFIER_KEYS),
              "missing_keys": list(result.missing_keys), "unexpected_keys": unexpected,
              "loaded_count": len(compatible), "key_count": len(source),
              "parameter_count": sum(p.numel() for p in model.parameters()),
              "trainable_parameter_count": sum(p.numel() for p in model.parameters() if p.requires_grad),
              "initial_model_state_sha256": state_hash(after),
              "config_before_three_class_transfer": original_config,
              "actual_config": config.to_dict(), "keys": key_report}
    return model, report

def stage_id(name):
    """Map HF names to official MMSeg stage_wise num_layers=6 (internal 8).

    Official downsample_layers all map 0, stages.i map i+1, other params 7.
    HF stores stage>0 downsample layers under encoder.stages.i.downsampling_layer.
    """
    if name.startswith("backbone.embeddings.") or ".downsampling_layer." in name:
        return 0
    if name.startswith("backbone.encoder.stages."):
        return int(name.split(".")[3]) + 1
    return 7

def build_optimizer(model, base_lr):
    groups = {}
    names_seen = []
    for name, parameter in model.named_parameters():
        assert parameter.requires_grad, name
        sid = stage_id(name)
        no_decay = parameter.ndim == 1 or name.endswith(".bias")
        key = (sid, no_decay)
        if key not in groups:
            scale = 0.9 ** (7 - sid)
            groups[key] = {"params": [], "param_names": [], "lr_scale": scale,
                           "lr": base_lr * scale, "initial_lr": base_lr * scale,
                           "weight_decay": 0.0 if no_decay else 0.05,
                           "group_name": f"stage_{sid}_{'no_decay' if no_decay else 'decay'}"}
        groups[key]["params"].append(parameter)
        groups[key]["param_names"].append(name)
        names_seen.append(name)
    assert len(names_seen) == len(set(names_seen))
    report = [{k: v for k, v in g.items() if k != "params"} for g in groups.values()]
    optimizer = torch.optim.AdamW(list(groups.values()), lr=base_lr,
                                 betas=(0.9, 0.999), eps=1e-8, weight_decay=0.05)
    return optimizer, report

def lr_factor(completed_updates, total_updates, warmup_updates=1500):
    """LR used by next update; update 1 starts at base LR * 1e-6."""
    assert 0 <= completed_updates <= total_updates and warmup_updates < total_updates
    if completed_updates < warmup_updates:
        return 1e-6 + (1.0 - 1e-6) * completed_updates / warmup_updates
    return max(0.0, (total_updates - completed_updates) /
               (total_updates - warmup_updates))

def set_lr(optimizer, completed_updates, total_updates):
    factor = lr_factor(completed_updates, total_updates)
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * factor
    return factor

def batch_plan(rows, microbatch=4, accumulation=2):
    """Full coverage: default tail 3+2; fallback tail microbatch size 3."""
    assert len(rows) == 4821
    assert (microbatch, accumulation) in ((4, 2), (2, 4))
    sizes = ([4] * 1204 + [3, 2] if microbatch == 4 else [2] * 2409 + [3])
    assert sum(sizes) == len(rows)
    microbatches, cursor = [], 0
    for size in sizes:
        microbatches.append(rows[cursor:cursor + size])
        cursor += size
    updates = [microbatches[i:i + accumulation]
               for i in range(0, len(microbatches), accumulation)]
    assert len(updates) == 603 and all(len(b) >= 2 for u in updates for b in u)
    assert [r for u in updates for b in u for r in b] == list(rows)
    return updates

def forward_logits(model, images):
    features = model.backbone.forward_with_filtered_kwargs(images).feature_maps
    main = model.decode_head(features)
    aux = model.auxiliary_head(features)
    size = images.shape[-2:]
    main = F.interpolate(main, size=size, mode="bilinear", align_corners=False)
    aux = F.interpolate(aux, size=size, mode="bilinear", align_corners=False)
    return main, aux

def weighted_denominator(target, weights):
    valid = target != 255
    return weights[target[valid]].sum()

def loss_numerator(main, aux, target, weights):
    main_loss = F.cross_entropy(main.float(), target, weight=weights,
                                ignore_index=255, reduction="sum")
    aux_loss = F.cross_entropy(aux.float(), target, weight=weights,
                               ignore_index=255, reduction="sum")
    return main_loss + 0.4 * aux_loss

def normalize_rgb(rgb):
    x = torch.from_numpy(np.asarray(rgb).copy()).permute(2, 0, 1).float() / 255.0
    return (x - torch.tensor([0.485, 0.456, 0.406])[:, None, None]) / torch.tensor(
        [0.229, 0.224, 0.225])[:, None, None]

def bn_state(model):
    return {k: t.detach().cpu().clone() for k, t in model.state_dict().items()
            if k.endswith(("running_mean", "running_var", "num_batches_tracked"))}
