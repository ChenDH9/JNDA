"""Portable PhenoBench native-cache, training, and validation prediction entrypoint."""
from __future__ import annotations

import argparse
import csv
import gc
import json
from pathlib import Path

from researchos.portable_data import sha256_file
from researchos.portable_original import bind_device, read_json

SEEDS = (20260723, 20260724, 20260725)


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def verify_cache_manifest(manifest: dict) -> None:
    from researchos.phenobench_adapter_identity import canonical_hash
    unsigned = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    if manifest.get("manifest_sha256") != canonical_hash(unsigned):
        raise ValueError("Native cache manifest identity differs")


def verify_native_pair_identity(left: dict, right: dict) -> None:
    fields = {"seed", "vae_cache_manifest_sha256", "native_train_manifest_sha256",
              "model_manifest_sha256", "prompt_cache_sha256"}
    if set(left) != fields or set(right) != fields or left != right or left["seed"] not in SEEDS:
        raise ValueError("Native Arm A/B seed, inputs, model, or prompt identities differ")
    for name in fields - {"seed"}:
        value = left[name]
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError("Native pair identity contains an invalid digest")


def verify_existing_native_pair(workspace: Path, arm: str, seed: int, identity: dict) -> str:
    from researchos import s8_native_training as native
    other = "B" if arm == "A" else "A"
    checkpoint = workspace / f"arm_{other}/seed_{seed}/checkpoint-14070"
    if not checkpoint.exists():
        return "OTHER_ARM_NOT_COMPLETE"
    state = read_json(checkpoint / "state.json")
    native.verify_checkpoint_receipt(checkpoint, expected_arm=other, expected_seed=seed,
        expected_step=14070, expected_binding=state["binding"],
        expected_sample_order_sha256=state["sample_order_sha256"])
    verify_native_pair_identity(identity, state["binding"].get("native_pair_identity", {}))
    return "PAIRED_INPUT_IDENTITY_VERIFIED"


def prepare(config: dict, workspace: Path) -> dict:
    data = Path(config["data_root"]).resolve(strict=True)
    train_images = data / "train/images"
    files = sorted(train_images.glob("*.png"))
    if len(files) != 1407 or len({p.name for p in files}) != 1407:
        raise ValueError("PhenoBench native training requires exactly 1,407 unique train PNGs")
    train = []
    for image in files:
        semantic = data / "train/semantics" / image.name
        if not semantic.is_file():
            raise FileNotFoundError(f"Missing train semantic mask: {image.name}")
        train.append({"sample_id": image.name, "split": "train",
            "image_path": image.relative_to(data).as_posix(), "image_sha256": sha256_file(image),
            "label_path": semantic.relative_to(data).as_posix(), "label_sha256": sha256_file(semantic)})
    with Path(config["validation_selection"]).open(encoding="utf-8-sig", newline="") as stream:
        selected = list(csv.DictReader(stream))
    selected.sort(key=lambda row: int(row["frozen_order_index"]))
    if len(selected) != 500 or len({r["sample_id"] for r in selected}) != 500:
        raise ValueError("Expected the frozen 500-image validation selection")
    if [int(r["frozen_order_index"]) for r in selected] != list(range(500)) or any(r["split"] != "val" for r in selected):
        raise ValueError("Validation split/order metadata differs")
    inputs = []
    for index, row in enumerate(selected):
        image = data / "val/images" / row["sample_id"]
        if not image.is_file():
            raise FileNotFoundError(f"Missing validation image: {row['sample_id']}")
        inputs.append({"index": index, "sample_id": row["sample_id"], "split": "val",
            "source_group": row["source_group"], "image_path": image.relative_to(data).as_posix(),
            "image_sha256": sha256_file(image)})
    write_json(workspace / "train.json", {"split": "train", "rows": train, "label_encoding": "phenobench_source_ids"})
    write_json(workspace / "validation.inputs.json", {"split": "val", "rows": inputs, "labels_present": False})
    return {"train_count": len(train), "validation_count": len(inputs), "semantic_pixels_decoded": False}


def _checked_path(data_root: Path, row: dict, kind: str) -> Path:
    path = (data_root / row[f"{kind}_path"]).resolve(strict=True)
    path.relative_to(data_root)
    if sha256_file(path) != row[f"{kind}_sha256"]:
        raise ValueError(f"Changed {kind} file: {row['sample_id']}")
    return path


def configure_runtime(config: dict, workspace: Path, arm: str, seed: int):
    from researchos import s8_runtime as runtime
    from researchos.phenobench_adapter_identity import model_identity, resolve_adapter
    runtime.ROOT = workspace
    runtime.MODEL_ROOT = Path(config["flux_model"]).resolve(strict=True)
    runtime.PHENOBENCH_MODEL_IDENTITY = model_identity(runtime.MODEL_ROOT)
    runtime.PHENOBENCH_S7_IDENTITY = None
    if arm == "A":
        binding = resolve_adapter(config["s7_adapters"][str(seed)], seed=seed,
            model_root=runtime.MODEL_ROOT, native_train_manifest_sha256=sha256_file(workspace / "train.json"))
        adapter = Path(binding["path"])
        alias = workspace / f"runs/S7_Ex01_phenobench_rgb_lora/fraction_100/seed_{seed}/attempt_001/pytorch_lora_weights.safetensors"
        alias.parent.mkdir(parents=True, exist_ok=True)
        if alias.exists() or alias.is_symlink():
            if alias.resolve() != adapter:
                raise ValueError("Workspace adapter alias already points to another artifact")
        else:
            alias.symlink_to(adapter)
        runtime.EXPECTED_ADAPTER_SHA256[seed] = binding["sha256"]
        runtime.PHENOBENCH_S7_IDENTITY = {key: value for key, value in binding.items() if key != "path"}
    return runtime


def build_cache(config: dict, workspace: Path) -> dict:
    import numpy as np
    import torch
    from PIL import Image
    from safetensors.torch import save_file
    from diffusers import Flux2KleinPipeline
    from researchos import s8_runtime as runtime
    from researchos.phenobench_labels import task_label_array
    from researchos.phenobench_adapter_identity import model_identity

    cache = workspace / "cache/vae"
    prompt_file = workspace / "cache/prompt.safetensors"
    if cache.exists() or prompt_file.exists():
        raise FileExistsError("Refusing to replace an existing PhenoBench cache")
    model = Path(config["flux_model"]).resolve(strict=True)
    base_model_identity = model_identity(model)
    data = Path(config["data_root"]).resolve(strict=True)
    rows = read_json(workspace / "train.json")["rows"]
    if len(rows) != 1407:
        raise ValueError("Expected 1,407 train samples")
    prompt_file.parent.mkdir(parents=True, exist_ok=True)
    pipeline = Flux2KleinPipeline.from_pretrained(str(model), vae=None, transformer=None,
        scheduler=None, torch_dtype=torch.bfloat16, local_files_only=True)
    pipeline.to("cuda:0")
    pipeline.text_encoder.eval()
    for parameter in pipeline.text_encoder.parameters():
        parameter.requires_grad_(False)
    with torch.inference_mode():
        embeds, ids = pipeline.encode_prompt(prompt=runtime.PROMPT, device=torch.device("cuda:0"),
            max_sequence_length=512, text_encoder_out_layers=(9, 18, 27))
    if tuple(embeds.shape) != (1, 512, 7680) or embeds.dtype != torch.bfloat16:
        raise ValueError("PhenoBench prompt embedding shape/dtype differs")
    if tuple(ids.shape) != (1, 512, 4) or ids.dtype != torch.int64:
        raise ValueError("PhenoBench text ID shape/dtype differs")
    save_file({"prompt_embeds": embeds.cpu().contiguous(), "txt_ids": ids.cpu().contiguous()}, str(prompt_file))
    pipeline.to("cpu")
    del pipeline, embeds, ids
    gc.collect()
    torch.cuda.empty_cache()
    pipeline = Flux2KleinPipeline.from_pretrained(str(model), transformer=None, text_encoder=None,
        tokenizer=None, scheduler=None, torch_dtype=torch.bfloat16, local_files_only=True)
    pipeline.to("cuda:0")
    pipeline.vae.eval()
    for parameter in pipeline.vae.parameters():
        parameter.requires_grad_(False)
    tokens, labels = [], []
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for row in rows:
            image_path = _checked_path(data, row, "image")
            label_path = _checked_path(data, row, "label")
            with Image.open(image_path) as image:
                if image.size != (1024, 1024):
                    raise ValueError("PhenoBench training RGB must be 1024x1024")
                _, tensor = runtime._resize_rgb_and_tensor(image)
            with Image.open(label_path) as image:
                if image.size != (1024, 1024):
                    raise ValueError("PhenoBench semantic masks must be 1024x1024")
                task = task_label_array(image)
                target_rgb = Image.fromarray(runtime.PALETTE[task])
                label = runtime._label_from_rgb(target_rgb)
            token = runtime.exact_vae_packed_tokens(pipeline, tensor, device="cuda:0")
            tokens.append(token.squeeze(0).cpu().contiguous())
            labels.append(label.contiguous())
    cache.mkdir(parents=True)
    save_file({"tokens": torch.stack(tokens), "labels": torch.stack(labels)}, str(cache / "tensors.safetensors"))
    from researchos.s8_native_training import canonical_json_sha256
    manifest = {"cache_kind": "vae_packed_tokens_bf16", "formal_cache": True,
        "arm": None, "seed": None, "s7_adapter_sha256": None,
        "sample_ids": [r["sample_id"] for r in rows], "train_manifest_sha256": sha256_file(workspace / "train.json"),
        "tensor_file_sha256": sha256_file(cache / "tensors.safetensors"), "model_identity": base_model_identity}
    if model_identity(model) != base_model_identity:
        raise ValueError("Model components changed during PhenoBench cache construction")
    manifest["manifest_sha256"] = canonical_json_sha256(manifest)
    write_json(cache / "manifest.json", manifest)
    return {"cache": str(cache), "samples": len(rows), "prompt_sha256": sha256_file(prompt_file)}


def build_arm_a_cache(config: dict, workspace: Path, seed: int) -> dict:
    import torch
    from safetensors.torch import save_file
    from researchos import s8_native_training as native
    runtime = configure_runtime(config, workspace, "A", seed)
    output = workspace / f"cache/arm_A_seed_{seed}"
    if output.exists():
        raise FileExistsError(output)
    source = native.load_cache_tensors(workspace / "cache/vae")
    verify_cache_manifest(source["manifest"])
    if (source["manifest"].get("model_identity") != runtime.PHENOBENCH_MODEL_IDENTITY
            or source["manifest"].get("train_manifest_sha256") != sha256_file(workspace / "train.json")):
        raise ValueError("Arm A and the shared VAE cache use different model or training-input identities")
    prompt_path = workspace / "cache/prompt.safetensors"
    prompt_sha = sha256_file(prompt_path)
    prompt = native.load_prompt_cache(prompt_path)
    native.seed_everything(seed)
    pipeline, transformer = runtime._load_transformer_pipeline(seed=seed, arm="A", device="cuda:0", include_vae=False)
    tokens = []
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for token in source["tokens"]:
            result = native.native_transformer_forward(transformer, token.unsqueeze(0).to("cuda:0"), **prompt)
            tokens.append(result.squeeze(0).cpu().contiguous())
    output.mkdir(parents=True)
    if sha256_file(prompt_path) != prompt_sha:
        raise ValueError("Prompt cache changed during Arm A native-cache construction")
    save_file({"tokens": torch.stack(tokens), "labels": source["labels"]}, str(output / "tensors.safetensors"))
    manifest = {"cache_kind": "native_tokens_bf16", "formal_cache": True, "arm": "A", "seed": seed,
        "s7_adapter_sha256": runtime.PHENOBENCH_S7_IDENTITY["sha256"],
        "s7_adapter_identity": runtime.PHENOBENCH_S7_IDENTITY, "sample_ids": source["sample_ids"],
        "prompt_cache_sha256": prompt_sha,
        "source_vae_manifest_sha256": source["manifest"]["manifest_sha256"],
        "tensor_file_sha256": sha256_file(output / "tensors.safetensors")}
    manifest["manifest_sha256"] = native.canonical_json_sha256(manifest)
    write_json(output / "manifest.json", manifest)
    return {"cache": str(output), "samples": len(tokens), "arm": "A", "seed": seed}


def train(config: dict, workspace: Path, arm: str, seed: int, process: str, gpu: dict) -> dict:
    from researchos import s8_native_training as native
    runtime = configure_runtime(config, workspace, arm, seed)
    native.seed_everything(seed)
    cache_path = workspace / (f"cache/arm_A_seed_{seed}" if arm == "A" else "cache/vae")
    cache = native.load_cache_tensors(cache_path)
    vae_manifest = read_json(workspace / "cache/vae/manifest.json")
    verify_cache_manifest(cache["manifest"])
    verify_cache_manifest(vae_manifest)
    if (vae_manifest.get("model_identity") != runtime.PHENOBENCH_MODEL_IDENTITY
            or vae_manifest.get("train_manifest_sha256") != sha256_file(workspace / "train.json")):
        raise ValueError("Native training inputs or base model differ from the shared VAE cache")
    if arm == "A" and (cache["manifest"].get("source_vae_manifest_sha256") != vae_manifest["manifest_sha256"]
            or cache["manifest"].get("s7_adapter_identity") != runtime.PHENOBENCH_S7_IDENTITY):
        raise ValueError("Arm A cache does not match the current paired inputs and S7 run identity")
    prompt_file = workspace / "cache/prompt.safetensors"
    if arm == "A" and cache["manifest"].get("prompt_cache_sha256") != sha256_file(prompt_file):
        raise ValueError("Arm A cached native fields use a different prompt-cache identity")
    prompt = native.load_prompt_cache(prompt_file)
    pair_identity = {"seed": seed, "vae_cache_manifest_sha256": vae_manifest["manifest_sha256"],
        "native_train_manifest_sha256": vae_manifest["train_manifest_sha256"],
        "model_manifest_sha256": runtime.PHENOBENCH_MODEL_IDENTITY["manifest_sha256"],
        "prompt_cache_sha256": sha256_file(prompt_file)}
    verify_existing_native_pair(workspace, arm, seed, pair_identity)
    transformer = None
    if arm == "B":
        pipeline, transformer = runtime._load_transformer_pipeline(seed=seed, arm="B", device="cuda:0",
            include_vae=False, gradient_checkpointing=True)
        transformer._s8_pipeline_owner = pipeline
    head = native.create_head(seed, device="cuda:0")
    binding = {"task_id": "S8_Ex01_native_field_readout", "arm": arm, "seed": seed, "attempt": 1,
        "train_count": len(cache["sample_ids"]), "config_sha256": native.canonical_json_sha256(config),
        "implementation_sha256": sha256_file(Path(native.__file__)),
        "runtime_sha256": sha256_file(Path(runtime.__file__)),
        "cache_manifest_sha256": cache["manifest"]["manifest_sha256"],
        "cache_tensor_sha256": cache["manifest"]["tensor_file_sha256"],
        "sample_order_sha256": native.sample_order_sha256(cache["sample_ids"], seed),
        "prompt_cache_sha256": sha256_file(prompt_file),
        "s7_adapter_sha256": cache["manifest"].get("s7_adapter_sha256"),
        "s7_adapter_identity": runtime.PHENOBENCH_S7_IDENTITY,
        "native_run_identity_kind": "public_native_reproduction",
        "native_pair_identity": pair_identity,
        "selected_gpu_uuid": gpu["visible_gpu_uuid"], "attention_backend": "torch_sdpa",
        "gradient_checkpointing": arm == "B", "vae_decoder_call_count": 0,
        "validation_access_count": 0, "final_test_access_count": 0, "development_smoke": False}
    return native.run_training_segment(arm=arm, seed=seed, transformer=transformer, head=head,
        cache=cache, prompt_cache=prompt, output_dir=workspace / f"arm_{arm}/seed_{seed}",
        mode=process, binding=binding, device="cuda:0")


def predict(config: dict, workspace: Path, arm: str, seed: int) -> dict:
    import torch
    from PIL import Image
    from diffusers import AutoencoderKLFlux2
    from researchos import s8_native_training as native
    runtime = configure_runtime(config, workspace, arm, seed)
    native.seed_everything(seed)
    pipeline, transformer = runtime._load_transformer_pipeline(seed=seed, arm=arm, device="cuda:0", include_vae=False)
    pipeline.vae = AutoencoderKLFlux2.from_pretrained(str(runtime.MODEL_ROOT), subfolder="vae",
        torch_dtype=torch.bfloat16, local_files_only=True).to("cuda:0").eval()
    head = native.create_head(seed, device="cuda:0")
    checkpoint = workspace / f"arm_{arm}/seed_{seed}/checkpoint-14070"
    state = read_json(checkpoint / "state.json")
    native.verify_checkpoint_receipt(checkpoint, expected_arm=arm, expected_seed=seed,
        expected_step=14070, expected_binding=state["binding"],
        expected_sample_order_sha256=state["sample_order_sha256"])
    if state["binding"]["prompt_cache_sha256"] != sha256_file(workspace / "cache/prompt.safetensors"):
        raise ValueError("Prediction prompt cache differs from the training binding")
    if state["binding"].get("native_pair_identity", {}).get("model_manifest_sha256") != runtime.PHENOBENCH_MODEL_IDENTITY["manifest_sha256"]:
        raise ValueError("Prediction base model differs from the native training pair identity")
    if arm == "A" and state["binding"].get("s7_adapter_identity") != runtime.PHENOBENCH_S7_IDENTITY:
        raise ValueError("Prediction S7 adapter differs from the training binding")
    verify_existing_native_pair(workspace, arm, seed, state["binding"]["native_pair_identity"])
    native.load_trainable_safetensors(checkpoint / "trainable.safetensors", transformer, head, arm)
    transformer.eval()
    head.eval()
    prompt = native.load_prompt_cache(workspace / "cache/prompt.safetensors")
    data = Path(config["data_root"]).resolve(strict=True)
    rows = read_json(workspace / "validation.inputs.json")["rows"]
    if len(rows) != 500:
        raise ValueError("Expected 500 frozen validation inputs")
    output = workspace / f"arm_{arm}/seed_{seed}/validation_predictions"
    output.mkdir(parents=True, exist_ok=True)
    inventory = []
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for row in rows:
            with Image.open(_checked_path(data, row, "image")) as image:
                _, tensor = runtime._resize_rgb_and_tensor(image)
            packed = runtime.exact_vae_packed_tokens(pipeline, tensor, device="cuda:0")
            field = native.native_transformer_forward(transformer, packed, **prompt)
            labels = native.head_logits(head, field).argmax(1).squeeze(0).byte().cpu().numpy()
            target = output / row["sample_id"]
            if target.exists():
                raise FileExistsError(target)
            Image.fromarray(labels).save(target)
            inventory.append({"sample_id": row["sample_id"], "prediction_path": target.relative_to(workspace).as_posix(),
                              "prediction_sha256": sha256_file(target)})
    result = {"arm": arm, "seed": seed, "prediction_count": len(inventory), "inventory": inventory,
        "native_run_identity_kind": "public_native_reproduction", "published_checkpoint": False,
        "native_pair_identity": state["binding"]["native_pair_identity"],
        "checkpoint_state_sha256": sha256_file(checkpoint / "state.json"),
        "checkpoint_trainable_sha256": sha256_file(checkpoint / "trainable.safetensors"),
        "s7_adapter_identity": runtime.PHENOBENCH_S7_IDENTITY}
    write_json(output / "inventory.json", result)
    return result


def evaluate(config: dict, workspace: Path, arm: str, seed: int | None) -> dict:
    import numpy as np
    from PIL import Image
    from researchos.phenobench_labels import task_label_array
    from researchos.metrics.segmentation import confusion_matrix, mean_iou, per_class_iou, pixel_accuracy
    workspace = workspace.resolve(strict=True)
    data = Path(config["data_root"]).resolve(strict=True)
    inputs = read_json(workspace / "validation.inputs.json")["rows"]
    if len(inputs) != 500 or len({row["sample_id"] for row in inputs}) != 500:
        raise ValueError("PhenoBench semantic evaluation requires exactly 500 frozen validation inputs")
    targets = {}
    for row in inputs:
        path = data / "val/semantics" / row["sample_id"]
        with Image.open(path) as image:
            target = task_label_array(image)
        targets[row["sample_id"]] = np.asarray(Image.fromarray(target).resize((512, 512), Image.Resampling.NEAREST))
    values = {}
    pairing = {}
    for current_seed in ((seed,) if seed is not None else SEEDS):
        directory = workspace / f"arm_{arm}/seed_{current_seed}/validation_predictions"
        inventory = read_json(directory / "inventory.json")
        other_arm = "B" if arm == "A" else "A"
        other_inventory = workspace / f"arm_{other_arm}/seed_{current_seed}/validation_predictions/inventory.json"
        pairing[str(current_seed)] = "OTHER_ARM_PREDICTIONS_NOT_PRESENT"
        if other_inventory.is_file():
            other = read_json(other_inventory)
            if other.get("arm") != other_arm or other.get("seed") != current_seed:
                raise ValueError("Other native arm prediction seed/identity differs")
            verify_native_pair_identity(inventory.get("native_pair_identity", {}), other.get("native_pair_identity", {}))
            pairing[str(current_seed)] = "PAIRED_INPUT_IDENTITY_VERIFIED"
        rows = inventory["inventory"]
        if inventory.get("arm") != arm or inventory.get("seed") != current_seed:
            raise ValueError("Prediction inventory arm/seed binding differs")
        if len(rows) != 500 or {r["sample_id"] for r in rows} != set(targets):
            raise ValueError("Prediction sample IDs differ from frozen validation inputs")
        total = np.zeros((3, 3), dtype=np.int64)
        for row in rows:
            path = (workspace / row["prediction_path"]).resolve(strict=True)
            path.relative_to(workspace)
            if sha256_file(path) != row["prediction_sha256"]:
                raise ValueError("Validation prediction digest changed")
            with Image.open(path) as image:
                prediction = np.asarray(image)
            if prediction.shape != (512, 512) or not set(np.unique(prediction)) <= {0, 1, 2}:
                raise ValueError("Validation prediction must be a 512x512 three-class map")
            total += confusion_matrix(prediction, targets[row["sample_id"]], num_classes=3)
        values[str(current_seed)] = {"confusion_matrix": total.tolist(), "miou": mean_iou(total),
            "class_iou": per_class_iou(total).tolist(), "pixel_accuracy": pixel_accuracy(total)}
    result = {"arm": arm, "sample_count": 500, "per_seed": values,
        "pairing_status": pairing,
        "mean_seed_miou": float(np.mean([value["miou"] for value in values.values()])),
        "aggregation": "pooled pixel confusion per seed, then arithmetic mean over requested seeds",
        "analysis_scope": "descriptive semantic evaluation; no historical gate or bootstrap inference"}
    write_json(workspace / f"arm_{arm}/validation_semantic_metrics.json", result)
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("check", "prepare", "cache", "cache-arm-a", "train", "predict", "evaluate"))
    parser.add_argument("--config", type=Path, default=Path("configs/original/phenobench.example.json"))
    parser.add_argument("--workspace", type=Path, default=Path("runs/phenobench"))
    parser.add_argument("--arm", choices=("A", "B"))
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--process", choices=("process_a", "process_b"))
    parser.add_argument("--device", default="0")
    args = parser.parse_args(argv)
    config = read_json(args.config)
    for key in ("data_root", "flux_model", "validation_selection", "s7_adapters"):
        if key not in config:
            parser.error(f"Configuration is missing {key}")
    workspace = args.workspace.resolve()
    if args.action == "check":
        missing = [name for name in ("data_root", "flux_model", "validation_selection") if not Path(config[name]).exists()]
        adapter_identities = {}
        adapter_errors = {}
        if args.arm == "A":
            seeds = (args.seed,) if args.seed else SEEDS
            for seed in seeds:
                item = config["s7_adapters"].get(str(seed), {})
                artifact_key = "manifest" if item.get("kind") == "public_s7_reproduction" else "path"
                if not Path(item.get(artifact_key, "missing")).is_file():
                    missing.append(f"s7_adapter_{seed}")
                    continue
                if item.get("kind") == "public_s7_reproduction" and not (workspace / "train.json").is_file():
                    missing.append("native_train_manifest")
                    continue
                if Path(config["flux_model"]).is_dir():
                    from researchos.phenobench_adapter_identity import resolve_adapter
                    try:
                        resolved = resolve_adapter(item, seed=seed, model_root=Path(config["flux_model"]),
                            native_train_manifest_sha256=sha256_file(workspace / "train.json")
                            if (workspace / "train.json").is_file() else "")
                        adapter_identities[str(seed)] = {"kind": resolved["kind"], "sha256": resolved["sha256"],
                            "published_checkpoint": resolved["published_checkpoint"]}
                    except (ValueError, FileNotFoundError, KeyError) as error:
                        adapter_errors[str(seed)] = str(error)
        result = {"status": "INVALID_ADAPTER_IDENTITY" if adapter_errors else "PASS" if not missing else "MISSING_INPUTS",
                  "missing": missing, "adapter_identities": adapter_identities, "adapter_errors": adapter_errors,
                  "train_count": 1407, "training_steps": 14070, "validation_count": 500, "models_loaded": False}
    elif args.action == "prepare":
        result = prepare(config, workspace)
    elif args.action == "evaluate":
        if args.arm is None:
            parser.error("Evaluation requires --arm")
        result = evaluate(config, workspace, args.arm, args.seed)
    else:
        if args.action in {"cache-arm-a", "train", "predict"} and args.seed is None:
            parser.error("This action requires --seed")
        if args.action in {"train", "predict"} and args.arm is None:
            parser.error("This action requires --arm")
        if args.action == "train" and args.process is None:
            parser.error("Training requires --process")
        gpu = bind_device(args.device)
        if args.action == "cache":
            result = build_cache(config, workspace)
        elif args.action == "cache-arm-a":
            result = build_arm_a_cache(config, workspace, args.seed)
        elif args.action == "train":
            result = train(config, workspace, args.arm, args.seed, args.process, gpu)
        else:
            result = predict(config, workspace, args.arm, args.seed)
    print(json.dumps(result, indent=2))
    return 2 if result.get("status") in {"MISSING_INPUTS", "INVALID_ADAPTER_IDENTITY"} else 0
