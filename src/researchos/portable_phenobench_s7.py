"""Invoke the pinned original PhenoBench S7 trainer with portable local assets."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from researchos.portable_original import bind_device, read_json
from researchos.portable_data import sha256_file

OFFICIAL_SHA256 = "6486a9d7b56264a047339c3e8291d0d55cae51266bb2c3c5d8d533872330e31e"
SEEDS = (20260723, 20260724, 20260725)


def canonical_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def load_manifest(path, *, project_root, validate_target_hashes=True, **kwargs):
    payload = read_json(Path(path))
    rows = payload.get("records", [])
    if payload.get("split") != "train" or payload.get("final_test") is not False or len(rows) != 1407:
        raise ValueError("S7 requires the complete 1,407-sample train-only manifest")
    if len({row["sample_id"] for row in rows}) != 1407:
        raise ValueError("S7 manifest contains duplicated sample IDs")
    unsigned = {key: value for key, value in payload.items() if key != "manifest_sha256"}
    if payload.get("manifest_sha256") != canonical_hash(unsigned):
        raise ValueError("S7 manifest digest differs")
    for row in rows:
        if row["split"] != "train" or row["instruction"] != payload["instruction"]:
            raise ValueError("S7 split/prompt binding differs")
        for key, digest in (("condition_image", "condition_sha256"), ("target_image", "target_sha256")):
            if sha256_file(Path(project_root) / row[key]) != row[digest]:
                raise ValueError(f"Changed S7 paired file: {row['sample_id']}")
    return payload, rows


def prepare(config: dict, workspace: Path, recipe: dict) -> dict:
    import numpy as np
    from PIL import Image
    from researchos.phenobench_labels import task_label_array, TASK_PALETTE
    data = Path(config["data_root"]).resolve(strict=True)
    rows = read_json(workspace / "train.json")["rows"]
    if len(rows) != 1407 or any(row["split"] != "train" for row in rows):
        raise ValueError("Run phenobench.py prepare for all 1,407 training inputs first")
    destination = workspace / "s7_inputs/targets"
    destination.mkdir(parents=True, exist_ok=True)
    records = []
    palette = np.asarray(TASK_PALETTE, dtype=np.uint8)
    for row in rows:
        image = (data / row["image_path"]).resolve(strict=True)
        label = (data / row["label_path"]).resolve(strict=True)
        if sha256_file(image) != row["image_sha256"] or sha256_file(label) != row["label_sha256"]:
            raise ValueError("PhenoBench training file changed after preparation")
        target = destination / row["sample_id"]
        with Image.open(label) as semantic:
            pixels = palette[task_label_array(semantic)]
        if target.exists():
            with Image.open(target) as existing:
                if not np.array_equal(np.asarray(existing.convert("RGB")), pixels):
                    raise ValueError("Existing S7 target differs")
        else:
            Image.fromarray(pixels).save(target)
        records.append({"sample_id": row["sample_id"], "split": "train",
            "condition_image": str(image), "target_image": target.relative_to(workspace).as_posix(),
            "condition_sha256": row["image_sha256"], "target_sha256": sha256_file(target),
            "instruction": recipe["prompt"]["text"]})
    payload = {"task_id": "S7_Ex01_phenobench_rgb_lora", "split": "train", "final_test": False,
        "pair_count": len(records), "instruction": recipe["prompt"]["text"],
        "source_native_train_manifest_sha256": sha256_file(workspace / "train.json"),
        "canonical_palette": [list(color) for color in TASK_PALETTE], "records": records}
    payload["manifest_sha256"] = canonical_hash(payload)
    path = workspace / "s7_inputs/train.json"
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return {"manifest": str(path), "pair_count": len(records), "manifest_sha256": payload["manifest_sha256"]}


def train(config: dict, workspace: Path, recipe: dict, official_path: Path, seed: int, process: str) -> dict:
    from researchos import phenobench_s7_bridge as bridge
    from researchos.phenobench_adapter_identity import training_identity, create_reproduction_binding
    bridge.ROOT = workspace
    bridge.MODEL_ROOT = Path(config["flux_model"]).resolve(strict=True)
    bridge.OFFICIAL_SOURCE = official_path.resolve(strict=True)
    bridge.PROMPT = recipe["prompt"]["text"]
    bridge.load_training_manifest = load_manifest
    manifest = workspace / "s7_inputs/train.json"
    payload, paired_records = load_manifest(manifest, project_root=workspace)
    native_train_sha = sha256_file(workspace / "train.json")
    native_rows = read_json(workspace / "train.json")["rows"]
    if (payload.get("source_native_train_manifest_sha256") != native_train_sha
            or {row["sample_id"]: row["condition_sha256"] for row in paired_records}
            != {row["sample_id"]: row["image_sha256"] for row in native_rows}):
        raise ValueError("Prepared S7 pairs do not originate from the current native training inputs")
    run_identity = training_identity(bridge.MODEL_ROOT, recipe)
    output = workspace / f"s7/seed_{seed}"
    start, stop = (0, 7035) if process == "process_a" else (7035, 14070)
    metadata_path = output / f"run_{process}.json"
    config_digest = canonical_hash({"recipe": recipe, "model_revision": bridge.REVISION})
    if process == "process_a" and output.exists() and any(output.iterdir()):
        raise FileExistsError("A new S7 training run requires an empty output directory")
    if metadata_path.exists():
        raise FileExistsError(metadata_path)
    if process == "process_b":
        prior = read_json(output / "run_process_a.json")
        checkpoint = output / "checkpoint-7035"
        required = {"optimizer.bin", "pytorch_lora_weights.safetensors", "random_states_0.pkl", "scheduler.bin"}
        if (prior.get("process_id") == os.getpid() or prior.get("final_global_step") != 7035
                or prior.get("manifest_sha256") != payload["manifest_sha256"]
                or prior.get("status") != "COMPLETE" or prior.get("seed") != seed
                or prior.get("training_identity") != run_identity
                or prior.get("native_train_manifest_sha256") != native_train_sha
                or prior.get("config_sha256") != config_digest
                or prior.get("checkpoint_files") != bridge._checkpoint_inventory(checkpoint)
                or not all((checkpoint / name).is_file() for name in required)):
            raise ValueError("S7 midpoint checkpoint/run identity does not match")
    output.mkdir(parents=True, exist_ok=True)
    official = bridge._load_official_module()
    official.save_model_card = lambda *args, **kwargs: None
    runtime_config = {"training": {**recipe["training"], "seed": seed},
        "runtime": {"instance_prompt": bridge.PROMPT}, "dataset_manifest": "s7_inputs/train.json"}
    arguments = bridge.build_official_arguments(config=runtime_config, output_dir=output,
        max_steps=stop, resume_from_checkpoint="checkpoint-7035" if process == "process_b" else None,
        checkpointing_steps=1407)
    arguments[arguments.index("--checkpoints_total_limit") + 1] = "12"
    parsed = official.parse_args(arguments)
    parsed.report_to = None
    official.args = parsed
    official.DreamBoothDataset = bridge.build_scientific_dataset_class(project_root=workspace,
        manifest_path=manifest, seed=seed, random_flip=False, condition_interpolation="bilinear", resume_step=start)
    sampler = bridge.build_deterministic_batch_sampler_class(resume_step=start, target_step=stop)
    official.BucketBatchSampler = sampler
    metrics = output / f"metrics_{process}.jsonl"
    bridge._install_metrics_accelerator(official, metrics, sampler)
    official.main(parsed)
    adapter = output / "pytorch_lora_weights.safetensors"
    checkpoint = output / f"checkpoint-{stop}"
    if not adapter.is_file() or not checkpoint.is_dir():
        raise RuntimeError("Pinned trainer did not emit the required adapter/checkpoint")
    summary = bridge._metric_summary(metrics, start_step=start, end_step=stop)
    batches = sum(int(row["yield_count"]) for row in sampler.iteration_records if row["phase"] == "training")
    if batches != stop - start:
        raise RuntimeError("S7 training batch count differs from the registered segment")
    if training_identity(bridge.MODEL_ROOT, recipe) != run_identity:
        raise RuntimeError("S7 model or implementation files changed during training")
    result = {"status": "COMPLETE", "seed": seed, "process": process, "process_id": os.getpid(),
        "start_global_step": start, "final_global_step": stop,
        "optimizer_steps_this_process": stop - start, "training_batches": batches, "pair_count": 1407,
        "config_sha256": config_digest, "manifest_sha256": payload["manifest_sha256"],
        "native_train_manifest_sha256": native_train_sha, "training_identity": run_identity,
        "official_trainer_sha256": OFFICIAL_SHA256, "metrics": summary, "metrics_sha256": sha256_file(metrics),
        "producer_run_sha256": sha256_file(output / "run_process_a.json") if process == "process_b" else None,
        "checkpoint_files": bridge._checkpoint_inventory(checkpoint),
        "adapter": {"path": str(adapter), "sha256": sha256_file(adapter)}}
    metadata_path.write_text(json.dumps(result, indent=2) + "\n")
    if process == "process_b":
        binding = create_reproduction_binding(output, seed=seed, identity=run_identity,
            native_train_sha=native_train_sha, s7_manifest_sha=payload["manifest_sha256"])
        (output / "adapter_binding.json").write_text(json.dumps(binding, indent=2) + "\n")
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("check", "prepare", "train"))
    parser.add_argument("--config", type=Path, default=Path("configs/original/phenobench.example.json"))
    parser.add_argument("--recipe", type=Path, default=Path("configs/original/phenobench_s7_prerequisite_recipe.json"))
    parser.add_argument("--workspace", type=Path, default=Path("runs/phenobench"))
    parser.add_argument("--official-trainer", type=Path,
        default=Path("third_party/diffusers_examples/train_dreambooth_lora_flux2_klein_img2img.py"))
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--process", choices=("process_a", "process_b"))
    parser.add_argument("--device", default="0")
    args = parser.parse_args(argv)
    config, recipe = read_json(args.config), read_json(args.recipe)
    from researchos.phenobench_adapter_identity import fixed_recipe_identity
    fixed_recipe_identity(recipe)
    workspace = args.workspace.resolve()
    if args.action == "check":
        present = args.official_trainer.is_file()
        result = {"official_trainer_present": present, "expected_sha256": OFFICIAL_SHA256,
            "official_trainer_matches": present and sha256_file(args.official_trainer) == OFFICIAL_SHA256,
            "model_present": Path(config["flux_model"]).is_dir(),
            "train_manifest_present": (workspace / "s7_inputs/train.json").is_file(), "models_loaded": False}
        result["status"] = "PASS" if all(result[name] for name in ("official_trainer_matches", "model_present", "train_manifest_present")) else "MISSING_INPUTS"
    elif args.action == "prepare":
        result = prepare(config, workspace, recipe)
    else:
        if args.seed is None or args.process is None:
            parser.error("Training requires --seed and --process")
        if not args.official_trainer.is_file() or sha256_file(args.official_trainer) != OFFICIAL_SHA256:
            parser.error("The exact pinned official trainer file is required; its SHA256 must match")
        bind_device(args.device)
        result = train(config, workspace, recipe, args.official_trainer, args.seed, args.process)
    print(json.dumps(result, indent=2))
    return 2 if result.get("status") == "MISSING_INPUTS" else 0
