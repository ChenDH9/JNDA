"""Portable file and device bindings for the original four-system implementation."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import shutil

SYSTEMS = ("Fresh_S7", "Fresh_Arm_A", "Fresh_Arm_B", "Fresh_SegFormer_B2")
SEEDS = (1701, 1702, 1703, 1704, 1705)
EXPECTED_PACKAGES = {"torch": "2.9.1", "torchvision": "0.24.1",
                     "transformers": "5.14.1", "peft": "0.19.1",
                     "numpy": "2.4.6", "Pillow": "12.3.0"}


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def load_config(path: Path) -> dict:
    value = read_json(path)
    required = {"data_root", "manifest_dir", "flux_model", "segformer_checkpoint"}
    if not required <= set(value.get("paths", {})):
        raise ValueError("Configuration requires data_root, manifest_dir, flux_model, segformer_checkpoint")
    if value.get("training", {}).get("Fresh_S7", {}).get("learning_rate") != 1e-4:
        raise ValueError("Original Fresh_S7 learning rate must remain 0.0001")
    return value


def configured_paths(config: dict) -> dict[str, Path]:
    # Relative paths are resolved against the invoking working directory.
    return {name: Path(value).expanduser().resolve() for name, value in config["paths"].items()}


def check_config(config: dict, *, action: str = "check", system: str | None = None) -> dict:
    paths = configured_paths(config)
    required = ["data_root", "manifest_dir"]
    if action in {"check", "train", "predict", "develop", "timing"}:
        if system != "Fresh_SegFormer_B2":
            required.append("flux_model")
        if system in {None, "Fresh_SegFormer_B2"}:
            required.append("segformer_checkpoint")
    missing = [name for name in required if not paths[name].exists()]
    manifests = ["train.json", "development.json", "replication-confirmation.inputs.json", "taxonomy.json"]
    missing.extend(f"manifest:{name}" for name in manifests if not (paths["manifest_dir"] / name).is_file())
    versions = {}
    for name, expected in EXPECTED_PACKAGES.items():
        try:
            actual = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            actual = None
        versions[name] = {"expected": expected, "installed": actual,
                          "matches": actual is not None and actual.split("+")[0] == expected}
    try:
        diffusers = importlib.metadata.version("diffusers")
    except importlib.metadata.PackageNotFoundError:
        diffusers = None
    return {"status": "PASS" if not missing else "MISSING_INPUTS", "missing": missing,
            "packages": versions, "diffusers_installed": diffusers,
            "diffusers_identity": "Use the immutable source distribution specified by the repository environment lock.",
            "models_loaded": False, "label_pixels_read": False}


def _copy_metadata(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.read_bytes() != source.read_bytes():
        raise ValueError(f"Existing runtime input differs from the configured manifest: {destination.name}")
    if not destination.exists():
        shutil.copyfile(source, destination)


def activate(config: dict, workspace: Path):
    """Bind user assets without changing retained model or statistical functions."""
    from researchos import s8r_ex01_training as training
    from researchos import s8r_ex01_evaluation as evaluation
    from researchos.s6_supervised import build_openmmlab_segformer_b2, load_openmmlab_segformer_b2_checkpoint

    paths = configured_paths(config)
    data_root, manifests = paths["data_root"], paths["manifest_dir"]
    metadata = workspace / ".inputs"
    preflight = metadata / "audit/S8R_Ex01_preregistration_preflight_v1_1"
    hardening = metadata / "audit/S8R_Ex01_preregistration_hardening"
    for source, target in (("train.json", "formal_train_manifest.json"),
                           ("development.json", "formal_development_manifest.json"),
                           ("replication-confirmation.inputs.json", "formal_confirmation_input_only_manifest.json")):
        _copy_metadata(manifests / source, preflight / target)
    _copy_metadata(manifests / "taxonomy.json", hardening / "complete_taxonomy_mapping.json")

    original_manifest = training._manifest
    original_taxonomy = training._taxonomy
    original_verified_path = training._verified_path
    original_confirmation_rows = evaluation._confirmation_rows
    original_evaluation_taxonomy = evaluation._taxonomy
    original_read_label = evaluation._read_label

    training._manifest = lambda root, **kwargs: original_manifest(metadata, **kwargs)
    training._taxonomy = lambda root: original_taxonomy(metadata)
    training._verified_path = lambda root, row, kind: original_verified_path(data_root, row, kind)
    training._model_root = lambda root: paths["flux_model"]

    def segformer_model(root):
        model = build_openmmlab_segformer_b2(num_classes=3)
        report = load_openmmlab_segformer_b2_checkpoint(model, paths["segformer_checkpoint"])
        return model, report

    training._segformer_model = segformer_model
    evaluation._confirmation_rows = lambda root: original_confirmation_rows(metadata)
    evaluation._taxonomy = lambda root: original_evaluation_taxonomy(metadata)
    evaluation._read_label = lambda root, binding, lookup: original_read_label(data_root, binding, lookup)

    def label_bindings(root, sample_ids):
        payload = read_json(manifests / "replication-confirmation.labels.json")
        if payload.get("split") != "replication-confirmation" or payload.get("label_encoding") != "source_ids":
            raise ValueError("Evaluation labels must use original source IDs in the primary partition")
        rows = payload["rows"]
        bindings = {row["sample_id"]: {"label_path": row["label_path"], "label_sha256": row["label_sha256"]} for row in rows}
        if len(bindings) != len(rows) or set(bindings) != sample_ids:
            raise ValueError("Evaluation label sample set is incomplete or duplicated")
        return bindings

    evaluation._label_bindings = label_bindings
    return training, evaluation


def bind_device(device: str) -> dict:
    os.environ["CUDA_VISIBLE_DEVICES"] = device
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "DIFFUSERS_OFFLINE", "HF_DATASETS_OFFLINE"):
        os.environ[name] = "1"
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Select exactly one available CUDA device with --device")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("The original training and inference require bfloat16 support")
    return {"visible_gpu_uuid": str(torch.cuda.get_device_properties(0).uuid)}


def paired_checkpoint(workspace: Path, seed: int, override: Path | None) -> Path:
    import torch
    directory = (override or workspace / "Fresh_S7" / f"seed_{seed}" / "checkpoint_epoch_10").resolve(strict=True)
    state = torch.load(directory / "training_state.pt", map_location="cpu", weights_only=False)
    if state.get("system") != "Fresh_S7" or state.get("seed") != seed or state.get("completed_epochs") != 10:
        raise ValueError("Arm A requires the same-seed final Fresh_S7 checkpoint")
    if not (directory / "pytorch_lora_weights.safetensors").is_file():
        raise FileNotFoundError("Paired Fresh_S7 adapter is absent")
    return directory


def collect_predictions(workspace: Path) -> list[dict]:
    rows = []
    for system in SYSTEMS:
        for seed in SEEDS:
            result = read_json(workspace / system / f"seed_{seed}" / "confirmation_prediction_receipt.json")
            if result.get("status") != "PASS" or result.get("system") != system or result.get("seed") != seed:
                raise ValueError("Prediction inventory system/seed binding differs")
            rows.extend(sorted(result["inventory"], key=lambda row: int(row["confirmation_index"])))
    return rows


def run_evaluation(workspace: Path, evaluation, output: Path) -> dict:
    from researchos.s8r_ex01_execution import TASK_ID, validate_joint_prediction_inventory
    rows = collect_predictions(workspace)
    seal = validate_joint_prediction_inventory(rows)
    # This in-memory compatibility object binds an explicitly requested evaluation
    # to the complete prediction inventory; no external authorization file is used.
    binding = {"task_id": TASK_ID, "authorized_mask_open_count": 844,
               "final_method_test_label_open_count": 0,
               "joint_inventory_sha256": seal["inventory_sha256"]}
    return evaluation.evaluate_confirmation(project_root=workspace, inventory_rows=rows,
        joint_seal=seal, label_release_token=binding, output_path=output)


def run_h4(workspace: Path, output: Path) -> dict:
    from researchos.s8r_ex01_execution import h4_bootstrap
    rows = []
    for system in ("Fresh_S7", "Fresh_Arm_B"):
        for seed in SEEDS:
            payload = read_json(workspace / system / f"seed_{seed}" / "cost40_timing_receipt.json")
            rows.extend(payload["timing_rows"])
    result = h4_bootstrap(rows)
    destination = workspace / output
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2) + "\n")
    return result


def write_final_predictions(training, config: dict, workspace: Path, output: Path,
                            system: str, seed: int, s7_checkpoint: Path | None) -> dict:
    from researchos.s8r_ex01_execution import sha256_file
    payload = read_json(configured_paths(config)["manifest_dir"] / "final-method-test.inputs.json")
    rows = payload.get("rows", [])
    if payload.get("split") != "final-method-test" or payload.get("labels_present") is not False or len(rows) != 1163:
        raise ValueError("Final prediction requires the 1,163-image input-only manifest")
    if (len({row["sample_id"] for row in rows}) != 1163
            or [int(row["index"]) for row in rows] != list(range(1163))
            or any(row.get("split") != "final-method-test" for row in rows)):
        raise ValueError("Final input sample identity, split, or order differs")
    if any(any("label" in key for key in row) for row in rows):
        raise ValueError("Final prediction input rows must not contain label fields")
    predictor = training._Predictor(root=workspace, output=output, system=system, seed=seed,
                                   s7_checkpoint=s7_checkpoint)
    directory = output / "final_predictions"
    inventory = []
    try:
        for row in rows:
            path = directory / f"{int(row['index']):04d}.png"
            if path.exists():
                raise FileExistsError(path)
            prediction = predictor.predict(training._load_rgb(workspace, row),
                sample_seed=training._sample_seed(seed, str(row["sample_id"])))
            training._save_prediction(path, prediction)
            inventory.append({"sample_id": row["sample_id"], "path": path.relative_to(output).as_posix(),
                "sha256": sha256_file(path), "source": row["source_set"],
                "session_id": row["recording_session_key"]})
    finally:
        predictor.close()
    result = {"system": system, "seed": seed, "split": "final-method-test",
        "checkpoint_sha256": sha256_file(output / "checkpoint_epoch_10/training_state.pt"), "rows": inventory}
    (output / "final_predictions.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("check", "train", "predict", "develop", "timing", "evaluate", "h4"))
    parser.add_argument("--config", type=Path, default=Path("configs/original/runtime.example.json"))
    parser.add_argument("--workspace", type=Path, default=Path("runs/original"))
    parser.add_argument("--system", choices=SYSTEMS)
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--split", choices=("replication-confirmation", "final-method-test"), default="replication-confirmation",
                        help="Input partition for the predict action")
    parser.add_argument("--process", choices=("process_a", "process_b"))
    parser.add_argument("--device", default="0", help="One CUDA ordinal or device UUID")
    parser.add_argument("--paired-s7", type=Path)
    parser.add_argument("--output", type=Path, help="Evaluation result path relative to --workspace")
    parser.add_argument("--check-config", action="store_true", help="Validate paths and inspect package versions without importing models")
    args = parser.parse_args(argv)
    if args.action == "h4" and not args.check_config:
        result = run_h4(args.workspace.resolve(strict=True), args.output or Path("h4.json"))
        print(json.dumps(result, indent=2))
        return 0
    config = load_config(args.config)
    check = check_config(config, action=args.action, system=args.system)
    if args.action == "check" or args.check_config:
        print(json.dumps(check, indent=2))
        return 0 if check["status"] == "PASS" else 2
    if check["missing"]:
        parser.error("Missing configured inputs: " + ", ".join(check["missing"]))
    workspace = args.workspace.resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    training, evaluation = activate(config, workspace)
    if args.action == "evaluate":
        result = run_evaluation(workspace, evaluation, args.output or Path("quality.json"))
    else:
        if args.system is None or args.seed is None:
            parser.error("This action requires --system and --seed")
        if args.action == "train" and args.process is None:
            parser.error("Training requires --process process_a or process_b")
        gpu = bind_device(args.device)
        s7 = paired_checkpoint(workspace, args.seed, args.paired_s7) if args.system == "Fresh_Arm_A" else None
        output = workspace / args.system / f"seed_{args.seed}"
        from researchos.s8r_ex01_execution import sha256_file
        kwargs = dict(project_root=workspace, config=config, config_path=args.config,
            config_sha256=sha256_file(args.config), system=args.system, seed=args.seed,
            output_dir=output, gpu=gpu, s7_checkpoint=s7)
        if args.action == "train":
            function = training.train_flux_segment if args.system == "Fresh_S7" else (
                training.train_segformer_segment if args.system == "Fresh_SegFormer_B2" else training.train_native_segment)
            result = function(**kwargs, process=args.process)
        elif args.action == "predict":
            if args.split == "final-method-test":
                result = write_final_predictions(training, config, workspace, output, args.system, args.seed, s7)
            else:
                result = training.write_confirmation_predictions(**kwargs)
                input_rows = training._manifest(workspace, split="replication-confirmation", labels_required=False)
                metadata = {row["sample_id"]: row for row in input_rows}
                compatible = {"system": args.system, "seed": args.seed,
                    "checkpoint_sha256": sha256_file(output / "checkpoint_epoch_10/training_state.pt"),
                    "rows": [{"sample_id": row["sample_id"],
                        "path": (workspace / row["prediction_path"]).relative_to(output).as_posix(),
                        "sha256": row["prediction_sha256"],
                        "source": metadata[row["sample_id"]]["source_set"],
                        "session_id": metadata[row["sample_id"]]["recording_session_key"]}
                        for row in result["inventory"]]}
                (output / "predictions.json").write_text(json.dumps(compatible, indent=2) + "\n")
        elif args.action == "develop":
            result = training.run_development(**kwargs)
        else:
            timing_source = configured_paths(config)["manifest_dir"] / "cost40.inputs.json"
            timing_target = workspace / ".inputs/cost40.inputs.json"
            _copy_metadata(timing_source, timing_target)
            config["efficiency"] = {"confirmatory_bootstrap": {"input_manifest": ".inputs/cost40.inputs.json"}}
            result = training.run_cost40_timing(**kwargs)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0
