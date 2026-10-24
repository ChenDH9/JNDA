"""Validate published and newly reproduced PhenoBench S7 adapter identities."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from researchos.portable_data import sha256_file

MODEL_REPOSITORY = "black-forest-labs/FLUX.2-klein-base-4B"
MODEL_REVISION = "8c44a2fbef88fae175da65df054db0f901aa9747"
OFFICIAL_TRAINER_SHA256 = "6486a9d7b56264a047339c3e8291d0d55cae51266bb2c3c5d8d533872330e31e"
ORIGINAL_RECIPE_SHA256 = "31e06b4ed7a1ee1cca3239e45eab65491958df4e3b55391809ab3e5f0500f964"
FROZEN_ADAPTER_SHA256 = {
    20260723: "fb6f0f5bac3543ba848af44df12b4b5fc1d5c58bf274a3dfca2a694227c03924",
    20260724: "feebcce77b90f14fa0a458a05e4e8ca93da618f4c5a1625a84d2e30fee46a431",
    20260725: "b54b1eebcf7f5e50dc56882ce9656b438f5af806c73759e4a5ea5ec0404fed45",
}
SOURCE_FILES = (
    "src/researchos/phenobench_s7_bridge.py",
    "src/researchos/portable_phenobench_s7.py",
    "src/researchos/phenobench_adapter_identity.py",
    "src/researchos/phenobench_labels.py",
    "src/researchos/flux2_klein_i2i/paired_transforms.py",
    "src/researchos/flux2_klein_i2i/real_trainer.py",
    "third_party/diffusers_examples/train_dreambooth_lora_flux2_klein_img2img.py",
)
_MODEL_CACHE = {}
ADAPTER_TENSOR_COUNT = 120
ADAPTER_PARAMETER_COUNT = 8355840


def canonical_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError("Expected an adapter identity object")
    return value


def fixed_recipe_identity(recipe: dict | None = None) -> str:
    if recipe is None:
        recipe = read_json(Path(__file__).resolve().parents[2] / "configs/original/phenobench_s7_prerequisite_recipe.json")
    digest = canonical_hash(recipe)
    if digest != ORIGINAL_RECIPE_SHA256:
        raise ValueError("S7 reproduction requires the original fixed scientific recipe")
    return digest


def model_identity(root: Path) -> dict:
    """Hash the actual model components; no model is imported or executed."""
    from jnda.controls.model_assets import FLUX_SNAPSHOT
    from jnda.controls.runtime import model_identity as verify_fixed_snapshot
    root = root.resolve(strict=True)
    if FLUX_SNAPSHOT["revision"] != MODEL_REVISION:
        raise ValueError("The registered FLUX snapshot revision differs")
    files = [root / row["path"] for row in FLUX_SNAPSHOT["files"]]
    key = (str(root), canonical_hash(FLUX_SNAPSHOT),
        tuple((path.relative_to(root).as_posix(), path.stat().st_size, path.stat().st_mtime_ns) for path in files))
    if key not in _MODEL_CACHE:
        verified = verify_fixed_snapshot(root, family="flux")
        rows = [{"path": row["path"], "size_bytes": row["bytes"], "sha256": row["sha256"]}
                for row in FLUX_SNAPSHOT["files"]]
        _MODEL_CACHE[key] = {"repository": MODEL_REPOSITORY, "revision": MODEL_REVISION,
            "files": rows, "manifest_sha256": canonical_hash(rows), **verified}
    return json.loads(json.dumps(_MODEL_CACHE[key]))


def training_identity(model_root: Path, recipe: dict | None = None) -> dict:
    repository = Path(__file__).resolve().parents[2]
    sources = {name: sha256_file(repository / name) for name in SOURCE_FILES}
    if sources[SOURCE_FILES[-1]] != OFFICIAL_TRAINER_SHA256:
        raise ValueError("Packaged official S7 trainer differs from its fixed source identity")
    return {"model": model_identity(model_root), "recipe_sha256": fixed_recipe_identity(recipe),
        "official_trainer_sha256": OFFICIAL_TRAINER_SHA256,
        "source_files": sources, "source_identity_sha256": canonical_hash(sources)}


def _relative_file(root: Path, value: str) -> Path:
    supplied = Path(value)
    if supplied.is_absolute() or ".." in supplied.parts:
        raise ValueError("S7 run identity references must be local relative paths")
    path = (root / supplied).resolve(strict=True)
    path.relative_to(root.resolve())
    if not path.is_file():
        raise ValueError("S7 run identity reference is not a file")
    return path


def _verify_checkpoint(output: Path, name: str, inventory: list[dict]) -> None:
    required = {"optimizer.bin", "pytorch_lora_weights.safetensors", "random_states_0.pkl", "scheduler.bin"}
    directory = output / name
    if not isinstance(inventory, list) or not required <= {row.get("path") for row in inventory}:
        raise ValueError("S7 complete optimizer/scheduler/RNG/adapter checkpoint is absent")
    actual_names = {p.relative_to(directory).as_posix() for p in directory.rglob("*") if p.is_file()}
    if actual_names != {row["path"] for row in inventory} or len(actual_names) != len(inventory):
        raise ValueError("S7 checkpoint inventory is incomplete or duplicated")
    for row in inventory:
        path = _relative_file(directory, row["path"])
        if path.stat().st_size != row["size_bytes"] or sha256_file(path) != row["sha256"]:
            raise ValueError("S7 checkpoint file differs from its run identity")


def _verify_stage(output: Path, phase: str, seed: int, identity: dict, native_train_sha: str, s7_manifest_sha: str) -> dict:
    value = read_json(output / f"run_{phase}.json")
    start, stop = (0, 7035) if phase == "process_a" else (7035, 14070)
    recipe = read_json(Path(__file__).resolve().parents[2] / "configs/original/phenobench_s7_prerequisite_recipe.json")
    fixed_recipe_identity(recipe)
    checks = (value.get("status") == "COMPLETE", value.get("seed") == seed,
        value.get("process") == phase, value.get("start_global_step") == start,
        value.get("final_global_step") == stop, value.get("optimizer_steps_this_process") == 7035,
        value.get("training_batches") == 7035, value.get("pair_count") == 1407,
        value.get("training_identity") == identity,
        value.get("config_sha256") == canonical_hash({"recipe": recipe, "model_revision": MODEL_REVISION}),
        value.get("native_train_manifest_sha256") == native_train_sha,
        value.get("manifest_sha256") == s7_manifest_sha,
        value.get("official_trainer_sha256") == OFFICIAL_TRAINER_SHA256)
    if not all(checks):
        raise ValueError(f"S7 {phase} identity is incomplete or mismatched")
    if type(value.get("process_id")) is not int or value["process_id"] <= 0:
        raise ValueError("S7 completed stage has no valid producer process identity")
    metrics = output / f"metrics_{phase}.jsonl"
    if sha256_file(metrics) != value.get("metrics_sha256"):
        raise ValueError("S7 optimizer metric file digest differs")
    rows = [json.loads(line) for line in metrics.read_text().splitlines() if line.strip()]
    if [row.get("step") for row in rows] != list(range(start + 1, stop + 1)):
        raise ValueError("S7 optimizer steps are not complete and contiguous")
    if any(type(row.get(key)) not in (int, float) or not math.isfinite(row[key])
           for row in rows for key in ("loss", "lr")):
        raise ValueError("S7 optimizer records contain invalid loss/LR values")
    if any(not math.isclose(float(row["lr"]), float(recipe["training"]["learning_rate"]), rel_tol=1e-12, abs_tol=0.0) for row in rows):
        raise ValueError("S7 optimizer records differ from the fixed constant learning rate")
    _verify_checkpoint(output, f"checkpoint-{stop}", value.get("checkpoint_files", []))
    return value


def verify_final_adapter_export(adapter: Path, checkpoint_adapter: Path) -> dict:
    """Compare tensor content after the official final BF16 export conversion."""
    import torch
    from safetensors.torch import load_file

    def normalized(path):
        values = load_file(str(path), device="cpu")
        result = {}
        for key, value in values.items():
            key = key.removeprefix("transformer.")
            if key in result or not any(marker in key for marker in (".lora_A.", ".lora_B.")):
                raise ValueError("Unexpected or colliding S7 adapter tensor name")
            result[key] = value
        return result

    final, checkpoint = normalized(adapter), normalized(checkpoint_adapter)
    if set(final) != set(checkpoint) or len(final) != ADAPTER_TENSOR_COUNT:
        raise ValueError("Final adapter tensor names/count differ from checkpoint-14070")
    if sum(value.numel() for value in final.values()) != ADAPTER_PARAMETER_COUNT:
        raise ValueError("Final adapter parameter count differs from the original rank-8 architecture")
    for key, value in final.items():
        previous = checkpoint[key]
        if (value.dtype != torch.bfloat16 or value.shape != previous.shape
                or not bool(torch.isfinite(value).all())
                or not torch.equal(value, previous.to(torch.bfloat16))):
            raise ValueError("Final adapter tensor differs from the registered BF16 checkpoint export")
    return {"status": "EXACT_AFTER_REGISTERED_BF16_CAST", "tensor_count": len(final),
        "parameter_count": sum(value.numel() for value in final.values()),
        "checkpoint_adapter_sha256": sha256_file(checkpoint_adapter),
        "final_adapter_sha256": sha256_file(adapter)}


def create_reproduction_binding(output: Path, *, seed: int, identity: dict,
                                native_train_sha: str, s7_manifest_sha: str) -> dict:
    """Issue the reusable adapter manifest only after both completed stages verify."""
    output = output.resolve(strict=True)
    a = _verify_stage(output, "process_a", seed, identity, native_train_sha, s7_manifest_sha)
    b = _verify_stage(output, "process_b", seed, identity, native_train_sha, s7_manifest_sha)
    if a.get("process_id") == b.get("process_id") or b.get("producer_run_sha256") != sha256_file(output / "run_process_a.json"):
        raise ValueError("S7 continuation is not bound to a distinct complete producer process")
    adapter = output / "pytorch_lora_weights.safetensors"
    actual_sha = sha256_file(adapter)
    if b.get("adapter", {}).get("sha256") != actual_sha:
        raise ValueError("Final S7 adapter differs from the completed run")
    export = verify_final_adapter_export(adapter, output / "checkpoint-14070/pytorch_lora_weights.safetensors")
    value = {"schema_version": 1, "identity_kind": "public_s7_reproduction", "status": "COMPLETE",
        "system": "PhenoBench_S7_RGB_LoRA", "seed": seed, "pair_count": 1407,
        "completed_epochs": 10, "final_global_step": 14070, "published_checkpoint": False,
        "artifact": {"path": adapter.name, "sha256": actual_sha, "size_bytes": adapter.stat().st_size},
        "training_identity": identity, "adapter_export": export, "native_train_manifest_sha256": native_train_sha,
        "s7_manifest_sha256": s7_manifest_sha,
        "stage_records": {phase: {"path": f"run_{phase}.json", "sha256": sha256_file(output / f"run_{phase}.json")}
                          for phase in ("process_a", "process_b")}}
    value["manifest_sha256"] = canonical_hash(value)
    return value


def resolve_adapter(binding: dict, *, seed: int, model_root: Path,
                    native_train_manifest_sha256: str) -> dict:
    if seed not in FROZEN_ADAPTER_SHA256:
        raise ValueError("Unknown PhenoBench seed")
    kind = binding.get("kind", "published_frozen")
    if kind == "published_frozen":
        path = Path(binding["path"]).expanduser().resolve(strict=True)
        expected = FROZEN_ADAPTER_SHA256[seed]
        if binding.get("sha256", expected) != expected or sha256_file(path) != expected:
            raise ValueError("Published S7 adapter must match the frozen SHA256 for the requested seed")
        return {"kind": kind, "seed": seed, "path": str(path), "sha256": expected, "published_checkpoint": True}
    if kind != "public_s7_reproduction" or "manifest" not in binding:
        raise ValueError("New S7 adapters require a public_s7_reproduction run manifest")
    if set(binding) - {"kind", "manifest"}:
        raise ValueError("New S7 adapters must be selected by their run manifest, not a free-form path/SHA override")
    manifest_path = Path(binding["manifest"]).expanduser().resolve(strict=True)
    manifest = read_json(manifest_path)
    unsigned = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    if manifest.get("manifest_sha256") != canonical_hash(unsigned):
        raise ValueError("S7 adapter run-manifest digest differs")
    checks = (manifest.get("identity_kind") == kind, manifest.get("seed") == seed,
        manifest.get("system") == "PhenoBench_S7_RGB_LoRA", manifest.get("status") == "COMPLETE",
        manifest.get("published_checkpoint") is False, manifest.get("completed_epochs") == 10,
        manifest.get("final_global_step") == 14070, manifest.get("pair_count") == 1407,
        manifest.get("native_train_manifest_sha256") == native_train_manifest_sha256)
    if not all(checks):
        raise ValueError("S7 adapter run identity is incomplete, belongs to another seed, or uses different training inputs")
    expected_identity = training_identity(model_root)
    if manifest.get("training_identity") != expected_identity:
        raise ValueError("S7 model, recipe, trainer, or source identity differs from the current reproduction")
    for phase in ("process_a", "process_b"):
        record = manifest.get("stage_records", {}).get(phase, {})
        if record.get("path") != f"run_{phase}.json" or sha256_file(_relative_file(manifest_path.parent, record["path"])) != record.get("sha256"):
            raise ValueError("S7 completed-stage binding differs")
    rebuilt = create_reproduction_binding(manifest_path.parent, seed=seed, identity=expected_identity,
        native_train_sha=native_train_manifest_sha256, s7_manifest_sha=manifest["s7_manifest_sha256"])
    if rebuilt != manifest:
        raise ValueError("S7 adapter manifest does not match the verified complete run")
    path = _relative_file(manifest_path.parent, manifest["artifact"]["path"])
    return {"kind": kind, "seed": seed, "path": str(path), "sha256": manifest["artifact"]["sha256"],
        "manifest_sha256": manifest["manifest_sha256"], "published_checkpoint": False,
        "training_identity": expected_identity}
