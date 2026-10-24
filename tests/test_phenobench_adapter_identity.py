"""Synthetic run-record checks for the two supported S7 adapter identities."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from safetensors.torch import save_file
from researchos import phenobench_adapter_identity as identity
from researchos.portable_data import sha256_file
from researchos.portable_phenobench import verify_native_pair_identity


class AdapterIdentityTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()
        self.model = self.root / "model"
        self.model.mkdir()
        model_file = self.model / "model_index.json"
        model_file.write_text('{"synthetic": true}')
        snapshot = {"revision": identity.MODEL_REVISION, "files": [
            {"path": model_file.name, "bytes": model_file.stat().st_size, "sha256": sha256_file(model_file)}]}
        # Replace only the published asset inventory with a tiny synthetic fixture.
        # Production constants and validation APIs have no skip-check option.
        self.snapshot_patch = patch("jnda.controls.model_assets.FLUX_SNAPSHOT", snapshot)
        self.snapshot_patch.start()
        self.count_patch = patch.multiple(identity, ADAPTER_TENSOR_COUNT=2, ADAPTER_PARAMETER_COUNT=8)
        self.count_patch.start()
        identity._MODEL_CACHE.clear()
        self.seed = 20260723
        self.native_sha = "1" * 64
        self.s7_sha = "2" * 64
        self.output = self.root / "s7"
        self.output.mkdir()
        tensors = {"transformer.layer.lora_A.weight": torch.arange(4, dtype=torch.float32).reshape(2, 2),
                   "transformer.layer.lora_B.weight": torch.arange(4, dtype=torch.float32).reshape(2, 2)}
        self.adapter = self.output / "pytorch_lora_weights.safetensors"
        save_file({key: value.to(torch.bfloat16) for key, value in tensors.items()}, str(self.adapter),
                  metadata={"synthetic_export": "bfloat16"})
        runtime = identity.training_identity(self.model)
        recipe_path = Path(identity.__file__).resolve().parents[2] / "configs/original/phenobench_s7_prerequisite_recipe.json"
        recipe = json.loads(recipe_path.read_text())
        config_sha = identity.canonical_hash({"recipe": recipe, "model_revision": identity.MODEL_REVISION})
        for phase, start, stop, pid in (("process_a", 0, 7035, 1001), ("process_b", 7035, 14070, 1002)):
            checkpoint = self.output / f"checkpoint-{stop}"
            checkpoint.mkdir()
            save_file(tensors, str(checkpoint / self.adapter.name), metadata={"synthetic_checkpoint": "float32"})
            for name in ("optimizer.bin", "random_states_0.pkl", "scheduler.bin"):
                (checkpoint / name).write_bytes(b"synthetic state")
            inventory = [{"path": path.name, "size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
                         for path in sorted(checkpoint.iterdir())]
            metrics = self.output / f"metrics_{phase}.jsonl"
            metrics.write_text("".join(json.dumps({"step": step, "loss": 1.0, "lr": 0.0001}) + "\n"
                                       for step in range(start + 1, stop + 1)))
            result = {"status": "COMPLETE", "seed": self.seed, "process": phase, "process_id": pid,
                "start_global_step": start, "final_global_step": stop, "optimizer_steps_this_process": 7035,
                "training_batches": 7035, "pair_count": 1407, "training_identity": runtime,
                "config_sha256": config_sha, "native_train_manifest_sha256": self.native_sha,
                "manifest_sha256": self.s7_sha, "official_trainer_sha256": identity.OFFICIAL_TRAINER_SHA256,
                "metrics_sha256": sha256_file(metrics), "checkpoint_files": inventory,
                "adapter": {"path": str(self.adapter), "sha256": sha256_file(self.adapter)},
                "producer_run_sha256": sha256_file(self.output / "run_process_a.json") if phase == "process_b" else None}
            (self.output / f"run_{phase}.json").write_text(json.dumps(result))
        self.manifest = identity.create_reproduction_binding(self.output, seed=self.seed, identity=runtime,
            native_train_sha=self.native_sha, s7_manifest_sha=self.s7_sha)
        self.manifest_path = self.output / "adapter_binding.json"
        self.manifest_path.write_text(json.dumps(self.manifest))
        self.binding = {"kind": "public_s7_reproduction", "manifest": str(self.manifest_path)}

    def tearDown(self):
        self.count_patch.stop()
        self.snapshot_patch.stop()
        identity._MODEL_CACHE.clear()
        self.temporary.cleanup()

    def resolve(self, binding=None, seed=None):
        return identity.resolve_adapter(binding or self.binding, seed=seed or self.seed,
            model_root=self.model, native_train_manifest_sha256=self.native_sha)

    def replace_manifest(self, value):
        value["manifest_sha256"] = identity.canonical_hash({key: item for key, item in value.items() if key != "manifest_sha256"})
        self.manifest_path.write_text(json.dumps(value))

    def test_new_complete_adapter_accepts_different_serialization_sha(self):
        self.assertNotEqual(sha256_file(self.adapter), sha256_file(self.output / "checkpoint-14070" / self.adapter.name))
        result = self.resolve()
        self.assertFalse(result["published_checkpoint"])
        self.assertEqual(result["kind"], "public_s7_reproduction")
        self.assertEqual(result["sha256"], sha256_file(self.adapter))

    def test_frozen_identity_is_seed_specific_and_does_not_accept_overrides(self):
        actual = sha256_file(self.adapter)
        with patch.dict(identity.FROZEN_ADAPTER_SHA256, {self.seed: actual, 20260724: "a" * 64}):
            frozen = {"kind": "published_frozen", "path": str(self.adapter), "sha256": actual}
            self.assertTrue(self.resolve(frozen)["published_checkpoint"])
            with self.assertRaises(ValueError):
                self.resolve(frozen, 20260724)
            with self.assertRaises(ValueError):
                self.resolve({**frozen, "sha256": "b" * 64})

    def test_new_manifest_rejects_wrong_seed_or_incomplete_state(self):
        with self.assertRaises(ValueError):
            self.resolve(seed=20260724)
        for key, invalid in (("status", "RUNNING"), ("final_global_step", 7035), ("published_checkpoint", True)):
            value = copy.deepcopy(self.manifest)
            value[key] = invalid
            self.replace_manifest(value)
            with self.assertRaises(ValueError):
                self.resolve()

    def test_model_and_source_identity_mismatches_are_rejected(self):
        value = copy.deepcopy(self.manifest)
        value["training_identity"]["source_identity_sha256"] = "f" * 64
        self.replace_manifest(value)
        with self.assertRaises(ValueError):
            self.resolve()
        self.replace_manifest(copy.deepcopy(self.manifest))
        (self.model / "model_index.json").write_text('{"synthetic": false}')
        with self.assertRaises(ValueError):
            self.resolve()

    def test_incomplete_optimizer_logs_and_wrong_final_tensors_are_rejected(self):
        path = self.output / "run_process_b.json"
        value = json.loads(path.read_text())
        value["optimizer_steps_this_process"] = 7034
        path.write_text(json.dumps(value))
        with self.assertRaises(ValueError):
            identity.create_reproduction_binding(self.output, seed=self.seed,
                identity=self.manifest["training_identity"], native_train_sha=self.native_sha, s7_manifest_sha=self.s7_sha)
        value["optimizer_steps_this_process"] = 7035
        altered = {"transformer.layer.lora_A.weight": torch.ones(2, 2, dtype=torch.bfloat16),
                   "transformer.layer.lora_B.weight": torch.ones(2, 2, dtype=torch.bfloat16)}
        save_file(altered, str(self.adapter))
        value["adapter"]["sha256"] = sha256_file(self.adapter)
        path.write_text(json.dumps(value))
        with self.assertRaises(ValueError):
            identity.create_reproduction_binding(self.output, seed=self.seed,
                identity=self.manifest["training_identity"], native_train_sha=self.native_sha, s7_manifest_sha=self.s7_sha)

    def test_free_form_new_adapter_and_changed_recipe_are_rejected(self):
        with self.assertRaises(ValueError):
            self.resolve({**self.binding, "path": str(self.adapter), "sha256": sha256_file(self.adapter)})
        with self.assertRaises(ValueError):
            identity.fixed_recipe_identity({"training": {"learning_rate": 0.001}})

    def test_native_pair_requires_same_seed_and_shared_input_identity(self):
        left = {"seed": self.seed, "vae_cache_manifest_sha256": "3" * 64,
            "native_train_manifest_sha256": self.native_sha, "model_manifest_sha256": "4" * 64,
            "prompt_cache_sha256": "5" * 64}
        verify_native_pair_identity(left, dict(left))
        with self.assertRaises(ValueError):
            verify_native_pair_identity(left, {**left, "seed": 20260724})
        with self.assertRaises(ValueError):
            verify_native_pair_identity(left, {**left, "native_train_manifest_sha256": "6" * 64})

    def test_stale_s7_inputs_are_rejected_before_model_identity_or_training(self):
        from researchos import portable_phenobench_s7 as portable
        from researchos import phenobench_s7_bridge as bridge
        workspace = self.root / "inputs"
        workspace.mkdir()
        (workspace / "train.json").write_text(json.dumps({"rows": [
            {"sample_id": "new_data", "image_sha256": "9" * 64}]}))
        stale = {"manifest_sha256": "7" * 64, "source_native_train_manifest_sha256": "8" * 64}
        records = [{"sample_id": "old_data", "condition_sha256": "0" * 64}]
        official = Path(identity.__file__).resolve().parents[2] / identity.SOURCE_FILES[-1]
        prior = (bridge.ROOT, bridge.MODEL_ROOT, bridge.OFFICIAL_SOURCE)
        try:
            with patch.object(portable, "load_manifest", return_value=(stale, records)), \
                    patch.object(identity, "training_identity", side_effect=AssertionError("must reject before assets load")), \
                    patch.object(bridge, "_load_official_module", side_effect=AssertionError("must not load trainer")):
                with self.assertRaisesRegex(ValueError, "Prepared S7 pairs"):
                    portable.train({"flux_model": str(self.model)}, workspace,
                        {"prompt": {"text": bridge.PROMPT}}, official, self.seed, "process_a")
        finally:
            bridge.ROOT, bridge.MODEL_ROOT, bridge.OFFICIAL_SOURCE = prior

    def test_arm_a_cache_rejects_a_changed_prompt_before_loading_tensors(self):
        from researchos import portable_phenobench as portable
        from researchos import s8_native_training as native
        workspace = self.root / "native"
        (workspace / "cache/vae").mkdir(parents=True)
        (workspace / "train.json").write_text('{"rows": []}')
        prompt = workspace / "cache/prompt.safetensors"
        prompt.write_bytes(b"changed synthetic prompt")
        model = {"manifest_sha256": "a" * 64}
        s7 = {"kind": "public_s7_reproduction", "seed": self.seed, "sha256": "b" * 64}
        vae = {"model_identity": model, "train_manifest_sha256": sha256_file(workspace / "train.json")}
        vae["manifest_sha256"] = identity.canonical_hash(vae)
        (workspace / "cache/vae/manifest.json").write_text(json.dumps(vae))
        arm_cache = {"source_vae_manifest_sha256": vae["manifest_sha256"], "s7_adapter_identity": s7,
                     "prompt_cache_sha256": "c" * 64}
        arm_cache["manifest_sha256"] = identity.canonical_hash(arm_cache)
        runtime = SimpleNamespace(PHENOBENCH_MODEL_IDENTITY=model, PHENOBENCH_S7_IDENTITY=s7)
        with patch.object(portable, "configure_runtime", return_value=runtime), \
                patch.object(native, "load_cache_tensors", return_value={"manifest": arm_cache}), \
                patch.object(native, "load_prompt_cache", side_effect=AssertionError("must reject before reading prompt tensors")):
            with self.assertRaisesRegex(ValueError, "different prompt-cache"):
                portable.train({}, workspace, "A", self.seed, "process_a", {})


if __name__ == "__main__":
    unittest.main()
