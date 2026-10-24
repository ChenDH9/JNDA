"""CPU-only checks for the PhenoBench S7 bridge and semantic evaluator."""
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from researchos import phenobench_s7_bridge as bridge
from researchos.portable_data import read_csv, sha256_file
from researchos.portable_phenobench import evaluate


class PhenoBenchPortableTests(unittest.TestCase):
    def test_s7_bounded_sampler_and_original_arguments(self):
        sampler_class = bridge.build_deterministic_batch_sampler_class(resume_step=7035, target_step=14070)
        sampler = sampler_class(list(range(1407)), batch_size=1, drop_last=True, shuffle_batches_each_epoch=False)
        self.assertEqual(len(list(iter(sampler))), 1407)
        sampler_class.training_started = True
        self.assertEqual(sum(len(list(iter(sampler))) for _ in range(5)), 7035)
        self.assertEqual(list(iter(sampler)), [])
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "transformer").mkdir()
            (root / "transformer/config.json").write_text('{"num_single_layers": 20}')
            prior_root, prior_model = bridge.ROOT, bridge.MODEL_ROOT
            bridge.ROOT = bridge.MODEL_ROOT = root
            try:
                recipe = json.loads((ROOT / "configs/original/phenobench_s7_prerequisite_recipe.json").read_text())
                config = {"training": {**recipe["training"], "seed": 20260723},
                    "runtime": {"instance_prompt": recipe["prompt"]["text"]}, "dataset_manifest": "train.json"}
                arguments = bridge.build_official_arguments(config=config, output_dir=root / "output",
                    max_steps=14070, resume_from_checkpoint="checkpoint-7035", checkpointing_steps=1407)
                self.assertEqual(arguments[arguments.index("--guidance_scale") + 1], "1.0")
                self.assertEqual(arguments[arguments.index("--max_train_steps") + 1], "14070")
                self.assertEqual(len(arguments[arguments.index("--lora_layers") + 1].split(",")), 25)
                self.assertIn("--cache_latents", arguments)
                self.assertIn("--offload", arguments)
            finally:
                bridge.ROOT, bridge.MODEL_ROOT = prior_root, prior_model

    def test_semantic_evaluation_of_500_synthetic_perfect_maps(self):
        selected = read_csv(ROOT / "data/phenobench_validation_selection.csv")
        mask = Image.fromarray(np.tile(np.arange(3, dtype=np.uint8), (3, 1)))
        raw_buffer, prediction_buffer = io.BytesIO(), io.BytesIO()
        mask.save(raw_buffer, format="PNG")
        mask.resize((512, 512), Image.Resampling.NEAREST).save(prediction_buffer, format="PNG")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data, workspace = root / "data", root / "workspace"
            labels = data / "val/semantics"
            directory = workspace / "arm_B/seed_20260723/validation_predictions"
            labels.mkdir(parents=True)
            directory.mkdir(parents=True)
            inputs, predictions = [], []
            for row in selected:
                sample_id = row["sample_id"]
                (labels / sample_id).write_bytes(raw_buffer.getvalue())
                path = directory / sample_id
                path.write_bytes(prediction_buffer.getvalue())
                inputs.append({"sample_id": sample_id})
                predictions.append({"sample_id": sample_id, "prediction_path": path.relative_to(workspace).as_posix(),
                                    "prediction_sha256": sha256_file(path)})
            (workspace / "validation.inputs.json").write_text(json.dumps({"rows": inputs}))
            (directory / "inventory.json").write_text(json.dumps({"arm": "B", "seed": 20260723, "inventory": predictions}))
            result = evaluate({"data_root": str(data)}, workspace, "B", 20260723)
            self.assertEqual(result["mean_seed_miou"], 1.0)
            self.assertEqual(np.array(result["per_seed"]["20260723"]["confusion_matrix"]).sum(), 500 * 512 * 512)


if __name__ == "__main__":
    unittest.main()
