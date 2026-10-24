"""Import/argument/data-interface checks only: never invoke official.main."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image
from researchos import phenobench_s7_bridge as bridge

ROOT = Path(__file__).resolve().parents[1]


class OfficialS7IntegrationTests(unittest.TestCase):
    def test_pinned_module_parser_and_synthetic_dataset_injection(self):
        prior = (bridge.ROOT, bridge.MODEL_ROOT, bridge.OFFICIAL_SOURCE)
        bridge.OFFICIAL_SOURCE = ROOT / "third_party/diffusers_examples/train_dreambooth_lora_flux2_klein_img2img.py"
        try:
            official = bridge._load_official_module()
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary).resolve()
                (root / "model/transformer").mkdir(parents=True)
                (root / "model/transformer/config.json").write_text('{"num_single_layers": 20}')
                bridge.ROOT, bridge.MODEL_ROOT = root, root / "model"
                recipe = json.loads((ROOT / "configs/original/phenobench_s7_prerequisite_recipe.json").read_text())
                config = {"training": {**recipe["training"], "seed": 20260723},
                    "runtime": {"instance_prompt": bridge.PROMPT}, "dataset_manifest": "train.json"}
                arguments = bridge.build_official_arguments(config=config, output_dir=root / "output",
                    max_steps=7035, resume_from_checkpoint=None, checkpointing_steps=1407)
                parsed = official.parse_args(arguments)
                parsed.report_to = None
                self.assertIsNone(parsed.report_to)
                self.assertEqual(parsed.optimizer.lower(), "adamw")
                self.assertFalse(parsed.use_8bit_adam)
                self.assertIsNone(parsed.bnb_quantization_config_path)
                self.assertIsNone(parsed.dataset_name)
                self.assertFalse(parsed.push_to_hub)
                self.assertFalse(parsed.upcast_before_saving)
                self.assertTrue(parsed.skip_final_inference)
                manifest = root / "train.json"
                manifest.write_text("{}")
                palette = ((0, 0, 0), (34, 139, 34), (255, 0, 255))
                records = []
                for index, color in enumerate(palette[:2]):
                    image, target = root / f"image_{index}.png", root / f"target_{index}.png"
                    Image.new("RGB", (2, 2), (30, 40, 50)).save(image)
                    Image.new("RGB", (2, 2), color).save(target)
                    records.append({"sample_id": f"synthetic_{index}", "condition_image": image.name, "target_image": target.name})
                payload = {"split": "train", "final_test": False, "instruction": bridge.PROMPT,
                    "canonical_palette": palette, "manifest_sha256": "0" * 64}
                with patch.object(bridge, "load_training_manifest", return_value=(payload, records)):
                    official.DreamBoothDataset = bridge.build_scientific_dataset_class(project_root=root,
                        manifest_path=manifest, seed=20260723, random_flip=False,
                        condition_interpolation="bilinear", resume_step=0)
                    dataset = official.DreamBoothDataset(instance_data_root=manifest, instance_prompt=bridge.PROMPT,
                        size=512, repeats=1, center_crop=False, buckets=[(512, 512)])
                    self.assertEqual(len(dataset), 2)
                    self.assertEqual(tuple(dataset[0]["instance_images"].shape), (3, 512, 512))
                    self.assertEqual(tuple(dataset[0]["cond_images"].shape), (3, 512, 512))
                    self.assertEqual(dataset[0]["instance_prompt"], bridge.PROMPT)
        finally:
            bridge.ROOT, bridge.MODEL_ROOT, bridge.OFFICIAL_SOURCE = prior


if __name__ == "__main__":
    unittest.main()
