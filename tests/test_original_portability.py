"""CPU checks using public metadata and synthetic files/arrays only."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from researchos.portable_data import COUNTS, partition_rows, prepare, taxonomy_rows
from researchos.portable_data import read_csv
from researchos.portable_phenobench import prepare as prepare_phenobench
from researchos.phenobench_labels import task_label_array
from researchos.s8r_ex01_evaluation import _confusion, _miou
from researchos.s8r_ex01_execution import FORMAL_SEEDS, h4_bootstrap


class OriginalPortabilityTests(unittest.TestCase):
    def test_public_partition_and_mapping_metadata(self):
        groups = partition_rows(ROOT / "data/cropandweed_partitions.csv")
        self.assertEqual({key: len(rows) for key, rows in groups.items()}, COUNTS)
        self.assertEqual(len(taxonomy_rows(ROOT / "data/cropandweed_class_mapping.csv")), 101)

    def test_manifest_builder_keeps_heldout_labels_separate(self):
        groups = partition_rows(ROOT / "data/cropandweed_partitions.csv")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "images").mkdir()
            (root / "labelIds/CropAndWeed").mkdir(parents=True)
            for rows in groups.values():
                for row in rows:
                    sample = row["sample_id"]
                    (root / "images" / f"{sample}.jpg").write_bytes(b"synthetic input bytes")
                    (root / "labelIds/CropAndWeed" / f"{sample}.png").write_bytes(b"synthetic label bytes")
            output = root / "manifests"
            prepare(data_root=root, image_dir=Path("images"), label_dir=Path("labelIds/CropAndWeed"),
                    partition_csv=ROOT / "data/cropandweed_partitions.csv",
                    taxonomy_csv=ROOT / "data/cropandweed_class_mapping.csv", output_dir=output)
            for split in ("replication-confirmation", "final-method-test"):
                inputs = json.loads((output / f"{split}.inputs.json").read_text())
                labels = json.loads((output / f"{split}.labels.json").read_text())
                self.assertFalse(inputs["labels_present"])
                self.assertTrue(all(not any("label" in key for key in row) for row in inputs["rows"]))
                self.assertEqual({r["sample_id"] for r in inputs["rows"]}, {r["sample_id"] for r in labels["rows"]})
                self.assertEqual(labels["label_encoding"], "source_ids")
            timing = json.loads((output / "cost40.inputs.json").read_text())
            self.assertEqual(len(timing["rows"]), 40)
            self.assertEqual(len({row["recording_session_key"] for row in timing["rows"]}), 40)
            self.assertEqual(sum(row["source_set"] == "ave" for row in timing["rows"]), 11)
            self.assertEqual(sum(row["source_set"] == "vwg" for row in timing["rows"]), 29)
            (root / "images" / f"{groups['train'][0]['sample_id']}.png").write_bytes(b"duplicate")
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                prepare(data_root=root, image_dir=Path("images"), label_dir=Path("labelIds/CropAndWeed"),
                        partition_csv=ROOT / "data/cropandweed_partitions.csv",
                        taxonomy_csv=ROOT / "data/cropandweed_class_mapping.csv", output_dir=root / "other")

    def test_synthetic_segmentation_and_pheno_mapping(self):
        target = np.array([[0, 1], [2, 255]], dtype=np.uint8)
        prediction = np.array([[0, 1], [2, 0]], dtype=np.uint8)
        np.testing.assert_array_equal(_confusion(target, prediction), np.eye(3, dtype=np.int64))
        self.assertEqual(_miou(_confusion(target, prediction)), 1.0)
        np.testing.assert_array_equal(task_label_array(np.array([[0, 1, 2, 3, 4]], dtype=np.uint8)),
                                      np.array([[0, 1, 2, 1, 2]], dtype=np.uint8))

    def test_pheno_prepare_uses_frozen_inputs_without_validation_labels(self):
        selected = read_csv(ROOT / "data/phenobench_validation_selection.csv")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in ("train/images", "train/semantics", "val/images"):
                (root / name).mkdir(parents=True)
            for index in range(1407):
                for name in ("train/images", "train/semantics"):
                    (root / name / f"synthetic_{index:04d}.png").write_bytes(b"synthetic bytes")
            for row in selected:
                (root / "val/images" / row["sample_id"]).write_bytes(b"synthetic input")
            output = root / "output"
            prepare_phenobench({"data_root": str(root),
                "validation_selection": str(ROOT / "data/phenobench_validation_selection.csv")}, output)
            inputs = json.loads((output / "validation.inputs.json").read_text())
            self.assertEqual([r["sample_id"] for r in inputs["rows"]],
                [r["sample_id"] for r in sorted(selected, key=lambda r: int(r["frozen_order_index"]))])
            self.assertFalse((root / "val/semantics").exists())
            self.assertTrue(all(not any("label" in key for key in row) for row in inputs["rows"]))

    def test_h4_paired_resampling_preserves_exact_constant_ratio(self):
        rows = []
        for source, count in (("ave", 11), ("vwg", 29)):
            for index in range(count):
                for seed in FORMAL_SEEDS:
                    for repeat in (1, 2, 3):
                        for system, multiplier in (("Fresh_S7", 2.0), ("Fresh_Arm_B", 1.0)):
                            rows.append({"source_set": source, "recording_session_key": f"{source}-{index:04d}",
                                "sample_id": f"{source}-{index:04d}-0000", "seed": seed, "repeat": repeat,
                                "system": system, "latency_ms": multiplier * (10 + index)})
        result = h4_bootstrap(rows, draws=50)
        self.assertEqual(result["ci95"], [2.0, 2.0])
        self.assertEqual(result["aggregate_geometric_mean_speedup"], 2.0)


if __name__ == "__main__":
    unittest.main()
