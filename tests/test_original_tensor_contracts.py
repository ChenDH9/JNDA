"""CPU-only synthetic tensor and checkpoint checks; no pretrained models."""
import os
from pathlib import Path
import sys
import tempfile
import unittest

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch
from researchos import s8_native_training as native
from researchos.s8r_ex01_training import _head_logits, _native_head


class OriginalTensorTests(unittest.TestCase):
    def test_head_unpatchifies_pixel_and_class_order(self):
        head = _native_head(1701, device="cpu")
        with torch.no_grad():
            head.weight.zero_()
            head.bias.copy_(torch.arange(768, dtype=torch.float32))
        logits = _head_logits(head, torch.zeros(1, 1024, 128))
        self.assertEqual(tuple(logits.shape), (1, 3, 512, 512))
        for y, x, channel in ((0, 0, 0), (7, 11, 2), (16, 16, 1), (511, 511, 2)):
            self.assertEqual(logits[0, channel, y, x].item(), ((y % 16) * 16 + x % 16) * 3 + channel)

    def test_native_field_coordinates_and_fixed_timestep(self):
        observed = {}

        def transformer(**kwargs):
            observed.update(kwargs)
            return (kwargs["hidden_states"],)

        tokens = torch.zeros(2, 1024, 128, dtype=torch.bfloat16)
        result = native.native_transformer_forward(transformer, tokens,
            prompt_embeds=torch.zeros(1, 512, 8, dtype=torch.bfloat16),
            txt_ids=torch.zeros(1, 512, 4, dtype=torch.int64))
        self.assertIs(result, tokens)
        self.assertEqual(tuple(observed["img_ids"].shape), (2, 1024, 4))
        self.assertEqual(observed["img_ids"][0, 33].tolist(), [0, 1, 1, 0])
        self.assertTrue(torch.allclose(observed["timestep"], torch.full((2,), 0.999)))
        self.assertEqual(observed["guidance"].tolist(), [0.0, 0.0])
        self.assertEqual(tuple(observed["encoder_hidden_states"].shape), (2, 512, 8))

    def test_native_checkpoint_preserves_head_and_optimizer_contract(self):
        head = native.create_head(20260723, device="cpu")
        optimizer, scheduler, _ = native.build_optimizer(None, head, "A")
        sample_ids = [f"synthetic_{index}" for index in range(1407)]
        order_sha = native.sample_order_sha256(sample_ids, 20260723)
        binding = {"train_count": 1407, "arm": "A", "seed": 20260723}
        original = {name: value.detach().clone() for name, value in head.named_parameters()}
        with tempfile.TemporaryDirectory() as directory:
            native.save_checkpoint(directory, step=7035, arm="A", seed=20260723,
                transformer=None, head=head, optimizer=optimizer, scheduler=scheduler,
                binding=binding, producer_context={}, sample_order_sha256_value=order_sha)
            checkpoint = Path(directory) / "checkpoint-7035"
            native.verify_checkpoint_receipt(checkpoint, expected_arm="A", expected_seed=20260723,
                expected_step=7035, expected_binding=binding, expected_sample_order_sha256=order_sha)
            with torch.no_grad():
                head.weight.zero_()
                head.bias.zero_()
            native.load_trainable_safetensors(checkpoint / "trainable.safetensors", None, head, "A")
            for name, value in head.named_parameters():
                self.assertTrue(torch.equal(value, original[name]))


if __name__ == "__main__":
    unittest.main()
