"""Original PhenoBench S7 dataset, sampler, and official-trainer integration."""
from __future__ import annotations
import hashlib
import importlib.util
import json
import math
from pathlib import Path
from typing import Any
from PIL import Image
from researchos.flux2_klein_i2i.real_trainer import lora_target_modules
from researchos.flux2_klein_i2i.paired_transforms import paired_transform
from researchos.portable_phenobench_s7 import load_manifest as load_training_manifest
ROOT = Path.cwd()
MODEL_ROOT = Path("models/FLUX.2-klein-base-4B")
REVISION = "8c44a2fbef88fae175da65df054db0f901aa9747"
OFFICIAL_SOURCE = Path("third_party/diffusers_examples/train_dreambooth_lora_flux2_klein_img2img.py")
OFFICIAL_SOURCE_SHA256 = "6486a9d7b56264a047339c3e8291d0d55cae51266bb2c3c5d8d533872330e31e"
PROMPT = "Create a spatially aligned three-class semantic segmentation mask of the input field image: background black, crop plants green, weeds magenta. Flat solid colors only; no text, legend, shading, outlines, or photograph."

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_official_module():
    if sha256_file(OFFICIAL_SOURCE) != OFFICIAL_SOURCE_SHA256:
        raise RuntimeError("pinned official Klein img2img trainer SHA256 mismatch")
    spec = importlib.util.spec_from_file_location(
        "researchos_pinned_flux2_klein_img2img", OFFICIAL_SOURCE
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load pinned official Klein img2img trainer")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _pil_to_normalized_tensor(image: Image.Image):
    import numpy as np
    import torch

    array = np.asarray(image, dtype=np.uint8).copy()
    return torch.from_numpy(array).permute(2, 0, 1).float().div_(127.5).sub_(1.0)


def build_deterministic_batch_sampler_class(*, resume_step: int, target_step: int):
    from torch.utils.data.sampler import BatchSampler

    class DeterministicBucketBatchSampler(BatchSampler):
        training_started = False
        training_iteration = 0
        iteration_records: list[dict[str, int | str]] = []

        def __init__(
            self,
            dataset,
            batch_size: int,
            drop_last: bool = False,
            shuffle_batches_each_epoch: bool = True,
        ):
            if batch_size != 1 or not drop_last:
                raise ValueError("Gate sampler requires batch_size=1 and drop_last=true")
            if shuffle_batches_each_epoch:
                raise ValueError("cached Gate sampler must remain deterministic")
            self.dataset = dataset
            self.batch_size = batch_size
            self.drop_last = drop_last
            self.batches = [[index] for index in range(len(dataset))]

        def __iter__(self):
            phase = "training" if type(self).training_started else "cache"
            if phase == "training":
                epoch_index = type(self).training_iteration
                type(self).training_iteration += 1
                remaining = target_step - resume_step - epoch_index * len(self.batches)
                limit = min(len(self.batches), max(remaining, 0))
            else:
                epoch_index = 0
                limit = len(self.batches)
            type(self).iteration_records.append(
                {"phase": phase, "epoch_index": epoch_index, "yield_count": limit}
            )
            yield from self.batches[:limit]

        def __len__(self):
            return len(self.batches)

    return DeterministicBucketBatchSampler


def build_official_arguments(
    *,
    config: dict[str, Any],
    output_dir: Path,
    max_steps: int,
    resume_from_checkpoint: str | None,
    checkpointing_steps: int,
) -> list[str]:
    training = config["training"]
    manifest = ROOT / config["dataset_manifest"]
    transformer_config = json.loads((MODEL_ROOT / "transformer/config.json").read_text())
    arguments = [
        "--pretrained_model_name_or_path",
        str(MODEL_ROOT),
        "--revision",
        REVISION,
        "--instance_data_dir",
        str(manifest),
        "--image_column",
        "target_image",
        "--cond_image_column",
        "condition_image",
        "--caption_column",
        "instruction",
        "--instance_prompt",
        config["runtime"]["instance_prompt"],
        "--output_dir",
        str(output_dir),
        "--resolution",
        str(training["resolution"]),
        "--rank",
        str(training["rank"]),
        "--lora_alpha",
        str(training["lora_alpha"]),
        "--lora_dropout",
        str(training["lora_dropout"]),
        "--train_batch_size",
        str(training["train_batch_size"]),
        "--gradient_accumulation_steps",
        str(training["gradient_accumulation_steps"]),
        "--learning_rate",
        str(training["learning_rate"]),
        "--lr_scheduler",
        str(training["lr_scheduler"]),
        "--lr_warmup_steps",
        str(training["lr_warmup_steps"]),
        "--max_grad_norm",
        str(training["max_grad_norm"]),
        "--max_sequence_length",
        str(training["max_sequence_length"]),
        "--weighting_scheme",
        str(training["weighting_scheme"]),
        "--guidance_scale",
        str(training["guidance_scale"]),
        "--max_train_steps",
        str(max_steps),
        "--checkpointing_steps",
        str(checkpointing_steps),
        "--checkpoints_total_limit",
        "2",
        "--seed",
        str(training["seed"]),
        "--mixed_precision",
        "bf16",
        "--optimizer",
        "adamw",
        "--dataloader_num_workers",
        "0",
        "--lora_layers",
        ",".join(lora_target_modules(int(transformer_config["num_single_layers"]))),
        "--gradient_checkpointing",
        "--cache_latents",
        "--offload",
        "--skip_final_inference",
    ]
    if training["center_crop"]:
        arguments.append("--center_crop")
    if training["random_flip"]:
        arguments.append("--random_flip")
    if resume_from_checkpoint is not None:
        arguments.extend(["--resume_from_checkpoint", resume_from_checkpoint])
    return arguments


def _checkpoint_inventory(path: Path) -> list[dict[str, Any]]:
    if not path.is_dir():
        return []
    return [
        {
            "path": str(item.relative_to(path)),
            "size_bytes": item.stat().st_size,
            "sha256": sha256_file(item),
        }
        for item in sorted(path.rglob("*"))
        if item.is_file()
    ]


def _install_metrics_accelerator(official, metrics_path: Path, sampler_class) -> None:
    """Capture the official loop's per-step loss/LR without changing its optimizer body."""
    original_accelerator = official.Accelerator
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text("", encoding="utf-8")

    class EvidenceAccelerator(original_accelerator):
        def prepare(self, *args, **kwargs):
            prepared = super().prepare(*args, **kwargs)
            sampler_class.training_started = True
            return prepared

        def log(self, values, step=None, **kwargs):
            if self.is_main_process:
                record: dict[str, int | float] = {"step": int(step)}
                for name, raw_value in values.items():
                    value = raw_value.item() if hasattr(raw_value, "item") else raw_value
                    if isinstance(value, (int, float)):
                        numeric = float(value)
                        if not math.isfinite(numeric):
                            raise FloatingPointError(f"non-finite training metric: {name}")
                        record[str(name)] = numeric
                with metrics_path.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(record, sort_keys=True) + "\n")
            return super().log(values, step=step, **kwargs)

    official.Accelerator = EvidenceAccelerator


def _metric_summary(metrics_path: Path, *, start_step: int, end_step: int) -> dict[str, Any]:
    rows = [
        json.loads(line)
        for line in metrics_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    expected_steps = list(range(start_step + 1, end_step + 1))
    actual_steps = [int(row["step"]) for row in rows]
    if actual_steps != expected_steps:
        raise RuntimeError(
            f"metric step sequence mismatch: expected {expected_steps[:1]}..{expected_steps[-1:]}, "
            f"observed {actual_steps[:1]}..{actual_steps[-1:]} ({len(actual_steps)} rows)"
        )
    if any("loss" not in row or "lr" not in row for row in rows):
        raise RuntimeError("each optimizer step must record finite loss and learning rate")
    losses = [float(row["loss"]) for row in rows]
    learning_rates = [float(row["lr"]) for row in rows]
    if not all(math.isfinite(value) for value in losses + learning_rates):
        raise FloatingPointError("training metrics contain non-finite values")
    return {
        "path": str(metrics_path),
        "row_count": len(rows),
        "first_step": actual_steps[0],
        "last_step": actual_steps[-1],
        "loss_first": losses[0],
        "loss_last": losses[-1],
        "loss_min": min(losses),
        "loss_max": max(losses),
        "learning_rate_first": learning_rates[0],
        "learning_rate_last": learning_rates[-1],
        "all_finite": True,
    }


def _stable_seed_order(records: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    return sorted(
        records,
        key=lambda row: hashlib.sha256(f"{seed}\0{row['sample_id']}".encode()).hexdigest(),
    )


def build_scientific_dataset_class(
    *,
    project_root: Path,
    manifest_path: Path,
    seed: int,
    random_flip: bool,
    condition_interpolation: str,
    resume_step: int,
):
    """Bind the official trainer to one hashed S7 manifest and deterministic cursor."""

    from torch.utils.data import Dataset

    class ScientificPairedDataset(Dataset):
        def __init__(
            self,
            instance_data_root,
            instance_prompt,
            size=512,
            repeats=1,
            center_crop=False,
            buckets=None,
        ):
            if Path(instance_data_root).resolve() != manifest_path.resolve():
                raise ValueError("official trainer received an unexpected S7 manifest")
            if repeats != 1:
                raise ValueError("S7 manifest records may not be implicitly repeated")
            payload, raw_records = load_training_manifest(
                manifest_path,
                project_root=project_root,
                validate_target_hashes=True,
            )
            if payload.get("final_test") is not False or payload.get("split") != "train":
                raise ValueError("S7 manifest must be train-only with final_test=false")
            if instance_prompt != PROMPT or payload.get("instruction") != PROMPT:
                raise ValueError("official instance prompt differs from frozen S7 prompt")
            if buckets != [(size, size)] or size != 512:
                raise ValueError("S7 training is frozen to one 512x512 bucket")
            records = _stable_seed_order(raw_records, seed)
            offset = resume_step % len(records)
            records = records[offset:] + records[:offset]
            palette = tuple(tuple(int(channel) for channel in row) for row in payload["canonical_palette"])
            self.pixel_values = []
            self.cond_pixel_values = []
            self.sample_ids = []
            for record in records:
                with Image.open(project_root / record["condition_image"]) as image:
                    condition = image.convert("RGB")
                with Image.open(project_root / record["target_image"]) as image:
                    target = image.convert("RGB")
                condition, target, _plan = paired_transform(
                    condition,
                    target,
                    output_size=(size, size),
                    seed=seed,
                    center_crop=center_crop,
                    random_flip=random_flip,
                    condition_interpolation=condition_interpolation,
                    canonical_palette=palette,
                )
                self.pixel_values.append((_pil_to_normalized_tensor(target), 0))
                self.cond_pixel_values.append((_pil_to_normalized_tensor(condition), 0))
                self.sample_ids.append(str(record["sample_id"]))
            self.instance_prompt = instance_prompt
            self.custom_instance_prompts = None
            self.buckets = list(buckets)
            self.num_instance_images = len(records)
            self._length = len(records)
            self.resume_offset = offset
            self.manifest_sha256 = payload["manifest_sha256"]

        def __len__(self):
            return self._length

        def __getitem__(self, index):
            target, bucket_idx = self.pixel_values[index % self.num_instance_images]
            condition, _ = self.cond_pixel_values[index % self.num_instance_images]
            return {
                "instance_images": target,
                "cond_images": condition,
                "instance_prompt": self.instance_prompt,
                "bucket_idx": bucket_idx,
                "sample_id": self.sample_ids[index % self.num_instance_images],
            }

    return ScientificPairedDataset
