"""Palette-safe paired geometry for RGB condition images and RGB masks."""

from __future__ import annotations

import math
import random
from collections.abc import Iterable
from dataclasses import asdict, dataclass

from PIL import Image, ImageOps

DEFAULT_PALETTE = ((0, 0, 0), (34, 139, 34), (255, 0, 255))


@dataclass(frozen=True)
class TransformPlan:
    resized_width: int
    resized_height: int
    crop_left: int
    crop_top: int
    output_width: int
    output_height: int
    horizontal_flip: bool

    def to_dict(self) -> dict:
        return asdict(self)


def palette_values(image: Image.Image) -> set[tuple[int, int, int]]:
    if image.mode != "RGB":
        raise ValueError("target mask must remain RGB")
    return {tuple(pixel) for pixel in image.get_flattened_data()}


def validate_palette(
    image: Image.Image,
    canonical_palette: Iterable[tuple[int, int, int]] = DEFAULT_PALETTE,
) -> set[tuple[int, int, int]]:
    observed = palette_values(image)
    allowed = {tuple(value) for value in canonical_palette}
    unexpected = observed - allowed
    if unexpected:
        raise ValueError(f"target mask contains non-canonical RGB values: {sorted(unexpected)[:8]}")
    return observed


def make_plan(
    input_size: tuple[int, int],
    output_size: tuple[int, int],
    *,
    seed: int,
    center_crop: bool = False,
    random_flip: bool = True,
) -> TransformPlan:
    width, height = input_size
    out_width, out_height = output_size
    if min(width, height, out_width, out_height) <= 0:
        raise ValueError("image and output dimensions must be positive")
    scale = max(out_width / width, out_height / height)
    resized_width = max(out_width, math.ceil(width * scale))
    resized_height = max(out_height, math.ceil(height * scale))
    max_left = resized_width - out_width
    max_top = resized_height - out_height
    rng = random.Random(seed)
    if center_crop:
        left, top = max_left // 2, max_top // 2
    else:
        left = rng.randint(0, max_left) if max_left else 0
        top = rng.randint(0, max_top) if max_top else 0
    flip = bool(random_flip and rng.random() < 0.5)
    return TransformPlan(
        resized_width,
        resized_height,
        left,
        top,
        out_width,
        out_height,
        flip,
    )


def apply_plan(
    condition: Image.Image,
    target_mask: Image.Image,
    plan: TransformPlan,
    *,
    condition_interpolation: str = "bilinear",
    canonical_palette: Iterable[tuple[int, int, int]] = DEFAULT_PALETTE,
) -> tuple[Image.Image, Image.Image]:
    if condition.mode != "RGB":
        raise ValueError("condition image must be RGB")
    if target_mask.mode != "RGB":
        raise ValueError("target mask must be RGB")
    if condition.size != target_mask.size:
        raise ValueError("paired condition and target must have identical source geometry")
    before = validate_palette(target_mask, canonical_palette)
    interpolation = {
        "bilinear": Image.Resampling.BILINEAR,
        "bicubic": Image.Resampling.BICUBIC,
    }.get(condition_interpolation)
    if interpolation is None:
        raise ValueError("condition interpolation must be bilinear or bicubic")
    resized = (plan.resized_width, plan.resized_height)
    condition_out = condition.resize(resized, resample=interpolation)
    target_out = target_mask.resize(resized, resample=Image.Resampling.NEAREST)
    box = (
        plan.crop_left,
        plan.crop_top,
        plan.crop_left + plan.output_width,
        plan.crop_top + plan.output_height,
    )
    condition_out = condition_out.crop(box)
    target_out = target_out.crop(box)
    if plan.horizontal_flip:
        condition_out = ImageOps.mirror(condition_out)
        target_out = ImageOps.mirror(target_out)
    after = validate_palette(target_out, canonical_palette)
    if not after.issubset(before):
        raise RuntimeError("palette pollution detected after nearest-neighbor target transform")
    return condition_out, target_out


def paired_transform(
    condition: Image.Image,
    target_mask: Image.Image,
    *,
    output_size: tuple[int, int],
    seed: int,
    center_crop: bool = False,
    random_flip: bool = True,
    condition_interpolation: str = "bilinear",
    canonical_palette: Iterable[tuple[int, int, int]] = DEFAULT_PALETTE,
) -> tuple[Image.Image, Image.Image, TransformPlan]:
    plan = make_plan(
        condition.size,
        output_size,
        seed=seed,
        center_crop=center_crop,
        random_flip=random_flip,
    )
    condition_out, target_out = apply_plan(
        condition,
        target_mask,
        plan,
        condition_interpolation=condition_interpolation,
        canonical_palette=canonical_palette,
    )
    return condition_out, target_out, plan
