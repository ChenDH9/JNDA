"""Isolated post-release quality evaluator for S8R_Ex01 v1.1."""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from researchos.s8r_ex01_execution import (
    EXPECTED_CONFIRMATION_COUNT,
    FORMAL_SEEDS,
    SYSTEMS,
    TASK_ID,
    TransactionError,
    canonical_json_sha256,
    sha256_file,
)

QUALITY_BOOTSTRAP_SEED = 2026073001
QUALITY_BOOTSTRAP_DRAWS = 10000


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(
                dict(value),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _taxonomy(root: Path) -> np.ndarray:
    payload = json.loads(
        (
            root
            / "audit/S8R_Ex01_preregistration_hardening/"
            "complete_taxonomy_mapping.json"
        ).read_text(encoding="utf-8")
    )
    rows = payload.get("mapping_rows")
    if not isinstance(rows, list) or len(rows) != 101:
        raise TransactionError("quality taxonomy is not 101/101")
    lookup = np.full(256, 255, dtype=np.uint8)
    observed = set()
    for row in rows:
        source = int(row["source_class_id"])
        target = int(row["target_class_id"])
        if source in observed or target not in {0, 1, 2, 255}:
            raise TransactionError("quality taxonomy row differs")
        observed.add(source)
        lookup[source] = target
    if observed != {*range(100), 255}:
        raise TransactionError("quality taxonomy coverage differs")
    return lookup


def _confirmation_rows(root: Path) -> list[dict[str, Any]]:
    payload = json.loads(
        (
            root
            / "audit/S8R_Ex01_preregistration_preflight_v1_1/"
            "formal_confirmation_input_only_manifest.json"
        ).read_text(encoding="utf-8")
    )
    rows = payload.get("rows")
    if (
        payload.get("split") != "replication-confirmation"
        or payload.get("labels_present") is not False
        or not isinstance(rows, list)
        or len(rows) != EXPECTED_CONFIRMATION_COUNT
    ):
        raise TransactionError("confirmation input-only manifest differs")
    return [dict(row) for row in rows]


def _label_bindings(root: Path, sample_ids: set[str]) -> dict[str, dict[str, str]]:
    path = (
        root
        / "audit/S8R_Ex00_asset_split_qualification/"
        "cropandweed_file_manifest.jsonl"
    )
    bindings = {}
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            sample_id = str(row.get("sample_id", ""))
            if sample_id in sample_ids:
                bindings[sample_id] = {
                    "label_path": str(row["label_path"]),
                    "label_sha256": str(row["files"]["label"]["sha256"]),
                }
    if set(bindings) != sample_ids:
        raise TransactionError("confirmation label binding set differs")
    return bindings


def _read_label(
    root: Path,
    binding: Mapping[str, str],
    lookup: np.ndarray,
) -> np.ndarray:
    path = (root / binding["label_path"]).resolve(strict=True)
    path.relative_to(root)
    if sha256_file(path) != binding["label_sha256"]:
        raise TransactionError("confirmation label SHA256 differs")
    with Image.open(path) as image:
        raw = np.asarray(
            image.resize((512, 512), Image.Resampling.NEAREST),
            dtype=np.uint8,
        )
    return lookup[raw].copy()


def _read_prediction(root: Path, row: Mapping[str, Any]) -> np.ndarray:
    path = (root / str(row["prediction_path"])).resolve(strict=True)
    path.relative_to(root)
    if sha256_file(path) != row["prediction_sha256"]:
        raise TransactionError("prediction SHA256 differs after joint seal")
    with Image.open(path) as image:
        if image.mode not in {"L", "P"} or image.size != (512, 512):
            raise TransactionError("prediction shape/mode differs")
        value = np.asarray(image, dtype=np.uint8).copy()
    if not set(np.unique(value)) <= {0, 1, 2}:
        raise TransactionError("prediction contains invalid class IDs")
    return value


def _confusion(target: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    valid = target != 255
    encoded = target[valid].astype(np.int64) * 3 + prediction[
        valid
    ].astype(np.int64)
    return np.bincount(encoded, minlength=9).reshape(3, 3).astype(
        np.int64
    )


def _miou(confusion: np.ndarray) -> float:
    intersection = np.diag(confusion).astype(np.float64)
    union = (
        confusion.sum(axis=0)
        + confusion.sum(axis=1)
        - intersection
    )
    if np.any(union <= 0):
        raise TransactionError("quality bootstrap produced undefined IoU")
    return float(np.mean(intersection / union))


def _contrast(
    values: np.ndarray,
    *,
    left: int,
    right: int,
) -> np.ndarray:
    return np.mean(values[:, left, :] - values[:, right, :], axis=1)


def evaluate_confirmation(
    *,
    project_root: str | Path,
    inventory_rows: Sequence[Mapping[str, Any]],
    joint_seal: Mapping[str, Any],
    label_release_token: Mapping[str, Any],
    output_path: str | Path,
) -> dict[str, Any]:
    """Open exactly 844 masks once and run the frozen quality bootstrap."""

    root = Path(project_root).resolve(strict=True)
    output = Path(output_path)
    output = output if output.is_absolute() else root / output
    output = output.resolve()
    output.relative_to(root)
    if (
        joint_seal.get("status") != "JOINT_PREDICTION_SEAL_READY"
        or joint_seal.get("prediction_count") != 16880
    ):
        raise TransactionError("quality evaluator joint seal is not ready")
    if (
        label_release_token.get("task_id") != TASK_ID
        or label_release_token.get("authorized_mask_open_count")
        != EXPECTED_CONFIRMATION_COUNT
        or label_release_token.get("final_method_test_label_open_count") != 0
        or label_release_token.get("joint_inventory_sha256")
        != joint_seal.get("inventory_sha256")
    ):
        raise TransactionError("quality evaluator release token differs")
    if len(inventory_rows) != 16880:
        raise TransactionError("quality inventory count differs")
    if output.exists():
        prior = json.loads(output.read_text(encoding="utf-8"))
        if (
            prior.get("joint_inventory_sha256")
            != joint_seal["inventory_sha256"]
            or prior.get("status") != "PASS"
        ):
            raise TransactionError("existing quality output is not idempotent")
        return prior

    confirmation = _confirmation_rows(root)
    expected_by_index = {
        int(row["index"]): row for row in confirmation
    }
    sample_ids = {str(row["sample_id"]) for row in confirmation}
    bindings = _label_bindings(root, sample_ids)
    lookup = _taxonomy(root)
    # This is the only semantic-mask open in the transaction: each unique
    # confirmation mask is read once, then reused in memory for all 20 cells.
    labels = {
        sample_id: _read_label(root, bindings[sample_id], lookup)
        for sample_id in sorted(sample_ids)
    }
    if len(labels) != EXPECTED_CONFIRMATION_COUNT:
        raise TransactionError("semantic mask open count differs")

    system_index = {system: index for index, system in enumerate(SYSTEMS)}
    seed_index = {seed: index for index, seed in enumerate(FORMAL_SEEDS)}
    sessions = sorted(
        {
            str(row["recording_session_key"])
            for row in confirmation
        }
    )
    session_index = {
        session: index for index, session in enumerate(sessions)
    }
    if len(sessions) != 92:
        raise TransactionError("confirmation session count differs")
    session_source = {}
    for row in confirmation:
        session = str(row["recording_session_key"])
        source = str(row["source_set"])
        prior = session_source.setdefault(session, source)
        if prior != source:
            raise TransactionError("session appears in multiple strata")
    if {
        source: sum(value == source for value in session_source.values())
        for source in ("ave", "vwg")
    } != {"ave": 26, "vwg": 66}:
        raise TransactionError("confirmation strata are not 26/66")

    confusions = np.zeros((4, 5, 92, 3, 3), dtype=np.int64)
    observed = set()
    for row in inventory_rows:
        system = str(row["system"])
        seed = int(row["seed"])
        index = int(row["confirmation_index"])
        expected = expected_by_index.get(index)
        if (
            system not in system_index
            or seed not in seed_index
            or expected is None
            or str(row.get("sample_id")) != str(expected["sample_id"])
        ):
            raise TransactionError("quality inventory binding differs")
        key = (system, seed, index)
        if key in observed:
            raise TransactionError("quality inventory has a duplicate")
        observed.add(key)
        prediction = _read_prediction(root, row)
        sample_id = str(expected["sample_id"])
        confusions[
            system_index[system],
            seed_index[seed],
            session_index[str(expected["recording_session_key"])],
        ] += _confusion(labels[sample_id], prediction)
    if len(observed) != 16880:
        raise TransactionError("quality inventory matrix is incomplete")

    pooled = confusions.sum(axis=2)
    point = np.empty((4, 5), dtype=np.float64)
    for system in range(4):
        for seed in range(5):
            point[system, seed] = _miou(pooled[system, seed])

    rng = np.random.Generator(np.random.PCG64(QUALITY_BOOTSTRAP_SEED))
    draws = np.empty((QUALITY_BOOTSTRAP_DRAWS, 4, 5), dtype=np.float64)
    strata_indices = {
        source: np.asarray(
            [
                session_index[session]
                for session in sessions
                if session_source[session] == source
            ],
            dtype=np.int64,
        )
        for source in ("ave", "vwg")
    }
    for draw in range(QUALITY_BOOTSTRAP_DRAWS):
        sampled = np.concatenate(
            [
                indices[
                    rng.integers(0, len(indices), size=len(indices))
                ]
                for indices in (
                    strata_indices["ave"],
                    strata_indices["vwg"],
                )
            ]
        )
        counts = np.bincount(sampled, minlength=92)
        draw_confusions = np.einsum(
            "n,abnij->abij", counts, confusions, optimize=True
        )
        for system in range(4):
            for seed in range(5):
                draws[draw, system, seed] = _miou(
                    draw_confusions[system, seed]
                )

    indices = system_index
    contrasts = {
        "H1": {
            "left": "Fresh_Arm_B",
            "right": "Fresh_SegFormer_B2",
            "margin": 0.0,
        },
        "H2": {
            "left": "Fresh_Arm_B",
            "right": "Fresh_S7",
            "margin": -0.005,
        },
        "H3": {
            "left": "Fresh_Arm_B",
            "right": "Fresh_Arm_A",
            "margin": 0.0,
        },
    }
    hypothesis = {}
    for name, specification in contrasts.items():
        left = indices[specification["left"]]
        right = indices[specification["right"]]
        seed_points = point[left] - point[right]
        draw_values = _contrast(draws, left=left, right=right)
        lower, upper = np.quantile(
            draw_values, [0.025, 0.975], method="linear"
        )
        passed = float(lower) > float(specification["margin"])
        if name == "H1":
            passed = passed and bool(np.all(seed_points > 0))
        hypothesis[name] = {
            **specification,
            "per_seed_point_differences": {
                str(seed): float(seed_points[index])
                for index, seed in enumerate(FORMAL_SEEDS)
            },
            "mean_point_difference": float(np.mean(seed_points)),
            "ci95": [float(lower), float(upper)],
            "draw_values_sha256": hashlib.sha256(
                draw_values.astype("<f8", copy=False).tobytes()
            ).hexdigest(),
            "status": "PASS" if passed else "FAIL",
        }

    result = {
        "schema_version": 1,
        "task_id": TASK_ID,
        "status": "PASS",
        "joint_inventory_sha256": joint_seal["inventory_sha256"],
        "semantic_label_open_count": EXPECTED_CONFIRMATION_COUNT,
        "replication_confirmation_label_open_count": (
            EXPECTED_CONFIRMATION_COUNT
        ),
        "final_method_test_label_open_count": 0,
        "point_miou": {
            system: {
                str(seed): float(point[system_index[system], seed_index[seed]])
                for seed in FORMAL_SEEDS
            }
            for system in SYSTEMS
        },
        "quality_bootstrap": {
            "method": (
                "paired source-stratified recording-session cluster bootstrap"
            ),
            "draws": QUALITY_BOOTSTRAP_DRAWS,
            "bit_generator": "PCG64",
            "seed": QUALITY_BOOTSTRAP_SEED,
            "quantile_method": "linear",
            "draw_matrix_sha256": hashlib.sha256(
                draws.astype("<f8", copy=False).tobytes()
            ).hexdigest(),
        },
        "hypotheses": hypothesis,
        "inventory_rows_sha256": canonical_json_sha256(
            list(inventory_rows)
        ),
    }
    _atomic_json(output, result)
    return result
