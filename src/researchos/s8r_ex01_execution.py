"""Deterministic data boundaries, RNG state, and paired statistics."""
from __future__ import annotations
import hashlib
import json
import math
import os
import random
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import numpy as np

TASK_ID = "S8R_Ex01_cleanroom_joint_native_replication_v1_1"


SYSTEMS = (
    "Fresh_S7",
    "Fresh_Arm_A",
    "Fresh_Arm_B",
    "Fresh_SegFormer_B2",
)


FORMAL_SEEDS = (1701, 1702, 1703, 1704, 1705)


ENGINEERING_SMOKE_SEED = 2026073011


TRAIN_SPLIT = "train"


DEVELOPMENT_SPLIT = "development"


REPLICATION_SPLIT = "replication-confirmation"


FINAL_METHOD_TEST_SPLIT = "final-method-test"


FORBIDDEN_OPTIMIZER_SPLITS = frozenset(
    {DEVELOPMENT_SPLIT, REPLICATION_SPLIT, FINAL_METHOD_TEST_SPLIT}
)


EXPECTED_CONFIRMATION_COUNT = 844


EXPECTED_FORMAL_CELLS = 20


EXPECTED_JOINT_PREDICTIONS = 16880


H4_BOOTSTRAP_SEED = 2026073002


H4_BOOTSTRAP_DRAWS = 10000


class S8RContractError(RuntimeError):
    """A fail-closed protocol or implementation-contract violation."""


class SplitAccessError(S8RContractError):
    """A split was used for a forbidden operation."""


class TransactionError(S8RContractError):
    """The immutable confirmation transaction is incomplete or consumed."""


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def cell_id(system: str, seed: int) -> str:
    if system not in SYSTEMS or seed not in FORMAL_SEEDS:
        raise S8RContractError("unknown formal system/seed")
    return f"{system}__seed_{seed}"


@dataclass
class SplitAccessGuard:
    """Count operations and fail before forward on optimizer contamination."""

    forward_count: int = 0
    backward_count: int = 0
    optimizer_step_count: int = 0
    development_forward_count: int = 0
    development_backward_count: int = 0
    development_optimizer_step_count: int = 0

    @staticmethod
    def _splits(batch_records: Sequence[Mapping[str, Any]]) -> set[str]:
        if not batch_records:
            raise SplitAccessError("empty batch is forbidden")
        splits = {str(row.get("split", "")) for row in batch_records}
        if "" in splits:
            raise SplitAccessError("batch record has no split")
        return splits

    def before_optimizer_forward(
        self, batch_records: Sequence[Mapping[str, Any]]
    ) -> None:
        splits = self._splits(batch_records)
        if splits != {TRAIN_SPLIT}:
            raise SplitAccessError(
                f"optimizer batch must be train-only before forward: {splits}"
            )
        self.forward_count += 1

    def record_backward(
        self, batch_records: Sequence[Mapping[str, Any]]
    ) -> None:
        splits = self._splits(batch_records)
        if splits != {TRAIN_SPLIT}:
            if DEVELOPMENT_SPLIT in splits:
                self.development_backward_count += 1
            raise SplitAccessError("backward is forbidden outside train")
        self.backward_count += 1

    def record_optimizer_step(
        self, batch_records: Sequence[Mapping[str, Any]]
    ) -> None:
        splits = self._splits(batch_records)
        if splits != {TRAIN_SPLIT}:
            if DEVELOPMENT_SPLIT in splits:
                self.development_optimizer_step_count += 1
            raise SplitAccessError("optimizer step is forbidden outside train")
        self.optimizer_step_count += 1

    def before_development_forward(
        self, batch_records: Sequence[Mapping[str, Any]]
    ) -> None:
        splits = self._splits(batch_records)
        if splits != {DEVELOPMENT_SPLIT}:
            raise SplitAccessError("development runner accepts development only")
        self.forward_count += 1
        self.development_forward_count += 1

    def receipt(self) -> dict[str, int]:
        return {
            "forward_count": self.forward_count,
            "backward_count": self.backward_count,
            "optimizer_step_count": self.optimizer_step_count,
            "development_forward_count": self.development_forward_count,
            "development_backward_count": self.development_backward_count,
            "development_optimizer_step_count": (
                self.development_optimizer_step_count
            ),
        }


def run_development_forward_only(
    model: Any,
    batches: Iterable[
        tuple[Sequence[Mapping[str, Any]], Mapping[str, Any]]
    ],
    *,
    guard: SplitAccessGuard,
    forward: Callable[[Any, Mapping[str, Any]], Any],
) -> list[Any]:
    """Run structural development QA under inference mode only."""

    import torch

    outputs: list[Any] = []
    model_training_before = bool(getattr(model, "training", False))
    if hasattr(model, "eval"):
        model.eval()
    try:
        with torch.inference_mode():
            for records, payload in batches:
                guard.before_development_forward(records)
                result = forward(model, payload)
                if torch.is_grad_enabled():
                    raise SplitAccessError(
                        "gradient mode enabled inside development runner"
                    )
                outputs.append(result)
    finally:
        if model_training_before and hasattr(model, "train"):
            model.train()
    if (
        guard.development_backward_count
        or guard.development_optimizer_step_count
    ):
        raise SplitAccessError("development mutation counters are nonzero")
    return outputs


def seed_all(seed: int) -> dict[str, Any]:
    """Bind every process-global RNG and return a DataLoader generator."""

    import torch

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    generator = torch.Generator()
    generator.manual_seed(seed)
    return {"seed": seed, "dataloader_generator": generator}


def dataloader_worker_init(worker_id: int, *, base_seed: int) -> None:
    worker_seed = (base_seed + worker_id) % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    try:
        import torch

        torch.manual_seed(worker_seed)
    except ImportError:
        pass


def capture_rng_state(generator: Any) -> dict[str, Any]:
    import torch

    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
        ),
        "dataloader_generator": generator.get_state(),
    }


def restore_rng_state(state: Mapping[str, Any], generator: Any) -> None:
    import torch

    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])
    generator.set_state(state["dataloader_generator"])


def stable_sample_order(
    sample_ids: Sequence[str], *, seed: int, epoch: int
) -> list[str]:
    if len(sample_ids) != len(set(sample_ids)) or not sample_ids:
        raise S8RContractError("sample IDs must be non-empty and unique")
    return sorted(
        sample_ids,
        key=lambda sample_id: hashlib.sha256(
            f"{TASK_ID}\0{seed}\0{epoch}\0{sample_id}".encode()
        ).hexdigest(),
    )


def _median(values: Sequence[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    if not len(array) or not np.all(np.isfinite(array)) or np.any(array <= 0):
        raise S8RContractError("latencies must be finite and positive")
    return float(np.median(array))


def _geometric_mean(values: Sequence[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    if not len(array) or not np.all(np.isfinite(array)) or np.any(array <= 0):
        raise S8RContractError(
            "geometric-mean inputs must be finite and positive"
        )
    return float(np.exp(np.mean(np.log(array))))


def _validated_latency_map(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[
    dict[tuple[str, int, str], float],
    dict[str, list[str]],
]:
    grouped: dict[tuple[str, int, str], list[float]] = defaultdict(list)
    session_source: dict[str, str] = {}
    for row in rows:
        system = str(row.get("system"))
        seed = int(row.get("seed", -1))
        session = str(row.get("recording_session_key"))
        source = str(row.get("source_set"))
        latency = float(row.get("latency_ms", math.nan))
        repeat = int(row.get("repeat", -1))
        if system not in {"Fresh_S7", "Fresh_Arm_B"}:
            raise S8RContractError("H4 contains an unregistered system")
        if seed not in FORMAL_SEEDS or not session:
            raise S8RContractError("H4 contains an invalid seed/session")
        if source not in {"ave", "vwg"}:
            raise S8RContractError("H4 contains an invalid stratum")
        if repeat not in {1, 2, 3}:
            raise S8RContractError("H4 repeats must be exactly 1,2,3")
        if not math.isfinite(latency) or latency <= 0:
            raise S8RContractError("H4 latency is NaN/Inf/nonpositive")
        prior = session_source.setdefault(session, source)
        if prior != source:
            raise S8RContractError("session belongs to multiple strata")
        grouped[(system, seed, session)].append(latency)
    strata = {
        source: sorted(
            session for session, observed in session_source.items()
            if observed == source
        )
        for source in ("ave", "vwg")
    }
    if {key: len(value) for key, value in strata.items()} != {
        "ave": 11,
        "vwg": 29,
    }:
        raise S8RContractError("H4 cost40 stratum counts are not 11/29")
    expected_keys = {
        (system, seed, session)
        for system in ("Fresh_S7", "Fresh_Arm_B")
        for seed in FORMAL_SEEDS
        for session in (*strata["ave"], *strata["vwg"])
    }
    if set(grouped) != expected_keys:
        raise S8RContractError("H4 timing matrix is incomplete or has extras")
    if any(len(values) != 3 for values in grouped.values()):
        raise S8RContractError("H4 requires exactly three repeats per cell")
    collapsed = {key: _median(value) for key, value in grouped.items()}
    return collapsed, strata


def h4_point_estimate(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    collapsed, strata = _validated_latency_map(rows)
    sessions = (*strata["ave"], *strata["vwg"])
    per_seed: dict[str, float] = {}
    for seed in FORMAL_SEEDS:
        s7 = _median(
            [collapsed[("Fresh_S7", seed, session)] for session in sessions]
        )
        arm_b = _median(
            [collapsed[("Fresh_Arm_B", seed, session)] for session in sessions]
        )
        per_seed[str(seed)] = s7 / arm_b
    aggregate = _geometric_mean(list(per_seed.values()))
    return {
        "per_seed_speedup": per_seed,
        "aggregate_geometric_mean_speedup": aggregate,
        "all_seed_points_at_least_1_5": all(
            value >= 1.5 for value in per_seed.values()
        ),
    }


def h4_bootstrap(
    rows: Sequence[Mapping[str, Any]],
    *,
    draws: int = H4_BOOTSTRAP_DRAWS,
    seed: int = H4_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Regenerate the paired source-stratified H4 bootstrap exactly."""

    if draws <= 0:
        raise S8RContractError("bootstrap draws must be positive")
    collapsed, strata = _validated_latency_map(rows)
    rng = np.random.Generator(np.random.PCG64(seed))
    draw_values = np.empty(draws, dtype=np.float64)
    for draw_index in range(draws):
        sampled = []
        for source in ("ave", "vwg"):
            sessions = np.asarray(strata[source], dtype=object)
            indices = rng.integers(0, len(sessions), size=len(sessions))
            sampled.extend(str(value) for value in sessions[indices])
        ratios = []
        for formal_seed in FORMAL_SEEDS:
            s7 = _median(
                [
                    collapsed[("Fresh_S7", formal_seed, session)]
                    for session in sampled
                ]
            )
            arm_b = _median(
                [
                    collapsed[("Fresh_Arm_B", formal_seed, session)]
                    for session in sampled
                ]
            )
            ratios.append(s7 / arm_b)
        draw_values[draw_index] = _geometric_mean(ratios)
    lower, upper = np.quantile(
        draw_values, [0.025, 0.975], method="linear"
    )
    point = h4_point_estimate(rows)
    return {
        **point,
        "method": "paired source-stratified recording-session bootstrap",
        "draws": draws,
        "bit_generator": "PCG64",
        "seed": seed,
        "ci95": [float(lower), float(upper)],
        "draw_values_sha256": hashlib.sha256(
            draw_values.astype("<f8", copy=False).tobytes()
        ).hexdigest(),
        "H4_STATUS": (
            "PASS"
            if float(lower) >= 1.5
            and point["all_seed_points_at_least_1_5"]
            else "FAIL"
        ),
    }


def expected_prediction_keys() -> set[tuple[str, int, str]]:
    return {
        (system, seed, str(index))
        for system in SYSTEMS
        for seed in FORMAL_SEEDS
        for index in range(EXPECTED_CONFIRMATION_COUNT)
    }


def validate_joint_prediction_inventory(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    keys: set[tuple[str, int, str]] = set()
    for row in rows:
        system = str(row.get("system"))
        seed = int(row.get("seed", -1))
        confirmation_index = str(row.get("confirmation_index"))
        path = Path(str(row.get("prediction_path", "")))
        if system not in SYSTEMS or seed not in FORMAL_SEEDS:
            raise TransactionError("joint inventory has unknown system/seed")
        if not confirmation_index.isdigit():
            raise TransactionError("joint inventory has invalid confirmation index")
        if path.suffix.lower() != ".png":
            raise TransactionError("prediction must be a PNG")
        if not isinstance(row.get("prediction_sha256"), str) or len(
            row["prediction_sha256"]
        ) != 64:
            raise TransactionError("prediction SHA256 is absent")
        key = (system, seed, confirmation_index)
        if key in keys:
            raise TransactionError("duplicate joint prediction")
        keys.add(key)
    expected = expected_prediction_keys()
    if keys != expected or len(rows) != EXPECTED_JOINT_PREDICTIONS:
        raise TransactionError(
            "joint inventory is not the exact 20 x 844 transaction"
        )
    return {
        "system_seed_cells": EXPECTED_FORMAL_CELLS,
        "prediction_count": len(rows),
        "inventory_sha256": canonical_json_sha256(list(rows)),
        "replication_label_open_count": 0,
        "final_method_test_label_open_count": 0,
        "status": "JOINT_PREDICTION_SEAL_READY",
    }


def hypothesis_statuses(
    *,
    h1_pass: bool,
    h2_pass: bool,
    h3_pass: bool,
    h4_pass: bool,
    engineering_pass: bool,
) -> dict[str, Any]:
    statuses = {
        "H1_STATUS": "PASS" if h1_pass else "FAIL",
        "H2_STATUS": "PASS" if h2_pass else "FAIL",
        "H3_STATUS": "PASS" if h3_pass else "FAIL",
        "H4_STATUS": "PASS" if h4_pass else "FAIL",
        "ENGINEERING_STATUS": "PASS" if engineering_pass else "FAIL",
    }
    global_pass = (
        engineering_pass and h1_pass and h2_pass and h3_pass and h4_pass
    )
    statuses["GLOBAL_CONFIRMATION_STATUS"] = (
        "PASS" if global_pass else "FAIL"
    )
    statuses["S10_UNLOCK_STATUS"] = (
        "ELIGIBLE_FOR_SEPARATE_USER_DECISION"
        if global_pass
        else "NOT_AUTHORIZED"
    )
    statuses["H1_CLAIM_RETAINED"] = h1_pass
    statuses["H1_CLAIM_BOUNDARY"] = (
        "Fresh Arm B versus Fresh SegFormer-B2 clean-room independent "
        "replication under the registered equal-10-pass protocol"
        if h1_pass
        else None
    )
    return statuses
