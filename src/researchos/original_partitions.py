"""Source-stratified session allocation and deterministic sample ordering."""
from __future__ import annotations
import hashlib
from collections.abc import Iterable

SPLIT_SALT = "S8R_CROPANDWEED_SPLIT_V1_20260730"


SPLIT_NAMES = (
    "train",
    "development",
    "replication-confirmation",
    "final-method-test",
)


SPLIT_RATIOS = (0.60, 0.15, 0.10, 0.15)


def sha256_lines(values: Iterable[str]) -> str:
    return sha256_bytes("".join(f"{value}\n" for value in values).encode("utf-8"))


def allocate_quotas(count: int) -> list[int]:
    raw = [count * ratio for ratio in SPLIT_RATIOS]
    quota = [int(value) for value in raw]
    remainder = count - sum(quota)
    order = sorted(
        range(len(raw)),
        key=lambda index: (-(raw[index] - quota[index]), index),
    )
    for index in order[:remainder]:
        quota[index] += 1
    return quota


def session_sort_key(source_set: str, session_id: str) -> tuple[str, str]:
    payload = f"{SPLIT_SALT}\0{source_set}\0{session_id}".encode()
    return hashlib.sha256(payload).hexdigest(), session_id


def sample_sort_key(split: str, sample_id: str) -> tuple[str, str]:
    payload = f"{SPLIT_SALT}\0{split}\0{sample_id}".encode()
    return hashlib.sha256(payload).hexdigest(), sample_id

from collections import defaultdict
from typing import Any

TASK_ID = "S8R_Ex01_preregistration_hardening_v1_0"


COST_SALT = "S8R_EX01_REPLICATION_COST40_V1_20260730"


def build_cost_manifest(
    split: dict[str, Any],
    file_manifest: list[dict[str, Any]],
) -> dict[str, Any]:
    replication_sessions = split["partitions"]["replication-confirmation"][
        "session_keys"
    ]
    by_source: dict[str, list[str]] = defaultdict(list)
    for session_key in replication_sessions:
        source = session_key.split("-", 1)[0]
        by_source[source].append(session_key)
    quotas = {"ave": 11, "vwg": 29}
    selected_sessions: list[str] = []
    for source, quota in quotas.items():
        ordered = sorted(
            by_source[source],
            key=lambda value: (
                hashlib.sha256(f"{COST_SALT}\0{value}".encode()).hexdigest(),
                value,
            ),
        )
        selected_sessions.extend(ordered[:quota])
    by_session: dict[str, list[dict[str, Any]]] = defaultdict(list)
    replication_ids = set(
        split["partitions"]["replication-confirmation"]["sample_order"]
    )
    for row in file_manifest:
        if row["sample_id"] in replication_ids:
            by_session[row["recording_session_key"]].append(row)
    selected_rows = []
    for session_key in selected_sessions:
        candidates = sorted(
            by_session[session_key],
            key=lambda row: (
                hashlib.sha256(
                    f"{COST_SALT}\0{row['sample_id']}".encode()
                ).hexdigest(),
                row["sample_id"],
            ),
        )
        if not candidates:
            raise RuntimeError(f"no sample for timing session {session_key}")
        row = candidates[0]
        selected_rows.append(
            {
                "sample_id": row["sample_id"],
                "source_set": row["source_set"],
                "recording_session_key": row["recording_session_key"],
                "image_path": row["image_path"],
                "image_sha256": row["files"]["image"]["sha256"],
            }
        )
    selected_rows.sort(key=lambda row: row["recording_session_key"])
    if len(selected_rows) != 40:
        raise RuntimeError("cost manifest must have exactly 40 rows")
    return {
        "schema_version": 1,
        "task_id": TASK_ID,
        "status": "FROZEN_INPUT_ONLY",
        "selection_salt": COST_SALT,
        "boundary": "replication-confirmation input only",
        "selection_rule": (
            "stratified deterministic hash rank of sessions, then deterministic "
            "hash rank of one sample within each selected session"
        ),
        "source_session_quotas": quotas,
        "unique_session_count": 40,
        "unique_sample_count": 40,
        "sample_order_sha256": sha256_lines(
            row["sample_id"] for row in selected_rows
        ),
        "contains_label_paths": False,
        "semantic_label_open_count": 0,
        "rows": selected_rows,
    }


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

