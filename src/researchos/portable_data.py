"""Build portable manifests from the released CropAndWeed partition tables."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

COUNTS = {"train": 4821, "development": 1206,
          "replication-confirmation": 844, "final-method-test": 1163}
SESSIONS = {"train": 557, "development": 140,
            "replication-confirmation": 92, "final-method-test": 140}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def partition_rows(path: Path) -> dict[str, list[dict[str, str]]]:
    from researchos.original_partitions import (
        SPLIT_NAMES, allocate_quotas, sample_sort_key, session_sort_key,
    )
    groups = {split: [] for split in COUNTS}
    seen: set[str] = set()
    session_partition: dict[str, str] = {}
    source_sessions: dict[str, set[str]] = defaultdict(set)
    for row in read_csv(path):
        split, sample_id = row["partition"], row["sample_id"]
        if split not in groups or sample_id in seen:
            raise ValueError("Unknown partition or duplicate sample ID")
        source, session, frame = sample_id.split("-")
        if source not in {"ave", "vwg"} or row["source"] != source:
            raise ValueError(f"Invalid source for {sample_id}")
        if row["recording_session_key"] != f"{source}-{session}" or row["frame_id"] != frame:
            raise ValueError(f"Inconsistent session/frame metadata for {sample_id}")
        previous = session_partition.setdefault(row["recording_session_key"], split)
        if previous != split:
            raise ValueError("Recording sessions overlap between partitions")
        seen.add(sample_id)
        source_sessions[source].add(session)
        groups[split].append(row)
    expected_assignment = {}
    for source, sessions in sorted(source_sessions.items()):
        ordered = sorted(sessions, key=lambda value: session_sort_key(source, value))
        offset = 0
        for split, count in zip(SPLIT_NAMES, allocate_quotas(len(ordered)), strict=True):
            for session in ordered[offset:offset + count]:
                expected_assignment[f"{source}-{session}"] = split
            offset += count
    if expected_assignment != session_partition:
        raise ValueError("Partition CSV differs from the registered session allocation")
    for split, rows in groups.items():
        rows.sort(key=lambda row: int(row["partition_order_index"]))
        if len(rows) != COUNTS[split] or len({row["recording_session_key"] for row in rows}) != SESSIONS[split]:
            raise ValueError(f"Unexpected sample/session count for {split}")
        if [int(row["partition_order_index"]) for row in rows] != list(range(len(rows))):
            raise ValueError(f"Non-contiguous sample order for {split}")
        ids = [row["sample_id"] for row in rows]
        if ids != sorted(ids, key=lambda value: sample_sort_key(split, value)):
            raise ValueError(f"Sample order differs from the registered hash order: {split}")
    return groups


def taxonomy_rows(path: Path) -> list[dict[str, int | str]]:
    rows = [{"source_class_id": int(row["source_label_id"]),
             "target_class_id": int(row["target_label_id"]),
             "source_class_name": row.get("source_label_name", ""),
             "target_class_name": row.get("target_class_name", ""),
             "resolution_rule": row.get("mapping_resolution_rule", "")}
            for row in read_csv(path)]
    if len(rows) != 101 or {row["source_class_id"] for row in rows} != {*range(100), 255}:
        raise ValueError("Taxonomy must contain exactly source IDs 0..99 and 255")
    if any(row["target_class_id"] not in {0, 1, 2, 255} for row in rows):
        raise ValueError("Taxonomy contains an invalid target ID")
    if {row["source_class_id"]: row["target_class_id"] for row in rows}.get(255) != 255:
        raise ValueError("Source 255 must map to ignore 255")
    return rows


def _index(directory: Path, extensions: set[str], expected: set[str], data_root: Path) -> dict[str, Path]:
    result = {}
    for path in sorted(directory.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in extensions or path.stem not in expected:
            continue
        if path.stem in result:
            raise ValueError(f"Duplicate file for sample {path.stem}")
        path = path.resolve(strict=True)
        path.relative_to(data_root)
        result[path.stem] = path
    if set(result) != expected:
        missing = sorted(expected - set(result))
        raise ValueError(f"Missing {len(missing)} files under {directory}; first IDs: {missing[:5]}")
    return result


def _write(path: Path, payload: dict) -> None:
    text = json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    if path.exists() and path.read_text() != text:
        raise FileExistsError(f"Refusing to replace a different manifest: {path}")
    path.write_text(text)


def prepare(*, data_root: Path, image_dir: Path, label_dir: Path,
            partition_csv: Path, taxonomy_csv: Path, output_dir: Path) -> dict:
    groups = partition_rows(partition_csv)
    mapping = taxonomy_rows(taxonomy_csv)
    data_root = data_root.resolve(strict=True)
    image_dir = (data_root / image_dir).resolve(strict=True)
    label_dir = (data_root / label_dir).resolve(strict=True)
    expected = {row["sample_id"] for rows in groups.values() for row in rows}
    images = _index(image_dir, {".jpg", ".jpeg", ".png"}, expected, data_root)
    labels = _index(label_dir, {".png"}, expected, data_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    primary_rows = []
    for split, source_rows in groups.items():
        inputs, targets = [], []
        for index, source in enumerate(source_rows):
            sample_id = source["sample_id"]
            row = {"index": index, "sample_id": sample_id, "split": split,
                   "source_set": source["source"], "recording_session_key": source["recording_session_key"],
                   "image_path": images[sample_id].relative_to(data_root).as_posix(),
                   "image_sha256": sha256_file(images[sample_id])}
            target = {"sample_id": sample_id, "split": split,
                      "label_path": labels[sample_id].relative_to(data_root).as_posix(),
                      "label_sha256": sha256_file(labels[sample_id]),
                      "label_encoding": "source_ids", "taxonomy": "taxonomy.json"}
            targets.append(target)
            if split in {"train", "development"}:
                row.update({key: target[key] for key in ("label_path", "label_sha256")})
            inputs.append(row)
        filename = f"{split}.json" if split in {"train", "development"} else f"{split}.inputs.json"
        _write(output_dir / filename, {"schema_version": 1, "split": split,
               "sample_count": len(inputs), "labels_present": split in {"train", "development"},
               "label_encoding": "source_ids" if split in {"train", "development"} else None,
               "rows": inputs})
        _write(output_dir / f"{split}.labels.json", {"schema_version": 1, "split": split,
               "labels_present": True, "label_encoding": "source_ids", "rows": targets})
        paths.extend([filename, f"{split}.labels.json"])
        if split == "replication-confirmation":
            primary_rows = inputs
    _write(output_dir / "taxonomy.json", {"schema_version": 1, "ignore_index": 255,
           "label_encoding": "source_ids", "mapping_rows": mapping})
    from researchos.original_partitions import build_cost_manifest
    split_payload = {"partitions": {"replication-confirmation": {
        "session_keys": sorted({r["recording_session_key"] for r in primary_rows}),
        "sample_order": [r["sample_id"] for r in primary_rows]}}}
    original_file_rows = [{**r, "files": {"image": {"sha256": r["image_sha256"]}}} for r in primary_rows]
    _write(output_dir / "cost40.inputs.json", build_cost_manifest(split_payload, original_file_rows))
    summary = {"schema_version": 1, "counts": COUNTS, "session_counts": SESSIONS,
               "partition_csv_sha256": sha256_file(partition_csv),
               "taxonomy_csv_sha256": sha256_file(taxonomy_csv),
               "label_pixels_decoded": False, "files": paths + ["taxonomy.json", "cost40.inputs.json"]}
    _write(output_dir / "manifest_summary.json", summary)
    return summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, default=Path("images"))
    parser.add_argument("--label-dir", type=Path, default=Path("labelIds/CropAndWeed"))
    parser.add_argument("--partition-csv", type=Path, default=Path("data/cropandweed_partitions.csv"))
    parser.add_argument("--taxonomy-csv", type=Path, default=Path("data/cropandweed_class_mapping.csv"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(prepare(**vars(args)), indent=2))
    return 0
