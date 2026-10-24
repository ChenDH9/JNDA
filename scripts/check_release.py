#!/usr/bin/env python3
"""Verify release file identities and the fixed study metadata offline."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=ROOT / 'RELEASE_MANIFEST.json')
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    for item in manifest['files']:
        relative = Path(item['path'])
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Release paths must be contained relative paths')
        path = ROOT / relative
        if path.is_symlink() or not path.is_file():
            raise ValueError(f'Missing or non-regular release file: {relative}')
        payload = path.read_bytes()
        if len(payload) != item['bytes'] or hashlib.sha256(payload).hexdigest() != item['sha256']:
            raise ValueError(f'Release file differs: {relative}')
    with (ROOT / 'data/cropandweed_partitions.csv').open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    expected = {'train': 4821, 'development': 1206, 'replication-confirmation': 844, 'final-method-test': 1163}
    if Counter(row['partition'] for row in rows) != expected:
        raise ValueError('Unexpected study partition counts')
    if len({row['sample_id'] for row in rows}) != 8034:
        raise ValueError('Duplicate study sample IDs')
    sessions = defaultdict(set)
    for row in rows:
        sessions[row['recording_session_key']].add(row['partition'])
    if len(sessions) != 929 or any(len(value) != 1 for value in sessions.values()):
        raise ValueError('Recording sessions overlap partitions')
    for name, count in expected.items():
        order = [int(row['partition_order_index']) for row in rows if row['partition'] == name]
        if order != list(range(count)):
            raise ValueError(f'Study order differs in {name}')
    with (ROOT / 'data/cropandweed_class_mapping.csv').open(encoding='utf-8-sig', newline='') as stream:
        labels = list(csv.DictReader(stream))
    if len(labels) != 101 or {int(row['source_label_id']) for row in labels} != set(range(100)) | {255}:
        raise ValueError('Incomplete source-label mapping')
    if not {int(row['target_label_id']) for row in labels}.issubset({0, 1, 2, 255}):
        raise ValueError('Unexpected target label')
    with (ROOT / 'data/phenobench_validation_selection.csv').open(encoding='utf-8-sig', newline='') as stream:
        validation = list(csv.DictReader(stream))
    if len(validation) != 500 or len({row['sample_id'] for row in validation}) != 500:
        raise ValueError('PhenoBench selection differs')
    print(json.dumps({'release': manifest['version'], 'verified_files': len(manifest['files']),
                      'cropandweed_samples': len(rows), 'sessions': len(sessions),
                      'phenobench_validation_samples': len(validation), 'status': 'PASS'}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
