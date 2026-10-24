"""Portable manifests and checked image access."""
import csv
import hashlib
import json
import os
from pathlib import Path


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def load_rows(path):
    path = Path(path)
    if path.suffix.lower() == '.csv':
        with path.open(newline='') as f:
            rows = list(csv.DictReader(f))
    else:
        value = json.loads(path.read_text())
        rows = value['rows'] if isinstance(value, dict) else value
    if not isinstance(rows, list) or not rows:
        raise ValueError('Manifest must contain a nonempty list of rows')
    rows = [dict(row) for row in rows]
    if len({row['sample_id'] for row in rows}) != len(rows):
        raise ValueError('Duplicate sample_id in manifest')
    for row in rows:
        row['source'] = row.get('source', row.get('source_set', row['sample_id'].split('-')[0]))
        row['session_id'] = row.get('session_id', row.get('recording_session_key'))
        if not row['session_id']:
            raise ValueError('Manifest must retain source/session identities')
    return rows


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    os.replace(tmp, path)


class Images:
    def __init__(self, data_root, taxonomy=None):
        import numpy as np
        self.root = Path(data_root).resolve(strict=True)
        self.lookup = None
        if taxonomy:
            p = Path(taxonomy)
            if p.suffix.lower() == '.csv':
                with p.open(newline='') as f:
                    rows = list(csv.DictReader(f))
            else:
                value = json.loads(p.read_text())
                rows = value['mapping_rows'] if isinstance(value, dict) else value
            lookup = np.full(256, 255, dtype=np.uint8)
            seen = set()
            for row in rows:
                source = int(row.get('source_class_id', row.get('source_label_id')))
                target = int(row.get('target_class_id', row.get('target_label_id')))
                if source in seen or target not in (0, 1, 2, 255):
                    raise ValueError('Invalid or duplicate taxonomy row')
                seen.add(source)
                lookup[source] = target
            if seen != {*range(100), 255}:
                raise ValueError('The published taxonomy must cover 0..99 and 255')
            self.lookup = lookup

    def path(self, row, kind):
        relative = row.get(kind + '_path')
        digest = row.get(kind + '_sha256')
        if not relative or not digest:
            raise ValueError('Use a prepared manifest containing explicit paths and SHA256 values')
        path = (self.root / relative).resolve(strict=True)
        path.relative_to(self.root)
        if sha256(path) != digest:
            raise ValueError(f'{kind} SHA256 mismatch for {row["sample_id"]}')
        return path

    def rgb(self, row):
        import numpy as np
        from PIL import Image
        with Image.open(self.path(row, 'image')) as im:
            return np.asarray(im.convert('RGB').resize((512, 512), Image.Resampling.BILINEAR), dtype=np.uint8).copy()

    def label(self, row):
        import numpy as np
        from PIL import Image
        if self.lookup is None:
            raise ValueError('Ground-truth access requires an explicit taxonomy')
        with Image.open(self.path(row, 'label')) as im:
            raw = np.asarray(im.resize((512, 512), Image.Resampling.NEAREST), dtype=np.uint8)
        if raw.ndim != 2:
            raise ValueError('Ground truth must be an indexed single-channel image')
        return self.lookup[raw].copy()
