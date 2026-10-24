#!/usr/bin/env python3
"""Download explicitly selected pretrained assets or verify local copies."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def verify(destination: Path, files: list[dict]) -> list[dict]:
    checked = []
    for item in files:
        relative = Path(item['path'])
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Model manifest paths must be relative and contained')
        path = destination / relative
        if not path.is_file():
            raise FileNotFoundError(f'Missing model file: {path}')
        if path.stat().st_size != item['bytes']:
            raise ValueError(f'Model file size mismatch: {relative}')
        digest = sha256_file(path)
        if digest != item['sha256']:
            raise ValueError(f'Model file checksum mismatch: {relative}')
        checked.append({'path': item['path'], 'bytes': item['bytes'], 'sha256': digest})
    return checked


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=('flux', 'segformer', 'upernet'), required=True)
    parser.add_argument('--destination', type=Path)
    parser.add_argument('--verify-only', action='store_true', help='Read and hash existing files without network access')
    args = parser.parse_args()
    catalog = json.loads((ROOT / 'configs/model_assets.json').read_text())
    entry = catalog[args.model]
    destination = args.destination or ROOT / entry['default_destination']
    if not args.verify_only:
        destination.mkdir(parents=True, exist_ok=True)
        if entry['provider'] == 'huggingface':
            from huggingface_hub import snapshot_download
            snapshot_download(repo_id=entry['repo_id'], revision=entry['revision'],
                              allow_patterns=[x['path'] for x in entry['files']],
                              local_dir=str(destination))
        else:
            for item in entry['files']:
                path = destination / item['path']
                if path.exists():
                    verify(destination, [item])
                    continue
                temporary = path.with_suffix(path.suffix + '.partial')
                try:
                    with urllib.request.urlopen(item['url'], timeout=60) as source, temporary.open('wb') as out:
                        for chunk in iter(lambda: source.read(4 * 1024 * 1024), b''):
                            out.write(chunk)
                    if temporary.stat().st_size != item['bytes'] or sha256_file(temporary) != item['sha256']:
                        raise ValueError(f'Download checksum mismatch: {item["path"]}')
                    os.replace(temporary, path)
                finally:
                    temporary.unlink(missing_ok=True)
    report = verify(destination, entry['files'])
    print(json.dumps({'model': args.model, 'destination': str(destination), 'verified_files': report}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
