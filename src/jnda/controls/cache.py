"""Lossless fixed-backbone feature caches for train/development images."""
import json
from pathlib import Path
from .io import Images, load_rows, sha256, write_json


def build(args):
    import torch
    from .native import NativeBackbone
    from .terminal import NativeOnlyBackbone
    from .runtime import configure, history
    backend = configure(args.device)
    old = history(args.model_root)
    train, dev = load_rows(args.train_manifest), load_rows(args.dev_manifest)
    if len(train) != 4821 or len(dev) != 1206:
        raise ValueError('Cache requires the complete4821/1206 train/development partitions')
    rows = train + dev
    if len(rows) != 6027 or len({r['sample_id'] for r in rows}) != 6027:
        raise ValueError('Cache requires exactly 4821 train and1206 development images')
    if any(r.get('split') not in ('train', 'development') for r in rows):
        raise ValueError('Cache construction is restricted to train/development')
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    data = Images(args.data_root)
    cls = NativeOnlyBackbone if args.feature == 'terminal' else NativeBackbone
    model = cls(old, Path(args.data_root), lora=False, seed=2701)
    records = []
    try:
        for row in rows:
            terminal, tap = model.features(data.rgb(row))
            value = terminal if args.feature == 'terminal' else tap
            expected = torch.bfloat16 if args.feature == 'terminal' else torch.float32
            width = 128 if args.feature == 'terminal' else 3072
            if value.dtype != expected or tuple(value.shape) != (1, 1024, width):
                raise RuntimeError('Producer feature dtype/shape differs from the recorded method')
            name = row['sample_id'] + '.pt'
            if Path(name).name != name:
                raise ValueError('Unsafe sample identifier')
            torch.save(value.detach().cpu().contiguous(), output / name)
            records.append({'sample_id': row['sample_id'], 'split': row['split'], 'file': name,
                            'sha256': sha256(output / name), 'image_sha256': row['image_sha256']})
        write_json(output / 'manifest.json', {'feature': args.feature, 'adapter_state': 'none',
                   'dtype': str(expected), 'shape': [1, 1024, width], 'producer': model.metadata,
                   'backend': backend, 'model_snapshot': old._supplementary_model_identity,
                   'rows': records, 'complete': True})
    finally:
        model.close()


class Cache:
    def __init__(self, root, feature, rows):
        self.root = Path(root).resolve(strict=True)
        m = json.loads((self.root / 'manifest.json').read_text())
        if not m.get('complete') or m.get('feature') != feature or m.get('adapter_state') != 'none':
            raise ValueError('Incompatible or incomplete fixed-backbone cache')
        from .model_assets import FLUX_SNAPSHOT
        import hashlib
        expected_fingerprint = hashlib.sha256(json.dumps(FLUX_SNAPSHOT, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        if m.get('model_snapshot', {}).get('snapshot_fingerprint') != expected_fingerprint:
            raise ValueError('Cache base model is not the fixed published snapshot')
        if m.get('producer', {}).get('prompt_sha256') != '13f46d9f3a35722b86f498a2a825f62a0907fb50b19ff727cdd089ee6bc810df' or m.get('producer', {}).get('prompt_state_sha256') != 'e9f95dd59f1bc79f8cc08516e4274e19d620b6567720580c08071118b1733fb9':
            raise ValueError('Cache prompt producer differs from the recorded method')
        if m.get('backend', {}).get('tf32_cudnn') is not False or m.get('backend', {}).get('tf32_matmul') is not False:
            raise ValueError('Cache backend differs from the fixed producer')
        self.manifest_sha256 = sha256(self.root / 'manifest.json')
        self.index = {r['sample_id']: r for r in m['rows']}
        if len(self.index) != len(m['rows']):
            raise ValueError('Duplicate cache entries')
        for r in rows:
            q = self.index.get(r['sample_id'])
            if not q or q['image_sha256'] != r['image_sha256']:
                raise ValueError('Cache image identity differs from the training manifest')
        self.feature = feature

    def get(self, sample_id):
        import torch
        r = self.index[sample_id]
        p = self.root / r['file']
        p.resolve().relative_to(self.root)
        if sha256(p) != r['sha256']:
            raise ValueError('Cache content SHA256 mismatch')
        v = torch.load(p, map_location='cpu', weights_only=True)
        dtype, width = (torch.bfloat16, 128) if self.feature == 'terminal' else (torch.float32, 3072)
        if v.dtype != dtype or tuple(v.shape) != (1, 1024, width):
            raise ValueError('Lossless cache dtype/shape changed')
        return v
