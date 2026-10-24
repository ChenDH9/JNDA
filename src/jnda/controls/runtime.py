"""Explicit CUDA and local-model configuration for the published methods."""
import os
from pathlib import Path


def configure(device='cuda:0', *, legacy=False):
    import torch
    if str(device) != 'cuda:0':
        raise ValueError('Select one GPU with CUDA_VISIBLE_DEVICES, then use --device cuda:0')
    if not torch.cuda.is_available():
        raise RuntimeError('The model path requires a CUDA GPU and the recorded dependencies')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = bool(legacy)
    torch.set_float32_matmul_precision('highest')
    torch.use_deterministic_algorithms(False)
    torch.set_num_threads(4)
    try:
        torch.set_num_interop_threads(4)
    except RuntimeError:
        pass
    return {'torch': torch.__version__, 'cuda': torch.version.cuda,
            'device': torch.cuda.get_device_name(0), 'tf32_cudnn': bool(legacy),
            'tf32_matmul': False, 'autocast': 'bfloat16'}


def history(model_root):
    from researchos import s8r_ex01_training as old
    root = Path(model_root).resolve(strict=True)
    if not (root / 'model_index.json').is_file():
        raise FileNotFoundError('Expected the complete local FLUX model snapshot')
    import hashlib
    if not hasattr(old, '_supplementary_original_prompt'):
        if hashlib.sha256(old.PROMPT.encode()).hexdigest() != '13f46d9f3a35722b86f498a2a825f62a0907fb50b19ff727cdd089ee6bc810df':
            raise ValueError('Original scientific prompt source changed')
        old._supplementary_original_prompt = old.PROMPT
    old.PROMPT = old._supplementary_original_prompt
    old._supplementary_model_identity = model_identity(root)
    old._model_root = lambda _: root
    return old


def model_identity(root, family="flux"):
    import hashlib
    import json
    from .io import sha256
    from .model_assets import FLUX_SNAPSHOT, UPERNET_SNAPSHOT
    root = Path(root).resolve(strict=True)
    snapshot = FLUX_SNAPSHOT if family == 'flux' else UPERNET_SNAPSHOT
    for row in snapshot['files']:
        path = (root / row['path']).resolve(strict=True)
        path.relative_to(root)
        if path.stat().st_size != row['bytes'] or sha256(path) != row['sha256']:
            raise ValueError('Fixed FLUX snapshot identity mismatch: ' + row['path'])
    fingerprint = hashlib.sha256(json.dumps(snapshot, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    return {'revision': snapshot['revision'], 'snapshot_fingerprint': fingerprint, 'verified_files': len(snapshot['files'])}
