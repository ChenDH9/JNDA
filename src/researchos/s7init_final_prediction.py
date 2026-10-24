"""Image-only final prediction for the independently trained S7-initialized route."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
from researchos.s8r_ex01_execution import S8RContractError, sha256_file

FORMAL_SEEDS = (1701, 1702, 1703, 1704, 1705)
PUBLIC_SYSTEM = 'S7_init_JNDA'
INTERNAL_SYSTEM = 'S7_Init_JNDA'


def validate_prediction(prediction: np.ndarray) -> tuple[int, ...]:
    if prediction.shape != (512, 512):
        raise S8RContractError('prediction shape differs')
    if prediction.dtype != np.uint8:
        raise S8RContractError('prediction dtype differs')
    classes = tuple(int(value) for value in np.unique(prediction))
    if not set(classes) <= {0, 1, 2}:
        raise S8RContractError('prediction contains an invalid class ID')
    return classes


def _sample_seed(seed: int, sample_id: str) -> int:
    return int.from_bytes(hashlib.sha256(f'{seed}\0{sample_id}'.encode()).digest()[:8], 'big') % (2**63 - 1)


def load_final_inputs(manifest_path: Path) -> list[dict]:
    body = json.loads(manifest_path.read_text())
    rows = body.get('rows', [])
    if body.get('split') != 'final-method-test' or body.get('labels_present') is not False:
        raise S8RContractError('Final prediction requires an image-only final-method-test manifest')
    if len(rows) != 1163 or len({r['sample_id'] for r in rows}) != 1163:
        raise S8RContractError('Final frame must contain1163 unique images')
    if [r['index'] for r in rows] != list(range(1163)) or any(r.get('split') != 'final-method-test' for r in rows):
        raise S8RContractError('Final input order or split differs')
    if any(any('label' in key for key in r) for r in rows):
        raise S8RContractError('Final prediction inputs must not contain labels')
    sessions = {(r['source_set'], r['recording_session_key']) for r in rows}
    if len(sessions) != 140 or {s: sum(x[0] == s for x in sessions) for s in ('ave', 'vwg')} != {'ave': 40, 'vwg': 100}:
        raise S8RContractError('Final source/session frame differs')
    return rows


def write_predictions(*, training, manifest_path: Path, workspace: Path, output: Path, seed: int) -> dict:
    if seed not in FORMAL_SEEDS:
        raise S8RContractError('Unknown formal seed')
    rows = load_final_inputs(manifest_path)
    # The retained fifth-route predictor reconstructs fresh LoRA structure and
    # loads the trained joint-native head and LoRA; it does not reload source S7.
    predictor = training._Predictor(root=workspace, output=output, system=INTERNAL_SYSTEM,
                                   seed=seed, s7_checkpoint=None)
    folder = output / 'final_predictions'
    folder.mkdir(exist_ok=False)
    inventory = []
    try:
        for row in rows:
            rgb = training._load_rgb(workspace, row)
            prediction = predictor.predict(rgb, sample_seed=_sample_seed(seed, row['sample_id']))
            validate_prediction(prediction)
            path = folder / f'{row["index"]:04d}.png'
            training._save_prediction(path, prediction)
            inventory.append({'sample_id': row['sample_id'], 'source': row['source_set'],
                              'session_id': row['recording_session_key'],
                              'path': path.relative_to(output).as_posix(), 'sha256': sha256_file(path)})
    finally:
        predictor.close()
    result = {'system': PUBLIC_SYSTEM, 'internal_system': INTERNAL_SYSTEM, 'seed': seed,
              'split': 'final-method-test', 'checkpoint_epoch': 10,
              'checkpoint_sha256': sha256_file(output / 'checkpoint_epoch_10/training_state.pt'),
              'rows': inventory}
    (output / 'final_predictions.json').write_text(json.dumps(result, indent=2) + '\n')
    return result
