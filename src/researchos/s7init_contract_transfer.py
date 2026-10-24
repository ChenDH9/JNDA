"""Portable same-seed adapter and local checkpoint identity checks."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
from typing import Any
from researchos.s8r_ex01_execution import S8RContractError

SYSTEM = "S7_Init_JNDA"

SEEDS = (1701, 1702, 1703, 1704, 1705)

SOURCE_SYSTEM = "Fresh_S7"

PROTOCOL_SHA256 = "adf3ff063ae71b7f2376b468a83e3445a5cc44f71ad6a5ad4aedab32263552af"

EXPERIMENT_CONFIG_SHA256 = "55ac98482f805c944b843e9e26ca18057fbfd4d8461e68836f667c79ba477d67"

EXPECTED_S7_ADAPTER_SHA256 = {
    1701: "31530b826f36afa622439ca802e8ca4ae001eef2479724580fb1de06f0563395",
    1702: "90f121bae25c3ad0946ed7483100c8c4b3b27f9d95992ae3e4eed160746166b4",
    1703: "d07382b49daefd66ab3fce625ad9cd02e7c15b03ed9b2dfb9788f57592a8eda1",
    1704: "7cc3d716ff07e0d408e35ff48676f483fa9db5f016f33e6bd5fd377671e5678b",
    1705: "b838a71ddcedd4a48465377128e9b1276d1aba0ffeafff182a600051c6ef7051",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise S8RContractError('Expected a JSON metadata object')
    return value


def verify_same_seed_s7_checkpoint(source_run_root: str | Path, seed: int) -> dict[str, Any]:
    """Check a complete same-seed S7 run, including newly reproduced checkpoints."""
    if seed not in SEEDS:
        raise S8RContractError('Seed is outside the fixed five-seed matrix')
    root = Path(source_run_root).resolve(strict=True)
    checkpoint = (root / SOURCE_SYSTEM / f'seed_{seed}' / 'checkpoint_epoch_10').resolve(strict=True)
    checkpoint.relative_to(root)
    receipt_path = checkpoint / 'formal_checkpoint_receipt.json'
    state_path = checkpoint / 'training_state.pt'
    adapter_path = checkpoint / 'pytorch_lora_weights.safetensors'
    receipt = read_json(receipt_path)
    state_sha, adapter_sha = sha256_file(state_path), sha256_file(adapter_path)
    checks = {
        'system': receipt.get('system') == SOURCE_SYSTEM,
        'seed': receipt.get('seed') == seed,
        'epoch': receipt.get('completed_epochs') == 10,
        'updates': receipt.get('global_step') == 48210,
        'scientific': receipt.get('scientific_checkpoint') is True,
        'status': receipt.get('status') == 'PASS',
        'training_state_hash': receipt.get('checkpoint_sha256') == state_sha,
        'adapter_hash': receipt.get('adapter_sha256') == adapter_sha,
        'paired_eligible': receipt.get('same_seed_arm_a_eligible') is True,
        'configuration_identity': isinstance(receipt.get('config_sha256'), str) and len(receipt['config_sha256']) == 64,
    }
    if not all(checks.values()):
        raise S8RContractError(f'Same-seed S7 checkpoint differs: {checks}')
    return {'checkpoint_dir': str(checkpoint), 'receipt_path': str(receipt_path),
            'receipt_sha256': sha256_file(receipt_path), 'training_state_path': str(state_path),
            'training_state_sha256': state_sha, 'adapter_path': str(adapter_path),
            'adapter_sha256': adapter_sha, 'seed': seed, 'checks': checks}


def load_portable_final_state(output: Path, *, system: str, seed: int, frozen_loader):
    """Keep model-state checks while replacing historical repository/clock bindings."""
    identity_path = output / 'run_identity.json'
    if not identity_path.is_file():
        # Complete original checkpoints retain their original strict metadata contract.
        return frozen_loader(output, system=system, seed=seed)
    if system != SYSTEM or seed not in SEEDS:
        raise S8RContractError('This checkpoint belongs to a different route or seed')
    import torch
    identity = read_json(identity_path)
    path = output / 'checkpoint_epoch_10' / 'training_state.pt'
    receipt = read_json(path.parent / 'formal_checkpoint_receipt.json')
    state = torch.load(path, map_location='cpu', weights_only=False)
    if not isinstance(state, dict):
        raise S8RContractError('Final checkpoint is not a mapping')
    context = state.get('producer_context', {})
    source_hash = identity['source_s7']['adapter_sha256']
    adapter = path.parent / 'pytorch_lora_weights.safetensors'
    checks = {
        'identity_system': identity.get('system') == SYSTEM,
        'identity_seed': identity.get('seed') == seed,
        'state_system': state.get('system') == system,
        'state_seed': state.get('seed') == seed,
        'state_epoch': state.get('completed_epochs') == 10,
        'state_updates': state.get('global_step') == 48210,
        'model_state_keys': isinstance(state.get('model_state'), dict) and set(state['model_state']) == {'head', 'lora'},
        'receipt_system': receipt.get('system') == system,
        'receipt_seed': receipt.get('seed') == seed,
        'receipt_epoch': receipt.get('completed_epochs') == 10,
        'receipt_updates': receipt.get('global_step') == 48210,
        'receipt_status': receipt.get('status') == 'PASS',
        'receipt_scientific': receipt.get('scientific_checkpoint') is True,
        'checkpoint_hash': receipt.get('checkpoint_sha256') == sha256_file(path),
        'trained_adapter_hash': adapter.is_file() and receipt.get('adapter_sha256') == sha256_file(adapter),
        'recipe_identity': receipt.get('config_sha256') == context.get('config_sha256') == identity['recipe_sha256'],
        'source_adapter_identity': receipt.get('source_s7_adapter_sha256') == context.get('source_s7_adapter_sha256') == source_hash,
        'run_provenance': receipt.get('run_provenance') == identity,
    }
    if not all(checks.values()):
        raise S8RContractError(f'Final S7-initialized checkpoint binding differs: {checks}')
    return state, path
