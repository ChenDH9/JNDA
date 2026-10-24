"""Portable training and final prediction for S7-initialized JNDA."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from researchos import portable_original as original
from researchos.s7init_contract_transfer import (
    SYSTEM, SEEDS, read_json, sha256_file, verify_same_seed_s7_checkpoint,
    load_portable_final_state,
)

SOURCE_COMMIT = '6d5a36a304b38b314d7b7076b2e12135723afcb3'


def activate(runtime: dict, workspace: Path):
    from researchos import s7init_training as training
    base, _ = original.activate(runtime, workspace)
    for name in ('_manifest', '_taxonomy', '_verified_path', '_model_root', '_segformer_model'):
        setattr(training, name, getattr(base, name))
    if not hasattr(training, '_portable_original_final_loader'):
        training._portable_original_final_loader = training._load_final_state
    training._load_final_state = lambda output, **kwargs: load_portable_final_state(
        output, frozen_loader=training._portable_original_final_loader, **kwargs)
    return training


def validate_recipe(path: Path) -> dict:
    from researchos import s7init_training as training
    recipe = read_json(path)
    training._validate_s7init_effective_config(config=recipe, config_path=path,
                                               config_sha256=sha256_file(path))
    init = recipe['initialization']
    if init.get('base_revision') != '8c44a2fbef88fae175da65df054db0f901aa9747' or init.get('source_system') != 'Fresh_S7' or init.get('source_checkpoint_epoch') != 10:
        raise ValueError('The source base model or S7 initialization endpoint changed')
    return recipe


def run_identity(*, runtime_path: Path, recipe_path: Path, runtime: dict, seed: int,
                 source: dict, model_identity: dict) -> dict:
    from researchos import s7init_training, s7init_lora
    paths = original.configured_paths(runtime)
    return {'schema': 's7_initialized_reproduction_identity_v1', 'system': SYSTEM, 'seed': seed,
            'upstream_source_commit': SOURCE_COMMIT,
            'public_training_source_sha256': sha256_file(Path(s7init_training.__file__)),
            'strict_lora_source_sha256': sha256_file(Path(s7init_lora.__file__)),
            'runtime_configuration_sha256': sha256_file(runtime_path),
            'recipe_sha256': sha256_file(recipe_path),
            'manifest_sha256': {name: sha256_file(paths['manifest_dir'] / name)
                                for name in ('train.json', 'development.json', 'taxonomy.json')},
            'source_s7': source, 'base_model': model_identity,
            'optimizer_source': 'new_categorical_optimizer_not_S7_state',
            'initialization': 'fresh_native_head_before_same_seed_S7_LoRA_overwrite'}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=('check', 'train', 'predict'))
    p.add_argument('--config', type=Path, default=Path('configs/original/runtime.example.json'))
    p.add_argument('--recipe', type=Path, default=Path('configs/original/s7_initialized.json'))
    p.add_argument('--workspace', type=Path, default=Path('runs/s7_initialized'))
    p.add_argument('--source-workspace', type=Path, default=Path('runs/original'))
    p.add_argument('--seed', type=int, choices=SEEDS)
    p.add_argument('--process', choices=('process_a', 'process_b'))
    p.add_argument('--device', default='0', help='One visible CUDA ordinal or device UUID')
    args = p.parse_args(argv)
    runtime = original.load_config(args.config)
    recipe = validate_recipe(args.recipe)
    paths = original.configured_paths(runtime)
    check = original.check_config(runtime, action='train' if args.action == 'train' else 'predict', system=SYSTEM)
    if args.action in ('check', 'predict') and not (paths['manifest_dir'] / 'final-method-test.inputs.json').is_file():
        check['missing'].append('manifest:final-method-test.inputs.json')
    if args.action == 'check':
        check['status'] = 'PASS' if not check['missing'] else 'MISSING_INPUTS'
        check.update(route=SYSTEM, recipe_valid=True, models_loaded=False, GPU_operations=0)
        print(json.dumps(check, indent=2))
        return 0 if check['status'] == 'PASS' else 2
    if check['missing']:
        p.error('Missing configured inputs: ' + ', '.join(check['missing']))
    if args.seed is None:
        p.error('--seed is required')
    if args.action == 'train' and args.process is None:
        p.error('Training requires --process process_a or process_b in separate invocations')
    workspace = args.workspace.resolve()
    output = workspace / f'seed_{args.seed}'
    if args.action == 'train' and args.process == 'process_a' and output.exists():
        raise FileExistsError('Process A requires a new per-seed output directory')
    if args.action == 'predict' and not (output / 'checkpoint_epoch_10/training_state.pt').is_file():
        raise FileNotFoundError('The own trained epoch10 checkpoint is absent')
    from jnda.controls.runtime import model_identity
    model = model_identity(paths['flux_model'])
    workspace.mkdir(parents=True, exist_ok=True)
    training = activate(runtime, workspace)
    if args.action == 'train':
        source = verify_same_seed_s7_checkpoint(args.source_workspace, args.seed)
        identity = run_identity(runtime_path=args.config, recipe_path=args.recipe,
                                runtime=runtime, seed=args.seed, source=source, model_identity=model)
        identity_path = output / 'run_identity.json'
        if args.process == 'process_a':
            output.mkdir()
            identity_path.write_text(json.dumps(identity, indent=2) + '\n')
        elif not identity_path.is_file() or read_json(identity_path) != identity:
            raise ValueError('Process B data, source adapter, model or recipe identity differs from process A')
        gpu = original.bind_device(args.device)
        result = training.train_native_segment(project_root=workspace, config=recipe,
            config_path=args.recipe.resolve(), config_sha256=sha256_file(args.recipe),
            system=SYSTEM, seed=args.seed, output_dir=output, gpu=gpu,
            s7_checkpoint=Path(source['checkpoint_dir']), process=args.process,
            audit_dir=output / 'scientific_audit', run_provenance=identity)
        if result.get('status') != 'PASS' or result.get('scientific_result') is not True:
            raise RuntimeError('The fixed training segment did not complete')
        (output / (args.process + '_result.json')).write_text(json.dumps(result, indent=2) + '\n')
    else:
        identity_path = output / 'run_identity.json'
        if identity_path.is_file() and read_json(identity_path)['base_model'] != model:
            raise ValueError('Prediction base-model snapshot differs from the training producer')
        original.bind_device(args.device)
        from researchos.s7init_final_prediction import write_predictions
        result = write_predictions(training=training, manifest_path=paths['manifest_dir'] / 'final-method-test.inputs.json',
                                   workspace=workspace, output=output, seed=args.seed)
    print(json.dumps({'system': SYSTEM, 'seed': args.seed, 'action': args.action,
                      'status': 'PASS', 'output': str(output),
                      'completed_epochs': result.get('completed_epochs'),
                      'predictions': len(result.get('rows', []))}, indent=2))
    return 0
