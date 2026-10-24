"""Portable execution of the fixed supplementary training recipes."""
import copy
import json
import math
import os
import time
from pathlib import Path
from .io import Images, load_rows, sha256, write_json
from .protocol import stable_order, t3_group_extension


def extension_decision(paths, output):
    curves, checkpoints = {}, {}
    for path in paths:
        body = json.loads(Path(path).read_text())
        seed = body['seed']
        if seed in curves or body['route'] != 'T3':
            raise ValueError('Supply exactly one complete T3 development curve per seed')
        curves[seed] = [r for r in body['curve'] if r['epoch'] <= 10]
        epoch10 = next(r for r in curves[seed] if r['epoch'] == 10)
        checkpoints[str(seed)] = epoch10['checkpoint_sha256']
    result = t3_group_extension(curves)
    write_json(output, {**result, 'epoch10_checkpoint_sha256': checkpoints})


def run(args):
    import numpy as np
    import torch
    from researchos import s8r_ex01_training as old
    from . import native, upernet
    from .terminal import NativeOnlyBackbone
    from .cache import Cache
    from .runtime import configure, history, model_identity
    route, seed = args.route, args.seed
    allowed_seeds = range(1701, 1706) if route in ('T1', 'B1') else range(1701, 1704)
    if seed not in allowed_seeds:
        raise ValueError('Seed is outside the published recipe')
    epochs = args.epochs if args.epochs is not None else (40 if route == 'B1' else 10 if route == 'T3' else 30)
    if (route == 'B1' and epochs != 40) or (route in ('T1', 'T2', 'T4') and epochs != 30) or (route == 'T3' and epochs not in (10, 30)):
        raise ValueError('Epoch budget differs from the published fixed recipe')
    decision = None
    extension_sha = None
    if route == 'T3' and epochs == 30:
        if not args.resume or not args.extension_decision:
            raise ValueError('T3 extension needs the own epoch10 checkpoint and three-seed development decision')
        decision = json.loads(Path(args.extension_decision).read_text())
        extension_sha = sha256(args.extension_decision)
        if decision['target_epochs_for_all_three'] != 30:
            raise ValueError('The fixed group decision does not select30 epochs')
    backend = configure(args.device)
    train, dev = load_rows(args.train_manifest), load_rows(args.dev_manifest)
    if len(train) != 4821 or len(dev) != 1206 or any(r.get('split') != 'train' for r in train) or any(r.get('split') != 'development' for r in dev):
        raise ValueError('Training requires complete fixed train4821/development1206 manifests')
    if set(r['sample_id'] for r in train) & set(r['sample_id'] for r in dev):
        raise ValueError('Train/development overlap')
    data = Images(args.data_root, args.taxonomy)
    output = Path(args.output)
    if not args.resume:
        output.mkdir(parents=True, exist_ok=False)
    elif not output.is_dir():
        raise ValueError('Resume output directory must already exist')
    config = {'route': route, 'seed': seed, 'train_manifest_sha256': sha256(args.train_manifest),
              'dev_manifest_sha256': sha256(args.dev_manifest), 'taxonomy_sha256': sha256(args.taxonomy),
              'epochs': epochs, 'head_lr': 0.0003 if route in ('T2', 'T3') else 0.001,
              'training_order': 'fixed SHA256 sample order', 'batch': 4 if route == 'B1' else 1,
              'accumulation': 2 if route == 'B1' else 1, 'backend': backend}
    cache = backbone = model = head = scheduler = None
    if route == 'B1':
        upernet_identity = model_identity(args.model_root, 'upernet')
        old.seed_all(seed)
        model, initialization = upernet.build_model(args.model_root, seed)
        model.to(args.device).train()
        optimizer, _ = upernet.build_optimizer(model, 1e-4)
        parameters = list(model.parameters())
        initial_hash = initialization['initial_model_state_sha256']
    else:
        if route == 'T1':
            initial = old._native_head(seed, device='cpu').state_dict()
        else:
            native.seed_cpu(seed)
            initial = native.SpatialHead(3072 if route == 'T4' else 128).state_dict()
        initial = {k: v.detach().clone() for k, v in initial.items()}
        initial_hash = native.state_hash(initial)
        if route == 'T3':
            old = history(args.model_root)
            backbone = NativeOnlyBackbone(old, Path(args.data_root), lora=True, seed=seed)
        else:
            if not args.cache:
                raise ValueError('Frozen-backbone training requires the matching lossless cache')
            cache = Cache(args.cache, 'projection-input' if route == 'T4' else 'terminal', train + dev)
        old.seed_all(seed)
        head = old._native_head(seed, device='cpu') if route == 'T1' else native.SpatialHead(3072 if route == 'T4' else 128)
        head.load_state_dict(initial, strict=True)
        head.to(args.device).train()
        optimizer, scheduler, _ = native.optimizer_for(head, route, backbone.transformer if backbone else None, config['head_lr'])
        parameters = list(head.parameters()) + ([p for p in backbone.transformer.parameters() if p.requires_grad] if backbone else [])
    # Binding the DataLoader generator must not reset the model-construction RNG.
    generator = torch.Generator(device='cpu').manual_seed(seed)
    curve, start_epoch, global_update = [], 0, 0
    config['cache_manifest_sha256'] = cache.manifest_sha256 if cache else None
    config['base_model_fingerprint'] = old._supplementary_model_identity['snapshot_fingerprint'] if backbone else (upernet_identity['snapshot_fingerprint'] if route == 'B1' else None)
    if args.resume:
        saved = torch.load(args.resume, map_location='cpu', weights_only=False)
        for key in ('route', 'seed', 'train_manifest_sha256', 'dev_manifest_sha256', 'taxonomy_sha256', 'head_lr', 'cache_manifest_sha256', 'base_model_fingerprint'):
            if saved['config'][key] != config[key]:
                raise ValueError('Resume scientific configuration changed: ' + key)
        if decision is not None:
            original10 = decision['epoch10_checkpoint_sha256'].get(str(seed))
            valid_first = saved['epoch'] == 10 and sha256(args.resume) == original10
            valid_later = saved['epoch'] > 10 and saved.get('extension_decision_sha256') == extension_sha and saved.get('epoch10_checkpoint_sha256') == original10
            if not (valid_first or valid_later):
                raise ValueError('Resume checkpoint is not on the same approved extension chain')
        if saved.get('initialization_sha256') != initial_hash or not saved.get('complete_epoch_boundary'):
            raise ValueError('Resume must use an own complete epoch checkpoint')
        (model if route == 'B1' else head).load_state_dict(saved['model' if route == 'B1' else 'head'], strict=True)
        if backbone:
            old._load_lora_state(backbone.transformer, saved['lora'])
        optimizer.load_state_dict(saved['optimizer'])
        if scheduler:
            scheduler.load_state_dict(saved['scheduler'])
        old.restore_rng_state(saved['rng'], generator)
        start_epoch, global_update, curve = saved['epoch'], saved['global_update'], list(saved['curve'])
        current_record = json.loads(Path(args.resume).with_suffix('.json').read_text())
        if current_record['epoch'] != start_epoch or current_record['checkpoint_sha256'] != sha256(args.resume):
            raise ValueError('Resume sidecar does not match the completed checkpoint')
        if not curve or curve[-1]['epoch'] < start_epoch:
            curve.append(current_record)
    write_json(output / 'recipe.json', config)
    write_json(output / 'initialization.json', {'seed': seed, 'state_sha256': initial_hash, 'paired_routes': ['T2', 'T3'] if route in ('T2', 'T3') else [route]})
    by_id = {r['sample_id']: r for r in train}
    weights = torch.tensor(native.CLASS_WEIGHTS, dtype=torch.float32, device=args.device)

    def forward(row):
        if route == 'B1':
            with torch.autocast('cuda', dtype=torch.bfloat16):
                return model(pixel_values=upernet.normalize_rgb(data.rgb(row)).unsqueeze(0).to(args.device)).logits
        feature = cache.get(row['sample_id']).to(args.device) if cache else backbone.features(data.rgb(row), gradients=False)[0]
        with torch.autocast('cuda', dtype=torch.bfloat16):
            return native.unpatchify(head(feature))

    def evaluate():
        modules = [model] if model else [head] + ([backbone.transformer] if backbone else [])
        modes = [(m, m.training) for top in modules for m in top.modules()]
        rng = old.capture_rng_state(generator)
        cm = np.zeros((3, 3), dtype=np.int64)
        try:
            for top in modules:
                top.eval()
            with torch.no_grad():
                for row in dev:
                    prediction = forward(row).argmax(1).squeeze(0).byte().cpu().numpy()
                    target = data.label(row)
                    valid = target != 255
                    cm += np.bincount((target[valid].astype(np.int64) * 3 + prediction[valid]).ravel(), minlength=9).reshape(3, 3)
        finally:
            for m, mode in modes:
                m.training = mode
            old.restore_rng_state(rng, generator)
        diagonal = np.diag(cm).astype(np.float64)
        union = cm.sum(0) + cm.sum(1) - diagonal
        ious = [float(diagonal[i] / union[i]) for i in range(3) if union[i] > 0]
        if not ious:
            raise ValueError('No defined class union on complete development split')
        return float(np.mean(ious)), cm.tolist()

    frozen_before = native.frozen_fingerprint(backbone.transformer) if backbone else None
    try:
        for epoch in range(start_epoch, epochs):
            rows = [by_id[sid] for sid in stable_order(list(by_id), seed, epoch)]
            groups = upernet.batch_plan(rows, 4, 2) if route == 'B1' else [[r] for r in rows]
            tick = time.monotonic()
            for group in groups:
                optimizer.zero_grad(set_to_none=True)
                if route == 'B1':
                    batches = []
                    for micro in group:
                        pairs = [old._segformer_transform(data.rgb(r), data.label(r), seed=seed, epoch=epoch, sample_id=r['sample_id']) for r in micro]
                        batches.append((torch.stack([p[0] for p in pairs]), torch.stack([p[1] for p in pairs])))
                    denominator = sum(upernet.weighted_denominator(y, torch.tensor(upernet.CLASS_WEIGHTS)).item() for _, y in batches)
                    if not math.isfinite(denominator) or denominator <= 0:
                        raise ValueError('Invalid effective-update weighted CE denominator')
                    upernet.set_lr(optimizer, global_update, 603 * 40)
                    for x, y in batches:
                        with torch.autocast('cuda', dtype=torch.bfloat16):
                            main, aux = upernet.forward_logits(model, x.to(args.device))
                        loss = upernet.loss_numerator(main, aux, y.to(args.device), weights) / denominator
                        if not bool(torch.isfinite(loss)):
                            raise FloatingPointError('Nonfinite UPerNet loss')
                        loss.backward()
                else:
                    row = group[0]
                    labels = torch.from_numpy(data.label(row)).long().unsqueeze(0).to(args.device)
                    feature = cache.get(row['sample_id']).to(args.device) if cache else backbone.features(data.rgb(row), gradients=True)[0]
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        logits = native.unpatchify(head(feature))
                        loss = native.weighted_loss(logits, labels, device=args.device)
                    loss.backward()
                if any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in parameters):
                    raise FloatingPointError('Missing or nonfinite trainable gradient')
                torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
                optimizer.step()
                if scheduler:
                    scheduler.step()
                global_update += 1
            torch.cuda.synchronize()
            training_seconds = time.monotonic() - tick
            score, cm = evaluate()
            state = {'config': config, 'initialization_sha256': initial_hash, 'epoch': epoch + 1,
                     'global_update': global_update, 'optimizer': optimizer.state_dict(),
                     'scheduler': scheduler.state_dict() if scheduler else None,
                     'rng': old.capture_rng_state(generator), 'curve': curve,
                     'complete_epoch_boundary': True,
                     'extension_decision_sha256': extension_sha,
                     'epoch10_checkpoint_sha256': decision['epoch10_checkpoint_sha256'][str(seed)] if decision else None,
                     'model' if route == 'B1' else 'head': (model if model else head).state_dict()}
            if backbone:
                state['lora'] = old._lora_state(backbone.transformer)
            path = output / f'epoch_{epoch + 1:03d}.pt'
            if path.exists():
                raise FileExistsError('Will not overwrite an existing endpoint: ' + str(path))
            torch.save(state, path)
            record = {'seed': seed, 'route': route, 'epoch': epoch + 1, 'split': 'development',
                      'full_dev': True, 'unique_sample_count': 1206, 'pooled_miou': score,
                      'confusion_matrix': cm, 'checkpoint': path.name, 'checkpoint_sha256': sha256(path),
                      'training_seconds': training_seconds}
            curve.append(record)
            # The payload stores preceding curves; the current epoch is recovered from this immutable sidecar.
            write_json(path.with_suffix('.json'), record)
            write_json(output / 'development.json', {'route': route, 'seed': seed, 'curve': curve})
            best = min(curve, key=lambda r: (-r['pooled_miou'], r['epoch']))
            write_json(output / 'endpoints.json', {'selected': best, 'selected_1to10': min([r for r in curve if r['epoch'] <= 10], key=lambda r: (-r['pooled_miou'], r['epoch'])) if epoch + 1 >= 10 else None, 'epoch10': next((r for r in curve if r['epoch'] == 10), None), 'last': record})
        if backbone and native.frozen_fingerprint(backbone.transformer) != frozen_before:
            raise RuntimeError('A frozen transformer parameter changed during joint training')
    finally:
        if backbone:
            backbone.close()
