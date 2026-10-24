"""Paired source/session bootstrap over complete fixed prediction groups."""
import json
import math
from pathlib import Path
import numpy as np
from PIL import Image
from jnda.controls.io import Images, load_rows, sha256, write_json
from . import metrics as M
from . import diagnostic_math as D


def common_draws(frame, split):
    expected = {'primary': (844, 26, 66), 'final': (1163, 40, 100), 'sensitivity': (92, 26, 66)}[split]
    if len(frame) != expected[0] or len({r['sample_id'] for r in frame}) != len(frame):
        raise ValueError('Incomplete or duplicate sampling frame')
    columns = [(source, session) for source in ('ave', 'vwg')
               for session in sorted({r['session_id'] for r in frame if r['source'] == source})]
    counts = [sum(source == s for source, _ in columns) for s in ('ave', 'vwg')]
    if counts != list(expected[1:]) or any(r['source'] not in ('ave', 'vwg') for r in frame):
        raise ValueError('Source/session frame differs from the fixed split')
    rng = np.random.Generator(np.random.PCG64(2026101901))
    weights = np.zeros((10000, len(columns)), dtype=np.uint16)
    for i in range(10000):
        offset = 0
        for count in counts:
            sampled = rng.integers(0, count, size=count)
            weights[i, offset:offset + count] = np.bincount(sampled, minlength=count)
            offset += count
    return weights, columns


def metric_values(features, diagnostics):
    features = np.asarray(features, dtype=np.float64)
    cm = features[:, :9].reshape(-1, 3, 3)
    diagonal = np.diagonal(cm, axis1=1, axis2=2)
    union = cm.sum(1) + cm.sum(2) - diagonal
    iou = D.safe_ratio(diagonal, union)
    valid = np.isfinite(iou)
    miou = D.safe_ratio(np.where(valid, iou, 0).sum(1), valid.sum(1))
    values = D.metrics_from_features(features)
    if not diagnostics:
        values = values[:, :12]
    return np.column_stack([miou, values])


def jsonable(value):
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(value) else None
    return value


def analyze(args):
    aliases = json.loads(Path(args.system_aliases).read_text()) if args.system_aliases else {}
    frame = load_rows(args.truth_manifest)
    if args.panel_ids:
        wanted = Path(args.panel_ids).read_text().splitlines()
        if len(wanted) != len(set(wanted)):
            raise ValueError('Duplicate panel identifiers')
        index = {r['sample_id']: r for r in frame}
        frame = [index[s] for s in wanted]
    draws, columns = common_draws(frame, args.split)
    session_index = {key: i for i, key in enumerate(columns)}
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    data = Images(args.data_root, args.taxonomy)
    truth = {r['sample_id']: data.label(r) for r in frame}
    groups, per_image, summaries, boot = {}, {}, [], {}
    names = ['pooled_miou'] + (D.METRICS if args.diagnostics else D.METRICS[:12])
    expected_ids = {r['sample_id'] for r in frame}
    for index_path in args.predictions:
        path = Path(index_path)
        binding = json.loads(path.read_text())
        key = (aliases.get(binding['system'], binding['system']), int(binding['seed']))
        if key in groups:
            raise ValueError('Duplicate system/seed prediction group')
        by_id = {r['sample_id']: r for r in binding['rows']}
        if len(by_id) != len(binding['rows']) or set(by_id) != expected_ids:
            raise ValueError('Prediction group does not exactly cover the frozen frame')
        image_features, session_features = [], np.zeros((len(columns), 57), dtype=np.float64)
        for row in frame:
            predrow = by_id[row['sample_id']]
            q = (path.parent / predrow.get('path', predrow.get('prediction_path'))).resolve(strict=True)
            if sha256(q) != predrow.get('sha256', predrow.get('prediction_sha256')):
                raise ValueError('Prediction SHA256 mismatch')
            with Image.open(q) as im:
                prediction = np.asarray(im)
            target = truth[row['sample_id']]
            if prediction.shape != target.shape or prediction.dtype != np.uint8:
                raise ValueError('Prediction size/dtype changed; predictions are never resized')
            cm = M.confusion_from_synthetic_arrays(prediction, target)
            if args.diagnostics:
                record = {'confusion_matrix': cm,
                          'boundary_counts': [M.boundary_counts(prediction, target, c, r) for c, r in D.BOUND_KEYS],
                          'weed_patches': M.weed_patch_records(prediction, target, sample_id=row['sample_id'], source=row['source'], session_id=row['session_id'])}
                vector = D.feature_row(record)
            else:
                vector = np.zeros(57)
                vector[:9] = np.asarray(cm).ravel()
            session_features[session_index[(row['source'], row['session_id'])]] += vector
            image_features.append(vector)
        points = metric_values(session_features.sum(0, keepdims=True), args.diagnostics)[0]
        groups[key] = points
        per_image[key] = np.asarray(image_features)
        boot[key] = metric_values(draws @ session_features, args.diagnostics)
        summaries.append({'system': key[0], 'seed': key[1], 'prediction_index_sha256': sha256(path),
                          'metrics': dict(zip(names, points)), 'confusion_matrix': session_features.sum(0)[:9].reshape(3, 3).astype(np.int64).tolist(),
                          'denominators': D.metric_denominators(session_features.sum(0)) if args.diagnostics else None})
    specs = json.loads(Path(args.comparisons).read_text())
    if isinstance(specs, dict):
        specs = specs['comparisons']
    contrasts = []
    for spec in specs:
        left, right, seeds = spec['left'], spec['right'], spec['seeds']
        if len(set(seeds)) != len(seeds) or not seeds:
            raise ValueError('Comparison seeds must be explicit and unique')
        if any((s, seed) not in groups for s in (left, right) for seed in seeds):
            raise ValueError('Missing one or more prescribed paired operands')
        point = np.stack([groups[(left, seed)] - groups[(right, seed)] for seed in seeds]).mean(0)
        values = np.stack([boot[(left, seed)] - boot[(right, seed)] for seed in seeds]).mean(0)
        result = []
        for i, name in enumerate(names):
            ci = D.conditional_CI(values[:, i])
            if not np.isfinite(point[i]):
                ci.update(low=None, high=None, NA_reason='A_PRESCRIBED_COMPLETE_SEED_OPERAND_IS_UNDEFINED')
            result.append({'metric': name, 'equal_seed_mean_left_minus_right': point[i],
                           'per_seed_differences': {str(seed): groups[(left, seed)][i] - groups[(right, seed)][i] for seed in seeds}, **ci})
        contrasts.append({'id': spec['id'], 'left': left, 'right': right, 'seeds': seeds, 'results': result})
        np.savez_compressed(output / (spec['id'] + '_bootstrap.npz'), values=values)
    np.savez_compressed(output / 'shared_session_draws.npz', weights=draws)
    write_json(output / 'analysis.json', jsonable({'split': args.split, 'frame_images': len(frame),
               'sessions': len(columns), 'draw_seed': 2026101901, 'draws': 10000,
               'sampling_unit': 'whole source-stratified recording session, shared across models,seeds,metrics',
               'interval_scope': 'session composition conditional on the fixed trained seeds; not retraining uncertainty',
               'seed_points': summaries, 'comparisons': contrasts}))
    if args.case_comparison:
        if not args.diagnostics:
            raise ValueError('Case selection needs --diagnostics')
        spec = next(s for s in specs if s['id'] == args.case_comparison)
        a, b = (spec['left'], 1701), (spec['right'], 1701)
        av, bv = metric_values(per_image[a], True), metric_values(per_image[b], True)
        records = []
        for i, row in enumerate(frame):
            difference = av[i, 0] - bv[i, 0]
            records.append({**row, 'index': i, 'difference': float(difference) if np.isfinite(difference) else None,
                            'small_pixels': int(per_image[a][i, 46]),
                            'small_recall': float(av[i, names.index('small_weed_pixel_recall')]) if np.isfinite(av[i, names.index('small_weed_pixel_recall')]) else None})
        selected, audit, shortfalls = D.select_cases(records)
        write_json(output / 'diagnostic_cases.json', jsonable({'selected': selected, 'audit': audit, 'shortfalls': shortfalls}))
