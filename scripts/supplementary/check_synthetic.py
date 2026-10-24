#!/usr/bin/env python3
"""Check metric identities and resampling using synthetic arrays only."""
import numpy as np
from jnda.evaluation import metrics as M
from jnda.evaluation import diagnostic_math as D
from jnda.evaluation.analyze import common_draws, metric_values
from jnda.controls.protocol import t3_group_extension


def main():
    g = np.zeros((32, 32), dtype=np.uint8)
    g[3:10, 3:10] = 1
    g[17:20, 17:20] = 2
    for c in (1, 2):
        for r in (1, 2, 4):
            assert M.boundary_counts(g, g, c, r)['scores']['f1'] == 1
    p = g.copy()
    p[17:20, 17:20] = 0
    p[17:19, 17:19] = 2
    patch = M.weed_patch_records(p, g, sample_id='synthetic', source='ave', session_id='synthetic-session')
    assert patch['records'][0]['area_pixels'] == 9
    assert patch['records'][0]['covered_pixels'] == 4
    assert patch['records'][0]['tiny'] and not patch['records'][0]['coverage_at_least_half']
    ignored = np.full_like(g, 255)
    assert M.boundary_counts(p, ignored, 2, 2)['scores']['f1'] is None
    try:
        M.confusion_from_synthetic_arrays(np.full_like(g, 255), g)
    except Exception:
        pass
    else:
        raise AssertionError('Invalid prediction class was accepted')
    a = np.zeros((1, 57)); a[0, 0] = 50
    assert metric_values(a, False)[0, 0] == 1
    frame = [{'sample_id': f'synthetic-{i}', 'source': 'ave' if i < 26 else 'vwg',
              'session_id': f'session-{i}'} for i in range(92)]
    draws, cols = common_draws(frame, 'sensitivity')
    assert draws.shape == (10000, 92)
    assert (draws[:, :26].sum(1) == 26).all() and (draws[:, 26:].sum(1) == 66).all()
    assert np.array_equal(draws, common_draws(frame, 'sensitivity')[0])
    ci = D.conditional_CI(np.zeros(10000))
    assert ci['low'] == ci['high'] == 0 and ci['NA_draws'] == 0
    values = np.zeros(10000); values[0] = np.nan
    assert D.conditional_CI(values)['low'] is None
    def curves(delta):
        return {seed: [{'seed': seed, 'epoch': epoch, 'split': 'development', 'full_dev': True,
                        'unique_sample_count': 1206, 'pooled_miou': 0.5 + (delta if epoch > 5 else 0),
                        'checkpoint_sha256': '0' * 64} for epoch in range(1, 11)] for seed in (1701, 1702, 1703)}
    assert t3_group_extension(curves(0))['target_epochs_for_all_three'] == 10
    assert t3_group_extension(curves(0.01))['target_epochs_for_all_three'] == 30
    print('Synthetic metric, NA, session resampling and fixed budget checks passed.')

if __name__ == '__main__':
    main()
