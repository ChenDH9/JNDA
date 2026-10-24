#!/usr/bin/env python3
"""Aggregate complete resource cells without mixing main and extra blocks."""
import argparse
import json
import statistics
from pathlib import Path

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cells', nargs='+', required=True)
    p.add_argument('--expected-seeds', nargs='+', type=int, required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    cells = [json.loads(Path(x).read_text()) for x in args.cells]
    if len({(x['route'], x['steps']) for x in cells}) != 1:
        raise ValueError('Aggregate one route/configuration at a time')
    main = [x for x in cells if x['panel_kind'] == 'main']
    if sorted(x['seed'] for x in main) != sorted(args.expected_seeds):
        raise ValueError('Exactly one main cell per prescribed seed is required')
    for cell in cells:
        records = cell['records']
        expected = 40 if cell['panel_kind'] == 'main' else 20
        ids = {r['sample_id'] for r in records}
        if len(ids) != expected or len(records) != expected * 3 or any(sorted(r['repeat'] for r in records if r['sample_id'] == sid) != [1, 2, 3] for sid in ids):
            raise ValueError('Incomplete or duplicated timing records')
    extra = [x for x in cells if x['panel_kind'] != 'main']
    if sorted(x['panel_kind'] for x in extra) != ['extra-1', 'extra-2'] or any(x['seed'] != 1701 for x in extra):
        raise ValueError('Two independent extra20-image blocks for seed1701 are required')
    first = next(x for x in main if x['seed'] == 1701)
    common_ids = {r['sample_id'] for r in extra[0]['records']}
    if {r['sample_id'] for r in extra[1]['records']} != common_ids or not common_ids <= {r['sample_id'] for r in first['records']}:
        raise ValueError('Extra blocks must use the same frozen20-image subset of main')
    def panel(cell, ids):
        return statistics.median([statistics.median([r['seconds'] for r in cell['records'] if r['sample_id'] == sid]) for sid in ids])
    latencies = [panel(first, common_ids)] + [panel(x, common_ids) for x in extra]
    output = {'route': main[0]['route'], 'steps': main[0]['steps'], 'seeds': args.expected_seeds,
              'latency_seconds': statistics.median([x['panel_latency_seconds'] for x in main]),
              'serial_throughput': sum(len(x['records']) for x in main) / sum(r['seconds'] for x in main for r in x['records']),
              'peak_allocated_bytes': max(x['peak_allocated_bytes'] for x in main),
              'peak_reserved_bytes': max(x['peak_reserved_bytes'] for x in main),
              'common20_three_block_latencies': latencies, 'common20_descriptive_range': [min(latencies), max(latencies)],
              'range_is_confidence_interval': False}
    Path(args.output).write_text(json.dumps(output, indent=2) + '\n')

if __name__ == '__main__':
    main()
