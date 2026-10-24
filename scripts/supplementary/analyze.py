#!/usr/bin/env python3
"""Evaluate fixed prediction files with paired source/session intervals."""
import argparse

def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['data-root', 'truth-manifest', 'taxonomy', 'comparisons', 'output']:
        p.add_argument('--' + name, required=True)
    p.add_argument('--predictions', nargs='+', required=True, help='One predictions.json per system/seed')
    p.add_argument('--split', choices=['primary', 'final', 'sensitivity'], required=True)
    p.add_argument('--panel-ids')
    p.add_argument('--system-aliases', help='Explicit name-only mapping for the original diagnostic groups')
    p.add_argument('--diagnostics', action='store_true', help='Also compute reliable boundaries and small weed patches')
    p.add_argument('--case-comparison', help='Comparison ID used by the fixed seed1701 diagnostic selector')
    args = p.parse_args()
    from jnda.evaluation.analyze import analyze
    analyze(args)

if __name__ == '__main__':
    main()
