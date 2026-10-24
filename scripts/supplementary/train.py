#!/usr/bin/env python3
"""Train one supplementary control with the fixed published recipe."""
import argparse

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--route', choices=['T1', 'T2', 'T3', 'T4', 'B1'], required=True)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--data-root', required=True)
    p.add_argument('--train-manifest', required=True)
    p.add_argument('--dev-manifest', required=True)
    p.add_argument('--taxonomy', required=True)
    p.add_argument('--model-root', help='Local FLUX snapshot for T3, or UPerNet snapshot for B1')
    p.add_argument('--cache', help='Lossless terminal or projection-input cache')
    p.add_argument('--output', required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--epochs', type=int)
    p.add_argument('--resume')
    p.add_argument('--extension-decision')
    args = p.parse_args()
    if args.route in ('T3', 'B1') and not args.model_root:
        p.error('--model-root is required for this route')
    from jnda.controls.train import run
    run(args)

if __name__ == '__main__':
    main()
