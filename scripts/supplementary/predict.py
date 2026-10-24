#!/usr/bin/env python3
"""Generate categorical predictions or measure one frozen resource panel."""
import argparse

def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['data-root', 'manifest', 'model-root', 'output']:
        p.add_argument('--' + name, required=True)
    p.add_argument('--route', choices=['T1', 'T2', 'T3', 'T4', 'B1', 'Fresh_S7', 'Fresh_Arm_B'], required=True)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--system', help='Evaluation group name, including endpoint identity')
    p.add_argument('--checkpoint')
    p.add_argument('--panel-ids', help='Ordered subset of IDs from the explicit input manifest')
    p.add_argument('--legacy-output', help='Original fixed-epoch checkpoint output tree for sensitivity')
    p.add_argument('--prompt', choices=['original', 'short', 'empty'], default='original')
    p.add_argument('--steps', type=int, default=10)
    p.add_argument('--timestep', type=float, default=0.999)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--timing', action='store_true')
    p.add_argument('--panel-kind', choices=['main', 'extra-1', 'extra-2'], default='main')
    args = p.parse_args()
    from jnda.evaluation.predict import run
    run(args)

if __name__ == '__main__':
    main()
