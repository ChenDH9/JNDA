#!/usr/bin/env python3
"""Build a lossless fixed-backbone train/development feature cache."""
import argparse

def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['data-root', 'train-manifest', 'dev-manifest', 'model-root', 'output']:
        p.add_argument('--' + name, required=True)
    p.add_argument('--feature', choices=['terminal', 'projection-input'], required=True)
    p.add_argument('--device', default='cuda:0')
    args = p.parse_args()
    from jnda.controls.cache import build
    build(args)

if __name__ == '__main__':
    main()
