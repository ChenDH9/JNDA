#!/usr/bin/env python3
"""Apply the once-only three-seed10-to30 development-curve rule."""
import argparse

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--curves', nargs=3, required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    from jnda.controls.train import extension_decision
    extension_decision(args.curves, args.output)

if __name__ == '__main__':
    main()
