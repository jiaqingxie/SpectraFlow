"""Entry point for auditable reviewer experiments; neural work defaults to CUDA."""
import argparse
import json
from pathlib import Path

from reviewer_experiments.common import OUT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare','train','status'])
    parser.add_argument('--output', type=Path, default=OUT)
    parser.add_argument('--device', choices=['cuda','cpu'], default='cuda')
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--run', help='Exact run name in run_manifest.json')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--budget-seconds', type=float, help='Explicit equal GPU-time training budget')
    args = parser.parse_args()
    if args.command == 'prepare':
        from reviewer_experiments.prepare import prepare
        prepare(args.output)
    elif args.command == 'train':
        from reviewer_experiments.training import train
        jobs = json.loads((args.output/'run_manifest.json').read_text())
        matches = [j for j in jobs if j['name']==args.run]
        if len(matches) != 1:
            parser.error('--run must select exactly one prepared run')
        train(matches[0], args)
    else:
        for name in ('preparation_summary.json',):
            path = args.output/name
            print(path.name, path.read_text() if path.exists() else 'not available')


if __name__ == '__main__':
    main()
