"""Create a relative-path configuration for resources inside this installation."""
import argparse
from pathlib import Path
from runtime_resources import ROOT, LOCAL_PATHS, save_new


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'config.local.json')
    args = parser.parse_args()
    if args.output.resolve().parent != ROOT:
        parser.error('Keep the configuration beside the pipeline files')
    for name, relative in LOCAL_PATHS.items():
        if not (ROOT/relative).exists():
            raise FileNotFoundError(f'Place the required {name} at {ROOT/relative}')
    save_new(args.output, LOCAL_PATHS)
    print('Local relative-path configuration saved:', args.output)
    print('No connection to the original development project is required.')


if __name__ == '__main__':
    main()
