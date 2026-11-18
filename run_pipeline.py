"""Run ASP-MT on a JSON list of {id, text, frequency?} objects."""
import argparse
from pathlib import Path

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--directory', type=Path, required=True)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--record-decisions', action='store_true', help='Write observational TRACE decision events; no decision changes')
    p.add_argument('--env-file', type=Path, help='Private credentials, loaded only for live execution')
    a = p.parse_args()
    from runtime_resources import load
    from pipeline import run
    return run(load(a.input), a.directory, resume=a.resume, env_file=a.env_file, record_decisions=a.record_decisions)
if __name__ == '__main__':
    raise SystemExit(main())
