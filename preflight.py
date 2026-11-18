"""Offline installation, resource and frozen-configuration checks; no service calls."""
import argparse
import importlib.metadata
from pathlib import Path
from runtime_resources import ROOT, bootstrap, implementation, resource, load, embedding_checkpoint, sha, configuration, LOCAL_PATHS

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.parse_args()
    bootstrap()
    failures = []
    config = configuration()
    for name in LOCAL_PATHS:
        if name not in config:
            failures.append('Missing standalone resource setting: ' + name)
        elif not Path(config[name]).is_relative_to(ROOT):
            failures.append('Standalone resource is outside this folder: ' + name)
        elif not Path(config[name]).exists():
            failures.append('Missing standalone resource: ' + name)
    import sys
    if not Path(sys.prefix).resolve().is_relative_to(ROOT):
        failures.append('Use this installation\'s .venv/Scripts/python.exe')
    if (ROOT / 'release.json').exists():
        for name, expected in load(ROOT / 'release.json')['files'].items():
            if sha(ROOT / name) != expected:
                failures.append('Changed release file: ' + name)
    for line in (ROOT / 'requirements.txt').read_text().splitlines():
        if not line or line.startswith('#'):
            continue
        name, expected = line.split('==')
        try:
            actual = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            actual = 'not installed'
        if actual != expected:
            failures.append(f'{name}: expected {expected}, found {actual}')
    for name in ('meddra_exact_reference.csv', 'llt.asc', 'pt.asc', 'lexicon_words.json', 'lexicon_phrases.json', 'lexicon_nominalizations.json', 'lexicon_agreement.json', 'meddra_vector_db_config.json'):
        if not resource(name).is_file():
            failures.append('Missing resource: ' + name)
    if not resource('meddra_vector_db').is_dir():
        failures.append('Missing MedDRA vector database')
    if not failures:
        from lexical_equivalence_evaluation import Reference
        ref = Reference(resource('meddra_exact_reference.csv'), resource('llt.asc'), resource('pt.asc'))
        from acceptance_policy import policy
        from pipeline import SETTINGS
        bound = implementation()
        embedding_checkpoint()
        lock = load(ROOT / 'resources.lock.json')
        for name, expected in lock.items():
            if bound.get('resource:' + name) != expected:
                failures.append('Changed frozen resource: ' + name)
        print('Current LLTs:', len(ref.rows), '| Reference release: unknown (hash-bound)')
        print('Policy: threshold T; score floor 0; actual margin > 1e-10; independent guards retained')
        print('Model:', SETTINGS['model'], '| Context:', SETTINGS['num_ctx'], '| Live calls: 0')
    for message in failures:
        print('FAIL:', message)
    if not failures:
        print('Offline preflight passed. Service availability/model digest are checked only during run.')
    return int(bool(failures))
if __name__ == '__main__':
    raise SystemExit(main())
