"""Explicit private resource configuration; no embedded terminology or credentials."""
import hashlib
import json
import os
from functools import lru_cache
from pathlib import Path
ROOT = Path(__file__).resolve().parent
LOCAL_PATHS = {
    'resource_root': 'resources/meddra',
    'lexicon_words.json': 'resources/lexicon/words.json',
    'lexicon_phrases.json': 'resources/lexicon/phrases.json',
    'lexicon_nominalizations.json': 'resources/lexicon/nominalizations.json',
    'lexicon_agreement.json': 'resources/lexicon/agreement.json',
    'embedding_checkpoint': 'models/BioLORD-2023',
    'training_terms': 'data/training/terms.csv',
    'training_cases': 'data/training/cases.csv',
    'test_terms': 'data/test/terms.csv',
    'test_cases': 'data/test/cases.csv',
}

def load(path):
    with Path(path).open(encoding='utf-8-sig') as f:
        return json.load(f)

def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

@lru_cache(maxsize=1)
def configuration():
    p = Path(os.environ.get('ASPMT_CONFIG', ROOT / 'config.local.json')).resolve()
    if not p.is_file():
        raise FileNotFoundError('Run configure.py first, or set ASPMT_CONFIG to your private JSON configuration.')
    c = load(p)
    for name in ('resource_root', 'lexicon_words.json', 'lexicon_phrases.json', 'lexicon_nominalizations.json', 'lexicon_agreement.json', 'embedding_checkpoint'):
        if name not in c:
            raise ValueError('Missing resource configuration: ' + name)
    for name in LOCAL_PATHS:
        if name in c:
            value = Path(c[name])
            c[name] = str((value if value.is_absolute() else p.parent / value).resolve())
    return c

def resource(name=''):
    c = configuration()
    return Path(c[name]) if name in c else Path(c['resource_root']) / name

@lru_cache(maxsize=1)
def implementation():
    """Fresh release bindings replace development-report hashes, not clinical checks."""
    out = {p.name: sha(p) for p in sorted(ROOT.glob('*.py'))}
    for name in ('meddra_exact_reference.csv', 'llt.asc', 'pt.asc', 'lexicon_words.json', 'lexicon_phrases.json', 'lexicon_nominalizations.json', 'lexicon_agreement.json'):
        out['resource:' + name] = sha(resource(name))
    lock = load(ROOT / 'resources.lock.json')
    for name, expected in lock.items():
        if out['resource:' + name] != expected:
            raise ValueError('Changed frozen resource: ' + name)
    out['resources.lock.json'] = sha(ROOT / 'resources.lock.json')
    out['model.lock.json'] = sha(ROOT / 'model.lock.json')
    out['embedding.lock.json'] = sha(ROOT / 'embedding.lock.json')
    out['release'] = 'ASP-MT-1.0.0-threshold-T'
    return out

@lru_cache(maxsize=1)
def embedding_checkpoint():
    lock = load(ROOT / 'embedding.lock.json')
    path = Path(configuration()['embedding_checkpoint'])
    for name, h in lock['files'].items():
        if sha(path / name) != h:
            raise ValueError('Changed local encoder file: ' + name)
    return path

def bootstrap():
    import sys
    sys.dont_write_bytecode = True
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    os.environ.setdefault('HF_HUB_OFFLINE', '1')
    os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')

def save_new(path, data):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('x', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
