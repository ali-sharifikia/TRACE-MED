"""Source-anchored phrase orthography, not a synonym or acceptance policy.

Frozen The frozen runtime policy applies. Only single ASCII
hyphens inside a witnessed alphabetic noun phrase become spaces. Every letter,
word order and outside character survives. No punctuation stripping, semantic
derivation, acronym expansion, nerve/nervous equivalence or transitive closure.
"""
from __future__ import annotations
from runtime_resources import resource as asp_resource
import copy
import re
from functools import lru_cache
from pathlib import Path
VERSION = 'phrase_normalization_v1'
DEFAULT_SOURCE = Path(asp_resource('lexicon_phrases.json'))
MAX_PHRASE_WORDS = 8
TOKENS = re.compile('(?<![A-Za-z0-9_+\\-])[A-Za-z]+(?:-[A-Za-z]+)*(?![A-Za-z0-9_+\\-])')
PROTECTED = frozenset('no not non without except excluding anti pre post and or before after during following due versus left right bilateral unilateral positive negative plus minus free total fasting nonfasting ionized conjugated unconjugated high low increased decreased elevated reduced b t nk cd'.split())

def norm(text):
    return ' '.join(text.casefold().split())

def phrase_spans(text):
    """Contiguous windows cannot start/end within a hyphen chain or cross marks."""
    tokens = list(TOKENS.finditer(text))
    result = []
    for start in range(len(tokens)):
        for end in range(start, len(tokens)):
            if end > start and (not text[tokens[end - 1].end():tokens[end].start()].isspace()):
                break
            a, b = (tokens[start].start(), tokens[end].end())
            value = text[a:b]
            words = value.replace('-', ' ').split()
            if len(words) > MAX_PHRASE_WORDS:
                break
            if '-' not in value:
                continue
            result.append({'text': value, 'span': [a, b], 'canonical': value.replace('-', ' '), 'key': norm(value.replace('-', ' ')), 'protected': sorted({w.casefold() for w in words} & PROTECTED)})
    return result

class PhraseNormalizer:

    @classmethod
    def from_file(cls, path):
        import json
        source = json.loads(Path(path).read_text(encoding='utf-8'))
        return cls(source['records'], source_binding=source['bindings'])

    def __init__(self, records, source_binding=None):
        if 'records' in records:
            source_binding = source_binding or records.get('bindings', {})
            records = records['records']
        self.source_binding = copy.deepcopy(source_binding or {})
        self.agreement = {}
        self.spelling = {}
        for row in records.get('LEX/LRAGR', []):
            f = row.get('fields', [])
            if len(f) != 6:
                raise ValueError('Expected official six-column LRAGR record')
            self.agreement.setdefault(norm(f[1]), []).append(copy.deepcopy(row))
        for row in records.get('LEX/LRSPL', []):
            f = row.get('fields', [])
            if len(f) != 3:
                raise ValueError('Expected official three-column LRSPL record')
            self.spelling.setdefault(norm(f[1]), []).append(copy.deepcopy(row))

    @lru_cache(maxsize=65536)
    def view(self, text):
        supported = []
        unresolved = []
        for candidate in phrase_spans(text):
            rows = self.agreement.get(candidate['key'], [])
            if not rows:
                continue
            if candidate['protected']:
                unresolved.append({**candidate, 'reason': 'protected_scope_or_attribute_tokens'})
                continue
            identities = {r['fields'][0] for r in rows}
            if len(identities) != 1 or {r['fields'][2] for r in rows} != {'noun'}:
                unresolved.append({**candidate, 'reason': 'competing_lexical_identity_or_POS', 'LRAGR': rows})
                continue
            eui = next(iter(identities))
            explicit = [r for r in self.spelling.get(norm(candidate['text']), []) if r['fields'][0] == eui and norm(r['fields'][2]) == candidate['key']]
            supported.append({**candidate, 'eui': eui, 'LRAGR': rows, 'LRSPL': explicit, 'route': 'source_explicit_spelling' if explicit else 'derived_noun_phrase_orthography', 'rule': 'single_ASCII_hyphen_to_space_within_verified_noun_phrase', 'source_binding': self.source_binding})
        conflicts = set()
        for i, left in enumerate(supported):
            if any((max(left['span'][0], r['span'][0]) < min(left['span'][1], r['span'][1]) for r in unresolved)):
                conflicts.add(i)
            for j in range(i + 1, len(supported)):
                right = supported[j]
                if max(left['span'][0], right['span'][0]) < min(left['span'][1], right['span'][1]):
                    if left['span'] != right['span'] or left['eui'] != right['eui']:
                        conflicts.update((i, j))
        for index in sorted(conflicts):
            unresolved.append({**supported[index], 'reason': 'overlapping_lexical_phrase_attachments'})
        selected = [c for i, c in enumerate(supported) if i not in conflicts]
        selected.sort(key=lambda r: r['span'])
        chars = list(text)
        for row in selected:
            a, b = row['span']
            if text[a:b] != row['text']:
                raise ValueError('Stale source span')
            chars[a:b] = row['canonical']
        return {'version': VERSION, 'original': text, 'view': ''.join(chars), 'transformations': selected, 'unresolved': unresolved}

    def pair(self, left, right):
        a, b = (self.view(left), self.view(right))
        equal = norm(a['view']) == norm(b['view'])
        verified = equal and (not (a['unresolved'] or b['unresolved']))
        return {'left': left, 'right': right, 'views': [a, b], 'verified': verified, 'authority': 'ordered_complete_phrase_after_source_anchored_orthography' if verified else None, 'proof': {'rule': 'same_complete_ordered_expression'} if verified else None, 'differences': [] if verified else ['unresolved_phrase_relationship' if equal else 'ordered_full_expression_not_established']}

def load_normalizer(path):
    return PhraseNormalizer.from_file(path)

@lru_cache(maxsize=1)
def default_normalizer():
    return load_normalizer(DEFAULT_SOURCE)
