"""Direct full-span NLM boundary spellings after the frozen v1 phrase view.

PHRASE_EVIDENCE_CONSISTENCY_V1_ITERATION_2.md applies. This is not synonymy:
letters, order, scope and specificity cannot change. LRSPL supplies a direct
alias/citation witness; LRAGR independently supplies its singular noun identity.
No source-pair or benchmark exceptions, transitive lookup or acceptance authority.
"""
from __future__ import annotations
from runtime_resources import resource as asp_resource
import copy
import json
import re
from functools import lru_cache
from pathlib import Path
import phrase_lexicon as previous
VERSION = 'phrase_normalization_v2'
DEFAULT_SOURCE = previous.DEFAULT_SOURCE
norm = previous.norm
FORM = re.compile('[A-Za-z]+(?:[ -][A-Za-z]+)*\\Z')

def letters(text):
    return ''.join(re.findall('[A-Za-z]', text)).casefold()

def protected(text):
    return sorted(set(re.findall('[a-z]+', text.casefold())) & previous.PROTECTED)

def overlap(a, b):
    return max(a[0], b[0]) < min(a[1], b[1])

class PhraseNormalizer:

    def __init__(self, records, source_binding=None):
        if 'records' in records:
            source_binding = source_binding or records.get('bindings', {})
            records = records['records']
        self.source_binding = copy.deepcopy(source_binding or {})
        self.previous = previous.PhraseNormalizer(records, source_binding=self.source_binding)
        self.agreement = self.previous.agreement
        self.rules = {}
        self.invalid = {}
        for key, rows in self.previous.spelling.items():
            pairs = [(r, norm(r['fields'][2])) for r in rows if FORM.fullmatch(key) and FORM.fullmatch(norm(r['fields'][2])) and (letters(key) == letters(r['fields'][2])) and (key != norm(r['fields'][2]))]
            if not pairs:
                continue
            reason = None
            agr = self.agreement.get(key, [])
            all_euis = {r['fields'][0] for r in rows}
            targets = {norm(r['fields'][2]) for r in rows}
            if len(all_euis) != 1 or len(targets) != 1:
                reason = 'competing_direct_spelling_identity_or_target'
            elif not agr or {r['fields'][0] for r in agr} != all_euis or {r['fields'][2] for r in agr} != {'noun'} or any(('thr_sing' not in r['fields'][3] for r in agr)):
                reason = 'missing_or_ambiguous_singular_noun_identity'
            elif protected(key) or any((protected(target) for _, target in pairs)):
                reason = 'protected_scope_or_attribute_tokens'
            elif max((len(re.findall('[A-Za-z]+', text)) for text in [key, *targets])) > previous.MAX_PHRASE_WORDS:
                reason = 'bounded_phrase_length_exceeded'
            elif len(pairs) != len(rows):
                reason = 'non_boundary_spelling_alternative'
            target = sorted(targets)[0] if len(targets) == 1 else None
            rule = {'alias': key, 'canonical': target, 'LRAGR': copy.deepcopy(agr), 'LRSPL': copy.deepcopy(rows), 'eui': next(iter(all_euis)) if len(all_euis) == 1 else None, 'source_binding': self.source_binding}
            if reason:
                self.invalid[key] = {**rule, 'reason': reason}
            else:
                self.rules[key] = rule
        aliases = sorted(set(self.rules) | set(self.invalid), key=lambda x: (-len(x), x))
        self.patterns = [(key, re.compile('(?<![A-Za-z0-9_+\\-])' + re.escape(key).replace('\\ ', '\\s+') + '(?![A-Za-z0-9_+\\-])', re.I)) for key in aliases]

    @classmethod
    def from_file(cls, path):
        source = json.loads(Path(path).read_text(encoding='utf-8'))
        return cls(source['records'], source_binding=source['bindings'])

    @lru_cache(maxsize=65536)
    def view(self, text):
        first = self.previous.view(text)
        stage = first['view']
        candidates, unresolved = ([], copy.deepcopy(first['unresolved']))
        for alias, pattern in self.patterns:
            for match in pattern.finditer(stage):
                a, b = match.span()
                common = {'text': text[a:b], 'original_text': text[a:b], 'matched_view_text': stage[a:b], 'span': [a, b], 'view_span_before_replacement': [a, b]}
                if alias in self.invalid:
                    unresolved.append({**self.invalid[alias], **common})
                    continue
                rule = copy.deepcopy(self.rules[alias])
                candidates.append({**rule, **common, 'route': 'source_explicit_phrase_boundary_spelling', 'rule': 'direct_LRSPL_identical_ordered_letters_singular_noun', 'preceding_phrase_proofs': [r for r in first['transformations'] if overlap(r['span'], [a, b])]})
        conflicts = set()
        for i, left in enumerate(candidates):
            if any((overlap(left['span'], r['span']) for r in unresolved)):
                conflicts.add(i)
            for j in range(i + 1, len(candidates)):
                right = candidates[j]
                if overlap(left['span'], right['span']):
                    conflicts.update((i, j))
        for i in sorted(conflicts):
            unresolved.append({**candidates[i], 'reason': 'overlapping_explicit_phrase_attachments'})
        if unresolved:
            return {'version': VERSION, 'original': text, 'view': text, 'transformations': [], 'unresolved': unresolved, 'source_view': first, 'offset_map': [[i, i + 1] for i in range(len(text))]}
        selected = sorted(candidates, key=lambda r: r['span'])
        result, offsets, emitted = ([], [], [])
        cursor = 0
        for row in selected:
            a, b = row['span']
            target = row['canonical']
            result.extend(stage[cursor:a])
            offsets.extend([[i, i + 1] for i in range(cursor, a)])
            start = len(result)
            positions = [a + m.start() for m in re.finditer('[A-Za-z]', text[a:b])]
            assert letters(text[a:b]) == letters(target)
            position = 0
            for char in target:
                if char.isalpha():
                    p = positions[position]
                    offsets.append([p, p + 1])
                    position += 1
                else:
                    left = positions[position - 1] + 1 if position else a
                    right = positions[position] if position < len(positions) else b
                    offsets.append([left, right])
                result.append(char)
            emitted.append({**row, 'output_span': [start, len(result)]})
            cursor = b
        result.extend(stage[cursor:])
        offsets.extend([[i, i + 1] for i in range(cursor, len(text))])
        return {'version': VERSION, 'original': text, 'view': ''.join(result), 'transformations': first['transformations'] + emitted, 'unresolved': [], 'source_view': first, 'offset_map': offsets}

    def pair(self, left, right):
        a, b = (self.view(left), self.view(right))
        verified = norm(a['view']) == norm(b['view']) and (not (a['unresolved'] or b['unresolved']))
        return {'left': left, 'right': right, 'views': [a, b], 'verified': verified, 'authority': 'direct_source_boundary_spelling_complete_expression' if verified else None, 'proof': {'rule': 'same_complete_ordered_expression'} if verified else None, 'differences': [] if verified else ['ordered_full_expression_or_identity_unestablished']}

def load_normalizer(path):
    return PhraseNormalizer.from_file(path)

@lru_cache(maxsize=1)
def default_normalizer():
    return load_normalizer(DEFAULT_SOURCE)
