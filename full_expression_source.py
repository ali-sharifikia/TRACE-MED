"""Frozen gold-free full-expression feasibility rules, 2026-10-04.

This is a pure comparison helper, NOT an acceptance policy or scoring rule.
Protocol (frozen before outcome scoring):
* L: explicit locative attachment of a complete, ordered, single symptom finding
  to one unchanged anatomical expression. Only sensory/swelling/bleeding heads
  from the existing grammar are allowed, not tumour/disease/toxicity/death heads.
  Prefix/suffix anatomy is compared with explicit `in/of` anatomy; no body-part
  synonyms, attachment flattening, lost modifiers or unspecified laterality.
* N: NLM LRNOM-attested -ness quality noun in `QUALITY of/in ARGUMENT` versus its
  adjective predication of exactly the same ordered argument. Other derivations
  (severity/severe, process/result, noun/verb, agent/patient) are NOT equivalence.
  All canonicalization and adjective predication requires complete LRAGR
  observations establishing adjective-only POS; a noun modifier such as
  `light reaction` is not parsed as asserted lightness, even if the input
  explicitly uses the nominal quality `lightness`.
* Numbers, negation, uncertain assertion, causal/temporal/composite relationships
  block both rules. Existing clinical attribute differences also block. Clinical
  parser's attachment-only uncertainty may be resolved ONLY by a positive frame.
* No code/parent/gold input, fuzzy similarity, thresholds, selection or closure.
* Original text/spans, source rows and exact transformation remain inspectable.
* No hypothetical proof counts as accepted mapping; integration must recheck all
  independent authority, reference, competing meaning and numerical requirements.

Novelty: existing v1.7 rotates a single finding head but cannot rotate anatomy
around an internally modified finding. Earlier LRNOM work was retrieval-only.
"""
from __future__ import annotations
from runtime_resources import resource as asp_resource
import copy
import re
from functools import lru_cache
from clinical_specificity import compare
from compound_validation import ANATOMY
from grammatical_discrimination import HEADS
VERSION = 'full_expression_source_v1'
LOCATIVE_HEADS = frozenset(HEADS & {'pain', 'sensation', 'paraesthesia', 'paresthesia', 'swelling', 'edema', 'haemorrhage', 'hemorrhage'})
UNSAFE = re.compile('[^A-Za-z\\s-]|\\b(?:no|not|non|without|except|excluding|possible|possibly|suspected|probable|uncertain|may|might|history|and|or|with|due|to|from|after|before|following|during|since|beyond|into|through|caused|related|normal|abnormal|high|low|increased|decreased|more|less|degree|extent|amount|state|level|levels|count|concentration|percentage|activity|duration|NOS|unspecified)\\b', re.I)
ATTACHMENT_ONLY = {'full_expression_attachment_unverified'}

def norm(value):
    return ' '.join(value.casefold().split())

def fragment(text, a, b):
    return {'text': text[a:b], 'span': [a, b]}

def _simple_finding(text):
    if UNSAFE.search(text) or re.search('\\b(?:of|in)\\b', text, re.I):
        return False
    words = norm(text).split()
    return sum((w in LOCATIVE_HEADS for w in words)) == 1 and words[-1] in LOCATIVE_HEADS and (not any((w in ANATOMY for w in words)))

@lru_cache(maxsize=65536)
def locative_frames(text):
    """Do not infer an implicit anatomical synonym or reorder finding tokens."""
    if UNSAFE.search(text):
        return ()
    atoms = '(?:' + '|'.join((re.escape(x) for x in sorted(ANATOMY, key=lambda x: (-len(x), x)))) + ')'
    patterns = (('explicit_locative', f'(?P<finding>.+?)\\s+(?:in|of)\\s+(?:the\\s+)?(?P<site>{atoms})'), ('site_prefix', f'(?P<site>{atoms})\\s+(?P<finding>.+)'), ('site_suffix', f'(?P<finding>.+?)\\s+(?P<site>{atoms})'))
    frames = []
    for rule, pattern in patterns:
        m = re.fullmatch(pattern, text, re.I)
        if not m or not _simple_finding(m['finding']):
            continue
        frames.append({'rule': rule, 'key': [norm(m['finding']), norm(m['site'])], 'finding': fragment(text, *m.span('finding')), 'site': fragment(text, *m.span('site')), 'head': norm(m['finding']).split()[-1], 'authority': 'bounded_locative_grammar; existing ANATOMY and clinical HEADS', 'retained_finding_order': True})
    return tuple(frames)

class FullExpressionProof:
    """Optional source records must be exact saved LRNOM rows, not synonyms."""

    def __init__(self, nominalization_rows=(), source_binding=None, agreement_rows=()):
        self.source_binding = copy.deepcopy(source_binding or {})
        self.qualities = {}
        self.agreement = {}
        for row in agreement_rows:
            f = row.get('fields', [])
            if len(f) != 6:
                raise ValueError('Expected official six-column LRAGR row')
            self.agreement.setdefault(norm(f[1]), []).append(copy.deepcopy(row))
        for row in nominalization_rows:
            f = row.get('fields', [])
            if len(f) != 6:
                raise ValueError('Expected official six-column LRNOM row')
            if f[2] == 'noun' and f[5] == 'adj' and norm(f[1]).endswith('ness') and re.fullmatch('[a-z]+', norm(f[1])) and re.fullmatch('[a-z]+', norm(f[4])):
                self.qualities.setdefault(norm(f[1]), []).append(copy.deepcopy(row))
        self.quality_words = {}
        for noun, rows in self.qualities.items():
            for word in {noun} | {norm(r['fields'][4]) for r in rows}:
                self.quality_words.setdefault(word, set()).add(noun)

    @lru_cache(maxsize=65536)
    def _quality_frames(self, text):
        if UNSAFE.search(text):
            return []
        result = []
        words = norm(text).split()
        if not words:
            return []
        options = self.quality_words.get(words[0], set()) | self.quality_words.get(words[-1], set())
        for noun in sorted(options):
            rows = self.qualities[noun]
            targets = {norm(r['fields'][4]) for r in rows}
            if len(targets) != 1:
                continue
            adjective = next(iter(targets))
            agreement = self.agreement.get(adjective, [])
            adjective_only = bool(agreement) and {r['fields'][2] for r in agreement} == {'adj'}
            patterns = (('quality_nominal', f'(?P<quality>{noun})\\s+(?:of|in)\\s+(?:the\\s+)?(?P<argument>.+)'), ('quality_prefix', f'(?P<quality>{adjective})\\s+(?P<argument>.+)'), ('quality_suffix', f'(?P<argument>.+?)\\s+(?P<quality>{adjective})'))
            for rule, pattern in patterns:
                m = re.fullmatch(pattern, text, re.I)
                if not m or re.search('\\b(?:of|in)\\b', m['argument'], re.I):
                    continue
                result.append({'rule': rule, 'key': [adjective, norm(m['argument'])], 'quality': fragment(text, *m.span('quality')), 'argument': fragment(text, *m.span('argument')), 'LRNOM': rows, 'source_binding': self.source_binding, 'LRAGR': agreement, 'adjective_only': adjective_only, 'authority': 'source-attested derivation PLUS bounded quality predication; not source synonym claim', 'retained_argument_order': True})
        return result

    @lru_cache(maxsize=65536)
    def view(self, text, *, route='both'):
        """One bounded role parse; use symmetrically, not as raw synonym expansion."""
        if route not in {'L', 'N', 'both'}:
            raise ValueError('Unknown full-expression proof route')
        parses = []
        if route in {'L', 'both'}:
            for frame in locative_frames(text):
                parses.append(('explicit_locative_full_finding', frame, frame['key'][0] + ' ' + frame['key'][1]))
        if route in {'N', 'both'}:
            for frame in self._quality_frames(text):
                if not frame['adjective_only']:
                    continue
                parses.append(('source_quality_predication', frame, frame['key'][0] + ' ' + frame['key'][1]))
        result = {'original': text, 'view': text, 'transformations': [], 'unresolved': [], 'authority': VERSION}
        views = {p[2] for p in parses}
        if len(views) > 1:
            result['unresolved'] = ['competing_full_expression_parses']
            return result
        if not parses:
            return result
        canonical = next(iter(views))
        clinical = compare(text, canonical)
        if clinical['unrepresented'] or clinical['unsupported_additions'] or clinical['contradictions'] or set(clinical['unresolved']) - ATTACHMENT_ONLY:
            result['unresolved'] = ['independent_clinical_or_scope_obstacle']
            return result
        result.update(view=canonical, clinical=clinical, transformations=[{'rule': name, 'frame': frame, 'canonical': canonical} for name, frame, _ in parses])
        return result

    def proof(self, text, label, *, route='both'):
        if route not in {'L', 'N', 'both'}:
            raise ValueError('Unknown full-expression proof route')
        clinical = compare(text, label)
        result = {'version': VERSION, 'original': text, 'candidate': label, 'positive': False, 'route': None, 'transformations': [], 'clinical': clinical, 'unresolved': ['complete_expression_not_proved']}
        if clinical['unrepresented'] or clinical['unsupported_additions'] or clinical['contradictions'] or set(clinical['unresolved']) - ATTACHMENT_ONLY:
            result['unresolved'] = ['independent_clinical_or_scope_obstacle']
            return result
        approaches = []
        if route in {'L', 'both'}:
            approaches.append(('explicit_locative_full_finding', locative_frames(text), locative_frames(label)))
        if route in {'N', 'both'}:
            approaches.append(('source_quality_predication', self._quality_frames(text), self._quality_frames(label)))
        for name, left, right in approaches:
            witnesses = [(a, b) for a in left for b in right if a['key'] == b['key'] and (name != 'source_quality_predication' or ((a['rule'] == 'quality_nominal' or a['adjective_only']) and (b['rule'] == 'quality_nominal' or b['adjective_only'])))]
            keys = {tuple(a['key']) for a, b in witnesses}
            if len(keys) == 1 and witnesses:
                result.update(positive=True, route=name, unresolved=[], transformations=[{'input': a, 'candidate': b} for a, b in witnesses], resolved_attachment=sorted(set(clinical['unresolved']) & ATTACHMENT_ONLY))
                return result
        return result

def proof(text, label):
    """Default is grammar-only; source-quality route needs explicit bound rows."""
    return FullExpressionProof().proof(text, label, route='L')

def view(text):
    return FullExpressionProof().view(text, route='L')
