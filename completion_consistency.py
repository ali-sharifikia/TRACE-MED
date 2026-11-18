"""Proof reconciliation, frozen wording repair and bounded event roles; no labels."""
from runtime_resources import resource as asp_resource
import copy, re
from functools import lru_cache
import cross_pt_expression as previous
import specificity_reconciliation as reconciliation
prior = previous.prior
MODES = ('R', 'C', 'RC', 'E')
proof_engine = previous.loader.isolated('_consistent_proof', previous.baseline.__file__)
repaired_proposal = previous.loader.isolated('_consistent_reconciliation', reconciliation.__file__)

@lru_cache(maxsize=65536)
def aligned_compare(text, label):
    old = previous.attributes(text, label)
    result = copy.deepcopy(old)
    ev = prior.ranker.comparison(text, label)
    s = prior.ranker.support(text, label)
    valid = bool(ev['equivalent'] and ev['source_frame']['safe'] and ev['candidate_frame']['safe'] and (not ev['protected_differences']) and (not s['conflicts']) and (not s['unsupported_specificity']) and (not s['unrepresented_input_information']) and (not old['unrepresented']) and (not old['unsupported_additions']) and (not old['contradictions']) and (old['unresolved'] == ['full_expression_attachment_unverified']))
    result['attachment_reconciliation'] = {'proved': valid, 'original_unresolved': old['unresolved'], 'authority': 'existing v1.9 safe complete argument/course comparison', 'comparison': ev}
    if valid:
        result.update(unresolved=[], full_equivalence_blocked=False, relationship='attachment_verified_by_existing_full_argument')
    return result
proof_engine.specificity = aligned_compare
repaired_proposal.compare = aligned_compare
EVENT_WORDS = prior.ranker.COURSE
EP = '(?:' + '|'.join(sorted(EVENT_WORDS, key=lambda x: (-len(x), x))) + ')'
EVENT_UNSAFE = re.compile('\\d|[^a-zA-Z\\s-]|\\b(?:and|or|with|without|not|no|non|possible|suspected|probable|may|might|after|before|following|during|since|due|from|to|because|history|new|onset)\\b', re.I)

@lru_cache(maxsize=32768)
def event_frame(text):
    out = {'original': text, 'view': text, 'transformation': None, 'unresolved': []}
    patterns = (('new_onset_prefix', '\\s*(?P<e>new[- ]onset)\\s+(?:of\\s+)?(?P<a>.+?)\\s*'), ('new_onset_suffix', '\\s*(?P<a>.+?)\\s+(?P<e>new[- ]onset)\\s*'), ('course_prefix', f'\\s*(?P<e>{EP})\\s+(?:of\\s+)?(?P<a>.+?)\\s*'), ('course_suffix', f'\\s*(?P<a>.+?)\\s+(?P<e>{EP})\\s*'))
    for rule, pattern in patterns:
        m = re.fullmatch(pattern, text, re.I)
        if not m:
            continue
        body = m['a']
        if EVENT_UNSAFE.search(body) or re.search('\\b' + EP + '\\b', body, re.I):
            out['unresolved'] = ['event_argument_scope_unresolved']
            return out
        ev = 'new onset' if rule.startswith('new_onset') else EVENT_WORDS[m['e'].casefold()]
        out['view'] = body + ' ' + ev
        out['transformation'] = {'rule': rule, 'event': ev, 'predicate_text': m['e'], 'predicate_span': list(m.span('e')), 'argument': body, 'argument_span': list(m.span('a')), 'authority': 'bounded event attachment; existing COURSE morphology', 'qualifiers_retained': True}
        return out
    return out

class EventLex:

    def __init__(self, lex):
        self.lex = previous.ExpressionLex(lex)

    def view(self, text):
        e = event_frame(text)
        old = self.lex.view(e['view'])
        return {**old, 'original': text, 'event_frame': copy.deepcopy(e), 'transformations': ([e['transformation']] if e['transformation'] else []) + old['transformations']}

def lex_for(lex, mode):
    return EventLex(lex) if mode == 'E' else previous.ExpressionLex(lex)

def analyze(case, ref, sem, lex, *, raw=None, mode='RC'):
    if mode not in MODES:
        raise ValueError('Unknown consistency mode')
    if case['baseline']['llt']:
        return {'accepted': False, 'representative': None, 'blockers': ['preserve_existing_automatic'], 'comparisons': {}}
    if case['baseline']['pt']:
        if len(case['baseline']['pt']) != 1:
            return {'accepted': False, 'representative': None, 'blockers': ['compound_PT_scope'], 'comparisons': {}}
        k = reconciliation.M.propose(case, ref, sem, lex, 'K')
        p = repaired_proposal if mode in ('C', 'RC', 'E') else reconciliation
        a = p.proposal(case, k, raw or {}, 'RW' if mode in ('R', 'RC', 'E') else 'CONTROL')
        a.update(branch='accepted_PT', comparisons=k['comparisons'], original_group_comparison=k, mode=mode)
        return a
    if mode == 'R':
        a = previous.analyze(case, ref, sem, lex, raw=raw)
    else:
        a = proof_engine.analyze(case, ref, sem, lex_for(lex, mode), raw=raw, reconcile=False)
        a['cross_PT'] = previous.parent_contrasts(case, a)
    a.update(mode=mode, branch='PT_recovery', event_frame=event_frame(case['input']))
    return a
