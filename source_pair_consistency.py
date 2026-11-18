"""NLM spelling witnesses in existing pair/group interfaces; no label access."""
from runtime_resources import resource as asp_resource
import copy, re, types
from functools import lru_cache
import completion_consistency as control
prior = control.prior
previous = control.previous
reconciliation = control.reconciliation
MODES = ('S',)
proof_engine = control.proof_engine

def lex_for(lex, mode):
    return control.lex_for(lex, 'RC')

class Spelling:

    def __init__(self, lex):
        self.lex = lex

    @lru_cache(maxsize=32768)
    def view(self, text):
        changes = []

        def replace(m):
            w = m[0]
            t = self.lex.token(w)
            if not t:
                return w
            rows = [r for r in self.lex.spl.get(prior.norm(w), []) if r['fields'][0] in self.lex.noun_euis]
            targets = {prior.norm(r['fields'][2]) for r in rows}
            if targets != {t['canonical']}:
                return w
            changes.append({'original': w, 'span': list(m.span()), 'canonical': t['canonical'], 'LRSPL': rows, 'noun_witnesses': t['witnesses']})
            return t['canonical']
        return {'original': text, 'view': re.sub('[A-Za-z]+', replace, text), 'transformations': changes}

def pair(original, x, y, ref, spelling):
    old = prior.group.compare_pair(original, x, y, ref)
    if old['status'] != 'unresolved' or old.get('retained_differences'):
        return old
    a, b = (ref.identity(x), ref.identity(y))
    if not a.get('current') or not b.get('current') or a['pt_code'] != b['pt_code']:
        return old
    va, vb = (spelling.view(a['llt_name']), spelling.view(b['llt_name']))
    if not (va['transformations'] or vb['transformations']):
        return old
    grammar = prior.group.wording_proof(va['view'], vb['view'])
    comparison = prior.ranker.comparison(va['view'], vb['view'])
    clinical = [control.aligned_compare(p, q) for p, q in ((va['view'], vb['view']), (vb['view'], va['view']))]
    input_checks = [control.aligned_compare(spelling.view(original)['view'], v['view']) for v in (va, vb)]
    if not grammar or not comparison['equivalent'] or comparison['protected_differences'] or (not comparison['source_frame']['safe']) or (not comparison['candidate_frame']['safe']) or any((c['full_equivalence_blocked'] for c in clinical)) or any((c['contradictions'] for c in input_checks)):
        return old
    return {**old, 'status': 'verified_equivalent', 'route': 'source_spelling_and_derived_grammar', 'rule': 'NLM_LRSPL_full_phrase', 'reasons': [], 'retained_differences': [], 'witness': {'left': va, 'right': vb, 'grammar': grammar, 'comparison': comparison, 'clinical': clinical, 'input_checks': input_checks}, 'previous': old}

def analyze(case, ref, sem, lex, *, raw=None, mode='S'):
    if mode != 'S':
        raise ValueError('Unknown source-pair condition')
    if case['baseline']['pt']:
        return control.analyze(case, ref, sem, lex, raw=raw, mode='RC')
    engine = control.previous.loader.isolated('_source_pair_hierarchy', control.proof_engine.__file__)
    engine.specificity = control.aligned_compare
    spelling = Spelling(lex)
    group = types.SimpleNamespace(**vars(prior.group))
    group.compare_pair = lambda text, x, y, r: pair(text, x, y, r, spelling)
    engine.prior = types.SimpleNamespace(**{**vars(prior), 'group': group})
    result = engine.analyze(case, ref, sem, control.lex_for(lex, 'RC'), raw=raw, reconcile=False)
    result['cross_PT'] = control.previous.parent_contrasts(case, result)
    return result

def parent_opportunity(case, analysis):
    """Diagnostic only: no candidate is chosen and no mapping is emitted."""
    out = {'passed': False, 'blockers': [], 'pt': None}
    if case['baseline']['pt']:
        out['blockers'] = ['already_has_PT']
        return out
    cs = analysis.get('comparisons', {})
    supported = sorted((k for k, v in cs.items() if v['input_proof']['positive']))
    parents = sorted({cs[k]['candidate']['pt_code'] for k in supported})
    out.update(supported=supported, parents=parents)
    if len(parents) != 1:
        out['blockers'] = ['no_unique_supported_parent']
        return out
    out['pt'] = parents[0]
    replaced = {'outside_full_input_proof', 'tied_meanings', 'representative_collision', 'margin_below_001', 'score_below_070'}
    out['blockers'] = [b for b in analysis['blockers'] if b not in replaced]
    for k in supported:
        e = cs[k]
        if e['retrieval_conflict'] or e['remaining_ranker_conflicts'] or e['remaining_ranker_specificity']:
            out['blockers'].append('supported_child_conflict:' + k)
    outside = sorted((k for k, v in cs.items() if v['candidate']['pt_code'] != out['pt']))
    score = max((cs[k]['score'] for k in supported))
    rival = max((cs[k]['score'] for k in outside), default=0.0)
    out.update(score=score, margin=score - rival, outside=outside)
    if score < 0.7:
        out['blockers'].append('parent_score_below_070')
    if score - rival < 0.01:
        out['blockers'].append('parent_margin_below_001')
    out['blockers'] = sorted(set(out['blockers']))
    out['passed'] = not out['blockers']
    return out
