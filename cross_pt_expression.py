"""Scoped complete-expression grammar and cross-parent contrasts; no label I/O."""
from runtime_resources import resource as asp_resource
import copy, re
from functools import lru_cache
import pt_hierarchy_recovery as baseline
import module_loading as loader
from clinical_specificity import compare as attributes
prior = baseline.prior
engine = loader.isolated('_cross_pt_expression_hierarchy', baseline.__file__)
PREDICATES = {**{w: 'increase' for w in ('increase', 'increases', 'increased', 'rise', 'rises', 'risen')}, **{w: 'decrease' for w in ('decrease', 'decreases', 'decreased', 'fall', 'falls', 'fallen', 'drop', 'drops', 'dropped', 'decline', 'declines', 'declined', 'reduction', 'reductions', 'reduced')}}
HEADS = ('rate', 'count', 'concentration', 'activity', 'percentage', 'proportion', 'duration', 'level', 'value', 'pressure', 'temperature', 'weight', 'volume', 'interval', 'density', 'length')
HEAD_FORMS = {h: h for h in HEADS}
HEAD_FORMS.update({h + 's': h for h in HEADS if not h.endswith('y')})
HEAD_FORMS.update(activities='activity', densities='density')
P = '(?:' + '|'.join(sorted(PREDICATES, key=len, reverse=True)) + ')'
UNSAFE = re.compile('\\d|[^a-zA-Z\\s-]|\\b(?:no|not|non|without|possible|possibly|suspected|probable|and|or|with|after|before|following|during|since|from|to|due|by|of|per|history|acute|chronic|transient|persistent|recurrent|sudden|rapid|gradual|severe|mild|moderate)\\b', re.I)

@lru_cache(maxsize=65536)
def quantity_frame(text):
    result = {'original': text, 'view': text, 'transformation': None, 'unresolved': []}
    patterns = (('nominal_quantity_change', f'\\s*(?:a\\s+|an\\s+)?(?P<p>{P})\\s+in\\s+(?P<a>.+?)\\s*'), ('prefix_quantity_change', f'\\s*(?P<p>{P})\\s+(?P<a>.+?)\\s*'), ('suffix_quantity_change', f'\\s*(?P<a>.+?)\\s+(?:(?:has|have)\\s+)?(?P<p>{P})\\s*'))
    for rule, pattern in patterns:
        m = re.fullmatch(pattern, text, re.I)
        if not m:
            continue
        arg = m['a']
        tokens = re.findall('[a-z]+', arg.casefold())
        if not tokens or tokens[-1] not in HEAD_FORMS:
            continue
        if UNSAFE.search(arg) or re.search('\\b(?:' + P + '|high|low|elevated|raised|abnormal|positive|negative|increased|decreased)\\b', arg, re.I):
            result['unresolved'] = ['quantity_argument_scope_not_simple']
            return result
        if re.search('\\s-|\\-\\s', arg):
            result['unresolved'] = ['quantity_argument_scope_not_simple']
            return result
        direction = PREDICATES[m['p'].casefold()]
        result['view'] = direction + ' in ' + arg
        result['transformation'] = {'rule': rule, 'authority': 'derived_bounded_quantity_grammar', 'predicate': m['p'], 'predicate_span': list(m.span('p')), 'direction': direction, 'argument': arg, 'argument_span': list(m.span('a')), 'quantity_head': HEAD_FORMS[tokens[-1]], 'retained_ordered_argument': True}
        return result
    return result

class ExpressionLex:

    def __init__(self, lex):
        self.lex = lex

    def view(self, text):
        q = quantity_frame(text)
        old = self.lex.view(q['view'])
        return {**old, 'original': text, 'quantity_frame': copy.deepcopy(q), 'transformations': ([q['transformation']] if q['transformation'] else []) + old['transformations']}

def full_support(text, label):
    a, b = (quantity_frame(text), quantity_frame(label))
    proof = baseline.input_proof(a['view'], b['view'])
    return {**proof, 'quantity_input': a, 'quantity_candidate': b}

def parent_contrasts(case, analysis):
    """No unknown relationship is converted to a clinical exclusion."""
    cs = analysis.get('comparisons', {})
    parents = {}
    for cid, c in cs.items():
        row = c['candidate']
        proof = c['input_proof']
        specific = proof['specificity']
        status = 'supported' if proof['positive'] else 'contradicted' if specific['contradictions'] else 'unresolved'
        detail = {'status': status, 'preserved': specific['preserved'], 'unrepresented': specific['unrepresented'], 'unsupported_additions': specific['unsupported_additions'], 'contradictions': specific['contradictions'], 'unresolved': sorted(set(specific['unresolved'] + ([] if proof['positive'] else ['full_expression_correspondence_unproved'])))}
        c['full_expression_comparison'] = detail
        pt = parents.setdefault(row['pt_code'], {'pt_code': row['pt_code'], 'pt_name': row['pt_name'], 'parent_label_comparison': attributes(case['input'], row['pt_name']), 'candidates': []})
        pt['candidates'].append(cid)

    def key(cid):
        c = cs[cid]
        s = c['full_expression_comparison']['status']
        return ({'supported': 2, 'unresolved': 1, 'contradicted': 0}[s], c['score'])
    for pt in parents.values():
        best = max((key(cid) for cid in pt['candidates']))
        pt['leading_candidates'] = sorted((cid for cid in pt['candidates'] if key(cid) == best))
        pt['comparison_key'] = list(best)
        pt['candidates'].sort()
    best = max((tuple(p['comparison_key']) for p in parents.values()), default=None)
    leaders = sorted((p for p in parents if tuple(parents[p]['comparison_key']) == best))
    rivals = []
    for p in leaders:
        for q in sorted(parents):
            if q == p:
                continue
            rivals.append({'leading_parent': p, 'rival_parent': q, 'leader_candidates': parents[p]['leading_candidates'], 'rival_candidates': parents[q]['leading_candidates'], 'leader_support': parents[p]['comparison_key'][0], 'rival_support': parents[q]['comparison_key'][0], 'relationship': 'competing_supported_meanings' if parents[p]['comparison_key'][0] == parents[q]['comparison_key'][0] == 2 else 'explicit_candidate_contradiction' if parents[q]['comparison_key'][0] == 0 else 'unresolved_rival_retained'})
    return {'parents': parents, 'leading_PT_set': leaders, 'rivals': rivals, 'authority': 'diagnostic ranking only; acceptance retains all outside scores'}

def analyze(case, ref, sem, lex, *, raw=None, reconcile=False):
    a = engine.analyze(case, ref, sem, ExpressionLex(lex), raw=raw, reconcile=False)
    a['cross_PT'] = parent_contrasts(case, a)
    a['input_quantity_frame'] = quantity_frame(case['input'])
    return a
