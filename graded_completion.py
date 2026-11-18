"""Gold-free reference completion; original source facts and scores stay bound."""
from runtime_resources import resource as asp_resource
import copy
import meaning_completion as previous

def expand(case, ref, sem, limit=500):
    groups = {}
    for r in case['candidates']:
        groups.setdefault(r['pt_code'], []).append(r)
    before = previous.prior.ranker.rank(case['input'], [], groups, sem)
    if set(before['features']) != set(case['scores']) or any((abs(before['features'][k]['score'] - s) > 1e-10 for k, s in case['scores'].items())):
        return (None, {'blocker': 'original_scores_not_reproduced'})
    considered = set(case['baseline']['pt']) or set(groups)
    if not considered <= set(ref.groups):
        return (None, {'blocker': 'unknown_parent_group'})
    original = {r['llt_code'] for r in case['candidates']}
    missing = set().union(*(ref.groups[p] for p in considered)) - original
    if len(original | missing) > limit:
        return (None, {'blocker': 'complete_pool_exceeds_500', 'required': len(original | missing)})
    c = copy.deepcopy(case)
    added = []
    for code in sorted(missing):
        r = ref.identity(code)
        if not r.get('current'):
            return (None, {'blocker': 'reference_identity_unavailable'})
        row = {k: r[k] for k in ('llt_code', 'llt_name', 'pt_code', 'pt_name')}
        row.update(candidate_id='LLT:' + code, is_current=True)
        c['candidates'].append(row)
        added.append(row)
    expanded = {}
    for r in c['candidates']:
        expanded.setdefault(r['pt_code'], []).append(r)
    ranked = previous.prior.ranker.rank(c['input'], [], expanded, sem)
    c['scores'] = {k: f['score'] for k, f in ranked['features'].items()}
    return (c, {'added': added, 'added_count': len(added), 'original_count': len(original), 'semantic_observations_missing': [r['candidate_id'] for r in added if r['candidate_id'] not in sem], 'original_scores_verified': True, 'derived_scores': c['scores']})

def propose(case, ref, sem, lex, mode):
    if mode not in ('F', 'K'):
        raise ValueError('Unknown condition')
    if mode == 'K':
        a = previous.hypothesis(case, ref, sem, lex, joint=False, graded=True)
        return {**a, 'mode': mode, 'expansion': None}
    c, ev = expand(case, ref, sem)
    if c is None:
        return {'accepted': False, 'representative': None, 'blockers': [ev['blocker']], 'expansion': ev, 'mode': mode}
    a = previous.hypothesis(c, ref, sem, lex, joint=True, graded=False)
    return {**a, 'mode': mode, 'expansion': ev}
