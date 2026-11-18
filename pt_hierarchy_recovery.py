"""Opt-in full-expression LLT proof and verified hierarchy; no labels or models."""
from runtime_resources import resource as asp_resource
import copy
from contextlib import contextmanager
from functools import lru_cache
import input_discrimination as lexical
import graded_completion as previous
from clinical_specificity import compare as specificity
from specificity_reconciliation import flag_proof
prior = previous.previous.prior

@lru_cache(maxsize=32768)
def input_proof(text, label):
    support = prior.ranker.support(text, label)
    comparison = prior.ranker.comparison(text, label)
    grammar = prior.group.wording_proof(text, label)
    clinical = specificity(text, label)
    positive = bool((grammar or comparison['equivalent']) and (not comparison['protected_differences']) and comparison['source_frame']['safe'] and (not support['conflicts']) and (not support['unsupported_specificity']) and (not support['unrepresented_input_information']) and (not clinical['full_equivalence_blocked']))
    return dict(positive=positive, grammar=grammar, comparison=comparison, support=support, specificity=clinical)

@contextmanager
def corrected_features(proven):
    """Serialized experimental use only; restore existing feature function."""
    old = lexical.llt_features

    def features(text, row, semantic=None, config=lexical.DEVELOPMENT):
        result = old(text, row, semantic, config)
        if row['candidate_id'] in proven:
            own = result['support']
            parent = result['parent_support']
            if not own['conflicts'] and (not own['unsupported_specificity']):
                result['hierarchy_correction'] = {'parent_lexical_support': copy.deepcopy(parent), 'original_score': result['score'], 'authority': 'input-bound full LLT proof and verified current parent'}
                result['conflicts'] = list(own['conflicts'])
                result['unsupported_specificity'] = list(own['unsupported_specificity'])
                result['score'] = max(0.0, min(1.0, sum(result['contributions'].values())))
        return result
    lexical.llt_features = features
    try:
        yield
    finally:
        lexical.llt_features = old

def analyze(case, ref, sem, lex, *, raw=None, reconcile=False):
    out = {'accepted': False, 'representative': None, 'blockers': [], 'comparisons': {}, 'group': [], 'pair_proofs': {}, 'reconcile': reconcile}
    b = out['blockers']
    if case['baseline']['pt'] or case['baseline']['llt']:
        b.append('preserve_existing_authority')
        return out
    for name in ('original_scope_available', 'input_bound', 'single_concept', 'scores_bound'):
        if not case.get(name):
            b.append(name)
    for name in ('fallback', 'exact_collision', 'numeric_uncertainty', 'exact_route'):
        if case.get(name):
            b.append(name)
    expanded, expansion = previous.expand(case, ref, sem)
    out['expansion'] = expansion
    if expanded is None:
        b.append(expansion['blocker'])
        return out
    rows = {r['candidate_id']: r for r in expanded['candidates']}
    views = {cid: prior.context_view(case['input'], r, 'ALL', lex) for cid, r in rows.items()}
    proof = {cid: input_proof(v['input']['view'], v['candidate']['view']) for cid, v in views.items()}
    proven = set()
    for cid, r in rows.items():
        identity = ref.identity(r['llt_code'])
        valid = bool(identity.get('current') and cid == 'LLT:' + r['llt_code'] and all((identity[k] == r[k] for k in ('llt_name', 'pt_name', 'pt_code'))))
        if not valid:
            b.append('invalid_identity:' + cid)
        if valid and proof[cid]['positive']:
            proven.add(cid)
    groups = {}
    for cid, r in rows.items():
        v = views[cid]
        groups.setdefault(r['pt_code'], []).append({**r, 'llt_name': v['candidate']['view'], 'pt_name': v['parent']['view']})
    text = lex.view(case['input'])['view']
    before = prior.ranker.rank(text, [], groups, sem)
    with corrected_features(proven):
        ranked = prior.ranker.rank(text, [], groups, sem)
    scores = {cid: ranked['features'][cid]['score'] for cid in rows}
    for cid in rows:
        f = ranked['features'][cid]
        legacy = case['features'].get(cid, {})
        retrieval = bool(legacy.get('retrieval_conflict'))
        reconciliation = flag_proof(case['input'], rows[cid]['llt_name'], (raw or {}).get(cid), proof[cid]['positive']) if reconcile and retrieval else None
        if reconciliation and reconciliation['proved']:
            retrieval = False
        out['comparisons'][cid] = {'candidate': rows[cid], 'views': views[cid], 'input_proof': proof[cid], 'score': scores[cid], 'original_score': before['features'][cid]['score'], 'parent_correction': f.get('hierarchy_correction'), 'retrieval_conflict': retrieval, 'reconciliation': reconciliation, 'remaining_ranker_conflicts': f['conflicts'], 'remaining_ranker_specificity': f['unsupported_specificity']}
    if not proven:
        b.append('no_full_input_proof')
        return out
    high = max((scores[cid] for cid in proven))
    tops = sorted((cid for cid in proven if abs(scores[cid] - high) <= 1e-10))
    anchor = min(tops, key=lambda cid: (prior.group.faithful_key(case['input'], rows[cid]['llt_name']), cid))

    def pair(a, z):
        key = '|'.join(sorted((a, z)))
        if key not in out['pair_proofs']:
            out['pair_proofs'][key] = prior.group.compare_pair(case['input'], rows[a]['llt_code'], rows[z]['llt_code'], ref)
        return out['pair_proofs'][key]
    members, outside, problems = prior.group.clique_group(anchor, sorted(rows), pair)
    out.update(group=members, outside=outside)
    b.extend(problems)
    if not set(tops) <= set(members):
        b.append('tied_meanings')
    if proven - set(members):
        b.append('outside_full_input_proof')
    for cid in members:
        e = out['comparisons'][cid]
        if cid not in proven:
            b.append('member_input_unproven:' + cid)
        if e['retrieval_conflict']:
            b.append('retrieval_conflict:' + cid)
        if e['remaining_ranker_conflicts'] or e['remaining_ranker_specificity']:
            b.append('clinical_conflict:' + cid)
    rival = max((scores[cid] for cid in outside), default=0.0)
    out['numeric'] = {'score': max((scores[cid] for cid in members)), 'outside_score': rival if outside else None, 'margin': max((scores[cid] for cid in members)) - rival, 'outside_rivals': [cid for cid in outside if abs(scores[cid] - rival) <= 1e-10]}
    if out['numeric']['score'] < 0.7:
        b.append('score_below_070')
    if out['numeric']['margin'] < 0.01:
        b.append('margin_below_001')
    keys = {cid: prior.group.faithful_key(case['input'], rows[cid]['llt_name']) for cid in members}
    chosen = [cid for cid, key in keys.items() if key == min(keys.values())]
    if len(chosen) != 1:
        b.append('representative_collision')
    out['blockers'] = sorted(set(b))
    out['accepted'] = not b
    if out['accepted']:
        out['representative'] = rows[chosen[0]]
    return out
