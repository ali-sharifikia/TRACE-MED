"""Gold-free available-resource hypotheses. No model calls or corpus exceptions."""
from runtime_resources import resource as asp_resource
import copy
from itertools import combinations
import general_llt as prior

from decision_recorder import computation

@computation('comparison.meaning_completion')
def hypothesis(case, ref, sem, lex, joint=False, graded=False):
    out = {'accepted': False, 'representative': None, 'blockers': [], 'group': [], 'numeric': None, 'comparisons': {}, 'pair_proofs': {}, 'joint': joint, 'graded': graded}
    b = out['blockers']
    base = case['baseline']
    if base['llt']:
        b.append('preserved_automatic')
        return out
    if len(base['pt']) > 1 or (not joint and len(base['pt']) != 1):
        b.append('PT_scope')
        return out
    for name in ('original_scope_available', 'input_bound', 'single_concept', 'scores_bound'):
        if not case.get(name):
            b.append(name)
    if base['pt'] and (not case.get('pt_authority')):
        b.append('PT_authority')
    for name in ('fallback', 'exact_collision', 'numeric_uncertainty', 'exact_route'):
        if case.get(name):
            b.append(name)
    raw = {}
    for r in case['candidates']:
        raw.setdefault(r['pt_code'], []).append(r)
    before = prior.ranker.rank(case['input'], [], raw, sem)
    if set(before['features']) != set(case['scores']) or any((abs(before['features'][k]['score'] - v) > 1e-10 for k, v in case['scores'].items())):
        b.append('original_scores_not_reproduced')
        return out
    views = {r['candidate_id']: prior.context_view(case['input'], r, 'ALL', lex) for r in case['candidates']}
    transformed = {pt: [{**r, 'llt_name': views[r['candidate_id']]['candidate']['view'], 'pt_name': views[r['candidate_id']]['parent']['view']} for r in rs] for pt, rs in raw.items()}
    ranked = prior.ranker.rank(lex.view(case['input'])['view'], [], transformed, sem)
    considered = {pt: rs for pt, rs in raw.items() if not base['pt'] or pt in base['pt']}
    cs = {r['candidate_id']: r for rs in considered.values() for r in rs}
    ids = sorted(cs)
    if not ids:
        b.append('no_candidates')
        return out
    if sum(map(len, considered.values())) != len(cs):
        b.append('duplicate_identity')
    for pt, rs in considered.items():
        if {r['llt_code'] for r in rs} != ref.groups.get(pt, set()):
            b.append('incomplete_group:' + pt)
    for cid, r in cs.items():
        actual = ref.identity(r['llt_code'])
        if not actual.get('current') or cid != 'LLT:' + r['llt_code'] or any((actual.get(k) != r[k] for k in ('llt_name', 'pt_name', 'pt_code'))):
            b.append('invalid_identity:' + cid)
        v = views[cid]
        f = ranked['features'][cid]
        s = prior.ranker.support(v['input']['view'], v['candidate']['view'])
        ps = prior.ranker.support(v['input']['view'], v['parent']['view'])
        comp = prior.ranker.comparison(v['input']['view'], v['candidate']['view'])
        conflicts = sorted(set(f['conflicts']) | set(s['conflicts']) | set(ps['conflicts']))
        specificity = sorted(set(f['unsupported_specificity']) | set(s['unsupported_specificity']) | set(ps['unsupported_specificity']))
        omitted = sorted(set(f['support']['unrepresented_input_information']) | set(case['features'].get(cid, {}).get('unrepresented_input_information', [])))
        protected = sorted(set(comp['protected_differences']))
        hard = conflicts or specificity or protected or omitted or (not comp['source_frame']['safe'])
        proof = prior.group.wording_proof(v['input']['view'], v['candidate']['view'])
        positive = bool((proof or comp['equivalent']) and (not hard))
        out['comparisons'][cid] = {'views': v, 'positive': positive, 'proof': proof, 'comparison': comp, 'conflicts': conflicts, 'specificity': specificity, 'omitted': omitted, 'protected': protected, 'safe': comp['source_frame']['safe'], 'score': f['score'], 'old_score': case['scores'][cid], 'retrieval_conflict': case['features'].get(cid, {}).get('retrieval_conflict', False)}
    scores = {cid: ranked['features'][cid]['score'] for cid in ids}
    maximum = max(scores.values())
    tops = [x for x in ids if abs(scores[x] - maximum) <= 1e-10]
    anchor = min(tops, key=lambda x: (prior.group.faithful_key(case['input'], cs[x]['llt_name']), x))

    def pair(a, z):
        key = '|'.join(sorted((a, z)))
        if key in out['pair_proofs']:
            return out['pair_proofs'][key]
        result = prior.group.compare_pair(case['input'], cs[a]['llt_code'], cs[z]['llt_code'], ref)
        va, vz = (views[a], views[z])
        if result['status'] not in ('exact', 'verified_equivalent') and cs[a]['pt_code'] == cs[z]['pt_code']:
            proof = prior.group.wording_proof(va['candidate']['view'], vz['candidate']['view'])
            comp = prior.ranker.comparison(va['candidate']['view'], vz['candidate']['view'])
            if proof and comp['equivalent'] and (not comp['protected_differences']):
                result = {**result, 'status': 'verified_equivalent', 'route': 'source_context', 'rule': 'v131_contextual_full_phrase', 'witness': {'left': va, 'right': vz, 'grammar': proof}}
        out['pair_proofs'][key] = result
        return result
    members, outside, problems = prior.group.clique_group(anchor, ids, pair)
    b.extend(problems)
    out.update(group=members, outside=outside, top_set=tops)
    if not set(tops) <= set(members):
        b.append('tied_different_meanings')
    for cid in members:
        e = out['comparisons'][cid]
        for k in ('conflicts', 'specificity', 'omitted', 'protected'):
            if e[k]:
                b.append(k + ':' + cid)
        if not e['safe']:
            b.append('unsafe_scope:' + cid)
        if e['retrieval_conflict']:
            b.append('retrieval_conflict:' + cid)
        if not graded and (not e['positive']):
            b.append('whole_phrase_unproven:' + cid)
    for cid in outside:
        if out['comparisons'][cid]['positive']:
            b.append('outside_supported_meaning:' + cid)
    keys = {x: prior.group.faithful_key(case['input'], cs[x]['llt_name']) for x in members}
    best = min(keys.values())
    representatives = [x for x in members if keys[x] == best]
    if len(representatives) != 1:
        b.append('representative_identity_collision')
    rival = max((scores[x] for x in outside), default=0.0)
    out['numeric'] = {'score': maximum, 'margin': maximum - rival, 'outside_score': rival if outside else None, 'outside_rivals': [x for x in outside if abs(scores[x] - rival) <= 1e-10]}
    if maximum < 0.7:
        b.append('score_below_070')
    if maximum - rival <= 1e-10:
        b.append('nonpositive_margin')
    if not graded and maximum - rival < 0.01:
        b.append('margin_below_001')
    out['blockers'] = sorted(set(b))
    out['accepted'] = not b
    if out['accepted']:
        out['representative'] = cs[representatives[0]]
    return out
