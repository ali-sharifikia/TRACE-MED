from runtime_resources import resource as asp_resource
from clinical_specificity import compare, interpret
from conflict_detection import detect_conflict_evidence, DIRECTIONAL_MODIFIERS, LATERALITY_TOKENS, TEMPORAL_MODIFIERS, SEVERITY_TOKENS
from fuzzy_matching import tokenize_term
from grammatical_discrimination import equivalence
import graded_completion as M
FIELDS = ('directionality_conflict', 'negation_conflict', 'laterality_conflict', 'temporal_conflict', 'severity_conflict', 'anatomical_conflict', 'unsupported_specificity')

def direction(text):
    return {a['value'] for a in interpret(text)['attributes'] if a['attribute'] == 'direction'}

def flag_proof(original, candidate, raw, positive):
    result = {'proved': False, 'candidate': candidate, 'original': original, 'raw': raw}
    if not raw or not positive:
        return result
    fuzzy = raw.get('fuzzy_evidence') or {}
    flags = {k for k in FIELDS if fuzzy.get(k)}
    result['original_flags'] = sorted(flags)
    if flags != {'unsupported_specificity'}:
        return result
    if set(fuzzy.get('conflict_flags', [])) - {'unsupported_specificity'}:
        return result
    if ' '.join((raw.get('fuzzy_query_term') or '').lower().split()) != ' '.join(original.lower().split()):
        return result
    if raw.get('llt_name') != candidate:
        return result
    x, y = (tokenize_term(original), tokenize_term(candidate))
    old = detect_conflict_evidence(x, y)
    result['recomputed_old_flags'] = old
    if {k for k in FIELDS if old.get(k)} != flags:
        return result
    categories = [('direction', DIRECTIONAL_MODIFIERS), ('laterality', LATERALITY_TOKENS), ('temporal', TEMPORAL_MODIFIERS), ('severity', SEVERITY_TOKENS)]
    introduced = [k for k, words in categories if not x & words and y & words]
    result['old_added_categories'] = introduced
    result['current_comparison'] = compare(original, candidate)
    same = direction(original) == direction(candidate) and len(direction(original)) == 1
    result['proved'] = bool(introduced == ['direction'] and same and (not result['current_comparison']['full_equivalence_blocked']))
    return result

def wording(text, label, e):
    if not e['positive']:
        return None
    if text.strip().casefold() == label.strip().casefold():
        return 3
    if equivalence(text, label)['equivalent']:
        return 2
    return 1

def proposal(case, k, raw, mode):
    cs = {r['candidate_id']: r for r in case['candidates']}
    blockers = list(k['blockers'])
    proofs = {}
    preferences = []
    group = k.get('group', [])
    comparisons = k.get('comparisons', {})
    if not group:
        return {'accepted': False, 'blockers': blockers or ['no_group'], 'prediction': None}
    for cid in group:
        e = compare(case['input'], cs[cid]['llt_name'])
        if e['full_equivalence_blocked']:
            blockers.append('specificity_guard:' + cid)
        if mode in ('R', 'RW') and 'retrieval_conflict:' + cid in blockers:
            proof = flag_proof(case['input'], cs[cid]['llt_name'], raw.get(cid), comparisons[cid]['positive'])
            proofs[cid] = proof
            if proof['proved']:
                blockers.remove('retrieval_conflict:' + cid)
    if mode in ('W', 'RW'):
        classes = {cid: wording(case['input'], cs[cid]['llt_name'], comparisons[cid]) for cid in comparisons}
        for cid in k.get('outside', []):
            tag = 'outside_supported_meaning:' + cid
            if tag in blockers and classes[cid] is not None and all((classes[g] is not None and classes[g] > classes[cid] for g in group)):
                blockers.remove(tag)
                preferences.append({'rival': cid, 'rival_class': classes[cid], 'group_classes': {g: classes[g] for g in group}, 'rival_remains_clinically_compatible': True})
    numeric = k.get('numeric')
    if not numeric or numeric['score'] < 0.7 or numeric['margin'] < 0.01:
        blockers.append('fixed_numeric_gate')
    if len(case['baseline']['pt']) != 1 or case['baseline']['llt']:
        blockers.append('not_deferred_single_PT')
    representative = min((cs[g] for g in group), key=lambda c: M.previous.prior.group.faithful_key(case['input'], c['llt_name']))
    return {'accepted': not blockers, 'blockers': sorted(set(blockers)), 'numeric': numeric, 'reconciliations': proofs, 'preferences': preferences, 'group': group, 'representative': representative, 'prediction': {'pt': case['baseline']['pt'], 'llt': [representative['llt_code']]} if not blockers else None}
