"""Direct candidate role proofs in existing meaning groups; labels never read."""
from runtime_resources import resource as asp_resource
import copy
import types
from pathlib import Path
import phrase_configuration as configured
import direct_role_pair as proof
full = configured.method
prior, previous, reconciliation = (full.prior, full.previous, full.reconciliation)
baseline, phrases = (full.baseline, full.phrases)
PHRASE_SOURCE = full.PHRASE_SOURCE
proof_engine = full.proof_engine
ACTIVE = full.ACTIVE
MODES = ('A', 'R', 'E')

def ComparisonLex(lex, mode, context=None):
    if mode not in MODES + ('PT',):
        raise ValueError('Unknown direct-role condition')
    return full.ComparisonLex(lex, 'PT', context)

def lex_for(lex, mode, context=None):
    return full.lex_for(lex, 'PT', context)
transported_compare = full.transported_compare

def analyze(case, ref, sem, lex, *, raw=None, mode='A', context=None):
    if mode not in MODES:
        raise ValueError('Unknown direct-role condition')
    context = context if context is not None else ACTIVE.get()
    old = full.analyze(case, ref, sem, lex, raw=raw, mode='PT', context=context)
    if mode == 'A' or case['baseline']['llt'] or len(case['baseline']['pt']) != 1:
        return {**old, 'experimental_condition': mode}
    wrapper = ComparisonLex(lex, mode, context)
    pair_audit = {}

    def pair(original, x, y, reference):
        existing = prior.group.compare_pair(original, x, y, reference)
        if existing['status'] != 'unresolved' or existing.get('retained_differences'):
            return existing
        a, b = (reference.identity(x), reference.identity(y))
        if not a.get('current') or not b.get('current') or a['pt_code'] != b['pt_code']:
            return existing
        va, vb = (wrapper.view(a['llt_name']), wrapper.view(b['llt_name']))
        direct = proof.prove(va, vb, mode=mode, clinical_compare=full.baseline_compare)
        input_checks = [full.transported_compare(original, n, wrapper, True) for n in (a['llt_name'], b['llt_name'])]
        valid = direct['positive'] and (not any((c['contradictions'] for c in input_checks)))
        record = {'left': a, 'right': b, 'views': [va, vb], 'direct': direct, 'input_checks': input_checks, 'previous': existing, 'authorized': bool(valid)}
        pair_audit['|'.join(sorted((x, y)))] = record
        if not valid:
            return existing
        return {**existing, 'status': 'verified_equivalent', 'route': 'derived_direct_role_comparison', 'rule': 'direct_role_group_' + mode, 'reasons': [], 'retained_differences': [], 'witness': record}
    engine = previous.loader.isolated('_direct_role_hypothesis_' + mode, Path(__file__).with_name('meaning_completion.py'))
    engine.prior = types.SimpleNamespace(**{**vars(prior), 'group': types.SimpleNamespace(**{**vars(prior.group), 'compare_pair': pair})})
    k = engine.hypothesis(case, ref, sem, wrapper, joint=False, graded=True)
    before = old.get('comparisons', {})
    after = k.get('comparisons', {})
    if set(before) != set(after) or any((abs(before[i]['score'] - after[i]['score']) > 1e-10 for i in before)):
        raise ValueError('Direct role grouping changed candidate scores or membership')
    group = k.get('group', [])
    new_edges = [r for r in pair_audit.values() if r['authorized'] and 'LLT:' + r['left']['llt_code'] in group and ('LLT:' + r['right']['llt_code'] in group)]
    if new_edges:
        for cid in group:
            if not after[cid]['positive']:
                k['blockers'].append('new_group_member_without_input_proof:' + cid)
    k['blockers'] = sorted(set(k['blockers']))
    k['accepted'] = not k['blockers']
    reconcile = previous.loader.isolated('_direct_role_reconcile_' + mode, reconciliation.__file__)
    reconcile.compare = lambda text, label: full.transported_compare(text, label, wrapper, True)
    result = reconcile.proposal(case, k, raw or {}, 'RW')
    result.update(branch='accepted_PT', comparisons=after, original_group_comparison=k, clinical_transport={cid: reconcile.compare(case['input'], next((r['llt_name'] for r in case['candidates'] if r['candidate_id'] == cid))) for cid in group}, role_group_evidence={'pairs': pair_audit, 'new_group_edges': len(new_edges), 'old_group': old.get('group', []), 'old_numeric': old.get('numeric'), 'old_blockers': old.get('blockers', []), 'scores_unchanged': True}, input_view=wrapper.view(case['input']), context_resolution=copy.deepcopy(context), experimental_condition=mode)
    return result
