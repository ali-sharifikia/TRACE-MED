"""Single final-policy adapter for bound phrase, role and threshold proofs.

Historical audit identifiers remain stable; original reviewer approval is never
manufactured. The inherited intermediate adapter overrides are consolidated here.
"""
import copy
from contextvars import ContextVar
from types import SimpleNamespace
import completion_authority as existing
import threshold_acceptance as method
from runtime_resources import implementation

CONTEXT = ContextVar('completion_consistency', default=None)
gate = existing.reuse.isolated('_threshold_completion', existing.__file__)
gate.policy.EXPERIMENT = 'threshold_acceptance_v1'
gate.policy.MARKER = 'Opt-in graded positive-margin acceptance; original reviewer unchanged.'


def hypothesis(case, ref, sem, lex, **kwargs):
    ctx = CONTEXT.get()
    if ctx is None:
        raise ValueError('Missing bound consistency condition')
    return method.analyze(case, ref, sem, lex, raw=ctx['raw'], mode=ctx['condition'])


gate.methods = SimpleNamespace(__file__=method.__file__, hypothesis=hypothesis, prior=method.prior)
gate.implementation = implementation
gate.policy.implementation = implementation
original_apply = gate.apply


def context_for(request):
    if request.get('context_probe') is not None:
        raise ValueError('Live context probes are not part of this release')
    return None


def resolve(request, response):
    return {'original': request['input'], 'case': request['case'], 'resolved': True,
            'condition': request['consistency_condition'], 'context': context_for(request)}


def support(text, label, resolution, fn=method.prior.ranker.support):
    # The active role adapter transports the full PT/LLT phrase proof.
    case, mode, context = resolution['case'], 'PT', resolution.get('context')
    wrapper = method.ComparisonLex(gate.LEX, mode, context)
    newly_viewed = bool(wrapper.view(text)['transported_transformations'] or wrapper.view(label)['transported_transformations'])
    if case['baseline']['pt'] and not newly_viewed:
        return method.prior.feature_support(text, label, case, 'ALL', wrapper, fn)
    ref = gate.policy.old.reference(str(gate.policy.old.REF), gate.policy.sha(gate.policy.old.REF))
    lex = method.lex_for(gate.LEX, mode, context)
    for row in sorted(case['candidates'], key=lambda r: r['candidate_id']):
        if label not in (row['llt_name'], row['pt_name']):
            continue
        identity = ref.identity(row['llt_code'])
        if not identity.get('current') or any(identity.get(k) != row[k] for k in ('llt_name', 'pt_code', 'pt_name')):
            continue
        view = method.prior.context_view(text, row, 'ALL', lex)
        proof = method.proof_engine.input_proof(view['input']['view'], view['candidate']['view'])
        transported = bool(wrapper.view(text)['transported_transformations'] or wrapper.view(row['llt_name'])['transported_transformations'])
        comparison = method.transported_compare(text, row['llt_name'], wrapper, transported)
        if proof['positive'] and (not transported or not comparison['full_equivalence_blocked']):
            result = copy.deepcopy(fn(view['input']['view'], view['candidate']['view']))
            original = fn(text, row['llt_name'])
            if transported:
                result['conflicts'] = sorted(set(result['conflicts']) | set(original['conflicts']))
            result['phrase_evidence_witness'] = {'current_identity': row, 'views': view, 'proof': proof,
                                                'clinical_transport': comparison, 'original_support': original}
            return result
    return fn(text, label)


def raw_from_input(inp):
    return {c.candidate_id: {k: c.model_dump(mode='json')[k]
                            for k in ('llt_name', 'fuzzy_query_term', 'fuzzy_evidence')}
            for c in inp.aggregation_result.candidates}


def apply(*, authority=None, enabled=True, **kw):
    if not enabled or authority is None:
        return original_apply(authority=None, enabled=False, **kw)
    context = context_for(authority['request'])
    ctx = authority.get('consistency_context')
    if not ctx or set(ctx) != {'raw', 'condition'} or ctx['condition'] not in method.MODES:
        raise ValueError('Invalid consistency condition')
    if authority.get('request', {}).get('consistency_condition') != ctx['condition']:
        raise ValueError('Inconsistent bound configuration')
    if ctx['raw'] != raw_from_input(kw['reviewer_input']):
        raise ValueError('Changed original retrieval evidence')
    context_token = method.ACTIVE.set(context)
    token = CONTEXT.set(ctx)
    try:
        return original_apply(authority=authority, enabled=True, **kw)
    finally:
        CONTEXT.reset(token)
        method.ACTIVE.reset(context_token)


def apply_after(current_decision, current_gate, *, authority=None, enabled=False, **kw):
    if not enabled or authority is None:
        return current_decision, current_gate
    current = gate.policy.decision_projection(current_decision, current_gate)
    if current['llt']:
        return current_decision, current_gate
    if current['pt'] != authority['request']['case']['baseline']['pt']:
        raise ValueError('Changed prior PT authority')
    return apply(authority=authority, enabled=True, **kw)


gate.policy.resolve = resolve
gate.policy.new_support = support
gate.policy.apply_development_gate = apply
gate.apply = apply
