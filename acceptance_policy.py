"""Frozen J -> P -> T acceptance and actual validation/finalization.

This is the same policy order as the evaluated threshold run. No gold enters here.
Upstream interpretation/review still uses the retained v1.9 gate on each round.
"""
import copy
import decision_recorder as trace
from runtime_resources import bootstrap, implementation, sha
bootstrap()
import completion_authority as underlying
import threshold_authority as policy
from schemas import ReviewerDecision, ReviewerInput, TermInput, NormalizationResult, ExactMatchResult
from reviewer_validation import reviewer_decision_validation_node
from coordinator import create_coordinator_node
from workflow_outcome import human_review_outcome_node, reviewer_final_outcome_node

def downstream(inp, decision, gate):
    state = {'reviewer_input': inp, 'reviewer_decision': decision, 'term_id': inp.term_id, 'term_input': TermInput(term=inp.original_term, frequency=1), 'normalization': NormalizationResult(original_term=inp.original_term, normalized_term=inp.normalized_term), 'input_interpretation': inp.input_interpretation, 'aggregation_result': inp.aggregation_result, 'review_round': inp.review_round, 'exact_result': ExactMatchResult(found=False), 'llt_selection_gate_result': gate}
    state.update(trace.observe('completion.validation', lambda: reviewer_decision_validation_node(state), state))
    state.update(trace.observe('completion.coordinator', lambda: create_coordinator_node()(state), state))
    fn = human_review_outcome_node if state['coordinator_result'].route.value == 'human_review' else reviewer_final_outcome_node
    state.update(trace.observe('completion.outcome', lambda: fn(state), state))
    return state

def apply(kw, fixed_policy, *, capture=False):
    old, scope = (underlying.policy.old, underlying.policy.scope)
    d, g = trace.observe('completion.retained', lambda: old.apply_development_gate(**kw), {k:v for k,v in kw.items() if k!='reranker'})
    projection = underlying.policy.decision_projection
    observed = {k: projection(d, g)[k] for k in ('pt', 'llt')}
    case = old.make_case(kw['reviewer_input'], observed, g.coverage_acceptance[scope.KEY], g, kw)
    sem = underlying.reuse.semantic_observations(kw['reviewer_input'])
    au = {'request': {'input': case['input'], 'case': case}, 'response': {}, 'semantics': sem, 'case_binding': scope.digest(case), 'implementation': underlying.implementation(), 'mode': 'J', 'cutoff': 0.01}
    with underlying.policy.integration(au):
        jd, jg = trace.observe('completion.J', lambda: underlying.apply(authority=au, **kw), au)
        retained = downstream(kw['reviewer_input'], jd, jg)
    if projection(jd, jg)['llt']:
        for skipped in ('P', 'T'):
            trace.emit('stage_skipped', {'reason': 'prior_completion_retained'}, stage='completion.' + skipped, status='skipped')
        return retained
    if projection(jd, jg)['pt'] != case['baseline']['pt']:
        raise ValueError('Unestablished changed PT authority')
    raw = {c.candidate_id: {k: c.model_dump(mode='json')[k] for k in ('llt_name', 'fuzzy_query_term', 'fuzzy_evidence')} for c in kw['reviewer_input'].aggregation_result.candidates}
    for mode in ('P', 'T'):
        bound = copy.deepcopy(au)
        bound.update(implementation=policy.implementation(), mode='R', cutoff=0.0, id=kw['reviewer_input'].term_id, fold='threshold', calibration_file=str(fixed_policy), calibration_sha256=sha(fixed_policy), consistency_context={'raw': raw, 'condition': mode})
        bound['request'].update(consistency_condition=mode, context_probe=None)
        with policy.gate.policy.integration(bound):
            nd, ng = trace.observe('completion.' + mode, lambda: policy.apply_after(d, g, authority=bound, enabled=True, **kw), bound)
            result = downstream(kw['reviewer_input'], nd, ng)
        if ng.coverage_acceptance[scope.KEY]['original_decision'] != kw['reviewer_decision'].model_dump(mode='json'):
            raise ValueError('Original reviewer evidence changed')
        if projection(nd, ng)['llt'] or mode == 'T':
            if mode == 'P':
                trace.emit('stage_skipped', {'reason': 'proof_priority_completion_retained'}, stage='completion.T', status='skipped')
            return result
    raise AssertionError('Unreachable policy state')

def replay(final_state, fixed_policy, reranker):
    """Return final runtime outcome; no fabricated original reviewer state."""
    gate = final_state.get('llt_selection_gate_result')
    if hasattr(gate, 'model_dump'):
        gate = gate.model_dump(mode='json')
    scope = (gate or {}).get('coverage_acceptance', {}).get(underlying.policy.scope.KEY)
    exact = final_state.get('exact_result')
    if hasattr(exact, 'model_dump'):
        exact = exact.model_dump(mode='json')
    if exact and exact.get('found'):
        return final_state
    if not scope:
        if not final_state.get('reviewer_input'):
            return final_state
        raise ValueError('Missing original reviewer scope; threshold result unestablished')
    ex = ExactMatchResult.model_validate(exact or {'found': False})
    kw = {'reviewer_input': ReviewerInput.model_validate(scope['original_input']), 'reviewer_decision': ReviewerDecision.model_validate(scope['original_decision']), 'reranker': reranker, 'reviewer_failure_diagnostics': final_state.get('reviewer_failure_diagnostics'), 'exact_match_found': ex.found, 'exact_ambiguity_candidates': tuple(ex.ambiguous_candidates)}
    result = apply(kw, fixed_policy)
    return {**final_state, **result}
