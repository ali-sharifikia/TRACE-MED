"""Active scoped acceptance gate. Historical audit identifiers remain stable."""
import traceback
from event_comparison import DEVELOPMENT, rank, decide, support
from assessment_scope import complete_decision, original_validated

def decision_projection(d, g):
    return {'pt': [d.resolved_pt_code] if d.resolved_pt_code else sorted({m.pt_code for m in d.component_matches}), 'llt': [s[4:] for s in d.selected_candidate_ids], 'review_pt': sorted({c.pt_code for c in g.human_review_candidates}), 'review_llt': [c.llt_code for c in g.human_review_candidates]}

def apply_development_gate(*, reviewer_input, reviewer_decision, config=DEVELOPMENT, **kwargs):
    from llt_selection_gate import apply_llt_selection_gate
    if not config.enabled:
        return apply_llt_selection_gate(reviewer_input=reviewer_input, reviewer_decision=reviewer_decision, **kwargs)
    inp = reviewer_input.model_copy(deep=True)
    original = reviewer_decision.model_copy(deep=True)
    original_validated(inp, original)
    bd, bg = apply_llt_selection_gate(reviewer_input=reviewer_input, reviewer_decision=reviewer_decision, **kwargs)
    actual_base = decision_projection(bd, bg)

    def bound_decide(evidence, ranking, baseline, **kw):
        baseline.clear()
        baseline.update(actual_base)
        return decide(evidence, ranking, baseline, **kw)
    d, g = _apply_discrimination(reviewer_input=reviewer_input, reviewer_decision=reviewer_decision, config=config, choose=bound_decide, **kwargs)
    assert inp == reviewer_input, 'Original reviewer request mutated'
    coverage = dict(g.coverage_acceptance)
    policy = dict(g.policy_configuration)
    if 'development_v1_6' in coverage:
        coverage['development_v1_7'] = coverage.pop('development_v1_6')
    if 'development_v1_6' in policy:
        policy['development_v1_7'] = policy.pop('development_v1_6')
    g = g.model_copy(update={'coverage_acceptance': coverage, 'policy_configuration': policy})
    return complete_decision(inp, original, d, g, config)

def _apply_discrimination(*, reviewer_input, reviewer_decision, reranker, config=DEVELOPMENT, choose, **kwargs):
    from llt_selection_gate import apply_llt_selection_gate, _retrieved_candidate_payloads, _as_human_review_candidate, _deferred_decision
    from adaptive_review_policy import prepare_policy_evidence
    from pt_coverage_policy import recovery_context
    from schemas import ReviewerDecision, ReviewerDecisionType, ComponentCandidateMatch, CandidateAssessment, HumanReviewPTGroup, LLTSelectionGateResult
    baseline, gate = apply_llt_selection_gate(reviewer_input=reviewer_input, reviewer_decision=reviewer_decision, reranker=reranker, **kwargs)
    if not config.enabled or kwargs.get('exact_match_found') or reviewer_input.input_interpretation.structure_type.value != 'single_concept':
        return (baseline, gate)
    selected = baseline.selected_candidate_ids
    ids = reviewer_decision.selected_candidate_ids
    raw = {c.candidate_id: c for c in reviewer_input.aggregation_result.candidates}
    available = {c.candidate_id: c for c in gate.ranked_candidates + gate.human_review_candidates}
    all_candidates = {**raw, **available}
    pt = [gate.resolved_pt_code] if gate.resolved_pt_code else sorted({all_candidates[x].pt_code for x in selected if x in all_candidates})
    automatic = baseline.decision_type in {ReviewerDecisionType.MATCHED_SINGLE, ReviewerDecisionType.MATCHED_MULTIPLE}
    base = {'pt': pt, 'llt': [x[4:] for x in selected] if automatic else [], 'review_pt': sorted({c.pt_code for c in gate.human_review_candidates}), 'review_llt': [c.llt_code for c in gate.human_review_candidates]}
    e = prepare_policy_evidence(reviewer_input, tuple(ids), single_selected_pt=pt[0] if len(pt) == 1 else None)
    if len(e.component_rankings) != 1:
        return (baseline, gate)
    pts = e.component_rankings[0]['pt_rankings']
    groups = {}
    typed = {}
    budget = 0
    for p in pts[:config.maximum_expanded_pt_groups]:
        rows = reranker.rank(input_term=reviewer_input.normalized_term, pt_code=p['pt_code'], retrieved_candidates=_retrieved_candidate_payloads(reviewer_input))
        if not rows or budget + len(rows) > config.maximum_reference_candidates:
            continue
        budget += len(rows)
        typed.update({r.candidate_id: r for r in rows})
        groups[p['pt_code']] = [{k: getattr(r, k) for k in ('candidate_id', 'llt_code', 'llt_name', 'pt_code', 'pt_name')} for r in rows]
    semantics = {c.candidate_id: c.semantic_evidence.cosine_similarity for c in raw.values() if c.semantic_evidence}
    context = recovery_context(reviewer_input, ids, selection_resolved=bool(ids), authority='runtime_reviewer_stage_with_repairs_preserved')
    components = reviewer_input.input_interpretation.asserted_components
    inp = {'original_input': reviewer_input.original_term, 'normalized_input': reviewer_input.normalized_term, 'route': 'reviewer', 'structure': 'single_concept', 'input_bound': len(components) == 1 and components[0].component_text == reviewer_input.normalized_term and (not components[0].is_negated), 'force_review': bool(kwargs.get('reviewer_failure_diagnostics')) or e.force_human_review, 'exact_collision': bool(kwargs.get('exact_ambiguity_candidates')) or e.normalized_exact_collision, 'reviewer_resolved': context['selection_resolved'] and context['current_reviewer_candidates_verified'], 'reviewer_provenance': {'scope': 'actual reviewer-stage decision, possibly deterministically repaired; not independent confirmation', 'selected_candidate_ids': ids, 'assessments': [a.model_dump(mode='json') for a in reviewer_decision.candidate_assessments if a.candidate_id in ids]}, 'reviewer_parents': context['reviewer_parent_codes'], 'complete_groups': list(groups), 'retrieval_conflicts': {k: v['retrieval_conflicts'] for k, v in context['candidate_evidence'].items()}}
    if len(pt) == 1 and gate.resolved_pt_name:
        inp['baseline_support'] = support(inp['original_input'], gate.resolved_pt_name)
    ranked = rank(inp['original_input'], pts, groups, semantics, config=config)
    output, audit = choose(inp, ranked, base, config=config)
    if output['pt'] == base['pt'] and output['llt'] == base['llt']:
        return (baseline, gate.model_copy(update={'coverage_acceptance': {**gate.coverage_acceptance, 'development_v1_6': audit}}))
    resolved = output['pt'][0] if len(output['pt']) == 1 else None
    current = {r['candidate_id']: r for rows in groups.values() for r in rows}
    if output['llt']:
        identifier = 'LLT:' + output['llt'][0]
        row = current[identifier]
        repaired = ReviewerDecision(term_id=baseline.term_id, review_round=baseline.review_round, decision_type='matched_single', selected_candidate_ids=[identifier], component_matches=[ComponentCandidateMatch(component_id=components[0].component_id, candidate_id=identifier, pt_code=row['pt_code'], rationale='Input-supported current LLT selected by opt-in development policy.')], candidate_assessments=[CandidateAssessment(candidate_id=identifier, verdict='supported', assessment_authority='deterministic_gate', rationale='Input-bound development discrimination.')], decision_rationale='Development v1.6 input discrimination; not an LLM acceptance.')
        shortlist = []
        pt_groups = []
        action = 'auto_accepted'
    else:
        allowed = [resolved] if resolved else [r['pt_code'] for r in ranked['pt_rankings'][:4]]
        shortlist = []
        pt_groups = []
        for p in allowed:
            rr = [r for r in ranked['groups'].get(p, []) if not ranked['features'][r['candidate_id']]['conflicts'] and (not ranked['features'][r['candidate_id']]['unsupported_specificity'])][:4]
            candidates = [_as_human_review_candidate(typed[r['candidate_id']], reviewer_candidate_ids=set(ids), reranker_shortlisted_ids={x['candidate_id'] for x in rr}) for r in rr]
            shortlist.extend(candidates)
            if candidates:
                pt_groups.append(HumanReviewPTGroup(pt_code=p, pt_name=candidates[0].pt_name, candidate_ids=[c.candidate_id for c in candidates], llt_candidates=candidates))
        name = next((p['pt_name'] for p in pts if p['pt_code'] == resolved), None)
        repaired = _deferred_decision(reviewer_input=reviewer_input, reviewer_decision=reviewer_decision, human_review_candidates=shortlist, pt_resolved=bool(resolved), resolved_pt_code=resolved, resolved_pt_name=name, decision_rationale='Development input-support assessment requires review.', human_review_reason='Unsupported attributes or indistinguishable LLT wording.')
        action = 'deferred_low_confidence' if resolved else 'deferred_pt_uncertainty'
    payload = gate.model_dump(mode='python')
    payload.update(version=config.version, action=action, gate_action=action, policy_configuration={**gate.policy_configuration, 'development_v1_6': config.to_dict()}, pt_top_code=ranked['pt_rankings'][0]['pt_code'] if ranked['pt_rankings'] else None, pt_top_score=ranked['pt_rankings'][0]['score'] if ranked['pt_rankings'] else None, pt_top_margin=audit.get('PT_margin'), pt_minimum_score=config.pt_score, pt_minimum_margin=config.pt_margin, minimum_score=config.llt_score, minimum_margin=config.llt_margin, top_score=audit.get('LLT_score'), llt_top_score=audit.get('LLT_score'), top_margin=audit.get('LLT_margin'), llt_top_margin=audit.get('LLT_margin'), resolved_pt_code=resolved, resolved_pt_name=next((p['pt_name'] for p in pts if p['pt_code'] == resolved), None), human_review_candidates=shortlist, human_review_pt_groups=pt_groups, ranked_candidates=[_as_human_review_candidate(r, reviewer_candidate_ids=set(ids), reranker_shortlisted_ids={r.candidate_id}) for r in typed.values()], pt_confidence_passed=bool(resolved), decision_authority='deterministic_input_discrimination', pt_confidence_failure_reasons=[] if resolved else audit['pt_reasons'] or ['development_input_support_or_confidence_unresolved'], reason='Opt-in development input discrimination: ' + audit['path'], gate_reason='Opt-in development input discrimination: ' + audit['path'], deterministic_repair_reasons=gate.deterministic_repair_reasons + ['development_v1_6_input_discrimination'], coverage_acceptance={**gate.coverage_acceptance, 'development_v1_6': audit})
    return (repaired, LLTSelectionGateResult.model_validate(payload))

def create_development_gate_node(*, reranker, config=DEVELOPMENT, capture=None, **kwargs):

    def note(stage, state):
        try:
            if capture:
                capture(stage, state, config)
        except Exception:
            pass

    def node(state):
        note('before_development_gate', state)
        exact = state.get('exact_result')
        try:
            decision, gate = apply_development_gate(reviewer_input=state['reviewer_input'], reviewer_decision=state['reviewer_decision'], reranker=reranker, config=config, exact_match_found=bool(exact and exact.found), exact_ambiguity_candidates=tuple(exact.ambiguous_candidates) if exact else (), reviewer_failure_diagnostics=state.get('reviewer_failure_diagnostics'), **kwargs)
        except Exception as error:
            note('development_gate_exception', {**state, 'integration_exception': {'failing_node': 'development_gate', 'exception_type': type(error).__name__, 'message': str(error), 'traceback': traceback.format_exc()}})
            raise
        update = {'reviewer_decision': decision, 'llt_selection_gate_result': gate}
        note('after_development_gate', {**state, **update})
        return update
    return node
