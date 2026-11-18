from runtime_resources import resource as asp_resource
from schemas import CoordinatorRoute, EvidenceExecutionBatchStatus, EvidenceExecutionStatus, HumanReviewCandidate, HumanReviewPTGroup, MatchMethod, ReviewerDecisionType, StandardizedConcept, TechnicalFailureDetail, WorkflowDecisionOrigin, WorkflowOutcome, WorkflowOutcomeStatus, ResolvedPTConcept
from state import StandardizationState

class WorkflowOutcomeError(ValueError):
    """
    Raised when a terminal outcome cannot safely
    be constructed from the workflow state.
    """

def _get_term_context(state: StandardizationState) -> tuple[str, str, str, int]:
    """
    Return the common fields required by every
    workflow outcome.
    """
    term_id = state.get('term_id')
    if term_id is None:
        raise WorkflowOutcomeError('Workflow outcome requires term_id.')
    term_input = state.get('term_input')
    if term_input is None:
        raise WorkflowOutcomeError('Workflow outcome requires term_input.')
    normalization = state.get('normalization')
    if normalization is None:
        raise WorkflowOutcomeError('Workflow outcome requires the normalization result.')
    return (term_id, term_input.term, normalization.normalized_term, term_input.frequency)

def _concept_data(candidate) -> dict:
    """
    Extract fields inherited from MeddraConcept.
    """
    return {'candidate_id': candidate.candidate_id, 'llt_code': candidate.llt_code, 'llt_name': candidate.llt_name, 'pt_code': candidate.pt_code, 'pt_name': candidate.pt_name, 'is_current': candidate.is_current}

def _supporting_methods(candidate) -> list[MatchMethod]:
    """
    Convert aggregated evidence sources to the
    public match-method enumeration.
    """
    evidence_sources = getattr(candidate, 'evidence_sources', None)
    if evidence_sources is None:
        return [MatchMethod.LLT_RERANKER]
    return [MatchMethod(source.value) for source in evidence_sources]

def _resolved_pt_concept(aggregation_result, candidate_ids: list[str], supplemental_candidates: list[HumanReviewCandidate] | None=None, gate_result=None, input_text: str | None=None) -> ResolvedPTConcept:
    """
    Resolve one PT from multiple deferred LLTs that
    are already known to belong to the same PT.
    """
    candidate_by_id = {candidate.candidate_id: candidate for candidate in aggregation_result.candidates}
    supplemental_candidate_ids: set[str] = set()
    for candidate in supplemental_candidates or []:
        supplemental_candidate_ids.add(candidate.candidate_id)
        candidate_by_id.setdefault(candidate.candidate_id, candidate)
    missing_candidate_ids = [candidate_id for candidate_id in candidate_ids if candidate_id not in candidate_by_id]
    if missing_candidate_ids:
        raise WorkflowOutcomeError(f'PT resolution references candidates missing from aggregation and the LLT selection gate: {sorted(missing_candidate_ids)}')
    candidates = [candidate_by_id[candidate_id] for candidate_id in candidate_ids]
    pt_identity = {(candidate.pt_code, candidate.pt_name) for candidate in candidates}
    if not candidates:
        audit = getattr(gate_result, 'coverage_acceptance', {})
        if not gate_result or not gate_result.pt_confidence_passed or (not gate_result.resolved_pt_code) or (not audit.get('pt_accepted')) or (audit.get('input_text') != input_text) or (audit.get('selected_pt') != gate_result.resolved_pt_code) or audit.get('pt_reasons') or (not audit.get('reference_complete')) or gate_result.reviewer_failure_diagnostics:
            raise WorkflowOutcomeError('Zero-LLT PT resolution requires independent, input-bound verified PT acceptance.')
        return ResolvedPTConcept(pt_code=gate_result.resolved_pt_code, pt_name=gate_result.resolved_pt_name, supporting_candidate_ids=[], supporting_methods=[MatchMethod.PT_RERANKER])
    if len(pt_identity) != 1:
        raise WorkflowOutcomeError(f'A PT-resolved LLT deferral requires all deferred candidates to belong to the same PT. Observed PTs: {sorted(pt_identity)}')
    pt_code, pt_name = next(iter(pt_identity))
    supporting_methods = sorted({method for candidate in candidates for method in _supporting_methods(candidate)} | ({MatchMethod.LLT_RERANKER} if set(candidate_ids) & supplemental_candidate_ids else set()), key=lambda method: method.value)
    return ResolvedPTConcept(pt_code=pt_code, pt_name=pt_name, supporting_candidate_ids=list(candidate_ids), supporting_methods=supporting_methods)

def _policy_audit(state: StandardizationState) -> dict:
    gate = state.get('llt_selection_gate_result')
    interpretation = state.get('input_interpretation')
    if interpretation is None and state.get('reviewer_input') is not None:
        interpretation = state['reviewer_input'].input_interpretation
    repairs = list(getattr(interpretation, 'deterministic_repair_reasons', []))
    exact_result = state.get('exact_result')
    exact_audit = {'exact_match_provenance': exact_result.exact_match_provenance, 'exact_match_version': exact_result.exact_match_version} if exact_result is not None else {}
    failure = state.get('reviewer_failure_diagnostics') or getattr(gate, 'reviewer_failure_diagnostics', {})
    exact_audit['reviewer_failure_diagnostics'] = failure
    exact_audit['decomposition_diagnostics'] = getattr(interpretation, 'decomposition_diagnostics', {})
    exact_audit['compound_acceptance'] = getattr(gate, 'compound_acceptance', {})
    exact_audit['pt_coverage_recovery'] = getattr(gate, 'pt_coverage_recovery', {})
    exact_audit['coverage_acceptance'] = getattr(gate, 'coverage_acceptance', {})
    if gate is None:
        return {'decision_authority': failure.get('authority', 'llm_with_component_validation'), 'deterministic_repair_reasons': repairs, **exact_audit}
    return {**exact_audit, 'decision_authority': gate.decision_authority or 'llm_with_deterministic_validation', 'policy_configuration': gate.policy_configuration, 'deterministic_repair_reasons': list(dict.fromkeys(repairs + gate.deterministic_repair_reasons)), 'shortlist_pruning_explanations': gate.shortlist_pruning_explanations}

def exact_match_outcome_node(state: StandardizationState) -> dict[str, WorkflowOutcome]:
    """
    Create the terminal outcome for a term that
    matched MedDRA during exact matching.
    """
    term_id, original_term, normalized_term, frequency = _get_term_context(state)
    exact_result = state.get('exact_result')
    if exact_result is None:
        raise WorkflowOutcomeError('Exact outcome requires exact_result.')
    if not exact_result.found or exact_result.concept is None:
        raise WorkflowOutcomeError('Exact outcome received a term without an exact match.')
    concept = StandardizedConcept(**_concept_data(exact_result.concept), supporting_methods=[MatchMethod.EXACT])
    outcome = WorkflowOutcome(term_id=term_id, original_term=original_term, normalized_term=normalized_term, frequency=frequency, status=WorkflowOutcomeStatus.MATCHED, decision_origin=WorkflowDecisionOrigin.EXACT, review_round=None, reviewer_decision_type=None, decision_authority='exact_matcher', exact_match_provenance=exact_result.exact_match_provenance, exact_match_version=exact_result.exact_match_version, selected_concepts=[concept], human_review_candidate_ids=[], technical_failures=[], decision_rationale='The normalized input term exactly matched a current MedDRA LLT.')
    return {'workflow_outcome': outcome}

def reviewer_final_outcome_node(state: StandardizationState) -> dict[str, WorkflowOutcome]:
    """
    Create a terminal matched or no-supported-
    match outcome from the reviewer decision.
    """
    term_id, original_term, normalized_term, frequency = _get_term_context(state)
    exact_result = state.get('exact_result')
    if exact_result is None:
        raise WorkflowOutcomeError('Reviewer outcome requires exact_result.')
    if exact_result.found:
        raise WorkflowOutcomeError('An exact-matched term must use the exact outcome node.')
    coordinator_result = state.get('coordinator_result')
    if coordinator_result is None:
        raise WorkflowOutcomeError('Reviewer outcome requires coordinator_result.')
    if coordinator_result.route != CoordinatorRoute.FINALIZE:
        raise WorkflowOutcomeError('Coordinator did not route this term to finalization.')
    reviewer_decision = state.get('reviewer_decision')
    if reviewer_decision is None:
        raise WorkflowOutcomeError('Reviewer outcome requires reviewer_decision.')
    if reviewer_decision.term_id != term_id:
        raise WorkflowOutcomeError('Reviewer decision term_id does not match state.')
    if reviewer_decision.review_round != coordinator_result.review_round:
        raise WorkflowOutcomeError('Reviewer and coordinator review rounds do not match.')
    if reviewer_decision.decision_type != coordinator_result.source_decision_type:
        raise WorkflowOutcomeError('Reviewer decision type does not match coordinator_result.')
    if reviewer_decision.selected_candidate_ids != coordinator_result.selected_candidate_ids:
        raise WorkflowOutcomeError('Reviewer and coordinator selected candidate IDs do not match.')
    aggregation_result = state.get('aggregation_result')
    if aggregation_result is None:
        raise WorkflowOutcomeError('Reviewer outcome requires aggregation_result.')
    candidate_by_id = {candidate.candidate_id: candidate for candidate in aggregation_result.candidates}
    if reviewer_decision.decision_type == ReviewerDecisionType.MATCHED_MULTIPLE:
        from compound_validation import COMPOUND_POLICY_VERSION
        gate = state.get('llt_selection_gate_result')
        audit = getattr(gate, 'compound_acceptance', {})
        if not audit.get('accepted') or audit.get('version') != COMPOUND_POLICY_VERSION or audit.get('term_id') != term_id or (audit.get('input_text') != normalized_term) or (audit.get('selected_candidate_ids') != reviewer_decision.selected_candidate_ids) or (audit.get('component_mappings') != [{'component_id': m.component_id, 'candidate_id': m.candidate_id, 'pt_code': m.pt_code} for m in reviewer_decision.component_matches]) or (not audit.get('components')) or (not all((c.get('accepted') for c in audit['components']))) or (audit.get('decomposition', {}).get('status') != 'accepted') or state.get('reviewer_failure_diagnostics') or getattr(gate, 'reviewer_failure_diagnostics', {}):
            raise WorkflowOutcomeError('Automatic multiple mapping requires a passing, input-bound component acceptance gate.')
    selected_concepts = []
    gate = state.get('llt_selection_gate_result')
    if gate is not None:
        for candidate in gate.ranked_candidates:
            if candidate.candidate_id not in candidate_by_id and candidate.candidate_id in reviewer_decision.selected_candidate_ids:
                audit = gate.coverage_acceptance
                development_authorized = False
                if 'assessment_scope_v1_7_1' in audit:
                    from assessment_scope import expanded_selection_authorized
                    inp = state.get('reviewer_input')
                    if inp is not None and inp.term_id == term_id and (inp.normalized_term == normalized_term) and (not state.get('reviewer_failure_diagnostics')):
                        development_authorized = expanded_selection_authorized(inp, reviewer_decision, gate, candidate)
                if not development_authorized and (not audit.get('applied') or not audit.get('llt_accepted') or (not audit.get('reference_complete')) or (audit.get('input_text') != normalized_term) or (audit.get('selected_llt') != candidate.llt_code) or (audit.get('selected_pt') != candidate.pt_code) or gate.reviewer_failure_diagnostics):
                    raise WorkflowOutcomeError('Expanded automatic LLT requires a passing input-bound current-reference gate.')
                candidate_by_id[candidate.candidate_id] = candidate
    for candidate_id in reviewer_decision.selected_candidate_ids:
        candidate = candidate_by_id.get(candidate_id)
        if candidate is None:
            raise WorkflowOutcomeError(f'A selected reviewer candidate is missing from aggregation: {candidate_id}')
        selected_concepts.append(StandardizedConcept(**_concept_data(candidate), supporting_methods=_supporting_methods(candidate)))
    if reviewer_decision.decision_type in {ReviewerDecisionType.MATCHED_SINGLE, ReviewerDecisionType.MATCHED_MULTIPLE}:
        status = WorkflowOutcomeStatus.MATCHED
    elif reviewer_decision.decision_type == ReviewerDecisionType.NO_SUPPORTED_MATCH:
        status = WorkflowOutcomeStatus.NO_SUPPORTED_MATCH
    else:
        raise WorkflowOutcomeError('Finalize route received a non-final reviewer decision.')
    outcome = WorkflowOutcome(term_id=term_id, original_term=original_term, normalized_term=normalized_term, frequency=frequency, status=status, **_policy_audit(state), decision_origin=WorkflowDecisionOrigin.REVIEWER, review_round=reviewer_decision.review_round, reviewer_decision_type=reviewer_decision.decision_type, selected_concepts=selected_concepts, human_review_candidate_ids=[], technical_failures=[], decision_rationale=reviewer_decision.decision_rationale)
    return {'workflow_outcome': outcome}

def human_review_outcome_node(state: StandardizationState) -> dict[str, WorkflowOutcome]:
    """
    Create a terminal unresolved outcome for a
    term routed to human review.
    """
    term_id, original_term, normalized_term, frequency = _get_term_context(state)
    coordinator_result = state.get('coordinator_result')
    if coordinator_result is None:
        raise WorkflowOutcomeError('Human-review outcome requires coordinator_result.')
    if coordinator_result.route != CoordinatorRoute.HUMAN_REVIEW:
        raise WorkflowOutcomeError('Coordinator did not route this term to human review.')
    reviewer_decision = state.get('reviewer_decision')
    reviewer_originated_types = {ReviewerDecisionType.HUMAN_REVIEW, ReviewerDecisionType.MATCHED_PT_LLT_HUMAN_REVIEW}
    if coordinator_result.source_decision_type in reviewer_originated_types:
        decision_origin = WorkflowDecisionOrigin.REVIEWER
        reviewer_decision_type = coordinator_result.source_decision_type
        if reviewer_decision is None:
            raise WorkflowOutcomeError('Reviewer-originated human review requires reviewer_decision.')
        decision_rationale = reviewer_decision.decision_rationale
    else:
        decision_origin = WorkflowDecisionOrigin.COORDINATOR
        reviewer_decision_type = coordinator_result.source_decision_type
        decision_rationale = coordinator_result.human_review_reason
    if not decision_rationale:
        raise WorkflowOutcomeError('Human-review outcome requires a decision rationale.')
    resolved_pt_concepts = []
    detailed_human_review_candidates: list[HumanReviewCandidate] = []
    detailed_human_review_pt_groups: list[HumanReviewPTGroup] = []
    gate_result = state.get('llt_selection_gate_result')
    if gate_result is not None:
        detailed_by_id = {candidate.candidate_id: candidate for candidate in gate_result.human_review_candidates}
        detailed_human_review_candidates = [detailed_by_id[candidate_id] for candidate_id in coordinator_result.human_review_candidate_ids if candidate_id in detailed_by_id]
        retained_ids = {candidate.candidate_id for candidate in detailed_human_review_candidates}
        for group in gate_result.human_review_pt_groups:
            nested = [candidate for candidate in group.llt_candidates if candidate.candidate_id in retained_ids]
            if nested:
                detailed_human_review_pt_groups.append(group.model_copy(update={'candidate_ids': [candidate.candidate_id for candidate in nested], 'llt_candidates': nested}))
    if reviewer_decision_type == ReviewerDecisionType.MATCHED_PT_LLT_HUMAN_REVIEW:
        aggregation_result = state.get('aggregation_result')
        if aggregation_result is None:
            raise WorkflowOutcomeError('PT-resolved LLT human review requires aggregation_result.')
        resolved_pt_concepts.append(_resolved_pt_concept(aggregation_result=aggregation_result, candidate_ids=list(coordinator_result.human_review_candidate_ids), supplemental_candidates=detailed_human_review_candidates, gate_result=gate_result, input_text=normalized_term))
    outcome = WorkflowOutcome(term_id=term_id, original_term=original_term, normalized_term=normalized_term, frequency=frequency, status=WorkflowOutcomeStatus.HUMAN_REVIEW, decision_origin=decision_origin, **_policy_audit(state), review_round=coordinator_result.review_round, reviewer_decision_type=reviewer_decision_type, selected_concepts=[], resolved_pt_concepts=resolved_pt_concepts, human_review_candidate_ids=list(coordinator_result.human_review_candidate_ids), human_review_candidates=detailed_human_review_candidates, human_review_pt_groups=detailed_human_review_pt_groups, technical_failures=[], decision_rationale=decision_rationale)
    return {'workflow_outcome': outcome}

def technical_failure_outcome_node(state: StandardizationState) -> dict[str, WorkflowOutcome]:
    """
    Create a terminal technical-failure outcome
    for a partial or failed retrieval batch.
    """
    term_id, original_term, normalized_term, frequency = _get_term_context(state)
    execution_batch = state.get('evidence_execution_batch')
    if execution_batch is None:
        raise WorkflowOutcomeError('Technical-failure outcome requires evidence_execution_batch.')
    if execution_batch.status == EvidenceExecutionBatchStatus.COMPLETED:
        raise WorkflowOutcomeError('A completed execution batch cannot produce a technical-failure outcome.')
    if execution_batch.term_id != term_id:
        raise WorkflowOutcomeError('Execution batch term_id does not match state.')
    failed_executions = [execution for execution in execution_batch.executions if execution.status == EvidenceExecutionStatus.FAILED]
    if not failed_executions:
        raise WorkflowOutcomeError('Technical-failure batch contains no failed executions.')
    failure_details = [TechnicalFailureDetail(request_id=execution.request.request_id, source=execution.request.source, action=execution.request.action, effective_query_term=execution.effective_query_term, error_message=execution.error_message or 'Retrieval failed without an error message.') for execution in failed_executions]
    outcome = WorkflowOutcome(term_id=term_id, original_term=original_term, normalized_term=normalized_term, frequency=frequency, status=WorkflowOutcomeStatus.TECHNICAL_FAILURE, decision_origin=WorkflowDecisionOrigin.SYSTEM, review_round=execution_batch.review_round, reviewer_decision_type=ReviewerDecisionType.REQUEST_MORE_EVIDENCE, selected_concepts=[], human_review_candidate_ids=[], technical_failures=failure_details, decision_rationale='One or more requested retrieval operations failed technically. No clinical standardization decision was made from the incomplete evidence.')
    return {'workflow_outcome': outcome}
