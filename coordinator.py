from runtime_resources import resource as asp_resource
from collections.abc import Sequence
from typing import Any
from reviewer_validation import validate_reviewer_decision
from schemas import CoordinatorResult, CoordinatorRoute, EvidenceRequest, EvidenceRequestHistoryEntry, ReviewerDecision, ReviewerDecisionType, ReviewerInput
from state import StandardizationState
DEFAULT_MAX_REVIEW_ROUNDS = 3

class CoordinatorError(ValueError):
    """
    Raised when the coordinator receives an
    inconsistent workflow state.
    """

def _normalize_query_term(query_term: str | None) -> str | None:
    """
    Normalize a focused-search query only for
    request-signature comparison.

    This does not modify the actual retrieval
    query.
    """
    if query_term is None:
        return None
    return ' '.join(query_term.casefold().split())

def _create_request_signature(request: EvidenceRequest) -> tuple[str, str, str | None, int | None, tuple[str, ...]]:
    """
    Create a deterministic request signature.

    request_id and reason are intentionally
    excluded because changing those fields does
    not create a new retrieval operation.
    """
    return (request.source.value, request.action.value, _normalize_query_term(request.query_term), request.requested_top_k, tuple(sorted(request.candidate_ids_of_interest)))

def _get_previous_request_signatures(term_id: str, request_history: Sequence[EvidenceRequestHistoryEntry]) -> set[tuple[str, str, str | None, int | None, tuple[str, ...]]]:
    """
    Return signatures for requests previously
    completed for the current term.
    """
    return {_create_request_signature(history_entry.request) for history_entry in request_history if history_entry.term_id == term_id}

def _all_candidate_ids(reviewer_input: ReviewerInput) -> list[str]:
    """
    Return all candidate IDs currently available
    to the reviewer.
    """
    return [candidate.candidate_id for candidate in reviewer_input.aggregation_result.candidates]

def coordinate_reviewer_decision(reviewer_input: ReviewerInput, reviewer_decision: ReviewerDecision, request_history: Sequence[EvidenceRequestHistoryEntry]=(), max_review_rounds: int=DEFAULT_MAX_REVIEW_ROUNDS, supplemental_candidates: Sequence[Any]=(), *, deterministic_gate=None) -> CoordinatorResult:
    """
    Convert one validated reviewer decision into
    a deterministic workflow route.

    This function does not execute retrieval.
    """
    if max_review_rounds < 1:
        raise CoordinatorError('max_review_rounds must be at least 1.')
    validated_decision = validate_reviewer_decision(reviewer_input=reviewer_input, deterministic_gate=deterministic_gate, reviewer_decision=reviewer_decision, supplemental_candidates=list(supplemental_candidates))
    decision_type = validated_decision.decision_type
    final_decision_types = {ReviewerDecisionType.MATCHED_SINGLE, ReviewerDecisionType.MATCHED_MULTIPLE, ReviewerDecisionType.NO_SUPPORTED_MATCH}
    if decision_type in final_decision_types:
        return CoordinatorResult(term_id=reviewer_input.term_id, review_round=reviewer_input.review_round, source_decision_type=decision_type, route=CoordinatorRoute.FINALIZE, selected_candidate_ids=list(validated_decision.selected_candidate_ids))
    if decision_type in {ReviewerDecisionType.HUMAN_REVIEW, ReviewerDecisionType.MATCHED_PT_LLT_HUMAN_REVIEW}:
        return CoordinatorResult(term_id=reviewer_input.term_id, review_round=reviewer_input.review_round, source_decision_type=decision_type, route=CoordinatorRoute.HUMAN_REVIEW, human_review_candidate_ids=list(validated_decision.human_review_candidate_ids), human_review_reason=validated_decision.human_review_reason)
    if decision_type != ReviewerDecisionType.REQUEST_MORE_EVIDENCE:
        raise CoordinatorError(f'Unsupported reviewer decision type: {decision_type!r}')
    if reviewer_input.review_round >= max_review_rounds:
        return CoordinatorResult(term_id=reviewer_input.term_id, review_round=reviewer_input.review_round, source_decision_type=decision_type, route=CoordinatorRoute.HUMAN_REVIEW, human_review_candidate_ids=_all_candidate_ids(reviewer_input), human_review_reason='The maximum number of automated review rounds was reached while the reviewer was still requesting additional evidence.')
    previous_signatures = _get_previous_request_signatures(term_id=reviewer_input.term_id, request_history=request_history)
    repeated_requests = [request for request in validated_decision.evidence_requests if _create_request_signature(request) in previous_signatures]
    if repeated_requests:
        repeated_request_ids = [request.request_id for request in repeated_requests]
        return CoordinatorResult(term_id=reviewer_input.term_id, review_round=reviewer_input.review_round, source_decision_type=decision_type, route=CoordinatorRoute.HUMAN_REVIEW, human_review_candidate_ids=_all_candidate_ids(reviewer_input), human_review_reason=f'The reviewer repeated evidence requests that had already been executed. Repeated request IDs: {sorted(repeated_request_ids)}')
    return CoordinatorResult(term_id=reviewer_input.term_id, review_round=reviewer_input.review_round, source_decision_type=decision_type, route=CoordinatorRoute.EXECUTE_EVIDENCE_REQUESTS, pending_evidence_requests=list(validated_decision.evidence_requests), next_review_round=reviewer_input.review_round + 1)

def create_coordinator_node(max_review_rounds: int=DEFAULT_MAX_REVIEW_ROUNDS):
    """
    Create the LangGraph-compatible coordinator
    node with a configured round limit.
    """
    if max_review_rounds < 1:
        raise ValueError('max_review_rounds must be at least 1.')

    def coordinator_node(state: StandardizationState) -> dict[str, object]:
        reviewer_input = state.get('reviewer_input')
        if reviewer_input is None:
            raise CoordinatorError('Coordinator requires reviewer_input.')
        reviewer_decision = state.get('reviewer_decision')
        if reviewer_decision is None:
            raise CoordinatorError('Coordinator requires a validated reviewer_decision.')
        request_history = state.get('evidence_request_history', [])
        gate_result = state.get('llt_selection_gate_result')
        supplemental_candidates = gate_result.human_review_candidates + gate_result.ranked_candidates if gate_result is not None else []
        result = coordinate_reviewer_decision(reviewer_input=reviewer_input, deterministic_gate=gate_result if gate_result is not None and 'assessment_scope_v1_7_1' in gate_result.coverage_acceptance else None, reviewer_decision=reviewer_decision, supplemental_candidates=supplemental_candidates, request_history=request_history, max_review_rounds=max_review_rounds)
        return {'coordinator_result': result, 'pending_evidence_requests': list(result.pending_evidence_requests)}
    return coordinator_node

def route_after_coordinator(state: StandardizationState) -> str:
    """
    LangGraph conditional-edge routing function.

    Returned values correspond to the node names
    that will be added to the graph later.
    """
    coordinator_result = state.get('coordinator_result')
    if coordinator_result is None:
        raise CoordinatorError('Coordinator routing requires coordinator_result.')
    return coordinator_result.route.value
