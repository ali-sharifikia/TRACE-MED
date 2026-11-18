from __future__ import annotations
from runtime_resources import resource as asp_resource
from pt_reranking import build_component_pt_rankings
from schemas import CandidateAssessment, CandidateReviewVerdict, InputStructureType, ReviewerDecision, ReviewerDecisionType, ReviewerInput

def apply_pt_disagreement_gate(reviewer_input: ReviewerInput, reviewer_decision: ReviewerDecision, *, candidate_pt_top_k: int=5) -> tuple[ReviewerDecision, list[str]]:
    """
    Escalate a single-concept matched decision when
    the reviewer-selected PT disagrees with the
    deterministic reranker's highest-ranked PT.

    The reranker never replaces the reviewer. It is
    used only as an independent disagreement signal.
    """
    if candidate_pt_top_k < 1:
        raise ValueError('candidate_pt_top_k must be at least 1.')
    if reviewer_decision.decision_type != ReviewerDecisionType.MATCHED_SINGLE:
        return (reviewer_decision, [])
    if reviewer_input.input_interpretation.structure_type != InputStructureType.SINGLE_CONCEPT:
        return (reviewer_decision, [])
    if len(reviewer_decision.selected_candidate_ids) != 1:
        return (reviewer_decision, [])
    candidate_by_id = {candidate.candidate_id: candidate for candidate in reviewer_input.aggregation_result.candidates}
    selected_candidate_id = reviewer_decision.selected_candidate_ids[0]
    selected_candidate = candidate_by_id.get(selected_candidate_id)
    if selected_candidate is None:
        return (reviewer_decision, [])
    component_rankings = build_component_pt_rankings(reviewer_input)
    if len(component_rankings) != 1:
        return (reviewer_decision, [])
    pt_rankings = component_rankings[0].get('pt_rankings', [])
    if not pt_rankings:
        return (reviewer_decision, [])
    reranker_winner = pt_rankings[0]
    reviewer_pt_code = selected_candidate.pt_code
    reranker_pt_code = reranker_winner['pt_code']
    if reviewer_pt_code == reranker_pt_code:
        return (reviewer_decision, [])
    deferred_pt_codes = [ranking['pt_code'] for ranking in pt_rankings[:candidate_pt_top_k]]
    if reviewer_pt_code not in deferred_pt_codes:
        deferred_pt_codes.append(reviewer_pt_code)
    group_by_pt_code = {group.pt_code: group for group in reviewer_input.pt_candidate_groups}
    human_review_candidate_ids = []
    for pt_code in deferred_pt_codes:
        group = group_by_pt_code.get(pt_code)
        if group is None:
            continue
        for candidate_id in group.candidate_ids:
            if candidate_id not in human_review_candidate_ids:
                human_review_candidate_ids.append(candidate_id)
    if selected_candidate_id not in human_review_candidate_ids:
        human_review_candidate_ids.append(selected_candidate_id)
    human_review_id_set = set(human_review_candidate_ids)
    candidate_assessments = []
    for candidate in reviewer_input.aggregation_result.candidates:
        if candidate.candidate_id in human_review_id_set:
            verdict = CandidateReviewVerdict.UNCERTAIN
            rationale = f"'{candidate.llt_name}' belongs to PT '{candidate.pt_name}', which is retained as a plausible PT option because the independent reviewer and deterministic PT reranker disagreed."
        else:
            verdict = CandidateReviewVerdict.NOT_SUPPORTED
            rationale = 'This candidate is outside the PT alternatives retained for resolving the reviewer-reranker disagreement.'
        candidate_assessments.append(CandidateAssessment(candidate_id=candidate.candidate_id, verdict=verdict, rationale=rationale))
    reranker_pt_name = reranker_winner['pt_name']
    reviewer_pt_ranking = next((ranking for ranking in pt_rankings if ranking['pt_code'] == reviewer_pt_code), None)
    reviewer_pt_rank = reviewer_pt_ranking['rank'] if reviewer_pt_ranking is not None else None
    repaired_decision = ReviewerDecision(term_id=reviewer_decision.term_id, review_round=reviewer_decision.review_round, decision_type=ReviewerDecisionType.HUMAN_REVIEW, selected_candidate_ids=[], component_matches=[], candidate_assessments=candidate_assessments, evidence_requests=[], human_review_candidate_ids=human_review_candidate_ids, decision_rationale=f"The clinical reviewer selected PT '{selected_candidate.pt_name}', while the independent deterministic PT reranker placed PT '{reranker_pt_name}' first. Because these independent PT-selection signals disagree, no PT or LLT is finalized automatically.", multiple_match_explanation=None, human_review_reason='Expert MedDRA coding review is required to resolve a PT-level disagreement between the clinical reviewer and the deterministic retrieval-based reranker.')
    repair_actions = [f'Escalated reviewer-reranker PT disagreement to human review. Reviewer PT: {reviewer_pt_code}; reranker PT: {reranker_pt_code}; reviewer PT rank: {reviewer_pt_rank}; retained PTs: {deferred_pt_codes}.']
    return (repaired_decision, repair_actions)
