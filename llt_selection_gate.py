from __future__ import annotations
from runtime_resources import resource as asp_resource
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any
from llt_reranking import LLTRankingEntry, PTConditionedLLTReranker
from pt_reranking import CONFLICT_FIELDS, build_component_pt_rankings
from schemas import CandidateAssessment, CandidateReviewVerdict, HumanReviewCandidate, HumanReviewPTGroup, LLTSelectionGateAction, LLTSelectionGateResult, ReviewerDecision, ReviewerDecisionType, ReviewerInput
from state import StandardizationState
from llt_reranking import normalize_llt_text
from dataclasses import replace
from pipeline_policy import DEFAULT_POLICY, PolicyConfig
from adaptive_review_policy import prepare_policy_evidence, apply_policy
LLT_SELECTION_GATE_VERSION = '1.5.0'
LLT_AUTO_ACCEPT_MINIMUM_SCORE = 0.9
LLT_AUTO_ACCEPT_MINIMUM_MARGIN = 0.1
LLT_AUTO_ACCEPT_PT_MINIMUM_SCORE = DEFAULT_POLICY.pt_minimum_score
LLT_AUTO_ACCEPT_PT_MINIMUM_MARGIN = DEFAULT_POLICY.pt_minimum_margin
LLT_DISAGREEMENT_SHORTLIST_SIZE = 5
LLT_DEFERRAL_ENRICHMENT_SIZE = 3
PT_UNCERTAINTY_SHORTLIST_GROUPS = 4
PT_UNCERTAINTY_LLTS_PER_GROUP = 2
SHORTLIST_SCORE_DELTA = 0.05
MAX_HUMAN_REVIEW_CANDIDATES = 8

@dataclass(frozen=True)
class _PTConfidenceDiagnostic:
    passed: bool
    resolved_pt_code: str | None
    resolved_pt_name: str | None
    top_code: str | None
    top_score: float | None
    top_margin: float | None
    failure_reasons: tuple[str, ...]
    component_pt_rankings: tuple[dict[str, Any], ...]

def _deduplicate(values: Iterable[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = str(value).strip()
        if normalized and normalized not in seen:
            result.append(normalized)
            seen.add(normalized)
    return result

def _candidate_margin(rankings: list[LLTRankingEntry]) -> float:
    if len(rankings) < 2:
        return 1.0
    return round(max(0.0, rankings[0].score - rankings[1].score), 6)

def _pt_candidate_margin(rankings: list[dict[str, Any]]) -> float:
    if len(rankings) < 2:
        return 1.0
    return round(max(0.0, float(rankings[0]['reranking_score']) - float(rankings[1]['reranking_score'])), 6)

def _explicit_candidate_conflicts(candidate: Any) -> list[str]:
    fuzzy_evidence = candidate.fuzzy_evidence
    if fuzzy_evidence is None:
        return []
    conflict_fields = (*CONFLICT_FIELDS, 'unsupported_specificity')
    return [f'explicit_{field_name}:{candidate.candidate_id}' for field_name in conflict_fields if bool(getattr(fuzzy_evidence, field_name, False))]

def _evaluate_pt_confidence(*, reviewer_input: ReviewerInput, reviewer_decision: ReviewerDecision, reviewer_candidates: list[Any], minimum_score: float, minimum_margin: float) -> _PTConfidenceDiagnostic:
    failure_reasons: list[str] = []
    pt_identity = {(candidate.pt_code, candidate.pt_name) for candidate in reviewer_candidates}
    resolved_pt_code: str | None = None
    resolved_pt_name: str | None = None
    if len(pt_identity) == 1:
        resolved_pt_code, resolved_pt_name = next(iter(pt_identity))
    else:
        failure_reasons.append('resolved_pt_is_not_unique')
    positive_components = [component for component in reviewer_input.input_interpretation.asserted_components if not component.is_negated]
    reviewer_ids = {candidate.candidate_id for candidate in reviewer_candidates}
    matched_component_ids = _deduplicate((component_match.component_id for component_match in reviewer_decision.component_matches if component_match.candidate_id in reviewer_ids))
    component_id: str | None = None
    if len(matched_component_ids) == 1:
        component_id = matched_component_ids[0]
    elif not matched_component_ids and len(positive_components) == 1:
        component_id = positive_components[0].component_id
    else:
        failure_reasons.append('resolved_component_is_not_unique')
    component_rankings = build_component_pt_rankings(reviewer_input)
    selected_component_ranking = next((ranking for ranking in component_rankings if ranking['component_id'] == component_id), None)
    pt_rankings = list(selected_component_ranking['pt_rankings']) if selected_component_ranking is not None else []
    top_code: str | None = None
    top_score: float | None = None
    top_margin: float | None = None
    if pt_rankings:
        top_ranking = pt_rankings[0]
        top_code = str(top_ranking['pt_code'])
        top_score = float(top_ranking['reranking_score'])
        top_margin = _pt_candidate_margin(pt_rankings)
        if resolved_pt_code is not None and top_code != resolved_pt_code:
            failure_reasons.append('pt_reranker_top_disagrees_with_resolved_pt')
        if top_score < minimum_score:
            failure_reasons.append('pt_top_score_below_minimum')
        if top_margin < minimum_margin:
            failure_reasons.append('pt_top_margin_below_minimum')
        aggregated_by_id = {candidate.candidate_id: candidate for candidate in reviewer_input.aggregation_result.candidates}
        conflict_candidate_ids = _deduplicate([str(top_ranking['best_candidate_id'])] + [candidate.candidate_id for candidate in reviewer_candidates])
        for candidate_id in conflict_candidate_ids:
            candidate = aggregated_by_id.get(candidate_id)
            if candidate is not None:
                failure_reasons.extend(_explicit_candidate_conflicts(candidate))
        modifier_reasons = top_ranking.get('score_breakdown', {}).get('modifier_conflict_reasons', [])
        failure_reasons.extend((f'explicit_modifier_conflict:{reason}' for reason in modifier_reasons))
    else:
        failure_reasons.append('pt_reranking_diagnostic_unavailable')
    if resolved_pt_code is not None:
        invalid_selected_ids = [candidate.candidate_id for candidate in reviewer_candidates if candidate.pt_code != resolved_pt_code]
        if invalid_selected_ids:
            failure_reasons.append('reviewer_candidate_not_in_resolved_pt')
        invalid_component_pt_codes = [component_match.pt_code for component_match in reviewer_decision.component_matches if component_match.candidate_id in reviewer_ids and component_match.pt_code != resolved_pt_code]
        if invalid_component_pt_codes:
            failure_reasons.append('reviewer_component_pt_does_not_contain_selected_candidate')
    unique_failures = tuple(_deduplicate(failure_reasons))
    return _PTConfidenceDiagnostic(passed=not unique_failures, resolved_pt_code=resolved_pt_code, resolved_pt_name=resolved_pt_name, top_code=top_code, top_score=top_score, top_margin=top_margin, failure_reasons=unique_failures, component_pt_rankings=tuple(pt_rankings))

def _retrieved_candidate_payloads(reviewer_input: ReviewerInput) -> list[dict[str, Any]]:
    return [{'candidate_id': candidate.candidate_id, 'llt_code': candidate.llt_code, 'pt_code': candidate.pt_code, 'source_count': candidate.source_count, 'semantic_similarity': float(candidate.semantic_evidence.cosine_similarity) if candidate.semantic_evidence is not None else 0.0} for candidate in reviewer_input.aggregation_result.candidates]

def _as_human_review_candidate(ranking: LLTRankingEntry, *, reviewer_candidate_ids: set[str], reranker_shortlisted_ids: set[str]) -> HumanReviewCandidate:
    return HumanReviewCandidate(candidate_id=ranking.candidate_id, llt_code=ranking.llt_code, llt_name=ranking.llt_name, pt_code=ranking.pt_code, pt_name=ranking.pt_name, is_current=True, reranker_rank=ranking.rank, reranker_score=ranking.score, was_retrieved=ranking.was_retrieved, retrieval_source_count=ranking.retrieval_source_count, reviewer_selected=ranking.candidate_id in reviewer_candidate_ids, reranker_shortlisted=ranking.candidate_id in reranker_shortlisted_ids)

def _retrieval_only_human_review_candidate(candidate: Any, *, reviewer_selected: bool) -> HumanReviewCandidate:
    return HumanReviewCandidate(candidate_id=candidate.candidate_id, llt_code=candidate.llt_code, llt_name=candidate.llt_name, pt_code=candidate.pt_code, pt_name=candidate.pt_name, is_current=candidate.is_current, reranker_rank=None, reranker_score=None, was_retrieved=True, retrieval_source_count=candidate.source_count, reviewer_selected=reviewer_selected, reranker_shortlisted=False)

def _build_human_review_candidates(*, reviewer_input: ReviewerInput, rankings: list[LLTRankingEntry], human_review_candidate_ids: list[str], reviewer_candidate_ids: list[str], reranker_shortlisted_ids: list[str]) -> list[HumanReviewCandidate]:
    ranking_by_id = {ranking.candidate_id: ranking for ranking in rankings}
    retrieved_by_id = {candidate.candidate_id: candidate for candidate in reviewer_input.aggregation_result.candidates}
    reviewer_id_set = set(reviewer_candidate_ids)
    reranker_id_set = set(reranker_shortlisted_ids)
    result: list[HumanReviewCandidate] = []
    for candidate_id in human_review_candidate_ids:
        ranking = ranking_by_id.get(candidate_id)
        if ranking is not None:
            result.append(_as_human_review_candidate(ranking, reviewer_candidate_ids=reviewer_id_set, reranker_shortlisted_ids=reranker_id_set))
            continue
        retrieved = retrieved_by_id.get(candidate_id)
        if retrieved is None:
            raise ValueError(f'An LLT human-review candidate was not found in either the MedDRA PT reference or aggregated retrieval: {candidate_id}')
        result.append(_retrieval_only_human_review_candidate(retrieved, reviewer_selected=candidate_id in reviewer_id_set))
    return result

def _candidate_retention_evidence(*, reviewer_input: ReviewerInput, candidate: HumanReviewCandidate, reviewer_candidate_ids: set[str], best_score_by_pt: dict[str, float], score_delta: float) -> HumanReviewCandidate:
    aggregated = next((item for item in reviewer_input.aggregation_result.candidates if item.candidate_id == candidate.candidate_id), None)
    reasons: list[str] = []
    methods: list[str] = []
    if candidate.candidate_id in reviewer_candidate_ids:
        reasons.append('reviewer_selected')
    if candidate.reranker_rank == 1:
        reasons.append('llt_reranker_top')
    best_score = best_score_by_pt.get(candidate.pt_code)
    if best_score is not None and candidate.reranker_score is not None and (best_score - candidate.reranker_score <= score_delta):
        reasons.append('within_score_delta')
    if normalize_llt_text(reviewer_input.original_term) == normalize_llt_text(candidate.llt_name):
        reasons.append('normalized_exact_surface')
    if aggregated is not None:
        fuzzy = aggregated.fuzzy_evidence
        semantic = aggregated.semantic_evidence
        umls = aggregated.umls_evidence
        if fuzzy is not None:
            methods.append('fuzzy')
            if fuzzy.rank == 1:
                reasons.append('top_fuzzy_candidate')
            if fuzzy.direct_pt_match_score is not None:
                reasons.append('direct_pt_name_evidence')
                methods.append('direct_pt_fuzzy')
            if fuzzy.introduced_by_pt_expansion:
                reasons.append('pt_to_llt_expansion')
        if semantic is not None:
            methods.append('semantic')
            if semantic.rank == 1:
                reasons.append('top_semantic_candidate')
        if umls is not None:
            methods.append('umls')
            reasons.append('umls_supported')
    return candidate.model_copy(update={'retention_reasons': _deduplicate(reasons), 'introduction_methods': _deduplicate(methods)})

def _structured_shortlist(*, reviewer_input: ReviewerInput, candidates: list[HumanReviewCandidate], reviewer_candidate_ids: list[str], pt_rankings: tuple[dict[str, Any], ...]=(), pt_group_count: int=PT_UNCERTAINTY_SHORTLIST_GROUPS, llts_per_pt: int=PT_UNCERTAINTY_LLTS_PER_GROUP, score_delta: float=SHORTLIST_SCORE_DELTA, maximum_total: int=MAX_HUMAN_REVIEW_CANDIDATES) -> tuple[list[HumanReviewCandidate], list[HumanReviewPTGroup], list[str]]:
    """Apply source-aware retention, deterministic caps, and PT grouping."""
    if min(pt_group_count, llts_per_pt, maximum_total) < 1:
        raise ValueError('Shortlist count limits must be at least 1.')
    candidate_by_id = {candidate.candidate_id: candidate for candidate in candidates}
    pt_rank_by_code = {str(item['pt_code']): int(item.get('rank') or index) for index, item in enumerate(pt_rankings, start=1)}
    pt_score_by_code = {str(item['pt_code']): float(item['reranking_score']) for item in pt_rankings}
    reviewer_ids = set(reviewer_candidate_ids)
    best_score_by_pt: dict[str, float] = {}
    for candidate in candidates:
        if candidate.reranker_score is not None:
            best_score_by_pt[candidate.pt_code] = max(candidate.reranker_score, best_score_by_pt.get(candidate.pt_code, 0.0))
    annotated = {candidate.candidate_id: _candidate_retention_evidence(reviewer_input=reviewer_input, candidate=candidate, reviewer_candidate_ids=reviewer_ids, best_score_by_pt=best_score_by_pt, score_delta=score_delta) for candidate in candidates}
    reason_priority = {'reviewer_selected': 0, 'normalized_exact_surface': 1, 'direct_pt_name_evidence': 2, 'top_fuzzy_candidate': 3, 'top_semantic_candidate': 3, 'umls_supported': 4, 'llt_reranker_top': 5, 'within_score_delta': 6, 'pt_to_llt_expansion': 7}
    leading_pt_codes = [str(item['pt_code']) for item in pt_rankings[:pt_group_count]]
    leading_pt_codes = _deduplicate(leading_pt_codes + [candidate_by_id[candidate_id].pt_code for candidate_id in reviewer_candidate_ids if candidate_id in candidate_by_id])
    eligible = [candidate for candidate in annotated.values() if candidate.pt_code in leading_pt_codes or any((reason in reason_priority for reason in candidate.retention_reasons))]
    eligible.sort(key=lambda candidate: (min([reason_priority[reason] for reason in candidate.retention_reasons] or [8]), pt_rank_by_code.get(candidate.pt_code, 10000), candidate.reranker_rank or 10000, -(candidate.reranker_score or 0.0), candidate.candidate_id))
    retained_ids: list[str] = []
    per_pt_counts: dict[str, int] = {}
    for candidate in eligible:
        is_mandatory = bool(set(candidate.retention_reasons) & {'reviewer_selected', 'normalized_exact_surface', 'direct_pt_name_evidence', 'top_fuzzy_candidate', 'top_semantic_candidate', 'umls_supported'})
        if not is_mandatory and per_pt_counts.get(candidate.pt_code, 0) >= llts_per_pt:
            continue
        if len(retained_ids) >= maximum_total:
            break
        retained_ids.append(candidate.candidate_id)
        per_pt_counts[candidate.pt_code] = per_pt_counts.get(candidate.pt_code, 0) + 1
    retained = [annotated[candidate_id] for candidate_id in retained_ids]
    retained.sort(key=lambda candidate: (pt_rank_by_code.get(candidate.pt_code, 10000), candidate.reranker_rank or 10000, candidate.candidate_id))
    groups: list[HumanReviewPTGroup] = []
    for pt_code in _deduplicate((candidate.pt_code for candidate in retained)):
        group_candidates = [candidate for candidate in retained if candidate.pt_code == pt_code]
        groups.append(HumanReviewPTGroup(pt_code=pt_code, pt_name=group_candidates[0].pt_name, pt_rank=pt_rank_by_code.get(pt_code), pt_score=pt_score_by_code.get(pt_code), candidate_ids=[candidate.candidate_id for candidate in group_candidates], llt_candidates=group_candidates, pruning_explanations=[]))
    explanations: list[str] = []
    if len(eligible) > len(retained):
        explanations.append(f'Retained {len(retained)} of {len(eligible)} eligible candidates under the deterministic overall cap of {maximum_total}.')
    if len({candidate.pt_code for candidate in candidates}) > len(groups):
        explanations.append(f'PT uncertainty was bounded to {pt_group_count} leading PT groups, plus groups required by source-leading or reviewer-selected evidence.')
    explanations.append(f"Within each non-mandatory PT group, at most {llts_per_pt} LLTs were retained, including candidates within {score_delta:.3f} of that PT's best score.")
    return (retained, groups, explanations)

def _rank_pt_if_available(*, reviewer_input: ReviewerInput, reranker: PTConditionedLLTReranker, pt_code: str) -> list[LLTRankingEntry]:
    try:
        return reranker.rank(input_term=reviewer_input.normalized_term, pt_code=pt_code, retrieved_candidates=_retrieved_candidate_payloads(reviewer_input))
    except ValueError:
        return []

def _build_pt_uncertainty_candidates(*, reviewer_input: ReviewerInput, reranker: PTConditionedLLTReranker, reviewer_candidate_ids: list[str], pt_diagnostic: _PTConfidenceDiagnostic) -> list[HumanReviewCandidate]:
    """Build one bounded shortlist spanning leading PT groups."""
    ranking_by_id: dict[str, LLTRankingEntry] = {}
    rankings_by_pt: dict[str, list[LLTRankingEntry]] = {}
    source_leading_ids = _deduplicate((candidate.candidate_id for candidate in reviewer_input.aggregation_result.candidates if candidate.fuzzy_evidence is not None and candidate.fuzzy_evidence.rank == 1 or (candidate.semantic_evidence is not None and candidate.semantic_evidence.rank == 1) or candidate.umls_evidence is not None or (candidate.fuzzy_evidence is not None and candidate.fuzzy_evidence.direct_pt_match_score is not None)))
    ordered_ids = _deduplicate(reviewer_candidate_ids + source_leading_ids)[:MAX_HUMAN_REVIEW_CANDIDATES]
    leading_pt_codes = _deduplicate((str(ranking['pt_code']) for ranking in pt_diagnostic.component_pt_rankings))
    leading_pt_codes = leading_pt_codes[:PT_UNCERTAINTY_SHORTLIST_GROUPS]
    if pt_diagnostic.resolved_pt_code is not None:
        leading_pt_codes = _deduplicate(leading_pt_codes + [pt_diagnostic.resolved_pt_code])
    for pt_code in leading_pt_codes:
        rankings = _rank_pt_if_available(reviewer_input=reviewer_input, reranker=reranker, pt_code=pt_code)
        if not rankings:
            continue
        rankings_by_pt[pt_code] = rankings
        for ranking in rankings:
            ranking_by_id.setdefault(ranking.candidate_id, ranking)
    for pt_code in leading_pt_codes:
        rankings = rankings_by_pt.get(pt_code, [])
        if rankings:
            ordered_ids = _deduplicate(ordered_ids + [rankings[0].candidate_id])
        if len(ordered_ids) >= MAX_HUMAN_REVIEW_CANDIDATES:
            break
    if len(ordered_ids) < MAX_HUMAN_REVIEW_CANDIDATES:
        for pt_code in leading_pt_codes:
            for ranking in rankings_by_pt.get(pt_code, [])[1:PT_UNCERTAINTY_LLTS_PER_GROUP]:
                ordered_ids = _deduplicate(ordered_ids + [ranking.candidate_id])
                if len(ordered_ids) >= MAX_HUMAN_REVIEW_CANDIDATES:
                    break
            if len(ordered_ids) >= MAX_HUMAN_REVIEW_CANDIDATES:
                break
    ordered_ids = ordered_ids[:MAX_HUMAN_REVIEW_CANDIDATES]
    return _build_human_review_candidates(reviewer_input=reviewer_input, rankings=list(ranking_by_id.values()), human_review_candidate_ids=ordered_ids, reviewer_candidate_ids=reviewer_candidate_ids, reranker_shortlisted_ids=ordered_ids)

def _build_deferred_assessments(*, reviewer_input: ReviewerInput, reviewer_decision: ReviewerDecision, human_review_candidates: list[HumanReviewCandidate], pt_resolved: bool) -> list[CandidateAssessment]:
    human_review_ids = {candidate.candidate_id for candidate in human_review_candidates}
    existing_assessments = {assessment.candidate_id: assessment for assessment in reviewer_decision.candidate_assessments}
    ordered_candidate_ids = _deduplicate([candidate.candidate_id for candidate in reviewer_input.aggregation_result.candidates] + [candidate.candidate_id for candidate in human_review_candidates])
    candidate_name_by_id = {candidate.candidate_id: candidate.llt_name for candidate in human_review_candidates}
    assessments: list[CandidateAssessment] = []
    for candidate_id in ordered_candidate_ids:
        existing = existing_assessments.get(candidate_id)
        if candidate_id in human_review_ids:
            if existing is not None and existing.verdict == CandidateReviewVerdict.UNCERTAIN:
                assessment = existing
            else:
                candidate_name = candidate_name_by_id.get(candidate_id, candidate_id)
                assessment = CandidateAssessment(assessment_authority='deterministic_gate', candidate_id=candidate_id, verdict=CandidateReviewVerdict.UNCERTAIN, rationale=f"'{candidate_name}' remains a plausible MedDRA option and is retained for expert review." if not pt_resolved else f"'{candidate_name}' remains a plausible LLT within the resolved PT and is retained for expert MedDRA coding review.")
        elif existing is not None:
            assessment = existing.model_copy(update={'assessment_authority': 'deterministic_gate', 'verdict': CandidateReviewVerdict.NOT_SUPPORTED})
        else:
            assessment = CandidateAssessment(assessment_authority='deterministic_gate', candidate_id=candidate_id, verdict=CandidateReviewVerdict.NOT_SUPPORTED, rationale='This candidate is outside the PT-resolved LLT shortlist.')
        assessments.append(assessment)
    return assessments

def _deferred_decision(*, reviewer_input: ReviewerInput, reviewer_decision: ReviewerDecision, human_review_candidates: list[HumanReviewCandidate], decision_rationale: str, human_review_reason: str, pt_resolved: bool=True, resolved_pt_code: str | None=None, resolved_pt_name: str | None=None) -> ReviewerDecision:
    return ReviewerDecision(unresolved_without_candidates=not human_review_candidates and (not pt_resolved), resolved_pt_code=resolved_pt_code if pt_resolved else None, resolved_pt_name=resolved_pt_name if pt_resolved else None, term_id=reviewer_decision.term_id, review_round=reviewer_decision.review_round, decision_type=ReviewerDecisionType.MATCHED_PT_LLT_HUMAN_REVIEW if pt_resolved else ReviewerDecisionType.HUMAN_REVIEW, selected_candidate_ids=[], component_matches=[], candidate_assessments=_build_deferred_assessments(reviewer_input=reviewer_input, reviewer_decision=reviewer_decision, human_review_candidates=human_review_candidates, pt_resolved=pt_resolved), evidence_requests=[], human_review_candidate_ids=[candidate.candidate_id for candidate in human_review_candidates], decision_rationale=decision_rationale, multiple_match_explanation=None, human_review_reason=human_review_reason)

def _gate_result(*, action: LLTSelectionGateAction, reason: str, reviewer_candidate_ids: list[str], minimum_score: float, minimum_margin: float, pt_minimum_score: float, pt_minimum_margin: float, pt_confidence_required: bool, pt_diagnostic: _PTConfidenceDiagnostic | None=None, reviewer_top_agreement: bool | None=None, llt_top_score: float | None=None, llt_top_margin: float | None=None, ranked_candidates: list[HumanReviewCandidate] | None=None, human_review_candidates: list[HumanReviewCandidate] | None=None, human_review_pt_groups: list[HumanReviewPTGroup] | None=None, shortlist_pruning_explanations: list[str] | None=None) -> LLTSelectionGateResult:
    return LLTSelectionGateResult(version=LLT_SELECTION_GATE_VERSION, action=action, reason=reason, pt_confidence_required=pt_confidence_required, pt_confidence_passed=pt_diagnostic.passed if pt_diagnostic is not None else None, resolved_pt_code=pt_diagnostic.resolved_pt_code if pt_diagnostic is not None else None, resolved_pt_name=pt_diagnostic.resolved_pt_name if pt_diagnostic is not None else None, pt_top_code=pt_diagnostic.top_code if pt_diagnostic is not None else None, pt_top_score=pt_diagnostic.top_score if pt_diagnostic is not None else None, pt_top_margin=pt_diagnostic.top_margin if pt_diagnostic is not None else None, pt_minimum_score=pt_minimum_score, pt_minimum_margin=pt_minimum_margin, pt_confidence_failure_reasons=list(pt_diagnostic.failure_reasons) if pt_diagnostic is not None else [], reviewer_candidate_ids=reviewer_candidate_ids, reviewer_top_agreement=reviewer_top_agreement, minimum_score=minimum_score, minimum_margin=minimum_margin, top_score=llt_top_score, top_margin=llt_top_margin, llt_top_score=llt_top_score, llt_top_margin=llt_top_margin, gate_action=action, gate_reason=reason, ranked_candidates=ranked_candidates or [], human_review_candidates=human_review_candidates or [], human_review_pt_groups=human_review_pt_groups or [], shortlist_pruning_explanations=shortlist_pruning_explanations or [], shortlist_configuration={'pt_group_count': PT_UNCERTAINTY_SHORTLIST_GROUPS, 'llts_per_pt': PT_UNCERTAINTY_LLTS_PER_GROUP, 'score_delta': SHORTLIST_SCORE_DELTA, 'maximum_total': MAX_HUMAN_REVIEW_CANDIDATES})

def _skipped_result(*, reason: str, minimum_score: float, minimum_margin: float, pt_minimum_score: float, pt_minimum_margin: float) -> LLTSelectionGateResult:
    return _gate_result(action=LLTSelectionGateAction.SKIPPED, reason=reason, reviewer_candidate_ids=[], minimum_score=minimum_score, minimum_margin=minimum_margin, pt_minimum_score=pt_minimum_score, pt_minimum_margin=pt_minimum_margin, pt_confidence_required=False)

def _legacy_apply_llt_selection_gate(*, reviewer_input: ReviewerInput, reviewer_decision: ReviewerDecision, reranker: PTConditionedLLTReranker, exact_match_found: bool=False, minimum_score: float=LLT_AUTO_ACCEPT_MINIMUM_SCORE, minimum_margin: float=LLT_AUTO_ACCEPT_MINIMUM_MARGIN, pt_minimum_score: float=LLT_AUTO_ACCEPT_PT_MINIMUM_SCORE, pt_minimum_margin: float=LLT_AUTO_ACCEPT_PT_MINIMUM_MARGIN) -> tuple[ReviewerDecision, LLTSelectionGateResult]:
    """
    Apply the calibrated PT-then-LLT confidence policy.

    A reviewer LLT remains automatic only when its PT first passes the
    deterministic component-level PT reranker check. PT uncertainty produces
    a full human-review decision spanning leading PT groups. Once the PT is
    trusted, the existing within-PT LLT agreement policy remains unchanged.
    """
    if exact_match_found:
        return (reviewer_decision, _skipped_result(reason='Exact-path terms are excluded from LLT reranking.', minimum_score=minimum_score, minimum_margin=minimum_margin, pt_minimum_score=pt_minimum_score, pt_minimum_margin=pt_minimum_margin))
    supported_decision_types = {ReviewerDecisionType.MATCHED_SINGLE, ReviewerDecisionType.MATCHED_PT_LLT_HUMAN_REVIEW, ReviewerDecisionType.HUMAN_REVIEW}
    if reviewer_decision.decision_type not in supported_decision_types:
        return (reviewer_decision, _skipped_result(reason='The reviewer decision does not resolve exactly one PT for LLT selection.', minimum_score=minimum_score, minimum_margin=minimum_margin, pt_minimum_score=pt_minimum_score, pt_minimum_margin=pt_minimum_margin))
    aggregated_by_id = {candidate.candidate_id: candidate for candidate in reviewer_input.aggregation_result.candidates}
    if reviewer_decision.decision_type == ReviewerDecisionType.MATCHED_SINGLE:
        reviewer_candidate_ids = list(reviewer_decision.selected_candidate_ids)
    else:
        reviewer_candidate_ids = list(reviewer_decision.human_review_candidate_ids)
    reviewer_candidates = [aggregated_by_id[candidate_id] for candidate_id in reviewer_candidate_ids if candidate_id in aggregated_by_id]
    if len(reviewer_candidates) != len(reviewer_candidate_ids):
        missing = sorted(set(reviewer_candidate_ids) - set(aggregated_by_id))
        raise ValueError(f'The reviewer resolved candidates that are missing from aggregation: {missing}')
    pt_diagnostic = _evaluate_pt_confidence(reviewer_input=reviewer_input, reviewer_decision=reviewer_decision, reviewer_candidates=reviewer_candidates, minimum_score=pt_minimum_score, minimum_margin=pt_minimum_margin)
    rankings = _rank_pt_if_available(reviewer_input=reviewer_input, reranker=reranker, pt_code=pt_diagnostic.resolved_pt_code) if pt_diagnostic.resolved_pt_code is not None else []
    top = rankings[0] if rankings else None
    top_margin = _candidate_margin(rankings) if rankings else None
    if not pt_diagnostic.passed:
        human_review_candidates = _build_pt_uncertainty_candidates(reviewer_input=reviewer_input, reranker=reranker, reviewer_candidate_ids=reviewer_candidate_ids, pt_diagnostic=pt_diagnostic)
        human_review_candidates, human_review_pt_groups, shortlist_pruning_explanations = _structured_shortlist(reviewer_input=reviewer_input, candidates=human_review_candidates, reviewer_candidate_ids=reviewer_candidate_ids, pt_rankings=pt_diagnostic.component_pt_rankings)
        failure_summary = ', '.join(pt_diagnostic.failure_reasons)
        gate_reason = f'The resolved PT did not satisfy the deterministic PT-confidence prerequisite: {failure_summary}.'
        repaired_decision = _deferred_decision(reviewer_input=reviewer_input, reviewer_decision=reviewer_decision, human_review_candidates=human_review_candidates, decision_rationale=gate_reason + ' The complete PT/LLT decision was deferred without retaining an automatic LLT selection.', human_review_reason='Expert MedDRA review is required because the PT itself is not confidently resolved.', pt_resolved=False)
        return (repaired_decision, _gate_result(action=LLTSelectionGateAction.DEFERRED_PT_UNCERTAINTY, reason=gate_reason, reviewer_candidate_ids=reviewer_candidate_ids, minimum_score=minimum_score, minimum_margin=minimum_margin, pt_minimum_score=pt_minimum_score, pt_minimum_margin=pt_minimum_margin, pt_confidence_required=True, pt_diagnostic=pt_diagnostic, reviewer_top_agreement=None if top is None or len(reviewer_candidate_ids) != 1 else reviewer_candidate_ids[0] == top.candidate_id, llt_top_score=top.score if top is not None else None, llt_top_margin=top_margin, ranked_candidates=human_review_candidates, human_review_candidates=human_review_candidates, human_review_pt_groups=human_review_pt_groups, shortlist_pruning_explanations=shortlist_pruning_explanations))
    if not rankings:
        raise ValueError('A PT passed confidence but no current LLTs were available for that PT.')
    pt_code = pt_diagnostic.resolved_pt_code
    pt_name = pt_diagnostic.resolved_pt_name
    assert pt_code is not None
    assert pt_name is not None
    assert top is not None
    assert top_margin is not None
    top_five = rankings[:LLT_DISAGREEMENT_SHORTLIST_SIZE]
    top_five_ids = [ranking.candidate_id for ranking in top_five]
    reviewer_id_set = set(reviewer_candidate_ids)
    top_five_id_set = set(top_five_ids)
    ranked_candidates = [_as_human_review_candidate(ranking, reviewer_candidate_ids=reviewer_id_set, reranker_shortlisted_ids=top_five_id_set) for ranking in top_five]
    if reviewer_decision.decision_type in {ReviewerDecisionType.MATCHED_PT_LLT_HUMAN_REVIEW, ReviewerDecisionType.HUMAN_REVIEW}:
        top_three_ids = [ranking.candidate_id for ranking in rankings[:LLT_DEFERRAL_ENRICHMENT_SIZE]]
        top_three_id_set = set(top_three_ids)
        deferral_ranked_candidates = [_as_human_review_candidate(ranking, reviewer_candidate_ids=reviewer_id_set, reranker_shortlisted_ids=top_three_id_set) for ranking in top_five]
        human_review_ids = _deduplicate(reviewer_candidate_ids + top_three_ids)
        human_review_candidates = _build_human_review_candidates(reviewer_input=reviewer_input, rankings=rankings, human_review_candidate_ids=human_review_ids, reviewer_candidate_ids=reviewer_candidate_ids, reranker_shortlisted_ids=top_three_ids)
        human_review_candidates, human_review_pt_groups, shortlist_pruning_explanations = _structured_shortlist(reviewer_input=reviewer_input, candidates=human_review_candidates, reviewer_candidate_ids=reviewer_candidate_ids, pt_rankings=pt_diagnostic.component_pt_rankings, pt_group_count=1, llts_per_pt=LLT_DEFERRAL_ENRICHMENT_SIZE, maximum_total=MAX_HUMAN_REVIEW_CANDIDATES)
        repaired_decision = _deferred_decision(reviewer_input=reviewer_input, reviewer_decision=reviewer_decision, human_review_candidates=human_review_candidates, decision_rationale=reviewer_decision.decision_rationale + ' The existing PT-resolved LLT deferral was enriched with the deterministic full-PT shortlist.', human_review_reason=reviewer_decision.human_review_reason or 'Expert MedDRA coding review is required to resolve the LLT.')
        return (repaired_decision, _gate_result(action=LLTSelectionGateAction.ENRICHED_EXISTING_DEFERRAL, reason="Preserved the reviewer LLT options and added the reranker's top three candidates.", reviewer_candidate_ids=reviewer_candidate_ids, reviewer_top_agreement=None, minimum_score=minimum_score, minimum_margin=minimum_margin, pt_minimum_score=pt_minimum_score, pt_minimum_margin=pt_minimum_margin, pt_confidence_required=True, pt_diagnostic=pt_diagnostic, llt_top_score=top.score, llt_top_margin=top_margin, ranked_candidates=deferral_ranked_candidates, human_review_candidates=human_review_candidates, human_review_pt_groups=human_review_pt_groups, shortlist_pruning_explanations=shortlist_pruning_explanations))
    reviewer_candidate_id = reviewer_candidate_ids[0]
    reviewer_top_agreement = reviewer_candidate_id == top.candidate_id
    single_llt_pt = len(rankings) == 1
    confident_agreement = reviewer_top_agreement and (single_llt_pt or (top.score >= minimum_score and top_margin >= minimum_margin))
    if confident_agreement:
        reason = 'The resolved PT has only one current LLT, so no within-PT choice remains.' if single_llt_pt else 'Reviewer and reranker agree and the calibrated score and margin thresholds are satisfied.'
        return (reviewer_decision, _gate_result(action=LLTSelectionGateAction.AUTO_ACCEPTED, reason=reason, reviewer_candidate_ids=reviewer_candidate_ids, reviewer_top_agreement=True, minimum_score=minimum_score, minimum_margin=minimum_margin, pt_minimum_score=pt_minimum_score, pt_minimum_margin=pt_minimum_margin, pt_confidence_required=True, pt_diagnostic=pt_diagnostic, llt_top_score=top.score, llt_top_margin=top_margin, ranked_candidates=ranked_candidates, human_review_candidates=[]))
    human_review_ids = _deduplicate(reviewer_candidate_ids + top_five_ids)
    human_review_candidates = _build_human_review_candidates(reviewer_input=reviewer_input, rankings=rankings, human_review_candidate_ids=human_review_ids, reviewer_candidate_ids=reviewer_candidate_ids, reranker_shortlisted_ids=top_five_ids)
    human_review_candidates, human_review_pt_groups, shortlist_pruning_explanations = _structured_shortlist(reviewer_input=reviewer_input, candidates=human_review_candidates, reviewer_candidate_ids=reviewer_candidate_ids, pt_rankings=pt_diagnostic.component_pt_rankings, pt_group_count=1, llts_per_pt=LLT_DISAGREEMENT_SHORTLIST_SIZE, maximum_total=MAX_HUMAN_REVIEW_CANDIDATES)
    if reviewer_top_agreement:
        action = LLTSelectionGateAction.DEFERRED_LOW_CONFIDENCE
        decision_rationale = 'The reviewer and deterministic full-PT reranker select the same LLT, but the calibrated score or separation margin is insufficient for automatic LLT selection.'
        human_review_reason = 'Expert MedDRA coding review is required because the agreeing LLT selection does not meet the calibrated confidence gate.'
    else:
        action = LLTSelectionGateAction.DEFERRED_DISAGREEMENT
        decision_rationale = 'The reviewer-selected LLT and the deterministic full-PT reranker disagree. Neither selection is allowed to overwrite the other automatically.'
        human_review_reason = 'Expert MedDRA coding review is required to resolve the reviewer/reranker LLT disagreement within the supported PT.'
    repaired_decision = _deferred_decision(reviewer_input=reviewer_input, reviewer_decision=reviewer_decision, human_review_candidates=human_review_candidates, decision_rationale=decision_rationale, human_review_reason=human_review_reason)
    return (repaired_decision, _gate_result(action=action, reason=human_review_reason, reviewer_candidate_ids=reviewer_candidate_ids, reviewer_top_agreement=reviewer_top_agreement, minimum_score=minimum_score, minimum_margin=minimum_margin, pt_minimum_score=pt_minimum_score, pt_minimum_margin=pt_minimum_margin, pt_confidence_required=True, pt_diagnostic=pt_diagnostic, llt_top_score=top.score, llt_top_margin=top_margin, ranked_candidates=ranked_candidates, human_review_candidates=human_review_candidates, human_review_pt_groups=human_review_pt_groups, shortlist_pruning_explanations=shortlist_pruning_explanations))

def apply_llt_selection_gate(*, reviewer_input: ReviewerInput, reviewer_decision: ReviewerDecision, reranker: PTConditionedLLTReranker, exact_match_found: bool=False, minimum_score: float | None=None, minimum_margin: float | None=None, pt_minimum_score: float | None=None, pt_minimum_margin: float | None=None, policy_config: PolicyConfig=DEFAULT_POLICY, pt_recovery_policy=None, coverage_policy=None, ranking_policy=None, exact_ambiguity_candidates: tuple=(), reviewer_failure_diagnostics: dict | None=None) -> tuple[ReviewerDecision, LLTSelectionGateResult]:
    """Apply v1.3 using the complete evidence pool and explicit PT groups."""
    minimum_score = policy_config.llt_minimum_score if minimum_score is None else minimum_score
    minimum_margin = policy_config.llt_minimum_margin if minimum_margin is None else minimum_margin
    pt_minimum_score = policy_config.pt_minimum_score if pt_minimum_score is None else pt_minimum_score
    pt_minimum_margin = policy_config.pt_minimum_margin if pt_minimum_margin is None else pt_minimum_margin
    config = replace(policy_config, pt_minimum_score=pt_minimum_score, pt_minimum_margin=pt_minimum_margin, llt_minimum_score=minimum_score, llt_minimum_margin=minimum_margin)
    from biomedical_wording import generic_context_base, is_generic_event
    from schemas import ComponentCandidateMatch
    generic_repair = False
    if reviewer_decision.decision_type == ReviewerDecisionType.MATCHED_MULTIPLE and generic_context_base(reviewer_input.normalized_term):
        by_id = {c.candidate_id: c for c in reviewer_input.aggregation_result.candidates}
        clinical_ids = [i for i in reviewer_decision.selected_candidate_ids if i in by_id and (not is_generic_event(by_id[i].llt_name)) and (not is_generic_event(by_id[i].pt_name))]
        if len(clinical_ids) == 1:
            identifier = clinical_ids[0]
            components = [c for c in reviewer_input.input_interpretation.asserted_components if not c.is_negated]
            if len(components) == 1:
                payload = reviewer_decision.model_dump(mode='python')
                payload.update(decision_type='matched_single', selected_candidate_ids=clinical_ids, multiple_match_explanation=None, component_matches=[ComponentCandidateMatch(component_id=components[0].component_id, pt_code=by_id[identifier].pt_code, candidate_id=identifier, rationale='Generic contextual suffix removed; retained the named clinical concept.')])
                payload['candidate_assessments'] = [a.model_copy(update={'verdict': CandidateReviewVerdict.NOT_SUPPORTED, 'rationale': 'Generic contextual suffix, not an independently asserted event.'}) if a.candidate_id in reviewer_decision.selected_candidate_ids and a.candidate_id != identifier else a for a in reviewer_decision.candidate_assessments]
                reviewer_decision = ReviewerDecision.model_validate(payload)
                generic_repair = True
    baseline, gate = _legacy_apply_llt_selection_gate(reviewer_input=reviewer_input, reviewer_decision=reviewer_decision, reranker=reranker, exact_match_found=exact_match_found, minimum_score=minimum_score, minimum_margin=minimum_margin, pt_minimum_score=max(0.9, pt_minimum_score), pt_minimum_margin=pt_minimum_margin)
    from clinical_features import automatic_mapping_uncertainties
    compound_numerical_uncertainty = reviewer_decision.decision_type == ReviewerDecisionType.MATCHED_MULTIPLE and any((automatic_mapping_uncertainties(c.component_text) for c in reviewer_input.input_interpretation.asserted_components if not c.is_negated))
    compound = reviewer_decision.decision_type == ReviewerDecisionType.MATCHED_MULTIPLE or (sum((not c.is_negated for c in reviewer_input.input_interpretation.asserted_components)) > 1 and reviewer_decision.decision_type in {ReviewerDecisionType.MATCHED_SINGLE, ReviewerDecisionType.MATCHED_PT_LLT_HUMAN_REVIEW})
    if exact_match_found or (not exact_ambiguity_candidates and (not compound_numerical_uncertainty) and (reviewer_decision.decision_type not in {ReviewerDecisionType.MATCHED_SINGLE, ReviewerDecisionType.MATCHED_PT_LLT_HUMAN_REVIEW, ReviewerDecisionType.HUMAN_REVIEW, ReviewerDecisionType.MATCHED_MULTIPLE})):
        return (baseline, gate.model_copy(update={'policy_configuration': config.to_dict(), 'decision_authority': 'exact_matcher' if exact_match_found else 'llm_with_component_validation'}))
    reviewer_ids = tuple(reviewer_decision.selected_candidate_ids or reviewer_decision.human_review_candidate_ids)
    raw = {c.candidate_id: c for c in reviewer_input.aggregation_result.candidates}
    baseline_llts = tuple(baseline.selected_candidate_ids)
    available = {c.candidate_id: c for c in gate.ranked_candidates + gate.human_review_candidates}
    available.update(raw)
    baseline_pts = tuple(sorted({available[i].pt_code for i in baseline_llts if i in available}))
    if baseline.decision_type == ReviewerDecisionType.MATCHED_PT_LLT_HUMAN_REVIEW and gate.resolved_pt_code:
        baseline_pts = (gate.resolved_pt_code,)
    supplemental = {c.candidate_id: c for c in gate.ranked_candidates + gate.human_review_candidates}
    if exact_ambiguity_candidates:
        for candidate in exact_ambiguity_candidates:
            supplemental[candidate.candidate_id] = HumanReviewCandidate(candidate_id=candidate.candidate_id, llt_code=candidate.llt_code, llt_name=candidate.llt_name, pt_code=candidate.pt_code, pt_name=candidate.pt_name, is_current=candidate.is_current, was_retrieved=False, retrieval_source_count=0, reviewer_selected=False, reranker_shortlisted=False)
    component_rankings = build_component_pt_rankings(reviewer_input)
    compound_audit = {}
    if compound:
        from reviewer_validation import validate_multiple_acceptance
        from compound_validation import assess_decomposition
        decomposition = assess_decomposition(reviewer_input.normalized_term, reviewer_input.input_interpretation.asserted_components)
        component_text = {c['component_id']: c['effective_text'] for c in decomposition['components']}
        component_llt_rankings = {}
        for match in reviewer_decision.component_matches:
            if match.component_id in component_text and reranker.entries_for_pt(match.pt_code):
                component_llt_rankings[match.component_id] = reranker.rank(input_term=component_text[match.component_id], pt_code=match.pt_code, retrieved_candidates=_retrieved_candidate_payloads(reviewer_input))
        compound_audit = validate_multiple_acceptance(reviewer_input, reviewer_decision, config, llt_rankings=component_llt_rankings, component_rankings=component_rankings)
        if reviewer_failure_diagnostics:
            compound_audit['accepted'] = False
            compound_audit['reasons'].append('clinical_validation_fallback_cannot_be_promoted')
        if exact_ambiguity_candidates:
            compound_audit['accepted'] = False
            compound_audit['reasons'].append('normalized_exact_collision_requires_review')
        if compound_audit['accepted']:
            return (reviewer_decision, LLTSelectionGateResult.model_validate({**gate.model_dump(mode='python'), 'version': LLT_SELECTION_GATE_VERSION, 'action': LLTSelectionGateAction.AUTO_ACCEPTED, 'gate_action': LLTSelectionGateAction.AUTO_ACCEPTED, 'reason': 'Every independently asserted component passed PT and LLT acceptance.', 'gate_reason': 'Every independently asserted component passed PT and LLT acceptance.', 'policy_configuration': config.to_dict(), 'compound_acceptance': compound_audit, 'decision_authority': 'deterministic_component_acceptance', 'pt_confidence_required': True, 'pt_confidence_passed': True, 'pt_minimum_score': config.pt_minimum_score, 'pt_minimum_margin': config.pt_minimum_margin}))
    expansion_pts = {r['pt_code'] for c in component_rankings for r in c['pt_rankings'][:config.maximum_pt_rank]}
    for pt in sorted(expansion_pts | set(baseline_pts)):
        for ranking in _rank_pt_if_available(reviewer_input=reviewer_input, reranker=reranker, pt_code=pt)[:config.llts_per_leading_pt]:
            supplemental[ranking.candidate_id] = _as_human_review_candidate(ranking, reviewer_candidate_ids=set(reviewer_ids), reranker_shortlisted_ids={ranking.candidate_id})
    selected_pts = {raw[i].pt_code for i in reviewer_decision.selected_candidate_ids if i in raw}
    evidence = prepare_policy_evidence(reviewer_input, reviewer_ids, baseline_pts, baseline_llts, next(iter(selected_pts)) if len(selected_pts) == 1 else None, tuple(supplemental.values()))
    if exact_ambiguity_candidates:
        evidence = replace(evidence, normalized_exact_collision=True, exact_collision_ids=tuple((c.candidate_id for c in exact_ambiguity_candidates)))
    if reviewer_failure_diagnostics and (reviewer_failure_diagnostics.get('retry_exhausted') or reviewer_failure_diagnostics.get('force_human_review')):
        evidence = replace(evidence, force_human_review=True)
    if compound:
        evidence = replace(evidence, force_human_review=True)
    result = apply_policy(evidence, config)
    from pt_coverage_policy import DEFAULT_PT_RECOVERY, recover_pt, recovery_context
    recovery_config = DEFAULT_PT_RECOVERY if pt_recovery_policy is None else pt_recovery_policy
    context = recovery_context(reviewer_input, reviewer_ids, selection_resolved=reviewer_decision.decision_type in {ReviewerDecisionType.MATCHED_SINGLE, ReviewerDecisionType.MATCHED_PT_LLT_HUMAN_REVIEW}, authority='validated_reviewer_stage_including_possible_deterministic_repairs')
    result, recovery_audit = recover_pt(evidence, result, context, policy=recovery_config, shortlist_config=config)
    from coverage_policy import assess
    from joint_ranking import RELEASE_COVERAGE, DEFAULT_RANKING, rank_pairs, assess_release, plausible_groups, reference_surface_index, surface
    coverage_config = RELEASE_COVERAGE if coverage_policy is None else coverage_policy
    ranking_config = DEFAULT_RANKING if ranking_policy is None else ranking_policy
    coverage_audit = {}
    if coverage_config.enabled and (not compound) and (not evidence.force_human_review) and (not exact_ambiguity_candidates):
        rows = component_rankings[0]['pt_rankings'] if len(component_rankings) == 1 else []
        full_rankings = {}
        if rows:
            code = rows[0]['pt_code']
            full_rankings[code] = _rank_pt_if_available(reviewer_input=reviewer_input, reranker=reranker, pt_code=code)
            for row in full_rankings[code]:
                supplemental[row.candidate_id] = _as_human_review_candidate(row, reviewer_candidate_ids=set(reviewer_ids), reranker_shortlisted_ids={row.candidate_id})
        full_evidence = prepare_policy_evidence(reviewer_input, reviewer_ids, single_selected_pt=evidence.single_selected_pt, supplemental=tuple(supplemental.values()))
        projected, coverage_audit = assess(full_evidence, context, full_rankings, config=coverage_config, shortlist_config=config, reference_complete=True)
        coverage_audit = {k: v for k, v in coverage_audit.items() if not k.startswith('_')}
        coverage_audit['applied'] = False
        if not result.llt_ids and projected.pt_codes and (not result.pt_codes or result.pt_codes == projected.pt_codes):
            result = projected
            evidence = full_evidence
            coverage_audit['applied'] = True
        if ranking_config.enabled:
            for pt in plausible_groups(full_evidence, ranking_config):
                if pt not in full_rankings:
                    full_rankings[pt] = _rank_pt_if_available(reviewer_input=reviewer_input, reranker=reranker, pt_code=pt)
                for row in full_rankings[pt]:
                    supplemental[row.candidate_id] = _as_human_review_candidate(row, reviewer_candidate_ids=set(reviewer_ids), reranker_shortlisted_ids={row.candidate_id})
            expanded = prepare_policy_evidence(reviewer_input, reviewer_ids, single_selected_pt=evidence.single_selected_pt, supplemental=tuple(supplemental.values()))
            semantics = {c.candidate_id: c.semantic_evidence.cosine_similarity for c in reviewer_input.aggregation_result.candidates if c.semantic_evidence}
            if not hasattr(reranker, '_v1_5_surface_index'):
                reranker._v1_5_surface_index = reference_surface_index(reranker)
            ranked = rank_pairs(expanded, full_rankings, semantics, config=ranking_config, literal_current_ids=reranker._v1_5_surface_index.get(surface(expanded.input_text), []))
            upgraded, ranking_audit, rank_acceptance = assess_release(expanded, context, ranked, result, coverage=coverage_config)
            if ranking_audit['applied']:
                result = upgraded
                evidence = expanded
                coverage_audit = {k: v for k, v in rank_acceptance.items() if not k.startswith('_')}
                coverage_audit['applied'] = True
                coverage_audit['acceptance_path'] = ranking_audit['reason']
            import json
            coverage_audit['joint_ranking'] = json.loads(json.dumps(ranking_audit))
    if compound:
        result.reasons.extend(compound_audit['reasons'])
    if generic_repair:
        result.reasons.append('generic_context_suffix_removed_from_matched_multiple')
    if result.llt_ids:
        decision = baseline
        if tuple(baseline.selected_candidate_ids) != result.llt_ids:
            identifier = result.llt_ids[0]
            current = {o.candidate.candidate_id: o.candidate for o in evidence.options}
            selected = current[identifier]
            decision = ReviewerDecision(term_id=reviewer_decision.term_id, review_round=reviewer_decision.review_round, decision_type=ReviewerDecisionType.MATCHED_SINGLE, selected_candidate_ids=[identifier], component_matches=[ComponentCandidateMatch(component_id=reviewer_input.input_interpretation.asserted_components[0].component_id, candidate_id=identifier, pt_code=selected.pt_code, rationale='Current-reference PT and LLT passed the deterministic coverage policy.')], candidate_assessments=[CandidateAssessment(candidate_id=i, assessment_authority='deterministic_gate', verdict=CandidateReviewVerdict.SUPPORTED if i == identifier else CandidateReviewVerdict.NOT_SUPPORTED, rationale='Deterministic current-reference selection.' if i == identifier else 'Not selected by the deterministic evidence policy.') for i in dict.fromkeys(list(raw) + [identifier])], decision_rationale='PT and LLT resolved by versioned deterministic coverage acceptance, not an additional LLM judgment.')
        action = LLTSelectionGateAction.AUTO_ACCEPTED
    else:
        decision = _deferred_decision(reviewer_input=reviewer_input, reviewer_decision=reviewer_decision, human_review_candidates=result.candidates, pt_resolved=len(result.pt_codes) == 1, resolved_pt_code=result.pt_codes[0] if len(result.pt_codes) == 1 else None, resolved_pt_name=next((o.candidate.pt_name for o in evidence.options if len(result.pt_codes) == 1 and o.candidate.pt_code == result.pt_codes[0]), None), decision_rationale='; '.join(result.reasons) or 'Adaptive evidence-preserving PT/LLT human review.', human_review_reason='Compare the retained current LLTs and their wording/specificity evidence; unresolved choices require human review.')
        action = LLTSelectionGateAction.DEFERRED_LOW_CONFIDENCE if result.pt_codes else LLTSelectionGateAction.DEFERRED_PT_UNCERTAINTY
        if gate.action == LLTSelectionGateAction.DEFERRED_DISAGREEMENT and result.pt_codes and (not result.ambiguity):
            action = gate.action
        if gate.action == LLTSelectionGateAction.ENRICHED_EXISTING_DEFERRAL and result.pt_codes:
            action = gate.action
    reason = '; '.join(result.reasons) or gate.reason
    resolved = result.pt_codes[0] if len(result.pt_codes) == 1 else None
    name = next((o.candidate.pt_name for o in evidence.options if o.candidate.pt_code == resolved), None)
    payload = gate.model_dump(mode='python')
    payload.update(version=LLT_SELECTION_GATE_VERSION, action=action, gate_action=action, reason=reason, gate_reason=reason, pt_minimum_score=config.pt_minimum_score, pt_minimum_margin=config.pt_minimum_margin, pt_confidence_required=True, pt_confidence_passed=bool(resolved), pt_confidence_failure_reasons=[] if resolved else gate.pt_confidence_failure_reasons or ['v1_3_pt_confidence_not_satisfied'], resolved_pt_code=resolved or (None if evidence.force_human_review else gate.resolved_pt_code), resolved_pt_name=name or (None if evidence.force_human_review else gate.resolved_pt_name), pt_top_score=result.top_score, pt_top_margin=result.top_margin, human_review_candidates=result.candidates, human_review_pt_groups=result.groups, shortlist_pruning_explanations=result.pruned, policy_configuration=config.to_dict(), shortlist_configuration={'base_pt_groups': config.base_pt_groups, 'maximum_pt_rank': config.maximum_pt_rank, 'llts_per_leading_pt': config.llts_per_leading_pt, 'score_delta': config.score_delta, 'maximum_candidates': config.maximum_candidates}, decision_authority=(reviewer_failure_diagnostics.get('authority', 'deterministic_clinical_validation_fallback') if reviewer_failure_diagnostics.get('force_human_review') else 'deterministic_clinical_validation_fallback') if reviewer_failure_diagnostics else 'deterministic_pt_coverage_recovery' if recovery_audit['accepted'] else 'deterministic_gate', reviewer_failure_diagnostics=reviewer_failure_diagnostics or {}, deterministic_repair_reasons=result.reasons, ambiguity_diagnostics=result.ambiguity, compound_acceptance=compound_audit, pt_coverage_recovery=recovery_audit, coverage_acceptance=coverage_audit)
    if coverage_audit:
        payload['ranked_candidates'] = list({c.candidate_id: c for c in gate.ranked_candidates + list(supplemental.values())}.values())
        if coverage_audit.get('applied') and result.pt_codes and (not evidence.force_human_review):
            payload['decision_authority'] = 'deterministic_coverage_policy'
            payload.update(minimum_score=coverage_config.llt_score, minimum_margin=coverage_config.llt_margin, pt_minimum_score=coverage_audit['pt_thresholds']['score'], pt_minimum_margin=coverage_audit['pt_thresholds']['margin'], reviewer_top_agreement=coverage_audit.get('reviewer_llt_agreement'), top_score=coverage_audit.get('llt_score'), llt_top_score=coverage_audit.get('llt_score'), top_margin=max(0.0, float(coverage_audit['llt_margin'])) if 'llt_margin' in coverage_audit else None, llt_top_margin=max(0.0, float(coverage_audit['llt_margin'])) if 'llt_margin' in coverage_audit else None)
    return (decision, LLTSelectionGateResult.model_validate(payload))

def create_llt_selection_gate_node(*, reranker: PTConditionedLLTReranker, minimum_score: float=LLT_AUTO_ACCEPT_MINIMUM_SCORE, minimum_margin: float=LLT_AUTO_ACCEPT_MINIMUM_MARGIN, pt_minimum_score: float=LLT_AUTO_ACCEPT_PT_MINIMUM_SCORE, pt_minimum_margin: float=LLT_AUTO_ACCEPT_PT_MINIMUM_MARGIN, pt_recovery_policy=None, coverage_policy=None, ranking_policy=None):
    """Create a LangGraph-compatible LLT selection gate node."""

    def llt_selection_gate_node(state: StandardizationState) -> dict[str, Any]:
        reviewer_input = state.get('reviewer_input')
        reviewer_decision = state.get('reviewer_decision')
        if reviewer_input is None:
            raise ValueError('LLT selection gate requires reviewer_input.')
        if reviewer_decision is None:
            raise ValueError('LLT selection gate requires reviewer_decision.')
        exact_result = state.get('exact_result')
        repaired_decision, gate_result = apply_llt_selection_gate(reviewer_input=reviewer_input, reviewer_decision=reviewer_decision, reranker=reranker, exact_match_found=exact_result is not None and exact_result.found, exact_ambiguity_candidates=tuple(exact_result.ambiguous_candidates) if exact_result is not None else (), reviewer_failure_diagnostics=state.get('reviewer_failure_diagnostics'), pt_recovery_policy=pt_recovery_policy, coverage_policy=coverage_policy, ranking_policy=ranking_policy, minimum_score=minimum_score, minimum_margin=minimum_margin, pt_minimum_score=pt_minimum_score, pt_minimum_margin=pt_minimum_margin)
        return {'reviewer_decision': repaired_decision, 'llt_selection_gate_result': gate_result}
    return llt_selection_gate_node
