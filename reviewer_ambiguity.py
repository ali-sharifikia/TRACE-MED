from __future__ import annotations
from runtime_resources import resource as asp_resource
import re
from typing import Any
from schemas import CandidateAssessment, CandidateReviewVerdict, InputStructureType, ReviewerDecision, ReviewerDecisionType, ReviewerInput
SPELLING_EQUIVALENTS = {'haematocrit': 'hematocrit', 'haemoglobin': 'hemoglobin', 'haemorrhage': 'hemorrhage', 'haemorrhagic': 'hemorrhagic', 'oedema': 'edema', 'anaemia': 'anemia', 'leukaemia': 'leukemia', 'tumour': 'tumor', 'tumours': 'tumors', 'foetal': 'fetal', 'oesophagus': 'esophagus', 'oesophageal': 'esophageal'}
TOKEN_EQUIVALENTS = {'increase': 'increased', 'increases': 'increased', 'elevation': 'increased', 'elevations': 'increased', 'elevated': 'increased', 'raised': 'increased', 'raises': 'increased', 'decrease': 'decreased', 'decreases': 'decreased', 'reduction': 'decreased', 'reductions': 'decreased', 'reduced': 'decreased', 'lowered': 'decreased', 'tests': 'test', 'disorders': 'disorder', 'infections': 'infection', 'reactions': 'reaction', 'manifestations': 'manifestation', 'eyes': 'eye', 'ligaments': 'ligament', 'rates': 'rate', 'counts': 'count', 'tumors': 'tumor', 'levels': 'level', 'values': 'value', 'abnormalities': 'abnormal', 'males': 'male', 'ocular': 'eye', 'mycotic': 'fungal', 'pulse': 'heart'}
IGNORED_EQUIVALENCE_TOKENS = {'nos', 'unspecified', 'of', 'in', 'on', 'the', 'level', 'value'}
MEANINGFUL_MODIFIER_TOKENS = {'high', 'low', 'increased', 'decreased', 'normal', 'abnormal', 'positive', 'negative', 'prolonged', 'shortened', 'aggravated', 'worsened', 'improved', 'acute', 'chronic', 'recurrent', 'primary', 'secondary', 'early', 'late', 'mild', 'moderate', 'severe', 'disseminated', 'systemic', 'left', 'right', 'bilateral'}

def _tokens(value: str) -> list[str]:
    return re.findall('[a-z0-9]+', value.casefold())

def _normalized_label_tokens(value: str) -> tuple[str, ...]:
    raw_tokens = _tokens(value)
    normalized_tokens: list[str] = []
    for index, raw_token in enumerate(raw_tokens):
        token = SPELLING_EQUIVALENTS.get(raw_token, raw_token)
        if token == 'c' and index > 0 and (raw_tokens[index - 1] in {'hdl', 'ldl'}):
            token = 'cholesterol'
        token = TOKEN_EQUIVALENTS.get(token, token)
        if token in IGNORED_EQUIVALENCE_TOKENS:
            continue
        normalized_tokens.append(token)
    return tuple(sorted(set(normalized_tokens)))

def _normalized_llt_label(value: str) -> str:
    return ' '.join(_normalized_label_tokens(value))

def _labels_are_equivalent(first_value: str, second_value: str) -> bool:
    return _normalized_label_tokens(first_value) == _normalized_label_tokens(second_value)

def _has_meaningful_modifier_difference(first_value: str, second_value: str) -> bool:
    first_modifiers = set(_normalized_label_tokens(first_value)) & MEANINGFUL_MODIFIER_TOKENS
    second_modifiers = set(_normalized_label_tokens(second_value)) & MEANINGFUL_MODIFIER_TOKENS
    return first_modifiers != second_modifiers and bool(first_modifiers or second_modifiers)

def _input_token_overlap(input_term: str, candidate_name: str) -> float:
    input_tokens = set(_tokens(input_term))
    candidate_tokens = set(_tokens(candidate_name))
    if not candidate_tokens:
        return 0.0
    return len(input_tokens & candidate_tokens) / len(candidate_tokens)

def _candidate_has_conflict(candidate: Any) -> bool:
    fuzzy_evidence = candidate.fuzzy_evidence
    if fuzzy_evidence is None:
        return False
    conflict_attributes = ('directionality_conflict', 'negation_conflict', 'laterality_conflict', 'temporal_conflict', 'severity_conflict', 'anatomical_conflict', 'unsupported_specificity')
    return any((bool(getattr(fuzzy_evidence, attribute, False)) for attribute in conflict_attributes))

def _candidate_support_score(candidate: Any) -> float:
    """
    Return a comparable 0-1 retrieval-support score.

    This is not a clinical confidence probability.
    It is used only to identify near-tied LLTs
    within an already selected PT.
    """
    scores: list[float] = []
    semantic_evidence = candidate.semantic_evidence
    if semantic_evidence is not None:
        similarity = getattr(semantic_evidence, 'cosine_similarity', None)
        if similarity is not None:
            scores.append(float(similarity))
    fuzzy_evidence = candidate.fuzzy_evidence
    if fuzzy_evidence is not None:
        ranking_score = getattr(fuzzy_evidence, 'ranking_score', None)
        if ranking_score is not None:
            scores.append(float(ranking_score) / 100.0)
    if not scores:
        return 0.0
    return max(scores)

def _candidate_source_count(candidate: Any) -> int:
    source_count = getattr(candidate, 'source_count', None)
    if source_count is not None:
        try:
            return int(source_count)
        except (TypeError, ValueError):
            pass
    evidence_sources = getattr(candidate, 'evidence_sources', None) or []
    return len({str(source) for source in evidence_sources})

def _selected_has_decisive_multisource_support(selected_candidate: Any, same_pt_candidates: list[Any], *, minimum_score_advantage: float=0.015) -> bool:
    competitors = [candidate for candidate in same_pt_candidates if candidate.candidate_id != selected_candidate.candidate_id]
    if not competitors:
        return False
    selected_source_count = _candidate_source_count(selected_candidate)
    if selected_source_count < 3:
        return False
    strongest_competitor = max(competitors, key=_candidate_support_score)
    competitor_source_count = _candidate_source_count(strongest_competitor)
    score_advantage = _candidate_support_score(selected_candidate) - _candidate_support_score(strongest_competitor)
    return selected_source_count >= competitor_source_count + 2 and score_advantage >= minimum_score_advantage

def find_same_pt_llt_ambiguity(reviewer_input: ReviewerInput, reviewer_decision: ReviewerDecision, *, score_margin: float=0.02, meaningful_modifier_margin: float=0.04, minimum_support_score: float=0.65) -> list[str]:
    """
    Identify substantively different, near-tied LLTs
    within the PT selected by a matched-single decision.

    Spelling-only variants are collapsed so that
    Hematocrit/Haematocrit alone does not cause
    human-review escalation.
    """
    if reviewer_decision.decision_type != ReviewerDecisionType.MATCHED_SINGLE:
        return []
    if reviewer_input.input_interpretation.structure_type != InputStructureType.SINGLE_CONCEPT:
        return []
    if len(reviewer_decision.selected_candidate_ids) != 1:
        return []
    candidates = reviewer_input.aggregation_result.candidates
    candidate_by_id = {candidate.candidate_id: candidate for candidate in candidates}
    selected_candidate_id = reviewer_decision.selected_candidate_ids[0]
    selected_candidate = candidate_by_id.get(selected_candidate_id)
    if selected_candidate is None:
        return []
    if _labels_are_equivalent(reviewer_input.normalized_term, selected_candidate.llt_name):
        return []
    same_pt_candidates = [candidate for candidate in candidates if candidate.pt_code == selected_candidate.pt_code and (not _candidate_has_conflict(candidate))]
    if len(same_pt_candidates) < 2:
        return []
    representative_by_label: dict[str, tuple[Any, float, float]] = {}
    for candidate in same_pt_candidates:
        normalized_label = _normalized_llt_label(candidate.llt_name)
        support_score = _candidate_support_score(candidate)
        lexical_overlap = _input_token_overlap(reviewer_input.normalized_term, candidate.llt_name)
        existing = representative_by_label.get(normalized_label)
        current_priority = (lexical_overlap, support_score)
        if existing is None or current_priority > (existing[1], existing[2]):
            representative_by_label[normalized_label] = (candidate, lexical_overlap, support_score)
    representatives = [(normalized_label, candidate, support_score) for normalized_label, (candidate, _, support_score) in representative_by_label.items() if support_score >= minimum_support_score]
    if len(representatives) < 2:
        return []
    best_score = max((support_score for _, _, support_score in representatives))
    best_representative = max(representatives, key=lambda item: item[2])
    best_candidate = best_representative[1]
    near_tied = []
    for normalized_label, candidate, support_score in representatives:
        effective_margin = meaningful_modifier_margin if _has_meaningful_modifier_difference(best_candidate.llt_name, candidate.llt_name) else score_margin
        if best_score - support_score <= effective_margin:
            near_tied.append((normalized_label, candidate, support_score))
    if len(near_tied) < 2:
        return []
    selected_normalized_label = _normalized_llt_label(selected_candidate.llt_name)
    near_tied_labels = {normalized_label for normalized_label, _, _ in near_tied}
    if selected_normalized_label not in near_tied_labels:
        return []
    near_tied.sort(key=lambda item: (-item[2], item[1].llt_name.casefold()))
    near_tied_label_order = {normalized_label: position for position, (normalized_label, _, _) in enumerate(near_tied)}
    ambiguous_candidates = [candidate for candidate in same_pt_candidates if _normalized_llt_label(candidate.llt_name) in near_tied_label_order]
    ambiguous_candidates.sort(key=lambda candidate: (near_tied_label_order[_normalized_llt_label(candidate.llt_name)], -_candidate_support_score(candidate), candidate.llt_name.casefold(), candidate.candidate_id))
    return [candidate.candidate_id for candidate in ambiguous_candidates]

def apply_same_pt_ambiguity_gate(reviewer_input: ReviewerInput, reviewer_decision: ReviewerDecision, *, score_margin: float=0.02, meaningful_modifier_margin: float=0.04, minimum_support_score: float=0.65) -> tuple[ReviewerDecision, list[str]]:
    ambiguous_candidate_ids = find_same_pt_llt_ambiguity(reviewer_input, reviewer_decision, score_margin=score_margin, meaningful_modifier_margin=meaningful_modifier_margin, minimum_support_score=minimum_support_score)
    if len(ambiguous_candidate_ids) < 2:
        return (reviewer_decision, [])
    ambiguous_id_set = set(ambiguous_candidate_ids)
    existing_assessments = {assessment.candidate_id: assessment for assessment in reviewer_decision.candidate_assessments}
    candidate_assessments = []
    for candidate in reviewer_input.aggregation_result.candidates:
        if candidate.candidate_id in ambiguous_id_set:
            assessment = CandidateAssessment(candidate_id=candidate.candidate_id, verdict=CandidateReviewVerdict.UNCERTAIN, rationale=f"'{candidate.llt_name}' is a plausible LLT within the supported PT '{candidate.pt_name}'. The available evidence does not clearly distinguish it from the other near-equivalent LLT alternatives.")
        else:
            existing_assessment = existing_assessments.get(candidate.candidate_id)
            if existing_assessment is not None:
                assessment = existing_assessment.model_copy(update={'verdict': CandidateReviewVerdict.NOT_SUPPORTED})
            else:
                assessment = CandidateAssessment(candidate_id=candidate.candidate_id, verdict=CandidateReviewVerdict.NOT_SUPPORTED, rationale='This candidate is not part of the unresolved same-PT LLT choice.')
        candidate_assessments.append(assessment)
    selected_candidate = next((candidate for candidate in reviewer_input.aggregation_result.candidates if candidate.candidate_id == reviewer_decision.selected_candidate_ids[0]))
    repaired_decision = ReviewerDecision(term_id=reviewer_decision.term_id, review_round=reviewer_decision.review_round, decision_type=ReviewerDecisionType.MATCHED_PT_LLT_HUMAN_REVIEW, selected_candidate_ids=[], component_matches=[], candidate_assessments=candidate_assessments, evidence_requests=[], human_review_candidate_ids=ambiguous_candidate_ids, decision_rationale=f"The PT-level clinical concept is supported, but multiple LLTs within the PT '{selected_candidate.pt_name}' have near-equivalent retrieval support. The pipeline cannot make a defensible automatic LLT selection.", multiple_match_explanation=None, human_review_reason='Expert MedDRA coding review is required to choose among near-equivalent LLTs within an otherwise supported PT.')
    repair_actions = [f'Escalated a near-tied same-PT LLT selection while preserving the PT mapping. Candidate IDs: {ambiguous_candidate_ids}']
    return (repaired_decision, repair_actions)
