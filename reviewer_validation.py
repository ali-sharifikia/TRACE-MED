from runtime_resources import resource as asp_resource
import re
from typing import Any
from schemas import CandidateReviewVerdict, EvidenceRequestAction, EvidenceSource, ReviewerDecision, ReviewerDecisionType, ReviewerInput, InputStructureType
from state import StandardizationState

class ReviewerDecisionValidationError(ValueError):
    """
    Raised when a structurally valid reviewer
    decision is incompatible with the evidence
    package supplied to the reviewer.
    """

class ClinicalMappingValidationError(ReviewerDecisionValidationError):
    """A supported clinical contradiction, not a structural or service error."""

    def __init__(self, candidate_ids, reasons):
        self.candidate_ids = sorted(set(candidate_ids))
        self.reasons = sorted(set(reasons))
        super().__init__('A final component mapping has explicit clinical conflicts: ' + ', '.join(self.reasons))

def validate_selected_clinical_mappings(reviewer_input, reviewer_decision, supplemental_candidates=None, *, deterministic_gate=None, measurement_evidence=False, shared_evidence=False):
    from pt_reranking import explicit_clinical_conflicts
    if reviewer_decision.decision_type not in {ReviewerDecisionType.MATCHED_SINGLE, ReviewerDecisionType.MATCHED_MULTIPLE}:
        return
    candidates = {c.candidate_id: c for c in _combined_candidates(reviewer_input, supplemental_candidates)}
    components = {c.component_id: c for c in reviewer_input.input_interpretation.asserted_components}
    from compound_validation import assess_decomposition
    scope = assess_decomposition(reviewer_input.normalized_term, list(components.values()))
    effective = {c['component_id']: c['effective_text'] for c in scope['components']}
    rejected, reasons = ([], [])
    for match in reviewer_decision.component_matches:
        if match.component_id not in components or match.candidate_id not in candidates:
            return
        component, candidate = (components[match.component_id], candidates[match.candidate_id])
        text = effective.get(component.component_id, component.component_text)
        conflicts = set(explicit_clinical_conflicts(text, candidate.llt_name))
        conflicts.update(explicit_clinical_conflicts(text, candidate.pt_name))
        if measurement_evidence or (deterministic_gate and deterministic_gate.coverage_acceptance.get('functionality_v1_19', {}).get('configuration', {}).get('measurement_repair')):
            from measurement_interpretation import compare
            labels = (candidate.llt_name, candidate.pt_name)
            checked = [compare(text, label) for label in labels]
            if all(('directionality_conflict' not in explicit_clinical_conflicts(text, label) or e['duration_direction_compatible'] for label, e in zip(labels, checked))):
                conflicts.discard('directionality_conflict')
        v120 = deterministic_gate.coverage_acceptance.get('development_v1_7', {}).get('comparison_v1_20', {}) if deterministic_gate else {}
        if shared_evidence or v120.get('added_LLT'):
            from clinical_interpretation import compare, canonical
            conflicts = set(explicit_clinical_conflicts(canonical(text), canonical(candidate.llt_name)))
            conflicts.update(explicit_clinical_conflicts(canonical(text), canonical(candidate.pt_name)))
            for label in (candidate.llt_name, candidate.pt_name):
                evidence = compare(text, label)
                conflicts.update(evidence['contradictions'])
            if all(('directionality_conflict' not in explicit_clinical_conflicts(canonical(text), canonical(label)) or compare(text, label)['duration_polarity_compatible'] for label in (candidate.llt_name, candidate.pt_name))):
                conflicts.discard('directionality_conflict')
        if deterministic_gate and 'terminology_v1_22' in deterministic_gate.coverage_acceptance:
            raise ValueError('This historical opt-in is not supported by ASP-MT 1.0')
        if deterministic_gate and 'terminology_v1_23' in deterministic_gate.coverage_acceptance:
            raise ValueError('This historical opt-in is not supported by ASP-MT 1.0')
        if deterministic_gate and 'wording_v1_24' in deterministic_gate.coverage_acceptance:
            raise ValueError('This historical opt-in is not supported by ASP-MT 1.0')
        if conflicts:
            rejected.append(match.candidate_id)
            reasons.extend(conflicts)
    if rejected:
        validate_reviewer_decision(reviewer_input, reviewer_decision, supplemental_candidates, check_clinical=False, deterministic_gate=deterministic_gate)
        raise ClinicalMappingValidationError(rejected, reasons)
LLT_REFERENCE_PATTERN = re.compile('\\bLLT\\s*:\\s*([A-Za-z0-9._-]*\\d[A-Za-z0-9._-]*)\\b', flags=re.IGNORECASE)
PT_REFERENCE_PATTERN = re.compile('\\bPT\\s*:\\s*([A-Za-z0-9._-]*\\d[A-Za-z0-9._-]*)\\b', flags=re.IGNORECASE)
UMLS_CUI_REFERENCE_PATTERN = re.compile('\\bC\\d{7}\\b', flags=re.IGNORECASE)

def _combined_candidates(reviewer_input: ReviewerInput, supplemental_candidates: list[Any] | None) -> list[Any]:
    """Merge aggregated and post-reviewer LLT candidates by identity."""
    candidates = list(reviewer_input.aggregation_result.candidates)
    candidate_by_id = {candidate.candidate_id: candidate for candidate in candidates}
    for candidate in supplemental_candidates or []:
        existing = candidate_by_id.get(candidate.candidate_id)
        if existing is not None:
            if existing.llt_code != candidate.llt_code or existing.pt_code != candidate.pt_code:
                raise ReviewerDecisionValidationError(f'A supplemental LLT candidate conflicts with an aggregated candidate identity: {candidate.candidate_id}')
            continue
        candidates.append(candidate)
        candidate_by_id[candidate.candidate_id] = candidate
    return candidates

def _validate_identifiers_in_rationales(reviewer_input: ReviewerInput, reviewer_decision: ReviewerDecision, supplemental_candidates: list[Any] | None=None) -> None:
    """
    Reject LLT or PT codes in free-text rationales when those
    codes were not present in the supplied reviewer evidence.
    """
    candidates = _combined_candidates(reviewer_input, supplemental_candidates)
    allowed_llt_codes = {str(candidate.llt_code).casefold() for candidate in candidates if candidate.llt_code is not None}
    allowed_pt_codes = {str(candidate.pt_code).casefold() for candidate in candidates if candidate.pt_code is not None}
    allowed_umls_cuis: set[str] = set()
    for candidate in candidates:
        umls_evidence = getattr(candidate, 'umls_evidence', None)
        if umls_evidence is not None and umls_evidence.umls_cui is not None:
            allowed_umls_cuis.add(str(umls_evidence.umls_cui).casefold())
    rationale_texts: list[str] = [reviewer_decision.decision_rationale]
    if reviewer_decision.multiple_match_explanation:
        rationale_texts.append(reviewer_decision.multiple_match_explanation)
    if reviewer_decision.human_review_reason:
        rationale_texts.append(reviewer_decision.human_review_reason)
    rationale_texts.extend((assessment.rationale for assessment in reviewer_decision.candidate_assessments))
    rationale_texts.extend((component_match.rationale for component_match in reviewer_decision.component_matches))
    rationale_texts.extend((evidence_request.reason for evidence_request in reviewer_decision.evidence_requests))
    referenced_llt_codes: set[str] = set()
    referenced_pt_codes: set[str] = set()
    referenced_umls_cuis: set[str] = set()
    for text in rationale_texts:
        if not text:
            continue
        referenced_llt_codes.update((match.casefold() for match in LLT_REFERENCE_PATTERN.findall(text)))
        referenced_pt_codes.update((match.casefold() for match in PT_REFERENCE_PATTERN.findall(text)))
        referenced_umls_cuis.update((match.casefold() for match in UMLS_CUI_REFERENCE_PATTERN.findall(text)))
    unknown_llt_codes = referenced_llt_codes - allowed_llt_codes
    unknown_pt_codes = referenced_pt_codes - allowed_pt_codes
    unknown_umls_cuis = referenced_umls_cuis - allowed_umls_cuis
    if unknown_llt_codes:
        raise ReviewerDecisionValidationError(f'Reviewer rationale contains LLT codes that were not supplied in the evidence: {sorted(unknown_llt_codes)}')
    if unknown_pt_codes:
        raise ReviewerDecisionValidationError(f'Reviewer rationale contains PT codes that were not supplied in the evidence: {sorted(unknown_pt_codes)}')
    if unknown_umls_cuis:
        raise ReviewerDecisionValidationError(f'Reviewer rationale contains UMLS CUIs that were not supplied in the evidence: {sorted(unknown_umls_cuis)}')

def validate_multiple_acceptance(reviewer_input, reviewer_decision, policy_config, **evidence):
    """Final component acceptance; structural reviewer validation alone is provisional.

The post-review gate supplies current-reference, per-component LLT ranks. A
failed assessment is deterministic deferral, not a retryable schema exception.
"""
    from compound_validation import assess_multiple_acceptance
    return assess_multiple_acceptance(reviewer_input, reviewer_decision, policy_config, **evidence)

def validate_reviewer_decision(reviewer_input: ReviewerInput, reviewer_decision: ReviewerDecision, supplemental_candidates: list[Any] | None=None, *, check_clinical: bool=True, deterministic_gate=None) -> ReviewerDecision:
    """
    Validate an LLM decision against the actual
    reviewer input for one term.

    Pydantic validates the decision internally.
    This function validates references between
    that decision and the supplied candidates.
    """
    deterministic_scope = False
    if deterministic_gate is not None:
        from assessment_scope import verify_contract
        deterministic_scope = verify_contract(reviewer_input, reviewer_decision, supplemental_candidates, deterministic_gate)
    if reviewer_decision.term_id != reviewer_input.term_id:
        raise ReviewerDecisionValidationError('Reviewer decision term_id does not match ReviewerInput.')
    if reviewer_decision.review_round != reviewer_input.review_round:
        raise ReviewerDecisionValidationError('Reviewer decision review_round does not match ReviewerInput.')
    structure_type = reviewer_input.input_interpretation.structure_type
    if reviewer_decision.decision_type == ReviewerDecisionType.MATCHED_MULTIPLE and structure_type != InputStructureType.MULTIPLE_CONCEPTS:
        raise ReviewerDecisionValidationError('MATCHED_MULTIPLE is only valid when InputInterpretation.structure_type is multiple_concepts.')
    if structure_type == InputStructureType.NEGATED_NON_EVENT and reviewer_decision.decision_type in {ReviewerDecisionType.MATCHED_SINGLE, ReviewerDecisionType.MATCHED_MULTIPLE}:
        raise ReviewerDecisionValidationError('A completely negated non-event cannot produce a final matched decision.')
    if structure_type == InputStructureType.AMBIGUOUS_CONCEPT and reviewer_decision.decision_type != ReviewerDecisionType.HUMAN_REVIEW:
        raise ReviewerDecisionValidationError('An input classified as AMBIGUOUS_CONCEPT requires HUMAN_REVIEW. Do not select one interpretation or return NO_SUPPORTED_MATCH.')
    component_matches = reviewer_decision.component_matches
    matched_decision_types = {ReviewerDecisionType.MATCHED_SINGLE, ReviewerDecisionType.MATCHED_MULTIPLE}
    if reviewer_decision.decision_type not in matched_decision_types and component_matches:
        raise ReviewerDecisionValidationError('Only a final matched decision may contain component_matches.')
    if component_matches:
        available_component_ids = {component.component_id for component in reviewer_input.input_interpretation.asserted_components}
        positive_component_ids = {component.component_id for component in reviewer_input.input_interpretation.asserted_components if not component.is_negated}
        mapped_component_ids = [component_match.component_id for component_match in component_matches]
        mapped_candidate_ids = [component_match.candidate_id for component_match in component_matches]
        if len(mapped_component_ids) != len(set(mapped_component_ids)):
            raise ReviewerDecisionValidationError('A positively asserted component cannot be mapped more than once.')
        if len(mapped_candidate_ids) != len(set(mapped_candidate_ids)):
            raise ReviewerDecisionValidationError('A selected candidate cannot be mapped to more than one component.')
        unknown_component_ids = set(mapped_component_ids) - available_component_ids
        if unknown_component_ids:
            raise ReviewerDecisionValidationError(f'component_matches references component IDs that were not supplied: {sorted(unknown_component_ids)}')
        if set(mapped_candidate_ids) != set(reviewer_decision.selected_candidate_ids):
            raise ReviewerDecisionValidationError('The candidate IDs in component_matches must exactly equal selected_candidate_ids.')
        if reviewer_decision.decision_type in matched_decision_types:
            missing_component_ids = positive_component_ids - set(mapped_component_ids)
            extra_component_ids = set(mapped_component_ids) - positive_component_ids
            if missing_component_ids:
                raise ReviewerDecisionValidationError(f'A final matched decision does not cover every positively asserted input component. Missing components: {sorted(missing_component_ids)}. Request focused retrieval for the missing component before finalizing the match.')
            if extra_component_ids:
                raise ReviewerDecisionValidationError(f'A final matched decision maps components that are not positively asserted: {sorted(extra_component_ids)}')
    available_candidates = _combined_candidates(reviewer_input, supplemental_candidates)
    candidate_by_id = {candidate.candidate_id: candidate for candidate in available_candidates}
    if len(candidate_by_id) != len(available_candidates):
        raise ReviewerDecisionValidationError('ReviewerInput contains duplicate candidate IDs.')
    _validate_identifiers_in_rationales(reviewer_input=reviewer_input, reviewer_decision=reviewer_decision, supplemental_candidates=supplemental_candidates)
    available_candidate_ids = set(candidate_by_id)
    assessed_candidate_ids = {assessment.candidate_id for assessment in reviewer_decision.candidate_assessments}
    selected_candidate_ids = set(reviewer_decision.selected_candidate_ids)
    human_review_candidate_ids = set(reviewer_decision.human_review_candidate_ids)
    request_candidate_ids = {candidate_id for request in reviewer_decision.evidence_requests for candidate_id in request.candidate_ids_of_interest}
    component_match_candidate_ids = {component_match.candidate_id for component_match in reviewer_decision.component_matches}
    referenced_candidate_ids = assessed_candidate_ids | selected_candidate_ids | human_review_candidate_ids | request_candidate_ids | component_match_candidate_ids
    unknown_candidate_ids = referenced_candidate_ids - available_candidate_ids
    if unknown_candidate_ids:
        raise ReviewerDecisionValidationError(f'Reviewer decision references candidate IDs that were not supplied: {sorted(unknown_candidate_ids)}')
    available_pt_codes = {candidate.pt_code for candidate in available_candidates}
    referenced_component_pt_codes = {component_match.pt_code for component_match in component_matches}
    unknown_component_pt_codes = referenced_component_pt_codes - available_pt_codes
    if unknown_component_pt_codes:
        raise ReviewerDecisionValidationError(f'component_matches references PT codes that were not supplied in the reviewer evidence: {sorted(unknown_component_pt_codes)}')
    invalid_pt_llt_links = []
    for component_match in component_matches:
        candidate = candidate_by_id[component_match.candidate_id]
        if candidate.pt_code != component_match.pt_code:
            invalid_pt_llt_links.append({'component_id': component_match.component_id, 'candidate_id': component_match.candidate_id, 'provided_pt_code': component_match.pt_code, 'expected_pt_code': candidate.pt_code})
    if invalid_pt_llt_links:
        raise ReviewerDecisionValidationError(f'A selected LLT must belong to the PT recorded in its component_matches entry. Invalid PT-to-LLT relationships: {invalid_pt_llt_links}')
    if reviewer_decision.decision_type == ReviewerDecisionType.MATCHED_PT_LLT_HUMAN_REVIEW:
        deferred_pt_codes = {candidate_by_id[candidate_id].pt_code for candidate_id in human_review_candidate_ids}
        explicit_pt = reviewer_decision.resolved_pt_code
        if explicit_pt is not None:
            identities = {(c.pt_code, c.pt_name) for c in candidate_by_id.values() if c.is_current}
            if (explicit_pt, reviewer_decision.resolved_pt_name) not in identities:
                raise ReviewerDecisionValidationError('Explicit resolved PT identity is absent from current supplied evidence.')
            if deferred_pt_codes and deferred_pt_codes != {explicit_pt}:
                raise ReviewerDecisionValidationError('Deferred LLT parent conflicts with explicit resolved PT.')
        if len(deferred_pt_codes) != 1 and (not (not deferred_pt_codes and explicit_pt)):
            raise ReviewerDecisionValidationError(f'MATCHED_PT_LLT_HUMAN_REVIEW requires every deferred LLT to belong to exactly one resolved PT. Observed PT codes: {sorted(deferred_pt_codes)}')
        missing_assessments = available_candidate_ids - assessed_candidate_ids
        if missing_assessments:
            raise ReviewerDecisionValidationError(f'A PT-resolved LLT deferral must assess every supplied candidate. Missing assessments: {sorted(missing_assessments)}')
        supported_candidate_ids = {assessment.candidate_id for assessment in reviewer_decision.candidate_assessments if assessment.verdict == CandidateReviewVerdict.SUPPORTED}
        if supported_candidate_ids:
            raise ReviewerDecisionValidationError(f'A PT-resolved LLT deferral cannot mark an LLT as finally supported. Use uncertain for the deferred LLTs. Supported candidate IDs: {sorted(supported_candidate_ids)}')
    final_decision_types = {ReviewerDecisionType.MATCHED_SINGLE, ReviewerDecisionType.MATCHED_MULTIPLE, ReviewerDecisionType.NO_SUPPORTED_MATCH}
    if reviewer_decision.decision_type in final_decision_types:
        missing_assessments = available_candidate_ids - assessed_candidate_ids
        if missing_assessments:
            raise ReviewerDecisionValidationError(f'A final decision must assess every candidate supplied to the reviewer. Missing assessments: {sorted(missing_assessments)}')
    if reviewer_decision.decision_type == ReviewerDecisionType.NO_SUPPORTED_MATCH and reviewer_input.review_round == 1 and (len(available_candidates) == 1):
        only_candidate = available_candidates[0]
        fuzzy_evidence = only_candidate.fuzzy_evidence
        if fuzzy_evidence is not None and fuzzy_evidence.directionality_conflict:
            raise ReviewerDecisionValidationError('NO_SUPPORTED_MATCH is premature on review round 1. The only retrieved candidate has an explicit directionality conflict with the positively asserted input concept. Reject that candidate and return REQUEST_MORE_EVIDENCE with an EXPAND_CANDIDATES request for fuzzy or semantic retrieval. requested_top_k must be greater than the current candidate count.')
    supported_candidate_ids = {assessment.candidate_id for assessment in reviewer_decision.candidate_assessments if assessment.verdict == CandidateReviewVerdict.SUPPORTED}
    matched_decision_types = {ReviewerDecisionType.MATCHED_SINGLE, ReviewerDecisionType.MATCHED_MULTIPLE}
    if reviewer_decision.decision_type in matched_decision_types and supported_candidate_ids != selected_candidate_ids:
        selected_without_support = selected_candidate_ids - supported_candidate_ids
        supported_without_selection = supported_candidate_ids - selected_candidate_ids
        raise ReviewerDecisionValidationError(f'For a final matched decision, the candidates marked supported must exactly equal selected_candidate_ids. Selected candidate IDs: {sorted(selected_candidate_ids)}. Candidates assessed as supported: {sorted(supported_candidate_ids)}. Selected but not assessed as supported: {sorted(selected_without_support)}. Assessed as supported but not selected: {sorted(supported_without_selection)}.')
    if reviewer_decision.decision_type == ReviewerDecisionType.MATCHED_MULTIPLE:
        from biomedical_wording import generic_context_base, is_generic_event
        if generic_context_base(reviewer_input.normalized_term) is not None:
            if any((is_generic_event(candidate_by_id[identifier].llt_name) or is_generic_event(candidate_by_id[identifier].pt_name) for identifier in reviewer_decision.selected_candidate_ids)):
                raise ReviewerDecisionValidationError('A generic contextual event/reaction suffix cannot be a separately matched clinical concept.')
        selected_candidates = [candidate_by_id[candidate_id] for candidate_id in reviewer_decision.selected_candidate_ids]
        selected_pt_codes = [candidate.pt_code for candidate in selected_candidates]
        if len(selected_pt_codes) != len(set(selected_pt_codes)):
            raise ReviewerDecisionValidationError('MATCHED_MULTIPLE cannot select multiple synonymous LLTs from the same PT. Select one LLT per distinct clinical concept.')
    positive_component_terms = {' '.join(component.component_text.casefold().split()) for component in reviewer_input.input_interpretation.asserted_components if not component.is_negated}
    if reviewer_input.input_interpretation.structure_type == InputStructureType.MULTIPLE_CONCEPTS and reviewer_input.review_round == 1 and (reviewer_decision.decision_type == ReviewerDecisionType.NO_SUPPORTED_MATCH):
        raise ReviewerDecisionValidationError(f'A multiple-concept input cannot be finalized as no_supported_match in the first review round. If a positively asserted component has no candidate, return request_more_evidence with a focused_search for that exact component. Positively asserted component terms: {sorted(positive_component_terms)}')
    if reviewer_decision.decision_type == ReviewerDecisionType.REQUEST_MORE_EVIDENCE and reviewer_input.input_interpretation.structure_type == InputStructureType.MULTIPLE_CONCEPTS:
        invalid_requests = [request.request_id for request in reviewer_decision.evidence_requests if request.action != EvidenceRequestAction.FOCUSED_SEARCH]
        if invalid_requests:
            raise ReviewerDecisionValidationError(f'Multiple-concept inputs must use focused_search for an unresolved asserted component. Invalid request IDs: {sorted(invalid_requests)}')
    request_signatures = set()
    for request in reviewer_decision.evidence_requests:
        normalized_query = request.query_term.casefold() if request.query_term is not None else None
        request_signature = (request.source, request.action, normalized_query, request.requested_top_k)
        if request_signature in request_signatures:
            raise ReviewerDecisionValidationError('The reviewer submitted duplicate evidence requests.')
        request_signatures.add(request_signature)
        if reviewer_input.input_interpretation.structure_type == InputStructureType.MULTIPLE_CONCEPTS and request.action == EvidenceRequestAction.EXPAND_CANDIDATES:
            raise ReviewerDecisionValidationError('Multiple-concept inputs must use focused_search for an unresolved asserted component. Do not expand candidates using the complete compound input.')
        if request.action == EvidenceRequestAction.EXPAND_CANDIDATES:
            if request.source == EvidenceSource.FUZZY:
                current_candidate_count = reviewer_input.aggregation_result.fuzzy_candidate_count
            elif request.source == EvidenceSource.SEMANTIC:
                current_candidate_count = reviewer_input.aggregation_result.semantic_candidate_count
            else:
                raise ReviewerDecisionValidationError('Only fuzzy and semantic retrieval can expand candidate counts.')
            if request.requested_top_k is None or request.requested_top_k <= current_candidate_count:
                raise ReviewerDecisionValidationError('An expanded candidate request must ask for more candidates than are currently available.')
        if request.action == EvidenceRequestAction.FOCUSED_SEARCH:
            if request.query_term is None:
                raise ReviewerDecisionValidationError('Focused search requires a query term.')
            normalized_focused_query = ' '.join(request.query_term.casefold().split())
            normalized_complete_input = ' '.join(reviewer_input.normalized_term.casefold().split())
            if normalized_focused_query == normalized_complete_input:
                raise ReviewerDecisionValidationError('A focused search must use a more focused subterm, not the unchanged normalized input.')
            if reviewer_input.input_interpretation.structure_type == InputStructureType.MULTIPLE_CONCEPTS and normalized_focused_query not in positive_component_terms:
                raise ReviewerDecisionValidationError(f'A focused-search query must exactly identify one positively asserted input component. Allowed component terms: {sorted(positive_component_terms)}')
    if reviewer_decision.decision_type in final_decision_types:
        uncertain_candidate_ids = [assessment.candidate_id for assessment in reviewer_decision.candidate_assessments if assessment.verdict == CandidateReviewVerdict.UNCERTAIN]
        if uncertain_candidate_ids and (not deterministic_scope):
            raise ReviewerDecisionValidationError(f'A final decision cannot contain uncertain candidate assessments. Request more evidence or human review instead. Uncertain candidates: {sorted(uncertain_candidate_ids)}')
    if check_clinical:
        validate_selected_clinical_mappings(reviewer_input, reviewer_decision, supplemental_candidates, deterministic_gate=deterministic_gate)
    return reviewer_decision

def reviewer_decision_validation_node(state: StandardizationState) -> dict[str, ReviewerDecision]:
    """
    LangGraph-compatible validation node.

    In the final graph, this runs immediately after
    the reviewer LLM node.
    """
    reviewer_input = state.get('reviewer_input')
    if reviewer_input is None:
        raise ValueError('Decision validation requires reviewer_input.')
    reviewer_decision = state.get('reviewer_decision')
    if reviewer_decision is None:
        raise ValueError('Decision validation requires reviewer_decision.')
    gate_result = state.get('llt_selection_gate_result')
    supplemental_candidates = gate_result.human_review_candidates + gate_result.ranked_candidates if gate_result is not None else []
    validated_decision = validate_reviewer_decision(reviewer_input=reviewer_input, reviewer_decision=reviewer_decision, supplemental_candidates=supplemental_candidates, deterministic_gate=gate_result if gate_result is not None and 'assessment_scope_v1_7_1' in gate_result.coverage_acceptance else None)
    if gate_result is None and validated_decision.decision_type == ReviewerDecisionType.MATCHED_MULTIPLE:
        from llt_selection_gate import apply_llt_selection_gate

        class UnavailableReference:

            def entries_for_pt(self, pt_code):
                return ()

            def rank(self, **kwargs):
                return []
        deferred, missing_reference_gate = apply_llt_selection_gate(reviewer_input=reviewer_input, reviewer_decision=validated_decision, reranker=UnavailableReference(), reviewer_failure_diagnostics=state.get('reviewer_failure_diagnostics'))
        validate_reviewer_decision(reviewer_input, deferred, missing_reference_gate.human_review_candidates)
        return {'reviewer_decision': deferred, 'llt_selection_gate_result': missing_reference_gate}
    return {'reviewer_decision': validated_decision}
