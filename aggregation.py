from runtime_resources import resource as asp_resource
from typing import Any
from schemas import AggregatedCandidate, AggregatedPTGroup, AggregationResult, EvidenceSource, FuzzyCandidate, MeddraConcept, SemanticCandidate, UMLSCandidate
from state import StandardizationState

def _extract_concept_data(candidate: MeddraConcept) -> dict[str, Any]:
    """
    Extract every field defined by the project's
    current MeddraConcept schema.

    This includes candidate_id and avoids manually
    duplicating the common fields.
    """
    return {field_name: getattr(candidate, field_name) for field_name in MeddraConcept.model_fields}

def _validate_same_concept(existing_data: dict[str, Any], candidate: MeddraConcept) -> None:
    """
    Ensure that evidence sources using the same
    LLT code also agree on its MedDRA metadata.
    """
    new_data = _extract_concept_data(candidate)
    conflicting_fields = [field_name for field_name in existing_data if existing_data[field_name] != new_data[field_name]]
    if conflicting_fields:
        raise ValueError(f'Conflicting MedDRA information for LLT code {candidate.llt_code}. Conflicting fields: {conflicting_fields}')

def aggregation_node(state: StandardizationState) -> dict[str, AggregationResult]:
    """
    Combine fuzzy, semantic and UMLS candidates
    by LLT code.

    This function does not rank, accept or reject
    candidates.
    """
    normalization_result = state.get('normalization')
    if normalization_result is None:
        raise ValueError('Aggregation requires the normalization result.')
    exact_result = state.get('exact_result')
    if exact_result is None:
        raise ValueError('Aggregation requires the exact-matching result.')
    if exact_result.found:
        raise ValueError('An exact-matched term should not be sent to candidate aggregation.')
    fuzzy_result = state.get('fuzzy_result')
    semantic_result = state.get('semantic_result')
    umls_result = state.get('umls_result')
    if fuzzy_result is None:
        raise ValueError('Aggregation requires the fuzzy matching result.')
    if semantic_result is None:
        raise ValueError('Aggregation requires the semantic matching result.')
    if umls_result is None:
        raise ValueError('Aggregation requires the UMLS lookup result.')
    normalized_term = normalization_result.normalized_term
    source_terms = {'fuzzy': fuzzy_result.normalized_input_term, 'semantic': semantic_result.normalized_input_term, 'umls': umls_result.normalized_input_term}
    inconsistent_sources = [source_name for source_name, source_term in source_terms.items() if source_term != normalized_term]
    if inconsistent_sources:
        raise ValueError(f'Candidate sources used different normalized input terms: {inconsistent_sources}')
    source_query_terms = {'fuzzy': fuzzy_result.retrieval_query_term or fuzzy_result.normalized_input_term, 'semantic': semantic_result.retrieval_query_term or semantic_result.normalized_input_term, 'umls': umls_result.retrieval_query_term or umls_result.normalized_input_term}
    direct_pt_sources: dict[str, set[EvidenceSource]] = {}
    direct_pt_variant_ids: dict[str, set[str]] = {}
    for source, result in ((EvidenceSource.FUZZY, fuzzy_result), (EvidenceSource.SEMANTIC, semantic_result), (EvidenceSource.UMLS, umls_result)):
        targeted_variants = {variant.variant_id: variant.target_pt_code for variant in result.query_variants if variant.target_pt_code is not None}
        for candidate in result.candidates:
            introduction_ids = set(getattr(candidate, 'introduction_query_variant_ids', []))
            matched_variant_ids = {variant_id for variant_id in introduction_ids if targeted_variants.get(variant_id) == candidate.pt_code}
            direct_pt_match = bool(getattr(candidate, 'direct_pt_match_score', None) is not None)
            if matched_variant_ids or direct_pt_match:
                direct_pt_sources.setdefault(candidate.pt_code, set()).add(source)
                direct_pt_variant_ids.setdefault(candidate.pt_code, set()).update(matched_variant_ids)
    accumulators: dict[str, dict[str, Any]] = {}

    def add_candidate(candidate: MeddraConcept, evidence_field: str) -> None:
        llt_code = candidate.llt_code
        if llt_code not in accumulators:
            accumulators[llt_code] = {'concept_data': _extract_concept_data(candidate), 'fuzzy_evidence': None, 'semantic_evidence': None, 'umls_evidence': None}
        accumulator = accumulators[llt_code]
        _validate_same_concept(accumulator['concept_data'], candidate)
        if accumulator[evidence_field] is not None:
            raise ValueError(f'Duplicate evidence from the same source for LLT {llt_code}.')
        accumulator[evidence_field] = candidate
    for candidate in fuzzy_result.candidates:
        add_candidate(candidate=candidate, evidence_field='fuzzy_evidence')
    for candidate in semantic_result.candidates:
        add_candidate(candidate=candidate, evidence_field='semantic_evidence')
    for candidate in umls_result.candidates:
        add_candidate(candidate=candidate, evidence_field='umls_evidence')
    aggregated_candidates = []
    for llt_code, accumulator in accumulators.items():
        evidence_sources = []
        fuzzy_evidence: FuzzyCandidate | None = accumulator['fuzzy_evidence']
        semantic_evidence: SemanticCandidate | None = accumulator['semantic_evidence']
        umls_evidence: UMLSCandidate | None = accumulator['umls_evidence']
        if fuzzy_evidence is not None:
            evidence_sources.append(EvidenceSource.FUZZY)
        if semantic_evidence is not None:
            evidence_sources.append(EvidenceSource.SEMANTIC)
        if umls_evidence is not None:
            evidence_sources.append(EvidenceSource.UMLS)
        aggregated_candidates.append(AggregatedCandidate(**accumulator['concept_data'], evidence_sources=evidence_sources, source_count=len(evidence_sources), fuzzy_evidence=fuzzy_evidence, semantic_evidence=semantic_evidence, umls_evidence=umls_evidence, fuzzy_query_term=source_query_terms['fuzzy'] if fuzzy_evidence is not None else None, semantic_query_term=source_query_terms['semantic'] if semantic_evidence is not None else None, umls_query_term=source_query_terms['umls'] if umls_evidence is not None else None))
    aggregated_candidates.sort(key=lambda candidate: (candidate.pt_code, candidate.llt_code))
    pt_accumulators: dict[str, dict[str, Any]] = {}
    for candidate in aggregated_candidates:
        pt_code = candidate.pt_code
        if pt_code not in pt_accumulators:
            pt_accumulators[pt_code] = {'pt_name': candidate.pt_name, 'candidate_ids': [], 'llt_codes': [], 'evidence_sources': set()}
        pt_accumulator = pt_accumulators[pt_code]
        if pt_accumulator['pt_name'] != candidate.pt_name:
            raise ValueError(f'Conflicting PT names found for PT code {pt_code}.')
        pt_accumulator['candidate_ids'].append(candidate.candidate_id)
        pt_accumulator['llt_codes'].append(candidate.llt_code)
        pt_accumulator['evidence_sources'].update(candidate.evidence_sources)
    source_order = (EvidenceSource.FUZZY, EvidenceSource.SEMANTIC, EvidenceSource.UMLS)
    pt_groups = []
    for pt_code in sorted(pt_accumulators):
        pt_accumulator = pt_accumulators[pt_code]
        candidate_pairs = sorted(zip(pt_accumulator['llt_codes'], pt_accumulator['candidate_ids']), key=lambda pair: pair[0])
        llt_codes = [llt_code for llt_code, _ in candidate_pairs]
        candidate_ids = [candidate_id for _, candidate_id in candidate_pairs]
        pt_evidence_sources = [source for source in source_order if source in pt_accumulator['evidence_sources']]
        pt_groups.append(AggregatedPTGroup(pt_code=pt_code, pt_name=pt_accumulator['pt_name'], candidate_ids=candidate_ids, llt_codes=llt_codes, llt_count=len(llt_codes), evidence_sources=pt_evidence_sources, source_count=len(pt_evidence_sources), direct_pt_evidence_sources=[source for source in source_order if source in direct_pt_sources.get(pt_code, set())], direct_pt_query_variant_ids=sorted(direct_pt_variant_ids.get(pt_code, set()))))
    fuzzy_candidate_count = len(fuzzy_result.candidates)
    semantic_candidate_count = len(semantic_result.candidates)
    umls_candidate_count = len(umls_result.candidates)
    aggregation_result = AggregationResult(normalized_input_term=normalized_term, fuzzy_candidate_count=fuzzy_candidate_count, semantic_candidate_count=semantic_candidate_count, umls_candidate_count=umls_candidate_count, total_source_candidates=fuzzy_candidate_count + semantic_candidate_count + umls_candidate_count, unique_llt_count=len(aggregated_candidates), multiple_source_llt_count=sum((candidate.source_count > 1 for candidate in aggregated_candidates)), unique_pt_count=len(pt_groups), multiple_source_pt_count=sum((group.source_count > 1 for group in pt_groups)), umls_lookup_status=umls_result.lookup_status, umls_error_message=umls_result.error_message, candidates=aggregated_candidates, pt_groups=pt_groups)
    return {'aggregation_result': aggregation_result}
