from runtime_resources import resource as asp_resource
from enum import Enum
from typing import Annotated, Any, Literal
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, model_validator

class SchemaModel(BaseModel):
    """
    Base configuration inherited by all project schemas.
    """
    model_config = ConfigDict(extra='forbid', str_strip_whitespace=True, populate_by_name=True)

class MatchMethod(str, Enum):
    EXACT = 'exact'
    FUZZY = 'fuzzy'
    SEMANTIC = 'semantic'
    UMLS = 'umls'
    LLT_RERANKER = 'llt_reranker'
    PT_RERANKER = 'pt_reranker'

class TermInput(SchemaModel):
    """
    One unique term entering the standardization workflow.

    The aliases correspond to the existing DataFrame columns.
    """
    term: str = Field(alias='Term', min_length=1, description='Original reaction term')
    frequency: int = Field(alias='Frequency', ge=1, description='Number of occurrences in the input dataset')

class NormalizationResult(SchemaModel):
    """
    Output of the safe normalization function.
    """
    original_term: str
    normalized_term: str
    applied_operations: list[str] = Field(default_factory=list)

class QueryExpansionType(str, Enum):
    """How a retrieval query variant was produced."""
    ORIGINAL = 'original'
    NORMALIZATION = 'normalization'
    BIOMEDICAL_ABBREVIATION = 'biomedical_abbreviation'
    MEDDRA_ACRONYM = 'meddra_acronym'
    UMLS_SYNONYM = 'umls_synonym'
    COMPONENT = 'component'

class QueryExpansionSource(str, Enum):
    """Terminology source responsible for a query expansion."""
    INPUT = 'input'
    BUILTIN_SAFE_RULE = 'builtin_safe_rule'
    MEDDRA_PT = 'meddra_pt'
    MEDDRA_LLT = 'meddra_llt'
    LOCAL_UMLS = 'local_umls'

class QueryVariant(SchemaModel):
    """Gold-independent, auditable retrieval-query expansion."""
    variant_id: str = Field(min_length=1)
    original_normalized_query: str = Field(min_length=1)
    expanded_query: str = Field(min_length=1)
    expansion_type: QueryExpansionType
    expansion_source: QueryExpansionSource
    confidence: float = Field(ge=0.0, le=1.0)
    ambiguity: bool = False
    context_status: Literal['supported', 'unresolved', 'context_incompatible'] = 'supported'
    context_reason: str | None = None
    target_pt_code: str | None = None
    target_concept_id: str | None = None

    @model_validator(mode='after')
    def validate_query_variant(self):
        if self.expansion_type == QueryExpansionType.ORIGINAL and self.expanded_query != self.original_normalized_query:
            raise ValueError('An original query variant must preserve the normalized query.')
        return self

class MeddraConcept(SchemaModel):
    """
    Common MedDRA representation used throughout the workflow.
    """
    candidate_id: str = Field(min_length=1, description='Internal identifier, normally LLT:<llt_code>')
    llt_code: str = Field(min_length=1)
    llt_name: str = Field(min_length=1)
    pt_code: str = Field(min_length=1)
    pt_name: str = Field(min_length=1)
    is_current: bool = True

class ExactMatchResult(SchemaModel):
    """
    Output produced by the exact-matching component.
    """
    found: bool
    concept: MeddraConcept | None = None
    match_method: Literal['exact'] | None = None
    matched_query_variant: str | None = None
    ambiguous_candidates: list[MeddraConcept] = Field(default_factory=list)
    ambiguity_reason: str | None = None
    exact_match_version: str | None = None
    exact_match_provenance: Literal['literal_exact', 'normalized_exact', 'alias_exact'] | None = None
    normalization_operations: list[str] = Field(default_factory=list)

    @model_validator(mode='after')
    def validate_exact_result(self):
        if self.found:
            if self.concept is None:
                raise ValueError('An exact match must contain a MedDRA concept.')
            if self.match_method != 'exact':
                raise ValueError("An exact match must use match_method='exact'.")
        else:
            if self.concept is not None:
                raise ValueError('An unmatched result cannot contain a concept.')
            if self.match_method is not None:
                raise ValueError('An unmatched result cannot contain a match method.')
        ambiguous_ids = [candidate.candidate_id for candidate in self.ambiguous_candidates]
        if len(ambiguous_ids) != len(set(ambiguous_ids)):
            raise ValueError('ambiguous_candidates contains duplicate candidate IDs.')
        if self.found and self.ambiguous_candidates:
            raise ValueError('A resolved exact match cannot also contain ambiguous candidates.')
        if self.ambiguous_candidates and (not self.ambiguity_reason):
            raise ValueError('Ambiguous exact candidates require ambiguity_reason.')
        if not self.ambiguous_candidates and self.ambiguity_reason is not None:
            raise ValueError('ambiguity_reason requires ambiguous exact candidates.')
        return self

class FuzzyCandidate(MeddraConcept):
    """
    One MedDRA candidate generated through fuzzy matching.
    """
    rank: int = Field(ge=1)
    fuzzy_score: float = Field(ge=0, le=100, description='RapidFuzz WRatio score')
    token_sort_score: float = Field(ge=0, le=100)
    token_set_score: float = Field(ge=0, le=100)
    character_similarity_score: float = Field(ge=0, le=100)
    shared_tokens: list[str] = Field(default_factory=list)
    missing_input_tokens: list[str] = Field(default_factory=list)
    added_candidate_tokens: list[str] = Field(default_factory=list)
    input_token_coverage: float = Field(ge=0, le=1)
    candidate_token_precision: float = Field(ge=0, le=1, description='Proportion of candidate tokens shared with the input term')
    token_overlap_f1: float = Field(ge=0, le=1, description='Harmonic mean of input coverage and candidate precision')
    ranking_score: float = Field(ge=0, le=100, description='General lexical score used only to rank the fuzzy candidates')
    directionality_conflict: bool = False
    negation_conflict: bool = False
    laterality_conflict: bool = False
    temporal_conflict: bool = False
    severity_conflict: bool = False
    anatomical_conflict: bool | None = Field(default=None, description='None means anatomical conflict was not evaluated by the lexical component.')
    unsupported_specificity: bool = False
    conflict_flags: list[str] = Field(default_factory=list)
    direct_pt_match_score: float | None = Field(default=None, ge=0, le=100)
    introduced_by_pt_expansion: bool = False
    introduction_query_variant_ids: list[str] = Field(default_factory=list)

class FuzzyMatchResult(SchemaModel):
    """
    Complete fuzzy candidate-generation result for one term.

    Candidates are possibilities, not accepted mappings.
    """
    normalized_input_term: str = Field(min_length=1)
    retrieval_query_term: str | None = Field(default=None, min_length=1)
    requested_top_k: int = Field(ge=1)
    candidates: list[FuzzyCandidate] = Field(default_factory=list)
    query_variants: list[QueryVariant] = Field(default_factory=list)

class SemanticCandidate(MeddraConcept):
    """
    One MedDRA candidate retrieved through vector similarity.
    """
    rank: int = Field(ge=1)
    cosine_similarity: float = Field(ge=-1, le=1, description='Cosine similarity between the input-term embedding and MedDRA LLT embedding')
    direct_pt_match_score: float | None = Field(default=None, ge=-1, le=1)
    introduced_by_pt_expansion: bool = False
    introduction_query_variant_ids: list[str] = Field(default_factory=list)

class SemanticMatchResult(SchemaModel):
    """
    Complete semantic retrieval result for one input term.

    Candidates are evidence, not accepted mappings.
    """
    normalized_input_term: str = Field(min_length=1)
    retrieval_query_term: str | None = Field(default=None, min_length=1)
    embedding_model: str = Field(min_length=1)
    requested_top_k: int = Field(ge=1)
    candidates: list[SemanticCandidate] = Field(default_factory=list)
    introduction_query_variant_ids: list[str] = Field(default_factory=list)
    query_variants: list[QueryVariant] = Field(default_factory=list)

class UMLSLookupStatus(str, Enum):
    CANDIDATES_FOUND = 'candidates_found'
    NO_SUPPORTED_MATCH = 'no_supported_match'
    LOOKUP_FAILED = 'lookup_failed'

class UMLSMappingMethod(str, Enum):
    DIRECT_MEDDRA_SEARCH = 'direct_meddra_search'
    CUI_ATOM_CROSSWALK = 'cui_atom_crosswalk'

class UMLSCandidate(MeddraConcept):
    """
    One current MedDRA LLT supported by UMLS
    evidence.

    Candidates are not ranked because UMLS is
    being used as a terminology crosswalk rather
    than a similarity search system.
    """
    umls_cui: str | None = None
    umls_term_name: str = Field(min_length=1)
    umls_source_code: str = Field(min_length=1)
    umls_source: str = 'MDR'
    umls_term_type: str | None = None
    mapping_method: UMLSMappingMethod
    introduction_query_variant_ids: list[str] = Field(default_factory=list)

class UMLSMatchResult(SchemaModel):
    """
    Complete UMLS lookup result for one input term.
    """
    normalized_input_term: str = Field(min_length=1)
    retrieval_query_term: str | None = Field(default=None, min_length=1)
    lookup_status: UMLSLookupStatus
    searched_cui: str | None = None
    searched_cui_name: str | None = None
    candidates: list[UMLSCandidate] = Field(default_factory=list)
    error_message: str | None = None
    query_variants: list[QueryVariant] = Field(default_factory=list)

class EvidenceSource(str, Enum):
    FUZZY = 'fuzzy'
    SEMANTIC = 'semantic'
    UMLS = 'umls'

class AggregatedCandidate(MeddraConcept):
    """
    One unique MedDRA LLT and all evidence
    supporting its retrieval.
    """
    evidence_sources: list[EvidenceSource] = Field(min_length=1)
    source_count: int = Field(ge=1, le=3)
    fuzzy_evidence: FuzzyCandidate | None = None
    semantic_evidence: SemanticCandidate | None = None
    umls_evidence: UMLSCandidate | None = None
    fuzzy_query_term: str | None = None
    semantic_query_term: str | None = None
    umls_query_term: str | None = None

    @model_validator(mode='after')
    def validate_evidence_sources(self):
        expected_sources = []
        if self.fuzzy_evidence is not None:
            expected_sources.append(EvidenceSource.FUZZY)
        if self.semantic_evidence is not None:
            expected_sources.append(EvidenceSource.SEMANTIC)
        if self.umls_evidence is not None:
            expected_sources.append(EvidenceSource.UMLS)
        if not expected_sources:
            raise ValueError('An aggregated candidate must contain evidence.')
        if self.evidence_sources != expected_sources:
            raise ValueError('evidence_sources does not match the supplied evidence objects.')
        if self.source_count != len(expected_sources):
            raise ValueError('source_count does not match the number of evidence sources.')
        evidence_query_pairs = [(self.fuzzy_evidence, self.fuzzy_query_term, 'fuzzy'), (self.semantic_evidence, self.semantic_query_term, 'semantic'), (self.umls_evidence, self.umls_query_term, 'umls')]
        for evidence, query_term, source_name in evidence_query_pairs:
            if evidence is None and query_term is not None:
                raise ValueError(f'{source_name}_query_term cannot exist without its evidence.')
            if query_term is not None and (not query_term.strip()):
                raise ValueError(f'{source_name}_query_term cannot be empty.')
        return self

class AggregatedPTGroup(SchemaModel):
    """
    Summary of all aggregated LLTs belonging to
    the same MedDRA PT.
    """
    pt_code: str = Field(min_length=1)
    pt_name: str = Field(min_length=1)
    candidate_ids: list[str] = Field(min_length=1)
    llt_codes: list[str] = Field(min_length=1)
    llt_count: int = Field(ge=1)
    evidence_sources: list[EvidenceSource] = Field(min_length=1)
    source_count: int = Field(ge=1, le=3)
    direct_pt_evidence_sources: list[EvidenceSource] = Field(default_factory=list)
    direct_pt_query_variant_ids: list[str] = Field(default_factory=list)

    @model_validator(mode='after')
    def validate_pt_group(self):
        if len(self.candidate_ids) != self.llt_count:
            raise ValueError('candidate_ids does not match llt_count.')
        if len(self.llt_codes) != self.llt_count:
            raise ValueError('llt_codes does not match llt_count.')
        if len(set(self.candidate_ids)) != len(self.candidate_ids):
            raise ValueError('A PT group contains duplicate candidate IDs.')
        if len(set(self.llt_codes)) != len(self.llt_codes):
            raise ValueError('A PT group contains duplicate LLT codes.')
        if len(set(self.evidence_sources)) != len(self.evidence_sources):
            raise ValueError('A PT group contains duplicate evidence sources.')
        if self.source_count != len(self.evidence_sources):
            raise ValueError('source_count does not match the PT evidence sources.')
        if not set(self.direct_pt_evidence_sources) <= set(self.evidence_sources):
            raise ValueError('Direct PT evidence sources must also support an LLT in the PT group.')
        if len(self.direct_pt_query_variant_ids) != len(set(self.direct_pt_query_variant_ids)):
            raise ValueError('Direct PT query variant IDs contain duplicates.')
        return self

class AggregationResult(SchemaModel):
    """
    Combined LLT candidates and PT-level groups.
    """
    normalized_input_term: str = Field(min_length=1)
    fuzzy_candidate_count: int = Field(ge=0)
    semantic_candidate_count: int = Field(ge=0)
    umls_candidate_count: int = Field(ge=0)
    total_source_candidates: int = Field(ge=0)
    unique_llt_count: int = Field(ge=0)
    multiple_source_llt_count: int = Field(ge=0)
    unique_pt_count: int = Field(ge=0)
    multiple_source_pt_count: int = Field(ge=0)
    umls_lookup_status: UMLSLookupStatus
    umls_error_message: str | None = None
    candidates: list[AggregatedCandidate] = Field(default_factory=list)
    pt_groups: list[AggregatedPTGroup] = Field(default_factory=list)

    @model_validator(mode='after')
    def validate_aggregation_counts(self):
        expected_total = self.fuzzy_candidate_count + self.semantic_candidate_count + self.umls_candidate_count
        if self.total_source_candidates != expected_total:
            raise ValueError('total_source_candidates does not match source counts.')
        if self.unique_llt_count != len(self.candidates):
            raise ValueError('unique_llt_count does not match the candidate list.')
        expected_multiple_llts = sum((candidate.source_count > 1 for candidate in self.candidates))
        if self.multiple_source_llt_count != expected_multiple_llts:
            raise ValueError('multiple_source_llt_count is incorrect.')
        if self.unique_pt_count != len(self.pt_groups):
            raise ValueError('unique_pt_count does not match the PT groups.')
        expected_multiple_pts = sum((group.source_count > 1 for group in self.pt_groups))
        if self.multiple_source_pt_count != expected_multiple_pts:
            raise ValueError('multiple_source_pt_count is incorrect.')
        candidate_by_id = {candidate.candidate_id: candidate for candidate in self.candidates}
        grouped_candidate_ids = [candidate_id for group in self.pt_groups for candidate_id in group.candidate_ids]
        if len(grouped_candidate_ids) != len(set(grouped_candidate_ids)):
            raise ValueError('An LLT candidate appears in more than one PT group.')
        if set(grouped_candidate_ids) != set(candidate_by_id):
            raise ValueError('PT groups do not contain every aggregated LLT candidate.')
        for group in self.pt_groups:
            for candidate_id in group.candidate_ids:
                candidate = candidate_by_id[candidate_id]
                if candidate.pt_code != group.pt_code or candidate.pt_name != group.pt_name:
                    raise ValueError('An LLT was assigned to the wrong PT group.')
        return self

class ReviewerPTCandidateGroup(SchemaModel):
    """
    Reviewer-facing representation of one MedDRA PT
    and all retrieved LLT candidates belonging to it.

    The PT is reviewed first. If the PT is supported,
    one of its LLT candidates may then be selected.
    """
    pt_code: str = Field(min_length=1)
    pt_name: str = Field(min_length=1)
    candidate_ids: list[str] = Field(min_length=1)
    llt_count: int = Field(ge=1)
    evidence_sources: list[EvidenceSource] = Field(min_length=1)
    source_count: int = Field(ge=1, le=3)
    direct_pt_evidence_sources: list[EvidenceSource] = Field(default_factory=list)
    direct_pt_query_variant_ids: list[str] = Field(default_factory=list)
    llt_candidates: list[AggregatedCandidate] = Field(min_length=1)

    @model_validator(mode='after')
    def validate_reviewer_pt_group(self):
        nested_candidate_ids = [candidate.candidate_id for candidate in self.llt_candidates]
        if self.candidate_ids != nested_candidate_ids:
            raise ValueError('candidate_ids must match the ordered LLT candidates contained in the reviewer PT group.')
        if len(self.candidate_ids) != len(set(self.candidate_ids)):
            raise ValueError('A reviewer PT group contains duplicate candidate IDs.')
        if self.llt_count != len(self.llt_candidates):
            raise ValueError('llt_count does not match the number of nested LLT candidates.')
        invalid_pt_candidates = [candidate.candidate_id for candidate in self.llt_candidates if candidate.pt_code != self.pt_code or candidate.pt_name != self.pt_name]
        if invalid_pt_candidates:
            raise ValueError(f'A reviewer PT group contains LLTs belonging to another PT: {invalid_pt_candidates}')
        source_order = (EvidenceSource.FUZZY, EvidenceSource.SEMANTIC, EvidenceSource.UMLS)
        expected_sources = [source for source in source_order if any((source in candidate.evidence_sources for candidate in self.llt_candidates))]
        if self.evidence_sources != expected_sources:
            raise ValueError('Reviewer PT evidence_sources do not match the nested LLT evidence.')
        if self.source_count != len(expected_sources):
            raise ValueError('Reviewer PT source_count does not match evidence_sources.')
        return self

class InputStructureType(str, Enum):
    """
    Clinical structure of one normalized input term.
    """
    SINGLE_CONCEPT = 'single_concept'
    MULTIPLE_CONCEPTS = 'multiple_concepts'
    NEGATED_NON_EVENT = 'negated_non_event'
    AMBIGUOUS_CONCEPT = 'ambiguous_concept'

class InputComponent(SchemaModel):
    """
    One explicit clinical component found in the
    normalized input.

    component_text must be copied from the normalized
    input rather than invented or paraphrased.
    """
    component_id: str = Field(min_length=1, pattern='^COMPONENT_[1-9][0-9]*$')
    component_text: str = Field(min_length=1)
    is_negated: bool = False

class InputInterpretation(SchemaModel):
    """
    Structured interpretation produced before the
    candidate reviewer runs.
    """
    term_id: str = Field(min_length=1)
    normalized_term: str = Field(min_length=1)
    structure_type: InputStructureType
    asserted_components: list[InputComponent] = Field(min_length=1)
    interpretation_rationale: str = Field(min_length=1)
    deterministic_repair_reasons: list[str] = Field(default_factory=list)
    interpreted_features: dict[str, Any] = Field(default_factory=dict)
    decomposition_diagnostics: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode='before')
    @classmethod
    def repair_generic_context(cls, data):
        from biomedical_wording import repair_generic_context_payload
        from clinical_features import interpreted_features, modifier_negations, event_is_negated
        if not isinstance(data, dict):
            return data
        data = repair_generic_context_payload(data)
        text = str(data.get('normalized_term', ''))
        data = {**data, 'interpreted_features': dict(interpreted_features(text))}
        components = data.get('asserted_components', [])
        components = [c.model_dump() if isinstance(c, SchemaModel) else c for c in components]
        if len(components) == 1 and isinstance(components[0], dict):
            source_features = interpreted_features(text)
            component_features = interpreted_features(str(components[0].get('component_text', '')))
            lost_modifier = source_features['negated_modifiers'] and source_features['negated_modifiers'] != component_features['negated_modifiers']
            lost_threshold = source_features['bounds'] and any((source_features[k] != component_features[k] for k in ('bounds', 'units')))
            if lost_modifier or lost_threshold:
                components = [{**components[0], 'component_text': text}]
                data = {**data, 'asserted_components': components, 'deterministic_repair_reasons': list(data.get('deterministic_repair_reasons', [])) + ['restored_protected_modifier_or_numerical_information_to_component']}
        if len(components) == 1 and modifier_negations(text) and (not event_is_negated(text)) and components[0].get('is_negated'):
            data = {**data, 'structure_type': 'single_concept', 'asserted_components': [{**components[0], 'is_negated': False}], 'deterministic_repair_reasons': list(data.get('deterministic_repair_reasons', [])) + ['negated_modifier_does_not_negate_clinical_event']}
        return data

    @model_validator(mode='after')
    def validate_input_interpretation(self):
        component_ids = [component.component_id for component in self.asserted_components]
        if len(component_ids) != len(set(component_ids)):
            raise ValueError('Input component IDs must be unique.')
        normalized_component_texts = [' '.join(component.component_text.casefold().split()) for component in self.asserted_components]
        if len(normalized_component_texts) != len(set(normalized_component_texts)):
            raise ValueError('Input components must have different component_text values.')
        normalized_input = ' '.join(self.normalized_term.casefold().split())
        for component_text in normalized_component_texts:
            if component_text not in normalized_input:
                raise ValueError(f"Every component_text must be copied from the normalized input. Invalid component: '{component_text}'.")
        component_count = len(self.asserted_components)
        if self.structure_type == InputStructureType.SINGLE_CONCEPT:
            if component_count != 1:
                raise ValueError('SINGLE_CONCEPT requires exactly one input component.')
            if self.asserted_components[0].is_negated:
                raise ValueError('A negated single component must use NEGATED_NON_EVENT.')
        elif self.structure_type == InputStructureType.MULTIPLE_CONCEPTS:
            if component_count < 2:
                raise ValueError('MULTIPLE_CONCEPTS requires at least two explicit components.')
        elif self.structure_type == InputStructureType.NEGATED_NON_EVENT:
            if not all((component.is_negated for component in self.asserted_components)):
                raise ValueError('Every component in a NEGATED_NON_EVENT must have is_negated=true.')
        elif self.structure_type == InputStructureType.AMBIGUOUS_CONCEPT:
            if component_count != 1:
                raise ValueError('AMBIGUOUS_CONCEPT represents one expression and requires exactly one component.')
            if self.asserted_components[0].is_negated:
                raise ValueError('An ambiguous component cannot also be marked as a negated non-event.')
        from compound_validation import assess_decomposition
        self.decomposition_diagnostics = assess_decomposition(self.normalized_term, self.asserted_components)
        return self

class ReviewerDecisionType(str, Enum):
    MATCHED_SINGLE = 'matched_single'
    MATCHED_MULTIPLE = 'matched_multiple'
    MATCHED_PT_LLT_HUMAN_REVIEW = 'matched_pt_llt_human_review'
    NO_SUPPORTED_MATCH = 'no_supported_match'
    REQUEST_MORE_EVIDENCE = 'request_more_evidence'
    HUMAN_REVIEW = 'human_review'

class CandidateReviewVerdict(str, Enum):
    SUPPORTED = 'supported'
    NOT_SUPPORTED = 'not_supported'
    UNCERTAIN = 'uncertain'

class EvidenceRequestAction(str, Enum):
    """
    EXPAND_CANDIDATES asks fuzzy or semantic
    retrieval to return more candidates for the
    existing normalized term.

    FOCUSED_SEARCH searches one explicitly stated
    subterm, usually for compound input.
    """
    EXPAND_CANDIDATES = 'expand_candidates'
    FOCUSED_SEARCH = 'focused_search'

class CandidateAssessment(SchemaModel):
    """
    The reviewer assessment of one aggregated
    LLT candidate.
    """
    assessment_authority: str = 'llm'
    candidate_id: str = Field(min_length=1)
    verdict: CandidateReviewVerdict
    rationale: str = Field(min_length=1)

class ComponentCandidateMatch(SchemaModel):
    """
    Connect one positively asserted input component
    to a selected MedDRA PT and one selected LLT
    candidate belonging to that PT.
    """
    component_id: str = Field(min_length=1)
    pt_code: str = Field(min_length=1)
    candidate_id: str = Field(min_length=1)
    rationale: str = Field(min_length=1)

class EvidenceRequest(SchemaModel):
    """
    A structured request sent to the deterministic
    coordinator, not directly executed by the LLM.
    """
    request_id: str = Field(min_length=1)
    source: EvidenceSource
    action: EvidenceRequestAction
    query_term: str | None = None
    requested_top_k: int | None = Field(default=None, ge=1, le=20)
    candidate_ids_of_interest: list[str] = Field(default_factory=list)
    reason: str = Field(min_length=1)

    @model_validator(mode='after')
    def validate_evidence_request(self):
        if len(self.candidate_ids_of_interest) != len(set(self.candidate_ids_of_interest)):
            raise ValueError('candidate_ids_of_interest contains duplicates.')
        if self.action == EvidenceRequestAction.EXPAND_CANDIDATES:
            if self.source not in {EvidenceSource.FUZZY, EvidenceSource.SEMANTIC}:
                raise ValueError('Only fuzzy or semantic retrieval can expand its candidate count.')
            if self.requested_top_k is None:
                raise ValueError('An expand-candidates request requires requested_top_k.')
            if self.query_term is not None:
                raise ValueError('An expand-candidates request uses the existing normalized term and must not provide query_term.')
        if self.action == EvidenceRequestAction.FOCUSED_SEARCH:
            if self.query_term is None or not self.query_term.strip():
                raise ValueError('A focused search requires query_term.')
            if self.source in {EvidenceSource.FUZZY, EvidenceSource.SEMANTIC} and self.requested_top_k is None:
                raise ValueError('A focused fuzzy or semantic search requires requested_top_k.')
            if self.source == EvidenceSource.UMLS and self.requested_top_k is not None:
                raise ValueError('UMLS focused lookup does not use requested_top_k.')
        return self

class RetrievalProvenance(SchemaModel):
    """
    Records which query produced candidates added
    during a reviewer feedback round.

    Candidate scores and ranks remain associated
    with this query rather than being interpreted
    as scores against the complete input term.
    """
    request_id: str = Field(min_length=1)
    review_round: int = Field(ge=1)
    source: EvidenceSource
    action: EvidenceRequestAction
    query_term: str = Field(min_length=1)
    requested_top_k: int | None = Field(default=None, ge=1, le=20)
    candidate_ids: list[str] = Field(default_factory=list)

    @model_validator(mode='after')
    def validate_provenance(self):
        if len(self.candidate_ids) != len(set(self.candidate_ids)):
            raise ValueError('Retrieval provenance contains duplicate candidate IDs.')
        if self.source in {EvidenceSource.FUZZY, EvidenceSource.SEMANTIC} and self.requested_top_k is None:
            raise ValueError('Fuzzy or semantic provenance requires requested_top_k.')
        if self.source == EvidenceSource.UMLS and self.requested_top_k is not None:
            raise ValueError('UMLS provenance must not contain requested_top_k.')
        return self

class ReviewerInput(SchemaModel):
    """
    Complete evidence package supplied to the
    reviewer for one exact-unmatched input term.
    """
    term_id: str = Field(min_length=1)
    original_term: str = Field(min_length=1)
    frequency: int = Field(ge=1)
    normalized_term: str = Field(min_length=1)
    input_interpretation: InputInterpretation
    review_round: int = Field(default=1, ge=1)
    aggregation_result: AggregationResult
    pt_candidate_groups: list[ReviewerPTCandidateGroup] = Field(default_factory=list)
    retrieval_provenance: list[RetrievalProvenance] = Field(default_factory=list)

    @model_validator(mode='after')
    def validate_reviewer_input(self):
        if self.normalized_term != self.aggregation_result.normalized_input_term:
            raise ValueError('Reviewer input and aggregation use different normalized terms.')
        candidate_by_id = {candidate.candidate_id: candidate for candidate in self.aggregation_result.candidates}
        expected_pt_candidate_groups = []
        for pt_group in self.aggregation_result.pt_groups:
            llt_candidates = [candidate_by_id[candidate_id] for candidate_id in pt_group.candidate_ids]
            expected_pt_candidate_groups.append(ReviewerPTCandidateGroup(pt_code=pt_group.pt_code, pt_name=pt_group.pt_name, candidate_ids=list(pt_group.candidate_ids), llt_count=pt_group.llt_count, evidence_sources=list(pt_group.evidence_sources), source_count=pt_group.source_count, direct_pt_evidence_sources=list(pt_group.direct_pt_evidence_sources), direct_pt_query_variant_ids=list(pt_group.direct_pt_query_variant_ids), llt_candidates=llt_candidates))
        if not self.pt_candidate_groups:
            self.pt_candidate_groups = expected_pt_candidate_groups
        elif self.pt_candidate_groups != expected_pt_candidate_groups:
            raise ValueError('pt_candidate_groups does not match the PT and LLT structure contained in aggregation_result.')
        provenance_request_keys = [(provenance.review_round, provenance.request_id) for provenance in self.retrieval_provenance]
        if len(provenance_request_keys) != len(set(provenance_request_keys)):
            raise ValueError('ReviewerInput contains duplicate provenance request IDs.')
        available_candidate_ids = {candidate.candidate_id for candidate in self.aggregation_result.candidates}
        unknown_provenance_ids = {candidate_id for provenance in self.retrieval_provenance for candidate_id in provenance.candidate_ids if candidate_id not in available_candidate_ids}
        if unknown_provenance_ids:
            raise ValueError(f'Retrieval provenance references candidates not present in the aggregation result: {sorted(unknown_provenance_ids)}')
        return self

    @model_validator(mode='after')
    def validate_input_interpretation(self) -> 'ReviewerInput':
        interpretation = self.input_interpretation
        if interpretation.term_id != self.term_id:
            raise ValueError('InputInterpretation term_id must match ReviewerInput term_id.')
        if interpretation.normalized_term != self.normalized_term:
            raise ValueError('InputInterpretation normalized_term must match ReviewerInput normalized_term.')
        return self

class ReviewerDecision(SchemaModel):
    """
    Structured response generated by the reviewer.

    selected_candidate_ids identifies the selected
    LLT candidates. Each matched component records
    its selected parent PT in component_matches.
    The deterministic validator confirms that every
    selected LLT belongs to the stated PT.
    """
    unresolved_without_candidates: bool = False
    resolved_pt_code: str | None = None
    resolved_pt_name: str | None = None
    term_id: str = Field(min_length=1)
    review_round: int = Field(ge=1)
    decision_type: ReviewerDecisionType
    selected_candidate_ids: list[str] = Field(default_factory=list)
    component_matches: list[ComponentCandidateMatch] = Field(default_factory=list)
    candidate_assessments: list[CandidateAssessment] = Field(default_factory=list)
    evidence_requests: list[EvidenceRequest] = Field(default_factory=list)
    human_review_candidate_ids: list[str] = Field(default_factory=list)
    decision_rationale: str = Field(min_length=1)
    multiple_match_explanation: str | None = None
    human_review_reason: str | None = None

    @model_validator(mode='after')
    def validate_reviewer_decision(self):
        if (self.resolved_pt_code is None) != (self.resolved_pt_name is None):
            raise ValueError('Explicit resolved PT code and name must occur together.')
        if self.resolved_pt_code is not None and (not self.resolved_pt_code.strip() or not self.resolved_pt_name.strip() or self.decision_type != ReviewerDecisionType.MATCHED_PT_LLT_HUMAN_REVIEW):
            raise ValueError('Explicit PT identity is valid only for PT-resolved LLT deferral.')
        if self.unresolved_without_candidates and (self.decision_type != ReviewerDecisionType.HUMAN_REVIEW or self.human_review_candidate_ids):
            raise ValueError('An explicit empty-shortlist marker is valid only for unresolved HUMAN_REVIEW.')
        selected_ids = self.selected_candidate_ids
        assessment_ids = [assessment.candidate_id for assessment in self.candidate_assessments]
        human_review_ids = self.human_review_candidate_ids
        request_ids = [request.request_id for request in self.evidence_requests]
        if len(selected_ids) != len(set(selected_ids)):
            raise ValueError('selected_candidate_ids contains duplicates.')
        if len(assessment_ids) != len(set(assessment_ids)):
            raise ValueError('A candidate was assessed more than once.')
        if len(human_review_ids) != len(set(human_review_ids)):
            raise ValueError('human_review_candidate_ids contains duplicates.')
        if len(request_ids) != len(set(request_ids)):
            raise ValueError('Evidence request IDs must be unique.')
        assessment_by_id = {assessment.candidate_id: assessment for assessment in self.candidate_assessments}
        unassessed_selected_ids = [candidate_id for candidate_id in selected_ids if candidate_id not in assessment_by_id]
        if unassessed_selected_ids:
            raise ValueError(f'Selected candidates must have candidate assessments: {unassessed_selected_ids}')
        unassessed_human_ids = [candidate_id for candidate_id in human_review_ids if candidate_id not in assessment_by_id]
        if unassessed_human_ids:
            raise ValueError(f'Human-review candidates must have candidate assessments: {unassessed_human_ids}')
        for candidate_id in selected_ids:
            if assessment_by_id[candidate_id].verdict != CandidateReviewVerdict.SUPPORTED:
                raise ValueError("A selected candidate must have verdict='supported'.")
        if self.decision_type == ReviewerDecisionType.MATCHED_SINGLE:
            if len(selected_ids) != 1:
                raise ValueError('MATCHED_SINGLE requires exactly one selected candidate.')
            if self.evidence_requests:
                raise ValueError('A final matched decision cannot request more evidence.')
            if human_review_ids:
                raise ValueError('A final matched decision cannot contain human-review candidates.')
            if self.multiple_match_explanation is not None:
                raise ValueError('MATCHED_SINGLE must not have multiple_match_explanation.')
            if self.human_review_reason is not None:
                raise ValueError('MATCHED_SINGLE must not have human_review_reason.')
        elif self.decision_type == ReviewerDecisionType.MATCHED_MULTIPLE:
            if len(selected_ids) < 2:
                raise ValueError('MATCHED_MULTIPLE requires at least two selected candidates.')
            if self.multiple_match_explanation is None or not self.multiple_match_explanation.strip():
                raise ValueError('MATCHED_MULTIPLE requires an explanation of why the input contains multiple concepts.')
            if self.evidence_requests:
                raise ValueError('A final matched decision cannot request more evidence.')
            if human_review_ids:
                raise ValueError('A final matched decision cannot contain human-review candidates.')
            if self.human_review_reason is not None:
                raise ValueError('MATCHED_MULTIPLE must not have human_review_reason.')
        elif self.decision_type == ReviewerDecisionType.NO_SUPPORTED_MATCH:
            if selected_ids:
                raise ValueError('NO_SUPPORTED_MATCH cannot select a candidate.')
            if self.evidence_requests:
                raise ValueError('Use REQUEST_MORE_EVIDENCE when more retrieval is needed.')
            if human_review_ids:
                raise ValueError('Use HUMAN_REVIEW when a person must inspect candidate options.')
            invalid_verdicts = [assessment.candidate_id for assessment in self.candidate_assessments if assessment.verdict != CandidateReviewVerdict.NOT_SUPPORTED]
            if invalid_verdicts:
                raise ValueError(f'NO_SUPPORTED_MATCH requires all assessed candidates to be not_supported: {invalid_verdicts}')
            if self.multiple_match_explanation is not None or self.human_review_reason is not None:
                raise ValueError('NO_SUPPORTED_MATCH contains fields for another decision type.')
        elif self.decision_type == ReviewerDecisionType.REQUEST_MORE_EVIDENCE:
            if selected_ids:
                raise ValueError('REQUEST_MORE_EVIDENCE is not a final selection.')
            if not self.evidence_requests:
                raise ValueError('REQUEST_MORE_EVIDENCE requires at least one structured request.')
            if human_review_ids:
                raise ValueError('A request-more-evidence decision cannot also request human review.')
            if self.multiple_match_explanation is not None or self.human_review_reason is not None:
                raise ValueError('REQUEST_MORE_EVIDENCE contains fields for another decision type.')
        elif self.decision_type == ReviewerDecisionType.MATCHED_PT_LLT_HUMAN_REVIEW:
            if selected_ids:
                raise ValueError('MATCHED_PT_LLT_HUMAN_REVIEW cannot contain a final LLT selection.')
            if not human_review_ids and self.resolved_pt_code is None:
                raise ValueError('MATCHED_PT_LLT_HUMAN_REVIEW requires explicit PT identity when no LLT is established.')
            if self.component_matches:
                raise ValueError('MATCHED_PT_LLT_HUMAN_REVIEW cannot contain component_matches because no LLT is final.')
            if self.evidence_requests:
                raise ValueError('MATCHED_PT_LLT_HUMAN_REVIEW cannot request more evidence.')
            if self.human_review_reason is None or not self.human_review_reason.strip():
                raise ValueError('MATCHED_PT_LLT_HUMAN_REVIEW requires a human-review reason.')
            invalid_human_review_verdicts = [candidate_id for candidate_id in human_review_ids if assessment_by_id[candidate_id].verdict != CandidateReviewVerdict.UNCERTAIN]
            if invalid_human_review_verdicts:
                raise ValueError(f"LLTs deferred after PT resolution must have verdict='uncertain': {invalid_human_review_verdicts}")
            if self.multiple_match_explanation is not None:
                raise ValueError('MATCHED_PT_LLT_HUMAN_REVIEW must not have multiple_match_explanation.')
        elif self.decision_type == ReviewerDecisionType.HUMAN_REVIEW:
            if not human_review_ids and (not self.unresolved_without_candidates):
                raise ValueError('HUMAN_REVIEW requires at least one candidate in human_review_candidate_ids.')
            if selected_ids:
                raise ValueError('HUMAN_REVIEW cannot contain a final selected candidate.')
            if self.evidence_requests:
                raise ValueError('Request additional evidence before escalating to a person.')
            if self.human_review_reason is None or not self.human_review_reason.strip():
                raise ValueError('HUMAN_REVIEW requires a reason.')
            invalid_human_review_verdicts = [candidate_id for candidate_id in human_review_ids if assessment_by_id[candidate_id].verdict != CandidateReviewVerdict.UNCERTAIN]
            if invalid_human_review_verdicts:
                raise ValueError(f"Candidates sent to human review must have verdict='uncertain': {invalid_human_review_verdicts}")
            if self.multiple_match_explanation is not None:
                raise ValueError('HUMAN_REVIEW must not have multiple_match_explanation.')
        return self

class MatchedSingleReviewerDecision(ReviewerDecision):
    """Final decision containing exactly one supported candidate."""
    decision_type: Literal[ReviewerDecisionType.MATCHED_SINGLE]
    selected_candidate_ids: list[str] = Field(min_length=1, max_length=1)
    candidate_assessments: list[CandidateAssessment] = Field(min_length=1)
    evidence_requests: list[EvidenceRequest] = Field(default_factory=list, max_length=0)
    human_review_candidate_ids: list[str] = Field(default_factory=list, max_length=0)
    multiple_match_explanation: None = None
    human_review_reason: None = None

class MatchedMultipleReviewerDecision(ReviewerDecision):
    """Final decision containing at least two distinct concepts."""
    decision_type: Literal[ReviewerDecisionType.MATCHED_MULTIPLE]
    selected_candidate_ids: list[str] = Field(min_length=2)
    candidate_assessments: list[CandidateAssessment] = Field(min_length=2)
    evidence_requests: list[EvidenceRequest] = Field(default_factory=list, max_length=0)
    human_review_candidate_ids: list[str] = Field(default_factory=list, max_length=0)
    multiple_match_explanation: str = Field(min_length=1)
    human_review_reason: None = None

class NoSupportedMatchReviewerDecision(ReviewerDecision):
    """Final decision in which every supplied candidate is rejected."""
    decision_type: Literal[ReviewerDecisionType.NO_SUPPORTED_MATCH]
    selected_candidate_ids: list[str] = Field(default_factory=list, max_length=0)
    candidate_assessments: list[CandidateAssessment] = Field()
    evidence_requests: list[EvidenceRequest] = Field(default_factory=list, max_length=0)
    human_review_candidate_ids: list[str] = Field(default_factory=list, max_length=0)
    multiple_match_explanation: None = None
    human_review_reason: None = None

class RequestMoreEvidenceReviewerDecision(ReviewerDecision):
    """Non-final decision containing at least one evidence request."""
    decision_type: Literal[ReviewerDecisionType.REQUEST_MORE_EVIDENCE]
    selected_candidate_ids: list[str] = Field(default_factory=list, max_length=0)
    evidence_requests: list[EvidenceRequest] = Field(min_length=1)
    human_review_candidate_ids: list[str] = Field(default_factory=list, max_length=0)
    multiple_match_explanation: None = None
    human_review_reason: None = None

class HumanReviewReviewerDecision(ReviewerDecision):
    """
    Decision escalating unresolved ambiguity
    to a human reviewer.
    """
    decision_type: Literal[ReviewerDecisionType.HUMAN_REVIEW]
    selected_candidate_ids: list[str] = Field(default_factory=list, max_length=0)
    candidate_assessments: list[CandidateAssessment] = Field(min_length=1)
    evidence_requests: list[EvidenceRequest] = Field(default_factory=list, max_length=0)
    human_review_candidate_ids: list[str] = Field(min_length=1)
    multiple_match_explanation: None = None
    human_review_reason: str = Field(min_length=1)
ReviewerStructuredDecision = Annotated[MatchedSingleReviewerDecision | MatchedMultipleReviewerDecision | NoSupportedMatchReviewerDecision | RequestMoreEvidenceReviewerDecision | HumanReviewReviewerDecision, Field(discriminator='decision_type')]
REVIEWER_DECISION_ADAPTER = TypeAdapter(ReviewerStructuredDecision)

class CoordinatorRoute(str, Enum):
    """
    The next workflow destination selected by the
    deterministic coordinator.
    """
    FINALIZE = 'finalize'
    EXECUTE_EVIDENCE_REQUESTS = 'execute_evidence_requests'
    HUMAN_REVIEW = 'human_review'

class EvidenceRequestHistoryEntry(SchemaModel):
    """
    One evidence request that has already been
    executed.

    Only completed requests should be added to this
    history. The coordinator uses the history to
    prevent identical retrieval loops.
    """
    term_id: str = Field(min_length=1)
    review_round: int = Field(ge=1)
    request: EvidenceRequest

class CoordinatorResult(SchemaModel):
    """
    Deterministic routing result produced from a
    validated reviewer decision.
    """
    term_id: str = Field(min_length=1)
    review_round: int = Field(ge=1)
    source_decision_type: ReviewerDecisionType
    route: CoordinatorRoute
    selected_candidate_ids: list[str] = Field(default_factory=list)
    pending_evidence_requests: list[EvidenceRequest] = Field(default_factory=list)
    next_review_round: int | None = Field(default=None, ge=2)
    human_review_candidate_ids: list[str] = Field(default_factory=list)
    human_review_reason: str | None = None

    @model_validator(mode='after')
    def validate_coordinator_result(self):
        if len(self.selected_candidate_ids) != len(set(self.selected_candidate_ids)):
            raise ValueError('selected_candidate_ids contains duplicates.')
        if len(self.human_review_candidate_ids) != len(set(self.human_review_candidate_ids)):
            raise ValueError('human_review_candidate_ids contains duplicates.')
        final_decision_types = {ReviewerDecisionType.MATCHED_SINGLE, ReviewerDecisionType.MATCHED_MULTIPLE, ReviewerDecisionType.NO_SUPPORTED_MATCH}
        if self.route == CoordinatorRoute.FINALIZE:
            if self.source_decision_type not in final_decision_types:
                raise ValueError('Only a final reviewer decision can use the finalize route.')
            if self.pending_evidence_requests:
                raise ValueError('A finalized result cannot contain pending evidence requests.')
            if self.next_review_round is not None:
                raise ValueError('A finalized result cannot contain next_review_round.')
            if self.human_review_candidate_ids:
                raise ValueError('A finalized result cannot contain human-review candidates.')
            if self.human_review_reason is not None:
                raise ValueError('A finalized result cannot contain human_review_reason.')
        elif self.route == CoordinatorRoute.EXECUTE_EVIDENCE_REQUESTS:
            if self.source_decision_type != ReviewerDecisionType.REQUEST_MORE_EVIDENCE:
                raise ValueError('Only request_more_evidence can route to evidence execution.')
            if not self.pending_evidence_requests:
                raise ValueError('Evidence execution requires at least one pending request.')
            if self.next_review_round != self.review_round + 1:
                raise ValueError('next_review_round must equal review_round + 1.')
            if self.selected_candidate_ids:
                raise ValueError('Evidence execution cannot contain final selections.')
            if self.human_review_candidate_ids:
                raise ValueError('Evidence execution cannot also route to human review.')
            if self.human_review_reason is not None:
                raise ValueError('Evidence execution cannot contain human_review_reason.')
        elif self.route == CoordinatorRoute.HUMAN_REVIEW:
            if self.pending_evidence_requests:
                raise ValueError('Human review cannot contain pending evidence requests.')
            if self.next_review_round is not None:
                raise ValueError('Human review cannot contain next_review_round.')
            if self.selected_candidate_ids:
                raise ValueError('Human review cannot contain final selections.')
            if self.human_review_reason is None or not self.human_review_reason.strip():
                raise ValueError('Human review requires a reason.')
        return self

class EvidenceExecutionStatus(str, Enum):
    """
    Whether one retrieval operation completed
    normally or encountered a technical failure.

    A completed search may legitimately return
    zero candidates.
    """
    COMPLETED = 'completed'
    FAILED = 'failed'

class EvidenceExecutionBatchStatus(str, Enum):
    """
    Overall status of all evidence requests from
    one reviewer decision.
    """
    COMPLETED = 'completed'
    PARTIAL = 'partial'
    FAILED = 'failed'

class EvidenceRequestExecutionResult(SchemaModel):
    """
    Auditable result of executing one structured
    reviewer evidence request.

    Exactly one source-specific result is normally
    present. A technical failure may contain no
    source result.
    """
    term_id: str = Field(min_length=1)
    review_round: int = Field(ge=1)
    request: EvidenceRequest
    effective_query_term: str = Field(min_length=1)
    status: EvidenceExecutionStatus
    fuzzy_result: FuzzyMatchResult | None = None
    semantic_result: SemanticMatchResult | None = None
    umls_result: UMLSMatchResult | None = None
    candidate_count: int = Field(ge=0)
    error_message: str | None = None

    @model_validator(mode='after')
    def validate_execution_result(self):
        source_results = {EvidenceSource.FUZZY: self.fuzzy_result, EvidenceSource.SEMANTIC: self.semantic_result, EvidenceSource.UMLS: self.umls_result}
        populated_sources = {source for source, result in source_results.items() if result is not None}
        expected_source = self.request.source
        unexpected_sources = populated_sources - {expected_source}
        if unexpected_sources:
            raise ValueError(f'Execution contains results from sources not requested by the reviewer: {sorted([source.value for source in unexpected_sources])}')
        expected_result = source_results[expected_source]
        if self.status == EvidenceExecutionStatus.COMPLETED:
            if expected_result is None:
                raise ValueError('A completed evidence request must contain its source-specific result.')
            if len(populated_sources) != 1:
                raise ValueError('A completed evidence request must contain exactly one source-specific result.')
            if self.error_message is not None:
                raise ValueError('A completed evidence request cannot contain error_message.')
            expected_candidate_count = len(expected_result.candidates)
            if self.candidate_count != expected_candidate_count:
                raise ValueError('candidate_count does not match the source-specific result.')
            normalized_effective_query = ' '.join(self.effective_query_term.casefold().split())
            normalized_result_query = ' '.join(expected_result.normalized_input_term.casefold().split())
            if normalized_effective_query != normalized_result_query:
                raise ValueError('effective_query_term does not match the term searched by the retrieval result.')
            if self.request.action == EvidenceRequestAction.FOCUSED_SEARCH:
                normalized_requested_query = ' '.join(self.request.query_term.casefold().split())
                if normalized_effective_query != normalized_requested_query:
                    raise ValueError("A focused-search execution must use the request's query_term.")
            if expected_source in {EvidenceSource.FUZZY, EvidenceSource.SEMANTIC}:
                if expected_result.requested_top_k != self.request.requested_top_k:
                    raise ValueError('The retrieval result used a different top_k than the reviewer requested.')
            if expected_source == EvidenceSource.UMLS and self.umls_result is not None and (self.umls_result.lookup_status == UMLSLookupStatus.LOOKUP_FAILED):
                raise ValueError('A failed UMLS lookup cannot be recorded as completed.')
        elif self.status == EvidenceExecutionStatus.FAILED:
            if self.candidate_count != 0:
                raise ValueError('A failed evidence request must have candidate_count=0.')
            if self.error_message is None or not self.error_message.strip():
                raise ValueError('A failed evidence request requires error_message.')
            if expected_result is not None and expected_source != EvidenceSource.UMLS:
                raise ValueError('A failed fuzzy or semantic request must not contain a retrieval result.')
            if self.umls_result is not None and self.umls_result.lookup_status != UMLSLookupStatus.LOOKUP_FAILED:
                raise ValueError('A failed UMLS execution may only retain a lookup_failed UMLS result.')
        return self

class EvidenceExecutionBatch(SchemaModel):
    """
    Results of all evidence requests generated by
    one reviewer decision.

    Reaggregation should proceed only when status
    is COMPLETED. PARTIAL and FAILED batches must
    not silently produce a new review round.
    """
    term_id: str = Field(min_length=1)
    review_round: int = Field(ge=1)
    requested_count: int = Field(ge=1)
    completed_count: int = Field(ge=0)
    failed_count: int = Field(ge=0)
    status: EvidenceExecutionBatchStatus
    executions: list[EvidenceRequestExecutionResult] = Field(min_length=1)

    @model_validator(mode='after')
    def validate_execution_batch(self):
        if self.requested_count != len(self.executions):
            raise ValueError('requested_count does not match the number of executions.')
        request_ids = [execution.request.request_id for execution in self.executions]
        if len(request_ids) != len(set(request_ids)):
            raise ValueError('An execution batch contains duplicate request IDs.')
        invalid_term_executions = [execution.request.request_id for execution in self.executions if execution.term_id != self.term_id]
        if invalid_term_executions:
            raise ValueError(f'Execution results use a different term_id: {sorted(invalid_term_executions)}')
        invalid_round_executions = [execution.request.request_id for execution in self.executions if execution.review_round != self.review_round]
        if invalid_round_executions:
            raise ValueError(f'Execution results use a different review_round: {sorted(invalid_round_executions)}')
        expected_completed_count = sum((execution.status == EvidenceExecutionStatus.COMPLETED for execution in self.executions))
        expected_failed_count = sum((execution.status == EvidenceExecutionStatus.FAILED for execution in self.executions))
        if self.completed_count != expected_completed_count:
            raise ValueError('completed_count is incorrect.')
        if self.failed_count != expected_failed_count:
            raise ValueError('failed_count is incorrect.')
        if self.completed_count + self.failed_count != self.requested_count:
            raise ValueError('Execution counts do not equal requested_count.')
        if self.failed_count == 0:
            expected_status = EvidenceExecutionBatchStatus.COMPLETED
        elif self.completed_count == 0:
            expected_status = EvidenceExecutionBatchStatus.FAILED
        else:
            expected_status = EvidenceExecutionBatchStatus.PARTIAL
        if self.status != expected_status:
            raise ValueError('Execution batch status does not match its completed and failed request counts.')
        return self

class WorkflowOutcomeStatus(str, Enum):
    """
    Final status assigned to one input term.
    """
    MATCHED = 'matched'
    NO_SUPPORTED_MATCH = 'no_supported_match'
    HUMAN_REVIEW = 'human_review'
    TECHNICAL_FAILURE = 'technical_failure'

class WorkflowDecisionOrigin(str, Enum):
    """
    Component responsible for the terminal
    outcome.
    """
    EXACT = 'exact'
    REVIEWER = 'reviewer'
    COORDINATOR = 'coordinator'
    SYSTEM = 'system'

class LLTSelectionGateAction(str, Enum):
    """Outcome of the deterministic post-reviewer LLT gate."""
    SKIPPED = 'skipped'
    AUTO_ACCEPTED = 'auto_accepted'
    DEFERRED_LOW_CONFIDENCE = 'deferred_low_confidence'
    DEFERRED_DISAGREEMENT = 'deferred_disagreement'
    DEFERRED_PT_UNCERTAINTY = 'deferred_pt_uncertainty'
    ENRICHED_EXISTING_DEFERRAL = 'enriched_existing_deferral'

class HumanReviewCandidate(MeddraConcept):
    """
    Self-contained LLT option exposed to a human reviewer.

    Reranker fields are optional because older and non-LLT-specific human
    review paths can still contain retrieval-only candidates.
    """
    reranker_rank: int | None = Field(default=None, ge=1)
    reranker_score: float | None = Field(default=None, ge=0.0, le=1.0)
    was_retrieved: bool | None = None
    retrieval_source_count: int = Field(default=0, ge=0)
    reviewer_selected: bool = False
    reranker_shortlisted: bool = False
    retention_reasons: list[str] = Field(default_factory=list)
    introduction_methods: list[str] = Field(default_factory=list)

    @model_validator(mode='after')
    def validate_reranker_fields(self):
        if (self.reranker_rank is None) != (self.reranker_score is None):
            raise ValueError('reranker_rank and reranker_score must either both be present or both be absent.')
        return self

class HumanReviewPTGroup(SchemaModel):
    """PT-preserving human-review shortlist with nested LLT options."""
    pt_code: str = Field(min_length=1)
    pt_name: str = Field(min_length=1)
    pt_rank: int | None = Field(default=None, ge=1)
    pt_score: float | None = Field(default=None, ge=0.0, le=1.0)
    candidate_ids: list[str] = Field(min_length=1)
    llt_candidates: list[HumanReviewCandidate] = Field(min_length=1)
    pruning_explanations: list[str] = Field(default_factory=list)

    @model_validator(mode='after')
    def validate_human_review_pt_group(self):
        nested_ids = [candidate.candidate_id for candidate in self.llt_candidates]
        if nested_ids != self.candidate_ids:
            raise ValueError('Human-review PT-group candidate_ids must match nested LLTs.')
        if len(nested_ids) != len(set(nested_ids)):
            raise ValueError('A human-review PT group contains duplicate LLTs.')
        if any((candidate.pt_code != self.pt_code for candidate in self.llt_candidates)):
            raise ValueError('A human-review PT group contains an LLT from another PT.')
        return self

class LLTSelectionGateResult(SchemaModel):
    """Auditable diagnostics emitted by the post-reviewer LLT gate."""
    version: str = Field(min_length=1)
    action: LLTSelectionGateAction
    reason: str = Field(min_length=1)
    policy_configuration: dict[str, Any] = Field(default_factory=dict)
    decision_authority: str | None = None
    reviewer_failure_diagnostics: dict[str, Any] = Field(default_factory=dict)
    deterministic_repair_reasons: list[str] = Field(default_factory=list)
    ambiguity_diagnostics: list[dict[str, Any]] = Field(default_factory=list)
    compound_acceptance: dict[str, Any] = Field(default_factory=dict)
    pt_coverage_recovery: dict[str, Any] = Field(default_factory=dict)
    coverage_acceptance: dict[str, Any] = Field(default_factory=dict)
    pt_confidence_required: bool
    pt_confidence_passed: bool | None = None
    resolved_pt_code: str | None = None
    resolved_pt_name: str | None = None
    pt_top_code: str | None = None
    pt_top_score: float | None = Field(default=None, ge=0.0, le=1.0)
    pt_top_margin: float | None = Field(default=None, ge=0.0, le=1.0)
    pt_minimum_score: float = Field(ge=0.0, le=1.0)
    pt_minimum_margin: float = Field(ge=0.0, le=1.0)
    pt_confidence_failure_reasons: list[str] = Field(default_factory=list)
    reviewer_candidate_ids: list[str] = Field(default_factory=list)
    reviewer_top_agreement: bool | None = None
    minimum_score: float = Field(ge=0.0, le=1.0)
    minimum_margin: float = Field(ge=0.0, le=1.0)
    top_score: float | None = Field(default=None, ge=0.0, le=1.0)
    top_margin: float | None = Field(default=None, ge=0.0, le=1.0)
    llt_top_score: float | None = Field(default=None, ge=0.0, le=1.0)
    llt_top_margin: float | None = Field(default=None, ge=0.0, le=1.0)
    gate_action: LLTSelectionGateAction
    gate_reason: str = Field(min_length=1)
    ranked_candidates: list[HumanReviewCandidate] = Field(default_factory=list)
    human_review_candidates: list[HumanReviewCandidate] = Field(default_factory=list)
    human_review_pt_groups: list[HumanReviewPTGroup] = Field(default_factory=list)
    shortlist_pruning_explanations: list[str] = Field(default_factory=list)
    shortlist_configuration: dict[str, float | int] = Field(default_factory=dict)

    @model_validator(mode='after')
    def validate_gate_result(self):
        if self.gate_action != self.action:
            raise ValueError('gate_action must match action.')
        if self.gate_reason != self.reason:
            raise ValueError('gate_reason must match reason.')
        if self.llt_top_score != self.top_score:
            raise ValueError('llt_top_score must match top_score.')
        if self.llt_top_margin != self.top_margin:
            raise ValueError('llt_top_margin must match top_margin.')
        if self.pt_confidence_required and self.pt_confidence_passed is None:
            raise ValueError('A required PT-confidence check must record whether it passed.')
        if not self.pt_confidence_required and self.pt_confidence_passed is not None:
            raise ValueError('A skipped PT-confidence check must record pt_confidence_passed=null.')
        if self.pt_confidence_passed is True and self.pt_confidence_failure_reasons:
            raise ValueError('A passing PT-confidence check cannot contain failure reasons.')
        if self.pt_confidence_passed is False and (not self.pt_confidence_failure_reasons):
            raise ValueError('A failing PT-confidence check must contain at least one failure reason.')
        for field_name in ('reviewer_candidate_ids', 'ranked_candidates', 'human_review_candidates'):
            values = getattr(self, field_name)
            identifiers = [value if isinstance(value, str) else value.candidate_id for value in values]
            if len(identifiers) != len(set(identifiers)):
                raise ValueError(f'{field_name} contains duplicate candidate IDs.')
        grouped_ids = [candidate_id for group in self.human_review_pt_groups for candidate_id in group.candidate_ids]
        if len(grouped_ids) != len(set(grouped_ids)):
            raise ValueError('human_review_pt_groups contains duplicate candidate IDs.')
        if grouped_ids and set(grouped_ids) != {candidate.candidate_id for candidate in self.human_review_candidates}:
            raise ValueError('Structured PT groups must contain the flat human-review shortlist.')
        if (self.resolved_pt_code is None) != (self.resolved_pt_name is None):
            raise ValueError('resolved_pt_code and resolved_pt_name must either both be present or both be absent.')
        return self

class StandardizedConcept(MeddraConcept):
    """
    One MedDRA concept selected as a final
    standardization result.
    """
    supporting_methods: list[MatchMethod] = Field(min_length=1)

    @model_validator(mode='after')
    def validate_supporting_methods(self):
        if len(self.supporting_methods) != len(set(self.supporting_methods)):
            raise ValueError('supporting_methods contains duplicates.')
        return self

class ResolvedPTConcept(SchemaModel):
    """
    A PT-level mapping that is finalized while the
    choice among its LLTs remains deferred.
    """
    pt_code: str = Field(min_length=1)
    pt_name: str = Field(min_length=1)
    supporting_candidate_ids: list[str] = Field(default_factory=list)
    supporting_methods: list[MatchMethod] = Field(min_length=1)

    @model_validator(mode='after')
    def validate_resolved_pt_concept(self):
        if len(self.supporting_candidate_ids) != len(set(self.supporting_candidate_ids)):
            raise ValueError('supporting_candidate_ids contains duplicates.')
        if len(self.supporting_methods) != len(set(self.supporting_methods)):
            raise ValueError('supporting_methods contains duplicates.')
        return self

class TechnicalFailureDetail(SchemaModel):
    """
    One evidence request that failed for a
    technical reason.
    """
    request_id: str = Field(min_length=1)
    source: EvidenceSource
    action: EvidenceRequestAction
    effective_query_term: str = Field(min_length=1)
    error_message: str = Field(min_length=1)

class WorkflowOutcome(SchemaModel):
    """
    Terminal standardized record for one unique
    input term.
    """
    reviewer_failure_diagnostics: dict[str, Any] = Field(default_factory=dict)
    compound_acceptance: dict[str, Any] = Field(default_factory=dict)
    pt_coverage_recovery: dict[str, Any] = Field(default_factory=dict)
    coverage_acceptance: dict[str, Any] = Field(default_factory=dict)
    decomposition_diagnostics: dict[str, Any] = Field(default_factory=dict)
    term_id: str = Field(min_length=1)
    original_term: str = Field(min_length=1)
    normalized_term: str = Field(min_length=1)
    frequency: int = Field(ge=1)
    status: WorkflowOutcomeStatus
    decision_origin: WorkflowDecisionOrigin
    decision_authority: str | None = None
    policy_configuration: dict[str, Any] = Field(default_factory=dict)
    deterministic_repair_reasons: list[str] = Field(default_factory=list)
    shortlist_pruning_explanations: list[str] = Field(default_factory=list)
    exact_match_provenance: str | None = None
    exact_match_version: str | None = None
    review_round: int | None = Field(default=None, ge=1)
    reviewer_decision_type: ReviewerDecisionType | None = None
    selected_concepts: list[StandardizedConcept] = Field(default_factory=list)
    resolved_pt_concepts: list[ResolvedPTConcept] = Field(default_factory=list)
    human_review_candidate_ids: list[str] = Field(default_factory=list)
    human_review_candidates: list[HumanReviewCandidate] = Field(default_factory=list)
    human_review_pt_groups: list[HumanReviewPTGroup] = Field(default_factory=list)
    technical_failures: list[TechnicalFailureDetail] = Field(default_factory=list)
    decision_rationale: str = Field(min_length=1)

    @model_validator(mode='after')
    def validate_workflow_outcome(self):
        selected_ids = [concept.candidate_id for concept in self.selected_concepts]
        resolved_pt_codes = [concept.pt_code for concept in self.resolved_pt_concepts]
        if len(resolved_pt_codes) != len(set(resolved_pt_codes)):
            raise ValueError('resolved_pt_concepts contains duplicate PT codes.')
        if len(selected_ids) != len(set(selected_ids)):
            raise ValueError('selected_concepts contains duplicate candidate IDs.')
        if len(self.human_review_candidate_ids) != len(set(self.human_review_candidate_ids)):
            raise ValueError('human_review_candidate_ids contains duplicates.')
        detailed_human_review_ids = [candidate.candidate_id for candidate in self.human_review_candidates]
        if len(detailed_human_review_ids) != len(set(detailed_human_review_ids)):
            raise ValueError('human_review_candidates contains duplicate candidate IDs.')
        if detailed_human_review_ids and set(detailed_human_review_ids) != set(self.human_review_candidate_ids):
            raise ValueError('Detailed human-review candidates must match human_review_candidate_ids.')
        grouped_human_review_ids = [candidate_id for group in self.human_review_pt_groups for candidate_id in group.candidate_ids]
        if grouped_human_review_ids and set(grouped_human_review_ids) != set(self.human_review_candidate_ids):
            raise ValueError('Workflow human-review PT groups must match candidate IDs.')
        failure_request_ids = [failure.request_id for failure in self.technical_failures]
        if len(failure_request_ids) != len(set(failure_request_ids)):
            raise ValueError('technical_failures contains duplicate request IDs.')
        if self.status == WorkflowOutcomeStatus.MATCHED:
            if not self.selected_concepts:
                raise ValueError('A matched outcome requires at least one selected concept.')
            if self.resolved_pt_concepts:
                raise ValueError('A matched outcome cannot contain deferred LLT PT resolutions.')
            if self.human_review_candidate_ids:
                raise ValueError('A matched outcome cannot contain human-review candidate IDs.')
            if self.technical_failures:
                raise ValueError('A matched outcome cannot contain technical failures.')
            if self.decision_origin == WorkflowDecisionOrigin.EXACT:
                if len(self.selected_concepts) != 1:
                    raise ValueError('An exact outcome requires exactly one selected concept.')
                if self.review_round is not None:
                    raise ValueError('An exact outcome cannot have a review round.')
                if self.reviewer_decision_type is not None:
                    raise ValueError('An exact outcome cannot have a reviewer decision type.')
                if self.selected_concepts[0].supporting_methods != [MatchMethod.EXACT]:
                    raise ValueError('An exact outcome must be supported only by the exact method.')
            elif self.decision_origin == WorkflowDecisionOrigin.REVIEWER:
                if self.review_round is None:
                    raise ValueError('A reviewer match requires review_round.')
                if self.reviewer_decision_type not in {ReviewerDecisionType.MATCHED_SINGLE, ReviewerDecisionType.MATCHED_MULTIPLE}:
                    raise ValueError('A reviewer match has an invalid decision type.')
            else:
                raise ValueError('A matched outcome must originate from exact matching or the reviewer.')
        elif self.status == WorkflowOutcomeStatus.NO_SUPPORTED_MATCH:
            if self.resolved_pt_concepts:
                raise ValueError('NO_SUPPORTED_MATCH cannot contain resolved PT concepts.')
            if self.selected_concepts:
                raise ValueError('NO_SUPPORTED_MATCH cannot contain selected concepts.')
            if self.human_review_candidate_ids:
                raise ValueError('NO_SUPPORTED_MATCH cannot contain human-review candidate IDs.')
            if self.technical_failures:
                raise ValueError('NO_SUPPORTED_MATCH cannot contain technical failures.')
            if self.decision_origin != WorkflowDecisionOrigin.REVIEWER:
                raise ValueError('NO_SUPPORTED_MATCH must originate from the reviewer.')
            if self.review_round is None:
                raise ValueError('NO_SUPPORTED_MATCH requires review_round.')
            if self.reviewer_decision_type != ReviewerDecisionType.NO_SUPPORTED_MATCH:
                raise ValueError('NO_SUPPORTED_MATCH has the wrong reviewer decision type.')
        elif self.status == WorkflowOutcomeStatus.HUMAN_REVIEW:
            if self.selected_concepts:
                raise ValueError('HUMAN_REVIEW cannot contain selected concepts.')
            if self.technical_failures:
                raise ValueError('HUMAN_REVIEW cannot contain technical failures.')
            if self.review_round is None:
                raise ValueError('HUMAN_REVIEW requires review_round.')
            if self.decision_origin not in {WorkflowDecisionOrigin.REVIEWER, WorkflowDecisionOrigin.COORDINATOR}:
                raise ValueError('HUMAN_REVIEW must originate from the reviewer or coordinator.')
            if self.reviewer_decision_type == ReviewerDecisionType.MATCHED_PT_LLT_HUMAN_REVIEW:
                if self.decision_origin != WorkflowDecisionOrigin.REVIEWER:
                    raise ValueError('A PT-resolved LLT deferral must originate from the reviewer.')
                if len(self.resolved_pt_concepts) != 1:
                    raise ValueError('MATCHED_PT_LLT_HUMAN_REVIEW requires exactly one resolved PT concept.')
                if set(self.resolved_pt_concepts[0].supporting_candidate_ids) != set(self.human_review_candidate_ids):
                    raise ValueError('The resolved PT supporting candidate IDs must equal the human-review LLT candidate IDs.')
            elif self.resolved_pt_concepts:
                raise ValueError('Only MATCHED_PT_LLT_HUMAN_REVIEW may contain resolved PT concepts.')
        elif self.status == WorkflowOutcomeStatus.TECHNICAL_FAILURE:
            if self.resolved_pt_concepts:
                raise ValueError('TECHNICAL_FAILURE cannot contain resolved PT concepts.')
            if self.selected_concepts:
                raise ValueError('TECHNICAL_FAILURE cannot contain selected concepts.')
            if self.human_review_candidate_ids:
                raise ValueError('TECHNICAL_FAILURE cannot contain human-review candidate IDs.')
            if not self.technical_failures:
                raise ValueError('TECHNICAL_FAILURE requires at least one failure detail.')
            if self.decision_origin != WorkflowDecisionOrigin.SYSTEM:
                raise ValueError('TECHNICAL_FAILURE must originate from the system.')
            if self.review_round is None:
                raise ValueError('TECHNICAL_FAILURE requires review_round.')
        return self
