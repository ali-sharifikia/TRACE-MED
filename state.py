from runtime_resources import resource as asp_resource
from typing_extensions import NotRequired, TypedDict
from schemas import ExactMatchResult, FuzzyMatchResult, SemanticMatchResult, UMLSMatchResult, NormalizationResult, TermInput, EvidenceExecutionBatch, WorkflowOutcome, InputInterpretation, AggregationResult, ReviewerDecision, ReviewerInput, CoordinatorResult, EvidenceRequest, EvidenceRequestHistoryEntry, RetrievalProvenance, LLTSelectionGateResult

class StandardizationState(TypedDict):
    """
    Shared state for one unique reaction term.
    """
    term_id: str
    specificity_evidence: NotRequired[dict]
    specificity_retrieval: NotRequired[dict]
    term_input: TermInput
    normalization: NotRequired[NormalizationResult]
    input_interpretation: NotRequired[InputInterpretation]
    exact_result: NotRequired[ExactMatchResult]
    fuzzy_result: NotRequired[FuzzyMatchResult]
    semantic_result: NotRequired[SemanticMatchResult]
    umls_result: NotRequired[UMLSMatchResult]
    aggregation_result: NotRequired[AggregationResult]
    reviewer_input: NotRequired[ReviewerInput]
    reviewer_decision: NotRequired[ReviewerDecision]
    reviewer_failure_diagnostics: NotRequired[dict]
    llt_selection_gate_result: NotRequired[LLTSelectionGateResult]
    review_round: NotRequired[int]
    evidence_request_history: NotRequired[list[EvidenceRequestHistoryEntry]]
    coordinator_result: NotRequired[CoordinatorResult]
    pending_evidence_requests: NotRequired[list[EvidenceRequest]]
    evidence_execution_batch: NotRequired[EvidenceExecutionBatch]
    evidence_execution_history: NotRequired[list[EvidenceExecutionBatch]]
    retrieval_provenance: NotRequired[list[RetrievalProvenance]]
    workflow_outcome: NotRequired[WorkflowOutcome]
