from runtime_resources import resource as asp_resource
from collections.abc import Sequence
from typing import Protocol
from schemas import CoordinatorRoute, EvidenceExecutionBatch, EvidenceExecutionBatchStatus, EvidenceExecutionStatus, EvidenceRequest, EvidenceRequestAction, EvidenceRequestExecutionResult, EvidenceRequestHistoryEntry, EvidenceSource, FuzzyMatchResult, SemanticMatchResult, UMLSLookupStatus, UMLSMatchResult
from state import StandardizationState

class FuzzyRetriever(Protocol):
    """
    Interface expected from fuzzy retrieval.

    An adapter will later connect this interface
    to the existing fuzzy matcher.
    """

    def __call__(self, *, query_term: str, top_k: int) -> FuzzyMatchResult:
        ...

class SemanticRetriever(Protocol):
    """
    Interface expected from semantic retrieval.
    """

    def __call__(self, *, query_term: str, top_k: int) -> SemanticMatchResult:
        ...

class UMLSRetriever(Protocol):
    """
    Interface expected from UMLS retrieval.
    """

    def __call__(self, *, query_term: str) -> UMLSMatchResult:
        ...

class EvidenceRequestExecutorError(ValueError):
    """
    Raised when the executor receives an
    inconsistent workflow state.
    """

def _normalize_text_for_comparison(text: str) -> str:
    return ' '.join(text.casefold().split())

def _effective_query_term(request: EvidenceRequest, normalized_input_term: str) -> str:
    """
    Determine which term must be sent to the
    retrieval component.

    Expand-candidates uses the complete normalized
    input term. Focused search uses the subterm
    requested by the reviewer.
    """
    if request.action == EvidenceRequestAction.EXPAND_CANDIDATES:
        return normalized_input_term
    if request.action == EvidenceRequestAction.FOCUSED_SEARCH:
        if request.query_term is None:
            raise EvidenceRequestExecutorError('Focused search is missing query_term.')
        query_term = request.query_term.strip()
        if not query_term:
            raise EvidenceRequestExecutorError('Focused-search query_term is empty.')
        return query_term
    raise EvidenceRequestExecutorError(f'Unsupported evidence-request action: {request.action!r}')

def _request_signature(request: EvidenceRequest) -> tuple[str, str, str | None, int | None, tuple[str, ...]]:
    """
    Identify the actual retrieval operation.

    request_id and reason are excluded because
    changing them does not change the operation.
    """
    normalized_query = None
    if request.query_term is not None:
        normalized_query = _normalize_text_for_comparison(request.query_term)
    return (request.source.value, request.action.value, normalized_query, request.requested_top_k, tuple(sorted(request.candidate_ids_of_interest)))

def _validate_pending_requests(pending_requests: Sequence[EvidenceRequest]) -> None:
    """
    Reject duplicate request IDs or duplicate
    retrieval operations in the same batch.
    """
    if not pending_requests:
        raise EvidenceRequestExecutorError('Evidence execution requires at least one pending request.')
    request_ids = [request.request_id for request in pending_requests]
    if len(request_ids) != len(set(request_ids)):
        raise EvidenceRequestExecutorError('Pending evidence requests contain duplicate request IDs.')
    signatures = [_request_signature(request) for request in pending_requests]
    if len(signatures) != len(set(signatures)):
        raise EvidenceRequestExecutorError('Pending evidence requests contain duplicate retrieval operations.')

class EvidenceRequestExecutor:
    """
    Execute structured reviewer requests using
    injected fuzzy, semantic, and UMLS retrievers.

    The executor performs retrieval only. It does
    not reaggregate candidates or increment the
    review round.
    """

    def __init__(self, *, fuzzy_retriever: FuzzyRetriever, semantic_retriever: SemanticRetriever, umls_retriever: UMLSRetriever) -> None:
        self._fuzzy_retriever = fuzzy_retriever
        self._semantic_retriever = semantic_retriever
        self._umls_retriever = umls_retriever

    def _execute_fuzzy_request(self, *, term_id: str, review_round: int, request: EvidenceRequest, effective_query_term: str) -> EvidenceRequestExecutionResult:
        if request.requested_top_k is None:
            raise EvidenceRequestExecutorError('Fuzzy retrieval requires requested_top_k.')
        fuzzy_result = self._fuzzy_retriever(query_term=effective_query_term, top_k=request.requested_top_k)
        fuzzy_result = FuzzyMatchResult.model_validate(fuzzy_result)
        return EvidenceRequestExecutionResult(term_id=term_id, review_round=review_round, request=request, effective_query_term=effective_query_term, status=EvidenceExecutionStatus.COMPLETED, fuzzy_result=fuzzy_result, semantic_result=None, umls_result=None, candidate_count=len(fuzzy_result.candidates), error_message=None)

    def _execute_semantic_request(self, *, term_id: str, review_round: int, request: EvidenceRequest, effective_query_term: str) -> EvidenceRequestExecutionResult:
        if request.requested_top_k is None:
            raise EvidenceRequestExecutorError('Semantic retrieval requires requested_top_k.')
        semantic_result = self._semantic_retriever(query_term=effective_query_term, top_k=request.requested_top_k)
        semantic_result = SemanticMatchResult.model_validate(semantic_result)
        return EvidenceRequestExecutionResult(term_id=term_id, review_round=review_round, request=request, effective_query_term=effective_query_term, status=EvidenceExecutionStatus.COMPLETED, fuzzy_result=None, semantic_result=semantic_result, umls_result=None, candidate_count=len(semantic_result.candidates), error_message=None)

    def _execute_umls_request(self, *, term_id: str, review_round: int, request: EvidenceRequest, effective_query_term: str) -> EvidenceRequestExecutionResult:
        umls_result = self._umls_retriever(query_term=effective_query_term)
        umls_result = UMLSMatchResult.model_validate(umls_result)
        if umls_result.lookup_status == UMLSLookupStatus.LOOKUP_FAILED:
            error_message = umls_result.error_message or 'UMLS lookup failed without an error message.'
            return EvidenceRequestExecutionResult(term_id=term_id, review_round=review_round, request=request, effective_query_term=effective_query_term, status=EvidenceExecutionStatus.FAILED, fuzzy_result=None, semantic_result=None, umls_result=umls_result, candidate_count=0, error_message=error_message)
        return EvidenceRequestExecutionResult(term_id=term_id, review_round=review_round, request=request, effective_query_term=effective_query_term, status=EvidenceExecutionStatus.COMPLETED, fuzzy_result=None, semantic_result=None, umls_result=umls_result, candidate_count=len(umls_result.candidates), error_message=None)

    def _failed_execution(self, *, term_id: str, review_round: int, request: EvidenceRequest, effective_query_term: str, error: Exception) -> EvidenceRequestExecutionResult:
        error_text = str(error).strip()
        if not error_text:
            error_text = f'The retrieval function raised {type(error).__name__}.'
        else:
            error_text = f'{type(error).__name__}: {error_text}'
        return EvidenceRequestExecutionResult(term_id=term_id, review_round=review_round, request=request, effective_query_term=effective_query_term, status=EvidenceExecutionStatus.FAILED, fuzzy_result=None, semantic_result=None, umls_result=None, candidate_count=0, error_message=error_text)

    def execute_request(self, *, term_id: str, review_round: int, normalized_input_term: str, request: EvidenceRequest) -> EvidenceRequestExecutionResult:
        """
        Execute one request and convert ordinary
        retrieval exceptions into an auditable
        failed execution result.
        """
        effective_query_term = _effective_query_term(request=request, normalized_input_term=normalized_input_term)
        try:
            if request.source == EvidenceSource.FUZZY:
                return self._execute_fuzzy_request(term_id=term_id, review_round=review_round, request=request, effective_query_term=effective_query_term)
            if request.source == EvidenceSource.SEMANTIC:
                return self._execute_semantic_request(term_id=term_id, review_round=review_round, request=request, effective_query_term=effective_query_term)
            if request.source == EvidenceSource.UMLS:
                return self._execute_umls_request(term_id=term_id, review_round=review_round, request=request, effective_query_term=effective_query_term)
            raise EvidenceRequestExecutorError(f'Unsupported evidence source: {request.source!r}')
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as error:
            return self._failed_execution(term_id=term_id, review_round=review_round, request=request, effective_query_term=effective_query_term, error=error)

    def execute_batch(self, *, term_id: str, review_round: int, normalized_input_term: str, pending_requests: Sequence[EvidenceRequest]) -> EvidenceExecutionBatch:
        """
        Execute every request from one reviewer
        decision.

        One failure does not prevent other requests
        in the same batch from being attempted.
        """
        _validate_pending_requests(pending_requests)
        executions = [self.execute_request(term_id=term_id, review_round=review_round, normalized_input_term=normalized_input_term, request=request) for request in pending_requests]
        completed_count = sum((execution.status == EvidenceExecutionStatus.COMPLETED for execution in executions))
        failed_count = len(executions) - completed_count
        if failed_count == 0:
            batch_status = EvidenceExecutionBatchStatus.COMPLETED
        elif completed_count == 0:
            batch_status = EvidenceExecutionBatchStatus.FAILED
        else:
            batch_status = EvidenceExecutionBatchStatus.PARTIAL
        return EvidenceExecutionBatch(term_id=term_id, review_round=review_round, requested_count=len(pending_requests), completed_count=completed_count, failed_count=failed_count, status=batch_status, executions=executions)

def create_evidence_request_executor_node(*, fuzzy_retriever: FuzzyRetriever, semantic_retriever: SemanticRetriever, umls_retriever: UMLSRetriever):
    """
    Create the LangGraph-compatible evidence
    request executor node.
    """
    executor = EvidenceRequestExecutor(fuzzy_retriever=fuzzy_retriever, semantic_retriever=semantic_retriever, umls_retriever=umls_retriever)

    def evidence_request_executor_node(state: StandardizationState) -> dict[str, object]:
        reviewer_input = state.get('reviewer_input')
        if reviewer_input is None:
            raise EvidenceRequestExecutorError('Evidence execution requires reviewer_input.')
        coordinator_result = state.get('coordinator_result')
        if coordinator_result is None:
            raise EvidenceRequestExecutorError('Evidence execution requires coordinator_result.')
        if coordinator_result.route != CoordinatorRoute.EXECUTE_EVIDENCE_REQUESTS:
            raise EvidenceRequestExecutorError('Coordinator did not route this term to evidence execution.')
        pending_requests = state.get('pending_evidence_requests', [])
        _validate_pending_requests(pending_requests)
        coordinator_request_ids = {request.request_id for request in coordinator_result.pending_evidence_requests}
        state_request_ids = {request.request_id for request in pending_requests}
        if coordinator_request_ids != state_request_ids:
            raise EvidenceRequestExecutorError('State pending requests do not match the coordinator result.')
        coordinator_request_by_id = {request.request_id: request for request in coordinator_result.pending_evidence_requests}
        for request in pending_requests:
            coordinator_request = coordinator_request_by_id[request.request_id]
            if request.model_dump(mode='json') != coordinator_request.model_dump(mode='json'):
                raise EvidenceRequestExecutorError(f'A pending request differs from the request approved by the coordinator: {request.request_id}')
        execution_batch = executor.execute_batch(term_id=reviewer_input.term_id, review_round=reviewer_input.review_round, normalized_input_term=reviewer_input.normalized_term, pending_requests=pending_requests)
        existing_history = list(state.get('evidence_request_history', []))
        completed_history_entries = [EvidenceRequestHistoryEntry(term_id=reviewer_input.term_id, review_round=reviewer_input.review_round, request=execution.request) for execution in execution_batch.executions if execution.status == EvidenceExecutionStatus.COMPLETED]
        updated_history = existing_history + completed_history_entries
        return {'evidence_execution_batch': execution_batch, 'evidence_request_history': updated_history, 'pending_evidence_requests': []}
    return evidence_request_executor_node

def route_after_evidence_execution(state: StandardizationState) -> str:
    """
    Route only a completely successful batch to
    reaggregation.

    Partial or failed batches require explicit
    failure handling and must not silently create
    a new reviewer round.
    """
    execution_batch = state.get('evidence_execution_batch')
    if execution_batch is None:
        raise EvidenceRequestExecutorError('Evidence-execution routing requires evidence_execution_batch.')
    if execution_batch.status == EvidenceExecutionBatchStatus.COMPLETED:
        return 'reaggregate_evidence'
    return 'evidence_execution_failed'
