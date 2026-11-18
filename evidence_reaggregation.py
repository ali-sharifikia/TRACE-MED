from runtime_resources import resource as asp_resource
from collections.abc import Sequence
from typing import Any
from aggregation import aggregation_node
from reviewer_input import reviewer_input_node
from schemas import AggregatedCandidate, AggregationResult, CoordinatorRoute, EvidenceExecutionBatch, EvidenceExecutionBatchStatus, EvidenceExecutionStatus, EvidenceRequestAction, EvidenceRequestExecutionResult, EvidenceSource, FuzzyMatchResult, RetrievalProvenance, SemanticMatchResult, UMLSLookupStatus, UMLSMatchResult
from state import StandardizationState

class EvidenceReaggregationError(ValueError):
    """
    Raised when feedback evidence cannot safely
    be merged into the existing aggregation.
    """

def _source_result(execution: EvidenceRequestExecutionResult):
    if execution.request.source == EvidenceSource.FUZZY:
        return execution.fuzzy_result
    if execution.request.source == EvidenceSource.SEMANTIC:
        return execution.semantic_result
    if execution.request.source == EvidenceSource.UMLS:
        return execution.umls_result
    raise EvidenceReaggregationError(f'Unsupported evidence source: {execution.request.source!r}')

def _merge_execution_history(*, existing_history: Sequence[EvidenceExecutionBatch], current_batch: EvidenceExecutionBatch) -> list[EvidenceExecutionBatch]:
    """
    Append the current successful batch while
    making the operation safe to retry.
    """
    history = list(existing_history)
    for previous_batch in history:
        if previous_batch.term_id != current_batch.term_id:
            raise EvidenceReaggregationError('Execution history contains a different term_id.')
        if previous_batch.review_round == current_batch.review_round:
            if previous_batch.model_dump(mode='json') == current_batch.model_dump(mode='json'):
                return sorted(history, key=lambda batch: batch.review_round)
            raise EvidenceReaggregationError('Execution history contains a different batch for the current review round.')
    history.append(current_batch)
    return sorted(history, key=lambda batch: batch.review_round)

def _completed_executions(execution_history: Sequence[EvidenceExecutionBatch]) -> list[EvidenceRequestExecutionResult]:
    executions = []
    for batch in execution_history:
        if batch.status != EvidenceExecutionBatchStatus.COMPLETED:
            raise EvidenceReaggregationError('Only completely successful execution batches can be reaggregated.')
        for execution in batch.executions:
            if execution.status != EvidenceExecutionStatus.COMPLETED:
                raise EvidenceReaggregationError('A completed batch contains a non-completed execution.')
            executions.append(execution)
    return executions

def _select_source_candidates(*, source: EvidenceSource, initial_candidates: Sequence[Any], normalized_input_term: str, executions: Sequence[EvidenceRequestExecutionResult]) -> tuple[list[Any], dict[str, str]]:
    """
    Select one evidence object per LLT for one
    retrieval source.

    Rules:

    1. The latest full-query expansion replaces
       the initial full-query candidate set.
    2. Full-query evidence takes priority when the
       same LLT also appears in a focused search.
    3. Focused-only candidates are added.
    4. For repeated focused retrieval of the same
       focused-only LLT, the latest evidence wins.
    """
    source_executions = [execution for execution in executions if execution.request.source == source]
    expansion_executions = [execution for execution in source_executions if execution.request.action == EvidenceRequestAction.EXPAND_CANDIDATES]
    if expansion_executions:
        selected_expansion = max(expansion_executions, key=lambda execution: (execution.review_round, execution.request.requested_top_k or 0))
        expansion_result = _source_result(selected_expansion)
        if expansion_result is None:
            raise EvidenceReaggregationError('Completed expansion is missing its source result.')
        full_query_candidates = list(expansion_result.candidates)
        full_query = selected_expansion.effective_query_term
    else:
        full_query_candidates = list(initial_candidates)
        full_query = normalized_input_term
    selected_by_llt = {candidate.llt_code: candidate for candidate in full_query_candidates}
    query_by_candidate_id = {candidate.candidate_id: full_query for candidate in full_query_candidates}
    full_query_llt_codes = set(selected_by_llt)
    focused_executions = sorted([execution for execution in source_executions if execution.request.action == EvidenceRequestAction.FOCUSED_SEARCH], key=lambda execution: (execution.review_round, execution.request.request_id))
    for execution in focused_executions:
        result = _source_result(execution)
        if result is None:
            raise EvidenceReaggregationError('Completed focused search is missing its source result.')
        for candidate in result.candidates:
            if candidate.llt_code in full_query_llt_codes:
                continue
            selected_by_llt[candidate.llt_code] = candidate
            query_by_candidate_id[candidate.candidate_id] = execution.effective_query_term
    selected_candidates = sorted(selected_by_llt.values(), key=lambda candidate: (candidate.pt_code, candidate.llt_code))
    return (selected_candidates, query_by_candidate_id)

def _build_combined_source_results(*, normalized_input_term: str, initial_fuzzy_result: FuzzyMatchResult, initial_semantic_result: SemanticMatchResult, initial_umls_result: UMLSMatchResult, executions: Sequence[EvidenceRequestExecutionResult]) -> tuple[FuzzyMatchResult, SemanticMatchResult, UMLSMatchResult, dict[EvidenceSource, dict[str, str]]]:
    fuzzy_candidates, fuzzy_queries = _select_source_candidates(source=EvidenceSource.FUZZY, initial_candidates=initial_fuzzy_result.candidates, normalized_input_term=normalized_input_term, executions=executions)
    semantic_candidates, semantic_queries = _select_source_candidates(source=EvidenceSource.SEMANTIC, initial_candidates=initial_semantic_result.candidates, normalized_input_term=normalized_input_term, executions=executions)
    umls_candidates, umls_queries = _select_source_candidates(source=EvidenceSource.UMLS, initial_candidates=initial_umls_result.candidates, normalized_input_term=normalized_input_term, executions=executions)
    requested_fuzzy_sizes = [initial_fuzzy_result.requested_top_k, len(fuzzy_candidates)]
    requested_semantic_sizes = [initial_semantic_result.requested_top_k, len(semantic_candidates)]
    semantic_models = {initial_semantic_result.embedding_model}
    umls_executions = []
    for execution in executions:
        if execution.request.source == EvidenceSource.FUZZY and execution.request.requested_top_k is not None:
            requested_fuzzy_sizes.append(execution.request.requested_top_k)
        if execution.request.source == EvidenceSource.SEMANTIC:
            if execution.request.requested_top_k is not None:
                requested_semantic_sizes.append(execution.request.requested_top_k)
            if execution.semantic_result is not None:
                semantic_models.add(execution.semantic_result.embedding_model)
        if execution.request.source == EvidenceSource.UMLS:
            umls_executions.append(execution)
    if len(semantic_models) != 1:
        raise EvidenceReaggregationError('Semantic feedback used a different embedding model from the initial semantic retrieval.')
    combined_fuzzy_result = FuzzyMatchResult(normalized_input_term=normalized_input_term, requested_top_k=max(requested_fuzzy_sizes), candidates=fuzzy_candidates)
    combined_semantic_result = SemanticMatchResult(normalized_input_term=normalized_input_term, embedding_model=initial_semantic_result.embedding_model, requested_top_k=max(requested_semantic_sizes), candidates=semantic_candidates)
    if umls_candidates:
        umls_status = UMLSLookupStatus.CANDIDATES_FOUND
        umls_error_message = None
    elif umls_executions:
        umls_status = UMLSLookupStatus.NO_SUPPORTED_MATCH
        umls_error_message = None
    else:
        umls_status = initial_umls_result.lookup_status
        umls_error_message = initial_umls_result.error_message
    combined_umls_result = UMLSMatchResult(normalized_input_term=normalized_input_term, lookup_status=umls_status, searched_cui=None, searched_cui_name=None, candidates=umls_candidates, error_message=umls_error_message)
    query_maps = {EvidenceSource.FUZZY: fuzzy_queries, EvidenceSource.SEMANTIC: semantic_queries, EvidenceSource.UMLS: umls_queries}
    return (combined_fuzzy_result, combined_semantic_result, combined_umls_result, query_maps)

def _apply_evidence_queries(*, aggregation_result: AggregationResult, query_maps: dict[EvidenceSource, dict[str, str]]) -> AggregationResult:
    updated_candidates = []
    for candidate in aggregation_result.candidates:
        candidate_data = candidate.model_dump(mode='python')
        candidate_data.update({'fuzzy_query_term': query_maps[EvidenceSource.FUZZY].get(candidate.candidate_id) if candidate.fuzzy_evidence is not None else None, 'semantic_query_term': query_maps[EvidenceSource.SEMANTIC].get(candidate.candidate_id) if candidate.semantic_evidence is not None else None, 'umls_query_term': query_maps[EvidenceSource.UMLS].get(candidate.candidate_id) if candidate.umls_evidence is not None else None})
        updated_candidates.append(AggregatedCandidate.model_validate(candidate_data))
    aggregation_data = aggregation_result.model_dump(mode='python')
    aggregation_data['candidates'] = updated_candidates
    return AggregationResult.model_validate(aggregation_data)

def _build_provenance(execution: EvidenceRequestExecutionResult) -> RetrievalProvenance:
    result = _source_result(execution)
    if result is None:
        raise EvidenceReaggregationError('Completed execution has no result for provenance.')
    return RetrievalProvenance(request_id=execution.request.request_id, review_round=execution.review_round, source=execution.request.source, action=execution.request.action, query_term=execution.effective_query_term, requested_top_k=execution.request.requested_top_k, candidate_ids=[candidate.candidate_id for candidate in result.candidates])

def _merge_provenance(*, existing_provenance: Sequence[RetrievalProvenance], executions: Sequence[EvidenceRequestExecutionResult]) -> list[RetrievalProvenance]:
    provenance_by_key = {(provenance.review_round, provenance.request_id): provenance for provenance in existing_provenance}
    for execution in executions:
        provenance = _build_provenance(execution)
        key = (provenance.review_round, provenance.request_id)
        existing = provenance_by_key.get(key)
        if existing is not None:
            if existing.model_dump(mode='json') != provenance.model_dump(mode='json'):
                raise EvidenceReaggregationError(f'Conflicting provenance exists for request {provenance.request_id} in round {provenance.review_round}.')
            continue
        provenance_by_key[key] = provenance
    return sorted(provenance_by_key.values(), key=lambda provenance: (provenance.review_round, provenance.request_id))

def _retain_current_provenance_candidates(*, provenance_records: Sequence[RetrievalProvenance], aggregation_result: AggregationResult) -> list[RetrievalProvenance]:
    """
    Keep historical retrieval records while
    removing references to candidates that were
    superseded by a later expansion.
    """
    current_candidate_ids = {candidate.candidate_id for candidate in aggregation_result.candidates}
    updated_records = []
    for provenance in provenance_records:
        provenance_data = provenance.model_dump(mode='python')
        provenance_data['candidate_ids'] = [candidate_id for candidate_id in provenance.candidate_ids if candidate_id in current_candidate_ids]
        updated_records.append(RetrievalProvenance.model_validate(provenance_data))
    return updated_records

def evidence_reaggregation_node(state: StandardizationState) -> dict[str, object]:
    """
    Merge all successful feedback retrievals,
    rebuild LLT/PT aggregation, and create the
    next ReviewerInput.
    """
    reviewer_input = state.get('reviewer_input')
    if reviewer_input is None:
        raise EvidenceReaggregationError('Reaggregation requires reviewer_input.')
    coordinator_result = state.get('coordinator_result')
    if coordinator_result is None:
        raise EvidenceReaggregationError('Reaggregation requires coordinator_result.')
    if coordinator_result.route != CoordinatorRoute.EXECUTE_EVIDENCE_REQUESTS:
        raise EvidenceReaggregationError('Coordinator did not route this term through evidence execution.')
    execution_batch = state.get('evidence_execution_batch')
    if execution_batch is None:
        raise EvidenceReaggregationError('Reaggregation requires an evidence execution batch.')
    if execution_batch.status != EvidenceExecutionBatchStatus.COMPLETED:
        raise EvidenceReaggregationError('Partial or failed execution batches cannot be reaggregated.')
    if execution_batch.term_id != reviewer_input.term_id:
        raise EvidenceReaggregationError('Execution batch term_id does not match ReviewerInput.')
    if execution_batch.review_round != reviewer_input.review_round:
        raise EvidenceReaggregationError('Execution batch review_round does not match ReviewerInput.')
    expected_next_round = reviewer_input.review_round + 1
    if coordinator_result.next_review_round != expected_next_round:
        raise EvidenceReaggregationError('Coordinator next_review_round is inconsistent.')
    initial_fuzzy_result = state.get('fuzzy_result')
    initial_semantic_result = state.get('semantic_result')
    initial_umls_result = state.get('umls_result')
    if initial_fuzzy_result is None:
        raise EvidenceReaggregationError('Initial fuzzy_result is missing.')
    if initial_semantic_result is None:
        raise EvidenceReaggregationError('Initial semantic_result is missing.')
    if initial_umls_result is None:
        raise EvidenceReaggregationError('Initial umls_result is missing.')
    execution_history = _merge_execution_history(existing_history=state.get('evidence_execution_history', []), current_batch=execution_batch)
    executions = _completed_executions(execution_history)
    normalized_input_term = reviewer_input.normalized_term
    combined_fuzzy_result, combined_semantic_result, combined_umls_result, query_maps = _build_combined_source_results(normalized_input_term=normalized_input_term, initial_fuzzy_result=initial_fuzzy_result, initial_semantic_result=initial_semantic_result, initial_umls_result=initial_umls_result, executions=executions)
    aggregation_state = dict(state)
    aggregation_state.update({'fuzzy_result': combined_fuzzy_result, 'semantic_result': combined_semantic_result, 'umls_result': combined_umls_result})
    aggregation_result = aggregation_node(aggregation_state)['aggregation_result']
    aggregation_result = _apply_evidence_queries(aggregation_result=aggregation_result, query_maps=query_maps)
    retrieval_provenance = _merge_provenance(existing_provenance=state.get('retrieval_provenance', []), executions=executions)
    retrieval_provenance = _retain_current_provenance_candidates(provenance_records=retrieval_provenance, aggregation_result=aggregation_result)
    next_state = dict(state)
    next_state.update({'aggregation_result': aggregation_result, 'review_round': expected_next_round, 'retrieval_provenance': retrieval_provenance, 'evidence_execution_history': execution_history})
    reviewer_update = reviewer_input_node(next_state)
    return {'aggregation_result': aggregation_result, 'review_round': expected_next_round, 'retrieval_provenance': retrieval_provenance, 'evidence_execution_history': execution_history, 'reviewer_input': reviewer_update['reviewer_input']}

def route_after_reaggregation(state: StandardizationState) -> str:
    """
    Return to the reviewer with the expanded
    evidence package.
    """
    reviewer_input = state.get('reviewer_input')
    if reviewer_input is None:
        raise EvidenceReaggregationError('Reaggregation routing requires the new reviewer_input.')
    return 'reviewer_llm'
