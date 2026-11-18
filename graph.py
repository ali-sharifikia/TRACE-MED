from __future__ import annotations
from runtime_resources import resource as asp_resource
from collections.abc import Callable
from typing import Any, Literal
from langgraph.graph import END, START, StateGraph
from aggregation import aggregation_node
from coordinator import create_coordinator_node, route_after_coordinator
from evidence_reaggregation import evidence_reaggregation_node, route_after_reaggregation
from evidence_request_executor import create_evidence_request_executor_node, route_after_evidence_execution
from normalization import normalization_node
from llt_reranking import PTConditionedLLTReranker
from llt_selection_gate import LLT_AUTO_ACCEPT_MINIMUM_MARGIN, LLT_AUTO_ACCEPT_PT_MINIMUM_MARGIN, LLT_AUTO_ACCEPT_PT_MINIMUM_SCORE, LLT_AUTO_ACCEPT_MINIMUM_SCORE, create_llt_selection_gate_node
from retrieval_query_expansion import expand_retrieval_query
from reviewer_input import reviewer_input_node
from reviewer_validation import reviewer_decision_validation_node
from state import StandardizationState
from workflow_outcome import exact_match_outcome_node, human_review_outcome_node, reviewer_final_outcome_node, technical_failure_outcome_node
from decision_recorder import node as record_node
NodeCallable = Callable[[StandardizationState], dict[str, Any]]
RetrieverCallable = Callable[..., Any]

def _route_value(route: Any) -> str:
    """
    Convert a string Enum or plain string route into
    the route name expected by LangGraph.
    """
    enum_value = getattr(route, 'value', None)
    if enum_value is not None:
        return str(enum_value)
    return str(route)

def route_after_exact_matching(state: StandardizationState) -> Literal['exact_match_outcome', 'input_interpretation']:
    """
    Exact matches can immediately produce a final outcome.

    Terms without an exact match are sent to the three
    candidate-retrieval methods.
    """
    exact_result = state.get('exact_result')
    if exact_result is None:
        raise ValueError("Exact-match routing requires state['exact_result'].")
    if exact_result.found:
        return 'exact_match_outcome'
    return 'input_interpretation'

def retrieval_dispatch_node(state: StandardizationState) -> dict[str, Any]:
    """
    Synchronization node used to start fuzzy, semantic,
    and UMLS retrieval in parallel.

    It does not modify the state.
    """
    exact_result = state.get('exact_result')
    if exact_result is None:
        raise ValueError("Retrieval dispatch requires state['exact_result'].")
    if exact_result.found:
        raise ValueError('An exact-matched term must not enter candidate retrieval.')
    return {}

def _get_normalized_query(state: StandardizationState) -> str:
    """
    Obtain the normalized input term for the initial
    retrieval nodes.
    """
    normalization_result = state.get('normalization')
    if normalization_result is None:
        raise ValueError("Candidate retrieval requires state['normalization'].")
    normalized_term = normalization_result.normalized_term
    if not normalized_term.strip():
        raise ValueError('The normalized retrieval query is empty.')
    return normalized_term

def _create_initial_retrieval_node(*, result_state_key: str, retriever: RetrieverCallable, top_k: int | None) -> NodeCallable:
    """
    Create an initial retrieval node.

    Fuzzy and semantic retrieval receive top_k.
    UMLS retrieval uses one query without top_k.
    """
    if top_k is not None and top_k < 1:
        raise ValueError('Initial retrieval top_k must be at least 1.')

    def initial_retrieval_node(state: StandardizationState) -> dict[str, Any]:
        original_query_term = _get_normalized_query(state)
        query_term = original_query_term if getattr(retriever, 'handles_query_variants', False) else expand_retrieval_query(original_query_term)
        interpretation = state.get('input_interpretation')
        positive_components = tuple((component.component_text for component in interpretation.asserted_components if not component.is_negated)) if interpretation is not None else ()
        if top_k is None:
            if getattr(retriever, 'handles_query_variants', False):
                result = retriever(query_term=query_term, positive_components=positive_components)
            else:
                result = retriever(query_term=query_term)
        elif getattr(retriever, 'handles_query_variants', False):
            result = retriever(query_term=query_term, top_k=top_k, positive_components=positive_components)
        else:
            result = retriever(query_term=query_term, top_k=top_k)
        result = result.model_copy(update={'normalized_input_term': original_query_term, 'retrieval_query_term': result.retrieval_query_term or query_term})
        return {result_state_key: result}
    return initial_retrieval_node

def _coordinator_router(state: StandardizationState) -> str:
    """
    Normalize the coordinator route for LangGraph.
    """
    return _route_value(route_after_coordinator(state))

def _evidence_execution_router(state: StandardizationState) -> str:
    """
    Normalize the evidence-execution route.
    """
    return _route_value(route_after_evidence_execution(state))

def _reaggregation_router(state: StandardizationState) -> str:
    """
    Normalize the reaggregation route.
    """
    return _route_value(route_after_reaggregation(state))

def create_standardization_graph(*, exact_matching_node: NodeCallable, input_interpretation_node: NodeCallable, fuzzy_retriever: RetrieverCallable, semantic_retriever: RetrieverCallable, umls_retriever: RetrieverCallable, reviewer_node: NodeCallable, llt_reranker: PTConditionedLLTReranker | None=None, llt_auto_accept_minimum_score: float=LLT_AUTO_ACCEPT_MINIMUM_SCORE, llt_auto_accept_minimum_margin: float=LLT_AUTO_ACCEPT_MINIMUM_MARGIN, llt_auto_accept_pt_minimum_score: float=LLT_AUTO_ACCEPT_PT_MINIMUM_SCORE, llt_auto_accept_pt_minimum_margin: float=LLT_AUTO_ACCEPT_PT_MINIMUM_MARGIN, fuzzy_top_k: int=5, semantic_top_k: int=5, umls_top_k: int=10, max_review_rounds: int=3, checkpointer: Any | None=None, clinical_specificity_reference: str | None=None, clinical_specificity_loinc: str | None=None) -> Any:
    """
    Assemble and compile the complete term-standardization
    LangGraph.

    External resources are supplied as dependencies so that
    MedDRA data, the embedding model, Qdrant, UMLS sessions,
    and Ollama are initialized once outside the graph.

    The returned compiled graph accepts one
    StandardizationState at a time.
    """
    if max_review_rounds < 1:
        raise ValueError('max_review_rounds must be at least 1.')
    initial_fuzzy_node = _create_initial_retrieval_node(result_state_key='fuzzy_result', retriever=fuzzy_retriever, top_k=fuzzy_top_k)
    initial_semantic_node = _create_initial_retrieval_node(result_state_key='semantic_result', retriever=semantic_retriever, top_k=semantic_top_k)
    initial_umls_node = _create_initial_retrieval_node(result_state_key='umls_result', retriever=umls_retriever, top_k=None)
    evidence_request_executor_node = create_evidence_request_executor_node(fuzzy_retriever=fuzzy_retriever, semantic_retriever=semantic_retriever, umls_retriever=umls_retriever)
    coordinator_node = create_coordinator_node(max_review_rounds=max_review_rounds)
    llt_selection_gate_node = None
    if llt_reranker is not None:
        llt_selection_gate_node = create_llt_selection_gate_node(reranker=llt_reranker, minimum_score=llt_auto_accept_minimum_score, minimum_margin=llt_auto_accept_minimum_margin, pt_minimum_score=llt_auto_accept_pt_minimum_score, pt_minimum_margin=llt_auto_accept_pt_minimum_margin)
    aggregation_impl = aggregation_node
    reviewer_input_impl = reviewer_input_node
    validation_impl = reviewer_decision_validation_node
    exact_outcome_impl = exact_match_outcome_node
    reviewer_outcome_impl = reviewer_final_outcome_node
    human_outcome_impl = human_review_outcome_node
    if clinical_specificity_reference is not None:
        from specificity_pipeline import SpecificityPolicy
        specificity = SpecificityPolicy(clinical_specificity_reference, clinical_specificity_loinc)
        exact_matching_node = specificity.exact(exact_matching_node)
        aggregation_impl = specificity.aggregation(fuzzy_retriever, semantic_retriever, umls_retriever)
        reviewer_input_impl = specificity.reviewer_input
        reviewer_node = specificity.decorate_reviewer(reviewer_node)
        validation_impl = specificity.validate(reviewer_decision_validation_node)
        exact_outcome_impl = specificity.outcome(exact_match_outcome_node)
        reviewer_outcome_impl = specificity.outcome(reviewer_final_outcome_node)
        human_outcome_impl = specificity.outcome(human_review_outcome_node)
    builder = StateGraph(StandardizationState)
    builder.add_node('normalization', record_node('normalization', normalization_node))
    builder.add_node('exact_matching', record_node('exact_matching', exact_matching_node))
    builder.add_node('input_interpretation', record_node('input_interpretation', input_interpretation_node))
    builder.add_node('retrieval_dispatch', record_node('retrieval_dispatch', retrieval_dispatch_node))
    builder.add_node('fuzzy_matching', record_node('fuzzy_matching', initial_fuzzy_node))
    builder.add_node('semantic_matching', record_node('semantic_matching', initial_semantic_node))
    builder.add_node('umls_matching', record_node('umls_matching', initial_umls_node))
    builder.add_node('aggregation', record_node('aggregation', aggregation_impl))
    builder.add_node('reviewer_input', record_node('reviewer_input', reviewer_input_impl))
    builder.add_node('reviewer_llm', record_node('reviewer_llm', reviewer_node))
    if llt_selection_gate_node is not None:
        builder.add_node('llt_selection_gate', record_node('llt_selection_gate', llt_selection_gate_node))
    builder.add_node('reviewer_validation', record_node('reviewer_validation', validation_impl))
    builder.add_node('coordinator', record_node('coordinator', coordinator_node))
    builder.add_node('evidence_request_executor', record_node('evidence_request_executor', evidence_request_executor_node))
    builder.add_node('evidence_reaggregation', record_node('evidence_reaggregation', evidence_reaggregation_node))
    builder.add_node('exact_match_outcome', record_node('exact_match_outcome', exact_outcome_impl))
    builder.add_node('reviewer_final_outcome', record_node('reviewer_final_outcome', reviewer_outcome_impl))
    builder.add_node('human_review_outcome', record_node('human_review_outcome', human_outcome_impl))
    builder.add_node('technical_failure_outcome', record_node('technical_failure_outcome', technical_failure_outcome_node))
    builder.add_edge(START, 'normalization')
    builder.add_edge('normalization', 'exact_matching')
    builder.add_conditional_edges('exact_matching', route_after_exact_matching, {'exact_match_outcome': 'exact_match_outcome', 'input_interpretation': 'input_interpretation'})
    builder.add_edge('exact_match_outcome', END)
    builder.add_edge('retrieval_dispatch', 'fuzzy_matching')
    builder.add_edge('retrieval_dispatch', 'semantic_matching')
    builder.add_edge('retrieval_dispatch', 'umls_matching')
    builder.add_edge('input_interpretation', 'retrieval_dispatch')
    builder.add_edge(['fuzzy_matching', 'semantic_matching', 'umls_matching'], 'aggregation')
    builder.add_edge('aggregation', 'reviewer_input')
    builder.add_edge('reviewer_input', 'reviewer_llm')
    if llt_selection_gate_node is None:
        builder.add_edge('reviewer_llm', 'reviewer_validation')
    else:
        builder.add_edge('reviewer_llm', 'llt_selection_gate')
        builder.add_edge('llt_selection_gate', 'reviewer_validation')
    builder.add_edge('reviewer_validation', 'coordinator')
    builder.add_conditional_edges('coordinator', _coordinator_router, {'finalize': 'reviewer_final_outcome', 'execute_evidence_requests': 'evidence_request_executor', 'human_review': 'human_review_outcome'})
    builder.add_conditional_edges('evidence_request_executor', _evidence_execution_router, {'reaggregate_evidence': 'evidence_reaggregation', 'evidence_execution_failed': 'technical_failure_outcome'})
    builder.add_conditional_edges('evidence_reaggregation', _reaggregation_router, {'reviewer_llm': 'reviewer_llm'})
    builder.add_edge('reviewer_final_outcome', END)
    builder.add_edge('human_review_outcome', END)
    builder.add_edge('technical_failure_outcome', END)
    if checkpointer is None:
        return builder.compile()
    return builder.compile(checkpointer=checkpointer)
