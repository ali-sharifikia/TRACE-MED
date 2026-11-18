from runtime_resources import resource as asp_resource
from schemas import RetrievalProvenance, ReviewerInput
from state import StandardizationState

def reviewer_input_node(state: StandardizationState) -> dict:
    """
    Construct the evidence package supplied to the
    reviewer agent.

    This node does not call an LLM or make a
    standardization decision.
    """
    term_input = state.get('term_input')
    if term_input is None:
        raise ValueError('Reviewer input requires term_input.')
    normalization_result = state.get('normalization')
    if normalization_result is None:
        raise ValueError('Reviewer input requires the normalization result.')
    input_interpretation = state.get('input_interpretation')
    if input_interpretation is None:
        raise ValueError('ReviewerInput cannot be created because input_interpretation is missing from state.')
    exact_result = state.get('exact_result')
    if exact_result is None:
        raise ValueError('Reviewer input requires the exact-matching result.')
    if exact_result.found:
        raise ValueError('An exact-matched term should not be sent to the reviewer.')
    aggregation_result = state.get('aggregation_result')
    if aggregation_result is None:
        raise ValueError('Reviewer input requires the aggregation result.')
    review_round = state.get('review_round', 1)
    if review_round < 1:
        raise ValueError('review_round must be at least 1.')
    retrieval_provenance = state.get('retrieval_provenance', [])
    reviewer_input = ReviewerInput(term_id=state['term_id'], original_term=term_input.term, frequency=term_input.frequency, normalized_term=normalization_result.normalized_term, input_interpretation=input_interpretation, review_round=review_round, aggregation_result=aggregation_result, retrieval_provenance=list(retrieval_provenance))
    return {'reviewer_input': reviewer_input, 'review_round': review_round}
