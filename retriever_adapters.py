from __future__ import annotations
from runtime_resources import resource as asp_resource
from decision_recorder import observe
from pathlib import Path
from typing import Any, TYPE_CHECKING
from schemas import FuzzyMatchResult, SemanticMatchResult, UMLSLookupStatus, UMLSMatchResult
from retrieval_query_expansion import build_retrieval_query_variants
from terminology_aliases import TerminologyAliasIndex
if TYPE_CHECKING:
    from fuzzy_matching import FuzzyMatcher
    from semantic_matching import SemanticMatcher
    from umls_matching import UMLSMatcher

def _active_query_variants(query_term: str, alias_index: TerminologyAliasIndex, *, maximum: int=4, positive_components: tuple[str, ...]=()):
    variants = build_retrieval_query_variants(query_term, alias_index=alias_index, positive_components=positive_components, max_variants=12)
    original = variants[:1]
    expanded = [variant for variant in variants[1:] if variant.context_status != 'context_incompatible' and (variant.confidence >= 0.9 or variant.ambiguity)]
    return (original + expanded)[:maximum]

class FuzzyRetrieverAdapter:
    """
    Adapt FuzzyMatcher to the interface required
    by EvidenceRequestExecutor.
    """

    def __init__(self, matcher: FuzzyMatcher, alias_index: TerminologyAliasIndex | None=None) -> None:
        self.matcher = matcher
        self.alias_index = alias_index or TerminologyAliasIndex([])
        self.handles_query_variants = True

    def __call__(self, *, query_term: str, top_k: int, positive_components: tuple[str, ...]=()) -> FuzzyMatchResult:
        variants = _active_query_variants(query_term, self.alias_index, positive_components=positive_components)
        candidate_by_id = {}
        original_candidate_ids: list[str] = []
        for variant_index, variant in enumerate(variants):
            result = observe('retrieval.fuzzy', lambda: self.matcher.generate_candidates(normalized_input_term=variant.expanded_query, top_k=top_k), {'query_variant': variant, 'top_k': top_k})
            if variant_index == 0:
                original_candidate_ids = [candidate.candidate_id for candidate in result.candidates]
            for candidate in result.candidates:
                existing = candidate_by_id.get(candidate.candidate_id)
                variant_ids = list(dict.fromkeys((list(existing.introduction_query_variant_ids) if existing is not None else []) + [variant.variant_id]))
                direct_scores = [score for score in (existing.direct_pt_match_score if existing is not None else None, candidate.direct_pt_match_score) if score is not None]
                merged_fields = {'introduction_query_variant_ids': variant_ids, 'direct_pt_match_score': max(direct_scores) if direct_scores else None, 'introduced_by_pt_expansion': bool(candidate.introduced_by_pt_expansion or (existing is not None and existing.introduced_by_pt_expansion))}
                updated = candidate.model_copy(update=merged_fields)
                if existing is None or updated.ranking_score > existing.ranking_score:
                    candidate_by_id[candidate.candidate_id] = updated
                else:
                    candidate_by_id[candidate.candidate_id] = existing.model_copy(update=merged_fields)
        candidates = sorted(candidate_by_id.values(), key=lambda candidate: (-candidate.ranking_score, -candidate.character_similarity_score, candidate.candidate_id))
        maximum_candidates = min(20, top_k if len(variants) == 1 else top_k * 2)
        retained_ids = list(original_candidate_ids)
        retained_ids.extend((candidate.candidate_id for candidate in candidates if candidate.candidate_id not in set(retained_ids)))
        retained_ids = retained_ids[:maximum_candidates]
        retained = [candidate_by_id[candidate_id] for candidate_id in retained_ids]
        retained.sort(key=lambda candidate: (-candidate.ranking_score, candidate.candidate_id))
        retained = [candidate.model_copy(update={'rank': rank}) for rank, candidate in enumerate(retained, start=1)]
        preferred_query = next((variant.expanded_query for variant in variants if not variant.ambiguity and variant.expanded_query != variant.original_normalized_query), variants[0].expanded_query)
        return FuzzyMatchResult(normalized_input_term=query_term, retrieval_query_term=preferred_query, requested_top_k=top_k, candidates=retained, query_variants=variants)

class SemanticRetrieverAdapter:
    """
    Adapt SemanticMatcher to the interface
    required by EvidenceRequestExecutor.
    """

    def __init__(self, matcher: SemanticMatcher, alias_index: TerminologyAliasIndex | None=None) -> None:
        self.matcher = matcher
        self.alias_index = alias_index or TerminologyAliasIndex([])
        self.handles_query_variants = True

    def __call__(self, *, query_term: str, top_k: int, positive_components: tuple[str, ...]=()) -> SemanticMatchResult:
        variants = _active_query_variants(query_term, self.alias_index, positive_components=positive_components)
        candidate_by_id = {}
        original_candidate_ids: list[str] = []
        embedding_model: str | None = None
        for variant_index, variant in enumerate(variants):
            result = observe('retrieval.semantic', lambda: self.matcher.generate_candidates(normalized_input_term=variant.expanded_query, top_k=top_k), {'query_variant': variant, 'top_k': top_k})
            embedding_model = result.embedding_model
            if variant_index == 0:
                original_candidate_ids = [candidate.candidate_id for candidate in result.candidates]
            for candidate in result.candidates:
                existing = candidate_by_id.get(candidate.candidate_id)
                variant_ids = list(dict.fromkeys((list(existing.introduction_query_variant_ids) if existing is not None else []) + [variant.variant_id]))
                updated = candidate.model_copy(update={'introduction_query_variant_ids': variant_ids})
                if existing is None or updated.cosine_similarity > existing.cosine_similarity:
                    candidate_by_id[candidate.candidate_id] = updated
                else:
                    candidate_by_id[candidate.candidate_id] = existing.model_copy(update={'introduction_query_variant_ids': variant_ids})
        ranked = sorted(candidate_by_id.values(), key=lambda candidate: (-candidate.cosine_similarity, candidate.candidate_id))
        maximum_candidates = min(20, top_k if len(variants) == 1 else top_k * 2)
        retained_ids = list(original_candidate_ids)
        retained_ids.extend((candidate.candidate_id for candidate in ranked if candidate.candidate_id not in set(retained_ids)))
        retained_ids = retained_ids[:maximum_candidates]
        retained = [candidate_by_id[candidate_id] for candidate_id in retained_ids]
        retained.sort(key=lambda candidate: (-candidate.cosine_similarity, candidate.candidate_id))
        retained = [candidate.model_copy(update={'rank': rank}) for rank, candidate in enumerate(retained, start=1)]
        return SemanticMatchResult(normalized_input_term=query_term, retrieval_query_term=variants[0].expanded_query, embedding_model=embedding_model or 'unknown', requested_top_k=top_k, candidates=retained, query_variants=variants)

class UMLSRetrieverAdapter:
    """
    Adapt UMLSMatcher to the interface required
    by EvidenceRequestExecutor.
    """

    def __init__(self, matcher: UMLSMatcher, alias_index: TerminologyAliasIndex | None=None) -> None:
        self.matcher = matcher
        self.alias_index = alias_index or TerminologyAliasIndex([])
        self.handles_query_variants = True

    def __call__(self, *, query_term: str, positive_components: tuple[str, ...]=()) -> UMLSMatchResult:
        variants = _active_query_variants(query_term, self.alias_index, maximum=3, positive_components=positive_components)
        candidate_by_id = {}
        failures: list[str] = []
        first_result = None
        for variant in variants:
            result = observe('retrieval.umls', lambda: self.matcher.lookup(normalized_term=variant.expanded_query), {'query_variant': variant})
            if first_result is None:
                first_result = result
            if result.error_message:
                failures.append(result.error_message)
            for candidate in result.candidates:
                existing = candidate_by_id.get(candidate.candidate_id)
                variant_ids = list(dict.fromkeys((list(existing.introduction_query_variant_ids) if existing is not None else []) + [variant.variant_id]))
                candidate_by_id[candidate.candidate_id] = candidate.model_copy(update={'introduction_query_variant_ids': variant_ids})
        assert first_result is not None
        candidates = sorted(candidate_by_id.values(), key=lambda candidate: candidate.candidate_id)
        status = UMLSLookupStatus.CANDIDATES_FOUND if candidates else UMLSLookupStatus.LOOKUP_FAILED if failures else UMLSLookupStatus.NO_SUPPORTED_MATCH
        return UMLSMatchResult(normalized_input_term=query_term, retrieval_query_term=variants[0].expanded_query, lookup_status=status, searched_cui=first_result.searched_cui, searched_cui_name=first_result.searched_cui_name, candidates=candidates, error_message=' | '.join(dict.fromkeys(failures)) or None, query_variants=variants)

class RetrieverAdapters:
    """
    Own the real matchers and their executor
    adapters.

    Fuzzy has no external resource requiring
    closure. Semantic releases the local Qdrant
    lock, and UMLS closes its HTTP session.
    """

    def __init__(self, *, fuzzy_matcher: FuzzyMatcher, semantic_matcher: SemanticMatcher, umls_matcher: UMLSMatcher, alias_index: TerminologyAliasIndex | None=None) -> None:
        self.fuzzy_matcher = fuzzy_matcher
        self.semantic_matcher = semantic_matcher
        self.umls_matcher = umls_matcher
        self.alias_index = alias_index or TerminologyAliasIndex([])
        self.fuzzy = FuzzyRetrieverAdapter(fuzzy_matcher, self.alias_index)
        self.semantic = SemanticRetrieverAdapter(semantic_matcher, self.alias_index)
        self.umls = UMLSRetrieverAdapter(umls_matcher, self.alias_index)
        self._closed = False

    def close(self) -> None:
        """
        Release Qdrant and HTTP resources.

        Calling close more than once is safe.
        """
        if self._closed:
            return
        self._closed = True
        semantic_error: Exception | None = None
        try:
            self.semantic_matcher.close()
        except Exception as error:
            semantic_error = error
        try:
            self.umls_matcher.close()
        except Exception:
            if semantic_error is None:
                raise
        if semantic_error is not None:
            raise semantic_error

    def __enter__(self) -> 'RetrieverAdapters':
        return self

    def __exit__(self, exc_type: type[BaseException] | None, exc_value: BaseException | None, traceback: Any) -> None:
        self.close()

def create_real_retriever_adapters(*, meddra_reference_path: str | Path, vector_database_path: str | Path, vector_config_path: str | Path, umls_api_key: str | None=None) -> RetrieverAdapters:
    """
    Load each expensive resource once and return
    adapters that reuse it across terms and review
    rounds.

    umls_api_key may be omitted when UMLS_API_KEY
    is available in the project .env file.
    """
    from fuzzy_matching import FuzzyMatcher
    from semantic_matching import SemanticMatcher
    from umls_matching import UMLSMatcher
    fuzzy_matcher = FuzzyMatcher(reference_path=meddra_reference_path, top_k=5)
    reference_path = Path(meddra_reference_path)
    local_umls_synonyms = reference_path.parent / 'umls_synonyms.csv'
    alias_index = TerminologyAliasIndex.from_meddra_reference(reference_path, local_umls_synonyms_path=local_umls_synonyms if local_umls_synonyms.exists() else None)
    semantic_matcher: SemanticMatcher | None = None
    umls_matcher: UMLSMatcher | None = None
    try:
        semantic_matcher = SemanticMatcher(vector_database_path=vector_database_path, config_path=vector_config_path, top_k=5, meddra_reference_path=meddra_reference_path, direct_pt_top_k=3, pt_expansion_llts_per_pt=2)
        umls_matcher = UMLSMatcher(meddra_reference_path=Path(meddra_reference_path), api_key=umls_api_key)
        return RetrieverAdapters(fuzzy_matcher=fuzzy_matcher, semantic_matcher=semantic_matcher, umls_matcher=umls_matcher, alias_index=alias_index)
    except Exception:
        if semantic_matcher is not None:
            semantic_matcher.close()
        if umls_matcher is not None:
            umls_matcher.close()
        raise
