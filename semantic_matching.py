from runtime_resources import embedding_checkpoint as asp_embedding
from runtime_resources import resource as asp_resource
import json
from pathlib import Path
import numpy as np
import torch
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from schemas import SemanticCandidate, SemanticMatchResult
from semantic_pt_index import SemanticPTIndex
from state import StandardizationState
REQUIRED_CONFIG_FIELDS = {'collection_name', 'embedding_model', 'embedding_dimension', 'distance', 'number_of_vectors'}

class SemanticMatcher:
    """
    Retrieve semantic MedDRA candidates from the persistent
    Qdrant vector database.

    The embedding model and vector database are opened once
    when this object is created.
    """

    def __init__(self, vector_database_path: str | Path, config_path: str | Path, top_k: int=5, meddra_reference_path: str | Path | None=None, direct_pt_top_k: int=3, pt_expansion_llts_per_pt: int=2) -> None:
        if top_k < 1:
            raise ValueError('top_k must be at least 1.')
        self.vector_database_path = Path(vector_database_path)
        self.config_path = Path(config_path)
        self.top_k = top_k
        if direct_pt_top_k < 0:
            raise ValueError('direct_pt_top_k cannot be negative.')
        self.direct_pt_top_k = direct_pt_top_k
        self.semantic_pt_index: SemanticPTIndex | None = None
        self.pt_embeddings: np.ndarray | None = None
        self.config = self._load_config()
        self.collection_name = self.config['collection_name']
        self.embedding_model_name = self.config['embedding_model']
        self.expected_embedding_dimension = int(self.config['embedding_dimension'])
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f'Loading semantic embedding model: {self.embedding_model_name}')
        print(f'Semantic model device: {self.device}')
        self.embedding_model = SentenceTransformer(str(asp_embedding()), device=self.device)
        actual_embedding_dimension = self.embedding_model.get_embedding_dimension()
        if actual_embedding_dimension is None:
            raise ValueError('The semantic model did not report its embedding dimension.')
        if actual_embedding_dimension != self.expected_embedding_dimension:
            raise ValueError(f'Embedding-model dimension does not match the vector database. Model: {actual_embedding_dimension}; database: {self.expected_embedding_dimension}.')
        if not self.vector_database_path.exists():
            raise FileNotFoundError(f'The MedDRA vector database was not found: {self.vector_database_path}')
        self.vector_client = QdrantClient(path=str(self.vector_database_path))
        if not self.vector_client.collection_exists(self.collection_name):
            self.vector_client.close()
            raise ValueError(f"The expected Qdrant collection '{self.collection_name}' was not found.")
        self.number_of_vectors = self.vector_client.count(collection_name=self.collection_name, exact=True).count
        if self.number_of_vectors == 0:
            self.vector_client.close()
            raise ValueError('The MedDRA vector collection is empty.')
        expected_vector_count = int(self.config['number_of_vectors'])
        if self.number_of_vectors != expected_vector_count:
            self.vector_client.close()
            raise ValueError(f'The vector database count differs from its configuration. Expected {expected_vector_count:,}; found {self.number_of_vectors:,}.')
        print(f'Loaded semantic collection with {self.number_of_vectors:,} LLT vectors.')
        if meddra_reference_path is not None and self.direct_pt_top_k > 0:
            self.semantic_pt_index = SemanticPTIndex(meddra_reference_path, llts_per_pt=pt_expansion_llts_per_pt)
            if self.semantic_pt_index.pt_names:
                print(f'Building in-memory semantic PT-name index for {len(self.semantic_pt_index.pt_names):,} current PTs.')
                self.pt_embeddings = self.embedding_model.encode(list(self.semantic_pt_index.pt_names), batch_size=64, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)

    def _load_config(self) -> dict:
        """
        Load the vector database configuration produced during
        the one-time ingestion step.
        """
        if not self.config_path.exists():
            raise FileNotFoundError(f'The vector database configuration file was not found: {self.config_path}')
        with self.config_path.open('r', encoding='utf-8') as config_file:
            config = json.load(config_file)
        missing_fields = REQUIRED_CONFIG_FIELDS - set(config)
        if missing_fields:
            raise ValueError(f'The vector database configuration is missing fields: {sorted(missing_fields)}')
        if config['distance'] != 'cosine':
            raise ValueError('The semantic node requires a cosine similarity collection.')
        return config

    def generate_candidates(self, normalized_input_term: str, top_k: int | None=None) -> SemanticMatchResult:
        """
        Embed one input term and retrieve the five closest
        MedDRA LLTs from Qdrant.
        """
        if not normalized_input_term:
            raise ValueError('The normalized input term cannot be empty.')
        requested_top_k = self.top_k if top_k is None else top_k
        if requested_top_k < 1:
            raise ValueError('top_k must be at least 1.')
        query_embedding = self.embedding_model.encode([normalized_input_term], convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)[0]
        result_limit = min(requested_top_k, self.number_of_vectors)
        query_result = self.vector_client.query_points(collection_name=self.collection_name, query=query_embedding.tolist(), limit=result_limit, with_payload=True, with_vectors=False)
        candidates: list[SemanticCandidate] = []
        for rank, point in enumerate(query_result.points, start=1):
            payload = point.payload or {}
            required_payload_fields = {'llt_code', 'llt_name', 'pt_code', 'pt_name'}
            missing_payload_fields = required_payload_fields - set(payload)
            if missing_payload_fields:
                raise ValueError(f'A semantic candidate is missing payload fields: {sorted(missing_payload_fields)}')
            candidate = SemanticCandidate(candidate_id=f"LLT:{payload['llt_code']}", rank=rank, llt_code=str(payload['llt_code']), llt_name=str(payload['llt_name']), pt_code=str(payload['pt_code']), pt_name=str(payload['pt_name']), is_current=True, cosine_similarity=round(float(point.score), 6))
            candidates.append(candidate)
        if self.semantic_pt_index is not None and self.pt_embeddings is not None and (len(self.semantic_pt_index.pt_names) > 0):
            pt_scores = self.pt_embeddings @ query_embedding
            direct_pt_count = min(self.direct_pt_top_k, len(self.semantic_pt_index.pt_names))
            ranked_pt_indices = sorted(range(len(pt_scores)), key=lambda index: (-float(pt_scores[index]), index))[:direct_pt_count]
            direct_candidates = self.semantic_pt_index.expand_ranked_pts(((index, float(pt_scores[index])) for index in ranked_pt_indices))
            by_candidate_id = {candidate.candidate_id: candidate for candidate in candidates}
            for direct_candidate in direct_candidates:
                existing = by_candidate_id.get(direct_candidate.candidate_id)
                if existing is None:
                    by_candidate_id[direct_candidate.candidate_id] = direct_candidate
                    continue
                by_candidate_id[direct_candidate.candidate_id] = existing.model_copy(update={'cosine_similarity': max(existing.cosine_similarity, direct_candidate.cosine_similarity), 'direct_pt_match_score': direct_candidate.direct_pt_match_score, 'introduced_by_pt_expansion': True})
            candidates = sorted(by_candidate_id.values(), key=lambda candidate: (-candidate.cosine_similarity, candidate.candidate_id))
            candidates = [candidate.model_copy(update={'rank': rank}) for rank, candidate in enumerate(candidates, start=1)]
        return SemanticMatchResult(normalized_input_term=normalized_input_term, embedding_model=self.embedding_model_name, requested_top_k=requested_top_k, candidates=candidates)

    def __call__(self, state: StandardizationState) -> dict[str, SemanticMatchResult]:
        """
        LangGraph-compatible semantic-matching node.
        """
        normalization = state.get('normalization')
        if normalization is None:
            raise ValueError('Semantic matching cannot run before normalization.')
        exact_result = state.get('exact_result')
        if exact_result is None:
            raise ValueError('Semantic matching cannot run before exact matching.')
        if exact_result.found:
            raise ValueError('The semantic node received a term that already had an exact match.')
        semantic_result = self.generate_candidates(normalization.normalized_term)
        return {'semantic_result': semantic_result}

    def close(self) -> None:
        """
        Close the local Qdrant database and release its file
        lock.
        """
        self.vector_client.close()
