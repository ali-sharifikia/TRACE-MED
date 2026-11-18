from runtime_resources import resource as asp_resource
import re
from collections.abc import Callable
from pathlib import Path
import pandas as pd
from rapidfuzz import fuzz, process
from schemas import FuzzyCandidate, FuzzyMatchResult, MeddraConcept
from state import StandardizationState
from conflict_detection import detect_conflict_evidence
REQUIRED_REFERENCE_COLUMNS = {'normalized_term', 'llt_code', 'llt_name', 'pt_code', 'pt_name'}
TOKEN_PATTERN = re.compile('[a-z0-9]+')
LEXICAL_STOPWORDS = {'a', 'an', 'the', 'of', 'in', 'on', 'at', 'to', 'for', 'from', 'by', 'with', 'and', 'or', 'as'}

def tokenize_term(normalized_term: str) -> set[str]:
    """
    Extract substantive lexical tokens used for coverage
    and candidate-ranking evidence.

    This does not modify the original or normalized term.
    """
    all_tokens = TOKEN_PATTERN.findall(normalized_term.casefold())
    return {token for token in all_tokens if token not in LEXICAL_STOPWORDS}

class FuzzyMatcher:
    """
    Retrieve fuzzy MedDRA candidates.

    The MedDRA reference is loaded once when this object is
    created and reused for every unmatched input term.
    """

    def __init__(self, reference_path: str | Path, top_k: int=5, direct_pt_top_k: int=3, pt_expansion_llts_per_pt: int=2) -> None:
        if top_k < 1:
            raise ValueError('top_k must be at least 1.')
        self.top_k = top_k
        if direct_pt_top_k < 1 or pt_expansion_llts_per_pt < 1:
            raise ValueError('Direct PT retrieval limits must be at least 1.')
        self.direct_pt_top_k = direct_pt_top_k
        self.pt_expansion_llts_per_pt = pt_expansion_llts_per_pt
        self.reference_path = Path(reference_path)
        self.reference_df = self._load_reference()
        self.normalized_choices = self.reference_df['normalized_term'].tolist()
        self.reference_records = self.reference_df.to_dict(orient='records')
        pt_df = self.reference_df[['pt_code', 'pt_name']].drop_duplicates(subset=['pt_code'], keep='first').sort_values('pt_code').reset_index(drop=True)
        self.pt_records = pt_df.to_dict(orient='records')
        self.pt_choices = [str(record['pt_name']).casefold() for record in self.pt_records]
        self.reference_indices_by_pt: dict[str, list[int]] = {}
        for reference_index, record in enumerate(self.reference_records):
            self.reference_indices_by_pt.setdefault(str(record['pt_code']), []).append(reference_index)

    def _load_reference(self) -> pd.DataFrame:
        """
        Load the processed MedDRA reference file.
        """
        if not self.reference_path.exists():
            raise FileNotFoundError(f'MedDRA reference file was not found: {self.reference_path}')
        reference_df = pd.read_csv(self.reference_path, dtype={'normalized_term': str, 'llt_code': str, 'llt_name': str, 'pt_code': str, 'pt_name': str}, keep_default_na=False)
        missing_columns = REQUIRED_REFERENCE_COLUMNS - set(reference_df.columns)
        if missing_columns:
            raise ValueError(f'The MedDRA reference file is missing columns: {sorted(missing_columns)}')
        if reference_df.empty:
            raise ValueError('The MedDRA reference file is empty.')
        return reference_df

    def generate_candidates(self, normalized_input_term: str, top_k: int | None=None) -> FuzzyMatchResult:
        """
        Retrieve a candidate pool and rank the candidates using
        general lexical similarity and token-overlap evidence.

        No candidate is automatically accepted or rejected.
        """
        if not normalized_input_term:
            raise ValueError('The normalized input term cannot be empty.')
        requested_top_k = self.top_k if top_k is None else top_k
        if requested_top_k < 1:
            raise ValueError('top_k must be at least 1.')
        retrieval_pool_size = min(max(50, requested_top_k * 10), len(self.normalized_choices))
        raw_matches = process.extract(normalized_input_term, self.normalized_choices, scorer=fuzz.WRatio, processor=None, limit=retrieval_pool_size, score_cutoff=None)
        original_reference_indices = {int(reference_index) for _, _, reference_index in raw_matches}
        direct_pt_score_by_reference_index: dict[int, float] = {}
        pt_matches = process.extract(normalized_input_term, self.pt_choices, scorer=fuzz.WRatio, processor=None, limit=min(self.direct_pt_top_k, len(self.pt_choices)), score_cutoff=65.0)
        for _, pt_score, pt_index in pt_matches:
            pt_code = str(self.pt_records[int(pt_index)]['pt_code'])
            pt_reference_indices = self.reference_indices_by_pt.get(pt_code, [])
            if not pt_reference_indices:
                continue
            pt_name_normalized = self.pt_choices[int(pt_index)]
            pt_name_rows = [index for index in pt_reference_indices if str(self.reference_records[index]['normalized_term']).casefold() == pt_name_normalized]
            lexical_rows = process.extract(normalized_input_term, {index: self.normalized_choices[index] for index in pt_reference_indices}, scorer=fuzz.WRatio, processor=None, limit=self.pt_expansion_llts_per_pt, score_cutoff=None)
            selected_indices = list(pt_name_rows[:1])
            selected_indices.extend((int(match[2]) for match in lexical_rows))
            selected_indices = list(dict.fromkeys(selected_indices))[:self.pt_expansion_llts_per_pt]
            for reference_index in selected_indices:
                direct_pt_score_by_reference_index[reference_index] = max(float(pt_score), direct_pt_score_by_reference_index.get(reference_index, 0.0))
                if reference_index not in original_reference_indices:
                    raw_matches.append((self.normalized_choices[reference_index], float(fuzz.WRatio(normalized_input_term, self.normalized_choices[reference_index])), reference_index))
        input_tokens = tokenize_term(normalized_input_term)
        candidate_records: list[dict] = []
        for candidate_normalized_term, fuzzy_score, reference_index in raw_matches:
            reference_record = self.reference_records[reference_index]
            candidate_tokens = tokenize_term(candidate_normalized_term)
            conflict_evidence = detect_conflict_evidence(input_tokens=input_tokens, candidate_tokens=candidate_tokens)
            shared_tokens = sorted(input_tokens & candidate_tokens)
            missing_input_tokens = sorted(input_tokens - candidate_tokens)
            added_candidate_tokens = sorted(candidate_tokens - input_tokens)
            if input_tokens:
                input_token_coverage = len(shared_tokens) / len(input_tokens)
            else:
                input_token_coverage = 0.0
            if candidate_tokens:
                candidate_token_precision = len(shared_tokens) / len(candidate_tokens)
            else:
                candidate_token_precision = 0.0
            coverage_precision_sum = input_token_coverage + candidate_token_precision
            if coverage_precision_sum > 0:
                token_overlap_f1 = 2 * input_token_coverage * candidate_token_precision / coverage_precision_sum
            else:
                token_overlap_f1 = 0.0
            token_sort_score = fuzz.token_sort_ratio(normalized_input_term, candidate_normalized_term)
            token_set_score = fuzz.token_set_ratio(normalized_input_term, candidate_normalized_term)
            character_similarity_score = fuzz.ratio(normalized_input_term, candidate_normalized_term)
            sequence_score = max(character_similarity_score, token_sort_score)
            ranking_score = (sequence_score + token_overlap_f1 * 100) / 2
            direct_pt_match_score = direct_pt_score_by_reference_index.get(int(reference_index))
            if direct_pt_match_score is not None:
                ranking_score = max(ranking_score, 0.72 * direct_pt_match_score + 0.28 * sequence_score)
            candidate_records.append({'candidate_id': f"LLT:{reference_record['llt_code']}", 'llt_code': reference_record['llt_code'], 'llt_name': reference_record['llt_name'], 'pt_code': reference_record['pt_code'], 'pt_name': reference_record['pt_name'], 'is_current': True, 'fuzzy_score': round(float(fuzzy_score), 2), 'token_sort_score': round(token_sort_score, 2), 'token_set_score': round(token_set_score, 2), 'character_similarity_score': round(character_similarity_score, 2), 'shared_tokens': shared_tokens, 'missing_input_tokens': missing_input_tokens, 'added_candidate_tokens': added_candidate_tokens, 'input_token_coverage': round(input_token_coverage, 4), 'candidate_token_precision': round(candidate_token_precision, 4), 'token_overlap_f1': round(token_overlap_f1, 4), 'ranking_score': round(ranking_score, 2), 'direct_pt_match_score': round(direct_pt_match_score, 2) if direct_pt_match_score is not None else None, 'introduced_by_pt_expansion': int(reference_index) not in original_reference_indices and direct_pt_match_score is not None, **conflict_evidence})
        merged_records: dict[str, dict] = {}
        for record in candidate_records:
            existing = merged_records.get(record['candidate_id'])
            if existing is None or record['ranking_score'] > existing['ranking_score']:
                merged_records[record['candidate_id']] = record
            elif record.get('direct_pt_match_score') is not None:
                existing['direct_pt_match_score'] = max(existing.get('direct_pt_match_score') or 0.0, record['direct_pt_match_score'])
                existing['introduced_by_pt_expansion'] = existing['introduced_by_pt_expansion'] or record['introduced_by_pt_expansion']
        candidate_records = list(merged_records.values())
        candidate_records.sort(key=lambda candidate: (candidate['ranking_score'], candidate['character_similarity_score'], candidate['token_sort_score'], candidate['fuzzy_score']), reverse=True)
        ranked_candidates: list[FuzzyCandidate] = []
        for rank, candidate_record in enumerate(candidate_records[:requested_top_k], start=1):
            ranked_candidates.append(FuzzyCandidate(rank=rank, **candidate_record))
        return FuzzyMatchResult(normalized_input_term=normalized_input_term, requested_top_k=requested_top_k, candidates=ranked_candidates)

def create_fuzzy_matching_node(reference_path: str | Path, top_k: int=5) -> Callable[[StandardizationState], dict[str, FuzzyMatchResult]]:
    """
    Create the LangGraph fuzzy-matching node.

    The MedDRA reference is loaded once when this function
    creates the node.
    """
    matcher = FuzzyMatcher(reference_path=reference_path, top_k=top_k)

    def fuzzy_matching_node(state: StandardizationState) -> dict[str, FuzzyMatchResult]:
        normalization = state.get('normalization')
        if normalization is None:
            raise ValueError('Fuzzy matching cannot run before normalization.')
        exact_result = state.get('exact_result')
        if exact_result is None:
            raise ValueError('Fuzzy matching cannot run before exact matching.')
        if exact_result.found:
            raise ValueError('The fuzzy-matching node received a term that already had an exact match.')
        fuzzy_result = matcher.generate_candidates(normalization.normalized_term)
        return {'fuzzy_result': fuzzy_result}
    return fuzzy_matching_node
