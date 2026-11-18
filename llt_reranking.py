from __future__ import annotations
from runtime_resources import resource as asp_resource
from clinical_features import clinical_scoring_text, interpreted_features, feature_conflicts
import csv
import re
import unicodedata
from collections import defaultdict
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Iterable
from terminology_aliases import SAFE_BIOMEDICAL_ALIASES
from pt_reranking import _detected_analytes
from biomedical_wording import canonical_biomedical_wording, generic_context_base
LLT_RERANKER_VERSION = '1.3.0'
LLT_RERANKING_WEIGHTS = {'token_f1': 0.47, 'character_similarity': 0.33, 'containment': 0.12, 'pt_alignment': 0.08, 'retrieval_source_bonus': 0.015, 'maximum_source_bonus': 0.04, 'semantic_similarity': 0.04}
REQUIRED_REFERENCE_COLUMNS = {'llt_code', 'llt_name', 'pt_code', 'pt_name'}
TOKEN_EQUIVALENTS = {'above': 'increased', 'elevated': 'increased', 'elevation': 'increased', 'elevations': 'increased', 'high': 'increased', 'increase': 'increased', 'increases': 'increased', 'raised': 'increased', 'below': 'decreased', 'decrease': 'decreased', 'decreases': 'decreased', 'low': 'decreased', 'reduced': 'decreased', 'reduction': 'decreased', 'prolongation': 'prolonged', 'shortening': 'shortened', 'abnormalities': 'abnormal', 'counts': 'count', 'events': 'event', 'infections': 'infection', 'levels': 'level', 'reactions': 'reaction', 'values': 'value', 'haematocrit': 'hematocrit', 'haemoglobin': 'hemoglobin', 'haemorrhage': 'hemorrhage', 'oedema': 'edema', 'anaemia': 'anemia', 'tumour': 'tumor', 'tumours': 'tumor'}
IGNORED_TOKENS = {'a', 'an', 'and', 'for', 'in', 'of', 'on', 'or', 'the', 'to', 'with'}
MODIFIER_FAMILIES = (frozenset({'increased', 'decreased', 'normal'}), frozenset({'positive', 'negative'}), frozenset({'prolonged', 'shortened'}), frozenset({'acute', 'chronic'}), frozenset({'left', 'right', 'bilateral'}), frozenset({'mild', 'moderate', 'severe'}), frozenset({'present', 'absent'}))

@dataclass(frozen=True, slots=True)
class LLTReferenceEntry:
    candidate_id: str
    llt_code: str
    llt_name: str
    pt_code: str
    pt_name: str
    is_current: bool = True

@dataclass(frozen=True, slots=True)
class LLTRankingEntry:
    rank: int
    candidate_id: str
    llt_code: str
    llt_name: str
    pt_code: str
    pt_name: str
    score: float
    score_gap_from_previous: float | None
    token_f1: float
    character_similarity: float
    containment: float
    pt_alignment: float
    modifier_penalty: float
    modifier_conflicts: tuple[str, ...]
    was_retrieved: bool
    retrieval_source_count: int
    semantic_similarity: float
    surface_match_type: str
    alias_equivalent: bool
    added_analytes: tuple[str, ...]
    unsupported_specificity_penalty: float
    unrounded_score: float | None = None

    def model_dump(self) -> dict:
        return asdict(self)

def _strip_accents(value: str) -> str:
    return ''.join((character for character in unicodedata.normalize('NFKD', value) if not unicodedata.combining(character)))

def normalize_llt_text(value: str) -> str:
    """Normalize wording for deterministic LLT comparison."""
    normalized = _strip_accents(canonical_biomedical_wording(clinical_scoring_text(generic_context_base(value) or value)))
    normalized = normalized.replace('≥', '>=')
    normalized = normalized.replace('≤', '<=')
    for alias, expansion in sorted(SAFE_BIOMEDICAL_ALIASES.items(), key=lambda item: -len(item[0])):
        normalized = re.sub(f'\\b{re.escape(alias)}\\b', f' {expansion} ', normalized)
    raw_tokens = re.findall('[a-z]+', normalized)
    tokens = [TOKEN_EQUIVALENTS.get(token, token) for token in raw_tokens if token not in IGNORED_TOKENS]
    return ' '.join(tokens)

def _token_f1(first: str, second: str) -> tuple[float, float]:
    first_tokens = set(first.split())
    second_tokens = set(second.split())
    if not first_tokens or not second_tokens:
        return (0.0, 0.0)
    overlap = len(first_tokens & second_tokens)
    precision = overlap / len(second_tokens)
    recall = overlap / len(first_tokens)
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    containment = float(first_tokens <= second_tokens or second_tokens <= first_tokens)
    return (f1, containment)

def _text_features(first_value: str, second_value: str) -> tuple[float, float, float]:
    first = normalize_llt_text(first_value)
    second = normalize_llt_text(second_value)
    token_f1, containment = _token_f1(first, second)
    character_similarity = SequenceMatcher(None, ' '.join(sorted(first.split())), ' '.join(sorted(second.split()))).ratio()
    return (token_f1, character_similarity, containment)

def _modifier_penalty(input_term: str, candidate_term: str) -> tuple[float, tuple[str, ...]]:
    input_tokens = set(normalize_llt_text(input_term).split())
    candidate_tokens = set(normalize_llt_text(candidate_term).split())
    penalty = 0.0
    conflicts: list[str] = []
    for family in MODIFIER_FAMILIES:
        input_modifiers = input_tokens & family
        candidate_modifiers = candidate_tokens & family
        if input_modifiers and candidate_modifiers and input_modifiers.isdisjoint(candidate_modifiers):
            penalty += 0.4
            conflicts.append(f'{sorted(input_modifiers)} vs {sorted(candidate_modifiers)}')
        elif input_modifiers and (not candidate_modifiers):
            penalty += 0.06
        elif candidate_modifiers and (not input_modifiers):
            penalty += 0.08
    return (min(0.5, penalty), tuple(conflicts))

def score_llt_candidate(input_term: str, candidate: LLTReferenceEntry, *, was_retrieved: bool=False, retrieval_source_count: int=0, semantic_similarity: float=0.0) -> dict:
    """Return an interpretable lexical score for one LLT inside a fixed PT."""
    normalized_input = normalize_llt_text(input_term)
    normalized_llt = normalize_llt_text(candidate.llt_name)
    token_f1, character_similarity, containment = _text_features(input_term, candidate.llt_name)
    pt_token_f1, pt_character_similarity, _ = _text_features(input_term, candidate.pt_name)
    pt_alignment = max(pt_token_f1, pt_character_similarity)
    modifier_penalty, modifier_conflicts = _modifier_penalty(input_term, candidate.llt_name)
    hard_conflicts = set(feature_conflicts(input_term, candidate.llt_name)) | set(feature_conflicts(input_term, candidate.pt_name))
    modifier_penalty = min(0.6, modifier_penalty + 0.3 * len(hard_conflicts - {'analyte_conflict', 'directionality_conflict'}))
    modifier_conflicts = tuple(sorted(set(modifier_conflicts) | hard_conflicts))
    raw_input = ' '.join(_strip_accents(input_term.casefold()).split())
    raw_llt = ' '.join(_strip_accents(candidate.llt_name.casefold()).split())
    punctuation_folded_input = re.sub('[^a-z0-9]+', ' ', raw_input).strip()
    punctuation_folded_llt = re.sub('[^a-z0-9]+', ' ', raw_llt).strip()
    if raw_input == raw_llt:
        surface_match_type = 'exact_normalized_surface'
    elif punctuation_folded_input == punctuation_folded_llt:
        surface_match_type = 'hyphenation_or_punctuation_only'
    elif normalized_input and sorted(normalized_input.split()) == sorted(normalized_llt.split()):
        surface_match_type = 'word_order_equivalent'
    else:
        surface_match_type = 'non_exact'
    alias_equivalent = bool(normalized_input and normalized_input == normalized_llt and (punctuation_folded_input != punctuation_folded_llt) and (not hard_conflicts) and (not interpreted_features(input_term)['bounds']))
    if hard_conflicts:
        surface_match_type = 'protected_clinical_conflict'
    elif surface_match_type == 'hyphenation_or_punctuation_only' and interpreted_features(input_term)['bounds'] != interpreted_features(candidate.llt_name)['bounds']:
        surface_match_type = 'numerical_constraints_not_equivalent'
    input_analytes = _detected_analytes(input_term)
    candidate_analytes = _detected_analytes(candidate.llt_name)
    added_analytes = candidate_analytes - input_analytes
    analyte_penalty = min(0.35, 0.22 * len(added_analytes))
    input_tokens = set(normalized_input.split())
    candidate_tokens = set(normalized_llt.split())
    specificity_tokens = {'hereditary', 'congenital', 'familial', 'primary', 'secondary', 'systemic', 'localized', 'disseminated', 'recurrent', 'severe', 'moderate', 'mild', 'drug', 'induced', 'postoperative'}
    specificity_tokens -= set().union(*MODIFIER_FAMILIES)
    added_specificity = (candidate_tokens & specificity_tokens) - input_tokens
    specificity_penalty = min(0.2, 0.08 * len(added_specificity))
    if normalized_input and normalized_input == normalized_llt and (not hard_conflicts) and (not interpreted_features(input_term)['bounds']):
        score = 1.0
    else:
        score = LLT_RERANKING_WEIGHTS['token_f1'] * token_f1 + LLT_RERANKING_WEIGHTS['character_similarity'] * character_similarity + LLT_RERANKING_WEIGHTS['containment'] * containment + LLT_RERANKING_WEIGHTS['pt_alignment'] * pt_alignment - modifier_penalty - analyte_penalty - specificity_penalty + min(LLT_RERANKING_WEIGHTS['maximum_source_bonus'], LLT_RERANKING_WEIGHTS['retrieval_source_bonus'] * retrieval_source_count) + LLT_RERANKING_WEIGHTS['semantic_similarity'] * max(0.0, semantic_similarity)
    return {'score': round(max(0.0, min(1.0, score)), 6), 'unrounded_score': max(0.0, min(1.0, score)), 'token_f1': round(token_f1, 6), 'character_similarity': round(character_similarity, 6), 'containment': round(containment, 6), 'pt_alignment': round(pt_alignment, 6), 'modifier_penalty': round(modifier_penalty, 6), 'modifier_conflicts': modifier_conflicts, 'was_retrieved': was_retrieved, 'retrieval_source_count': retrieval_source_count, 'semantic_similarity': round(max(0.0, semantic_similarity), 6), 'surface_match_type': surface_match_type, 'alias_equivalent': alias_equivalent, 'added_analytes': tuple(sorted(added_analytes)), 'unsupported_specificity_penalty': round(specificity_penalty, 6)}

class PTConditionedLLTReranker:
    """Enumerate and rank all current LLTs beneath a resolved MedDRA PT."""

    def __init__(self, reference_path: str | Path) -> None:
        self.reference_path = Path(reference_path)
        if not self.reference_path.exists():
            raise FileNotFoundError(f'The MedDRA reference file was not found: {self.reference_path}')
        self._entries_by_pt: dict[str, tuple[LLTReferenceEntry, ...]] = {}
        self._load_reference()

    def _load_reference(self) -> None:
        by_pt: dict[str, list[LLTReferenceEntry]] = defaultdict(list)
        with self.reference_path.open('r', encoding='utf-8-sig', newline='') as handle:
            reader = csv.DictReader(handle)
            columns = set(reader.fieldnames or [])
            missing = REQUIRED_REFERENCE_COLUMNS - columns
            if missing:
                raise ValueError(f'The MedDRA reference file is missing columns: {sorted(missing)}')
            seen_llt_codes: set[str] = set()
            for row_number, row in enumerate(reader, start=2):
                llt_code = str(row['llt_code']).strip()
                llt_name = str(row['llt_name']).strip()
                pt_code = str(row['pt_code']).strip()
                pt_name = str(row['pt_name']).strip()
                if not all((llt_code, llt_name, pt_code, pt_name)):
                    raise ValueError(f'Blank MedDRA identity field at reference row {row_number}.')
                if llt_code in seen_llt_codes:
                    raise ValueError(f'Duplicate LLT code in reference: {llt_code}')
                seen_llt_codes.add(llt_code)
                is_current = str(row.get('is_current', 'true')).strip().casefold()
                if is_current in {'false', '0', 'no', 'n'}:
                    continue
                by_pt[pt_code].append(LLTReferenceEntry(candidate_id=f'LLT:{llt_code}', llt_code=llt_code, llt_name=llt_name, pt_code=pt_code, pt_name=pt_name))
        self._entries_by_pt = {pt_code: tuple(sorted(entries, key=lambda entry: entry.llt_code)) for pt_code, entries in by_pt.items()}

    def entries_for_pt(self, pt_code: str) -> tuple[LLTReferenceEntry, ...]:
        return self._entries_by_pt.get(str(pt_code), ())

    def rank(self, *, input_term: str, pt_code: str, retrieved_candidates: Iterable[dict]=(), top_k: int | None=None) -> list[LLTRankingEntry]:
        if not input_term.strip():
            raise ValueError('input_term cannot be blank.')
        if top_k is not None and top_k < 1:
            raise ValueError('top_k must be at least 1 when provided.')
        entries = self.entries_for_pt(pt_code)
        if not entries:
            raise ValueError(f'No current LLTs were found for PT {pt_code}.')
        retrieval_by_code = {str(candidate.get('llt_code', '')): candidate for candidate in retrieved_candidates if str(candidate.get('pt_code', '')) == str(pt_code)}
        scored: list[tuple[LLTReferenceEntry, dict]] = []
        for entry in entries:
            retrieved = retrieval_by_code.get(entry.llt_code)
            scored.append((entry, score_llt_candidate(input_term, entry, was_retrieved=retrieved is not None, retrieval_source_count=int(retrieved.get('source_count', 0)) if retrieved is not None else 0, semantic_similarity=float(retrieved.get('semantic_similarity', 0.0)) if retrieved is not None else 0.0)))
        scored.sort(key=lambda item: (-item[1]['score'], -int(item[1]['was_retrieved']), -item[1]['retrieval_source_count'], item[0].llt_code))
        rankings: list[LLTRankingEntry] = []
        previous_score: float | None = None
        for rank, (entry, features) in enumerate(scored, start=1):
            score = features['score']
            rankings.append(LLTRankingEntry(rank=rank, candidate_id=entry.candidate_id, llt_code=entry.llt_code, llt_name=entry.llt_name, pt_code=entry.pt_code, pt_name=entry.pt_name, score=score, unrounded_score=features['unrounded_score'], score_gap_from_previous=None if previous_score is None else round(previous_score - score, 6), token_f1=features['token_f1'], character_similarity=features['character_similarity'], containment=features['containment'], pt_alignment=features['pt_alignment'], modifier_penalty=features['modifier_penalty'], modifier_conflicts=features['modifier_conflicts'], was_retrieved=features['was_retrieved'], retrieval_source_count=features['retrieval_source_count'], semantic_similarity=features['semantic_similarity'], surface_match_type=features['surface_match_type'], alias_equivalent=features['alias_equivalent'], added_analytes=features['added_analytes'], unsupported_specificity_penalty=features['unsupported_specificity_penalty']))
            previous_score = score
        from decision_recorder import reference_inventory
        reference_inventory(rankings, input_term=input_term, pt_code=pt_code, return_limit=top_k)
        return rankings[:top_k] if top_k is not None else rankings
