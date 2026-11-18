from runtime_resources import resource as asp_resource
from collections.abc import Callable
from pathlib import Path
import re
import pandas as pd
from normalization import normalization_variants
from biomedical_wording import canonical_biomedical_wording
from terminology_aliases import SAFE_BIOMEDICAL_ALIASES
from pt_reranking import explicit_clinical_conflicts
from schemas import ExactMatchResult, MeddraConcept
from state import StandardizationState
REQUIRED_REFERENCE_COLUMNS = {'normalized_term', 'llt_code', 'llt_name', 'pt_code', 'pt_name'}
EXACT_MATCH_VERSION = '1.3.1'

def _alias_key(value: str) -> str:
    text = canonical_biomedical_wording(value)
    for alias, expansion in sorted(SAFE_BIOMEDICAL_ALIASES.items(), key=lambda item: -len(item[0])):
        text = re.sub(f'\\b{re.escape(alias)}\\b', expansion, text)
    return ' '.join(text.replace('-', ' ').split())

def load_exact_lookup(reference_path: str | Path) -> dict[str, MeddraConcept]:
    """
    Load the processed MedDRA reference file and create an
    in-memory exact-matching dictionary.

    The dictionary is loaded once when the exact-matching node
    is created, not once for every input term.
    """
    reference_path = Path(reference_path)
    if not reference_path.exists():
        raise FileNotFoundError(f'The MedDRA exact-reference file was not found: {reference_path}')
    reference_df = pd.read_csv(reference_path, dtype={'normalized_term': str, 'llt_code': str, 'llt_name': str, 'pt_code': str, 'pt_name': str}, keep_default_na=False)
    missing_columns = REQUIRED_REFERENCE_COLUMNS - set(reference_df.columns)
    if missing_columns:
        raise ValueError(f'The MedDRA reference file is missing columns: {sorted(missing_columns)}')
    if reference_df['normalized_term'].eq('').any():
        raise ValueError('The MedDRA reference contains blank normalized terms.')
    duplicated_terms = reference_df.loc[reference_df['normalized_term'].duplicated(keep=False), 'normalized_term'].drop_duplicates().tolist()
    if duplicated_terms:
        raise ValueError(f'Duplicate normalized MedDRA terms were found: {duplicated_terms[:10]}')
    exact_lookup: dict[str, MeddraConcept] = {}
    for row in reference_df.itertuples(index=False):
        concept = MeddraConcept(candidate_id=f'LLT:{row.llt_code}', llt_code=row.llt_code, llt_name=row.llt_name, pt_code=row.pt_code, pt_name=row.pt_name, is_current=True)
        exact_lookup[row.normalized_term] = concept
    return exact_lookup

def load_exact_variant_lookup(reference_path: str | Path) -> dict[str, tuple[MeddraConcept, ...]]:
    """Build an ambiguity-preserving index of safe surface variants."""
    reference_path = Path(reference_path)
    reference_df = pd.read_csv(reference_path, dtype={'normalized_term': str, 'llt_code': str, 'llt_name': str, 'pt_code': str, 'pt_name': str}, keep_default_na=False)
    missing_columns = REQUIRED_REFERENCE_COLUMNS - set(reference_df.columns)
    if missing_columns:
        raise ValueError(f'The MedDRA reference file is missing columns: {sorted(missing_columns)}')
    variants: dict[str, dict[str, MeddraConcept]] = {}
    for row in reference_df.itertuples(index=False):
        if str(getattr(row, 'is_current', 'true')).casefold() not in {'true', '1', 'y', 'yes'}:
            continue
        concept = MeddraConcept(candidate_id=f'LLT:{row.llt_code}', llt_code=row.llt_code, llt_name=row.llt_name, pt_code=row.pt_code, pt_name=row.pt_name, is_current=True)
        source_forms = (row.normalized_term, row.llt_name)
        for source_form in source_forms:
            for variant, _ in normalization_variants(source_form):
                variants.setdefault(variant, {})[concept.candidate_id] = concept
    return {variant: tuple(sorted(concepts.values(), key=lambda concept: concept.candidate_id)) for variant, concepts in variants.items()}

def create_exact_matching_node(reference_path: str | Path) -> Callable[[StandardizationState], dict[str, ExactMatchResult]]:
    """
    Create the LangGraph exact-matching node.

    The MedDRA reference is loaded when this function is called.
    The returned node then reuses the same lookup dictionary for
    every input term.
    """
    variant_lookup = load_exact_variant_lookup(reference_path)
    all_concepts = {candidate.candidate_id: candidate for values in variant_lookup.values() for candidate in values}
    literal_lookup: dict[str, list[MeddraConcept]] = {}
    alias_lookup: dict[str, list[MeddraConcept]] = {}
    for candidate in all_concepts.values():
        literal_lookup.setdefault(candidate.llt_name.casefold().strip(), []).append(candidate)
        alias_lookup.setdefault(_alias_key(candidate.llt_name), []).append(candidate)

    def exact_matching_node(state: StandardizationState) -> dict[str, ExactMatchResult]:
        """
        Search for the normalized input term in the MedDRA
        exact-reference dictionary.
        """
        normalization_result = state.get('normalization')
        if normalization_result is None:
            raise ValueError('Exact matching cannot run before term normalization.')
        normalized_term = normalization_result.normalized_term
        candidate_by_id: dict[str, MeddraConcept] = {}
        matched_variants: list[str] = []
        literal_candidates = literal_lookup.get(normalization_result.original_term.casefold().strip(), [])
        provenance = 'literal_exact' if len(literal_candidates) == 1 else 'normalized_exact'
        for variant, _ in normalization_variants(normalized_term):
            variant_candidates = variant_lookup.get(variant, ())
            if variant_candidates and variant not in matched_variants:
                matched_variants.append(variant)
            for candidate in variant_candidates:
                candidate_by_id[candidate.candidate_id] = candidate
        if len(literal_candidates) == 1:
            candidate_by_id = {candidate.candidate_id: candidate for candidate in literal_candidates}
            matched_variants = [normalization_result.original_term]
        elif not candidate_by_id:
            provenance = 'alias_exact'
            candidate_by_id = {candidate.candidate_id: candidate for candidate in alias_lookup.get(_alias_key(normalized_term), [])}
            matched_variants = [_alias_key(normalized_term)]
        candidate_by_id = {identifier: candidate for identifier, candidate in candidate_by_id.items() if not explicit_clinical_conflicts(normalized_term, candidate.llt_name)}
        candidates = sorted(candidate_by_id.values(), key=lambda candidate: candidate.candidate_id)
        if len(candidates) == 1:
            result = ExactMatchResult(found=True, concept=candidates[0], match_method='exact', matched_query_variant=matched_variants[0], exact_match_version=EXACT_MATCH_VERSION, exact_match_provenance=provenance, normalization_operations=normalization_result.applied_operations)
        elif candidates:
            result = ExactMatchResult(found=False, concept=None, match_method=None, exact_match_version=EXACT_MATCH_VERSION, exact_match_provenance=provenance, normalization_operations=normalization_result.applied_operations, ambiguous_candidates=candidates, ambiguity_reason='Safe normalized lookup variants resolved to multiple current MedDRA LLTs or PTs; normal reviewer routing is required.')
        else:
            result = ExactMatchResult(found=False, concept=None, match_method=None, exact_match_version=EXACT_MATCH_VERSION)
        return {'exact_result': result}
    return exact_matching_node
