from __future__ import annotations
from runtime_resources import resource as asp_resource
import re
from re import Pattern
from clinical_features import interpreted_features
from schemas import QueryExpansionSource, QueryExpansionType, QueryVariant
from terminology_aliases import TerminologyAliasIndex, build_query_variants
RETRIEVAL_QUERY_EXPANSION_VERSION = '2.0.1'
QUERY_EXPANSION_RULES: tuple[tuple[Pattern[str], str], ...] = ((re.compile('\\bnon[\\s-]*hdl[\\s-]*(?:c|cholesterol)?\\b', re.IGNORECASE), 'non high density lipoprotein cholesterol'), (re.compile('\\bhdl(?:[\\s-]*(?:c|cholesterol))?\\b', re.IGNORECASE), 'high density lipoprotein cholesterol'), (re.compile('\\bldl(?:[\\s-]*(?:c|cholesterol))?\\b', re.IGNORECASE), 'low density lipoprotein cholesterol'), (re.compile('\\bqtc(?:[\\s-]+interval)?\\b', re.IGNORECASE), 'corrected qt interval'), (re.compile('\\bast\\b', re.IGNORECASE), 'aspartate aminotransferase'), (re.compile('\\balt\\b', re.IGNORECASE), 'alanine aminotransferase'), (re.compile('\\bcpk\\b', re.IGNORECASE), 'creatine phosphokinase'), (re.compile('\\bck\\b', re.IGNORECASE), 'creatine kinase'), (re.compile('\\biop\\b', re.IGNORECASE), 'intraocular pressure'), (re.compile('\\bwbc\\b', re.IGNORECASE), 'white blood cell'), (re.compile('\\buln\\b', re.IGNORECASE), 'upper limit of normal'))

def expand_retrieval_query(query_term: str, *, alias_index: TerminologyAliasIndex | None=None) -> str:
    """
    Expand conservative biomedical abbreviations for
    candidate retrieval only.

    The normalized input stored in workflow state is
    not changed.
    """
    if not isinstance(query_term, str):
        raise TypeError('query_term must be a string.')
    expanded_query = interpreted_features(query_term)['retrieval_text']
    if not expanded_query:
        raise ValueError('query_term cannot be empty.')
    for pattern, replacement in QUERY_EXPANSION_RULES:
        expanded_query = pattern.sub(replacement, expanded_query)
    rule_expanded = ' '.join(expanded_query.split())
    if alias_index is None:
        return rule_expanded
    variants = build_retrieval_query_variants(query_term, alias_index=alias_index)
    preferred = next((variant for variant in variants if variant.expanded_query != variant.original_normalized_query and (not variant.ambiguity) and (variant.context_status == 'supported')), None)
    return preferred.expanded_query if preferred is not None else rule_expanded

def build_retrieval_query_variants(query_term: str, *, alias_index: TerminologyAliasIndex | None=None, positive_components: tuple[str, ...]=(), max_variants: int=12) -> list[QueryVariant]:
    """Return original plus every bounded, provenance-bearing query variant."""
    variants = build_query_variants(query_term, alias_index=alias_index, positive_components=positive_components, max_variants=max_variants)
    original = variants[0].original_normalized_query
    rule_expanded = interpreted_features(query_term)['retrieval_text']
    for pattern, replacement in QUERY_EXPANSION_RULES:
        rule_expanded = pattern.sub(replacement, rule_expanded)
    rule_expanded = ' '.join(rule_expanded.split())
    payloads = [variant.model_dump(mode='python') for variant in variants]
    if rule_expanded != original:
        payloads.insert(1, QueryVariant(variant_id='query_variant_rule', original_normalized_query=original, expanded_query=rule_expanded, expansion_type=QueryExpansionType.BIOMEDICAL_ABBREVIATION, expansion_source=QueryExpansionSource.BUILTIN_SAFE_RULE, confidence=0.99, ambiguity=False).model_dump(mode='python'))
    deduplicated: list[dict] = []
    seen: set[tuple[str, str | None, str | None]] = set()
    for payload in payloads:
        key = (payload['expanded_query'], payload.get('target_pt_code'), payload.get('target_concept_id'))
        if key not in seen:
            deduplicated.append(payload)
            seen.add(key)
    return [QueryVariant(**{**payload, 'variant_id': f'query_variant_{index:02d}'}) for index, payload in enumerate(deduplicated[:max_variants], start=1)]
