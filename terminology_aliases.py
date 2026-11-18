from __future__ import annotations
from runtime_resources import resource as asp_resource
import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from normalization import normalize_term, normalization_variants
from clinical_features import expansion_context_status
from schemas import QueryExpansionSource, QueryExpansionType, QueryVariant
TERMINOLOGY_ALIAS_VERSION = '1.0.1'
ACRONYM_TOKEN_PATTERN = re.compile('[a-z0-9]+')
ACRONYM_STOPWORDS = {'a', 'an', 'and', 'at', 'by', 'for', 'from', 'in', 'of', 'on', 'or', 'the', 'to', 'with', 'without'}
SAFE_BIOMEDICAL_ALIASES: dict[str, str] = {'alt': 'alanine aminotransferase', 'sgpt': 'alanine aminotransferase', 'alanine transaminase': 'alanine aminotransferase', 'ast': 'aspartate aminotransferase', 'sgot': 'aspartate aminotransferase', 'aspartate transaminase': 'aspartate aminotransferase', 'cdad': 'clostridium difficile associated diarrhoea', 'dress': 'drug reaction with eosinophilia and systemic symptoms', 'onj': 'osteonecrosis of jaw', 'pjp': 'pneumocystis jirovecii pneumonia', 'pml': 'progressive multifocal leukoencephalopathy', 'rpls': 'reversible posterior leukoencephalopathy syndrome', 'sjs': 'stevens johnson syndrome', 'tls': 'tumour lysis syndrome', 'vte': 'venous thromboembolism', 'hdl': 'high density lipoprotein cholesterol', 'hdl-c': 'high density lipoprotein cholesterol', 'ldl': 'low density lipoprotein cholesterol', 'ldl-c': 'low density lipoprotein cholesterol', 'qtc': 'corrected qt interval', 'cpk': 'creatine phosphokinase', 'ck': 'creatine kinase', 'iop': 'intraocular pressure', 'wbc': 'white blood cell', 'uln': 'upper limit of normal', 'lln': 'lower limit of normal'}

def canonical_alias_text(value: str) -> str:
    variants = normalization_variants(value)
    hyphen_space = next((text for text, operation in variants if operation == 'hyphen_to_space'), variants[0][0])
    return ' '.join(hyphen_space.split())

def generated_acronym(value: str) -> str | None:
    tokens = [token for token in ACRONYM_TOKEN_PATTERN.findall(value.casefold()) if token not in ACRONYM_STOPWORDS and (not token.isdigit())]
    if len(tokens) < 3:
        return None
    acronym = ''.join((token[0] for token in tokens))
    if not 3 <= len(acronym) <= 10:
        return None
    return acronym

@dataclass(frozen=True, slots=True)
class AliasEntry:
    alias: str
    expansion: str
    source: QueryExpansionSource
    confidence: float
    target_pt_code: str | None = None
    target_concept_id: str | None = None

class TerminologyAliasIndex:
    """A gold-independent acronym/synonym index with ambiguity preserved."""

    def __init__(self, entries: Iterable[AliasEntry]) -> None:
        by_alias: dict[str, dict[tuple[str, str | None, str | None], AliasEntry]] = {}
        for entry in entries:
            alias = canonical_alias_text(entry.alias)
            if not alias:
                continue
            key = (entry.expansion, entry.target_pt_code, entry.target_concept_id)
            by_alias.setdefault(alias, {})[key] = entry
        self._by_alias = {alias: tuple(sorted(values.values(), key=lambda entry: (entry.expansion, entry.target_pt_code or '', entry.target_concept_id or ''))) for alias, values in by_alias.items()}

    @classmethod
    def from_meddra_reference(cls, reference_path: str | Path, *, local_umls_synonyms_path: str | Path | None=None) -> 'TerminologyAliasIndex':
        reference_path = Path(reference_path)
        entries: list[AliasEntry] = []
        label_targets: dict[str, list[tuple[str, str]]] = {}
        with reference_path.open('r', encoding='utf-8-sig', newline='') as handle:
            reader = csv.DictReader(handle)
            required = {'llt_code', 'llt_name', 'pt_code', 'pt_name'}
            missing = required - set(reader.fieldnames or [])
            if missing:
                raise ValueError(f'The MedDRA reference is missing columns: {sorted(missing)}')
            seen_pts: set[str] = set()
            for row in reader:
                if str(row.get('is_current', 'true')).strip().casefold() in {'false', '0', 'no', 'n'}:
                    continue
                llt_code = str(row['llt_code']).strip()
                llt_name = str(row['llt_name']).strip()
                pt_code = str(row['pt_code']).strip()
                pt_name = str(row['pt_name']).strip()
                label_targets.setdefault(canonical_alias_text(llt_name), []).append((pt_code, f'LLT:{llt_code}'))
                llt_acronym = generated_acronym(llt_name)
                if llt_acronym:
                    entries.append(AliasEntry(alias=llt_acronym, expansion=canonical_alias_text(llt_name), source=QueryExpansionSource.MEDDRA_LLT, confidence=0.94, target_pt_code=pt_code, target_concept_id=f'LLT:{llt_code}'))
                if pt_code not in seen_pts:
                    seen_pts.add(pt_code)
                    label_targets.setdefault(canonical_alias_text(pt_name), []).append((pt_code, None))
                    pt_acronym = generated_acronym(pt_name)
                    if pt_acronym:
                        entries.append(AliasEntry(alias=pt_acronym, expansion=canonical_alias_text(pt_name), source=QueryExpansionSource.MEDDRA_PT, confidence=0.96, target_pt_code=pt_code))
        for alias, expansion in SAFE_BIOMEDICAL_ALIASES.items():
            targets = label_targets.get(canonical_alias_text(expansion), [])
            if targets:
                for pt_code, concept_id in targets:
                    entries.append(AliasEntry(alias=alias, expansion=canonical_alias_text(expansion), source=QueryExpansionSource.BUILTIN_SAFE_RULE, confidence=0.99, target_pt_code=pt_code, target_concept_id=concept_id))
            else:
                entries.append(AliasEntry(alias=alias, expansion=canonical_alias_text(expansion), source=QueryExpansionSource.BUILTIN_SAFE_RULE, confidence=0.97))
        if local_umls_synonyms_path is not None:
            umls_path = Path(local_umls_synonyms_path)
            if umls_path.exists():
                entries.extend(_load_local_umls_aliases(umls_path))
        return cls(entries)

    def entries_for(self, alias: str) -> tuple[AliasEntry, ...]:
        return self._by_alias.get(canonical_alias_text(alias), ())

    def expand_tokens(self, query: str) -> list[tuple[str, AliasEntry, bool]]:
        """Return every plausible expansion; ambiguity is never collapsed."""
        canonical_query = canonical_alias_text(query)
        tokens = canonical_query.split()
        results: list[tuple[str, AliasEntry, bool]] = []
        alias_spans = [canonical_query] + tokens
        for alias in dict.fromkeys(alias_spans):
            entries = self.entries_for(alias)
            if not entries:
                continue
            target_pts = {entry.target_pt_code or entry.expansion for entry in entries}
            ambiguous = len(target_pts) > 1
            for entry in entries:
                if alias == canonical_query:
                    expanded = entry.expansion
                else:
                    expanded = re.sub(f'\\b{re.escape(alias)}\\b', entry.expansion, normalize_term(query).normalized_term)
                results.append((expanded, entry, ambiguous))
        unique: dict[tuple[str, str | None, str | None], tuple[str, AliasEntry, bool]] = {}
        for result in results:
            expanded, entry, _ = result
            unique[expanded, entry.target_pt_code, entry.target_concept_id] = result
        return list(unique.values())

def _load_local_umls_aliases(path: Path) -> list[AliasEntry]:
    entries: list[AliasEntry] = []
    with path.open('r', encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or [])
        if not {'synonym', 'preferred_term'} <= columns:
            raise ValueError('A local UMLS synonym file requires synonym and preferred_term columns.')
        for row in reader:
            synonym = str(row.get('synonym', '')).strip()
            preferred = str(row.get('preferred_term', '')).strip()
            if not synonym or not preferred:
                continue
            entries.append(AliasEntry(alias=synonym, expansion=canonical_alias_text(preferred), source=QueryExpansionSource.LOCAL_UMLS, confidence=float(row.get('confidence') or 0.9), target_pt_code=str(row.get('pt_code') or '').strip() or None, target_concept_id=str(row.get('concept_id') or '').strip() or None))
    return entries

def build_query_variants(original_normalized_query: str, *, alias_index: TerminologyAliasIndex | None=None, positive_components: Iterable[str]=(), max_variants: int=12) -> list[QueryVariant]:
    """Build deterministic query variants without accepting evaluation gold."""
    if max_variants < 1:
        raise ValueError('max_variants must be at least 1.')
    original = normalize_term(original_normalized_query).normalized_term
    if not original:
        raise ValueError('original_normalized_query cannot be empty.')
    pending: list[dict] = [{'expanded_query': original, 'expansion_type': QueryExpansionType.ORIGINAL, 'expansion_source': QueryExpansionSource.INPUT, 'confidence': 1.0, 'ambiguity': False, 'target_pt_code': None, 'target_concept_id': None}]
    for variant, operation in normalization_variants(original_normalized_query):
        normalized_variant = normalize_term(variant).normalized_term
        if normalized_variant != original:
            pending.append({'expanded_query': normalized_variant, 'expansion_type': QueryExpansionType.NORMALIZATION, 'expansion_source': QueryExpansionSource.BUILTIN_SAFE_RULE, 'confidence': 0.98 if 'plural' not in operation else 0.92, 'ambiguity': 'plural' in operation, 'target_pt_code': None, 'target_concept_id': None})
    if alias_index is not None:
        for expanded, entry, ambiguous in alias_index.expand_tokens(original):
            status, context_reason = expansion_context_status(original, entry.expansion, ambiguous=ambiguous)
            pending.append({'expanded_query': expanded, 'expansion_type': QueryExpansionType.UMLS_SYNONYM if entry.source == QueryExpansionSource.LOCAL_UMLS else QueryExpansionType.MEDDRA_ACRONYM if entry.source in {QueryExpansionSource.MEDDRA_PT, QueryExpansionSource.MEDDRA_LLT} else QueryExpansionType.BIOMEDICAL_ABBREVIATION, 'expansion_source': entry.source, 'confidence': entry.confidence, 'ambiguity': status == 'unresolved', 'context_status': status, 'context_reason': context_reason, 'target_pt_code': entry.target_pt_code, 'target_concept_id': entry.target_concept_id})
    for component in positive_components:
        component_text = normalize_term(component).normalized_term
        if component_text and component_text != original:
            pending.append({'expanded_query': component_text, 'expansion_type': QueryExpansionType.COMPONENT, 'expansion_source': QueryExpansionSource.INPUT, 'confidence': 1.0, 'ambiguity': False, 'target_pt_code': None, 'target_concept_id': None})
    deduplicated: list[dict] = []
    seen: set[tuple[str, str | None, str | None]] = set()
    for item in pending:
        key = (item['expanded_query'], item['target_pt_code'], item['target_concept_id'])
        if key not in seen:
            deduplicated.append(item)
            seen.add(key)
    return [QueryVariant(variant_id=f'query_variant_{index:02d}', original_normalized_query=original, **item) for index, item in enumerate(deduplicated[:max_variants], start=1)]
