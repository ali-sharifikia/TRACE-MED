from __future__ import annotations
from runtime_resources import resource as asp_resource
from clinical_features import clinical_scoring_text, interpreted_features, feature_conflicts, event_is_negated, automatic_mapping_uncertainties
import re
from difflib import SequenceMatcher
from typing import Any
from schemas import AggregatedCandidate, ReviewerInput
from terminology_aliases import SAFE_BIOMEDICAL_ALIASES
from biomedical_wording import canonical_biomedical_wording, generic_context_base
PT_RERANKER_VERSION = '2.2.3'
PT_RERANKING_WEIGHTS = {'evidence_strength': 0.46, 'label_alignment': 0.32, 'component_coverage': 0.14, 'query_alignment': 0.0, 'source_diversity': 0.08}
PT_RERANKER_WEIGHTS = PT_RERANKING_WEIGHTS
PT_RERANKER_CONFLICT_PENALTY_PER_FLAG = 0.12
PT_RERANKER_MAX_CONFLICT_PENALTY = 0.6
TOKEN_EQUIVALENTS = {'elevation': 'increased', 'elevations': 'increased', 'elevated': 'increased', 'increase': 'increased', 'increases': 'increased', 'raised': 'increased', 'above': 'increased', 'high': 'increased', 'excess': 'increased', 'reduction': 'decreased', 'reductions': 'decreased', 'reduced': 'decreased', 'decrease': 'decreased', 'decreases': 'decreased', 'below': 'decreased', 'low': 'decreased', 'prolonging': 'prolonged', 'prolongation': 'prolonged', 'prolongations': 'prolonged', 'lengthened': 'prolonged', 'shortening': 'shortened', 'abnormalities': 'abnormal', 'reactions': 'reaction', 'events': 'event', 'infections': 'infection', 'rates': 'rate', 'counts': 'count', 'values': 'value', 'levels': 'level', 'males': 'male', 'females': 'female', 'mycotic': 'fungal', 'multiorgan': 'systemic', 'thrombotic': 'thrombosis', 'thromboses': 'thrombosis', 'dystrophies': 'dystrophy', 'dyskinesia': 'dysfunction', 'oropharyngeal': 'pharyngeal', 'haematocrit': 'hematocrit', 'haemoglobin': 'hemoglobin', 'haemorrhage': 'hemorrhage', 'oedema': 'edema', 'anaemia': 'anemia', 'tumour': 'tumor', 'tumours': 'tumor', 'sgot': 'aspartateaminotransferase', 'sgpt': 'alanineaminotransferase', 'transaminase': 'aminotransferase', 'no': 'negated', 'not': 'negated', 'without': 'negated', 'absent': 'negated'}
PHRASE_EQUIVALENTS = (('\\b(?:increase|increases|increased|elevation|elevations|elevated)[\\s-]+in[\\s-]+(?:the[\\s-]+)?qt(?:c)?(?:[\\s-]+interval)?\\b', ' qt prolonged '), ('\\bqt(?:c)?(?:[\\s-]+interval)?[\\s-]+(?:increase|increases|increased|elevation|elevations|elevated)\\b', ' qt prolonged '), ('\\bnon[\\s-]+high[\\s-]+density[\\s-]+lipoprotein(?:[\\s-]+cholesterol)?\\b', ' nonhdlcholesterol '), ('\\bnon[\\s-]*hdl[\\s-]*c\\b', ' nonhdlcholesterol '), ('\\bhigh[\\s-]+density[\\s-]+lipoprotein(?:[\\s-]+cholesterol)?\\b', ' hdlcholesterol '), ('\\bhdl[\\s-]*c\\b', ' hdlcholesterol '), ('\\blow[\\s-]+density[\\s-]+lipoprotein(?:[\\s-]+cholesterol)?\\b', ' ldlcholesterol '), ('\\bldl[\\s-]*c\\b', ' ldlcholesterol '), ('\\bpre[\\s-]+hypertension\\b', ' prehypertension '), ('\\bast\\b', ' aspartate aminotransferase '), ('\\balt\\b', ' alanine aminotransferase '), ('\\bqtc\\b', ' qt '))
IGNORED_TOKENS = {'a', 'an', 'and', 'for', 'in', 'of', 'on', 'or', 'the', 'to', 'with', 'nos', 'unspecified'}
CONFLICT_FIELDS = ('directionality_conflict', 'negation_conflict', 'laterality_conflict', 'temporal_conflict', 'severity_conflict', 'anatomical_conflict')
MODIFIER_FAMILIES = (frozenset({'increased', 'decreased', 'normal'}), frozenset({'positive', 'negative'}), frozenset({'prolonged', 'shortened'}), frozenset({'left', 'right', 'bilateral'}), frozenset({'mild', 'moderate', 'severe'}), frozenset({'new', 'acute', 'chronic', 'recurrent', 'persistent', 'intermittent'}), frozenset({'negated', 'present'}))
SPECIFICITY_MODIFIERS = frozenset({'hereditary', 'congenital', 'familial', 'primary', 'secondary', 'systemic', 'localized', 'disseminated', 'recurrent', 'severe', 'moderate', 'mild', 'failure'})
ANALYTE_PATTERNS = {'alanine_aminotransferase': (re.compile('\\b(?:alt|sgpt)\\b'), re.compile('\\balanine\\s+(?:amino)?trans(?:ferase|aminase)\\b')), 'aspartate_aminotransferase': (re.compile('\\b(?:ast|sgot)\\b'), re.compile('\\baspartate\\s+(?:amino)?trans(?:ferase|aminase)\\b')), 'hdl_cholesterol': (re.compile('\\bhdl(?:\\s*c)?\\b'), re.compile('\\bhigh\\s+density\\s+lipoprotein(?:\\s+cholesterol)?\\b')), 'ldl_cholesterol': (re.compile('\\bldl(?:\\s*c)?\\b'), re.compile('\\blow\\s+density\\s+lipoprotein(?:\\s+cholesterol)?\\b')), 'hemoglobin': (re.compile('\\bha?emoglobin\\b'),), 'hematocrit': (re.compile('\\bha?ematocrit\\b'),), 'creatine_kinase': (re.compile('\\b(?:ck|cpk)\\b'), re.compile('\\bcreatine\\s+(?:kinase|phosphokinase)\\b'))}
ANATOMY_GROUPS = (frozenset({'face', 'facial'}), frozenset({'tongue', 'lingual'}), frozenset({'eye', 'ocular', 'ophthalmic'}), frozenset({'ear', 'aural', 'otic'}), frozenset({'arm', 'upperlimb'}), frozenset({'leg', 'lowerlimb'}), frozenset({'kidney', 'renal'}), frozenset({'liver', 'hepatic'}), frozenset({'lung', 'pulmonary'}), frozenset({'heart', 'cardiac'}), frozenset({'infusion', 'injection', 'implant', 'catheter'}))
ETIOLOGY_TOKENS = frozenset({'drug', 'induced', 'infectious', 'postoperative', 'transplantation', 'radiation', 'traumatic', 'allergic', 'autoimmune', 'idiopathic'})

def _detected_analytes(value: str) -> set[str]:
    normalized = interpreted_features(value)['retrieval_text'].replace('-', ' ')
    return {analyte for analyte, patterns in ANALYTE_PATTERNS.items() if any((pattern.search(normalized) for pattern in patterns))} | set(interpreted_features(value)['analytes'])

def _anatomy_group_indices(tokens: set[str]) -> set[int]:
    return {index for index, group in enumerate(ANATOMY_GROUPS) if tokens & group}

def _normalize_text(value: str) -> str:
    normalized = canonical_biomedical_wording(clinical_scoring_text(generic_context_base(value) or value))
    for pattern, replacement in PHRASE_EQUIVALENTS:
        normalized = re.sub(pattern, replacement, normalized)
    for alias, expansion in sorted(SAFE_BIOMEDICAL_ALIASES.items(), key=lambda item: -len(item[0])):
        normalized = re.sub(f'\\b{re.escape(alias)}\\b', f' {expansion} ', normalized)
    normalized = re.sub('\\b(?:upper|lower)\\s+limit\\s+of\\s+normal\\b', ' ', normalized)
    return normalized

def _tokens(value: str) -> list[str]:
    tokens = re.findall('[a-z0-9]+', _normalize_text(value))
    return [TOKEN_EQUIVALENTS.get(token, token) for token in tokens if token not in IGNORED_TOKENS and (not token.isdigit())]

def _text_similarity(first_value: str, second_value: str) -> float:
    first_tokens = _tokens(first_value)
    second_tokens = _tokens(second_value)
    first_set = set(first_tokens)
    second_set = set(second_tokens)
    if not first_set or not second_set:
        return 0.0
    intersection_count = len(first_set & second_set)
    precision = intersection_count / len(second_set)
    recall = intersection_count / len(first_set)
    if precision + recall:
        token_f1 = 2.0 * precision * recall / (precision + recall)
    else:
        token_f1 = 0.0
    sequence_similarity = SequenceMatcher(None, ' '.join(sorted(first_tokens)), ' '.join(sorted(second_tokens))).ratio()
    containment = float(first_set <= second_set or second_set <= first_set)
    return min(1.0, 0.55 * token_f1 + 0.3 * sequence_similarity + 0.15 * containment)

def _query_alignment(component_text: str, candidate: AggregatedCandidate) -> float:
    query_terms = [query_term for query_term in (candidate.fuzzy_query_term, candidate.semantic_query_term, candidate.umls_query_term) if query_term is not None]
    if not query_terms:
        return 0.0
    return max((_text_similarity(component_text, query_term) for query_term in query_terms))

def _clinical_modifier_penalty(component_text: str, candidate_text: str) -> tuple[float, list[str]]:
    """
    Compare explicit clinical modifiers independently
    of retrieval source.

    This catches conflicts in semantic-only and UMLS
    candidates, which do not carry fuzzy conflict
    flags.
    """
    component_tokens = set(_tokens(component_text))
    candidate_tokens = set(_tokens(candidate_text))
    penalty = 0.0
    reasons = []
    for modifier_family in MODIFIER_FAMILIES:
        component_modifiers = component_tokens & modifier_family
        candidate_modifiers = candidate_tokens & modifier_family
        if component_modifiers and candidate_modifiers and component_modifiers.isdisjoint(candidate_modifiers):
            penalty += 0.3
            reasons.append(f'conflicting modifiers: {sorted(component_modifiers)} vs {sorted(candidate_modifiers)}')
        elif component_modifiers and (not candidate_modifiers):
            penalty += 0.07
            reasons.append(f'candidate omits modifier: {sorted(component_modifiers)}')
        elif not component_modifiers and candidate_modifiers:
            if automatic_mapping_uncertainties(component_text) and candidate_modifiers <= {'increased', 'decreased', 'normal'}:
                continue
            penalty += 0.08
            reasons.append(f'candidate adds modifier: {sorted(candidate_modifiers)}')
    already_scored_modifiers = set().union(*MODIFIER_FAMILIES)
    unsupported_specificity = candidate_tokens & SPECIFICITY_MODIFIERS - component_tokens - already_scored_modifiers
    if unsupported_specificity:
        penalty += min(0.2, 0.1 * len(unsupported_specificity))
        reasons.append(f'candidate adds specificity: {sorted(unsupported_specificity)}')
    component_analytes = _detected_analytes(component_text)
    candidate_analytes = _detected_analytes(candidate_text)
    added_analytes = candidate_analytes - component_analytes
    missing_analytes = component_analytes - candidate_analytes
    if added_analytes:
        penalty += min(0.35, 0.22 * len(added_analytes))
        reasons.append(f'candidate adds analyte: {sorted(added_analytes)}')
    if missing_analytes:
        penalty += min(0.24, 0.16 * len(missing_analytes))
        reasons.append(f'candidate omits analyte: {sorted(missing_analytes)}')
    component_anatomy = _anatomy_group_indices(component_tokens)
    candidate_anatomy = _anatomy_group_indices(candidate_tokens)
    if component_anatomy and candidate_anatomy and component_anatomy.isdisjoint(candidate_anatomy):
        penalty += 0.25
        reasons.append('candidate conflicts with the asserted anatomical site')
    elif candidate_anatomy - component_anatomy:
        penalty += 0.1
        reasons.append('candidate adds an unsupported anatomical/procedural site')
    added_etiology = (candidate_tokens & ETIOLOGY_TOKENS) - component_tokens - unsupported_specificity
    if added_etiology:
        penalty += min(0.2, 0.1 * len(added_etiology))
        reasons.append(f'candidate adds etiology/cause: {sorted(added_etiology)}')
    modifier_tokens = set().union(*MODIFIER_FAMILIES)
    substantive_missing = component_tokens - candidate_tokens - modifier_tokens
    if len(substantive_missing) >= 2:
        penalty += min(0.16, 0.04 * len(substantive_missing))
        reasons.append(f'candidate misses input components: {sorted(substantive_missing)}')
    for conflict in feature_conflicts(component_text, candidate_text):
        if conflict not in {'analyte_conflict', 'directionality_conflict'}:
            penalty += 0.3
            reasons.append(conflict)
    return (min(0.6, penalty), reasons)

def _conflict_penalty(component_text: str, candidate: AggregatedCandidate) -> tuple[float, float, float, list[str]]:
    evidence = candidate.fuzzy_evidence
    fuzzy_conflict_penalty = 0.0
    if evidence is not None:
        conflict_count = sum((bool(getattr(evidence, field_name, False)) for field_name in CONFLICT_FIELDS))
        fuzzy_conflict_penalty = min(0.42, 0.12 * conflict_count)
    modifier_penalty, modifier_conflict_reasons = _clinical_modifier_penalty(component_text, f'{candidate.llt_name} {candidate.pt_name}')
    canonical_conflicts = explicit_clinical_conflicts(component_text, f'{candidate.llt_name} {candidate.pt_name}')
    separate_features = set(feature_conflicts(component_text, candidate.llt_name)) | set(feature_conflicts(component_text, candidate.pt_name))
    combined_features = set(feature_conflicts(component_text, f'{candidate.llt_name} {candidate.pt_name}'))
    for conflict in sorted(separate_features - combined_features):
        if conflict not in {'analyte_conflict', 'directionality_conflict'}:
            modifier_penalty = min(0.6, modifier_penalty + 0.3)
            modifier_conflict_reasons.append(conflict)
    canonical_conflicts = set(canonical_conflicts) | set(explicit_clinical_conflicts(component_text, candidate.llt_name)) | set(explicit_clinical_conflicts(component_text, candidate.pt_name))
    if evidence is not None:
        uncovered_flags = {field for field in CONFLICT_FIELDS if getattr(evidence, field, False) and field not in canonical_conflicts}
        fuzzy_conflict_penalty = min(0.42, 0.12 * len(uncovered_flags))
    total_penalty = min(PT_RERANKER_MAX_CONFLICT_PENALTY, fuzzy_conflict_penalty + modifier_penalty)
    return (total_penalty, fuzzy_conflict_penalty, modifier_penalty, modifier_conflict_reasons)

def _score_candidate(component_text: str, candidate: AggregatedCandidate) -> dict[str, Any]:
    fuzzy_evidence = candidate.fuzzy_evidence
    semantic_evidence = candidate.semantic_evidence
    fuzzy_strength = float(fuzzy_evidence.ranking_score) / 100.0 if fuzzy_evidence is not None else 0.0
    semantic_strength = float(semantic_evidence.cosine_similarity) if semantic_evidence is not None else 0.0
    umls_strength = 0.95 if candidate.umls_evidence is not None else 0.0
    llt_alignment = _text_similarity(component_text, candidate.llt_name)
    pt_alignment = _text_similarity(component_text, candidate.pt_name)
    label_alignment = max(llt_alignment, 0.95 * pt_alignment)
    fuzzy_coverage = float(fuzzy_evidence.input_token_coverage) if fuzzy_evidence is not None else 0.0
    component_coverage = max(llt_alignment, fuzzy_coverage)
    query_alignment = _query_alignment(component_text, candidate)
    evidence_strength = max(fuzzy_strength, semantic_strength, umls_strength)
    source_diversity = min(candidate.source_count, 3) / 3.0
    penalty, fuzzy_conflict_penalty, modifier_conflict_penalty, modifier_conflict_reasons = _conflict_penalty(component_text, candidate)
    component_token_set = set(_tokens(component_text))
    candidate_text = f'{candidate.llt_name} {candidate.pt_name}'
    candidate_token_set = set(_tokens(candidate_text))
    component_analytes = _detected_analytes(component_text)
    candidate_analytes = _detected_analytes(candidate_text)
    score = PT_RERANKING_WEIGHTS['evidence_strength'] * evidence_strength + PT_RERANKING_WEIGHTS['label_alignment'] * label_alignment + PT_RERANKING_WEIGHTS['component_coverage'] * component_coverage + PT_RERANKING_WEIGHTS['query_alignment'] * query_alignment + PT_RERANKING_WEIGHTS['source_diversity'] * source_diversity - penalty
    score = max(0.0, min(1.0, score))
    return {'candidate_id': candidate.candidate_id, 'score': round(score, 6), 'unrounded_score': score, 'fuzzy_strength': round(fuzzy_strength, 6), 'semantic_strength': round(semantic_strength, 6), 'umls_strength': round(umls_strength, 6), 'label_alignment': round(label_alignment, 6), 'component_coverage': round(component_coverage, 6), 'query_alignment': round(query_alignment, 6), 'source_diversity': round(source_diversity, 6), 'conflict_penalty': round(penalty, 6), 'fuzzy_conflict_penalty': round(fuzzy_conflict_penalty, 6), 'modifier_conflict_penalty': round(modifier_conflict_penalty, 6), 'modifier_conflict_reasons': modifier_conflict_reasons, 'interpreted_features': interpreted_features(component_text), 'explicit_clinical_conflicts': sorted(set(explicit_clinical_conflicts(component_text, candidate.llt_name)) | set(explicit_clinical_conflicts(component_text, candidate.pt_name))), 'automatic_mapping_uncertainties': automatic_mapping_uncertainties(component_text), 'missing_input_tokens': sorted(component_token_set - candidate_token_set), 'added_candidate_tokens': sorted(candidate_token_set - component_token_set), 'input_analytes': sorted(component_analytes), 'candidate_analytes': sorted(candidate_analytes), 'added_analytes': sorted(candidate_analytes - component_analytes), 'missing_analytes': sorted(component_analytes - candidate_analytes)}

def explicit_clinical_conflicts(input_text: str, candidate_text: str) -> tuple[str, ...]:
    """Distinct protected contradictions; ordinary missing tokens stay soft."""
    first, second = (set(_tokens(input_text)), set(_tokens(candidate_text)))
    families = {'directionality_conflict': {'increased', 'decreased', 'normal', 'prolonged', 'shortened'}, 'laterality_conflict': {'left', 'right', 'bilateral'}, 'temporal_conflict': {'new', 'acute', 'chronic', 'recurrent', 'persistent', 'intermittent'}, 'severity_conflict': {'mild', 'moderate', 'severe'}}
    conflicts = set()
    for label, family in families.items():
        a, b = (first & family, second & family)
        if a and b and a.isdisjoint(b):
            conflicts.add(label)
    if event_is_negated(input_text) != event_is_negated(candidate_text):
        conflicts.add('negation_conflict')
    a, b = (first & {'positive', 'negative'}, second & {'positive', 'negative'})
    if a and b and a.isdisjoint(b):
        conflicts.add('negation_conflict')
    a, b = (_anatomy_group_indices(first), _anatomy_group_indices(second))
    if a and b and a.isdisjoint(b):
        conflicts.add('anatomical_conflict')
    a, b = (_detected_analytes(input_text), _detected_analytes(candidate_text))
    if a and b and (a != b):
        conflicts.add('analyte_conflict')
    if (second & ETIOLOGY_TOKENS) - first:
        conflicts.add('etiology_conflict')
    conflicts.update(feature_conflicts(input_text, candidate_text))
    return tuple(sorted(conflicts))

def _rank_component(*, component_id: str, component_text: str, reviewer_input: ReviewerInput) -> dict[str, Any]:
    aggregation_result = reviewer_input.aggregation_result
    candidate_by_id = {candidate.candidate_id: candidate for candidate in aggregation_result.candidates}
    scored_groups = []
    for group in aggregation_result.pt_groups:
        candidate_scores = [_score_candidate(component_text, candidate_by_id[candidate_id]) for candidate_id in group.candidate_ids]
        candidate_scores.sort(key=lambda item: (-item['score'], item['candidate_id']))
        best_candidate = candidate_scores[0]
        scored_groups.append({'pt_code': group.pt_code, 'pt_name': group.pt_name, 'candidate_ids': list(group.candidate_ids), 'best_candidate_id': best_candidate['candidate_id'], 'reranking_score': best_candidate['score'], 'unrounded_reranking_score': best_candidate['unrounded_score'], 'score_breakdown': {key: value for key, value in best_candidate.items() if key not in {'candidate_id', 'score'}}})
    scored_groups.sort(key=lambda item: (-item['reranking_score'], item['pt_code']))
    previous_score = None
    for rank, group in enumerate(scored_groups, start=1):
        current_score = group['reranking_score']
        group['rank'] = rank
        group['score_gap_from_previous'] = None if previous_score is None else round(previous_score - current_score, 6)
        previous_score = current_score
    return {'component_id': component_id, 'component_text': component_text, 'pt_rankings': scored_groups}

def build_component_pt_rankings(reviewer_input: ReviewerInput) -> list[dict[str, Any]]:
    """
    Rank the retrieved PT groups separately for
    every positively asserted input component.

    The score is a deterministic retrieval-priority
    score. It is not a clinical confidence
    probability and does not automatically select
    a PT.
    """
    positive_components = [component for component in reviewer_input.input_interpretation.asserted_components if not component.is_negated]
    from compound_validation import assess_decomposition
    scope = assess_decomposition(reviewer_input.normalized_term, positive_components)
    effective = {c['component_id']: c['effective_text'] for c in scope['components']}
    return [_rank_component(component_id=component.component_id, component_text=effective.get(component.component_id, component.component_text), reviewer_input=reviewer_input) for component in positive_components]
