"""Shared opt-in typed interpretation. No codes, labels, corpus or service I/O.

The vocabulary comes from the existing v1.10 measurement representation and
v1.6/clinical_features helpers. The extension is duration morphology, not an
investigation/synonym inventory. Missing attributes are not contradictions.
"""
from runtime_resources import resource as asp_resource
import copy
import re
from functools import lru_cache
import reference_wording as previous
from clinical_features import BOUND, preserve_numeric_notation, interpreted_features
from input_discrimination import representations, support as legacy_support
from event_comparison import comparison
VERSION = '1.19.0'
DURATION = '\\b(?:duration|intervals?|time|prolongations?)\\b'
LONG = '\\b(?:prolong(?:s|ed|ing|ation|ations)?|lengthen(?:s|ed|ing)?)\\b'
SHORT = '\\b(?:shorten(?:s|ed|ing)?|shortening)\\b'
RELATION = '\\b(?:and|or|with|without|after|before|following|due|from|to|during)\\b|[;,]'

def spans(text, pattern):
    return [{'text': m[0], 'span': [m.start(), m.end()], 'surface': 'original'} for m in re.finditer(pattern, text, re.I)]

@lru_cache(maxsize=32768)
def _frame(text):
    old = previous.measurement(text)
    raw = preserve_numeric_notation(text)
    duration = bool(re.search(DURATION, raw, re.I))
    directions = {old['direction']} if old['direction'] else set()
    if duration:
        if re.search(LONG, raw, re.I):
            directions.add('increased')
        if re.search(SHORT, raw, re.I):
            directions.add('decreased')
    kinds = set(old['measurement_types'])
    if duration:
        kinds.discard('time')
        kinds.add('duration')
    if old['generic_level_wording']:
        kinds.add('level')
    entity = old['entity_wording']
    if duration:
        entity = re.sub(DURATION + '|' + LONG + '|' + SHORT, ' ', entity, flags=re.I)
        entity = ' '.join(entity.split())
    numeric = [{**m, 'notation_surface': raw} for m in spans(raw, BOUND.pattern)]
    uncertainty = []
    if len(directions) > 1:
        uncertainty.append('multiple_directions')
    if len(kinds) > 1:
        uncertainty.append('multiple_measurement_kinds')
    if re.search(RELATION, raw, re.I):
        uncertainty.append('relational_or_multiple_scope')
    if old['negated'] or old['uncertainty']:
        uncertainty.append('negated_or_uncertain_assertion')
    if old['bounds'] and (not old['direction']):
        uncertainty.append('numerical_abnormality_not_established')
    return {'version': VERSION, 'original': text, 'investigation': entity, 'kind': sorted(kinds), 'specimen': old['specimen'], 'qualifiers': old['qualifiers'], 'direction': next(iter(directions)) if len(directions) == 1 else None, 'bounds': old['bounds'], 'numeric_evidence': numeric, 'units': old['units'], 'numeric_content': old['numeric_content'], 'course': old['course'], 'negated': old['negated'], 'uncertainty': uncertainty, 'supporting_spans': {'duration': spans(text, DURATION), 'direction': spans(text, LONG + '|' + SHORT + '|' + '\\b(?:' + '|'.join(previous.FAMILY) + ')\\b'), 'specimen': spans(text, '\\b(?:' + '|'.join(previous.SPECIMEN) + ')\\b'), 'qualifiers': spans(text, '\\b(?:' + '|'.join(previous.QUALIFIERS) + ')\\b')}, 'legacy_frame': old, 'resource': 'discrimination_v1_10.measurement; clinical_features.BOUND; bounded duration morphology'}

def interpret(text):
    return copy.deepcopy(_frame(text))

def prepare_retrieval(text):
    """Input contract for opt-in retrieval preparation; never removes wording.

    This task does not invoke retrieval or claim changed candidate availability.
    """
    return {'original_input': text, 'query_text': text, 'measurement': interpret(text), 'query_transformations': [], 'retrieval_executed': False}

def compare(text, label):
    a, b = (_frame(text), _frame(label))
    s = legacy_support(text, label)
    conflicts = list(s['conflicts'])
    omitted = []
    added = []
    facts = []
    same_entity = bool(a['investigation'] and a['investigation'] == b['investigation'])
    duration = bool(same_entity and a['kind'] == b['kind'] == ['duration'])
    equal_fields = True
    for field in ('kind', 'specimen', 'qualifiers', 'direction', 'course', 'bounds'):
        x, y = (a[field], b[field])
        if x and x == y:
            facts.append(field)
        elif x and (not y):
            omitted.append(field)
            equal_fields = False
        elif y and (not x):
            added.append(field)
            equal_fields = False
        elif x != y:
            equal_fields = False
            if field in ('direction', 'specimen', 'kind', 'qualifiers', 'bounds'):
                conflicts.append('measurement_' + field + '_conflict')
    for field in ('units', 'numeric_content'):
        if [x['text'] for x in a[field]] != [x['text'] for x in b[field]]:
            equal_fields = False
            omitted.append(field + '_not_preserved')
    transformation = bool(duration and equal_fields and a['direction'] and (not a['uncertainty']) and (not b['uncertainty']))
    direction_compatible = bool(a['kind'] == b['kind'] == ['duration'] and a['direction'] and (a['direction'] == b['direction']) and (not a['uncertainty']) and (not b['uncertainty']))
    if direction_compatible:
        conflicts = [x for x in conflicts if x != 'directionality_conflict']
    clinical = not conflicts and (not s['unsupported_specificity']) and (not added)
    supported = bool(transformation and clinical)
    return {'input': text, 'candidate_label': label, 'input_frame': a, 'candidate_frame': b, 'supported_facts': facts + (['same_investigation'] if same_entity else []), 'explicit_contradictions': sorted(set(conflicts)), 'unrepresented_input_information': sorted(set(omitted + s['unrepresented_input_information'])), 'unsupported_candidate_specificity': sorted(set(added + s['unsupported_specificity'])), 'transformations': [{'rule': 'same_investigation_duration_direction', 'input_spans': a['supporting_spans'], 'candidate_spans': b['supporting_spans'], 'resource': a['resource']}] if transformation else [], 'duration_equivalent': supported, 'duration_direction_compatible': direction_compatible, 'same_investigation': same_entity, 'remaining_identity_uncertainty': 'clinical duration equivalence alone does not select among multiple current identities', 'legacy_support': s}

def support(text, label):
    s = copy.deepcopy(legacy_support(text, label))
    c = compare(text, label)
    if c['duration_direction_compatible']:
        s['conflicts'] = [x for x in s['conflicts'] if x != 'directionality_conflict']
    s['measurement_v1_19'] = c
    return s

def explain(text, label, measurement=True):
    c = compare(text, label)
    legacy = c['legacy_support']
    e = comparison(text, label)
    return {'supported_facts': c['supported_facts'], 'explicit_contradictions': c['explicit_contradictions'] if measurement else legacy['conflicts'], 'unrepresented_input_information': c['unrepresented_input_information'], 'unsupported_candidate_specificity': c['unsupported_candidate_specificity'] if measurement else legacy['unsupported_specificity'], 'meaning_preserving_transformations': c['transformations'] if measurement else [], 'duration_polarity_correction': bool(measurement and c['duration_direction_compatible'] and ('directionality_conflict' in legacy['conflicts'])), 'investigation_identity_established': c['same_investigation'], 'duration_direction_evidence': {'input': c['input_frame']['supporting_spans'], 'candidate': c['candidate_frame']['supporting_spans']} if c['duration_direction_compatible'] else None, 'existing_event_correspondence': e['equivalent'], 'positive_complete_correspondence': bool(c['duration_equivalent'] if measurement and c['transformations'] else e['equivalent']), 'strict_identity_uncertainty': c['remaining_identity_uncertainty'], 'input_span': [0, len(text)], 'source': text, 'label': label, 'resource': 'existing v1.9 comparison and v1.10 measurement; v1.19 duration extension'}

def prepare_query_variants(text, **kwargs):
    """Opt-in retrieval preparation using unchanged existing query generation.

    Return typed original/variant evidence beside QueryVariant objects; callers
    need not infer clinical support from an expansion's text or confidence.
    No search, model call, annotation wording or candidate expansion occurs here.
    """
    from retrieval_query_expansion import build_retrieval_query_variants
    variants = build_retrieval_query_variants(text, **kwargs)
    return {**prepare_retrieval(text), 'variants': variants, 'variant_measurements': {v.variant_id: interpret(v.expanded_query) for v in variants}}
