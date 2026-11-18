"""Shared label-independent interpretation, attribute coverage and identity evidence.

No reference, corpus, label or service I/O. Existing language resources are reused;
unknown residual words are uncertainty, not an invented clinical relation.
"""
from runtime_resources import resource as asp_resource
import copy, re
from functools import lru_cache
from collections import Counter
import clinical_features as clinical
import reference_wording as measurement
import event_comparison as event
import input_discrimination as lexical
import grammatical_discrimination as grammar
import measurement_interpretation as duration
from biomedical_wording import canonical_biomedical_wording
from pt_reranking import SPECIFICITY_MODIFIERS, ETIOLOGY_TOKENS, ANATOMY_GROUPS
from compound_validation import ANATOMY, SITE_EQUIVALENTS
VERSION = '1.20.0'
RESOURCE = 'clinical_features; v1.6 grammar/aliases; v1.7 roles; v1.9 event; v1.10 measurement; v1.19 duration'
DIRECTION_PATTERN = '\\b(?:' + '|'.join(sorted(measurement.FAMILY, key=lambda s: (-len(s), s))) + ')\\b'
RELATION = grammar.RELATIONAL

def canonical(text):
    """Direction morphology, not semantic synonym rewriting or token deletion."""

    def change(m):
        if re.match('\\s+density\\b', text[m.end():], re.I):
            return m[0]
        w = m[0].casefold()
        family = measurement.FAMILY[w]
        return {'increase': 'increased', 'decrease': 'decreased', 'elevation': 'elevated', 'reduction': 'reduced'}.get(family, m[0])
    return re.sub(DIRECTION_PATTERN, change, text, flags=re.I)

def _words(text):
    s = canonical_biomedical_wording(text)
    return [measurement.MORPH.get(w, lexical.GRAMMAR.get(w, w)) for w in re.findall('[a-z]+[0-9]*(?:-[a-z]+)?', s)]

@lru_cache(maxsize=32768)
def _frame(text):
    normalized = canonical(text)
    m = measurement.measurement(normalized)
    d = duration.interpret(normalized)
    c = copy.deepcopy(clinical.interpreted_features(normalized))
    words = _words(normalized)
    recognized = sorted({w for w in words if clinical.LAB_WORDS.fullmatch(w) and w not in {'count', 'concentration', 'level', 'percentage'}})
    c['analytes'] = sorted(set(c['analytes']) | set(m['detected_analytes']) | set(recognized))
    sites = sorted({SITE_EQUIVALENTS.get(w, w) for w in words if w in ANATOMY})
    scopes = [{'text': x[0], 'span': [x.start(), x.end()]} for x in RELATION.finditer(text)]
    uncertainty = [{'text': x[0], 'span': [x.start(), x.end()]} for x in re.finditer('\\b(?:possible|suspected|probable|likely|uncertain|may|might)\\b', text, re.I)]
    kinds = [k for k in d['kind'] if k != 'level']
    lab = bool(m['laboratory'] or recognized or c['analytes'] or clinical.LAB_WORDS.search(' '.join(words)))
    domain = 'duration' if 'duration' in kinds else 'measurement' if lab else 'event_change' if m['direction'] else 'event'
    direction = d['direction'] or c['direction']
    spans = [{'text': x[0], 'span': [x.start(), x.end()], 'canonical': canonical(x[0])} for x in re.finditer(DIRECTION_PATTERN, text, re.I)]
    transitions = lexical.representations(text)['transition']
    unresolved = []
    if scopes:
        unresolved.append('relationship_attachment_requires_preservation')
    if len({x['canonical'] for x in spans}) > 1:
        unresolved.append('multiple_direction_expressions')
    if transitions:
        unresolved.append('state_transition_not_single_direction')
    if uncertainty:
        unresolved.append('uncertain_assertion')
    if c['event_negated'] or c['negated_modifiers']:
        unresolved.append('negated_assertion_or_modifier')
    if m['bounds'] and (not direction):
        unresolved.append('numerical_abnormality_unknown')
    attributes = {'direction': [direction] if direction else [], 'kind': kinds, 'specimen': m['specimen'], 'qualifiers': m['qualifiers'], 'anatomy': sites, 'course': sorted(event.frame(normalized)['course'] and [event.frame(normalized)['course']] or []), 'specificity': sorted(set(words) & (SPECIFICITY_MODIFIERS | lexical.PROTECTED)), 'etiology': sorted(set(words) & ETIOLOGY_TOKENS), 'assertion': ['negated'] if c['event_negated'] else ['uncertain'] if uncertainty else ['asserted']}
    return {'version': VERSION, 'original': text, 'canonical_clinical_text': normalized, 'domain': domain, 'event_or_investigation': m['entity_meaning'], 'original_entity_wording': m['entity_wording'], 'attributes': attributes, 'generic_level_or_value': bool(re.search('\\b(?:levels?|values?)\\b', text, re.I)), 'clinical': c, 'numeric': {'bounds': m['bounds'], 'values': m['numeric_content'], 'units': m['units']}, 'course_frame': event.frame(normalized), 'relations': scopes, 'transition': transitions, 'unresolved': unresolved, 'original_tokens': _words(text), 'direction_attachment': {'kind': domain, 'argument': m['entity_wording'], 'resolved': not unresolved}, 'spans': {'direction': spans, 'uncertainty': uncertainty, 'tokens': [{'text': x[0], 'span': [x.start(), x.end()]} for x in re.finditer('\\S+', text)]}, 'transformations': [{'rule': 'existing_direction_inflection', 'original': text, 'result': normalized, 'source_spans': spans, 'resource': 'discrimination_v1_10.FAMILIES'}] if normalized != text else [], 'resource': RESOURCE}

def interpret(text):
    return copy.deepcopy(_frame(text))

def _numeric(frame):
    n = frame['numeric']
    return {'bounds': n['bounds'], 'values': [x['text'] for x in n['values']], 'units': [x['text'] for x in n['units']]}

@lru_cache(maxsize=131072)
def _compare(text, label):
    a, b = (_frame(text), _frame(label))
    legacy = lexical.support(a['canonical_clinical_text'], b['canonical_clinical_text'])
    dur = duration.compare(a['canonical_clinical_text'], b['canonical_clinical_text'])
    conflicts = set(legacy['conflicts'])
    supported = []
    omitted = []
    added = []
    unresolved = []
    if dur['duration_direction_compatible']:
        conflicts.discard('directionality_conflict')
    for key, x in a['attributes'].items():
        y = b['attributes'][key]
        for value in sorted(set(x) & set(y)):
            supported.append({'attribute': key, 'value': value})
        for value in sorted(set(x) - set(y)):
            omitted.append({'attribute': key, 'value': value})
        for value in sorted(set(y) - set(x)):
            added.append({'attribute': key, 'value': value})
        if x and y and (x != y) and (key in ('direction', 'kind', 'specimen', 'assertion')):
            conflicts.add('explicit_' + key + '_difference')
    numeric_equal = _numeric(a) == _numeric(b)
    if not numeric_equal:
        if any(_numeric(a).values()):
            omitted.append({'attribute': 'numerical_content', 'value': _numeric(a)})
        if any(_numeric(b).values()):
            added.append({'attribute': 'numerical_content', 'value': _numeric(b)})
        if a['numeric']['bounds'] and b['numeric']['bounds']:
            conflicts.add('explicit_numerical_difference')
    same_entity = bool(a['event_or_investigation'] and a['event_or_investigation'] == b['event_or_investigation'])
    level_redundant = bool(same_entity and (a['domain'] == 'measurement' or b['domain'] == 'measurement') and (not a['attributes']['kind']) and (not b['attributes']['kind']))

    def identity_text(t):
        return re.sub('\\b(?:levels?|values?)\\b', '', t, flags=re.I) if level_redundant else t
    ev = event.comparison(a['canonical_clinical_text'], b['canonical_clinical_text'])
    clinical_grammar = grammar.equivalence(identity_text(a['canonical_clinical_text']), identity_text(b['canonical_clinical_text']))
    known = bool(ev['equivalent'] or clinical_grammar['equivalent'] or dur['duration_equivalent'])
    at = Counter(lexical.representations(identity_text(a['canonical_clinical_text']))['meaning_tokens'])
    bt = Counter(lexical.representations(identity_text(b['canonical_clinical_text']))['meaning_tokens'])
    extra = sorted((bt - at).elements())
    lost = sorted((at - bt).elements())
    if not known:
        added.extend(({'attribute': 'unestablished_candidate_word', 'value': w} for w in extra))
        omitted.extend(({'attribute': 'unrepresented_input_word', 'value': w} for w in lost))
    if a['relations'] != b['relations'] and (not known):
        unresolved.append('relationship_or_order_correspondence_unestablished')
    if not known:
        unresolved.append('complete_event_or_investigation_correspondence_unestablished')
    if a['unresolved'] or b['unresolved']:
        unresolved += sorted(set(a['unresolved'] + b['unresolved']))
    if a['generic_level_or_value'] != b['generic_level_or_value'] and (not level_redundant):
        (omitted if a['generic_level_or_value'] else added).append({'attribute': 'level_or_value', 'value': 'unestablished_redundancy'})
    added.extend(({'attribute': 'existing_specificity', 'value': x} for x in legacy['unsupported_specificity']))
    literal = lexical.surface(text) == lexical.surface(label)
    exact_attributes = not omitted and (not added) and (not conflicts)
    adequate = bool(known and exact_attributes and (not unresolved or literal))
    original_grammar = grammar.equivalence(identity_text(text), identity_text(label))
    identity_class = 3 if adequate and literal else 2 if adequate and original_grammar['equivalent'] else 1 if adequate else 0
    transforms = list(a['transformations']) + list(b['transformations'])
    if level_redundant and a['generic_level_or_value'] != b['generic_level_or_value']:
        transforms.append({'rule': 'generic_level_value_for_same_measurement', 'resource': RESOURCE, 'input': text, 'candidate': label})
    if original_grammar['equivalent'] and (not literal):
        transforms.append({'rule': original_grammar['reason'], 'resource': 'grammatical_discrimination_v1_7', 'input': text, 'candidate': label})
    return {'supported': supported, 'omitted': omitted, 'unsupported': added, 'contradictions': sorted(conflicts), 'unresolved': sorted(set(unresolved)), 'adequate_concept': adequate, 'identity_class': identity_class, 'identity_basis': {0: 'not_established', 1: 'clinical_correspondence_only', 2: 'original_bounded_grammar', 3: 'current_literal'}[identity_class], 'transformations': transforms, 'same_investigation': same_entity, 'generic_level_redundant': level_redundant, 'legacy_support': legacy, 'duration_polarity_compatible': dur['duration_direction_compatible'], 'source_span': [0, len(text)], 'candidate_span': [0, len(label)], 'resource': RESOURCE}

def compare(text, label):
    return copy.deepcopy(_compare(text, label))

def support(text, label):
    c = compare(text, label)
    s = copy.deepcopy(c['legacy_support'])
    s.update(conflicts=c['contradictions'], unsupported_specificity=[x['attribute'] + ':' + str(x['value']) for x in c['unsupported']], unrepresented_input_information=[x['attribute'] + ':' + str(x['value']) for x in c['omitted']], source_features=interpret(text)['clinical'], candidate_features=interpret(label)['clinical'], comparison_v1_20=c)
    return s

def prepare_query_variants(text, **kwargs):
    from retrieval_query_expansion import build_retrieval_query_variants
    query = canonical(text)
    return {'original_input': text, 'query_text': query, 'interpretation': interpret(text), 'variants': build_retrieval_query_variants(query, **kwargs), 'retrieval_executed': False}
