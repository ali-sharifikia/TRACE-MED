"""Original-text clinical specificity observations; no labels or code preferences.

Absence from a label is unresolved, not proof of absence from its concept.
This is a veto/evidence layer, never positive semantic acceptance authority.
"""
from __future__ import annotations
from runtime_resources import resource as asp_resource
import re
from clinical_features import LAB_WORDS, BOUND, NEGATABLE_MODIFIERS
from compound_validation import ANATOMY
from reference_wording import SPECIMEN, FAMILY, DIRECTION
VERSION = '1.34.0'

def _norm(text):
    return re.sub('\\s+', ' ', text.casefold().strip())

def interpret(text):
    attributes, uncertainty = ([], [])

    def add(kind, match, value=None, rule=None, **extra):
        attributes.append(dict(attribute=kind, value=value or _norm(match[0]), text=match[0], span=[match.start(), match.end()], evidence='explicit_source_text', rule=rule or kind, **extra))
    patterns = {'specimen': '\\b(?:' + '|'.join(sorted(SPECIMEN)) + ')\\b', 'fraction': '\\b(?:free|total|ionized|conjugated|unconjugated)\\b', 'fasting': '\\b(?:non[- ]?fasting|fasting|postprandial)\\b', 'measurement_kind': '\\b(?:count|concentration|activity|percentage|proportion|duration)\\b|%', 'anatomy': '\\b(?:' + '|'.join(sorted(ANATOMY | {'intracranial', 'pineal'})) + ')\\b', 'laterality': '\\b(?:left|right|bilateral|unilateral)\\b', 'extent': '\\b(?:localized|localised|local|disseminated|diffuse|generalized|generalised|invasive)\\b', 'position': '\\b(?:sitting|standing|supine|prone|upright|recumbent)\\b', 'behaviour': '\\b(?:benign|malignant|malignancies|malignancy|in situ|uncertain behaviour|uncertain behavior)\\b', 'primary_status': '\\b(?:second[- ]primary|primary|secondary|metastatic)\\b', 'cell_subset': '\\b(?:CD\\d+[+-]?|B[- ]cell|T[- ]cell|NK[- ]cell)\\b', 'course': '\\b(?:transient|persistent|recurrent|acute|chronic)\\b', 'severity': '\\b(?:mild|moderate|severe)\\b', 'cell_measure': '\\babsolute\\b', 'structure': '\\b(?:cystic|cyst|cysts|invasion|infiltration)\\b'}
    exclusion_words = NEGATABLE_MODIFIERS | {'infective', 'melanoma'}
    exclusions = list(re.finditer('\\bnon[-\\s]?(' + '|'.join(sorted(exclusion_words, key=len, reverse=True)) + ')\\b', text, re.I))
    for m in exclusions:
        tail = re.match('\\s+([^,;]+)', text[m.end():])
        add('exclusion', m, m[1].casefold(), 'bounded_non_modifier', scope_span=[m.end() + tail.start(1), m.end() + tail.end(1)] if tail else None)
        if not tail:
            uncertainty.append('exclusion_attachment_unknown')
    for kind, pattern in patterns.items():
        for m in re.finditer(pattern, text, re.I):
            add(kind, m)
    for m in re.finditer('\\b(?:no|not|without|excluding|except)\\b', text, re.I):
        add('negation_or_exclusion_scope', m, scope=text[m.end():])
        uncertainty.append('negation_or_exclusion_attachment_requires_full_expression')
    for m in re.finditer('\\b(?:invasion|infiltration)\\s+(beyond|into|of|within)\\s+(?:the\\s+)?([\\w -]+?)(?=[,;.]|$)', text, re.I):
        add('relationship', m, _norm(m[0]), 'event_relation_target', relation=m[1].casefold(), target=_norm(m[2]))
    if re.search('\\b(?:invasion|infiltration)\\b', text, re.I) and (not any((a['attribute'] == 'relationship' for a in attributes))):
        uncertainty.append('invasion_attachment_unresolved')
    for m in re.finditer('\\b(?:possible|suspected|probable|may|might|uncertain)\\b', text, re.I):
        add('assertion_uncertainty', m)
    for m in re.finditer('\\b(?:' + '|'.join(sorted(FAMILY, key=len, reverse=True)) + ')\\b', text, re.I):
        if re.match('\\s+density\\b', text[m.end():], re.I):
            uncertainty.append('direction_like_entity_token_not_a_result')
            continue
        add('direction', m, DIRECTION[FAMILY[m[0].casefold()]], 'existing_measurement_direction')
    for m in LAB_WORDS.finditer(text):
        if m[0].casefold() not in {'count', 'concentration', 'level', 'percentage'}:
            add('analyte', m, rule='clinical_features.LAB_WORDS')
    for m in BOUND.finditer(text):
        add('numerical_bound', m)
    for m in re.finditer('\\b(?:mg|g|mmol|mol|iu|u)\\s*/\\s*(?:dl|ml|l)\\b|/\\s*(?:ul|ml|l)\\b', text, re.I):
        add('unit', m)
    for m in re.finditer('(?<!\\w)\\d+(?:\\.\\d+)?', text):
        if not any((a['span'][0] <= m.start() < a['span'][1] for a in attributes)):
            add('number', m)
    attributes.sort(key=lambda a: (a['span'], a['attribute']))
    return dict(version=VERSION, original=text, attributes=attributes, unresolved=sorted(set(uncertainty)), limits=['unrecognized analytes and implicit terminology relationships are not inferred'])

def compare(original, candidate):
    a, b = (interpret(original), interpret(candidate))
    preserved, unrepresented, added, conflicts = ([], [], [], [])

    def grouped(frame):
        out = {}
        for item in frame['attributes']:
            out.setdefault(item['attribute'], {}).setdefault(item['value'], []).append(item)
        return out
    x, y = (grouped(a), grouped(b))
    for key in sorted(x.keys() | y.keys()):
        av, bv = (x.get(key, {}), y.get(key, {}))
        for value in sorted(av.keys() & bv.keys()):
            preserved.append(dict(attribute=key, value=value, input=av[value], candidate=bv[value]))
        for value in sorted(av.keys() - bv.keys()):
            unrepresented.append(dict(attribute=key, value=value, evidence=av[value], status='not_explicitly_represented; implicit meaning unverified'))
        for value in sorted(bv.keys() - av.keys()):
            added.append(dict(attribute=key, value=value, evidence=bv[value], status='not_supported_by_original_text'))
        if av and bv and (av.keys() != bv.keys()) and (key in {'direction', 'fraction', 'fasting', 'measurement_kind', 'laterality', 'primary_status', 'position', 'cell_subset', 'numerical_bound', 'number', 'relationship', 'negation_or_exclusion_scope'}):
            conflicts.append(dict(attribute=key, input=sorted(av), candidate=sorted(bv), status='explicit_distinction_not_interchangeable'))
    if any((v in x.get('structure', {}) for v in ('invasion', 'infiltration'))) and any((v in y.get('structure', {}) for v in ('cyst', 'cysts'))):
        conflicts.append(dict(attribute='event_relationship', status='invasion_is_not_cyst'))
    for item in a['attributes']:
        if item['attribute'] == 'exclusion' and re.search('\\b' + re.escape(item['value']) + '\\b', candidate, re.I) and (item['value'] not in y.get('exclusion', {})):
            conflicts.append(dict(attribute='exclusion', value=item['value'], status='excluded_modifier_asserted_or_scope_unknown'))
    literal = _norm(original) == _norm(candidate)
    unresolved = [] if literal else sorted(set(a['unresolved'] + b['unresolved']))
    if unrepresented:
        unresolved.append('implicit_attribute_representation_unverified')
    if not literal and re.search('\\b(?:with|without|of|in|beyond|into|excluding|except)\\b', original, re.I):
        from grammatical_discrimination import equivalence
        if not equivalence(original, candidate)['equivalent']:
            unresolved.append('full_expression_attachment_unverified')
    blocked = bool(unrepresented or added or conflicts or unresolved)
    return dict(version=VERSION, original=original, candidate=candidate, input_frame=a, candidate_frame=b, preserved=preserved, unrepresented=unrepresented, unsupported_additions=added, contradictions=conflicts, unresolved=sorted(set(unresolved)), full_equivalence_blocked=blocked, relationship='explicit_conflict' if conflicts else 'unsupported_specificity' if added else 'potentially_broader_unverified' if unrepresented else 'unresolved' if unresolved else 'no_detected_attribute_obstacle', positive_equivalence_authority=False)
