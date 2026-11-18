"""Shared, deterministic clinical language features. No terminology codes or labels."""
from __future__ import annotations
from runtime_resources import resource as asp_resource
import re
from decimal import Decimal
from functools import lru_cache
CLINICAL_FEATURES_VERSION = '1.4.0'
NEGATABLE_MODIFIERS = frozenset({'pathological', 'traumatic', 'infectious', 'allergic', 'autoimmune', 'malignant', 'specific', 'cardiac', 'renal', 'hepatic', 'surgical', 'inflammatory', 'obstructive', 'congenital', 'hereditary'})
COMPARISON = '(?P<operator><=|>=|<|>|=|less\\s+than|greater\\s+than|at\\s+least|at\\s+most|below|above)'
NUMBER = '(?P<mantissa>[+-]?(?:\\d+(?:\\.\\d+)?|\\.\\d+))'
SCALE = '(?:\\s*(?:[x*\\u00d7]\\s*10\\s*\\^?\\s*|e)(?P<exponent>[+-]?\\d+))?'
BOUND = re.compile(COMPARISON + '\\s*' + NUMBER + SCALE, re.I)
VOLUME = re.compile('/\\s*([mun\\u03bc\\u00b5]?l)\\b|\\b([mun\\u03bc\\u00b5]?l)\\s*\\^?-1\\b', re.I)
LAB_WORDS = re.compile('\\b(?:count|concentration|level|percentage|platelet|neutrophil|lymphocyte|leukocyte|hemoglobin|hematocrit|aminotransferase|bilirubin)\\b', re.I)
COUNT_ABBREVIATIONS = {'anc': 'absolute neutrophil count', 'alc': 'absolute lymphocyte count', 'wbc': 'white blood cell count', 'rbc': 'red blood cell count'}
NUMBER_WORDS = {word: str(n) for n, word in enumerate('zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty'.split())}
REFERENCE_WORDING = '(?:uln|lln|upper limit of normal|lower limit of normal)'
REFERENCE_MULTIPLIER = re.compile('\\b(' + '|'.join(NUMBER_WORDS) + '|\\d+(?:\\.\\d+)?)\\s*(?:times|fold|[x*])\\s*(?:the\\s+)?(' + REFERENCE_WORDING + ')\\b', re.I)

def canonical_reference_multipliers(text: str) -> str:
    """Bounded number-word equivalence only next to an explicit normal bound.

An unqualified 'two times' or 'two tablets' is not an abnormal laboratory value.
Bare '2 times ULN' asserts a ratio (=2); '<2 times ULN' retains its comparator.
"""

    def expand(match):
        previous = text[:match.start()]
        has_operator = re.search('(?:<=|>=|<|>|=|less\\s+than|greater\\s+than|at\\s+least|at\\s+most|below|above)\\s*$', previous, re.I)
        number = NUMBER_WORDS.get(match[1].casefold(), match[1])
        reference = 'uln' if match[2].casefold() in {'uln', 'upper limit of normal'} else 'lln'
        return ('' if has_operator else '=') + number + '*' + reference
    return REFERENCE_MULTIPLIER.sub(expand, text)

def preserve_numeric_notation(text: str) -> str:
    """Make superscript exponents explicit before Unicode compatibility folding."""
    superscripts = str.maketrans('⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺', '0123456789-+')
    text = re.sub('(?<=[\\dalL])([\\u2070\\u00b9\\u00b2\\u00b3\\u2074-\\u2079\\u207b\\u207a]+)', lambda m: '^' + m[1].translate(superscripts), text)
    return text.replace('≥', '>=').replace('≤', '<=')

def modifier_negations(text: str) -> set[str]:
    return {m[1].casefold() for m in re.finditer('\\bnon[\\s\\-\\u2010-\\u2015]+([a-z]+)\\b', text, re.I) if m[1].casefold() in NEGATABLE_MODIFIERS}

def event_is_negated(text: str) -> bool:
    return bool(re.search('\\b(?:no|not|without|denies|denied|absent)\\b|\\babsence\\s+of\\b', text, re.I))

@lru_cache(maxsize=32768)
def interpreted_features(value: str) -> dict:
    text = canonical_reference_multipliers(preserve_numeric_notation(value.casefold()))
    bounds = []
    operator_map = {'less than': '<', 'greater than': '>', 'below': '<', 'above': '>', 'at least': '>=', 'at most': '<='}
    for m in BOUND.finditer(text):
        exponent = int(m['exponent'] or 0)
        canonical_value = str(Decimal(m['mantissa']).scaleb(exponent).normalize()) if abs(exponent) <= 100 else None
        reference_match = re.match('\\s*[*x]?\\s*(?:the\\s+)?(' + REFERENCE_WORDING + ')\\b', text[m.end():], re.I)
        reference = None if not reference_match else 'uln' if reference_match[1] in {'uln', 'upper limit of normal'} else 'lln'
        bounds.append({'surface': m[0], 'operator': operator_map.get(' '.join(m['operator'].split()), m['operator']), 'mantissa': m['mantissa'], 'exponent': exponent, 'value': canonical_value, 'normal_reference': reference})
    units = [next((group for group in m.groups() if group)) for m in VOLUME.finditer(text)]
    if '%' in text or re.search('\\bpercent(?:age)?\\b', text):
        units.append('%')
    laboratory_context = bool(LAB_WORDS.search(text) or (bounds and units) or any((b['normal_reference'] for b in bounds)))
    retrieval_text = text
    expansions = []
    for acronym, expansion in COUNT_ABBREVIATIONS.items():
        if re.search(f'\\b{acronym}\\b', text):
            supported = laboratory_context and '%' not in units
            expansions.append({'acronym': acronym, 'expansion': expansion, 'status': 'supported' if supported else 'unresolved', 'reason': 'count_or_volume_context' if supported else 'insufficient_measurement_context'})
            if supported:
                retrieval_text = re.sub(f'\\b{acronym}\\b', expansion, retrieval_text)
    analytes = set()
    if re.search('\\b(?:anca|[cp]-anca)\\b|antineutrophil\\s+cytoplasmic\\s+antibod', retrieval_text):
        analytes.add('antineutrophil_cytoplasmic_antibody')
    for name, pattern in {'neutrophil': '\\bneutrophil(?:s)?\\b', 'lymphocyte': '\\blymphocyte(?:s)?\\b', 'basophil': '\\bbasophil(?:s)?\\b', 'platelet': '\\bplatelet(?:s)?\\b', 'white_blood_cell': '\\bwhite blood cell', 'red_blood_cell': '\\bred blood cell'}.items():
        if re.search(pattern, retrieval_text):
            analytes.add(name)
    measurement = 'percentage' if '%' in units else 'absolute_count' if 'absolute' in retrieval_text and 'count' in retrieval_text or units else 'count' if 'count' in retrieval_text else None
    direction = None
    direction_source = None
    if re.search('\\b(?:decreased|low|reduced|decrease)\\b', retrieval_text):
        direction, direction_source = ('decreased', 'explicit_wording')
    elif re.search('\\b(?:increased|high|elevated|increase)\\b', retrieval_text):
        direction, direction_source = ('increased', 'explicit_wording')
    if direction is None:
        for bound in bounds:
            if bound['value'] is None:
                continue
            quantity = Decimal(bound['value'])
            if bound['normal_reference'] == 'uln' and (bound['operator'] in ('>', '>=', '=') and quantity > 1 or (bound['operator'] == '>' and quantity == 1)):
                direction, direction_source = ('increased', 'explicit_normal_reference_bound')
            elif bound['normal_reference'] == 'lln' and (bound['operator'] in ('<', '<=', '=') and quantity < 1 or (bound['operator'] == '<' and quantity == 1)):
                direction, direction_source = ('decreased', 'explicit_normal_reference_bound')
    return {'version': CLINICAL_FEATURES_VERSION, 'retrieval_text': retrieval_text, 'laboratory_context': laboratory_context, 'analytes': sorted(analytes), 'measurement_type': measurement, 'bounds': bounds, 'units': units, 'direction': direction, 'direction_source': direction_source, 'acronym_expansions': expansions, 'negated_modifiers': sorted(modifier_negations(text)), 'event_negated': event_is_negated(text)}

def clinical_scoring_text(value: str) -> str:
    """Clinical lexical features; numerical constraints remain separate data."""
    features = interpreted_features(value)
    text = BOUND.sub(' ', features['retrieval_text'])
    text = VOLUME.sub(' ', text)
    text = re.sub('\\b(?:uln|lln|upper limit of normal|lower limit of normal)\\b', ' ', text)
    text = re.sub('(?<![a-z])[+-]?\\d+(?:\\.\\d+)?(?:[x*\\u00d7]10\\s*\\^?\\s*[+-]?\\d+|e[+-]?\\d+)?', ' ', text)
    text = re.sub('\\b(?:less|greater|than|above|below|at least|at most)\\b|[<>=^%*/]', ' ', text)
    for modifier in modifier_negations(text):
        text = re.sub(f'\\bnon[\\s\\-\\u2010-\\u2015]+{modifier}\\b', 'non' + modifier, text)
    if features['direction'] and features['direction_source'] == 'explicit_normal_reference_bound':
        text += ' ' + features['direction']
    return ' '.join(text.split())

def feature_conflicts(first: str, second: str) -> tuple[str, ...]:
    a, b = (interpreted_features(first), interpreted_features(second))
    conflicts = set()
    an, bn = (set(a['negated_modifiers']), set(b['negated_modifiers']))
    if an and bn and (an != bn) or any((re.search(f'\\b{modifier}\\b', second, re.I) and modifier not in bn for modifier in an)):
        conflicts.add('modifier_negation_conflict')
    if any((re.search(f'\\b{modifier}\\b', first, re.I) and modifier not in an for modifier in bn)):
        conflicts.add('modifier_negation_conflict')
    if a['analytes'] and b['analytes'] and (set(a['analytes']) != set(b['analytes'])):
        conflicts.add('analyte_conflict')
    if {a['measurement_type'], b['measurement_type']} & {'count', 'absolute_count'} and 'percentage' in {a['measurement_type'], b['measurement_type']}:
        conflicts.add('measurement_type_conflict')
    if a['bounds'] and b['bounds']:
        if a['units'] and b['units'] and (set(a['units']) != set(b['units'])):
            conflicts.add('measurement_unit_conflict')
        first_bounds = {(x['operator'], x['value'], x['normal_reference']) for x in a['bounds']}
        second_bounds = {(x['operator'], x['value'], x['normal_reference']) for x in b['bounds']}
        if first_bounds != second_bounds:
            conflicts.add('threshold_conflict')
        if {x['operator'][0] for x in a['bounds']} != {x['operator'][0] for x in b['bounds']}:
            conflicts.add('directionality_conflict')
    return tuple(sorted(conflicts))

def automatic_mapping_uncertainties(text: str) -> tuple[str, ...]:
    features = interpreted_features(text)
    if features['laboratory_context'] and features['bounds'] and (features['direction'] is None):
        return ('numerical_abnormal_direction_unresolved',)
    return ()

def expansion_context_status(query: str, expansion: str, *, ambiguous: bool) -> tuple[str, str]:
    source = interpreted_features(query)
    if feature_conflicts(query, expansion):
        return ('context_incompatible', 'protected_clinical_feature_conflict')
    if source['laboratory_context'] and (not LAB_WORDS.search(expansion)):
        return ('context_incompatible', 'laboratory_context_without_matching_measurement_expansion')
    if source['measurement_type'] == 'absolute_count' and re.search('percent|antibod|\\banca\\b', expansion, re.I):
        return ('context_incompatible', 'measurement_or_analyte_identity_conflict')
    if source['laboratory_context'] and LAB_WORDS.search(expansion):
        return ('supported', 'laboratory_wording_and_surrounding_measurement_context')
    return ('unresolved', 'multiple_context_free_expansions') if ambiguous else ('supported', 'unambiguous_reference_alias')
