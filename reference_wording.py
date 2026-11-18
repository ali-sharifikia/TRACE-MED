"""Coherent, input-bound measurement comparison; no labels, codes or I/O.

Parent context is compatibility evidence, never an LLT score contribution.
All new authorization still goes through the existing current-reference gate.
"""
from runtime_resources import resource as asp_resource
import re
from dataclasses import dataclass, replace
from functools import lru_cache
import event_comparison as previous
from clinical_features import interpreted_features, clinical_scoring_text, preserve_numeric_notation
from pt_reranking import _detected_analytes
from terminology_aliases import generated_acronym
old = previous.old

@dataclass(frozen=True)
class MeasurementConfig(previous.ComparisonConfig):
    version: str = '1.10.0'
    name: str = 'v1.10_measurement_context'
    recover_pt: bool = True
    measurement_ranking: bool = True
    measurement_acceptance: bool = True
    parent_context: bool = True
DEVELOPMENT = MeasurementConfig()
REPRESENTATION = replace(DEVELOPMENT, name='v1.10_measurement_representation', measurement_ranking=False, measurement_acceptance=False, parent_context=False)
ACCEPTANCE = replace(DEVELOPMENT, name='v1.10_parent_context_baseline_order', measurement_ranking=False)
RANKING = replace(DEVELOPMENT, name='v1.10_measurement_ranking', measurement_acceptance=False, parent_context=False)
NO_PARENT = replace(DEVELOPMENT, name='v1.10_measurement_without_parent', parent_context=False)
CONFIGS = {c.name: c for c in (REPRESENTATION, ACCEPTANCE, RANKING, NO_PARENT, DEVELOPMENT)}
DEVELOPMENT = ACCEPTANCE
FAMILIES = {'increase': ('increase', 'increased', 'increases'), 'decrease': ('decrease', 'decreased', 'decreases'), 'elevation': ('elevation', 'elevated', 'elevations'), 'reduction': ('reduction', 'reduced', 'reductions'), 'raised': ('raised',), 'high': ('high',), 'low': ('low',), 'abnormal': ('abnormal', 'abnormality', 'abnormalities')}
FAMILY = {w: k for k, words in FAMILIES.items() for w in words}
DIRECTION = {k: 'increased' if k in ('increase', 'elevation', 'raised', 'high') else 'decreased' if k in ('decrease', 'reduction', 'low') else 'abnormal' for k in FAMILIES}
SPECIMEN = {'blood', 'serum', 'plasma', 'urine', 'saliva', 'csf'}
TYPES = {'activity': 'activity', 'concentration': 'concentration', 'count': 'count', 'counts': 'count', 'percentage': 'percentage', 'percent': 'percentage', 'ratio': 'ratio', 'time': 'time'}
QUALIFIERS = {'total', 'free', 'ionized', 'conjugated', 'unconjugated', 'absolute', 'fasting'}
MORPH = {'enzymes': 'enzyme', 'hormones': 'hormone', 'factors': 'factor', 'platelets': 'platelet', 'leukocytes': 'leukocyte', 'lymphocytes': 'lymphocyte', 'neutrophils': 'neutrophil'}
SCOPE = re.compile('\\b(?:and|or|with|without|after|before|following|from|to|due|history|possible|suspected|transient|recurrence|recurrent|worsening)\\b|[;,]')

def spans(text, pattern):
    return [{'span': [m.start(), m.end()], 'text': m[0]} for m in re.finditer(pattern, text, re.I)]

@lru_cache(maxsize=65536)
def measurement(text):
    surface = old.prior.surface(preserve_numeric_notation(text))
    legacy = interpreted_features(surface)
    found = [m for m in re.finditer('\\b(?:' + '|'.join(FAMILY) + ')\\b', surface) if not re.match('\\s+density\\b', surface[m.end():])]
    families = sorted({FAMILY[m[0]] for m in found})
    directions = sorted({DIRECTION[k] for k in families})
    normalized = re.sub('\\b(?:' + '|'.join(FAMILY) + ')\\b', lambda m: DIRECTION[FAMILY[m[0]]] if not re.match('\\s+density\\b', surface[m.end():]) else m[0], surface)
    clinical_input = re.sub('\\b(high|low)\\s+density\\b', lambda m: m[1] + 'density', normalized)
    active = interpreted_features(clinical_input)
    types = sorted({TYPES[m[0]] for m in re.finditer('\\b(?:' + '|'.join(TYPES) + ')\\b', surface)})
    if '%' in surface and 'percentage' not in types:
        types.append('percentage')
    specimen = sorted(set(re.findall('\\b(?:' + '|'.join(SPECIMEN) + ')\\b', surface)))
    qualifiers = sorted(set(re.findall('\\b(?:' + '|'.join(QUALIFIERS) + ')\\b', surface)))
    lexical = clinical_scoring_text(surface)
    words = re.findall('[a-z]+[0-9]*(?:-[a-z]+)?', lexical)
    core = []
    for i, w in enumerate(words):
        if w in FAMILY and (not (i + 1 < len(words) and words[i + 1] == 'density')):
            continue
        if w in SPECIMEN or w in TYPES or w in QUALIFIERS or (w in ('in', 'of', 'the', 'level', 'levels', 'value', 'values')):
            continue
        core.append(MORPH.get(w, w))
    entity = ' '.join(core)
    aliases = []
    meaning = entity
    for a, b in sorted(old.prior.ALIASES.items(), key=lambda p: (-len(p[0]), p[0])):
        matches = spans(surface, '\\b' + re.escape(a) + '\\b')
        if matches:
            aliases.append({'original': a, 'expansion': b, 'status': 'supported_existing_resource', 'resource': 'input_discrimination_v1_6.ALIASES', 'evidence': matches})
            meaning = re.sub('\\b' + re.escape(a) + '\\b', b, meaning)
    meaning = ' '.join((previous.SITE.get(w, w) for w in old.prior.representations(meaning)['meaning_tokens']))
    laboratory = bool(_detected_analytes(surface) or active['laboratory_context'] or specimen or types or re.search('\\b(?:enzymes?|hormones?)\\b', surface))
    units = spans(surface, '(?:[a-zμµ]+(?:/[a-zμµ]+)+|%|10\\s*\\^[+-]?\\d+)')
    numeric = spans(surface, '(?<![a-z])\\d+(?:\\.\\d+)?(?:\\s*[x*]\\s*10\\s*\\^[+-]?\\d+)?')
    exclusive = any((set(q) <= set(qualifiers) for q in (('total', 'free'), ('conjugated', 'unconjugated'))))
    uncertainty = spans(surface, '\\b(?:possible|suspected|probable|likely|uncertain|may|might)\\b')
    safe = bool(core and (not SCOPE.search(surface)) and (not active['event_negated']) and (not uncertainty) and (len(directions) <= 1) and (len(specimen) <= 1) and (len(types) <= 1) and (not exclusive))
    return {'version': '1.10.0', 'original': text, 'surface': surface, 'entity_wording': entity, 'entity_meaning': meaning, 'detected_analytes': sorted(_detected_analytes(normalized)), 'measured_entity_status': 'explicit_measurement_argument' if laboratory and core else 'unresolved_without_measurement_context', 'aliases': aliases, 'measurement_types': types, 'specimen': specimen, 'qualifiers': qualifiers, 'direction': directions[0] if len(directions) == 1 else active['direction'], 'direction_families': families, 'direction_evidence': [{'span': [m.start(), m.end()], 'text': m[0], 'family': FAMILY[m[0]], 'meaning': DIRECTION[FAMILY[m[0]]]} for m in found], 'bounds': active['bounds'], 'units': units, 'numeric_content': numeric, 'generic_level_wording': bool(re.search('\\blevels?\\b', surface)), 'attribute_spans': {k: spans(surface, '\\b(?:' + '|'.join(v) + ')\\b') for k, v in (('specimen', SPECIMEN), ('qualifiers', QUALIFIERS), ('measurement_types', TYPES))}, 'negated': active['event_negated'], 'uncertainty': uncertainty, 'course': old.representation(text)['course'], 'scope_safe': safe, 'laboratory': laboratory, 'normalized_clinical': active, 'legacy_clinical': legacy, 'normalization_input': normalized, 'clinical_normalization_input': clinical_input, 'protected_density_spans': spans(surface, '\\b(?:high|low)\\s+density\\b'), 'scope_evidence': spans(surface, SCOPE.pattern), 'authority': 'original input; bounded morphology and existing language resources'}

def context_aliases(pt_rows):
    choices = {}
    for row in pt_rows:
        m = measurement(row['pt_name'])
        entity = m['entity_wording']
        a = generated_acronym(entity)
        if a and m['laboratory']:
            choices.setdefault(a, set()).add(entity)
    return {a: sorted(v) for a, v in choices.items()}

def entity(frame, aliases):
    value = frame['entity_meaning']
    evidence = []
    for a, choices in sorted(aliases.items()):
        if re.search('\\b' + re.escape(a) + '\\b', value):
            status = 'supported' if len(choices) == 1 else 'unresolved'
            evidence.append({'original': a, 'expansions': choices, 'status': status, 'resource': 'generated acronym of compared current PT measured entity', 'source': frame['surface'], 'spans': spans(frame['surface'], '\\b' + re.escape(a) + '\\b')})
            if len(choices) == 1:
                value = re.sub('\\b' + re.escape(a) + '\\b', measurement(choices[0])['entity_meaning'], value)
    return (value, evidence)

def pair(text, row, pt_row, aliases, *, parent_context=True):
    a, b, p = (measurement(text), measurement(row['llt_name']), measurement(row['pt_name']))
    verified = bool(row.get('candidate_id') == 'LLT:' + str(row.get('llt_code')) and row.get('pt_code') == pt_row.get('pt_code') and (row.get('pt_name') == pt_row.get('pt_name')))
    ae, ax = entity(a, aliases)
    be, bx = entity(b, aliases)
    pe, px = entity(p, aliases)
    same = bool(ae and ae == be and (ae == pe))
    evidence = []
    blocked = []
    for field in ('measurement_types', 'specimen', 'qualifiers', 'direction'):
        x, y, z = (a[field], b[field], p[field])
        if x and y and (x != y):
            status = 'conflicting'
        elif x and y == x:
            status = 'supported_LLT'
        elif not x and y:
            status = 'unsupported_addition'
        elif x and (not y) and parent_context and verified and same and (z == x):
            status = 'supported_PT'
        elif x and (not y):
            status = 'unencoded'
        else:
            status = 'not_asserted'
        evidence.append({'attribute': field, 'input': x, 'LLT': y, 'PT': z, 'status': status, 'sources': {'input': a['surface'], 'LLT': b['surface'], 'PT': p['surface']}})
        if status in ('conflicting', 'unsupported_addition', 'unencoded'):
            blocked.append(field + ':' + status)
        if field in ('measurement_types', 'direction') and x and z and (x != z):
            blocked.append(field + ':explicit_parent_conflict')
        if field in ('measurement_types', 'direction', 'qualifiers') and z and (not x):
            blocked.append(field + ':parent_attribute_not_established_by_input')
        if field == 'qualifiers' and x and z and (set(x) != set(z)):
            blocked.append('qualifiers:explicit_parent_distinction_unresolved')
        if field == 'specimen' and x and z and (x != z) and (z != ['blood']):
            blocked.append('specimen:explicit_parent_conflict')
    if a['measurement_types'] and b['generic_level_wording'] or (b['measurement_types'] and a['generic_level_wording']):
        blocked.append('explicit_measurement_type_versus_unspecified_level')
    if a['bounds'] or b['bounds'] or a['units'] or b['units']:
        if a['bounds'] != b['bounds'] or [x['text'] for x in a['units']] != [x['text'] for x in b['units']]:
            blocked.append('numerical_information_unencoded_or_conflicting')
    if [x['text'] for x in a['numeric_content']] != [x['text'] for x in b['numeric_content']]:
        blocked.append('numeric_content_not_preserved')
    if not verified:
        blocked.append('identity_parent_scope_mismatch')
    if not same:
        blocked.append('measured_entity_unresolved_or_different')
    if not all((f['scope_safe'] for f in (a, b, p))):
        blocked.append('negation_temporal_or_relational_scope')
    if not (a['laboratory'] or p['laboratory']):
        blocked.append('measurement_context_unestablished')
    if any((x['status'] == 'unresolved' for x in ax + bx + px)):
        blocked.append('ambiguous_contextual_alias')
    direct = bool(a['entity_wording'] == b['entity_wording'] and a['direction_families'] == b['direction_families'])
    return {'source_frame': a, 'LLT_frame': b, 'PT_frame': p, 'attributes': evidence, 'source_aliases': ax, 'LLT_aliases': bx, 'PT_aliases': px, 'identity_parent_consistent': verified, 'same_measured_entity': same, 'compatible': not blocked, 'blockers': blocked, 'original_entity_and_direction_family_preserved': direct, 'parent_context_used': any((x['status'] == 'supported_PT' for x in evidence)), 'reference_authority': 'compared current-reference group; completeness and input binding verified by gate, not inferred from wording', 'parent_score_contribution': 0.0}

def rank(text, pt_rows, groups, semantics, *, config=DEVELOPMENT):
    r = previous.rank(text, pt_rows, groups, semantics, config=config)
    aliases = context_aliases(pt_rows)
    parents = {p['pt_code']: p for p in pt_rows}
    for pt, rows in r['groups'].items():
        fs = r['features']
        pairs = {x['candidate_id']: pair(text, x, parents.get(pt, {}), aliases, parent_context=config.parent_context) for x in rows}
        compatible = [cid for cid, e in pairs.items() if e['compatible'] and (not fs[cid]['conflicts']) and (not fs[cid]['unsupported_specificity'])]
        direct = [cid for cid in compatible if pairs[cid]['original_entity_and_direction_family_preserved']]
        preferred = direct if direct else compatible
        for x in rows:
            cid = x['candidate_id']
            f = fs[cid]
            e = pairs[cid]
            f['measurement_v1_10'] = {**e, 'compatible_competitors': [k for k in compatible if k != cid], 'preferred_identities': preferred}
            f['baseline_v1_9_score'] = f['score']
            f['baseline_v1_9_wording'] = f['wording_equivalent']
            unique = len(preferred) == 1 and cid == preferred[0]
            if config.measurement_ranking and unique and e['compatible'] and (not e['parent_context_used']) and (not f['unresolved_reference_wording_alternatives']):
                if e['original_entity_and_direction_family_preserved']:
                    target = sum((v for k, v in f['contributions'].items() if k not in ('bounded_correspondence', 'event_argument_correspondence')))
                    target += max(0.0, 0.55 - f['contributions']['input_wording']) + max(0.0, 0.35 - f['contributions']['clinical_meaning'])
                    if not f['original_v1_7_equivalent']:
                        target -= 0.04
                    increment = max(0.0, min(1.0, target) - f['score'])
                    f['score'] += increment
                    f['contributions']['measurement_original_morphology'] = increment
            authorized = bool(config.measurement_acceptance and unique and e['compatible'])
            if authorized:
                f['wording_equivalent'] = True
            f['acceptance_evidence'].update(measurement_compatibility=authorized, wording_authorized=f['wording_equivalent'], parent_context_used=e['parent_context_used'], parent_score_contribution=0.0)
            f['ordering_evidence'].update(strategy=config.name, score=f['score'], baseline_v1_9_score=f['baseline_v1_9_score'])
        r['groups'][pt] = sorted(rows, key=lambda x: (-fs[x['candidate_id']]['score'], x['candidate_id']))
    return r

def decide(inp, ranking, baseline, *, config=DEVELOPMENT):
    original = previous.rank(inp['original_input'], ranking['pt_rankings'], ranking['groups'], {k: f['semantic'] for k, f in ranking['features'].items() if f['semantic'] is not None})
    pt_out, pt_audit = previous.decide(inp, original, baseline)
    out, audit = previous.decide(inp, ranking, {**pt_out, 'llt': baseline.get('llt', []) if pt_out['pt'] == baseline.get('pt', []) else []}, config=replace(config, recover_pt=False))
    out['pt'] = pt_out['pt']
    if not out['pt']:
        out['llt'] = []
    audit['PT_measurement_control'] = {'policy': 'unchanged v1.9_event_comparison_PT', 'decision': pt_out['pt'], 'acceptance': pt_audit.get('PT_acceptance')}
    audit['LLT_stage_configuration'] = audit.get('configuration')
    audit['configuration'] = config.to_dict()
    audit['measurement_configuration'] = config.to_dict()
    return (out, audit)
support = previous.support
