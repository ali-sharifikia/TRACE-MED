"""Deterministic, input-bound grammatical equivalence. No labels or model calls."""
from runtime_resources import resource as asp_resource
from dataclasses import dataclass, replace
from collections import Counter
from functools import lru_cache
import re
import input_discrimination as prior
VERSION = '1.7.0'

@dataclass(frozen=True)
class GrammarConfig(prior.DevelopmentConfig):
    version: str = VERSION
    pt_ranking: bool = False
    llt_path: str = 'supported_wording'
    grammar: bool = True
    ambiguity: bool = True
    name: str = 'v1.7_scoped_grammar'
DEVELOPMENT = GrammarConfig()
BASELINE = replace(prior.DEVELOPMENT, pt_ranking=False)
CONFIGS = {DEVELOPMENT.name: DEVELOPMENT, 'v1.7_no_ambiguity': replace(DEVELOPMENT, ambiguity=False, name='v1.7_no_ambiguity'), 'v1.7_no_semantics': replace(DEVELOPMENT, semantic=False, name='v1.7_no_semantics'), 'v1.7_no_specificity_score': replace(DEVELOPMENT, specificity=False, name='v1.7_no_specificity_score'), 'v1.7_no_grammar': replace(DEVELOPMENT, grammar=False, name='v1.7_no_grammar'), 'v1.7_full_PT_ablation': replace(DEVELOPMENT, pt_ranking=True, name='v1.7_full_PT_ablation')}
DIRECTION = {'increase', 'decrease', 'high', 'low', 'elevated', 'elevation', 'reduced', 'reduction', 'abnormal', 'positive', 'negative', 'change'}
HEADS = {'infection', 'swelling', 'tumor', 'neoplasm', 'fistula', 'toxicity', 'disorder', 'disease', 'reaction', 'arrhythmia', 'thrombosis', 'edema', 'haemorrhage', 'hemorrhage', 'pain', 'sensation', 'death', 'anemia', 'anaemia', 'oropharyngitis', 'tuberculosis', 'depression', 'paraesthesia', 'paresthesia'}
MEASURE_QUALIFIERS = {'total', 'conjugated', 'unconjugated', 'systolic', 'diastolic', 'absolute', 'estimated'}
SPECIMENS = {'blood', 'serum', 'plasma', 'urine'}
ATTRIBUTE_ROLES = {'deep': 'depth', 'superficial': 'depth', 'venous': 'vessel', 'arterial': 'vessel', 'acute': 'duration', 'chronic': 'duration', 'recurrent': 'recurrence'}
RELATIONAL = re.compile('\\b(without|with|due to|following|before|after|history of|caused by|secondary to|to|from|and|or|versus)\\b|[;,]', re.I)
NOS = re.compile('\\b(?:nos|unspecified|non[- ]specific|nonspecific)\\b', re.I)

@lru_cache(maxsize=65536)
def phrase(text):
    r = prior.representations(text)
    words = list(r['wording_tokens'])
    surface = prior.surface(text)
    links = [{'text': m[0], 'span': [m.start(), m.end()]} for m in RELATIONAL.finditer(surface)]
    guarded = bool(links or re.search('\\b(?:no|not|non-)\\b', surface) or r['clinical']['negated_modifiers'] or r['transition'])
    key = ('ordered', tuple(words))
    rule = 'ordered_bounded_inflection'
    directions = [(i, w) for i, w in enumerate(words) if w in DIRECTION]
    single_specimen_predicate = len(words) == 3 and words[0] in SPECIMENS and (len(directions) == 1) and (directions[0][0] == 1) and (directions[0][1] in {'positive', 'negative'})
    if not guarded and len(directions) == 1 and (directions[0][0] in (0, len(words) - 1) or single_specimen_predicate) and (len(words) > 1):
        i, d = directions[0]
        body = words[:i] + words[i + 1:]
        qualifiers = tuple(sorted((w for w in body if w in MEASURE_QUALIFIERS | SPECIMENS)))
        core = tuple((w for w in body if w not in MEASURE_QUALIFIERS | SPECIMENS))
        key = ('finding', d, core, qualifiers)
        rule = 'single_finding_direction_frame'
    elif not guarded and len(directions) == 0:
        heads = [(i, w) for i, w in enumerate(words) if w in HEADS]
        if len(heads) == 1 and heads[0][0] in (0, len(words) - 1):
            i, h = heads[0]
            body = tuple(words[:i] + words[i + 1:])
            if all((w in ATTRIBUTE_ROLES for w in body)) and len({ATTRIBUTE_ROLES[w] for w in body}) == len(body):
                body = tuple(sorted(body, key=lambda w: ATTRIBUTE_ROLES[w]))
            if body:
                key = ('entity', h, body)
                rule = 'single_head_complement_rotation'
        elif len(words) == 2 and len(set(words) & SPECIMENS) == 1:
            key = ('specimen', next((w for w in words if w in SPECIMENS)), next((w for w in words if w not in SPECIMENS)))
            rule = 'single_specimen_argument'
    return {'key': key, 'rule': rule, 'wording_tokens': words, 'relations': links, 'guarded_scope': guarded, 'numbers': r['clinical']['bounds'], 'units': r['clinical']['units'], 'source': text, 'provenance': 'closed grammatical argument frames; original scope retained'}

def equivalence(text, label):
    a, b = (phrase(text), phrase(label))
    same = a['key'] == b['key'] and bool(a['wording_tokens'])
    if a['guarded_scope'] or b['guarded_scope']:
        same = prior.surface(text) == prior.surface(label)
    sa = prior.support(text, label)
    if sa['conflicts'] or sa['unsupported_specificity']:
        same = False
    return {'equivalent': same, 'source_frame': a, 'candidate_frame': b, 'correspondences': sorted((Counter(a['wording_tokens']) & Counter(b['wording_tokens'])).elements()), 'added_words': sorted((Counter(b['wording_tokens']) - Counter(a['wording_tokens'])).elements()), 'omitted_words': sorted((Counter(a['wording_tokens']) - Counter(b['wording_tokens'])).elements()), 'reason': a['rule'] if same else 'scope_or_argument_or_wording_not_established', 'support': sa}

def ambiguity_signature(label, strip_measurement=True):
    stripped = NOS.sub('', label)
    if strip_measurement:
        stripped = re.sub('\\b(?:levels?|values?)\\b', '', stripped, flags=re.I)
    return phrase(stripped)['key']

def rank(text, pt_rows, groups, semantics, *, config=DEVELOPMENT):
    result = prior.rank(text, pt_rows, groups, semantics, config=config)
    for pt, rows in result['groups'].items():
        for row in rows:
            f = result['features'][row['candidate_id']]
            g = equivalence(text, row['llt_name'])
            f['grammatical_equivalence'] = g
            if config.grammar:
                f['wording_equivalent'] = g['equivalent']
                same_bag = Counter(g['source_frame']['wording_tokens']) == Counter(g['candidate_frame']['wording_tokens'])
                if same_bag and (g['source_frame']['guarded_scope'] or g['candidate_frame']['guarded_scope']) and (not g['equivalent']):
                    f['score'] = 0.0
                    f['conflicts'] = sorted(set(f['conflicts']) | {'unresolved_relational_scope'})
            rivals = []
            if config.ambiguity and (not NOS.search(text)):
                strip_measurement = not bool(re.search('\\b(?:levels?|values?)\\b', text, re.I))
                signature = ambiguity_signature(row['llt_name'], strip_measurement)
                for other in rows:
                    if other['candidate_id'] == row['candidate_id']:
                        continue
                    other_support = prior.support(text, other['llt_name'])
                    if other_support['conflicts'] or other_support['unsupported_specificity']:
                        continue
                    if ambiguity_signature(other['llt_name'], strip_measurement) == signature:
                        rivals.append({'candidate_id': other['candidate_id'], 'llt_name': other['llt_name'], 'reason': 'NOS_or_optional_measurement_wording_does_not_establish_unique_identity'})
            f['unresolved_reference_wording_alternatives'] = rivals
        result['groups'][pt] = sorted(rows, key=lambda r: (-result['features'][r['candidate_id']]['score'], r['candidate_id']))
    result['version'] = VERSION
    return result

def decide(input_evidence, ranking, baseline, *, config=DEVELOPMENT):
    original_groups = ranking['groups']
    semantic = {k: f['semantic'] for k, f in ranking['features'].items() if f['semantic'] is not None}
    pt_prior = prior.rank(input_evidence['original_input'], ranking['pt_rankings'], original_groups, semantic, config=config)
    pt_output, pt_audit = prior.decide(input_evidence, pt_prior, baseline, config=config)
    seeded = {**pt_output, 'llt': list(baseline.get('llt', [])) if pt_output['pt'] == baseline.get('pt', []) else []}
    output, audit = prior.decide(input_evidence, ranking, seeded, config=config)
    output['pt'] = pt_output['pt']
    if not output['pt']:
        output['llt'] = []
    audit.update(version=VERSION, configuration=config.to_dict(), deterministic_configuration=config.name, PT_acceptance=pt_audit, experimental_model_selector_enabled=False)
    for key in ('PT_selected', 'PT_score', 'PT_margin', 'competitor', 'reviewer_agreement', 'PT_support', 'pt_reasons'):
        if key in pt_audit:
            audit[key] = pt_audit[key]
    if output['pt'] != baseline.get('pt', []):
        audit['applied'] = True
        audit['path'] = 'independent_PT_policy/' + pt_audit['path']
    applicable = input_evidence.get('route') != 'exact' and input_evidence.get('structure') == 'single_concept' and input_evidence.get('input_bound') and (not input_evidence.get('force_review')) and (not input_evidence.get('exact_collision'))
    if applicable and output['llt']:
        selected = [ranking['features'].get('LLT:' + code) for code in output['llt']]
        reasons = []
        if any((not f for f in selected)):
            reasons.append('missing_selected_reference_evidence')
        if any((f and f['unresolved_reference_wording_alternatives'] for f in selected)):
            reasons.append('reference_wording_identity_unresolved')
        if any((f and (f['conflicts'] or f['unsupported_specificity']) for f in selected)):
            reasons.append('selected_scope_or_specificity_not_supported')
        if reasons:
            output['llt'] = []
            output['review_pt'] = list(output['pt'])
            output['review_llt'] = [r['llt_code'] for p in output['pt'] for r in ranking['groups'].get(p, []) if not ranking['features'][r['candidate_id']]['conflicts']][:4]
            audit['llt_reasons'] = sorted(set(audit['llt_reasons'] + reasons))
            audit['path'] = 'scoped_grammar_review'
            audit['applied'] = True
        else:
            audit['path'] = 'scoped_grammatical_LLT_acceptance'
    return (output, audit)
support = prior.support
