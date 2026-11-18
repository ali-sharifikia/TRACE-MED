"""Opt-in, bounded input-to-LLT correspondence. No labels or terminology codes."""
from runtime_resources import resource as asp_resource
import re
from dataclasses import dataclass, replace
from functools import lru_cache
import grammatical_discrimination as old
import input_discrimination as prior
from pt_reranking import _detected_analytes

@dataclass(frozen=True)
class DiscriminationConfig(old.GrammarConfig):
    version: str = '1.8.0'
    name: str = 'v1.8_correspondence'
    representation: bool = True
    rerank: bool = True
    accept_correspondence: bool = True
    recover_pt: bool = True
DEVELOPMENT = DiscriminationConfig()
CONFIGS = {DEVELOPMENT.name: DEVELOPMENT, 'v1.8_adapter_baseline': replace(DEVELOPMENT, representation=False, rerank=False, accept_correspondence=False, recover_pt=False, name='v1.8_adapter_baseline'), 'v1.8_representation_only': replace(DEVELOPMENT, rerank=False, accept_correspondence=False, recover_pt=False, name='v1.8_representation_only'), 'v1.8_ranking_only': replace(DEVELOPMENT, accept_correspondence=False, recover_pt=False, name='v1.8_ranking_only'), 'v1.8_no_PT_recovery': replace(DEVELOPMENT, recover_pt=False, name='v1.8_no_PT_recovery'), 'v1.8_no_specificity_score': replace(DEVELOPMENT, specificity=False, name='v1.8_no_specificity_score'), 'v1.8_no_semantic': replace(DEVELOPMENT, semantic=False, name='v1.8_no_semantic')}
INFLECTION = {'ecchymoses': 'ecchymosis', 'emboli': 'embolus', 'crises': 'crisis', 'fractures': 'fracture', 'abnormalities': 'abnormal', 'abnormality': 'abnormal', 'dyspneae': 'dyspnea'}
COURSE = {'worsening': 'aggravated', 'worsened': 'aggravated', 'exacerbation': 'aggravated', 'exacerbations': 'aggravated', 'exacerbate': 'aggravated', 'aggravation': 'aggravated', 'recurrence': 'recurrent', 'recurrences': 'recurrent', 'progression': 'progressive', 'progressing': 'progressive', 'aggravated': 'aggravated', 'recurrent': 'recurrent', 'progressive': 'progressive'}

@lru_cache(maxsize=65536)
def representation(text):
    original = prior.surface(text)
    clinical = prior.representations(text)['clinical']
    changes = []

    def note(rule, m, replacement, source=None):
        source = original if source is None else source
        changes.append({'rule': rule, 'span': [m.start(), m.end()], 'supporting_text': m[0], 'replacement': replacement, 'source': source, 'span_scope': 'specific transformation input; original wording retained independently'})
        return replacement
    value = re.sub('\\b(?:' + '|'.join(INFLECTION) + ')\\b', lambda m: note('closed_inflection', m, INFLECTION[m[0]]), original)
    if not re.search('\\b(?:and|or|without|with|after|before|following|due to)\\b|[;,]', value):
        m = re.search('\\s+(?:adverse\\s+)?events?$', value)
        if m and len(value[:m.start()].split()) >= 1:
            note('named_event_context_suffix', m, '', value)
            value = value[:m.start()]
    laboratory = bool(_detected_analytes(original) or clinical['laboratory_context'] or (re.search('\\b(?:blood|serum|plasma|urine)\\b', original) and clinical['direction']))
    if laboratory:
        value = re.sub('\\b(?:elevated|elevation|elevations)\\b', lambda m: note('laboratory_elevation_direction', m, 'increased', value), value)
        value = re.sub('\\b(?:levels?|values?)\\b', lambda m: note('laboratory_level_argument', m, '', value), value)
    course = []
    for m in re.finditer('\\b(?:' + '|'.join(COURSE) + ')\\b', original):
        course.append(COURSE[m[0]])
        note('event_course_attachment', m, COURSE[m[0]])
    if course and len(course) == 1 and (not re.search('\\b(?:and|or|with|without|after|before)\\b|[;]', value)):
        value = re.sub('^(?:' + '|'.join(COURSE) + ')\\s+(?:of\\s+)?', '', value)
        value = re.sub('[, ]+\\b(?:' + '|'.join(COURSE) + ')$', '', value)
        value = value.strip(' ,') + ' ' + course[0]
    value = ' '.join(value.split())
    return {'original': text, 'surface': original, 'comparison_text': value, 'clinical': clinical, 'transformations': changes, 'course': course, 'laboratory': laboratory, 'provenance': 'bounded v1.8 grammar; original wording and clinical numbers retained'}

@lru_cache(maxsize=65536)
def correspondence(text, label):
    a, b = (representation(text), representation(label))
    eq = old.equivalence(a['comparison_text'], b['comparison_text'])
    am = prior.representations(a['comparison_text'])
    bm = prior.representations(b['comparison_text'])
    alias_equal = bool(a['laboratory'] and b['laboratory'] and am['meaning_tokens'] and (not eq['source_frame']['guarded_scope']) and (not eq['candidate_frame']['guarded_scope']) and (old.phrase(' '.join(am['meaning_tokens']))['key'] == old.phrase(' '.join(bm['meaning_tokens']))['key']))
    s = prior.support(text, label)
    protected = set(s['conflicts'] + s['unsupported_specificity'])
    for role in ('count', 'percentage', 'percent', 'concentration', 'total', 'conjugated', 'unconjugated', 'upper', 'lower'):
        x = bool(re.search('\\b' + role + '\\b', a['comparison_text']))
        y = bool(re.search('\\b' + role + '\\b', b['comparison_text']))
        if x != y:
            protected.add('unresolved_role:' + role)
    if a['course'] != b['course'] and (a['course'] or b['course']):
        protected.add('event_course_not_preserved')
    return {'equivalent': bool((eq['equivalent'] or alias_equal) and (not protected)), 'source': a, 'candidate': b, 'contextual_alias_equivalence': alias_equal, 'protected_differences': sorted(protected), 'grammar': eq, 'meaning_equivalence_is_not_identity': True}

def rank(text, pt_rows, groups, semantics, *, config=DEVELOPMENT):
    result = old.rank(text, pt_rows, groups, semantics, config=config)
    for pt, rows in result['groups'].items():
        literal = [r['candidate_id'] for r in rows if result['features'][r['candidate_id']]['wording_equivalent'] and (not result['features'][r['candidate_id']]['conflicts']) and (not result['features'][r['candidate_id']]['unsupported_specificity'])]
        for r in rows:
            f = result['features'][r['candidate_id']]
            c = correspondence(text, r['llt_name'])
            f['correspondence_v1_8'] = c
            f['original_v1_7_score'] = f['score']
            f['original_v1_7_equivalent'] = f['wording_equivalent']
            if config.representation and config.rerank and c['equivalent'] and (not f['conflicts']) and (not f['unsupported_specificity']):
                increment = max(0.0, 0.55 - f['contributions']['input_wording']) + max(0.0, 0.35 - f['contributions']['clinical_meaning'])
                if not f['original_v1_7_equivalent']:
                    increment = max(0.0, increment - 0.04)
                f['contributions']['bounded_correspondence'] = increment
                f['score'] = min(1.0, f['score'] + increment)
            if config.accept_correspondence and config.representation:
                f['wording_equivalent'] = f['wording_equivalent'] or (c['equivalent'] and (not literal))
            f['stronger_original_wording_candidates'] = literal
            peers = [x for x in rows if x['candidate_id'] != r['candidate_id'] and correspondence(text, x['llt_name'])['equivalent']]
            if config.representation and c['equivalent'] and peers and (not f['original_v1_7_equivalent']) and (not literal):
                f['unresolved_reference_wording_alternatives'] += [{'candidate_id': x['candidate_id'], 'llt_name': x['llt_name'], 'reason': 'multiple_bounded_correspondences_do_not_identify_one_LLT'} for x in peers]
        result['groups'][pt] = sorted(rows, key=lambda r: (-result['features'][r['candidate_id']]['score'], r['candidate_id']))
    result['version'] = '1.8.0'
    return result

def decide(inp, ranking, baseline, *, config=DEVELOPMENT):
    out, audit = old.decide(inp, ranking, baseline, config=config)
    if config.recover_pt and (not out['pt']):
        candidate, pa = prior.decide(inp, ranking, baseline, config=config)
        if candidate['pt']:
            out, audit = old.decide(inp, ranking, candidate, config=config)
            audit['PT_correspondence_recovery'] = pa
    audit['implementation_version'] = '1.8.0'
    audit['correspondence_configuration'] = config.to_dict()
    return (out, audit)
support = prior.support
