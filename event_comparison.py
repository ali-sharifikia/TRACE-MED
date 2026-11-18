"""Input-bound comparison of event arguments and course; opt-in, no labels/I/O."""
from runtime_resources import resource as asp_resource
import re
from dataclasses import dataclass, replace
from collections import Counter
from functools import lru_cache
import wording_priority as previous
from pt_reranking import ANATOMY_GROUPS
old = previous.previous

@dataclass(frozen=True)
class ComparisonConfig(previous.ControlledConfig):
    version: str = '1.9.0'
    name: str = 'v1.9_event_comparison'
    ordering: str = 'original'
    pt_direction_guard: bool = True
    comparison_ranking: bool = True
    comparison_acceptance: bool = True
    recover_pt: bool = False
    restrict_new_evidence: bool = True
LLT_ONLY = ComparisonConfig()
RANK_ONLY = replace(LLT_ONLY, name='v1.9_comparison_ranking', comparison_acceptance=False)
ACCEPT_ONLY = replace(LLT_ONLY, name='v1.9_comparison_acceptance', comparison_ranking=False)
PT_ONLY = replace(LLT_ONLY, name='v1.9_PT_convergence', comparison_ranking=False, comparison_acceptance=False, recover_pt=True)
COMBINED_PT = replace(LLT_ONLY, name='v1.9_event_comparison_PT', recover_pt=True)
UNRESTRICTED = replace(LLT_ONLY, name='v1.9_unrestricted_comparison', restrict_new_evidence=False)
DEVELOPMENT = COMBINED_PT
CONFIGS = {c.name: c for c in (RANK_ONLY, ACCEPT_ONLY, LLT_ONLY, PT_ONLY, COMBINED_PT, UNRESTRICTED)}
COURSE = {**old.COURSE, 'potentiation': 'aggravated', 'potentiated': 'aggravated', 'reactivation': 'reactivated', 'reactivated': 'reactivated'}
COURSE_PATTERN = '\\b(?:' + '|'.join(sorted(COURSE, key=lambda x: (-len(x), x))) + ')\\b'
UNSAFE = re.compile('\\b(?:and|or|without|with|after|before|following|due to|not|no|history)\\b|[;:]')
SITE = {w: sorted(g)[0] for g in ANATOMY_GROUPS if 'infusion' not in g for w in g}

@lru_cache(maxsize=65536)
def frame(text):
    base = old.representation(text)
    surface = base['surface']
    matches = list(re.finditer(COURSE_PATTERN, surface))
    safe = not UNSAFE.search(surface)
    body = surface
    course = None
    transforms = []
    if len(matches) == 1 and safe:
        m = matches[0]
        if m.start() == 0:
            body = re.sub('^\\s*(?:of\\s+)?', '', surface[m.end():])
            course = COURSE[m[0]]
        elif not surface[m.end():].strip(' ,'):
            body = surface[:m.start()].strip(' ,')
            course = COURSE[m[0]]
        if course:
            transforms.append({'rule': 'single_event_course_argument', 'source': surface, 'span': [m.start(), m.end()], 'supporting_text': m[0], 'course': course, 'argument': body})
    if matches and (not course):
        safe = False
    if old.old.RELATIONAL.search(body):
        safe = False
    argument_representation = old.representation(body)
    transformed = argument_representation['comparison_text']
    words = list(old.prior.representations(transformed)['wording_tokens'])
    normalized = []
    for w in words:
        value = SITE.get(w, w)
        normalized.append(value)
        if value != w:
            transforms.append({'rule': 'existing_anatomical_noun_adjective', 'resource': 'pt_reranking.ANATOMY_GROUPS (anatomy only)', 'source': surface, 'token': w, 'replacement': value, 'spans': [[m.start(), m.end()] for m in re.finditer('\\b' + re.escape(w) + '\\b', surface)]})
    key = old.old.phrase(' '.join(normalized))['key']
    if safe and len(normalized) == 2 and (normalized.count('fracture') == 1):
        key = ('single_fracture_argument', next((x for x in normalized if x != 'fracture')))
    return {'source': text, 'surface': surface, 'argument': body, 'key': key, 'course': course, 'safe': bool(safe), 'transformations': transforms, 'argument_transformations': argument_representation['transformations'], 'original_clinical': base['clinical'], 'laboratory': base['laboratory']}

@lru_cache(maxsize=131072)
def comparison(text, label):
    a, b = (frame(text), frame(label))
    s = old.support(text, label)
    body_equal = bool(a['key'] == b['key'] and a['argument'] and b['argument'])
    protected = list(s['conflicts']) + list(s['unsupported_specificity'])
    aa, bb = (old.representation(text), old.representation(label))
    for role in ('count', 'percentage', 'percent', 'concentration', 'total', 'conjugated', 'unconjugated', 'upper', 'lower'):
        if bool(re.search('\\b' + role + '\\b', aa['comparison_text'])) != bool(re.search('\\b' + role + '\\b', bb['comparison_text'])):
            protected.append('protected_role:' + role)
    if s['added_specimen']:
        protected.append('added_specimen')
    source_specimens = set(old.prior.representations(text)['meaning_tokens']) & old.prior.SPECIMEN
    target_specimens = set(old.prior.representations(label)['meaning_tokens']) & old.prior.SPECIMEN
    if source_specimens != target_specimens:
        protected.append('specimen_not_preserved')
    course_equal = a['course'] == b['course']
    eligible = a['safe'] and b['safe'] and body_equal and (not protected)
    measurement_context = any((f['laboratory'] or f['original_clinical']['direction'] or f['original_clinical']['measurement_type'] or f['original_clinical']['bounds'] or f['original_clinical']['units'] for f in (a, b)))
    return {'source_frame': a, 'candidate_frame': b, 'shared_argument': body_equal, 'protected_differences': sorted(set(protected)), 'omitted_course': a['course'] if a['course'] and (not b['course']) else None, 'added_course': b['course'] if b['course'] and (not a['course']) else None, 'conflicting_course': bool(a['course'] and b['course'] and (not course_equal)), 'equivalent': bool(eligible and course_equal), 'course_omission_competitor': bool(eligible and a['course'] and (not b['course'])), 'measurement_context': bool(measurement_context), 'new_event_evidence': bool(eligible and course_equal and (not measurement_context) and (not old.correspondence(text, label)['equivalent'])), 'authority': 'original input spans, bounded grammar, existing anatomical resource; not reviewer rationale'}

from decision_recorder import computation

@computation('ranking.event_comparison')
def rank(text, pt_rows, groups, semantics, *, config=DEVELOPMENT):
    result = previous.rank(text, pt_rows, groups, semantics, config=config)
    for pt, rows in result['groups'].items():
        fs = result['features']
        evidence = {r['candidate_id']: comparison(text, r['llt_name']) for r in rows}
        equivalents = [r['candidate_id'] for r in rows if evidence[r['candidate_id']]['equivalent'] and (not fs[r['candidate_id']]['conflicts']) and (not fs[r['candidate_id']]['unsupported_specificity'])]
        for row in rows:
            cid = row['candidate_id']
            f = fs[cid]
            ev = evidence[cid]
            f['comparison_v1_9'] = {**ev, 'equivalent_competitors': [x for x in equivalents if x != cid]}
            f['baseline_v1_8_1_score'] = f['score']
            f['baseline_v1_8_1_wording'] = f['wording_equivalent']
            f['acceptance_evidence']['pre_comparison_wording_authorized'] = f['wording_equivalent']
            authorized = ev['new_event_evidence'] if config.restrict_new_evidence else ev['equivalent']
            if config.comparison_ranking and authorized and (not f['conflicts']) and (not f['unsupported_specificity']):
                target = sum((v for k, v in f['contributions'].items() if k != 'bounded_correspondence'))
                target += max(0.0, 0.55 - f['contributions']['input_wording']) + max(0.0, 0.35 - f['contributions']['clinical_meaning'])
                if not f['original_v1_7_equivalent']:
                    target -= 0.04
                increment = max(0.0, target - f['score'])
                f['score'] = min(1.0, f['score'] + increment)
                f['contributions']['event_argument_correspondence'] = increment
            if config.comparison_ranking and ev['course_omission_competitor'] and equivalents:
                penalty = min(f['score'], f['contributions']['clinical_meaning'])
                f['score'] -= penalty
                f['contributions']['asserted_course_omission'] = -penalty
            if config.comparison_acceptance and authorized and (len(equivalents) == 1):
                f['wording_equivalent'] = True
            f['acceptance_evidence']['event_argument_comparison'] = bool(config.comparison_acceptance and authorized and (len(equivalents) == 1))
            f['acceptance_evidence']['wording_authorized'] = f['wording_equivalent']
            f['ordering_evidence'].update(strategy=config.name, score=f['score'], baseline_score=f['baseline_v1_8_1_score'])
        result['groups'][pt] = sorted(rows, key=lambda r: (-fs[r['candidate_id']]['score'], r['candidate_id']))
    return result

def decide(inp, ranking, baseline, *, config=DEVELOPMENT):
    out, audit = previous.decide(inp, ranking, baseline, config=config)
    audit['candidate_comparison_configuration'] = config.to_dict()
    audit['implementation_version'] = config.version
    if 'ordering_contract' in audit:
        audit['ordering_contract']['baseline_strategy'] = audit['ordering_contract']['strategy']
        audit['ordering_contract']['strategy'] = config.name
    return (out, audit)
support = old.support
