"""Opt-in input-grounded PT/LLT discrimination. Pure runtime code; no labels/I/O."""
from runtime_resources import resource as asp_resource
from dataclasses import dataclass, asdict, replace
from functools import lru_cache
from difflib import SequenceMatcher
import re
import unicodedata
from math import isfinite
from biomedical_wording import canonical_biomedical_wording, generic_context_base
from clinical_features import clinical_scoring_text, interpreted_features, feature_conflicts, automatic_mapping_uncertainties
from terminology_aliases import SAFE_BIOMEDICAL_ALIASES
from compound_validation import component_semantic_issues, ANATOMY, SITE_EQUIVALENTS, MALIGNANCY
from pt_reranking import explicit_clinical_conflicts
from joint_ranking import transition, FORMS, STOP
VERSION = '1.6.0'

@dataclass(frozen=True)
class DevelopmentConfig:
    version: str = VERSION
    enabled: bool = True
    wording: bool = True
    specificity: bool = True
    pt_ranking: bool = True
    semantic: bool = True
    maximum_expanded_pt_groups: int = 10
    maximum_reference_candidates: int = 1500
    pt_score: float = 0.5
    pt_margin: float = 0.001
    llt_score: float = 0.7
    llt_margin: float = 0.01
    llt_path: str = 'ordered_grammar'

    def __post_init__(self):
        if self.llt_path not in {'ordered_grammar', 'supported_wording'}:
            raise ValueError('Unknown development LLT acceptance path')
        if not 1 <= self.maximum_expanded_pt_groups <= 10 or not 1 <= self.maximum_reference_candidates <= 1500:
            raise ValueError('Reference budget outside declared bounds')
        if any((not isfinite(v) or not 0 <= v <= 1 for v in (self.pt_score, self.pt_margin, self.llt_score, self.llt_margin))):
            raise ValueError('Invalid development threshold')

    def to_dict(self):
        return asdict(self)
DEVELOPMENT = DevelopmentConfig()
DISABLED = replace(DEVELOPMENT, enabled=False)
GRAMMAR = {'increases': 'increase', 'increased': 'increase', 'decreases': 'decrease', 'decreased': 'decrease', 'elevations': 'elevation', 'reductions': 'reduction', 'fistulas': 'fistula', 'disorders': 'disorder', 'infections': 'infection', 'tumors': 'tumor', 'neoplasms': 'neoplasm', 'tests': 'test', 'cysts': 'cyst', 'eyes': 'eye', 'ears': 'ear', 'arms': 'arm', 'legs': 'leg', 'knees': 'knee', 'hips': 'hip', 'malignancies': 'malignancy', 'values': 'value', 'levels': 'level', 'counts': 'count', 'changes': 'change'}
ALIASES = {**SAFE_BIOMEDICAL_ALIASES, 't4': 'thyroxine', 't3': 'triiodothyronine', 'egfr': 'estimated glomerular filtration rate', 'gfr': 'glomerular filtration rate'}
WEAK = {'level', 'value', 'finding', 'disorder', 'disease', 'event', 'effect', 'reaction', 'test'}
PROTECTED = {'developmental', 'congenital', 'hereditary', 'traumatic', 'pathological', 'acute', 'chronic', 'recurrent', 'severe', 'mild', 'moderate', 'primary', 'secondary', 'neonatal', 'juvenile', 'senile', 'malignant', 'metastatic'}
SPECIMEN = {'blood', 'serum', 'plasma', 'urine'}

def surface(text):
    return ' '.join(unicodedata.normalize('NFKC', text).casefold().split())

@lru_cache(maxsize=65536)
def representations(text):
    original = surface(text)
    bounded = original
    words = {'one': '1', 'two': '2', 'three': '3', 'four': '4', 'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9', 'ten': '10'}
    bounded = re.sub('\\b(' + '|'.join(words) + ')\\s+times\\s+(?:the\\s+)?(?=uln|lln|upper limit|lower limit)', lambda m: words[m[1]] + ' * ', bounded)
    bounded = re.sub('\\b(above|below)\\s+(?:the\\s+)?(upper|lower) limit of normal', lambda m: ('> ' if m[1] == 'above' else '< ') + '1 * ' + ('uln' if m[2] == 'upper' else 'lln'), bounded)
    clinical = interpreted_features(bounded)
    measurement = re.search('\\b(fall|drop|decline|rise)\\s+in\\s+(?:the\\s+)?(?:estimated|measured|blood|serum|plasma|urine|level|count|rate|gfr|egfr)\\b', original)
    meaning_source = generic_context_base(bounded) or bounded
    if measurement:
        direction = 'increased' if measurement[1] == 'rise' else 'decreased'
        meaning_source = bounded[:measurement.start(1)] + direction + bounded[measurement.end(1):]
    lexical = clinical_scoring_text(meaning_source)
    raw = clinical_scoring_text(bounded)
    tokens = tuple((GRAMMAR.get(w, w) for w in re.findall('[a-z]+[0-9]*(?:-[a-z]+)?', raw) if w not in STOP))
    meaning = canonical_biomedical_wording(lexical)
    for a, b in sorted(ALIASES.items(), key=lambda p: -len(p[0])):
        meaning = re.sub('\\b' + re.escape(a) + '\\b', b, meaning)
    meaning = meaning.replace('oesophag', 'esophag')
    mt = tuple((FORMS.get(GRAMMAR.get(w, w), GRAMMAR.get(w, w)) for w in re.findall('[a-z]+[0-9]*(?:-[a-z]+)?', meaning) if w not in STOP))
    return {'original': text, 'wording_tokens': tokens, 'meaning_tokens': mt, 'clinical': interpreted_features(meaning_source), 'original_clinical': clinical, 'meaning_source': meaning_source, 'measurement_change_span': [measurement.start(), measurement.end()] if measurement else None, 'provenance': 'bounded grammar/biomedical aliases plus existing clinical_features; numbers retained separately', 'transition': transition(text)}

def fit(a, b):
    a, b = (set(a), set(b))
    weight = lambda w: 0.15 if w in WEAK else 1.0
    n = sum((weight(w) for w in a & b))
    den = sum((weight(w) for w in a)) + sum((weight(w) for w in b))
    return 2 * n / den if den else 0.0

@lru_cache(maxsize=131072)
def support(text, label):
    a, b = (representations(text), representations(label))
    at, bt = (set(a['meaning_tokens']), set(b['meaning_tokens']))
    conflicts = set(explicit_clinical_conflicts(a['meaning_source'], label)) | set(feature_conflicts(a['meaning_source'], label))
    conflicts.update(component_semantic_issues(' '.join(a['meaning_tokens']), ' '.join(b['meaning_tokens'])))
    unsupported = (bt & PROTECTED) - at
    if at & MALIGNANCY:
        unsupported.discard('malignant')
    if 'time' in bt and 'bleeding' in bt and (not {'time', 'measurement', 'assay', 'test'} & at):
        unsupported.add('measurement_time')
    if 'count' in bt and (not {'count', 'cell', 'platelet', 'lymphocyte', 'neutrophil', 'leukocyte', 'blood'} & at) and (not a['clinical']['laboratory_context']) and (not a['clinical']['analytes']):
        unsupported.add('cell_count_measurement')
    sites = {SITE_EQUIVALENTS.get(t, t) for t in bt & ANATOMY}
    source_sites = {SITE_EQUIVALENTS.get(t, t) for t in at & ANATOMY}
    if sites - source_sites:
        unsupported.update(('anatomy:' + x for x in sites - source_sites))
    if a['measurement_change_span'] and bt <= {'fall', 'falling'}:
        conflicts.add('measurement_change_not_physical_event')
    if a['transition'] and b['transition'] and all(a['transition'].values()) and all(b['transition'].values()) and (a['transition'] != b['transition']):
        conflicts.add('polarity_transition_conflict')
    unrepresented = []
    if a['clinical']['bounds'] and (not b['clinical']['bounds']):
        unrepresented.append('input_numeric_bound_not_encoded_in_label')
    if 'total' in at and 'total' not in bt:
        unrepresented.append('total_not_encoded_in_label')
    if a['transition'] and (not b['transition']):
        unrepresented.append('transition_not_encoded_in_label')
    added_specimen = sorted((bt & SPECIMEN) - at)
    attrs = []
    for word in sorted(bt & PROTECTED):
        spans = [[m.start(), m.end()] for m in re.finditer('\\b' + re.escape(word) + '\\b', text, re.I)]
        attrs.append({'attribute': word, 'status': 'explicit_support' if spans else 'bounded_equivalence' if word in at else 'unsupported_added_specificity', 'source_spans': spans, 'source_text': text})
    return {'conflicts': sorted(conflicts), 'unsupported_specificity': sorted(unsupported), 'attributes': attrs, 'unrepresented_input_information': unrepresented, 'added_specimen': added_specimen, 'meaning_fit': fit(at, bt), 'source_features': a['clinical'], 'candidate_features': b['clinical'], 'provenance': 'input_discrimination_v1_6.support; label omission is not an explicit contradiction'}

def llt_features(text, row, semantic=None, config=DEVELOPMENT):
    if semantic is not None and (not isfinite(semantic) or not -1 <= semantic <= 1):
        raise ValueError('Invalid candidate-specific semantic evidence')
    a, b = (representations(text), representations(row['llt_name']))
    s = support(text, row['llt_name'])
    parent = support(text, row['pt_name'])
    wording = fit(a['wording_tokens'], b['wording_tokens'])
    meaning = s['meaning_fit']
    contributions = {'input_wording': 0.55 * wording if config.wording else 0.55 * meaning, 'clinical_meaning': 0.35 * meaning, 'candidate_semantic': 0.1 * semantic if config.semantic and semantic is not None else 0.0, 'added_specimen': -0.06 * len(s['added_specimen']) if config.specificity else 0.0, 'added_NOS': -0.08 * bool(set(b['wording_tokens']) & {'nos', 'unspecified', 'nonspecific'} - set(a['wording_tokens'])) if config.specificity else 0.0}
    score = sum(contributions.values())
    conflicts = sorted(set(s['conflicts'] + parent['conflicts']))
    unsupported = sorted(set(s['unsupported_specificity'] + parent['unsupported_specificity']))
    if conflicts or (config.specificity and unsupported):
        score = 0.0
    equal = bool(a['wording_tokens'] and set(a['wording_tokens']) == set(b['wording_tokens']))
    return {'score': max(0.0, min(1.0, score)), 'contributions': contributions, 'wording_fit': wording, 'meaning_fit': meaning, 'wording_equivalent': equal, 'semantic': semantic, 'semantic_available': semantic is not None, 'support': s, 'parent_support': parent, 'conflicts': conflicts, 'unsupported_specificity': unsupported, 'input_representation': a, 'candidate_representation': b, 'parent_name_equal': surface(row['llt_name']) == surface(row['pt_name']), 'parent_equality_weight': 0.0}

def rank(text, pt_rows, groups, semantics, *, config=DEVELOPMENT):
    if len({r['pt_code'] for r in pt_rows}) != len(pt_rows):
        raise ValueError('Duplicate original PT competitors')
    ranked = {}
    features = {}
    for pt, rows in groups.items():
        if len({r['llt_code'] for r in rows}) != len(rows):
            raise ValueError('Duplicate LLT identities')
        for row in rows:
            if row['pt_code'] != pt or row['candidate_id'] != 'LLT:' + row['llt_code']:
                raise ValueError('Invalid current parent')
            features[row['candidate_id']] = llt_features(text, row, semantics.get(row['candidate_id']), config)
        ranked[pt] = sorted(rows, key=lambda r: (-features[r['candidate_id']]['score'], r['candidate_id']))
    anchors = {p for p, rr in ranked.items() if any((features[r['candidate_id']]['wording_equivalent'] and (not features[r['candidate_id']]['conflicts']) and (not features[r['candidate_id']]['unsupported_specificity']) for r in rr))}
    pts = []
    for row in pt_rows:
        pt = row['pt_code']
        s = support(text, row['pt_name'])
        old = row.get('unrounded_reranking_score', row.get('reranking_score'))
        best = ranked.get(pt, [])
        best_support = max((features[r['candidate_id']]['meaning_fit'] for r in best if not features[r['candidate_id']]['conflicts'] and (not features[r['candidate_id']]['unsupported_specificity'])), default=0.0)
        score = old
        if config.pt_ranking and old is not None:
            score = max(old, 0.98) if anchors == {pt} and (not representations(text)['clinical']['bounds']) else old
            if s['conflicts'] or (config.specificity and s['unsupported_specificity']):
                score = 0.0
        pts.append({**row, 'original_score': old, 'score': score, 'input_support': s, 'best_LLT_meaning_fit': best_support})
    pts.sort(key=lambda r: (-(r['score'] if r['score'] is not None else -1), r['pt_code']))
    return {'version': VERSION, 'configuration': config.to_dict(), 'pt_rankings': pts, 'groups': ranked, 'features': features, 'original_competitor_codes': [r['pt_code'] for r in pt_rows], 'pt_ties': [r['pt_code'] for r in pts if pts and r['score'] == pts[0]['score']]}

def decide(input_evidence, ranking, baseline, *, config=DEVELOPMENT):
    """Development decisions; caller supplies verified reference and baseline gate output.

    Compound decisions remain with the established component gate. This path
    never accepts a subset or manufactures reference completeness.
    """
    text = input_evidence['original_input']
    out = {k: list(baseline.get(k, [])) for k in ('pt', 'llt', 'review_pt', 'review_llt')}
    audit = {'version': VERSION, 'configuration': config.to_dict(), 'input': text, 'authority': 'deterministic_input_discrimination', 'reviewer_provenance': input_evidence.get('reviewer_provenance'), 'reviewer_agreement_is_independent_evidence': False, 'input_binding': {k: input_evidence.get(k) for k in ('normalized_input', 'structure', 'input_bound', 'force_review', 'exact_collision')}, 'PT_competitors': [{'pt_code': p['pt_code'], 'pt_name': p['pt_name'], 'score': p['score'], 'original_score': p['original_score']} for p in ranking['pt_rankings']], 'pt_reasons': [], 'llt_reasons': [], 'path': 'preserved_production', 'applied': False}
    if not config.enabled:
        return (out, audit)
    if input_evidence.get('route') == 'exact':
        return (out, {**audit, 'path': 'historical_exact_preserved'})
    if input_evidence.get('force_review') or input_evidence.get('exact_collision'):
        return ({**out, 'pt': [], 'llt': []}, {**audit, 'path': 'forced_deferral_preserved'})
    if input_evidence.get('structure') != 'single_concept' or not input_evidence.get('input_bound'):
        return (out, {**audit, 'path': 'existing_compound_or_interpretation_gate_only'})
    pts = ranking['pt_rankings']
    by_pt = {r['pt_code']: r for r in pts}
    accepted = out['pt'][0] if len(out['pt']) == 1 else None
    if accepted:
        row = by_pt.get(accepted)
        s = row['input_support'] if row else input_evidence.get('baseline_support')
        audit['baseline_PT_support'] = s
        if not s or s['conflicts'] or s['unsupported_specificity']:
            audit['pt_reasons'].append('baseline_PT_support_not_established')
            out['pt'], out['llt'] = ([], [])
            accepted = None
            audit['applied'] = True
    if not accepted and len(pts) > 1:
        if any((not isinstance(p['score'], (int, float)) or not isfinite(p['score']) or (not 0 <= p['score'] <= 1) for p in pts)):
            return ({**out, 'pt': [], 'llt': []}, {**audit, 'pt_reasons': audit['pt_reasons'] + ['missing_or_invalid_distinct_competitor_score']})
        top = pts[0]
        margin = top['score'] - pts[1]['score']
        s = top['input_support']
        agreement = input_evidence.get('reviewer_resolved') and input_evidence.get('reviewer_parents') == [top['pt_code']]
        audit.update(PT_selected=top['pt_code'], PT_score=top['score'], PT_margin=margin, competitor=pts[1]['pt_code'], reviewer_agreement=bool(agreement), PT_support=s)
        anchors = {p for p, rows in ranking['groups'].items() if any((ranking['features'][r['candidate_id']]['wording_equivalent'] and (not ranking['features'][r['candidate_id']]['conflicts']) and (not ranking['features'][r['candidate_id']]['unsupported_specificity']) for r in rows))}
        anchor = anchors == {top['pt_code']} and (not representations(text)['clinical']['bounds'])
        if top['score'] >= config.pt_score and margin > 0 and (margin >= config.pt_margin) and (agreement or anchor) and (not s['conflicts']) and (not s['unsupported_specificity']) and (not automatic_mapping_uncertainties(representations(text)['meaning_source'])) and (top['pt_code'] in input_evidence['complete_groups']) and (not input_evidence.get('retrieval_conflicts', {}).get(top.get('best_candidate_id'))):
            accepted = top['pt_code']
            out['pt'] = [accepted]
            out['llt'] = []
            audit.update(applied=True, path='input_wording_convergence' if anchor else 'supported_reranking_with_recorded_agreement')
    if not accepted:
        out.update(pt=[], llt=[])
        return (out, audit)
    rows = ranking['groups'].get(accepted, [])
    if out['llt']:
        selected = [ranking['features'].get('LLT:' + x) for x in out['llt']]
        if all((f and (not f['conflicts']) and (not f['unsupported_specificity']) for f in selected)):
            return (out, audit)
        out['llt'] = []
        audit['llt_reasons'].append('previous_LLT_support_not_established')
        audit['applied'] = True
    if not rows:
        return (out, {**audit, 'llt_reasons': audit['llt_reasons'] + ['missing_current_group']})
    first = rows[0]
    f = ranking['features'][first['candidate_id']]
    second = ranking['features'][rows[1]['candidate_id']]['score'] if len(rows) > 1 else 0.0
    margin = f['score'] - second
    audit.update(LLT_selected=first['llt_code'], LLT_score=f['score'], LLT_margin=margin, LLT_features=f, LLT_competitors=[{'id': r['candidate_id'], 'score': ranking['features'][r['candidate_id']]['score']} for r in rows[1:]])
    reasons = audit['llt_reasons']
    if accepted not in input_evidence['complete_groups']:
        reasons.append('full_reference_group_incomplete')
    if f['conflicts'] or f['unsupported_specificity']:
        reasons.append('clinical_or_specificity_rejection')
    if margin <= 0 or margin < config.llt_margin:
        reasons.append('tie_or_insufficient_wording_margin')
    equivalent = [r for r in rows if ranking['features'][r['candidate_id']]['wording_equivalent'] and (not ranking['features'][r['candidate_id']]['conflicts']) and (not ranking['features'][r['candidate_id']]['unsupported_specificity'])]
    if len(equivalent) > 1:
        reasons.append('wording_equivalence_collision_semantic_not_a_tie_breaker')
    if f['score'] < config.llt_score:
        reasons.append('insufficient_input_score')
    if not f['wording_equivalent']:
        reasons.append('approximate_wording_not_yet_supported_for_automatic_LLT')
    if config.llt_path == 'ordered_grammar' and f['input_representation']['wording_tokens'] != f['candidate_representation']['wording_tokens']:
        reasons.append('reordered_wording_remains_experimental')
    if automatic_mapping_uncertainties(representations(text)['meaning_source']):
        reasons.append('numerical_meaning_unresolved')
    if 'transition_not_encoded_in_label' in f['support']['unrepresented_input_information']:
        reasons.append('transition_unresolved')
    if input_evidence.get('retrieval_conflicts', {}).get(first['candidate_id']):
        reasons.append('saved_retrieval_conflict')
    if not reasons:
        out.update(llt=[first['llt_code']], review_pt=[], review_llt=[])
        audit.update(applied=True, path='input_supported_current_LLT_wording')
    else:
        out.update(llt=[], review_pt=[accepted], review_llt=[r['llt_code'] for r in rows[:4] if not ranking['features'][r['candidate_id']]['conflicts'] and (not ranking['features'][r['candidate_id']]['unsupported_specificity'])])
    return (out, audit)
