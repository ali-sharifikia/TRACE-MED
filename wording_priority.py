"""Controlled ordering versus correspondence authorization; no labels or codes."""
from runtime_resources import resource as asp_resource
import copy, re
from dataclasses import dataclass, replace
import clinical_discrimination as previous

@dataclass(frozen=True)
class ControlledConfig(previous.DiscriminationConfig):
    version: str = '1.8.1'
    name: str = 'v1.8.1_baseline_order'
    ordering: str = 'baseline'
    recover_pt: bool = False
    pt_direction_guard: bool = False
BASELINE_ORDER = ControlledConfig()
ORIGINAL = replace(BASELINE_ORDER, name='v1.8.1_original_wording', ordering='original')
PT_ONLY = replace(BASELINE_ORDER, name='v1.8.1_PT_guard_only', ordering='combined', recover_pt=True, pt_direction_guard=True)
COMBINED = replace(ORIGINAL, name='v1.8.1_original_wording_PT_guard', pt_direction_guard=True)
CONFIGS = {c.name: c for c in (BASELINE_ORDER, ORIGINAL, PT_ONLY, COMBINED)}
DEVELOPMENT = COMBINED

def rank(text, pt_rows, groups, semantics, *, config=DEVELOPMENT):
    cfg = replace(config, rerank=config.ordering != 'baseline')
    result = previous.rank(text, pt_rows, groups, semantics, config=cfg)
    representation = previous.prior.representations(text)
    abbreviations = {w for w in representation['wording_tokens'] if re.fullmatch('[a-z][a-z0-9]{1,4}', w) and w not in representation['meaning_tokens'] and (w not in previous.old.DIRECTION) and (w not in ('rise', 'raise', 'fall', 'level', 'value', 'total'))}
    for pt, rows in result['groups'].items():
        features = result['features']
        supported = [r['candidate_id'] for r in rows if features[r['candidate_id']]['original_v1_7_equivalent'] and (not features[r['candidate_id']]['conflicts']) and (not features[r['candidate_id']]['unsupported_specificity']) and (not features[r['candidate_id']]['unresolved_reference_wording_alternatives'])]
        preserved = [r['candidate_id'] for r in rows if abbreviations and abbreviations <= set(features[r['candidate_id']]['candidate_representation']['wording_tokens']) and features[r['candidate_id']]['correspondence_v1_8']['equivalent'] and features[r['candidate_id']]['correspondence_v1_8']['grammar']['equivalent'] and (not features[r['candidate_id']]['conflicts']) and (not features[r['candidate_id']]['unsupported_specificity']) and (not any((a['reason'] != 'multiple_bounded_correspondences_do_not_identify_one_LLT' for a in features[r['candidate_id']]['unresolved_reference_wording_alternatives'])))]
        alias_rival = any((features[r['candidate_id']]['correspondence_v1_8']['equivalent'] and (not abbreviations <= set(features[r['candidate_id']]['candidate_representation']['wording_tokens'])) for r in rows))
        preferred = supported if supported else preserved if alias_rival else []
        prefer = config.ordering == 'original' and len(preferred) == 1
        for r in rows:
            f = features[r['candidate_id']]
            if prefer:
                f['score'] = f['original_v1_7_score']
                f['contributions'].pop('bounded_correspondence', None)
                if not supported:
                    f['wording_equivalent'] = r['candidate_id'] == preferred[0]
                    if f['wording_equivalent']:
                        f['unresolved_reference_wording_alternatives'] = [a for a in f['unresolved_reference_wording_alternatives'] if a['reason'] != 'multiple_bounded_correspondences_do_not_identify_one_LLT']
            f['ordering_evidence'] = {'strategy': config.ordering, 'score': f['score'], 'original_score': f['original_v1_7_score'], 'original_wording_priority_applied': prefer, 'unique_original_wording_candidate': preferred[0] if prefer else None, 'wording_priority_basis': 'conservative_original' if supported else 'bounded_grammar_preserving_entity_wording', 'source_abbreviations': sorted(abbreviations), 'scope': 'whole current PT group; all competitors retained'}
            f['acceptance_evidence'] = {'bounded_correspondence': f['correspondence_v1_8']['equivalent'], 'original_equivalence': f['original_v1_7_equivalent'], 'wording_authorized': f['wording_equivalent'], 'extra_acceptance_score': None, 'score_scale': 'same as this variant ordering; no separate score substituted'}
        result['groups'][pt] = sorted(rows, key=lambda r: (-features[r['candidate_id']]['score'], r['candidate_id']))
    result['version'] = config.version
    return result

def direction_evidence(text, pt_name, competitors):
    """A result direction is not represented by its bare laboratory test concept.

    Require a supplied, supported directional counterpart of the same named
    measurement. This is not a blanket prohibition on terminology abstraction.
    """
    source = previous.representation(text)
    candidate = previous.representation(pt_name)
    interpreted = previous.prior.representations(source['comparison_text'])['clinical']
    direction = interpreted['direction']
    alternatives = []
    scope_safe = not re.search('\\b(?:and|or|without|with|after|before|not|no)\\b|[;,]', source['surface'])
    if scope_safe and source['laboratory'] and (direction in ('increased', 'decreased')) and (not candidate['clinical']['direction']):
        core = previous.prior.representations(pt_name)['meaning_tokens']
        input_tokens = previous.prior.representations(source['comparison_text'])['meaning_tokens']
        for row in competitors:
            r = previous.representation(row['pt_name'])
            tokens = previous.prior.representations(row['pt_name'])['meaning_tokens']
            remaining = [t for t in tokens if t not in ('increased', 'decreased')]
            s = previous.support(text, row['pt_name'])
            if r['clinical']['direction'] == direction and tuple(remaining) == tuple(core) and all((t in input_tokens for t in core)) and (not s['conflicts']) and (not s['unsupported_specificity']):
                alternatives.append(row['pt_code'])
    return {'source': source, 'interpreted_direction': direction, 'candidate': candidate, 'directional_counterparts': sorted(alternatives), 'blocked': bool(alternatives), 'authority': 'input direction and supplied same-measurement counterpart; not reviewer rationale'}

def decide(inp, ranking, baseline, *, config=DEVELOPMENT):
    out, audit = previous.decide(inp, ranking, baseline, config=config)
    audit['selection_origin'] = 'preserved_actual_base_decision' if out['llt'] and out['llt'] == baseline.get('llt', []) else 'current_variant_leader_or_deferral'
    if config.pt_direction_guard and len(out['pt']) == 1 and (inp.get('route') != 'exact') and (inp.get('structure') == 'single_concept'):
        selected = next((r for r in ranking['pt_rankings'] if r['pt_code'] == out['pt'][0]), None)
        if selected:
            evidence = direction_evidence(inp['original_input'], selected['pt_name'], ranking['pt_rankings'])
            audit['PT_direction_guard'] = evidence
            if evidence['blocked']:
                out = {'pt': [], 'llt': [], 'review_pt': [r['pt_code'] for r in ranking['pt_rankings'][:4]], 'review_llt': [r['llt_code'] for p in ranking['pt_rankings'][:4] for r in ranking['groups'].get(p['pt_code'], [])[:4]]}
                audit.update(applied=True, path='unrepresented_measurement_direction', pt_reasons=audit['pt_reasons'] + ['asserted_direction_has_supported_current_counterpart'])
    if len(out['pt']) == 1:
        rows = ranking['groups'].get(out['pt'][0], [])
        if rows:
            fs = ranking['features']
            leader = rows[0]['candidate_id']
            score = fs[leader]['score']
            rival = max((fs[r['candidate_id']]['score'] for r in rows[1:]), default=0.0)
            audit['ordering_contract'] = {'strategy': config.ordering, 'leader': leader, 'leader_score': score, 'maximum_competitor_score': rival, 'margin': score - rival, 'ids': [r['candidate_id'] for r in rows], 'score_is_maximum': all((score >= fs[r['candidate_id']]['score'] for r in rows)), 'tied_leaders': [r['candidate_id'] for r in rows if fs[r['candidate_id']]['score'] == score], 'acceptance_evidence': fs[leader]['acceptance_evidence']}
            if audit.get('LLT_selected') == leader[4:]:
                assert abs(audit['LLT_margin'] - (score - rival)) < 1e-12
            if config.ordering == 'baseline' and out['llt'] and (out['llt'] != baseline.get('llt', [])):
                assert out['llt'] == [leader[4:]], 'Controlled ordering cannot select a lower-ranked alternative'
    audit['implementation_version'] = config.version
    return (out, audit)
support = previous.support
