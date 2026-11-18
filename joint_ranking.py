"""Bounded, label-free cross-PT/LLT ranking. No model or reference I/O here."""
from runtime_resources import resource as asp_resource
from dataclasses import dataclass, asdict, replace
from difflib import SequenceMatcher
from functools import lru_cache
import re
import unicodedata
from biomedical_wording import canonical_biomedical_wording, generic_context_base
from clinical_features import interpreted_features, feature_conflicts, automatic_mapping_uncertainties
from compound_validation import component_semantic_issues
from llt_reranking import normalize_llt_text
from pt_reranking import explicit_clinical_conflicts
from coverage_policy import DEFAULT_COVERAGE, assess
VERSION = '1.5.0'
V1_4_COVERAGE = DEFAULT_COVERAGE
DEVELOPMENT_COVERAGE = replace(V1_4_COVERAGE, pt_unagreed_score=None, pt_unagreed_margin=None)
RELEASE_COVERAGE = replace(DEVELOPMENT_COVERAGE, version=VERSION)

@dataclass(frozen=True)
class RankingConfig:
    version: str = VERSION
    enabled: bool = True
    base_pt_groups: int = 4
    maximum_pt_groups: int = 6
    pt_expansion_delta: float = 0.2
    maximum_reference_candidates: int = 1500
    alias_weight: float = 0.35
    wording_weight: float = 0.3
    order_weight: float = 0.2
    parent_weight: float = 0.1
    semantic_weight: float = 0.05
    old_pt_weight: float = 0.35
    added_nos_penalty: float = 0.1
    missing_qualifier_penalty: float = 0.12
    pt_strategy: str = 'surface_anchor'

    def __post_init__(self):
        if self.pt_strategy not in {'surface_anchor', 'experimental_blend'}:
            raise ValueError('Unknown PT ranking strategy')
        if not 1 <= self.base_pt_groups <= self.maximum_pt_groups <= 10 or self.maximum_reference_candidates < 1:
            raise ValueError('Invalid bounded reference expansion')

    def to_dict(self):
        return asdict(self)
DEFAULT_RANKING = RankingConfig()
DISABLED_RANKING = replace(DEFAULT_RANKING, enabled=False)
STOP = {'a', 'an', 'the', 'of', 'in', 'on', 'at', 'for', 'to', 'and', 'or', 'with'}
GENERIC = {'event', 'reaction', 'effect', 'disorder', 'disease', 'finding', 'symptom', 'condition', 'test'}
FORMS = {'increases': 'increased', 'increase': 'increased', 'elevated': 'increased', 'elevation': 'increased', 'elevations': 'increased', 'raised': 'increased', 'decrease': 'decreased', 'decreases': 'decreased', 'reduced': 'decreased', 'reduction': 'decreased', 'worsened': 'worsening', 'aggravated': 'worsening', 'aggravation': 'worsening', 'prolongation': 'prolonged', 'prolonging': 'prolonged', 'shortening': 'shortened', 'impairment': 'impaired', 'impair': 'impaired', 'impairing': 'impaired', 'visual': 'vision', 'swollen': 'swelling', 'abnormalities': 'abnormal', 'platelets': 'platelet', 'triglycerides': 'triglyceride', 'enzymes': 'enzyme', 'symptoms': 'symptom', 'signs': 'sign', 'haematocrit': 'hematocrit', 'haemoglobin': 'hemoglobin', 'haemorrhage': 'hemorrhage', 'unspecified': 'nos', 'nonspecific': 'nos'}
QUALIFIERS = {'high', 'low', 'increased', 'decreased', 'positive', 'negative', 'normal', 'abnormal', 'prolonged', 'shortened', 'worsening', 'progression', 'acute', 'chronic', 'recurrent', 'severe', 'mild', 'moderate', 'left', 'right', 'bilateral', 'localised', 'localized', 'systemic', 'congenital', 'hereditary', 'primary', 'secondary', 'nos'}

@lru_cache(maxsize=65536)
def surface(text):
    return ' '.join(unicodedata.normalize('NFKC', text).casefold().split())

@lru_cache(maxsize=65536)
def wording(text, representation=True):
    value = canonical_biomedical_wording(generic_context_base(text) or text)
    words = re.findall('[a-z]+', value)
    return tuple((FORMS.get(w, w) if representation else w for w in words if w not in STOP))

def weighted_f1(a, b):
    a, b = (set(a), set(b))
    weight = lambda w: 0.2 if w in GENERIC else 1.5 if w in QUALIFIERS else 1.0
    overlap = sum((weight(w) for w in a & b))
    total = sum((weight(w) for w in a)) + sum((weight(w) for w in b))
    return 2 * overlap / total if total else 0.0

def transition(text):
    value = surface(text)
    match = re.search('\\b(negative|positive)\\b.*?\\bto\\s+(?:a\\s+)?(negative|positive)\\b', value)
    if match:
        return {'from': match[1], 'to': match[2]}
    if re.search('\\b(?:seroconversion|seroconverted|conversion)\\b', value):
        return {'from': None, 'to': None}
    return None

def pair_features(text, row, *, semantic=None, config=DEFAULT_RANKING, representation=True):
    original = wording(text, representation)
    label = wording(row.llt_name, representation)
    parent = wording(row.pt_name, representation)
    alias = weighted_f1(normalize_llt_text(text).split(), normalize_llt_text(row.llt_name).split())
    wording_fit = weighted_f1(original, label)
    order = SequenceMatcher(None, ' '.join(original), ' '.join(label), autojunk=False).ratio()
    parent_fit = weighted_f1(original, parent)
    conflicts = set(explicit_clinical_conflicts(text, row.llt_name)) | set(explicit_clinical_conflicts(text, row.pt_name))
    conflicts.update(feature_conflicts(text, row.llt_name))
    conflicts.update(feature_conflicts(text, row.pt_name))
    conflicts.update(component_semantic_issues(text, row.llt_name + ' ' + row.pt_name))
    source_transition, target_transition = (transition(text), transition(row.llt_name))
    if source_transition and target_transition and all(source_transition.values()) and all(target_transition.values()) and (source_transition != target_transition):
        conflicts.add('polarity_transition_conflict')
    lost_transition = bool(source_transition and (not target_transition))
    a, b = (set(original), set(label))
    missing = sorted((a & QUALIFIERS) - b)
    added = sorted((b & QUALIFIERS) - a)
    contributions = {'clinical_alias_fit': config.alias_weight * alias, 'surface_wording_fit': config.wording_weight * wording_fit, 'word_order_fit': config.order_weight * order, 'parent_wording_fit': config.parent_weight * parent_fit, 'saved_semantic': config.semantic_weight * (semantic if semantic is not None else 0.0), 'missing_qualifier': -min(0.3, config.missing_qualifier_penalty * len(missing)), 'added_nos': -config.added_nos_penalty * ('nos' in added), 'lost_transition': -0.2 * lost_transition}
    literal = surface(text) == surface(row.llt_name)
    score = max(0.0, min(1.0, sum(contributions.values())))
    if literal:
        score = 1.0
    if conflicts:
        score = 0.0
    return {'score': score, 'literal_surface': literal, 'safe_wording_equal': bool(original and original == label), 'alias_equal': normalize_llt_text(text) == normalize_llt_text(row.llt_name), 'semantic_similarity': semantic, 'semantic_available': semantic is not None, 'evidence_scope': 'saved_semantic_and_local_lexical' if semantic is not None else 'local_lexical_and_reference_only', 'missing_qualifiers': missing, 'added_qualifiers': added, 'transition_lost': lost_transition, 'input_transition': source_transition, 'candidate_transition': target_transition, 'conflicts': sorted(conflicts), 'contributions': contributions, 'literal_override': literal and (not conflicts), 'input_numeric': interpreted_features(text), 'candidate_numeric': interpreted_features(row.llt_name)}

def plausible_groups(evidence, config=DEFAULT_RANKING):
    if len(evidence.component_rankings) != 1:
        return []
    rows = evidence.component_rankings[0]['pt_rankings']
    if not rows:
        return []
    top = rows[0]['unrounded_reranking_score']
    return [r['pt_code'] for i, r in enumerate(rows[:config.maximum_pt_groups]) if i < config.base_pt_groups or top - r['unrounded_reranking_score'] <= config.pt_expansion_delta]

def rank_pairs(evidence, full, semantics, *, config=DEFAULT_RANKING, representation=True, literal_current_ids=None):
    """Rank complete supplied PT pools; semantic evidence is candidate-keyed.

    Code ordering provides stable display only; equal clinical scores share rank.
    Missing/oversized pools are explicitly excluded, not called full reference.
    """
    groups = {}
    features = {}
    exclusions = []
    budget = 0
    old = evidence.component_rankings[0]['pt_rankings'] if len(evidence.component_rankings) == 1 else []
    for pt in plausible_groups(evidence, config):
        rows = full.get(pt, [])
        if not rows or budget + len(rows) > config.maximum_reference_candidates:
            exclusions.append({'pt_code': pt, 'reason': 'missing_or_over_budget_full_PT_reference'})
            continue
        if len({r.llt_code for r in rows}) != len(rows) or any((r.pt_code != pt or r.candidate_id != 'LLT:' + r.llt_code for r in rows)):
            raise ValueError('Invalid current-reference candidate/parent pool')
        budget += len(rows)
        scored = []
        for row in rows:
            f = pair_features(evidence.input_text, row, semantic=semantics.get(row.candidate_id), config=config, representation=representation)
            features[row.candidate_id] = f
            scored.append(replace(row, score=f['score'], unrounded_score=f['score']))
        scored.sort(key=lambda r: (-r.score, r.candidate_id))
        rank = 0
        previous = None
        groups[pt] = []
        for i, row in enumerate(scored):
            if row.score != previous:
                rank = i + 1
            groups[pt].append(replace(row, rank=rank, score_gap_from_previous=None if previous is None else previous - row.score))
            previous = row.score
    anchors = {pt for pt, rows in groups.items() if any((features[r.candidate_id]['safe_wording_equal'] and (not features[r.candidate_id]['conflicts']) and (not features[r.candidate_id]['input_numeric'].get('bounds')) and (not automatic_mapping_uncertainties(evidence.input_text)) for r in rows))}
    anchor = next(iter(anchors)) if len(anchors) == 1 else None
    ranked = []
    for row in old:
        if row['pt_code'] not in groups:
            continue
        best = groups[row['pt_code']][0]
        score = config.old_pt_weight * row['unrounded_reranking_score'] + (1 - config.old_pt_weight) * best.score if config.pt_strategy == 'experimental_blend' else 1.0 if row['pt_code'] == anchor else row['unrounded_reranking_score']
        ranked.append({**row, 'reranking_score': score, 'unrounded_reranking_score': score, 'joint_best_candidate_id': best.candidate_id, 'original_pt_score': row['unrounded_reranking_score'], 'surface_anchor': row['pt_code'] == anchor})
    ranked.sort(key=lambda r: (-r['unrounded_reranking_score'], r['pt_code']))
    represented = {r['pt_code'] for r in ranked}
    ranked.extend((r for r in old if r['pt_code'] not in represented))
    ranked.sort(key=lambda r: (-r['unrounded_reranking_score'], r['pt_code']))
    return {'version': VERSION, 'configuration': config.to_dict(), 'groups': groups, 'features': features, 'pt_rankings': ranked, 'exclusions': exclusions, 'reference_candidates_processed': budget, 'representation_enabled': representation, 'literal_current_ids': literal_current_ids}

def assess_release(evidence, context, ranking, baseline, *, coverage=RELEASE_COVERAGE):
    """Deploy only the supported narrow discrimination path, not the blend.

    Existing automatic decisions/independent PTs remain authoritative. New LLTs
    require a unique literal current surface AND all the original acceptance
    checks. Approximate ranking is review/experiment evidence, not an arbitrary
    new strict LLT choice. Gold labels are never accepted by this interface.
    """
    audit = {'version': VERSION, 'configuration': ranking['configuration'], 'authority': 'deterministic_pair_ranking', 'input_text': evidence.input_text, 'pt_rankings': ranking['pt_rankings'], 'exclusions': ranking['exclusions'], 'reference_candidates_processed': ranking['reference_candidates_processed'], 'candidates': [{'candidate_id': r.candidate_id, 'pt_code': pt, 'score': r.score, 'rank': r.rank, 'features': ranking['features'][r.candidate_id]} for pt, rows in ranking['groups'].items() for r in rows], 'applied': False}
    if not ranking['configuration']['enabled'] or baseline.llt_ids or evidence.force_human_review or evidence.normalized_exact_collision:
        audit['reason'] = 'disabled_or_preserved_automatic_or_forced_deferral'
        return (baseline, audit, None)
    decision, acceptance = assess_ranked(evidence, context, ranking, coverage=coverage, baseline=baseline)
    if decision is None:
        return (baseline, {**audit, 'reason': 'unsupported_structure'}, acceptance)
    if decision.llt_ids:
        chosen = decision.llt_ids[0]
        f = ranking['features'].get(chosen, {})
        literals = [i for i, v in ranking['features'].items() if v['literal_surface'] and (not v['conflicts'])]
        if not f.get('literal_surface') or literals != [chosen] or ranking.get('literal_current_ids') != [chosen]:
            audit['reason'] = 'approximate_or_non_unique_literal_selection_not_deployable'
            return (baseline, audit, acceptance)
    if not decision.pt_codes or (baseline.pt_codes and decision.pt_codes != baseline.pt_codes):
        audit['reason'] = 'no_supported_upgrade'
        return (baseline, audit, acceptance)
    if (decision.pt_codes, decision.llt_ids) == (baseline.pt_codes, baseline.llt_ids):
        audit['reason'] = 'no_automatic_change'
        return (baseline, audit, acceptance)
    audit.update(applied=True, reason='unique_literal_llt_with_shared_guards' if decision.llt_ids else 'supported_surface_PT_anchor', pt_resolved=True, llt_resolved=bool(decision.llt_ids), selected_pt=decision.pt_codes[0], selected_llt_ids=list(decision.llt_ids), acceptance={k: v for k, v in acceptance.items() if not k.startswith('_')})
    return (decision, audit, acceptance)

def reference_surface_index(reranker):
    """Current-reference-only identity index; no retrieval/semantic fabrication."""
    index = {}
    for rows in reranker._entries_by_pt.values():
        for row in rows:
            if row.is_current:
                index.setdefault(surface(row.llt_name), []).append(row.candidate_id)
    return {name: sorted(ids) for name, ids in index.items()}

def assess_ranked(evidence, context, ranking, *, coverage=DEVELOPMENT_COVERAGE, baseline=None):
    """Shared acceptance on reranked evidence, retaining proven baseline codes."""
    if len(evidence.component_rankings) != 1:
        return (None, {'reason': 'compound_or_missing_component_not_replayable'})
    from adaptive_review_policy import ReviewOption
    rows = ranking['pt_rankings']
    options = []
    for option in evidence.options:
        f = ranking['features'].get(option.candidate.candidate_id)
        if f:
            candidate = option.candidate.model_copy(update={'reranker_score': f['score']})
            option = replace(option, candidate=candidate, score=f['score'], conflicts=tuple(sorted(set(option.conflicts) | set(f['conflicts']))))
        options.append(option)
    modified = replace(evidence, options=tuple(options), component_rankings=({**evidence.component_rankings[0], 'pt_rankings': rows},))
    decision, audit = assess(modified, context, ranking['groups'], config=coverage, reference_complete=bool(rows and rows[0]['pt_code'] in ranking['groups']))
    audit['ranking_version'] = VERSION
    if baseline and (baseline.llt_ids or (baseline.pt_codes and decision.pt_codes != baseline.pt_codes)):
        return (baseline, {**audit, 'applied': False, 'baseline_preserved': True})
    return (decision, {**audit, 'applied': True})
