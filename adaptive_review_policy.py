"""Shared deterministic PT promotion, LLT ambiguity and shortlist policy.

Only evidence objects are accepted. The offline evaluator owns label access.
"""
from __future__ import annotations
from runtime_resources import resource as asp_resource
from dataclasses import dataclass
from difflib import SequenceMatcher
from collections import defaultdict
from biomedical_wording import generic_context_base, is_generic_event
from clinical_features import automatic_mapping_uncertainties, interpreted_features
from llt_reranking import LLTReferenceEntry, normalize_llt_text, score_llt_candidate
from pipeline_policy import DEFAULT_POLICY, PolicyConfig
from pt_reranking import build_component_pt_rankings, explicit_clinical_conflicts
from schemas import HumanReviewCandidate, HumanReviewPTGroup, ReviewerInput, ReviewerDecision

@dataclass(frozen=True)
class ReviewOption:
    candidate: HumanReviewCandidate
    score: float
    semantic: float
    umls: bool
    sources: int
    conflicts: tuple[str, ...]
    component_ids: tuple[str, ...]
    alias_exact: bool

@dataclass(frozen=True)
class PolicyEvidence:
    input_text: str
    structure_type: str
    component_rankings: tuple[dict, ...]
    options: tuple[ReviewOption, ...]
    reviewer_ids: tuple[str, ...]
    baseline_pt_codes: tuple[str, ...]
    baseline_llt_ids: tuple[str, ...]
    single_selected_pt: str | None
    preprocessing_reasons: tuple[str, ...] = ()
    normalized_exact_collision: bool = False
    exact_collision_ids: tuple[str, ...] = ()
    force_human_review: bool = False
    independently_resolved_pt_codes: tuple[str, ...] = ()

@dataclass
class PolicyDecision:
    pt_codes: tuple[str, ...]
    llt_ids: tuple[str, ...]
    candidates: list[HumanReviewCandidate]
    groups: list[HumanReviewPTGroup]
    pruned: list[str]
    reasons: list[str]
    ambiguity: list[dict]
    top_score: float | None
    top_margin: float | None

def prepare_policy_evidence(reviewer_input: ReviewerInput, reviewer_ids: tuple[str, ...], baseline_pt_codes: tuple[str, ...]=(), baseline_llt_ids: tuple[str, ...]=(), single_selected_pt: str | None=None, supplemental: tuple[HumanReviewCandidate, ...]=()) -> PolicyEvidence:
    """Compute label-independent features once, shared by runtime and replay."""
    components = [c for c in reviewer_input.input_interpretation.asserted_components if not c.is_negated]
    from compound_validation import assess_decomposition
    scope = assess_decomposition(reviewer_input.normalized_term, components)
    effective = {c['component_id']: c['effective_text'] for c in scope['components']}
    rankings = build_component_pt_rankings(reviewer_input)
    retrieved = {c.candidate_id: c for c in reviewer_input.aggregation_result.candidates}
    candidates = {c.candidate_id: c for c in supplemental}
    for c in retrieved.values():
        candidates[c.candidate_id] = HumanReviewCandidate(candidate_id=c.candidate_id, llt_code=c.llt_code, llt_name=c.llt_name, pt_code=c.pt_code, pt_name=c.pt_name, is_current=c.is_current, was_retrieved=True, retrieval_source_count=c.source_count, reviewer_selected=c.candidate_id in reviewer_ids, reranker_shortlisted=False)
    options = []
    for candidate in candidates.values():
        raw = retrieved.get(candidate.candidate_id)
        semantic = raw.semantic_evidence.cosine_similarity if raw and raw.semantic_evidence else 0.0
        sources = raw.source_count if raw else candidate.retrieval_source_count
        conflicts_by_component = {c.component_id: tuple(sorted(set(explicit_clinical_conflicts(effective[c.component_id], candidate.llt_name)) | set(explicit_clinical_conflicts(effective[c.component_id], candidate.pt_name)))) for c in components}
        eligible_components = tuple((c.component_id for c in components if not conflicts_by_component[c.component_id]))
        conflicts = () if eligible_components else tuple(sorted({v for values in conflicts_by_component.values() for v in values}))
        if not candidate.is_current:
            conflicts += ('non_current_llt',)
        if generic_context_base(reviewer_input.normalized_term) and (is_generic_event(candidate.llt_name) or is_generic_event(candidate.pt_name)):
            conflicts += ('generic_context_is_not_an_independent_event',)
        entry = LLTReferenceEntry(candidate.candidate_id, candidate.llt_code, candidate.llt_name, candidate.pt_code, candidate.pt_name, candidate.is_current)
        feature_sets = [score_llt_candidate(effective[c.component_id], entry, was_retrieved=raw is not None, retrieval_source_count=sources, semantic_similarity=semantic) for c in components if c.component_id in eligible_components]
        features = max(feature_sets, key=lambda item: item['score']) if feature_sets else {'score': 0.0, 'alias_equivalent': False}
        candidate = candidate.model_copy(update={'reranker_score': features['score'], 'reranker_rank': 1, 'reviewer_selected': candidate.candidate_id in reviewer_ids})
        literal_equal = reviewer_input.normalized_term.casefold().strip() == candidate.llt_name.casefold().strip()
        safe_exact = literal_equal or (not interpreted_features(reviewer_input.normalized_term)['bounds'] and (features['alias_equivalent'] or normalize_llt_text(reviewer_input.normalized_term) == normalize_llt_text(candidate.llt_name)))
        options.append(ReviewOption(candidate, features['score'], semantic, bool(raw and raw.umls_evidence), sources, conflicts, eligible_components, bool(safe_exact)))
    return PolicyEvidence(reviewer_input.normalized_term, reviewer_input.input_interpretation.structure_type.value, tuple(rankings), tuple(sorted(options, key=lambda o: o.candidate.candidate_id)), reviewer_ids, baseline_pt_codes, baseline_llt_ids, single_selected_pt, tuple(reviewer_input.input_interpretation.deterministic_repair_reasons))

def _ambiguity(input_text: str, selected: ReviewOption, other: ReviewOption, config: PolicyConfig) -> dict | None:
    if selected.candidate.pt_code != other.candidate.pt_code or other.conflicts:
        return None
    first = normalize_llt_text(selected.candidate.llt_name)
    second = normalize_llt_text(other.candidate.llt_name)
    surface = SequenceMatcher(None, ' '.join(sorted(first.split())), ' '.join(sorted(second.split()))).ratio()
    nonspecific = {'nos', 'nonspecific', 'non', 'specific', 'unspecified'}
    core_same = set(first.split()) - nonspecific == set(second.split()) - nonspecific
    literal_selected = input_text.casefold().strip() == selected.candidate.llt_name.casefold().strip()
    if literal_selected:
        return None
    modifier_equivalent = not explicit_clinical_conflicts(selected.candidate.llt_name, other.candidate.llt_name)
    plausible = other.score >= config.llt_minimum_score - config.ambiguity_score_delta
    close = selected.score - other.score <= config.ambiguity_score_delta
    semantic_support = other.semantic >= config.ambiguity_semantic and other.sources >= 1 and (other.semantic >= selected.semantic - 0.04)
    supported_alternative = core_same and semantic_support or (surface >= config.ambiguity_surface and semantic_support)
    if plausible and close and modifier_equivalent and supported_alternative:
        return {'candidate_id': other.candidate.candidate_id, 'same_pt': True, 'surface_similarity': round(surface, 6), 'semantic_similarity': other.semantic, 'nos_or_alias_core_equivalent': core_same, 'score_gap': round(selected.score - other.score, 6), 'modifier_equivalent': True, 'specificity_difference': sorted(set(first.split()) ^ set(second.split())), 'reviewer_selected': selected.candidate.reviewer_selected, 'reviewer_reranker_agreement': selected.score >= other.score, 'reason': 'same_pt_wording_or_specificity_not_reliably_distinguished'}
    return None

def apply_policy(evidence: PolicyEvidence, config: PolicyConfig=DEFAULT_POLICY) -> PolicyDecision:
    """Replayable policy without model, terminology I/O, or evaluation labels."""
    by_id = {o.candidate.candidate_id: o for o in evidence.options}
    by_pt: dict[str, list[ReviewOption]] = defaultdict(list)
    for option in evidence.options:
        by_pt[option.candidate.pt_code].append(option)
    for values in by_pt.values():
        values.sort(key=lambda o: (-o.score, -o.sources, -o.semantic, o.candidate.candidate_id))
    pt_order = []
    pt_scores = {}
    component_tops = set()
    for component in evidence.component_rankings:
        for item in component['pt_rankings']:
            pt_scores[item['pt_code']] = max(pt_scores.get(item['pt_code'], 0.0), item['reranking_score'])
        if component['pt_rankings']:
            component_tops.add(component['pt_rankings'][0]['pt_code'])
    for rank in range(max((len(c['pt_rankings']) for c in evidence.component_rankings), default=0)):
        for component in evidence.component_rankings:
            if rank < len(component['pt_rankings']):
                pt = component['pt_rankings'][rank]['pt_code']
                if pt not in pt_order:
                    pt_order.append(pt)
    for pt in sorted(by_pt):
        if pt not in pt_order:
            pt_order.append(pt)
    pt_rank = {pt: rank for rank, pt in enumerate(pt_order, 1)}
    top = evidence.component_rankings[0]['pt_rankings'] if evidence.component_rankings else []
    top_score = top[0]['reranking_score'] if top else None
    margin = round(top_score - top[1]['reranking_score'], 6) if len(top) > 1 else 1.0 if top else None
    pts = tuple(evidence.baseline_pt_codes)
    llt_ids = tuple(evidence.baseline_llt_ids)
    reasons = list(evidence.preprocessing_reasons)
    ambiguity = []
    uncertainties = sorted({reason for text in [evidence.input_text] + [component.get('component_text', '') for component in evidence.component_rankings] for reason in automatic_mapping_uncertainties(text)})
    block_automatic = evidence.force_human_review or bool(uncertainties)
    if block_automatic:
        pts, llt_ids = ((), ())
        reasons.extend(uncertainties)
        if evidence.force_human_review:
            reasons.append('reviewer_clinical_validation_exhausted_requires_human_review')
    if evidence.normalized_exact_collision:
        pts, llt_ids = ((), ())
        reasons.append('ambiguous_normalized_or_alias_exact_collision_requires_human_review')
    if any((identifier not in by_id or by_id[identifier].conflicts for identifier in llt_ids)):
        llt_ids = ()
        pts = ()
        reasons.append('automatic_mapping_deferred_due_to_explicit_clinical_conflict')
    if pts and (not llt_ids) and any((pt not in evidence.independently_resolved_pt_codes and (not any((not option.conflicts for option in by_pt[pt]))) for pt in pts)):
        pts = ()
        reasons.append('resolved_pt_has_no_clinically_consistent_current_llt')
    if not pts and (not block_automatic) and (not evidence.normalized_exact_collision) and (evidence.structure_type == 'single_concept') and (len(evidence.component_rankings) == 1) and top:
        best = by_id.get(top[0]['best_candidate_id'])
        internally_consistent = evidence.single_selected_pt in {None, top[0]['pt_code']}
        if top_score >= config.pt_minimum_score and margin >= config.pt_minimum_margin and (best is not None) and (not best.conflicts) and internally_consistent:
            pts = (top[0]['pt_code'],)
            reasons.append('deterministic_unique_single_component_pt_promotion')
    if llt_ids and config.ambiguity_enabled:
        for identifier in llt_ids:
            selected = by_id[identifier]
            for other in by_pt[selected.candidate.pt_code]:
                if other.candidate.candidate_id == identifier:
                    continue
                diagnostic = _ambiguity(evidence.input_text, selected, other, config)
                if diagnostic:
                    ambiguity.append(diagnostic)
        if ambiguity:
            llt_ids = ()
            reasons.append('same_pt_llt_ambiguity_preserves_automatic_pt')
    if llt_ids:
        return PolicyDecision(pts, llt_ids, [], [], [], reasons, ambiguity, top_score, margin)
    resolved = pts[0] if len(pts) == 1 else None
    allowed_pts = set([resolved] if resolved else pt_order[:config.base_pt_groups])
    if not resolved:
        for pt in pt_order[config.base_pt_groups:config.maximum_pt_rank]:
            independent = any((o.sources >= 2 or o.umls or o.semantic >= config.strong_semantic for o in by_pt[pt] if not o.conflicts))
            close = (top_score or 0.0) - pt_scores.get(pt, 0.0) <= config.score_delta
            if independent or close or pt in component_tops:
                allowed_pts.add(pt)
        allowed_pts.update((by_id[i].candidate.pt_code for i in evidence.reviewer_ids if i in by_id))
        allowed_pts.update((by_id[i].candidate.pt_code for i in evidence.exact_collision_ids if i in by_id))
        allowed_pts.update(component_tops)
    retained: dict[str, list[str]] = {}
    pruning: dict[str, str] = {}
    eligible = {}
    for option in evidence.options:
        identifier, pt = (option.candidate.candidate_id, option.candidate.pt_code)
        if option.conflicts:
            pruning[identifier] = 'explicit_conflict:' + ','.join(option.conflicts)
        elif resolved and pt != resolved:
            pruning[identifier] = 'outside_resolved_pt'
        elif pt not in allowed_pts:
            pruning[identifier] = 'outside_adaptive_pt_evidence_window'
        else:
            eligible[identifier] = option

    def keep(identifier, reason):
        if identifier in eligible:
            retained.setdefault(identifier, []).append(reason)
    for identifier in evidence.reviewer_ids:
        keep(identifier, 'reviewer_selected_mandatory')
    for identifier in evidence.exact_collision_ids:
        keep(identifier, 'ambiguous_normalized_or_alias_exact_mandatory')
    for pt in pt_order:
        safe = [o for o in by_pt[pt] if o.candidate.candidate_id in eligible]
        if not safe:
            continue
        keep(safe[0].candidate.candidate_id, 'resolved_or_leading_pt' if pt == resolved or pt in component_tops else 'base_or_adaptive_pt_representation')
    leading_pts = {resolved} if resolved else component_tops
    for pt in sorted(leading_pts, key=lambda code: pt_rank[code]):
        for option in [o for o in by_pt[pt] if o.candidate.candidate_id in eligible][:config.llts_per_leading_pt]:
            keep(option.candidate.candidate_id, 'leading_or_resolved_pt_llt_allocation')
    for identifier, option in eligible.items():
        if option.alias_exact:
            keep(identifier, 'normalized_or_safe_alias_exact')
    mandatory_count = len(retained)
    remaining = sorted((o for identifier, o in eligible.items() if identifier not in retained), key=lambda o: (-(o.sources >= 2), -o.umls, -(o.semantic >= config.strong_semantic), pt_rank[o.candidate.pt_code], -o.score, -o.sources, o.candidate.candidate_id))
    for option in remaining:
        identifier = option.candidate.candidate_id
        pt = option.candidate.pt_code
        pt_count = sum((eligible[i].candidate.pt_code == pt for i in retained))
        if pt_count >= config.llts_per_leading_pt:
            pruning[identifier] = 'per_pt_allocation_after_mandatory_retention'
        elif len(retained) < config.maximum_candidates:
            keep(identifier, 'multiple_sources' if option.sources >= 2 else 'strong_semantic_or_umls' if option.umls or option.semantic >= config.strong_semantic else 'within_group_wording_alternative')
        else:
            pruning[identifier] = 'total_cap_after_mandatory_retention'
    candidates = []
    for identifier, why in retained.items():
        option = eligible[identifier]
        candidates.append(option.candidate.model_copy(update={'retention_reasons': sorted(set(why)), 'reranker_shortlisted': True, 'reranker_rank': 1 + next((index for index, other in enumerate(by_pt[option.candidate.pt_code]) if other.candidate.candidate_id == identifier)), 'introduction_methods': ['retrieved' if option.candidate.was_retrieved else 'pt_expansion']}))
    groups = []
    for pt in pt_order:
        nested = [c for c in candidates if c.pt_code == pt]
        if nested:
            groups.append(HumanReviewPTGroup(pt_code=pt, pt_name=nested[0].pt_name, pt_rank=pt_rank[pt], pt_score=pt_scores.get(pt), candidate_ids=[c.candidate_id for c in nested], llt_candidates=nested, pruning_explanations=[f'{identifier}: {reason}' for identifier, reason in sorted(pruning.items()) if by_id[identifier].candidate.pt_code == pt]))
    pruned = [f'{identifier}: {reason}' for identifier, reason in sorted(pruning.items())]
    if mandatory_count > config.maximum_candidates:
        pruned.append(f'soft_cap_overflow: {mandatory_count} mandatory candidates retained; configured cap={config.maximum_candidates}')
    if not candidates:
        reasons.append('no_viable_candidate_after_conflict_and_pt_checks')
        pts = tuple((pt for pt in pts if pt in evidence.independently_resolved_pt_codes and (not block_automatic) and (not evidence.normalized_exact_collision)))
    return PolicyDecision(pts, (), candidates, groups, pruned, reasons, ambiguity, top_score, margin)
