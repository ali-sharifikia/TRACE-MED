"""Additive single-component PT recovery; no labels, I/O or LLT acceptance.

The compatibility-stage PT/LLT gates and compound thresholds are unchanged.
This ranking margin is NOT a probability or independent statistical evidence.
"""
from runtime_resources import resource as asp_resource
from dataclasses import asdict, dataclass, replace
from decimal import Decimal
from math import isfinite
from adaptive_review_policy import apply_policy
from clinical_features import automatic_mapping_uncertainties
from compound_validation import component_semantic_issues
from pipeline_policy import DEFAULT_POLICY

@dataclass(frozen=True)
class PTRecoveryConfig:
    version: str = '1.3.3'
    enabled: bool = True
    minimum_score: float = 0.88
    minimum_margin: float = 0.08
    requires_resolved_reviewer_agreement: bool = True
    scope: str = 'single_concept_PT_only'

    def __post_init__(self):
        if not (0 <= self.minimum_score <= 1 and 0 < self.minimum_margin <= 1):
            raise ValueError('PT recovery thresholds must be bounded, with a positive margin.')
        if not self.requires_resolved_reviewer_agreement or self.scope != 'single_concept_PT_only':
            raise ValueError('PT recovery cannot disable agreement or expand to compounds.')

    def to_dict(self):
        return asdict(self)
DEFAULT_PT_RECOVERY = PTRecoveryConfig()
DISABLED_PT_RECOVERY = replace(DEFAULT_PT_RECOVERY, enabled=False)

def recovery_context(reviewer_input, reviewer_ids, *, selection_resolved, authority):
    """Allowlisted runtime evidence. Correlated candidates/queries are not votes."""
    candidates = {c.candidate_id: c for c in reviewer_input.aggregation_result.candidates}
    ids = tuple(reviewer_ids)
    parents = {candidates[i].pt_code for i in ids if i in candidates}
    current = bool(ids) and all((i in candidates and candidates[i].is_current for i in ids))
    from llt_selection_gate import _explicit_candidate_conflicts
    return {'reviewer_candidate_ids': list(ids), 'selection_resolved': bool(selection_resolved), 'reviewer_parent_codes': sorted(parents), 'current_reviewer_candidates_verified': current, 'reviewer_authority': authority, 'candidate_evidence': {i: {'pt_code': c.pt_code, 'llt_code': c.llt_code, 'is_current': c.is_current, 'retrieval_conflicts': _explicit_candidate_conflicts(c), 'method_families': sorted({s.value for s in c.evidence_sources if getattr(c, s.value + '_evidence', None) is not None})} for i, c in candidates.items()}, 'source_input': reviewer_input.normalized_term, 'interpretation_status': reviewer_input.input_interpretation.structure_type.value, 'provenance': 'validated_reviewer_decision_and_aggregated_current_candidates'}

def recover_pt(evidence, baseline, context, *, policy=DEFAULT_PT_RECOVERY, shortlist_config=DEFAULT_POLICY):
    """Return (decision, audit). Existing automatic decisions are left intact.

No LLT can be newly accepted here. A deferred compatible within-PT shortlist
is built by the existing policy only after all PT recovery checks pass.
"""
    audit = {'version': policy.version, 'configuration': policy.to_dict(), 'accepted': False, 'authority': 'deterministic_pt_coverage_recovery', 'acceptance_path': 'resolved_reviewer_agreement_with_bounded_pt_margin', 'reasons': [], 'baseline_pt_codes': list(baseline.pt_codes), 'llt_acceptance_unchanged': True, 'compound_acceptance_unchanged': True}
    if evidence.force_human_review or evidence.normalized_exact_collision:
        audit['reasons'] = ['forced_human_review_or_exact_collision_cannot_be_promoted']
        return (apply_policy(replace(evidence, baseline_pt_codes=(), baseline_llt_ids=()), shortlist_config), audit)
    if baseline.pt_codes or baseline.llt_ids:
        audit['reasons'] = ['existing_automatic_decision_unchanged']
        return (baseline, audit)
    reasons = []
    if not policy.enabled:
        reasons.append('recovery_disabled')
    if evidence.force_human_review:
        reasons.append('forced_human_review_cannot_be_promoted')
    if evidence.normalized_exact_collision:
        reasons.append('normalized_exact_collision_requires_review')
    if evidence.structure_type != 'single_concept' or len(evidence.component_rankings) != 1:
        reasons.append('not_one_unambiguous_asserted_component')
    if len(evidence.component_rankings) == 1:
        from biomedical_wording import generic_context_base
        component = evidence.component_rankings[0].get('component_text', '')
        if component not in {evidence.input_text, generic_context_base(evidence.input_text)}:
            reasons.append('single_component_does_not_represent_complete_input')
    rankings = evidence.component_rankings[0]['pt_rankings'] if len(evidence.component_rankings) == 1 else []
    if len(rankings) < 2:
        reasons.append('competitor_evidence_missing')
    codes = [p['pt_code'] for p in rankings]
    if len(codes) != len(set(codes)):
        reasons.append('duplicate_pt_group_evidence')
    top = rankings[0] if rankings else {}
    best_id = top.get('best_candidate_id')
    options = {o.candidate.candidate_id: o for o in evidence.options}
    best = options.get(best_id)
    checks = dict(context or {})
    if checks.get('source_input') != evidence.input_text or checks.get('interpretation_status') != evidence.structure_type:
        reasons.append('missing_or_stale_interpretation_evidence')
    agrees = bool(checks.get('selection_resolved') and checks.get('current_reviewer_candidates_verified') and (checks.get('reviewer_parent_codes') == [top.get('pt_code')]))
    if not agrees:
        reasons.append('resolved_reviewer_pt_agreement_missing_or_conflicting')
    if evidence.single_selected_pt not in {None, top.get('pt_code')}:
        reasons.append('reviewer_selection_contradicts_leading_pt')
    ids = list(checks.get('reviewer_candidate_ids') or [])
    if not ids or any((i not in options or options[i].conflicts or options[i].candidate.pt_code != top.get('pt_code') for i in ids)):
        reasons.append('reviewer_candidate_identity_or_clinical_consistency_failed')
    if not best or best.conflicts or (not best.candidate.is_current) or (best.candidate.pt_code != top.get('pt_code')):
        reasons.append('leader_identity_or_clinical_consistency_failed')
    raw = checks.get('candidate_evidence') or {}
    retrieval_conflicts = sorted({v for i in set(ids + [best_id]) for v in raw.get(i, {}).get('retrieval_conflicts', [])})
    if any((i not in raw for i in ids + [best_id])):
        reasons.append('retrieval_identity_evidence_missing')
    if any((raw.get(i, {}).get('pt_code') != top.get('pt_code') or not raw.get(i, {}).get('is_current') for i in ids + [best_id])):
        reasons.append('serialized_current_parent_inconsistent')
    if any((i in options and raw.get(i, {}).get('llt_code') != options[i].candidate.llt_code for i in ids + [best_id])):
        reasons.append('serialized_candidate_identity_inconsistent')
    ranking_conflicts = top.get('score_breakdown', {}).get('modifier_conflict_reasons', [])
    if retrieval_conflicts or ranking_conflicts:
        reasons.append('existing_retrieval_or_modifier_conflict_guard_failed')
    uncertainties = sorted({r for text in [evidence.input_text] + [c.get('component_text', '') for c in evidence.component_rankings] for r in automatic_mapping_uncertainties(text)})
    reasons.extend(uncertainties)
    if best:
        reasons.extend(component_semantic_issues(evidence.input_text, best.candidate.llt_name + ' ' + best.candidate.pt_name))
    scores = [p.get('unrounded_reranking_score') for p in rankings]
    valid_scores = bool(scores) and all((isinstance(s, (int, float)) and isfinite(s) and (0 <= s <= 1) for s in scores))
    margin = None
    if not valid_scores:
        reasons.append('unrounded_score_evidence_missing_or_invalid')
    else:
        score = Decimal(str(scores[0]))
        competitor = max((Decimal(str(s)) for s in scores[1:]), default=score)
        margin = score - competitor
        if score < Decimal(str(policy.minimum_score)):
            reasons.append('pt_score_below_recovery_floor')
        if margin < Decimal(str(policy.minimum_margin)):
            reasons.append('distinct_pt_margin_below_recovery_floor')
    competitor_row = max(rankings[1:], key=lambda p: p.get('unrounded_reranking_score', -1)) if len(rankings) > 1 and valid_scores else None
    methods = sorted(set(raw.get(best_id, {}).get('method_families', [])))
    audit.update(selected_pt=top.get('pt_code'), best_candidate_id=best_id, top_score=scores[0] if scores else None, margin=str(margin) if margin is not None else None, strongest_competitor={k: competitor_row.get(k) for k in ('pt_code', 'pt_name', 'unrounded_reranking_score', 'best_candidate_id')} if competitor_row else None, relevant_competitor_count=max(0, len(rankings) - 1), competitors_removed=0, reviewer_reranker_agreement=agrees, reviewer_authority=checks.get('reviewer_authority', 'unavailable'), reviewer_candidate_ids=ids, supporting_method_families=methods, evidence_semantics='Distinct retrieval method families, not independent observations; aliases and query variants are not additional votes.', clinical_checks={'candidate_conflicts': list(best.conflicts) if best else ['missing_candidate'], 'retrieval_conflicts': retrieval_conflicts, 'ranking_conflicts': ranking_conflicts, 'uncertainties': uncertainties}, interpretation_status=evidence.structure_type, llts='No new strict LLT decision: full-reference LLT/compatibility/ambiguity gate did not supply an automatic LLT. Retain compatible within-PT alternatives.')
    if reasons:
        audit['reasons'] = sorted(set(reasons))
        return (baseline, audit)
    resolved = apply_policy(replace(evidence, baseline_pt_codes=(top['pt_code'],), baseline_llt_ids=()), shortlist_config)
    if resolved.pt_codes != (top['pt_code'],) or any((c.pt_code != top['pt_code'] for c in resolved.candidates)):
        audit['reasons'] = ['compatible_resolved_pt_shortlist_parent_inconsistent']
        return (baseline, audit)
    resolved.reasons.append('pt_coverage_recovery_resolved_reviewer_agreement_with_bounded_margin')
    audit['accepted'] = True
    audit['reasons'] = ['all_single_component_pt_recovery_checks_passed_llt_deferred']
    return (resolved, audit)
