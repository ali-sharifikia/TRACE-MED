"""Label-independent coverage policy with full-current-reference LLT checks.

Scores are rankings, not probabilities. This module performs no I/O. The caller
supplies reference-complete ranks; absent evidence is a deferral, never a vote.
"""
from runtime_resources import resource as asp_resource
from dataclasses import asdict, dataclass, replace
from decimal import Decimal
from math import isfinite
from adaptive_review_policy import apply_policy, _ambiguity
from biomedical_wording import generic_context_base
from clinical_features import automatic_mapping_uncertainties
from compound_validation import component_semantic_issues
from pipeline_policy import DEFAULT_POLICY
from pt_reranking import explicit_clinical_conflicts
VERSION = '1.4.0'

@dataclass(frozen=True)
class CoverageConfig:
    version: str = VERSION
    enabled: bool = False
    pt_score: float = 0.88
    pt_margin: float = 0.08
    pt_agreement: bool = True
    pt_unagreed_score: float | None = None
    pt_unagreed_margin: float | None = None
    llt_score: float = 0.9
    llt_margin: float = 0.1
    llt_agreement: bool = True
    llt_pt_score: float = 0.9
    llt_pt_margin: float = 0.1
    ambiguity_window: float = 0.22
    nos_specificity_window: float = 0.22

    def __post_init__(self):
        if (self.pt_unagreed_score is None) != (self.pt_unagreed_margin is None):
            raise ValueError('The stricter unagreed PT path requires both thresholds.')
        if self.pt_unagreed_score is not None and (not self.pt_agreement or not self.pt_score <= self.pt_unagreed_score <= 1 or (not self.pt_margin <= self.pt_unagreed_margin <= 1)):
            raise ValueError('Unagreed thresholds must be at least as strict as the agreement path.')
        if any((not isfinite(v) or not 0 <= v <= 1 for v in (self.pt_score, self.pt_margin, self.llt_score, self.llt_margin, self.llt_pt_score, self.llt_pt_margin, self.ambiguity_window, self.nos_specificity_window))):
            raise ValueError('Coverage thresholds must be finite and bounded.')

    def to_dict(self):
        return asdict(self)
DEFAULT_COVERAGE = CoverageConfig(enabled=True, pt_score=0.5, pt_margin=0.001, pt_agreement=True, pt_unagreed_score=0.8, pt_unagreed_margin=0.03, llt_score=0.7, llt_margin=0.01, llt_agreement=False, llt_pt_score=0.88, llt_pt_margin=0.08, ambiguity_window=0.1)
DISABLED_COVERAGE = replace(DEFAULT_COVERAGE, enabled=False)

def pt_requirements(config, agreement):
    if config.pt_agreement and (not agreement) and (config.pt_unagreed_score is not None):
        return (config.pt_unagreed_score, config.pt_unagreed_margin, 'stricter_current_reference_ranking_without_resolved_reviewer_selection')
    return (config.pt_score, config.pt_margin, 'resolved_reviewer_agreement' if config.pt_agreement else 'current_reference_ranking')

def ambiguity_applies(finding, config, shortlist_config=DEFAULT_POLICY):
    """Do not relax genuine NOS specificity uncertainty with cosmetic wording.

The finding already requires same current PT, compatible clinical modifiers,
strong semantic evidence and surface/core equivalence. Plain/NOS differences
retain the established specificity window; only other wording uses the trial
window. A spelling preference is not evidence resolving unspecifiedness.
"""
    difference = set(finding.get('specificity_difference', []))
    nos_specificity = bool(finding.get('nos_or_alias_core_equivalent') and difference and (difference <= {'nos', 'nonspecific', 'non', 'specific', 'unspecified'}))
    window = max(config.ambiguity_window, config.nos_specificity_window) if nos_specificity else config.ambiguity_window
    return finding['score_gap'] <= window and finding['alternative_score'] >= shortlist_config.llt_minimum_score - window

def resolved_pt_review(evidence, pt_code, acceptance, *, shortlist_config=DEFAULT_POLICY):
    """Represent independently verified PT certainty with zero, one or many LLTs.

This consumes an input-bound deterministic acceptance, never a reviewer claim.
No compatible LLT is invented, and an empty shortlist remains LLT-unresolved.
"""
    if not acceptance.get('pt_accepted') or acceptance.get('input_text') != evidence.input_text or acceptance.get('selected_pt') != pt_code or (not acceptance.get('reference_complete')) or acceptance.get('pt_reasons') or evidence.force_human_review or evidence.normalized_exact_collision:
        raise ValueError('An independently verified, input-bound PT acceptance is required.')
    return apply_policy(replace(evidence, baseline_pt_codes=(pt_code,), baseline_llt_ids=(), independently_resolved_pt_codes=(pt_code,)), shortlist_config)

def rethreshold(permissive, audit, deferred, config):
    """Apply declared thresholds to cached, identical clinical assessments.

The cached assessment must use zero floors and no required agreement. Clinical
rejections (including ties) cannot be cleared by this calibration operation.
"""
    source = audit['configuration']
    if any((source[k] for k in ('pt_score', 'pt_margin', 'pt_agreement', 'llt_score', 'llt_margin', 'llt_agreement', 'llt_pt_score', 'llt_pt_margin'))) or source['ambiguity_window'] != 0:
        raise ValueError('Rethresholding requires the unfiltered clinical assessment.')
    revised = {**audit, 'configuration': config.to_dict(), 'pt_reasons': list(audit['pt_reasons']), 'llt_reasons': list(audit['llt_reasons'])}
    if not config.enabled:
        revised['pt_reasons'].append('policy_disabled')
    if audit['pt_accepted']:
        required_score, required_margin, path = pt_requirements(config, audit['reviewer_pt_agreement'])
        revised['pt_thresholds'] = {'score': required_score, 'margin': required_margin}
        revised['pt_acceptance_path'] = path
        if Decimal(str(audit['top_pt_score'])) < Decimal(str(required_score)):
            revised['pt_reasons'].append('pt_score_below_threshold')
        if Decimal(audit['pt_margin']) < Decimal(str(required_margin)):
            revised['pt_reasons'].append('pt_margin_below_threshold_or_tied')
        if config.pt_agreement and (not audit['reviewer_pt_agreement']) and (config.pt_unagreed_score is None):
            revised['pt_reasons'].append('resolved_reviewer_agreement_missing')
    revised['pt_accepted'] = audit['pt_accepted'] and (not revised['pt_reasons'])
    if not revised['pt_accepted']:
        revised['llt_accepted'] = False
        return (deferred, revised)
    if 'llt_score' in audit:
        if Decimal(str(audit['llt_score'])) < Decimal(str(config.llt_score)):
            revised['llt_reasons'].append('llt_score_below_threshold')
        if Decimal(audit['llt_margin']) < Decimal(str(config.llt_margin)):
            revised['llt_reasons'].append('llt_margin_below_threshold_or_tied')
        if config.llt_agreement and (not audit['reviewer_llt_agreement']):
            revised['llt_reasons'].append('resolved_reviewer_llt_agreement_missing')
        if Decimal(str(audit['top_pt_score'])) < Decimal(str(config.llt_pt_score)) or Decimal(audit['pt_margin']) < Decimal(str(config.llt_pt_margin)):
            revised['llt_reasons'].append('llt_compatibility_stage_pt_confidence')
        revised['ambiguity'] = [v for v in audit.get('potential_ambiguities', []) if ambiguity_applies(v, config)]
        if revised['ambiguity']:
            revised['llt_reasons'].append('genuine_same_pt_wording_ambiguity')
    revised['llt_accepted'] = audit['llt_accepted'] and (not revised['llt_reasons'])
    if revised['llt_accepted']:
        return (permissive, revised)
    return (replace(permissive, llt_ids=(), candidates=audit['_pt_review_candidates'], groups=audit['_pt_review_groups']), revised)

def assess(evidence, context, full_llt_rankings, *, config=DEFAULT_COVERAGE, shortlist_config=DEFAULT_POLICY, reference_complete=False):
    """Reconstruct acceptance independently of historical automatic decisions.

Only single components use this additional path. Compounds continue through the
existing shared all-components validation (including full-reference checks).
"""
    deferred = apply_policy(replace(evidence, baseline_pt_codes=(), baseline_llt_ids=(), force_human_review=True), shortlist_config)
    audit = dict(version=VERSION, configuration=config.to_dict(), authority='deterministic_coverage_policy', pt_accepted=False, llt_accepted=False, pt_reasons=[], llt_reasons=[], acceptance_path='current_reference_ranking_with_clinical_validation', reference_complete=reference_complete, input_text=evidence.input_text)
    reasons = audit['pt_reasons']
    if not config.enabled:
        reasons.append('policy_disabled')
    if evidence.force_human_review or evidence.normalized_exact_collision:
        reasons.append('forced_deferral_or_exact_collision')
    if evidence.structure_type != 'single_concept' or len(evidence.component_rankings) != 1:
        reasons.append('interpretation_not_one_unambiguous_asserted_component')
    rankings = evidence.component_rankings[0]['pt_rankings'] if len(evidence.component_rankings) == 1 else []
    if len(evidence.component_rankings) == 1 and evidence.component_rankings[0]['component_text'] not in {evidence.input_text, generic_context_base(evidence.input_text)}:
        reasons.append('component_does_not_represent_entire_input')
    if context.get('source_input') != evidence.input_text or context.get('interpretation_status') != evidence.structure_type:
        reasons.append('missing_or_stale_interpretation_evidence')
    reasons.extend(automatic_mapping_uncertainties(evidence.input_text))
    if len(rankings) < 2 or len({p['pt_code'] for p in rankings}) != len(rankings):
        reasons.append('distinct_competitor_evidence_missing_or_duplicated')
    if not rankings:
        return (deferred, audit)
    top = rankings[0]
    options = {o.candidate.candidate_id: o for o in evidence.options}
    best = options.get(top['best_candidate_id'])
    raw = context.get('candidate_evidence', {})
    top_raw = raw.get(top['best_candidate_id'], {})
    entries = full_llt_rankings.get(top['pt_code'], [])
    if len({e.candidate_id for e in entries}) != len(entries) or any((e.pt_code != top['pt_code'] or e.pt_name != top['pt_name'] or e.candidate_id != 'LLT:' + e.llt_code for e in entries)):
        reasons.append('current_reference_pool_identity_inconsistent')
    current = next((e for e in entries if e.candidate_id == top['best_candidate_id']), None)
    if not reference_complete or not current or (not best) or any((current.pt_code != best.candidate.pt_code, current.llt_code != best.candidate.llt_code, current.llt_name != best.candidate.llt_name, current.pt_name != best.candidate.pt_name)):
        reasons.append('current_reference_identity_or_parent_unverified')
    if not best or best.conflicts or (not top_raw.get('is_current')) or (top_raw.get('pt_code') != top['pt_code']):
        reasons.append('leader_clinical_or_identity_conflict')
    if best:
        reasons.extend(component_semantic_issues(evidence.input_text, best.candidate.llt_name + ' ' + best.candidate.pt_name))
    retrieval_conflicts = top_raw.get('retrieval_conflicts', [])
    modifier_conflicts = top.get('score_breakdown', {}).get('modifier_conflict_reasons', [])
    if retrieval_conflicts or modifier_conflicts:
        reasons.append('existing_retrieval_or_modifier_conflict_guard')
    agreement = bool(context.get('selection_resolved') and context.get('current_reviewer_candidates_verified') and (context.get('reviewer_parent_codes') == [top['pt_code']]))
    if agreement:
        reference_by_id = {e.candidate_id: e for e in entries}
        for identifier in context.get('reviewer_candidate_ids', []):
            option = options.get(identifier)
            reference = reference_by_id.get(identifier)
            if not option or option.conflicts or raw.get(identifier, {}).get('retrieval_conflicts') or (not reference) or (reference.llt_name != option.candidate.llt_name) or (reference.pt_code != option.candidate.pt_code):
                reasons.append('reviewer_agreement_clinical_or_identity_unverified')
                break
    required_score, required_margin, path = pt_requirements(config, agreement)
    audit.update(pt_thresholds={'score': required_score, 'margin': required_margin}, pt_acceptance_path=path)
    if config.pt_agreement and (not agreement) and (config.pt_unagreed_score is None):
        reasons.append('resolved_reviewer_agreement_missing')
    if context.get('selection_resolved') and context.get('reviewer_parent_codes') != [top['pt_code']]:
        reasons.append('resolved_reviewer_disagrees_with_pt')
    scores = [p.get('unrounded_reranking_score') for p in rankings]
    valid = all((isinstance(s, (int, float)) and isfinite(s) and (0 <= s <= 1) for s in scores))
    margin = None
    if not valid:
        reasons.append('raw_score_evidence_missing')
    else:
        margin = Decimal(str(scores[0])) - max((Decimal(str(s)) for s in scores[1:]), default=Decimal(str(scores[0])))
        if Decimal(str(scores[0])) < Decimal(str(required_score)):
            reasons.append('pt_score_below_threshold')
        if margin <= 0 or margin < Decimal(str(required_margin)):
            reasons.append('pt_margin_below_threshold_or_tied')
    audit.update(selected_pt=top['pt_code'], selected_pt_name=top['pt_name'], top_pt_score=scores[0], pt_margin=str(margin), pt_competitors=[{k: p.get(k) for k in ('pt_code', 'pt_name', 'unrounded_reranking_score')} for p in rankings[1:]], reviewer_pt_agreement=agreement, reviewer_authority=context.get('reviewer_authority'), retrieval_methods=sorted(set(top_raw.get('method_families', []))), retrieval_conflicts=retrieval_conflicts, modifier_conflicts=modifier_conflicts, interpretation_status=evidence.structure_type)
    if reasons:
        return (deferred, audit)
    audit['pt_accepted'] = True
    result = resolved_pt_review(evidence, top['pt_code'], audit, shortlist_config=shortlist_config)
    audit['llt_shortlist_size'] = len(result.candidates)
    audit['_pt_review_candidates'] = result.candidates
    audit['_pt_review_groups'] = result.groups
    llt_reasons = audit['llt_reasons']
    if not entries:
        llt_reasons.append('no_established_current_llt')
        return (result, audit)
    selected = entries[0]
    selected_option = options.get(selected.candidate_id)
    llt_scores = [e.unrounded_score for e in entries]
    if not all((isinstance(s, (int, float)) and isfinite(s) and (0 <= s <= 1) for s in llt_scores)):
        llt_reasons.append('raw_llt_score_evidence_missing')
        return (result, audit)
    llt_margin = Decimal(str(llt_scores[0])) - max((Decimal(str(s)) for s in llt_scores[1:]), default=Decimal('0'))
    llt_agreement = context.get('selection_resolved') and context.get('reviewer_candidate_ids') == [selected.candidate_id]
    if config.llt_agreement and (not llt_agreement):
        llt_reasons.append('resolved_reviewer_llt_agreement_missing')
    if Decimal(str(scores[0])) < Decimal(str(config.llt_pt_score)) or margin < Decimal(str(config.llt_pt_margin)):
        llt_reasons.append('llt_compatibility_stage_pt_confidence')
    if selected_option is None or selected_option.conflicts:
        llt_reasons.append('llt_clinical_evidence_missing_or_conflicting')
    if selected.modifier_conflicts or selected.unsupported_specificity_penalty or selected.added_analytes:
        llt_reasons.append('llt_modifier_or_specificity_conflict')
    llt_reasons.extend(explicit_clinical_conflicts(evidence.input_text, selected.llt_name))
    llt_reasons.extend(component_semantic_issues(evidence.input_text, selected.llt_name + ' ' + selected.pt_name))
    if raw.get(selected.candidate_id, {}).get('retrieval_conflicts'):
        llt_reasons.append('llt_retrieval_conflict')
    if Decimal(str(llt_scores[0])) < Decimal(str(config.llt_score)):
        llt_reasons.append('llt_score_below_threshold')
    if llt_margin <= 0 or llt_margin < Decimal(str(config.llt_margin)):
        llt_reasons.append('llt_margin_below_threshold_or_tied')
    ambiguity, potential_ambiguities = ([], [])
    if selected_option:
        for other in evidence.options:
            if other.candidate.candidate_id != selected.candidate_id:
                finding = _ambiguity(evidence.input_text, selected_option, other, replace(shortlist_config, ambiguity_score_delta=1.0))
                if finding:
                    finding['score_gap'] = selected_option.score - other.score
                    finding['alternative_score'] = other.score
                    potential_ambiguities.append(finding)
                    if ambiguity_applies(finding, config, shortlist_config):
                        ambiguity.append(finding)
    if ambiguity:
        llt_reasons.append('genuine_same_pt_wording_ambiguity')
    audit.update(selected_llt=selected.llt_code, selected_llt_parent=selected.pt_code, llt_score=llt_scores[0], llt_margin=str(llt_margin), llt_competitors=[{'llt_code': e.llt_code, 'score': e.unrounded_score} for e in entries[1:]], reviewer_llt_agreement=bool(llt_agreement), ambiguity=ambiguity, potential_ambiguities=potential_ambiguities)
    if not llt_reasons:
        result.llt_ids = (selected.candidate_id,)
        result.candidates, result.groups = ([], [])
        audit['llt_accepted'] = True
    result.reasons.append('coverage_policy_pt_and_llt_resolved' if audit['llt_accepted'] else 'coverage_policy_pt_resolved_llt_deferred')
    return (result, audit)
