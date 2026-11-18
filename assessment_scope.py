"""Input-bound deterministic assessment scope; no ranking or acceptance changes."""
from runtime_resources import resource as asp_resource
import hashlib, json
from schemas import CandidateAssessment, ReviewerDecision, ReviewerInput
from grammatical_discrimination import support
VERSION = '1.7.1'
KEY = 'assessment_scope_v1_7_1'

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()

def identities(reviewer_input, supplemental):
    result = {}
    for c in list(reviewer_input.aggregation_result.candidates) + list(supplemental or []):
        row = {k: getattr(c, k) for k in ('candidate_id', 'llt_code', 'llt_name', 'pt_code', 'pt_name', 'is_current')}
        if row['candidate_id'] != 'LLT:' + row['llt_code']:
            raise ValueError('Invalid candidate identity')
        if c.candidate_id in result and result[c.candidate_id] != row:
            raise ValueError('Contradictory current parent/identity')
        result[c.candidate_id] = row
    return result

def original_validated(inp, decision):
    from reviewer_validation import validate_reviewer_decision, ReviewerDecisionValidationError
    validate_reviewer_decision(inp, decision, check_clinical=False)
    if not decision.evidence_requests:
        missing = {c.candidate_id for c in inp.aggregation_result.candidates} - {a.candidate_id for a in decision.candidate_assessments}
        if missing:
            raise ReviewerDecisionValidationError('Original reviewer assessment scope incomplete: ' + str(sorted(missing)))

def combined_assessments(inp, original, final, universe, *, support_fn=None, preserved_ids=()):
    compare_support = support_fn or support
    original_by_id = {a.candidate_id: a for a in original.candidate_assessments}
    selected = set(final.selected_candidate_ids)
    assessments = []
    evidence = {}
    from compound_validation import assess_decomposition
    decomposition = assess_decomposition(inp.normalized_term, list(inp.input_interpretation.asserted_components))
    component_text = {c['component_id']: c['effective_text'] for c in decomposition['components']}
    selected_text = {m.candidate_id: component_text.get(m.component_id, inp.original_term) for m in final.component_matches}
    for identifier, row in sorted(universe.items()):
        text = selected_text.get(identifier, inp.original_term) if inp.input_interpretation.structure_type.value == 'multiple_concepts' else inp.original_term
        active_support = support if identifier in preserved_ids else compare_support
        llt = active_support(text, row['llt_name'])
        pt = active_support(text, row['pt_name'])
        blocked = bool(llt['conflicts'] or pt['conflicts'] or llt['unsupported_specificity'] or pt['unsupported_specificity'] or (not row['is_current']))
        existing = original_by_id.get(identifier)
        detail = {'assessment_input': text, 'LLT_support': llt, 'PT_support': pt, 'origin': 'original_reviewer_scope' if identifier in {c.candidate_id for c in inp.aggregation_result.candidates} else 'reference_expansion', 'original_assessment': existing.model_dump(mode='json') if existing else None}
        if identifier in selected:
            if blocked:
                raise ValueError('Deterministic selection has conflicting or unsupported attributes: ' + identifier)
            a = CandidateAssessment(candidate_id=identifier, verdict='supported', assessment_authority='deterministic_gate', rationale='Selected by the input-bound deterministic gate; current identity and input support checked. Not a model assessment.')
        elif identifier in final.human_review_candidate_ids:
            a = CandidateAssessment(candidate_id=identifier, verdict='uncertain', assessment_authority='deterministic_gate', rationale='Exact LLT selection remains unresolved. Clinical support and conflict findings are recorded separately; shortlist membership is not support.')
        elif existing is not None and existing.verdict != 'supported':
            a = existing.model_copy(deep=True)
        else:
            a = CandidateAssessment(candidate_id=identifier, verdict='not_supported' if blocked else 'uncertain', assessment_authority='deterministic_gate', rationale='Input/candidate conflict or unsupported added attribute.' if blocked else 'No verified clinical rejection. Exact LLT selection is unresolved for this alternative; nonselection is not clinical incompatibility.')
        detail['final_assessment'] = a.model_dump(mode='json')
        evidence[identifier] = detail
        assessments.append(a)
    return (assessments, evidence)

def complete_decision(inp, original, final, gate, config):
    original_validated(inp, original)
    supplemental = gate.human_review_candidates + gate.ranked_candidates
    universe = identities(inp, supplemental)
    missing = set(final.selected_candidate_ids + final.human_review_candidate_ids) - set(universe)
    if missing:
        raise ValueError('Invented final identity: ' + str(sorted(missing)))
    support_fn = None
    if config.to_dict().get('functionality_v1_19') and config.to_dict().get('measurement_repair'):
        from measurement_interpretation import support as support_fn
    preserved_ids = ()
    if config.to_dict().get('shared_interpretation_v1_20'):
        from clinical_interpretation import support as support_fn
        comparison = gate.coverage_acceptance.get('development_v1_7', {}).get('comparison_v1_20')
        preserved_ids = ['LLT:' + x for x in comparison['preserved_LLTs']] if comparison else final.selected_candidate_ids
    if 'terminology_v1_22' in gate.coverage_acceptance:
        raise ValueError('This historical opt-in is not supported by ASP-MT 1.0')
    if 'terminology_v1_23' in gate.coverage_acceptance:
        raise ValueError('This historical opt-in is not supported by ASP-MT 1.0')
    if 'wording_v1_24' in gate.coverage_acceptance:
        raise ValueError('This historical opt-in is not supported by ASP-MT 1.0')
    assessments, evidence = combined_assessments(inp, original, final, universe, support_fn=support_fn, preserved_ids=preserved_ids)
    result = ReviewerDecision.model_validate({**final.model_dump(mode='python'), 'candidate_assessments': assessments})
    payload = {'version': VERSION, 'original_input': inp.model_dump(mode='json'), 'original_decision': original.model_dump(mode='json'), 'input_hash': digest(inp.model_dump(mode='json')), 'universe': universe, 'original_scope': sorted((c.candidate_id for c in inp.aggregation_result.candidates)), 'pre_repair_decision': final.model_dump(mode='json'), 'final_decision_hash': digest(result.model_dump(mode='json')), 'candidate_assessment_evidence': evidence, 'configuration': config.to_dict(), 'gate_binding': digest(gate.model_dump(mode='json')), 'authority': 'deterministic_gate_assessment_completion; original reviewer snapshot preserved separately'}
    repaired = gate.model_copy(update={'coverage_acceptance': {**gate.coverage_acceptance, KEY: payload}})
    return (result, repaired)

def verify_contract(inp, decision, supplemental, gate):
    from reviewer_validation import ReviewerDecisionValidationError
    payload = gate.coverage_acceptance.get(KEY)
    if not payload:
        return False
    try:
        if payload['version'] != VERSION or payload['input_hash'] != digest(inp.model_dump(mode='json')):
            raise ValueError('Stale input binding')
        if payload['final_decision_hash'] != digest(decision.model_dump(mode='json')):
            raise ValueError('Stale final decision binding')
        scope = identities(inp, supplemental)
        if scope != payload['universe']:
            raise ValueError('Candidate scope/parent binding changed')
        base = gate.model_copy(update={'coverage_acceptance': {k: v for k, v in gate.coverage_acceptance.items() if k != KEY}})
        if digest(base.model_dump(mode='json')) != payload['gate_binding']:
            raise ValueError('Stale gate acceptance binding')
        actual_configuration = gate.coverage_acceptance.get('development_v1_7', {}).get('configuration')
        if actual_configuration is not None and actual_configuration != payload['configuration']:
            raise ValueError('Stale development configuration')
        original_input = ReviewerInput.model_validate(payload['original_input'])
        original = ReviewerDecision.model_validate(payload['original_decision'])
        if original_input.model_dump(mode='json') != inp.model_dump(mode='json'):
            raise ValueError('Original input mismatch')
        original_validated(original_input, original)
        support_fn = None
        if payload.get('configuration', {}).get('functionality_v1_19') and payload['configuration'].get('measurement_repair'):
            from measurement_interpretation import support as support_fn
        preserved_ids = ()
        if payload.get('configuration', {}).get('shared_interpretation_v1_20'):
            from clinical_interpretation import support as support_fn
            comparison = gate.coverage_acceptance.get('development_v1_7', {}).get('comparison_v1_20')
            preserved_ids = ['LLT:' + x for x in comparison['preserved_LLTs']] if comparison else decision.selected_candidate_ids
        if 'terminology_v1_22' in gate.coverage_acceptance:
            raise ValueError('This historical opt-in is not supported by ASP-MT 1.0')
        if 'terminology_v1_23' in gate.coverage_acceptance:
            raise ValueError('This historical opt-in is not supported by ASP-MT 1.0')
        if 'wording_v1_24' in gate.coverage_acceptance:
            raise ValueError('This historical opt-in is not supported by ASP-MT 1.0')
        expected, evidence = combined_assessments(inp, original, decision, scope, support_fn=support_fn, preserved_ids=preserved_ids)
        if [a.model_dump(mode='json') for a in expected] != [a.model_dump(mode='json') for a in decision.candidate_assessments]:
            raise ValueError('Altered deterministic assessments')
        if evidence != payload['candidate_assessment_evidence']:
            raise ValueError('Altered assessment evidence')
        if payload.get('configuration', {}).get('functionality_v1_19'):
            raise ValueError('This historical opt-in is not supported by ASP-MT 1.0')
        if payload.get('configuration', {}).get('shared_interpretation_v1_20'):
            raise ValueError('This historical opt-in is not supported by ASP-MT 1.0')
        if 'terminology_v1_21' in gate.coverage_acceptance:
            raise ValueError('Unsupported historical acceptance authority')
            verify_knowledge(inp, decision, gate)
        if 'terminology_v1_22' in gate.coverage_acceptance:
            raise ValueError('This historical opt-in is not supported by ASP-MT 1.0')
        if 'terminology_v1_23' in gate.coverage_acceptance:
            raise ValueError('This historical opt-in is not supported by ASP-MT 1.0')
        if 'wording_v1_24' in gate.coverage_acceptance:
            raise ValueError('This historical opt-in is not supported by ASP-MT 1.0')
        if 'equivalence_group_v1_25' in gate.coverage_acceptance:
            from equivalence_authority import verify_contract as verify_group
            verify_group(inp, decision, gate)
        if 'specialist_v1_26' in gate.coverage_acceptance:
            from extension_authority import verify_contract as verify_specialist
            verify_specialist(inp, decision, gate)
        if payload.get('configuration', {}).get('pt_scope_guard_v1_14'):
            raise ValueError('This historical opt-in is not supported by ASP-MT 1.0')
        elif payload.get('configuration', {}).get('pt_scope_guard'):
            raise ValueError('This historical opt-in is not supported by ASP-MT 1.0')
        if 'pairwise_v1_11' in gate.coverage_acceptance:
            if payload.get('configuration', {}).get('grouped_confidence_v1_16'):
                raise ValueError('This historical opt-in is not supported by ASP-MT 1.0')
            elif payload.get('configuration', {}).get('rival_evidence_v1_15'):
                raise ValueError('This historical opt-in is not supported by ASP-MT 1.0')
            elif payload.get('configuration', {}).get('nonlinear_agreement_v1_14'):
                raise ValueError('This historical opt-in is not supported by ASP-MT 1.0')
            else:
                raise ValueError('Unsupported historical acceptance authority')
            verify_learned_acceptance(inp, decision, gate)
    except (KeyError, ValueError, TypeError) as error:
        raise ReviewerDecisionValidationError('Invalid deterministic assessment contract: ' + str(error)) from error
    return True

def expanded_selection_authorized(inp, decision, gate, candidate):
    """Bridge the unchanged development acceptance audit to outcome assembly.

    The LLT acceptance branch in v1.6/v1.7 requires a fully enumerated current
    group before emitting an applied, reason-free LLT acceptance. Verify that
    actual audit, not merely membership in the expanded ranking or shortlist.
    """
    if not verify_contract(inp, decision, gate.human_review_candidates + gate.ranked_candidates, gate):
        return False
    if 'terminology_v1_21' in gate.coverage_acceptance:
        raise ValueError('Unsupported historical acceptance authority')
        if authorize(inp, decision, gate, candidate):
            return True
    if 'terminology_v1_22' in gate.coverage_acceptance:
        raise ValueError('This historical opt-in is not supported by ASP-MT 1.0')
    if 'terminology_v1_23' in gate.coverage_acceptance:
        raise ValueError('This historical opt-in is not supported by ASP-MT 1.0')
    if 'wording_v1_24' in gate.coverage_acceptance:
        raise ValueError('This historical opt-in is not supported by ASP-MT 1.0')
    if 'equivalence_group_v1_25' in gate.coverage_acceptance:
        from equivalence_authority import expanded_selection_authorized as authorize
        if authorize(inp, decision, gate, candidate):
            return True
    if 'specialist_v1_26' in gate.coverage_acceptance:
        from extension_authority import expanded_selection_authorized as authorize
        if authorize(inp, decision, gate, candidate):
            return True
    if gate.coverage_acceptance.get(KEY, {}).get('configuration', {}).get('shared_interpretation_v1_20'):
        raise ValueError('This historical opt-in is not supported by ASP-MT 1.0')
    if gate.coverage_acceptance.get(KEY, {}).get('configuration', {}).get('functionality_v1_19'):
        raise ValueError('This historical opt-in is not supported by ASP-MT 1.0')
    if gate.coverage_acceptance.get('pairwise_v1_11', {}).get('acceptance', {}).get('applied'):
        if gate.coverage_acceptance.get(KEY, {}).get('configuration', {}).get('grouped_confidence_v1_16'):
            raise ValueError('This historical opt-in is not supported by ASP-MT 1.0')
        elif gate.coverage_acceptance.get(KEY, {}).get('configuration', {}).get('rival_evidence_v1_15'):
            raise ValueError('This historical opt-in is not supported by ASP-MT 1.0')
        elif gate.coverage_acceptance.get(KEY, {}).get('configuration', {}).get('nonlinear_agreement_v1_14'):
            raise ValueError('This historical opt-in is not supported by ASP-MT 1.0')
        else:
            raise ValueError('Unsupported historical acceptance authority')
        return bool(verify_learned_acceptance(inp, decision, gate) and candidate.is_current and (candidate.candidate_id in decision.selected_candidate_ids) and (candidate.pt_code == gate.resolved_pt_code))
    audit = gate.coverage_acceptance.get('development_v1_7', {})
    binding = audit.get('input_binding', {})
    features = audit.get('LLT_features', {})
    config = audit.get('baseline_configuration', {}) if 'pairwise_v1_11' in gate.coverage_acceptance else audit.get('configuration', {})
    return bool(audit.get('applied') and audit.get('path') == 'scoped_grammatical_LLT_acceptance' and (audit.get('authority') == 'deterministic_input_discrimination') and (audit.get('input') == inp.original_term) and (binding.get('normalized_input') == inp.normalized_term) and binding.get('input_bound') and (binding.get('structure') == 'single_concept') and (not binding.get('force_review')) and (not binding.get('exact_collision')) and (not gate.reviewer_failure_diagnostics) and (not audit.get('llt_reasons')) and (gate.action == 'auto_accepted') and gate.pt_confidence_passed and (gate.resolved_pt_code == candidate.pt_code) and candidate.is_current and (decision.selected_candidate_ids == [candidate.candidate_id]) and (audit.get('LLT_selected') == candidate.llt_code) and features.get('wording_equivalent') and (not features.get('conflicts')) and (not features.get('unsupported_specificity')) and (not features.get('unresolved_reference_wording_alternatives')) and (audit.get('LLT_score', -1) >= config.get('llt_score', float('inf'))) and (audit.get('LLT_margin', -1) > 0) and (audit.get('LLT_margin', -1) >= config.get('llt_margin', float('inf'))))
