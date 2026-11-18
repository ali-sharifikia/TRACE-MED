"""Isolated opt-in v1.25, with retained v1.9 or v1.24 output authority."""
from runtime_resources import resource as asp_resource
from functools import lru_cache
from pathlib import Path
from assessment_scope import KEY, complete_decision, digest
from event_comparison import DEVELOPMENT, rank
from acceptance_gate import decision_projection
from io_utils import sha, load
from lexical_equivalence_evaluation import Reference
from equivalence_group import propose
import reference_context as old
REF = old.REF
NAMESPACE = 'equivalence_group_v1_25'
STORE = None

def implementation():
    from runtime_resources import implementation
    return implementation()

@lru_cache(maxsize=2)
def reference(path, h):
    if sha(path) != h:
        raise ValueError('Stale reference')
    return Reference(path, Path(path).with_name('llt.asc'), Path(path).with_name('pt.asc'))

def make_case(inp, baseline, scope, gate, kw):
    """Same original ranking inputs and code as runtime_case v1.21; retain features."""
    from adaptive_review_policy import prepare_policy_evidence
    from clinical_features import automatic_mapping_uncertainties
    from input_discrimination import representations
    from assessment_scope import original_validated
    from pt_coverage_policy import recovery_context
    from schemas import ReviewerDecision
    candidates = list(scope.get('universe', {}).values())
    groups = {}
    for c in candidates:
        groups.setdefault(c['pt_code'], []).append(c)
    e = prepare_policy_evidence(inp, tuple(scope.get('original_decision', {}).get('selected_candidate_ids', [])))
    pts = e.component_rankings[0]['pt_rankings'] if len(e.component_rankings) == 1 else []
    sem = {c.candidate_id: c.semantic_evidence.cosine_similarity for c in inp.aggregation_result.candidates if c.semantic_evidence}
    ranked = rank(inp.original_term, pts, groups, sem) if pts and groups else {'features': {}}
    features = {}
    ids = tuple(scope.get('original_decision', {}).get('selected_candidate_ids', []))
    context = recovery_context(inp, ids, selection_resolved=bool(ids), authority='runtime_reviewer_stage_with_repairs_preserved')
    retrieval = {k: v['retrieval_conflicts'] for k, v in context['candidate_evidence'].items()}
    original = scope.get('original_decision')
    valid = False
    if original:
        original_validated(inp, ReviewerDecision.model_validate(original))
        valid = True
    audit = gate.coverage_acceptance.get('development_v1_7', {})
    leader = audit.get('LLT_selected')
    for cid, f in ranked['features'].items():
        features[cid] = {'wording_equivalent': f['wording_equivalent'], 'conflicts': f['conflicts'], 'unsupported_specificity': f['unsupported_specificity'], 'unrepresented_input_information': f['support']['unrepresented_input_information'], 'alternatives': [a['candidate_id'] for a in f.get('unresolved_reference_wording_alternatives', [])], 'retrieval_conflict': bool(retrieval.get(cid))}
    scores = {k: f['score'] for k, f in ranked['features'].items()}
    saved = {c['id']: c['score'] for c in audit.get('LLT_competitors', [])}
    if leader is not None and audit.get('LLT_score') is not None:
        saved['LLT:' + str(leader)] = audit['LLT_score']
    scores_bound = all((k in scores and abs(scores[k] - v) < 1e-10 for k, v in saved.items()))
    components = inp.input_interpretation.asserted_components
    pt = baseline['pt'][0] if len(baseline['pt']) == 1 else None
    ref = reference(str(REF), sha(REF))
    group = groups.get(pt, [])
    return {'input': inp.original_term, 'baseline': baseline, 'candidates': candidates, 'scores': scores, 'features': features, 'original_scope_available': valid, 'pt_authority': bool(pt and gate.resolved_pt_code == pt and gate.pt_confidence_passed), 'input_bound': len(components) == 1 and components[0].component_text == inp.normalized_term and (not components[0].is_negated), 'single_concept': inp.input_interpretation.structure_type.value == 'single_concept', 'complete_group': bool(pt and {c['llt_code'] for c in group} == ref.groups.get(pt, set())), 'scores_bound': scores_bound, 'saved_score_observations': len(saved), 'fallback': bool(kw.get('reviewer_failure_diagnostics')) or e.force_human_review, 'exact_collision': bool(kw.get('exact_ambiguity_candidates')) or e.normalized_exact_collision, 'exact_route': bool(kw.get('exact_match_found')), 'numeric_uncertainty': bool(automatic_mapping_uncertainties(representations(inp.original_term)['meaning_source'])), 'old_reasons': audit.get('llt_reasons', []), 'reference_sha256': sha(REF), 'input_hash': digest(inp.model_dump(mode='json')), 'original_decision_hash': digest(original) if original else None}

def apply_development_gate(*, baseline_mode='v1.9', enabled=True, **kwargs):
    if baseline_mode not in ('v1.9', 'v1.24'):
        raise ValueError('Unsupported baseline')
    from acceptance_gate import apply_development_gate as baseline
    d, g = baseline(**kwargs)
    if baseline_mode == 'v1.24':
        raise ValueError('This historical opt-in is not supported by ASP-MT 1.0')
    if not enabled or decision_projection(d, g)['llt'] or len(decision_projection(d, g)['pt']) != 1:
        return (d, g)
    if KEY not in g.coverage_acceptance:
        return (d, g)
    inp = kwargs['reviewer_input']
    original = kwargs['reviewer_decision']
    case = make_case(inp, decision_projection(d, g), g.coverage_acceptance[KEY], g, kwargs)
    audit = propose(case, reference(str(REF), sha(REF)))
    if audit['accepted']:
        from schemas import ReviewerDecision, ComponentCandidateMatch, CandidateAssessment
        c = audit['representative']
        comp = inp.input_interpretation.asserted_components[0]
        d = ReviewerDecision(term_id=inp.term_id, review_round=inp.review_round, decision_type='matched_single', selected_candidate_ids=[c['candidate_id']], component_matches=[ComponentCandidateMatch(component_id=comp.component_id, candidate_id=c['candidate_id'], pt_code=c['pt_code'], rationale='Verified pairwise equivalent meaning group; deterministic, not original reviewer approval.')], candidate_assessments=[CandidateAssessment(candidate_id=c['candidate_id'], verdict='supported', assessment_authority='deterministic_gate', rationale='Input-bound group proof and current identity; original assessment preserved separately.')], decision_rationale='Opt-in v1.25 equivalent wording group completion.')
        g = type(g).model_validate({**g.model_dump(mode='python'), 'action': 'auto_accepted', 'gate_action': 'auto_accepted', 'human_review_candidates': [], 'human_review_pt_groups': [], 'decision_authority': 'deterministic_input_discrimination', 'reason': 'v1.25 verified equivalent meaning group', 'gate_reason': 'v1.25 verified equivalent meaning group'})
    payload = {'version': '1.25.0', 'baseline_mode': baseline_mode, 'audit': audit, 'case_binding': digest(case), 'baseline': case['baseline'], 'implementation': implementation(), 'reference_sha256': sha(REF), 'raw_reference_hashes': {n: sha(REF.with_name(n)) for n in ('llt.asc', 'pt.asc')}, 'v124_store_sha256': sha(STORE) if baseline_mode == 'v1.24' else None, 'gate_kwargs': {'reviewer_failure_diagnostics': kwargs.get('reviewer_failure_diagnostics'), 'exact_match_found': bool(kwargs.get('exact_match_found')), 'exact_ambiguity_candidates': [c.model_dump(mode='json') for c in kwargs.get('exact_ambiguity_candidates', ())]}}
    base = g.model_copy(update={'coverage_acceptance': {**{k: v for k, v in g.coverage_acceptance.items() if k != KEY}, NAMESPACE: payload}})
    return complete_decision(inp, original, d, base, DEVELOPMENT)

def verify_contract(inp, d, g):
    from schemas import ReviewerDecision, AggregatedCandidate
    p = g.coverage_acceptance[NAMESPACE]
    if p['implementation'] != implementation() or p['reference_sha256'] != sha(REF):
        raise ValueError('Stale group implementation/reference')
    if p['raw_reference_hashes'] != {n: sha(REF.with_name(n)) for n in ('llt.asc', 'pt.asc')}:
        raise ValueError('Stale raw reference')
    if p['baseline_mode'] == 'v1.24' and p['v124_store_sha256'] != sha(STORE):
        raise ValueError('Stale v1.24 store')
    kw = dict(p['gate_kwargs'])
    kw['exact_ambiguity_candidates'] = tuple((AggregatedCandidate.model_validate(x) for x in kw['exact_ambiguity_candidates']))
    nd, ng = apply_development_gate(baseline_mode=p['baseline_mode'], reviewer_input=inp, reviewer_decision=ReviewerDecision.model_validate(g.coverage_acceptance[KEY]['original_decision']), reranker=old.verifier_reranker(str(REF), sha(REF)), **kw)
    if nd != d or ng != g:
        raise ValueError('Group authority does not recompute')
    return True

def expanded_selection_authorized(inp, d, g, c):
    verify_contract(inp, d, g)
    a = g.coverage_acceptance[NAMESPACE]['audit']
    return bool(a['accepted'] and c.is_current and (a['representative']['candidate_id'] == c.candidate_id) and (d.selected_candidate_ids == [c.candidate_id]) and (c.pt_code == g.resolved_pt_code) and g.pt_confidence_passed and (g.action == 'auto_accepted'))
