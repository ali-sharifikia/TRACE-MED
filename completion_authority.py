"""Offline-only isolated dispatch; explicit full-phrase or calibrated authority."""
from runtime_resources import resource as asp_resource
from pathlib import Path
import meaning_completion as methods
import module_loading as reuse
policy = reuse.isolated('_coverage_dispatch', Path(reuse.__file__).with_name('source_evidence_authority.py'))
SOURCE = Path(asp_resource('lexicon_words.json'))
LEX = methods.prior.Lexicon(policy.old.load(SOURCE))
policy.EXPERIMENT = 'coverage_v1_32'
policy.MARKER = 'Opt-in v1.32 comparison/calibrated selection; deterministic experimental authority, not original reviewer approval.'

def implementation():
    from runtime_resources import implementation
    return implementation()

def resolve(request, response):
    return {'original': request['input'], 'case': request['case'], 'resolved': True}

def support(text, label, resolution, fn=methods.prior.ranker.support):
    return methods.prior.feature_support(text, label, resolution['case'], 'ALL', LEX, fn)

def apply(*, authority=None, enabled=True, **kw):
    d, g = policy.old.apply_development_gate(**kw)
    if not enabled or authority is None or policy.decision_projection(d, g)['llt']:
        return (d, g)
    inp = kw['reviewer_input']
    original = kw['reviewer_decision']
    scope = policy.scope
    baseline = {k: policy.decision_projection(d, g)[k] for k in ('pt', 'llt')}
    case = policy.old.make_case(inp, baseline, g.coverage_acceptance[scope.KEY], g, kw)
    if authority['implementation'] != implementation() or scope.digest(case) != authority['case_binding']:
        raise ValueError('Stale source/candidate authority')
    if scope.digest(authority['request']['case']) != scope.digest(case) or authority['request']['input'] != inp.original_term:
        raise ValueError('Stale input authority')
    sem = reuse.semantic_observations(inp)
    if sem != authority['semantics']:
        raise ValueError('Changed original semantic evidence')
    mode = authority['mode']
    cutoff = authority['cutoff']
    if mode not in ('P', 'J', 'R'):
        raise ValueError('Unknown mode')
    if mode != 'R' and cutoff != 0.01:
        raise ValueError('Changed fixed margin')
    if mode == 'R':
        path = Path(authority['calibration_file'])
        if policy.sha(path) != authority['calibration_sha256']:
            raise ValueError('Changed calibration binding')
        result = policy.old.load(path)
        fold = str(authority['fold'])
        if cutoff != result['calibration'][fold]['chosen']['cutoff']:
            raise ValueError('Changed calibrated cutoff')
        if authority['id'] not in result['calibration'][fold]['assessment_ids']:
            raise ValueError('Not an assessment-fold input')
        if authority['id'] in result['calibration'][fold]['calibration_ids']:
            raise ValueError('Calibration leakage')
    a = methods.hypothesis(case, policy.old.reference(str(policy.old.REF), policy.sha(policy.old.REF)), sem, LEX, joint=mode != 'P', graded=mode == 'R')
    if not a['accepted'] or cutoff is None or a['numeric']['margin'] < cutoff:
        return (d, g)
    from schemas import ReviewerDecision, ComponentCandidateMatch, CandidateAssessment
    c = a['representative']
    comp = inp.input_interpretation.asserted_components[0]
    d = ReviewerDecision(term_id=inp.term_id, review_round=inp.review_round, decision_type='matched_single', selected_candidate_ids=[c['candidate_id']], component_matches=[ComponentCandidateMatch(component_id=comp.component_id, candidate_id=c['candidate_id'], pt_code=c['pt_code'], rationale=policy.MARKER)], candidate_assessments=[CandidateAssessment(candidate_id=c['candidate_id'], verdict='supported', assessment_authority='deterministic_gate', rationale=policy.MARKER)], decision_rationale=policy.MARKER)
    g = type(g).model_validate({**g.model_dump(mode='python'), 'action': 'auto_accepted', 'gate_action': 'auto_accepted', 'pt_confidence_required': True, 'pt_confidence_passed': True, 'pt_confidence_failure_reasons': [], 'resolved_pt_code': c['pt_code'], 'resolved_pt_name': c['pt_name'], 'human_review_candidates': [], 'human_review_pt_groups': [], 'decision_authority': 'deterministic_input_discrimination', 'reason': policy.MARKER, 'gate_reason': policy.MARKER})
    payload = {'experiment': policy.EXPERIMENT, 'authority': authority, 'audit': a, 'retained_v125': g.coverage_acceptance.get(policy.old.NAMESPACE), 'gate_kwargs': {'reviewer_failure_diagnostics': kw.get('reviewer_failure_diagnostics'), 'exact_match_found': bool(kw.get('exact_match_found')), 'exact_ambiguity_candidates': [r.model_dump(mode='json') for r in kw.get('exact_ambiguity_candidates', ())]}}
    g = g.model_copy(update={'coverage_acceptance': {**{k: v for k, v in g.coverage_acceptance.items() if k not in (scope.KEY, policy.old.NAMESPACE)}, policy.dispatch.NAMESPACE: payload}})
    with policy.integration(authority):
        return scope.complete_decision(inp, original, d, g, policy.old.DEVELOPMENT)
policy.resolve = resolve
policy.new_support = support
policy.implementation = implementation
policy.apply_development_gate = apply
