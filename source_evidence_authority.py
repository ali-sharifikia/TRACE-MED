"""Opt-in source-span interpretation. No labels, corpus exception lists or live I/O."""
from runtime_resources import resource as asp_resource
import copy, re, json
from contextlib import contextmanager
from pathlib import Path
import sys
import equivalence_authority as old
import extension_authority as dispatch
import assessment_scope as scope
from equivalence_group import propose as group_propose
from event_comparison import support, comparison
from grammatical_discrimination import support as assessment_support
from pt_reranking import explicit_clinical_conflicts
from acceptance_gate import decision_projection
from io_utils import sha
EXPERIMENT = 'source_evidence_authority'
MARKER = 'Opt-in v1.28 source-span correspondence; not original reviewer approval.'
SCHEMA = {'type': 'object', 'additionalProperties': False, 'required': ['judgment', 'expansion', 'reason'], 'properties': {'judgment': {'type': 'string', 'enum': ['supported', 'uncertain']}, 'expansion': {'anyOf': [{'type': 'string'}, {'type': 'null'}]}, 'reason': {'type': 'string'}}}

def norm(s):
    return ' '.join(s.casefold().split())

def candidates_for(text, rows, names):
    """Candidate names select relevance only. Preserve every source alternative."""
    found = []
    for m in re.finditer('\\b[A-Za-z]{2,10}\\b', text):
        rs = [r for r in rows if norm(r['fields'][1]) == norm(m[0]) and re.fullmatch('[A-Z]{2,10}', r['fields'][1])]
        if not rs:
            continue
        alternatives = sorted({r['fields'][4] for r in rs}, key=lambda x: (norm(x), x))
        relevant = [x for x in alternatives if len(x.split()) >= 2 and any((re.search('(?<!\\w)' + re.escape(norm(x)) + '(?!\\w)', norm(n)) for n in names))]
        if relevant:
            found.append({'start': m.start(), 'end': m.end(), 'original_span': m[0], 'alternatives': alternatives, 'source_rows': rs, 'relevance_only': relevant})
    return found

def validate(value):
    if not isinstance(value, dict) or set(value) != set(SCHEMA['required']):
        raise ValueError('Unexpected response fields')
    if value['judgment'] not in ('supported', 'uncertain') or not isinstance(value['reason'], str) or (not value['reason'].strip()):
        raise ValueError('Missing judgment/reason')
    if value['judgment'] == 'supported' and (not isinstance(value['expansion'], str) or not value['expansion'].strip()):
        raise ValueError('Missing supported expansion')
    if value['judgment'] == 'uncertain' and value['expansion'] is not None:
        raise ValueError('Uncertain must not select an expansion')
    return value

def resolve(request, response):
    validate(response)
    text = request['input']
    s = request['span']
    if text[s['start']:s['end']] != s['original_span']:
        raise ValueError('Stale original span')
    out = {'original': text, 'interpreted': None, 'resolved': False, 'response': response, 'span': s, 'authority': 'model-assessed context; source-explicit available expansion, not verified clinical equivalence'}
    if response['judgment'] == 'uncertain':
        out['reason'] = 'explicit_abstention'
        return out
    witnesses = [r for r in s['source_rows'] if norm(r['fields'][4]) == norm(response['expansion'])]
    if not witnesses:
        out['reason'] = 'expansion_not_source_witnessed'
        return out
    expansion = sorted({r['fields'][4] for r in witnesses})[0]
    out.update(resolved=True, reason='source_witnessed_model_context', witnesses=witnesses, interpreted=text[:s['start']] + expansion + text[s['end']:])
    return out

def new_support(original, label, resolution, fn=support):
    before = fn(original, label)
    if not resolution['resolved'] or original != resolution['original']:
        return before
    after = copy.deepcopy(fn(resolution['interpreted'], label))
    hard = set(explicit_clinical_conflicts(original, label)) | set(explicit_clinical_conflicts(resolution['interpreted'], label))
    after['conflicts'] = sorted(set(after['conflicts']) | hard)
    return after

def propose(case, ref, resolution):
    if resolution['original'] != case['input']:
        raise ValueError('Interpretation/input mismatch')
    if not resolution['resolved']:
        return {'accepted': False, 'blockers': [resolution['reason']], 'resolution': resolution}
    c = copy.deepcopy(case)
    evidence = {}
    for candidate in c['candidates']:
        cid = candidate['candidate_id']
        if cid not in c['features']:
            continue
        name = candidate['llt_name']
        s = new_support(case['input'], name, resolution)
        pt = new_support(case['input'], candidate['pt_name'], resolution)
        ev = comparison(resolution['interpreted'], name)
        from equivalence_group import wording_proof
        proof = wording_proof(resolution['interpreted'], name)
        positive = bool((ev['equivalent'] or proof) and (not s['conflicts']) and (not s['unsupported_specificity']) and (not pt['conflicts']) and (not pt['unsupported_specificity']))
        f = c['features'][cid]
        evidence[cid] = {'before': case['features'][cid], 'interpreted_support': s, 'parent_support': pt, 'comparison': ev, 'lexical_proof': proof, 'positive': positive}
        f['wording_equivalent'] = positive
        f['conflicts'] = sorted(set(s['conflicts']) | set(pt['conflicts']))
        f['unsupported_specificity'] = sorted(set(s['unsupported_specificity']) | set(pt['unsupported_specificity']))
    result = group_propose(c, ref)
    result['resolution'] = resolution
    result['source_comparisons'] = evidence
    if not case.get('saved_score_observations'):
        result['blockers'].append('no_saved_score_observations')
    if len(result.get('group', [])) == 1 and all((evidence[x]['positive'] for x in result['group'])):
        result['blockers'] = [b for b in result['blockers'] if b != 'no_multiple_verified_equivalent_identities']
    result['blockers'] = sorted(set(result['blockers']))
    result['accepted'] = not result['blockers']
    if result['accepted']:
        cid = result['representative_convention']['candidate']
        result['representative'] = next((x for x in c['candidates'] if x['candidate_id'] == cid))
    else:
        result['representative'] = None
    return result

def implementation():
    from runtime_resources import implementation
    return implementation()

def apply_development_gate(*, authority=None, enabled=True, **kw):
    d, g = old.apply_development_gate(**kw)
    if not enabled or authority is None or decision_projection(d, g)['llt'] or (len(decision_projection(d, g)['pt']) != 1):
        return (d, g)
    inp = kw['reviewer_input']
    original = kw['reviewer_decision']
    case = old.make_case(inp, decision_projection(d, g), g.coverage_acceptance[scope.KEY], g, kw)
    if authority['implementation'] != implementation() or authority['case_binding'] != scope.digest(case):
        raise ValueError('Stale lexicon authority')
    if not authority.get('execution') or authority['request']['input'] != inp.original_term:
        raise ValueError('Missing model/input provenance')
    for path, h in authority['source_bindings'].items():
        if sha(path) != h:
            raise ValueError('Stale lexicon source')
    if len(authority['source_bindings']) != 1:
        raise ValueError('Expected one frozen source store')
    store = json.loads(Path(next(iter(authority['source_bindings']))).read_text(encoding='utf-8'))
    span = authority['request']['span']
    rows = [r for r in store['records']['LEX/LRABR'] if norm(r['fields'][1]) == norm(span['original_span']) and re.fullmatch('[A-Z]{2,10}', r['fields'][1])]
    if rows != span['source_rows'] or sorted({r['fields'][4] for r in rows}, key=lambda x: (norm(x), x)) != span['alternatives']:
        raise ValueError('Altered/omitted source alternatives')
    resolution = resolve(authority['request'], authority['response'])
    audit = propose(case, old.reference(str(old.REF), sha(old.REF)), resolution)
    if not audit['accepted']:
        return (d, g)
    from schemas import ReviewerDecision, ComponentCandidateMatch, CandidateAssessment
    c = audit['representative']
    comp = inp.input_interpretation.asserted_components[0]
    d = ReviewerDecision(term_id=inp.term_id, review_round=inp.review_round, decision_type='matched_single', selected_candidate_ids=[c['candidate_id']], component_matches=[ComponentCandidateMatch(component_id=comp.component_id, candidate_id=c['candidate_id'], pt_code=c['pt_code'], rationale=MARKER)], candidate_assessments=[CandidateAssessment(candidate_id=c['candidate_id'], verdict='supported', assessment_authority='deterministic_gate', rationale=MARKER)], decision_rationale=MARKER)
    g = type(g).model_validate({**g.model_dump(mode='python'), 'action': 'auto_accepted', 'gate_action': 'auto_accepted', 'human_review_candidates': [], 'human_review_pt_groups': [], 'decision_authority': 'deterministic_input_discrimination', 'reason': MARKER, 'gate_reason': MARKER})
    payload = {'experiment': EXPERIMENT, 'authority': authority, 'audit': audit, 'retained_v125': g.coverage_acceptance.get(old.NAMESPACE), 'gate_kwargs': {'reviewer_failure_diagnostics': kw.get('reviewer_failure_diagnostics'), 'exact_match_found': bool(kw.get('exact_match_found')), 'exact_ambiguity_candidates': [c.model_dump(mode='json') for c in kw.get('exact_ambiguity_candidates', ())]}}
    g = g.model_copy(update={'coverage_acceptance': {**{k: v for k, v in g.coverage_acceptance.items() if k not in (scope.KEY, old.NAMESPACE)}, dispatch.NAMESPACE: payload}})
    with integration(authority):
        return scope.complete_decision(inp, original, d, g, old.DEVELOPMENT)

def verify(inp, d, g):
    from schemas import ReviewerDecision, AggregatedCandidate
    p = g.coverage_acceptance[dispatch.NAMESPACE]
    kw = dict(p['gate_kwargs'])
    kw['exact_ambiguity_candidates'] = tuple((AggregatedCandidate.model_validate(c) for c in kw['exact_ambiguity_candidates']))
    nd, ng = apply_development_gate(authority=p['authority'], reviewer_input=inp, reviewer_decision=ReviewerDecision.model_validate(g.coverage_acceptance[scope.KEY]['original_decision']), reranker=old.old.verifier_reranker(str(old.REF), sha(old.REF)), **kw)
    if d != nd or g != ng:
        raise ValueError('Lexicon proof does not recompute')
    return True

@contextmanager
def integration(authority):
    """Single-process, isolated opt-in; old functions restored even on failure."""
    ov, oe, oc = (dispatch.verify_contract, dispatch.expanded_selection_authorized, scope.combined_assessments)
    resolution = resolve(authority['request'], authority['response'])

    def current(g):
        return g.coverage_acceptance.get(dispatch.NAMESPACE, {}).get('experiment') == EXPERIMENT

    def v(i, d, g):
        return verify(i, d, g) if current(g) else ov(i, d, g)

    def e(i, d, g, c):
        if not current(g):
            return oe(i, d, g, c)
        verify(i, d, g)
        a = g.coverage_acceptance[dispatch.NAMESPACE]['audit']
        return bool(a['accepted'] and c.is_current and (a['representative']['candidate_id'] == c.candidate_id) and (c.pt_code == g.resolved_pt_code) and g.pt_confidence_passed)

    def combined(i, o, d, u, **kw):
        if d.decision_rationale == MARKER:
            if i.original_term != resolution['original']:
                raise ValueError('Stale interpretation context')
            kw['support_fn'] = lambda text, label: new_support(text, label, resolution, assessment_support)
        return oc(i, o, d, u, **kw)
    dispatch.verify_contract, dispatch.expanded_selection_authorized, scope.combined_assessments = (v, e, combined)
    try:
        yield
    finally:
        dispatch.verify_contract, dispatch.expanded_selection_authorized, scope.combined_assessments = (ov, oe, oc)
