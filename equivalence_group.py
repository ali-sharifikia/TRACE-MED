"""Opt-in meaning-group proof. No corpus labels, IDs, error categories or I/O."""
from runtime_resources import resource as asp_resource
import re
from functools import lru_cache
from itertools import combinations
import lexical_equivalence_evaluation as lexical
from clinical_features import NEGATABLE_MODIFIERS
from grammatical_discrimination import ATTRIBUTE_ROLES, equivalence
from pt_reranking import ANATOMY_GROUPS
MODIFIERS = frozenset(NEGATABLE_MODIFIERS) | frozenset(ATTRIBUTE_ROLES) | {'normal', 'abnormal'} | {w for group in ANATOMY_GROUPS for w in group if w.endswith(('al', 'ic', 'ary'))}
VERSION = '1.25.0'

@lru_cache(maxsize=50000)
def adjective_frame(value):
    words = lexical.text(value).split()
    if len(words) != 2 or lexical.SCOPE.search(value) or (not re.fullmatch('[A-Za-z ]+', value)):
        return None
    modifiers = [w for w in words if w in MODIFIERS]
    if len(modifiers) != 1:
        return None
    m = modifiers[0]
    head = words[1] if words[0] == m else words[0]
    if len(head) < 3 or head in MODIFIERS | lexical.SPECIMENS | lexical.PROPERTIES | set(lexical.DIRECTIONS) | set().union(*lexical.ATTRIBUTES.values()):
        return None
    if any((t.isupper() and len(t) > 1 for t in value.split())):
        return None
    return {'modifier': m, 'head': head, 'ordered_tokens': words, 'resource': 'existing NEGATABLE_MODIFIERS / ATTRIBUTE_ROLES / ANATOMY_GROUPS adjective forms; normal/abnormal descriptors'}

def wording_proof(a, b):
    proof = lexical.linguistic(a, b)
    if proof:
        return proof
    x, y = (adjective_frame(a), adjective_frame(b))
    if x and y and ((x['modifier'], x['head']) == (y['modifier'], y['head'])):
        return {'route': 'B', 'rule': 'bounded_single_modifier_head_inversion', 'witness': {'left': x, 'right': y, 'unchanged_tokens': True}}
    return None

def compare_pair(original, a, b, ref):
    result = lexical.compare_pair(original, a, b, ref)
    if result['status'] in ('exact', 'verified_equivalent'):
        return result
    x, y = (ref.identity(a), ref.identity(b))
    if not x.get('current') or not y.get('current') or x['pt_code'] != y['pt_code']:
        return result
    proof = wording_proof(x['llt_name'], y['llt_name'])
    if proof and (not lexical.input_guard(original, x['llt_name'], y['llt_name'])):
        return {**result, 'status': 'verified_equivalent', 'reasons': [], 'retained_differences': [], **proof}
    return result

def score_sets(predicted, expected, original, ref):
    p = sorted(set(predicted))
    g = sorted(set(expected))
    pairs = {(a, b): compare_pair(original, a, b, ref) for a in p for b in g}

    def matching(states):
        found = {}

        def visit(a, seen):
            for b in g:
                if b in seen or pairs[a, b]['status'] not in states:
                    continue
                seen.add(b)
                if b not in found or visit(found[b], seen):
                    found[b] = a
                    return True
            return False
        for a in p:
            visit(a, set())
        return [{'predicted': a, 'expected': b, 'route': pairs[a, b]['route'], 'rule': pairs[a, b]['rule']} for b, a in sorted(found.items())]
    links = matching({'exact', 'verified_equivalent'})
    ok = bool(p) and len(p) == len(g) == len(links)
    possible = matching({'exact', 'verified_equivalent', 'unresolved'})
    parents = lambda cs: {ref.rows[c]['pt_code'] for c in cs} if cs and all((c in ref.rows for c in cs)) else None
    pp, gp = (parents(p), parents(g))
    return {'strict': bool(p) and p == g, 'equivalence': ok, 'correspondences': links, 'pairs': list(pairs.values()), 'parent_agreement': pp == gp if pp is not None and gp is not None else None, 'disposition': 'agreement' if ok else 'cardinality_difference' if len(p) != len(g) else 'unresolved' if len(possible) == len(p) else 'retained_attribute_difference'}

def clique_group(anchor, ids, pair):
    """No closure. Every included edge and every crossing edge is checked."""
    members = sorted((x for x in ids if x == anchor or pair(anchor, x)['status'] == 'verified_equivalent'))
    blockers = []
    if any((pair(a, b)['status'] != 'verified_equivalent' for a, b in combinations(members, 2))):
        blockers.append('non_clique_equivalence')
    outside = sorted(set(ids) - set(members))
    if any((pair(a, b)['status'] == 'verified_equivalent' for a in members for b in outside)):
        blockers.append('overlapping_equivalence_membership')
    return (members, outside, blockers)

def faithful_key(original, name):
    literal = lexical.text(original) == lexical.text(name)
    grammar = equivalence(original, name)['equivalent']
    a = re.findall('[a-z]+', lexical.text(original))
    b = re.findall('[a-z]+', lexical.text(name))
    shared_prefix = 0
    for x, y in zip(a, b):
        if x != y:
            break
        shared_prefix += 1
    return (-int(literal), -int(grammar), -shared_prefix, lexical.text(name))

from decision_recorder import computation

@computation('comparison.equivalence_group')
def propose(case, ref):
    result = {'version': VERSION, 'accepted': False, 'representative': None, 'blockers': [], 'group': [], 'pair_proofs': {}, 'outside': [], 'numeric': None, 'input_support': {}, 'old_reasons': case.get('old_reasons', [])}
    b = result['blockers']
    base = case['baseline']
    pt = base['pt'][0] if len(base['pt']) == 1 else None
    if base['llt'] or pt is None:
        b.append('retained_automatic_or_PT_unresolved')
        return result
    for name in ('original_scope_available', 'pt_authority', 'input_bound', 'single_concept', 'complete_group', 'scores_bound'):
        if not case.get(name):
            b.append(name + '_unestablished')
    for name in ('fallback', 'exact_collision', 'numeric_uncertainty', 'exact_route'):
        if case.get(name):
            b.append(name)
    cs = {c['candidate_id']: c for c in case['candidates'] if c['pt_code'] == pt}
    ids = sorted(cs)
    scores = case['scores']
    fs = case['features']
    if not ids or any((x not in scores or x not in fs for x in ids)):
        b.append('missing_scores_or_features')
        return result
    if len(cs) != len([c for c in case['candidates'] if c['pt_code'] == pt]):
        b.append('duplicate_candidate_identity')
    for x, c in cs.items():
        row = ref.identity(c['llt_code'])
        if not row.get('current') or x != 'LLT:' + c['llt_code'] or any((row[k] != c[k] for k in ('llt_name', 'pt_code', 'pt_name'))):
            b.append('invalid_current_identity')
    if {c['llt_code'] for c in cs.values()} != ref.groups.get(pt, set()):
        b.append('incomplete_actual_current_group')
    maximum = max((scores[x] for x in ids))
    tops = sorted((x for x in ids if abs(scores[x] - maximum) <= 1e-10))
    anchor = min(tops, key=lambda x: faithful_key(case['input'], cs[x]['llt_name']))

    def pair(a, c):
        x, y = sorted((a, c))
        key = x + '|' + y
        if key not in result['pair_proofs']:
            result['pair_proofs'][key] = compare_pair(case['input'], cs[x]['llt_code'], cs[y]['llt_code'], ref)
        return result['pair_proofs'][key]
    group, outside, problems = clique_group(anchor, ids, pair)
    b.extend(problems)
    result.update(anchor=anchor, group=group, outside=outside, top_set=tops)
    if len(group) < 2:
        b.append('no_multiple_verified_equivalent_identities')
    if not set(tops) <= set(group):
        b.append('competing_maximum_meanings')
    old_rival = max((scores[x] for x in ids if x != anchor), default=0.0)
    outscore = max((scores[x] for x in outside), default=0.0)
    result['numeric'] = {'score': maximum, 'ordinary_margin': maximum - old_rival, 'ordinary_closest': [x for x in ids if x != anchor and abs(scores[x] - old_rival) <= 1e-10], 'group_margin': maximum - outscore, 'strongest_outside': [x for x in outside if abs(scores[x] - outscore) <= 1e-10], 'outside_maximum': outscore if outside else None, 'sole_group': not outside, 'sole_group_convention': 'existing v1.6 zero comparison baseline, not a candidate' if not outside else None, 'passed': maximum >= 0.7 and maximum - outscore > 0 and (maximum - outscore >= 0.01)}
    if not result['numeric']['passed']:
        b.append('group_score_or_margin')
    for x in ids:
        f = fs[x]
        proof = wording_proof(case['input'], cs[x]['llt_name'])
        if lexical.input_guard(case['input'], cs[x]['llt_name'], cs[x]['llt_name']):
            proof = None
        guards = list(f.get('conflicts', [])) + list(f.get('unsupported_specificity', []))
        supported = bool((f.get('wording_equivalent') or proof) and (not guards))
        result['input_support'][x] = {'positive': supported, 'existing_wording_support': f.get('wording_equivalent'), 'new_lexical_proof': proof, 'clinical_guards': guards, 'saved_retrieval_conflict': f.get('retrieval_conflict', False), 'unrepresented_input_information': f.get('unrepresented_input_information', []), 'recorded_identity_alternatives': f.get('alternatives', []), 'score': scores[x]}
        if x in group:
            if not supported:
                b.append('group_member_input_support_unestablished:' + x)
            if f.get('retrieval_conflict'):
                b.append('saved_retrieval_conflict:' + x)
            if 'transition_not_encoded_in_label' in f.get('unrepresented_input_information', []):
                b.append('transition_unresolved:' + x)
            for rival in f.get('alternatives', []):
                if rival not in group:
                    b.append('outside_recorded_identity_alternative:' + rival)
        elif supported:
            b.append('outside_supported_wording_identity:' + x)
    replaced = {'tie_or_insufficient_wording_margin', 'wording_equivalence_collision_semantic_not_a_tie_breaker', 'reference_wording_identity_unresolved', 'approximate_wording_not_yet_supported_for_automatic_LLT'}
    b.extend(('inherited:' + r for r in case.get('old_reasons', []) if r not in replaced))
    names = [lexical.text(cs[x]['llt_name']) for x in group]
    if len(names) != len(set(names)):
        b.append('identical_wording_distinct_identity_collision')
    representative = min(group, key=lambda x: faithful_key(case['input'], cs[x]['llt_name']))
    result['representative_convention'] = {'candidate': representative, 'key': faithful_key(case['input'], cs[representative]['llt_name']), 'authority': 'input literal/grammar/ordered-token preference; final official-name alphabetical convention is not clinical confidence'}
    result['blockers'] = sorted(set(b))
    result['accepted'] = not result['blockers']
    if result['accepted']:
        result['representative'] = cs[representative]
    return result
