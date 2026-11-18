"""General source/reference transformations; no gold, term-ID or correction tables."""
from runtime_resources import resource as asp_resource
import copy, re
from pathlib import Path
from functools import lru_cache
import module_loading as reuse
import equivalence_group as group
import event_comparison as ranker
from grammatical_discrimination import SPECIMENS, DIRECTION, MEASURE_QUALIFIERS, ATTRIBUTE_ROLES
MODES = ('CONTROL', 'L', 'H', 'X', 'ALL')
PROTECTED = set(SPECIMENS) | set(DIRECTION) | set(MEASURE_QUALIFIERS) | set(ATTRIBUTE_ROLES) | {'count', 'counts', 'concentration', 'concentrations', 'percentage', 'percent', 'activity', 'duration', 'time', 'interval', 'left', 'right', 'bilateral', 'no', 'not', 'without', 'non', 'and', 'or', 'after', 'before', 'following', 'due', 'to', 'from', 'total', 'free', 'fasting', 'nos', 'unspecified', 'level', 'levels', 'value', 'values', 'high', 'low', 'increased', 'decreased'}

def norm(s):
    return ' '.join(s.lower().split())

class Lexicon:

    def __init__(self, source):
        self.source = source
        self.agr = {}
        self.noun_euis = set()
        self.spl = {}
        for r in source['records']['LEX/LRAGR']:
            f = r['fields']
            self.agr.setdefault(norm(f[1]), []).append(r)
            if f[2] == 'noun':
                self.noun_euis.add(f[0])
        for r in source['records']['LEX/LRSPL']:
            self.spl.setdefault(norm(r['fields'][1]), []).append(r)

    @lru_cache(maxsize=20000)
    def token(self, word):
        w = norm(word)
        rows = self.agr.get(w, [])
        spell = self.spl.get(w, [])
        if not re.fullmatch('[a-z]+', w) or w in PROTECTED or word.isupper():
            return None
        if rows and any((r['fields'][2] != 'noun' for r in rows)):
            return None
        agr = [r for r in rows if r['fields'][2] == 'noun' and ('thr_sing' in r['fields'][3] or 'thr_plur' in r['fields'][3])]
        spl = [r for r in spell if r['fields'][0] in self.noun_euis]
        targets = {norm(r['fields'][5]) for r in agr} | {norm(r['fields'][2]) for r in spl}
        if len(targets) != 1:
            return None
        target = next(iter(targets))
        if target == w or target in PROTECTED or (not re.fullmatch('[a-z]+', target)):
            return None
        return {'canonical': target, 'witnesses': agr + spl, 'rule': 'documented_noun_citation_form', 'original': word}

    def view(self, text):
        if re.search('\\d|\\b(?:one|two|three|four|five|six|seven|eight|nine|ten|multiple)\\b', text, re.I):
            return {'original': text, 'view': text, 'transformations': []}
        parts = []
        last = 0
        proof = []
        for m in re.finditer('[A-Za-z]+', text):
            w = self.token(m[0])
            parts.append(text[last:m.start()])
            parts.append(w['canonical'] if w else m[0])
            last = m.end()
            if w:
                proof.append({**w, 'span': [m.start(), m.end()]})
        parts.append(text[last:])
        return {'original': text, 'view': ''.join(parts), 'transformations': proof}

def specimens(text):
    return sorted({m.lower() for m in re.findall('\\b(?:blood|serum|plasma|urine)\\b', text, re.I)})

def context_view(text, row, mode, lex):
    inp = lex.view(text) if mode in ('L', 'ALL') else {'original': text, 'view': text, 'transformations': []}
    cand = lex.view(row['llt_name']) if mode in ('L', 'ALL') else {'original': row['llt_name'], 'view': row['llt_name'], 'transformations': []}
    parent = lex.view(row['pt_name']) if mode in ('L', 'ALL') else {'original': row['pt_name'], 'view': row['pt_name'], 'transformations': []}
    inherited = None
    a, b, p = (specimens(inp['view']), specimens(cand['view']), specimens(parent['view']))
    if mode in ('H', 'ALL') and len(a) == 1 and (not b) and (a == p):
        inherited = {'attribute': 'specimen', 'value': a[0], 'official_llt': row['llt_code'], 'official_pt': row['pt_code'], 'parent_name': row['pt_name'], 'authority': 'current reference parent context; not inferred patient information'}
        leading = re.match('^(?:increased|decreased|elevated|raised|reduced|increase|decrease|low|high|abnormal|positive|negative)\\b', cand['view'], re.I)
        view = cand['view'][:leading.end()] + ' ' + a[0] + cand['view'][leading.end():] if leading else a[0] + ' ' + cand['view']
        cand = {**cand, 'view': view}
    return {'input': inp, 'candidate': cand, 'parent': parent, 'inherited': inherited}

def feature_support(text, label, case, mode, lex, fn=ranker.support):
    rows = [r for r in case['candidates'] if r['llt_name'] == label and r['pt_code'] in case['baseline']['pt']]
    if not rows:
        a = lex.view(text)['view'] if mode in ('L', 'ALL') else text
        b = lex.view(label)['view'] if mode in ('L', 'ALL') else label
        return fn(a, b)
    views = [context_view(text, r, mode, lex) for r in rows]
    if len({v['candidate']['view'] for v in views}) != 1:
        return fn(text, label)
    return fn(views[0]['input']['view'], views[0]['candidate']['view'])

def contradiction_margin(score, outside, scores, features):
    excluded = {cid: features[cid]['conflicts'] for cid in outside if features[cid].get('conflicts')}
    retained = [cid for cid in outside if cid not in excluded]
    maximum = max((scores[cid] for cid in retained), default=0.0)
    return {'excluded': excluded, 'remaining_competitors': retained, 'margin': score - maximum, 'passed': score >= 0.7 and score - maximum >= 0.01 and (score > maximum), 'outside_maximum': maximum if retained else None, 'authority': 'explicit conflicts only; no candidate score changes'}

def propose(case, ref, semantics, mode, lex):
    if mode not in MODES:
        raise ValueError('Unknown mode')
    if case['baseline']['llt']:
        return {'accepted': False, 'blockers': ['retained_automatic'], 'representative': None}
    rawgroups = {}
    for r in case['candidates']:
        rawgroups.setdefault(r['pt_code'], []).append(r)
    before = ranker.rank(case['input'], [], rawgroups, semantics)
    if set(before['features']) != set(case['scores']) or any((abs(before['features'][cid]['score'] - v) > 1e-10 for cid, v in case['scores'].items())):
        raise ValueError('Original ranking does not reproduce')
    if mode == 'CONTROL':
        return group.propose(case, ref)
    c = copy.deepcopy(case)
    views = {r['candidate_id']: context_view(case['input'], r, mode, lex) for r in c['candidates']}
    input_view = lex.view(case['input'])['view'] if mode in ('L', 'ALL') else case['input']
    gs = {pt: [{**r, 'llt_name': views[r['candidate_id']]['candidate']['view'], 'pt_name': views[r['candidate_id']]['parent']['view']} for r in rs] for pt, rs in rawgroups.items()}
    ranked = ranker.rank(input_view, [], gs, semantics) if mode in ('L', 'H', 'ALL') else before
    ev = {}
    for r in c['candidates']:
        cid = r['candidate_id']
        f = c['features'][cid]
        rf = ranked['features'][cid]
        v = views[cid]
        comparison = ranker.comparison(v['input']['view'], v['candidate']['view'])
        s = ranker.support(v['input']['view'], v['candidate']['view'])
        ps = ranker.support(v['input']['view'], v['parent']['view'])
        conflicts = sorted(set(s['conflicts']) | set(ps['conflicts']) | set(rf['conflicts']))
        specificity = sorted(set(s['unsupported_specificity']) | set(ps['unsupported_specificity']) | set(rf['unsupported_specificity']))
        proof = group.wording_proof(v['input']['view'], v['candidate']['view'])
        positive = bool((comparison['equivalent'] or proof or rf['wording_equivalent']) and (not comparison['protected_differences']) and (not conflicts) and (not specificity))
        changed = v['input']['view'] != case['input'] or v['candidate']['view'] != r['llt_name'] or v['parent']['view'] != r['pt_name']
        if changed:
            f.update(wording_equivalent=positive, conflicts=conflicts, unsupported_specificity=specificity)
            f['alternatives'] = sorted(set(f['alternatives']) | {a['candidate_id'] for a in rf.get('unresolved_reference_wording_alternatives', [])})
        c['scores'][cid] = rf['score']
        ev[cid] = {'views': v, 'comparison': comparison, 'positive': positive, 'changed_view': changed, 'original_score': case['scores'][cid], 'score': rf['score'], 'semantic': rf['semantic'], 'conflicts': conflicts, 'unsupported_specificity': specificity}
    if mode in ('L', 'H', 'ALL') and any((x['changed_view'] for x in ev.values())):
        c['old_reasons'] = [x for x in c['old_reasons'] if x not in ('insufficient_input_score', 'clinical_or_specificity_rejection', 'selected_scope_or_specificity_not_supported')]
    isolated = reuse.isolated('_general_groups', group.__file__)

    def pair(original, x, y, r):
        result = group.compare_pair(original, x, y, r)
        if result['status'] in ('exact', 'verified_equivalent'):
            return result
        a, b = (r.identity(x), r.identity(y))
        if not a.get('current') or not b.get('current') or a['pt_code'] != b['pt_code']:
            return result
        vx, vy = (views['LLT:' + x], views['LLT:' + y])
        if not (ev['LLT:' + x]['changed_view'] or ev['LLT:' + y]['changed_view']):
            return result
        p = group.wording_proof(vx['candidate']['view'], vy['candidate']['view'])
        comp = ranker.comparison(vx['candidate']['view'], vy['candidate']['view'])
        if not p or not comp['equivalent'] or comp['protected_differences']:
            return result
        return {**result, 'status': 'verified_equivalent', 'route': 'experimental_source_reference', 'rule': 'contextual_full_phrase_grammar', 'reasons': [], 'retained_differences': [], 'witness': {'left': vx, 'right': vy, 'proof': p}}
    isolated.compare_pair = pair
    a = isolated.propose(c, ref)
    if len(a['group']) == 1 and ev[a['group'][0]]['changed_view'] and a['input_support'][a['group'][0]]['positive']:
        a['blockers'] = [b for b in a['blockers'] if b != 'no_multiple_verified_equivalent_identities']
    if mode in ('X', 'ALL') and a.get('numeric'):
        a['contradiction_margin'] = contradiction_margin(a['numeric']['score'], a['outside'], c['scores'], c['features'])
        if a['contradiction_margin']['passed']:
            a['blockers'] = [b for b in a['blockers'] if b != 'group_score_or_margin']
    if not case['saved_score_observations']:
        a['blockers'].append('no_saved_score_observations')
    a['blockers'] = sorted(set(a['blockers']))
    a['accepted'] = not a['blockers']
    a['representative'] = next((r for r in c['candidates'] if r['candidate_id'] == a['representative_convention']['candidate'])) if a['accepted'] else None
    a.update(experiment='general_llt_v1_31', mode=mode, comparisons=ev, new_scores=c['scores'])
    return a
