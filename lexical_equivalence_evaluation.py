"""Reporting-only, pairwise lexical proofs. No predictions, labels or acceptance."""
from runtime_resources import resource as asp_resource
import csv, re
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
VERSION = 'reporting-lexical-equivalence-1'
GUIDE = 'https://alt.meddra.org/files_acrobat/intguide_29_1_English.pdf'
SOURCE_PAIRS = [('arthritis', 'joint inflammation', 'synonym', '3.1, printed p7 / PDF p13'), ('acquired immunodeficiency syndrome', 'aids', 'lexical variant', '3.1, printed p7 / PDF p13'), ('biopsy tongue', 'tongue biopsy', 'lexical inversion', '3.1, printed p7 / PDF p13'), ('diarrhoea', 'diarrhea', 'spelling variant', '4.1, printed p14 / PDF p20')]
SPECIMENS = {'blood', 'serum', 'plasma', 'urine'}
PROPERTIES = {'level', 'levels', 'count', 'concentration', 'activity', 'proportion', 'percentage', 'duration', 'ratio'}
DIRECTIONS = {'increase': 'increase', 'increased': 'increase', 'decrease': 'decrease', 'decreased': 'decrease'}
SCOPE = re.compile('\\b(?:no|not|without|non|denies|denied|absence|absent|possible|possibly|suspected|probable|uncertain|history|previous|prior|before|after|following|during|then|from|to|with|due|secondary|and|or|versus|nos|unspecified|nonspecific)\\b|[0-9<>=/%;,]', re.I)
ATTRIBUTES = {'specimen': SPECIMENS, 'property': PROPERTIES, 'qualifier': {'free', 'total', 'fasting', 'conjugated', 'unconjugated', 'absolute', 'relative'}, 'laterality': {'left', 'right', 'bilateral', 'unilateral'}, 'course': {'acute', 'chronic', 'recurrent', 'persistent', 'transient', 'progressive'}, 'specificity': {'major', 'minor', 'severe', 'mild', 'localized', 'generalized', 'malignant', 'benign', 'infectious', 'traumatic'}, 'site': {'tongue', 'lip', 'lips', 'leg', 'arm', 'face', 'cord', 'brain', 'lung', 'liver', 'kidney'}}

def text(s):
    return ' '.join(str(s).casefold().split())

@lru_cache(maxsize=60000)
def frame(s):
    """Ordered nominal phrase plus a SINGLE boundary predicate; no bag-of-words."""
    s = text(s)
    t = s.split()
    result = {'original': s, 'parsed': False, 'transformations': []}
    if SCOPE.search(s) or not re.fullmatch('[a-z ]+', s):
        return result
    positions = [i for i, w in enumerate(t) if w in DIRECTIONS]
    if len(positions) != 1 or positions[0] not in (0, len(t) - 1) or len(t) < 2:
        return result
    i = positions[0]
    predicate = t[i]
    body = t[1:] if i == 0 else t[:-1]
    if any((w in {'low', 'high', 'raised', 'elevated', 'reduced', 'abnormal', 'positive', 'negative'} for w in body)):
        return result
    if body[-1:] == ['levels']:
        body = body[:-1] + ['level']
        result['transformations'].append('explicit terminal measurement noun levels -> level')
    specimen = None
    specs = [j for j, w in enumerate(body) if w in SPECIMENS]
    if len(specs) == 1:
        j = specs[0]
        rest = body[:j] + body[j + 1:]
        if (len(rest) == 1 or (len(rest) == 2 and rest[-1] in PROPERTIES)) and len(rest[0]) >= 3 and (rest[0] not in PROPERTIES | SPECIMENS):
            if j in (0, 1):
                specimen = body[j]
                body = rest
                result['transformations'].append('specimen before/after single head; specimen retained')
    result.update(parsed=True, predicate=DIRECTIONS[predicate], ordered_entity=body, specimen=specimen, predicate_position='prefix' if i == 0 else 'suffix')
    if predicate != DIRECTIONS[predicate]:
        result['transformations'].append('boundary predicate grammatical inflection')
    return result

def differences(a, b):
    """Explicit surface attributes only. Absence means specificity, not contradiction."""
    ta = set(re.findall('[a-z]+', text(a)))
    tb = set(re.findall('[a-z]+', text(b)))
    out = []
    for kind, words in ATTRIBUTES.items():
        x, y = (ta & words, tb & words)
        if kind == 'property':
            x = {w.rstrip('s') if w == 'levels' else w for w in x}
            y = {w.rstrip('s') if w == 'levels' else w for w in y}
        if x != y:
            out.append({'attribute': kind, 'left': sorted(x), 'right': sorted(y), 'interpretation': 'explicit attribute or specificity difference; not necessarily opposing clinical truth'})
    da = {DIRECTIONS[w] for w in ta if w in DIRECTIONS}
    db = {DIRECTIONS[w] for w in tb if w in DIRECTIONS}
    if da and db and (da != db):
        out.append({'attribute': 'direction', 'left': sorted(da), 'right': sorted(db)})
    na = ta & {'no', 'not', 'without', 'non'}
    nb = tb & {'no', 'not', 'without', 'non'}
    if na != nb:
        out.append({'attribute': 'negation', 'left': sorted(na), 'right': sorted(nb)})
    va = re.findall('[<>]=?|[0-9]+(?:\\.[0-9]+)?|[%/]', a)
    vb = re.findall('[<>]=?|[0-9]+(?:\\.[0-9]+)?|[%/]', b)
    if va != vb:
        out.append({'attribute': 'numerical_surface', 'left': va, 'right': vb})
    return out

def input_guard(original, a, b):
    if SCOPE.search(text(original)):
        return ['original input assertion/numeric/relationship scope not established for new credit']
    tokens = set(re.findall('[a-z]+', text(original)))
    plus = bool(tokens & {'increase', 'increases', 'increased', 'increasing'})
    minus = bool(tokens & {'decrease', 'decreases', 'decreased', 'decreasing'})
    reasons = []
    if plus and minus:
        reasons.append('multiple input direction assertions')
    for name in (a, b):
        f = frame(name)
        if f['parsed'] and (plus and f['predicate'] == 'decrease' or (minus and f['predicate'] == 'increase')):
            reasons.append('input direction conflicts with proposed pair')
        s = set(re.findall('[a-z]+', text(name))) & SPECIMENS
        i = tokens & SPECIMENS
        if s and i and (s != i):
            reasons.append('input specimen conflicts with proposed pair')
    return sorted(set(reasons))

def linguistic(a, b):
    a, b = (text(a), text(b))
    for x, y, kind, section in SOURCE_PAIRS:
        if {a, b} == {x, y}:
            return {'route': 'A', 'rule': 'official_explicit_complete_name_pair', 'witness': {'source': GUIDE, 'section': section, 'names': [x, y], 'relationship': kind}}
    if a == b:
        return {'route': 'B', 'rule': 'case_whitespace_only', 'witness': {'left': a, 'right': b}}
    fa, fb = (frame(a), frame(b))
    if fa['parsed'] and fb['parsed'] and all((fa[k] == fb[k] for k in ('predicate', 'ordered_entity', 'specimen'))):
        return {'route': 'B', 'rule': 'bounded_ordered_predicate_measurement_frame', 'witness': {'left': fa, 'right': fb}}
    return None

class Reference:

    def __init__(self, csv_path, llt_path, pt_path):
        self.rows = {}
        self.groups = defaultdict(set)
        raw = {}
        pt = {}
        for line in Path(llt_path).read_text(encoding='ascii').splitlines():
            v = line.split('$')
            assert len(v) == 12 and v[0] not in raw
            raw[v[0]] = {'name': v[1], 'pt': v[2], 'current': v[9] == 'Y'}
        for line in Path(pt_path).read_text(encoding='ascii').splitlines():
            v = line.split('$')
            assert len(v) == 12 and v[0] not in pt
            pt[v[0]] = v[1]
        with Path(csv_path).open(encoding='utf-8-sig') as reference_stream:
            for row in csv.DictReader(reference_stream):
                code = row['llt_code']
                assert code not in self.rows
                r = raw[code]
                assert r['current'] and (r['name'], r['pt']) == (row['llt_name'], row['pt_code']) and (pt[r['pt']] == row['pt_name'])
                self.rows[code] = {**row, 'current': True, 'identity_evidence': 'exact CSV reconciled to llt.asc current Y and pt.asc'}
                self.groups[row['pt_code']].add(code)
        assert set(self.rows) == {k for k, v in raw.items() if v['current']}
        self.raw = raw
        self.metadata = {'release': 'unknown; no independently bound release identifier', 'current_rows': len(self.rows), 'raw_LLT_rows': len(raw), 'PT_rows': len(pt), 'full_current_reference_reconciliation': True}

    def identity(self, code):
        if code in self.rows:
            return self.rows[code]
        r = self.raw.get(code)
        return {'llt_code': code, 'identity_evidence': 'noncurrent' if r and (not r['current']) else 'missing_or_unresolved', 'raw': r, 'current': False}

def compare_pair(original, left, right, ref):
    a, b = (ref.identity(left), ref.identity(right))
    r = {'left': left, 'right': right, 'left_identity': a, 'right_identity': b, 'status': 'unresolved', 'route': None, 'rule': None, 'reasons': [], 'retained_differences': []}
    if left == right:
        r.update(status='exact', rule='strict_identity')
        return r
    if not a.get('current') or not b.get('current'):
        r['reasons'] = ['missing_or_noncurrent_reference_identity']
        return r
    da = differences(a['llt_name'], b['llt_name'])
    r['retained_differences'] = da
    if a['pt_code'] != b['pt_code']:
        r['reasons'].append('different_current_parents; not proof of clinical non-equivalence')
        if da:
            r['status'] = 'meaningful_difference'
        return r
    proof = linguistic(a['llt_name'], b['llt_name'])
    guard = input_guard(original, a['llt_name'], b['llt_name'])
    if proof and (not guard):
        r.update(status='verified_equivalent', **proof)
        return r
    r['reasons'] = guard or ['no complete pair-specific lexical proof; same parent is insufficient']
    if da:
        r['status'] = 'meaningful_difference'
    return r

def match_sets(predicted, expected, original, ref):
    """Bipartite one-to-one matching; never collapse codes by parent/equivalence."""
    p = sorted(set(predicted))
    g = sorted(set(expected))
    pairs = {(x, y): compare_pair(original, x, y, ref) for x in p for y in g}

    def matching(allowed):
        chosen = {}

        def visit(x, seen):
            for y in g:
                if y in seen or pairs[x, y]['status'] not in allowed:
                    continue
                seen.add(y)
                if y not in chosen or visit(chosen[y], seen):
                    chosen[y] = x
                    return True
            return False
        for x in p:
            visit(x, set())
        return [{'predicted': x, 'expected': y, 'status': pairs[x, y]['status'], 'route': pairs[x, y]['route'], 'rule': pairs[x, y]['rule']} for y, x in sorted(chosen.items())]
    links = matching({'exact', 'verified_equivalent'})
    ok = bool(p) and len(p) == len(g) == len(links)
    potential = matching({'exact', 'verified_equivalent', 'unresolved'})
    status = 'agreement' if ok else 'cardinality_difference' if len(p) != len(g) else 'unresolved' if len(p) == len(potential) else 'meaningful_difference'
    parents = lambda codes: {ref.rows[c]['pt_code'] for c in codes} if codes and all((c in ref.rows for c in codes)) else None
    pp, gp = (parents(p), parents(g))
    return {'strict': bool(p) and p == g, 'equivalence': ok, 'disposition': status, 'correspondences': links, 'parent_agreement': pp == gp if pp is not None and gp is not None else None, 'predicted_current_parents': sorted(pp) if pp is not None else None, 'expected_current_parents': sorted(gp) if gp is not None else None, 'pairs': list(pairs.values())}
