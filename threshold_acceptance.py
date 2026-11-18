"""User-selected graded threshold endpoint. No label access or live I/O."""
from runtime_resources import resource as asp_resource
import copy
import math
import proof_priority as proof
MODES = ('P', 'T')
REPLACE = {'fixed_numeric_gate', 'score_below_070', 'margin_below_001'}

def __getattr__(name):
    return getattr(proof, name)

from decision_recorder import computation

@computation('comparison.threshold')
def revise(case, original, mode):
    if mode not in MODES:
        raise ValueError('Unknown threshold condition')
    retained = proof.revise(case, original, 'P')
    if mode == 'P' or retained['accepted']:
        return retained
    out = copy.deepcopy(original)
    base = case['baseline']
    reasons = [b for b in out.get('blockers', []) if b not in REPLACE]
    if base['llt'] or len(base['pt']) != 1 or out.get('branch') != 'accepted_PT':
        reasons.append('not_deferred_single_PT')
    group = out.get('group', [])
    rep = out.get('representative')
    n = out.get('numeric') or {}
    score, margin = (n.get('score'), n.get('margin'))
    if not group or not rep or rep.get('candidate_id') not in group:
        reasons.append('missing_group_representative')
    elif rep['pt_code'] not in base['pt']:
        reasons.append('wrong_parent')
    if not all((isinstance(x, (int, float)) and (not isinstance(x, bool)) and math.isfinite(x) for x in (score, margin))):
        reasons.append('missing_or_invalid_scores')
    elif not 0 <= score <= 1 or margin <= 1e-10:
        reasons.append('nonpositive_or_invalid_numerics')
    out['threshold_policy'] = {'score_floor': 0.0, 'margin_floor': 0.0, 'strict_positive_margin_epsilon': 1e-10, 'original_blockers': out.get('blockers', []), 'retained_blockers': sorted(set(reasons)), 'all_members_have_input_proof': bool(group) and all((out.get('comparisons', {}).get(c, {}).get('positive') for c in group)), 'authority': 'experimental graded acceptance; not verified full-expression equivalence'}
    out['experimental_condition'] = mode
    out['blockers'] = sorted(set(reasons))
    out['accepted'] = not reasons
    if not reasons:
        out['prediction'] = {'pt': list(base['pt']), 'llt': [rep['llt_code']]}
    return out

def analyze(case, ref, sem, lex, *, raw=None, mode='P', context=None):
    original = proof.old.analyze(case, ref, sem, lex, raw=raw, mode='E', context=context)
    return revise(case, original, mode)
