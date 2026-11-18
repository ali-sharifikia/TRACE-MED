"""Opt-in complete-input authority over lexical numeric gates; never reads labels."""
from runtime_resources import resource as asp_resource
import copy
import math
import direct_role_group as old
MODES = ('A', 'G', 'M', 'P')

def __getattr__(name):
    return getattr(old, name)

from decision_recorder import computation

@computation('comparison.proof_priority')
def revise(case, original, mode):
    if mode not in MODES:
        raise ValueError('Unknown proof-priority condition')
    out = copy.deepcopy(original)
    out['experimental_condition'] = mode
    if mode in ('A', 'G') or out['accepted']:
        return out
    if case['baseline']['llt'] or len(case['baseline']['pt']) != 1:
        return out
    group = out.get('group', [])
    comparisons = out.get('comparisons', {})
    n = out.get('numeric') or {}
    missing = []
    if not group or any((not comparisons.get(c, {}).get('positive') for c in group)):
        missing.append('complete_input_proof_missing')
    outside = sorted(set(comparisons) - set(group))
    positive_rivals = [c for c in outside if comparisons[c].get('positive')]
    if positive_rivals:
        missing.append('outside_positive_input_proof')
    score, margin = (n.get('score'), n.get('margin'))
    if not all((isinstance(v, (float, int)) and (not isinstance(v, bool)) and math.isfinite(v) for v in (score, margin))):
        missing.append('invalid_numerics')
    else:
        if not 0 <= score <= 1 or margin <= 1e-10:
            missing.append('no_positive_full_pool_margin')
        if mode == 'M' and score < 0.7:
            missing.append('retained_score_floor')
    removable = {'fixed_numeric_gate'} | ({'score_below_070'} if mode == 'P' else set())
    retained = [b for b in out['blockers'] if b not in removable]
    if retained:
        missing.append('independent_requirements_remain')
    out['proof_priority'] = {'eligible': not missing, 'failures': sorted(set(missing)), 'original_blockers': list(out['blockers']), 'retained_blockers': retained, 'discharged': sorted(set(out['blockers']) & removable) if not missing else [], 'outside_candidates': outside, 'outside_positive_candidates': positive_rivals, 'original_numerics': copy.deepcopy(n), 'authority': 'existing positive full-input proof; experimental numeric override', 'unknown_rivals_excluded': False}
    if not missing:
        representative = out.get('representative')
        if not representative or representative['candidate_id'] not in group:
            raise ValueError('Missing unchanged group representative')
        out['blockers'] = []
        out['accepted'] = True
        out['prediction'] = {'pt': list(case['baseline']['pt']), 'llt': [representative['llt_code']]}
    return out

def analyze(case, ref, sem, lex, *, raw=None, mode='A', context=None):
    original = old.analyze(case, ref, sem, lex, raw=raw, mode='A' if mode == 'A' else 'E', context=context)
    return revise(case, original, mode)
