from runtime_resources import resource as asp_resource
DIRECTIONAL_OPPOSITIONS = (({'increase', 'increased', 'increasing', 'elevated', 'raised', 'high'}, {'decrease', 'decreased', 'decreasing', 'reduced', 'lowered', 'low'}), ({'positive'}, {'negative'}), ({'prolonged', 'longer'}, {'shortened', 'shorter'}))
DIRECTIONAL_MODIFIERS = set().union(*[left_group | right_group for left_group, right_group in DIRECTIONAL_OPPOSITIONS])
NEGATION_TOKENS = {'no', 'not', 'without', 'absent', 'absence', 'denied', 'denies'}
LATERALITY_TOKENS = {'left', 'right', 'bilateral', 'unilateral'}
TEMPORAL_OPPOSITIONS = (({'acute'}, {'chronic'}), ({'early'}, {'late'}))
TEMPORAL_MODIFIERS = set().union(*[left_group | right_group for left_group, right_group in TEMPORAL_OPPOSITIONS])
SEVERITY_TOKENS = {'mild', 'moderate', 'severe'}

def contains_opposition(input_tokens: set[str], candidate_tokens: set[str], opposition_groups: tuple) -> bool:
    """
    Determine whether two token sets contain opposing
    modifiers.
    """
    for left_group, right_group in opposition_groups:
        input_has_left = bool(input_tokens & left_group)
        input_has_right = bool(input_tokens & right_group)
        candidate_has_left = bool(candidate_tokens & left_group)
        candidate_has_right = bool(candidate_tokens & right_group)
        if input_has_left and candidate_has_right:
            return True
        if input_has_right and candidate_has_left:
            return True
    return False

def detect_laterality_conflict(input_tokens: set[str], candidate_tokens: set[str]) -> bool:
    """
    Detect clear laterality disagreements.
    """
    if 'left' in input_tokens and 'right' in candidate_tokens:
        return True
    if 'right' in input_tokens and 'left' in candidate_tokens:
        return True
    one_sided_tokens = {'left', 'right', 'unilateral'}
    if 'bilateral' in input_tokens and bool(candidate_tokens & one_sided_tokens):
        return True
    if 'bilateral' in candidate_tokens and bool(input_tokens & one_sided_tokens):
        return True
    return False

def detect_severity_conflict(input_tokens: set[str], candidate_tokens: set[str]) -> bool:
    """
    Detect differing explicit severity modifiers.
    """
    input_severity = input_tokens & SEVERITY_TOKENS
    candidate_severity = candidate_tokens & SEVERITY_TOKENS
    return bool(input_severity and candidate_severity and (input_severity != candidate_severity))

def detect_unsupported_specificity(input_tokens: set[str], candidate_tokens: set[str]) -> bool:
    """
    Detect whether the candidate introduces a protected
    modifier category absent from the input.

    This does not attempt to detect anatomical specificity.
    """
    protected_categories = (DIRECTIONAL_MODIFIERS, LATERALITY_TOKENS, TEMPORAL_MODIFIERS, SEVERITY_TOKENS)
    for modifier_category in protected_categories:
        input_modifiers = input_tokens & modifier_category
        candidate_modifiers = candidate_tokens & modifier_category
        if not input_modifiers and candidate_modifiers:
            return True
    return False

def detect_conflict_evidence(input_tokens: set[str], candidate_tokens: set[str]) -> dict:
    """
    Generate general conflict evidence for one input-candidate
    pair.

    These flags provide evidence only. They do not reject,
    remove, rerank or accept candidates.
    """
    directionality_conflict = contains_opposition(input_tokens=input_tokens, candidate_tokens=candidate_tokens, opposition_groups=DIRECTIONAL_OPPOSITIONS)
    input_has_negation = bool(input_tokens & NEGATION_TOKENS)
    candidate_has_negation = bool(candidate_tokens & NEGATION_TOKENS)
    negation_conflict = input_has_negation != candidate_has_negation
    laterality_conflict = detect_laterality_conflict(input_tokens=input_tokens, candidate_tokens=candidate_tokens)
    temporal_conflict = contains_opposition(input_tokens=input_tokens, candidate_tokens=candidate_tokens, opposition_groups=TEMPORAL_OPPOSITIONS)
    severity_conflict = detect_severity_conflict(input_tokens=input_tokens, candidate_tokens=candidate_tokens)
    unsupported_specificity = detect_unsupported_specificity(input_tokens=input_tokens, candidate_tokens=candidate_tokens)
    conflict_flags: list[str] = []
    if directionality_conflict:
        conflict_flags.append('directionality_conflict')
    if negation_conflict:
        conflict_flags.append('negation_conflict')
    if laterality_conflict:
        conflict_flags.append('laterality_conflict')
    if temporal_conflict:
        conflict_flags.append('temporal_conflict')
    if severity_conflict:
        conflict_flags.append('severity_conflict')
    if unsupported_specificity:
        conflict_flags.append('unsupported_specificity')
    return {'directionality_conflict': directionality_conflict, 'negation_conflict': negation_conflict, 'laterality_conflict': laterality_conflict, 'temporal_conflict': temporal_conflict, 'severity_conflict': severity_conflict, 'anatomical_conflict': None, 'unsupported_specificity': unsupported_specificity, 'conflict_flags': conflict_flags}
