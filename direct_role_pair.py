"""Direct candidate-pair role proof; no labels, scores, IDs or I/O.

The frozen runtime policy defines R/E. Lexical preservation is a veto,
never a bag-of-words equivalence rule. Positive authority always needs direct,
bidirectional complete-role comparison AND complete bidirectional clinical checks.
"""
from runtime_resources import resource as asp_resource
from collections import Counter
import copy
import re
import event_comparison as ranker
import grammatical_discrimination as grammar
VERSION = 'direct_role_pair_v1'
MODES = ('R', 'E')
UNSAFE = re.compile('[^A-Za-z\\s-]|\\b(?:no|not|non|without|except|excluding|and|or|with|versus|due|from|to|after|before|during|following|history|previous|prior|possible|possibly|suspected|probable|uncertain|may|might|nos|unspecified|nonspecific|beyond|through|into)\\b', re.I)
CLINICAL_FIELDS = ('contradictions', 'unrepresented', 'unsupported_additions', 'unresolved')

def _view(value):
    if isinstance(value, str):
        return {'original': value, 'view': value, 'transformations': [], 'unresolved': []}
    if not isinstance(value, dict) or not isinstance(value.get('view'), str):
        raise ValueError('Expected complete source view or original string')
    return copy.deepcopy(value)

def _uncertain(view):
    if view.get('unresolved'):
        return True
    for key in ('phrase_view', 'source_view', 'scope_spelling', 'event_frame'):
        value = view.get(key)
        if isinstance(value, dict) and _uncertain(value):
            return True
    return False

def _tokens(text):
    return list(grammar.phrase(text)['wording_tokens'])

def _recorded_tokens(text, frame):
    """E can explain only transformations already explicitly recorded by v1.9."""
    tokens = _tokens(text)
    evidence = []
    for row in frame.get('transformations', []):
        if row.get('rule') == 'existing_anatomical_noun_adjective':
            before, after = (row['token'], row['replacement'])
            source = row.get('source', '')
            spans = row.get('spans', [])
            if not spans or any((source[a:b].casefold() != before for a, b in spans)) or ranker.SITE.get(before) != after:
                return (tokens, evidence, ['invalid_existing_anatomy_witness'])
            changed = sum((w == before for w in tokens))
            tokens = [after if w == before else w for w in tokens]
            evidence.append({**copy.deepcopy(row), 'applied_tokens': changed})
        elif row.get('rule') == 'single_event_course_argument':
            before, after = (row['supporting_text'], row['course'])
            source = row.get('source', '')
            a, b = row['span']
            if source[a:b] != before or ranker.COURSE.get(before) != after:
                return (tokens, evidence, ['invalid_existing_course_witness'])
            old = _tokens(before)
            if len(old) != 1:
                return (tokens, evidence, ['unresolved_course_token_attachment'])
            changed = sum((w == old[0] for w in tokens))
            tokens = [after if w == old[0] else w for w in tokens]
            evidence.append({**copy.deepcopy(row), 'applied_tokens': changed})
    return (tokens, evidence, [])

def prove(left_view, right_view, mode='R', clinical_compare=None):
    """Prove a DIRECT whole-label relationship, never via a shared input/PT.

    `clinical_compare(a,b)` must return the existing complete clinical comparison
    schema. Runtime integration separately validates current identities, common
    parent, original input support, complete rivals and every acceptance guard.
    """
    if mode not in MODES:
        raise ValueError('Unknown direct role proof condition')
    if not callable(clinical_compare):
        raise ValueError('A bound complete clinical comparison is required')
    views = [_view(left_view), _view(right_view)]
    a, b = (v['view'] for v in views)
    result = {'version': VERSION, 'mode': mode, 'positive': False, 'proof': None, 'reasons': [], 'views': views, 'direct': [], 'clinical': [], 'preservation': {}, 'authority': None}
    reasons = result['reasons']
    if not a.strip() or not b.strip():
        reasons.append('empty_complete_expression')
    if any((_uncertain(v) for v in views)):
        reasons.append('unresolved_source_view')
    if any((UNSAFE.search(v['view']) for v in views)):
        reasons.append('numerical_polarity_or_relationship_scope_outside_rule')
    if reasons:
        return result
    directions = [ranker.comparison(a, b), ranker.comparison(b, a)]
    result['direct'] = directions
    for comparison in directions:
        if not comparison['equivalent'] or not comparison['shared_argument'] or (not comparison['source_frame']['safe']) or (not comparison['candidate_frame']['safe']) or comparison['protected_differences'] or comparison['omitted_course'] or comparison['added_course'] or comparison['conflicting_course']:
            reasons.append('direct_complete_role_correspondence_unestablished')
    clinical = [clinical_compare(a, b), clinical_compare(b, a)]
    result['clinical'] = clinical
    for comparison in clinical:
        if not isinstance(comparison, dict) or any((k not in comparison for k in (*CLINICAL_FIELDS, 'full_equivalence_blocked'))):
            reasons.append('missing_complete_clinical_comparison')
        elif comparison['full_equivalence_blocked'] or any((comparison[k] for k in CLINICAL_FIELDS)):
            reasons.append('independent_clinical_difference_or_uncertainty')
    tokens = [_tokens(a), _tokens(b)]
    if not all(tokens):
        reasons.append('missing_lexical_content')
    transformed = copy.deepcopy(tokens)
    witnesses = [[], []]
    if mode == 'E':
        for index, (text, frame) in enumerate(zip((a, b), (directions[0]['source_frame'], directions[0]['candidate_frame']))):
            transformed[index], witnesses[index], failed = _recorded_tokens(text, frame)
            reasons.extend(failed)
    preserved = Counter(transformed[0]) == Counter(transformed[1])
    result['preservation'] = {'original_wording_tokens': tokens, 'checked_tokens': transformed, 'recorded_transformations': witnesses, 'equal_preserved_content': preserved, 'role': 'necessary negative preservation check; never positive bag-of-words authority', 'existing_grammar': grammar.phrase(a), 'other_existing_grammar': grammar.phrase(b)}
    if not preserved:
        reasons.append('unexplained_lexical_difference')
    result['reasons'] = sorted(set(reasons))
    result['positive'] = not result['reasons']
    if result['positive']:
        result['authority'] = 'direct_existing_bounded_role_grammar_not_input_transitivity'
        result['proof'] = {'rule': 'bidirectional_complete_role_with_preserved_lexical_content', 'mode': mode, 'role_key': directions[0]['source_frame']['key'], 'course': directions[0]['source_frame']['course'], 'source_correspondence': witnesses, 'qualification': 'derived role proof, not independent source-certified synonymy'}
    return result
