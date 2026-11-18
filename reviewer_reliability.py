"""Label-free, opt-in reviewer contract repair; never supplies mapping support."""
from runtime_resources import resource as asp_resource
import json
from reviewer_validation import ReviewerDecisionValidationError
VERSION = '1.9.reliability.1'

class ConflictingAssessmentDuplicates(ReviewerDecisionValidationError):
    pass

def consolidate_review_assessments(inp, raw):
    """Only consolidate consistent duplicates in an explicitly unresolved response.

    Unknown identities/fields, malformed assessments and conflicting duplicates
    are not discarded. Original payload and each merged assessment are audited.
    Matched responses continue through their existing repair path unchanged.
    """
    try:
        payload = json.loads(raw)
    except (ValueError, TypeError):
        return (raw, [])
    if not isinstance(payload, dict):
        return (raw, [])
    rows = payload.get('candidate_assessments')
    if not isinstance(rows, list):
        return (raw, [])
    supplied = {c.candidate_id for c in inp.aggregation_result.candidates}
    grouped = {}
    for a in rows:
        if not isinstance(a, dict) or a.get('candidate_id') not in supplied:
            return (raw, [])
        grouped.setdefault(a['candidate_id'], []).append(a)
    for identifier, assessments in grouped.items():
        signatures = [{k: v for k, v in a.items() if k != 'rationale'} for a in assessments]
        if any((s != signatures[0] for s in signatures[1:])):
            raise ConflictingAssessmentDuplicates('Conflicting duplicate assessments remain unresolved: ' + identifier)
    if payload.get('decision_type') != 'human_review':
        return (raw, [])
    if payload.get('selected_candidate_ids') or payload.get('component_matches') or payload.get('resolved_pt_code'):
        return (raw, [])
    edits = []
    out = []
    for identifier, assessments in grouped.items():
        first = assessments[0]
        if len(assessments) > 1:
            signatures = [{k: v for k, v in a.items() if k != 'rationale'} for a in assessments]
            if any((s != signatures[0] for s in signatures[1:])):
                raise ConflictingAssessmentDuplicates('Conflicting duplicate assessments remain unresolved: ' + identifier)
            if any((not isinstance(a.get('rationale'), str) or not a['rationale'].strip() for a in assessments)):
                return (raw, [])
            first = {**first, 'rationale': '\n'.join(dict.fromkeys((a['rationale'] for a in assessments))), 'assessment_authority': 'deterministic_duplicate_consolidation'}
            edits.append({'candidate_id': identifier, 'original_assessments': assessments, 'consolidated': first, 'authority': 'consistent_verdict_and_fields_only; all rationales retained'})
        out.append(first)
    if not edits:
        return (raw, [])
    return (json.dumps({**payload, 'candidate_assessments': out}), edits)

def terminal_clinical_sequence(rejections, attempt_count):
    """Two consecutive final structurally valid clinical rejections, not any two errors."""
    return attempt_count >= 2 and [r['attempt'] for r in rejections[-2:]] == [attempt_count - 1, attempt_count]
