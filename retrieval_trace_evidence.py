"""Read observed retrieval/completion history. No execution or reconstruction."""
import json
from decision_recorder import INVENTORY_VERSION, UPSTREAM_OUTCOMES
from validate_decision_trace import read_events, validate_term, EvidenceError


def load_history(reader, term_id):
    result = {'initial': {}, 'executions': [], 'gates': [], 'inventories': [],
              'audit': {'available': False, 'complete': False, 'findings': []}}
    relative = 'traces/events/' + term_id + '.jsonl'
    if reader.read_bytes(relative) is None:
        return result
    audit = result['audit']
    audit['available'] = True
    metadata = reader.read('traces/recorder.json') or {}
    reader.read('traces/summaries/' + term_id + '.json')
    events, errors = read_events(reader.directory, term_id)
    validation = validate_term(reader.directory, term_id)
    audit['findings'] = validation['errors']
    audit['inventory_version'] = metadata.get('retrieval_inventory_version')
    audit['event_file'] = relative
    audit['trace_completeness'] = validation['trace_completeness']
    # Use only the validator's supported, structurally valid prefix.
    if errors:
        audit['findings'] = [*audit['findings'], *errors]
    seen_terminal = False
    completion_finished = False
    failed = False

    def snapshot(event):
        refs = [r for r in event['artifact_refs'] if r['role'] == 'snapshot']
        if len(refs) != 1:
            raise EvidenceError('missing_snapshot', event['event_id'])
        ref = refs[0]
        data = reader.read_bytes(ref['path'])
        if data is None:
            raise EvidenceError('missing_snapshot', ref['path'])
        import hashlib
        if len(data) != ref['bytes'] or hashlib.sha256(data).hexdigest() != ref['sha256']:
            raise EvidenceError('corrupt_snapshot', ref['path'])
        return json.loads(data), {'artifact': ref['path'], 'event_id': event['event_id'],
                                  'event_file': relative, 'pointer': ''}

    for event in events:
        stage, kind = event['stage'], event['event_type']
        failed |= kind == 'stage_failed'
        seen_terminal |= stage in UPSTREAM_OUTCOMES and kind == 'stage_completed'
        completion_finished |= stage == 'completion.replay' and kind == 'stage_completed'
        try:
            if kind == 'stage_completed' and stage in ('exact_matching', 'fuzzy_matching', 'semantic_matching', 'umls_matching'):
                value, ref = snapshot(event)
                key = 'exact_result' if stage == 'exact_matching' else stage.replace('_matching', '_result')
                if key in value:
                    result['initial'][key] = {'value': value[key], 'evidence': {**ref, 'pointer': '/' + key}}
            elif kind == 'stage_completed' and stage == 'evidence_request_executor':
                value, ref = snapshot(event)
                for execution in (value.get('evidence_execution_batch') or {}).get('executions', []):
                    result['executions'].append({'value': execution, 'evidence': ref})
            elif kind == 'stage_completed' and stage in ('llt_selection_gate', 'completion.retained', 'completion.J', 'completion.P', 'completion.T'):
                value, ref = snapshot(event)
                gate = value.get('llt_selection_gate_result') if isinstance(value, dict) else value[1]
                result['gates'].append({'value': gate, 'phase': 'upstream' if stage == 'llt_selection_gate' else 'completion',
                                       'evidence': {**ref, 'pointer': '/llt_selection_gate_result' if isinstance(value, dict) else '/1'}})
            elif kind == 'stage_started' and stage in ('completion.J', 'completion.P', 'completion.T'):
                value, ref = snapshot(event)
                case = value.get('request', {}).get('case', {})
                if isinstance(case.get('candidates'), list):
                    result['inventories'].append({'candidates': case['candidates'], 'phase': 'completion',
                                                 'evidence': {**ref, 'pointer': '/request/case/candidates'}})
            elif kind == 'reference_candidates_available':
                value, ref = snapshot(event)
                info = event['payload']
                if info.get('inventory_version') != INVENTORY_VERSION or info.get('phase') not in ('upstream', 'completion'):
                    continue
                result['inventories'].append({'candidates': value['candidates'], 'phase': info['phase'],
                    'evidence': {**ref, 'pointer': '/candidates', 'ranking_scope': info['scope'],
                                 'input_term': info['input_term'], 'pt_code': info['pt_code'], 'return_limit': info['return_limit']}})
        except (EvidenceError, KeyError, IndexError, TypeError, ValueError) as exc:
            audit['findings'].append({'event_id': event['event_id'], 'kind': 'retrieval_snapshot_unavailable', 'detail': str(exc)})
    audit['complete'] = bool(metadata.get('retrieval_inventory_version') == INVENTORY_VERSION
        and validation['trace_completeness'] == 'complete' and not audit['findings']
        and seen_terminal and completion_finished and not failed)
    if not audit['complete']:
        audit['limitation'] = 'Only observed valid inventories used; no assertion of exhaustive history.'
    return result
