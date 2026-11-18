"""TRACE observational recorder. Never used to authorize or alter a decision.

Disabled unless a run explicitly enables it. Failures are fail-open for mapping,
but fail-closed for trace completeness. All writes are new artifacts or JSONL
appends. No network, terminology lookup, model invocation or label access.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from enum import Enum
from functools import wraps
from pathlib import Path
import dataclasses
import hashlib
import inspect
import json
import math
import os
import re
import sys
import threading

VERSION = 'trace.decision-events/2'
SUPPORTED_VERSIONS = ('trace.decision-events/1', VERSION)
# Only these graph terminals produce the state consumed by completion.replay.
# Nested completion.outcome observations must never substitute for them.
UPSTREAM_OUTCOMES = frozenset(('exact_match_outcome', 'reviewer_final_outcome',
                              'human_review_outcome', 'technical_failure_outcome'))
CURRENT = ContextVar('trace_recorder', default=None)
SPAN = ContextVar('trace_span', default=None)
STAGE = ContextVar('trace_stage', default='run')
ROUND = ContextVar('trace_round', default=None)
INVENTORY_PHASE = ContextVar('trace_inventory_phase', default=None)
INVENTORY_VERSION = 'trace.retrieval-inventories/1'
CAUSE = ContextVar('trace_cause', default=None)
DEPENDENCIES = {
    'normalization': ('run',), 'exact_matching': ('normalization',),
    'input_interpretation': ('exact_matching',), 'retrieval_dispatch': ('input_interpretation',),
    'fuzzy_matching': ('retrieval_dispatch',), 'semantic_matching': ('retrieval_dispatch',),
    'umls_matching': ('retrieval_dispatch',),
    'aggregation': ('fuzzy_matching', 'semantic_matching', 'umls_matching'),
    'reviewer_input': ('aggregation',), 'reviewer_llm': ('reviewer_input', 'evidence_reaggregation'),
    'llt_selection_gate': ('reviewer_llm',), 'reviewer_validation': ('llt_selection_gate', 'reviewer_llm'),
    'coordinator': ('reviewer_validation',), 'evidence_request_executor': ('coordinator',),
    'evidence_reaggregation': ('evidence_request_executor',),
    'exact_match_outcome': ('exact_matching',), 'reviewer_final_outcome': ('coordinator',),
    'human_review_outcome': ('coordinator',), 'technical_failure_outcome': ('evidence_request_executor',),
}
ALTERNATIVE_DEPS = {'reviewer_llm', 'reviewer_validation'}
SECRET_KEY = re.compile(r'(?:api.?key|authorization|password|secret|access.?token|refresh.?token|cookie|headers|environment|env_file)', re.I)


def utc():
    return datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')


def digest_bytes(data):
    return hashlib.sha256(data).hexdigest()


def encoded(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')


def objects(value):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from objects(child)
    elif isinstance(value, list):
        for child in value:
            yield from objects(child)


def identities(value):
    """Definitions only from full identity records, never from a guessed code."""
    fields = ('candidate_id', 'llt_code', 'llt_name', 'pt_code', 'pt_name')
    rows = {}
    for obj in objects(value):
        if all(obj.get(k) is not None for k in fields):
            row = {k: str(obj[k]) for k in fields}
            rows[json.dumps(row, sort_keys=True)] = row
    return list(rows.values())


def candidate_ids(value):
    result = set()
    for obj in objects(value):
        for key, item in obj.items():
            if key == 'candidate_id' and isinstance(item, str):
                result.add(item)
            elif key.endswith('candidate_ids') and isinstance(item, list):
                result.update(x for x in item if isinstance(x, str))
    return sorted(result)


def output_authority(record):
    """Read saved declarations; do not infer clinical acceptance from scores."""
    outcome = record.get('outcome') or {}
    gate = record.get('final_gate') or {}
    exact = (record.get('upstream_state') or {}).get('exact_result') or {}
    if record.get('status') != 'completed':
        return {'kind': 'execution_failure', 'decision_authority': outcome.get('decision_authority'), 'branch': None}
    if exact.get('found'):
        return {'kind': 'exact_reference', 'decision_authority': outcome.get('decision_authority'),
                'branch': 'exact_matching', 'provenance': exact.get('exact_match_provenance')}
    branches = []
    thresholds = None
    for obj in objects(gate.get('coverage_acceptance', {})):
        au = obj.get('authority')
        if isinstance(au, dict) and isinstance(au.get('request'), dict):
            req = au['request']
            branches.append(req.get('consistency_condition') or au.get('mode'))
            audit=obj.get('audit') or {}
            if thresholds is None and audit.get('accepted'):
                if audit.get('threshold_policy'):
                    policy=audit['threshold_policy']
                    thresholds={k:policy[k] for k in ('score_floor','margin_floor','strict_positive_margin_epsilon') if k in policy}
                    thresholds['source']='threshold_acceptance.revise: threshold_policy'
                elif (audit.get('proof_priority') or {}).get('eligible'):
                    thresholds={'score_floor':0.0,'strict_positive_margin_epsilon':1e-10,
                                'source':'proof_priority.revise: eligible P override'}
                elif au.get('mode')=='J' or req.get('consistency_condition') in ('P','T'):
                    thresholds={'score_floor':0.7,'margin_floor':0.01,
                                'source':'retained fixed checks: meaning_completion / specificity_reconciliation; no numeric override applied'}
    return {'kind': 'saved_policy_declaration', 'decision_authority': outcome.get('decision_authority'),
            'gate_authority': gate.get('decision_authority'), 'branch': next((x for x in branches if x), 'retained_gate' if gate else 'reviewer_and_validation'),
            'retained_policy_namespaces': sorted(gate.get('coverage_acceptance', {})),
            'decision_origin': outcome.get('decision_origin'),
            'applicable_completion_thresholds': thresholds,
            'threshold_limitation': None if thresholds else 'No completion override established; use recorded gate configuration and full audit. Scores are not probabilities.',
            'gate_thresholds': {k: gate[k] for k in ('minimum_score', 'minimum_margin', 'pt_minimum_score',
                'pt_minimum_margin', 'top_score', 'top_margin', 'llt_top_score', 'llt_top_margin') if k in gate},
            'applicable_policy_evidence': [{k:{f:v for f,v in obj[k].items() if f in
                ('score','margin','score_floor','margin_floor','strict_positive_margin_epsilon','authority',
                 'eligible','accepted','failures','retained_blockers','discharged','outside_score')}
                for k in ('threshold_policy','proof_priority','numeric') if isinstance(obj.get(k),dict)}
                for obj in objects(gate.get('coverage_acceptance', {}))
                if any(isinstance(obj.get(k),dict) for k in ('threshold_policy','proof_priority','numeric'))]}


class Recorder:
    def __init__(self, directory, term_id, run_id):
        self.directory = Path(directory).resolve()
        self.term_id, self.run_id = term_id, run_id
        self.path = self.directory / 'traces' / 'events' / (term_id + '.jsonl')
        self.lock = threading.RLock()
        self.sequence, self.latest, self.known, self.errors = 0, {}, {}, []
        self.upstream_outputs = []
        self.broken, self.last = False, None
        self.secrets = tuple(v for k, v in os.environ.items() if SECRET_KEY.search(k) and len(v) >= 5)
        # This constructor is used only for a newly reserved input; never append
        # reconstructed events to a completed or interrupted historical attempt.
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open('x', encoding='utf-8'):
            pass

    def clean(self, value, key=''):
        if SECRET_KEY.search(key):
            return '[REDACTED]'
        if dataclasses.is_dataclass(value):
            return self.clean(dataclasses.asdict(value))
        if hasattr(value, 'model_dump'):
            return self.clean(value.model_dump(mode='json'))
        if isinstance(value, Enum):
            return self.clean(value.value)
        if isinstance(value, dict):
            return {str(k): self.clean(v, str(k)) for k, v in value.items()}
        if isinstance(value, (tuple, list, set)):
            return [self.clean(v) for v in value]
        if isinstance(value, Path):
            value = str(value)
        if isinstance(value, str):
            for secret in self.secrets:
                value = value.replace(secret, '[REDACTED]')
            value = re.sub(r'(?i)(api[-_]?key|token|password|authorization)(\s*[=:]\s*)[^\s&"\']+', r'\1\2[REDACTED]', value)
            value = re.sub(r'(https?://)[^/\s:@]+:[^/\s@]+@', r'\1[REDACTED]@', value)
            return value
        if isinstance(value, float) and not math.isfinite(value):
            return {'unserializable_numeric': str(value)}
        if value is None or isinstance(value, (bool, int, float)):
            return value
        return {'not_snapshotted_type': type(value).__name__}

    def fail(self, error):
        self.broken = True
        detail = {'utc_timestamp': utc(), 'term_id': self.term_id, 'error_type': type(error).__name__}
        self.errors.append(detail)
        try:
            with (self.directory / 'traces' / 'recorder_errors.jsonl').open('a', encoding='utf-8') as f:
                f.write(encoded(detail).decode() + '\n')
        except Exception:
            pass
        print(f'TRACE recording failure for {self.term_id}: {type(error).__name__}; mapping behavior unchanged', file=sys.stderr)

    def reference(self, path, role):
        path = Path(path).resolve()
        rel = path.relative_to(self.directory).as_posix()
        data = path.read_bytes()
        return {'path': rel, 'sha256': digest_bytes(data), 'bytes': len(data), 'role': role}

    def snapshot(self, value, role):
        value = self.clean(value)
        data = encoded(value)
        path = self.directory / 'traces' / 'evidence' / (digest_bytes(data) + '.json')
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with path.open('xb') as f:
                f.write(data)
                f.flush()
                os.fsync(f.fileno())
        except FileExistsError:
            if path.read_bytes() != data:
                raise ValueError('Snapshot hash collision or changed evidence')
        return self.reference(path, role), value

    def emit(self, event_type, payload=None, *, status='completed', stage=None, parents=None,
             evidence_kind='computed', artifacts=(), review_round=None, snapshot=None):
        with self.lock:
            if self.broken:
                return None
            try:
                refs = [self.reference(path, role) for path, role in artifacts]
                value = self.clean(payload or {})
                definitions_from = value
                if snapshot is not None:
                    ref, snap = self.snapshot(snapshot, 'snapshot')
                    refs.append(ref)
                    definitions_from = [value, snap]
                definitions, views = [], []
                for identity in identities(definitions_from):
                    cid=identity['candidate_id']
                    previous=self.known.get(cid)
                    if evidence_kind=='model_reported':
                        views.append({'candidate_id':cid,'observed_identity':identity,
                                      'scope':'model_claim_not_identity_authority'})
                    elif previous is None:
                        self.known[cid]=identity
                        definitions.append(identity)
                    elif previous!=identity:
                        # Ranking sometimes compares a source-normalized label.
                        # Never silently rename an already observed identity.
                        views.append({'candidate_id':cid,'observed_identity':identity,
                                      'scope':'comparison_or_unverified_identity_variant'})
                candidates = [{'candidate_id': c, 'resolution': 'observed_identity' if c in self.known else 'unresolved_reported_reference'}
                              for c in candidate_ids(definitions_from)]
                self.sequence += 1
                eid = f'{self.term_id}:{self.sequence:06d}'
                dependencies = parents if parents is not None else ([CAUSE.get() or SPAN.get()] if (CAUSE.get() or SPAN.get()) else ([self.last] if self.last else []))
                event = {'schema_version': VERSION, 'run_id': self.run_id, 'term_id': self.term_id,
                    'event_id': eid, 'sequence': self.sequence, 'utc_timestamp': utc(),
                    'stage': stage or STAGE.get(), 'event_type': event_type, 'execution_status': status,
                    'review_round': review_round if review_round is not None else ROUND.get(),
                    'component_ids': sorted({str(o['component_id']) for o in objects(definitions_from) if o.get('component_id')}),
                    'parent_event_ids': list(dict.fromkeys(dependencies)), 'evidence_kind': evidence_kind,
                    'candidate_definitions': definitions, 'candidate_views': views, 'candidate_references': candidates,
                    'payload': value, 'artifact_refs': refs}
                with self.path.open('a', encoding='utf-8', newline='\n') as f:
                    f.write(encoded(event).decode() + '\n')
                    f.flush()
                    os.fsync(f.fileno())
                self.last = eid
                CAUSE.set(eid)
                return eid
            except Exception as error:
                self.fail(error)
                return None

    def start(self, row):
        eid = self.emit('term_started', row, stage='run', parents=[], artifacts=[(self.directory/'contract.json', 'run_contract')])
        self.latest['run'] = (self.sequence, eid)

    def completion_handoff(self, state):
        """Bind by observed producer output, not wall time or task-local CAUSE.

        LangGraph returns merged state outside its worker context. Compare the
        producer's entire output projection to that state; do not infer a
        producer from the mapping status or from the last event in the file.
        """
        state = self.clean(state)
        with self.lock:
            matches = [o for o in self.upstream_outputs if o['output'] and
                       all(k in state and state[k] == v for k, v in o['output'].items())]
            status = ('observed' if len(matches) == 1 else 'ambiguous' if matches else
                      'state_mismatch' if self.upstream_outputs else 'missing')
            producer = matches[0] if status == 'observed' else None
            handoff = {'status': status, 'producer_event_id': producer['event_id'] if producer else None,
                       'producer_stage': producer['stage'] if producer else None,
                       'matched_output_keys': sorted(producer['output']) if producer else [],
                       'consumed_state_sha256': digest_bytes(encoded(state)),
                       'workflow_outcome_sha256': digest_bytes(encoded(state.get('workflow_outcome'))),
                       'available_producer_event_ids': [o['event_id'] for o in self.upstream_outputs],
                       'provenance': 'observed completion input snapshot; no upstream reconstruction'}
            if producer:
                return [producer['event_id']], handoff
            gap = self.emit('observation_gap', {'upstream_handoff': handoff,
                'reason': 'No uniquely matching observed upstream terminal; supplied state is available, execution history is not.'},
                stage='completion.handoff', parents=[self.latest['run'][1]], evidence_kind='observed_state')
            return [gap] if gap else [], handoff

    def finish(self, record_path):
        try:
            record = json.loads(Path(record_path).read_text(encoding='utf-8'))
            self.emit('term_final', {'status': record['status'], 'prediction': record['prediction'],
                'outcome_status': (record.get('outcome') or {}).get('status'), 'authority': output_authority(record),
                'policy_replay_failure': record.get('policy_replay_failure')}, stage='final',
                status='completed' if record['status']=='completed' else 'failed',
                artifacts=[(record_path, 'saved_record')])
            # Validator derives completeness from disk rather than trusting memory.
            from validate_decision_trace import validate_term
            validation = validate_term(self.directory, self.term_id, require_summary=False)
            summary = {'schema_version': VERSION, 'run_id': self.run_id, 'term_id': self.term_id,
                'outcome': {k:self.clean((record.get('outcome') or {}).get(k)) for k in ('status','decision_origin','reviewer_decision_type','review_round')}, 'prediction': record['prediction'],
                'saved_record': self.reference(record_path,'saved_record'),
                'acceptance_authority': output_authority(record), 'trace_completeness': validation['trace_completeness'],
                'validation_errors': validation['errors'], 'recorder_errors': self.errors,
                'final_event_id': self.last, 'event_count': self.sequence}
            path=self.directory/'traces'/'summaries'/(self.term_id+'.json')
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open('x', encoding='utf-8') as f:
                json.dump(summary, f, ensure_ascii=False, indent=2, allow_nan=False)
        except Exception as error:
            self.fail(error)


def emit(event_type, payload=None, **kwargs):
    recorder = CURRENT.get()
    if recorder is not None:
        return recorder.emit(event_type, payload, **kwargs)
    return None


@contextmanager
def term_recording(directory, row, enabled):
    recorder = None
    if enabled:
        try:
            contract = Path(directory)/'contract.json'
            recorder = Recorder(directory, row['id'], 'TRACE-' + digest_bytes(contract.read_bytes())[:24])
            recorder.start(row)
        except Exception as error:
            print(f'TRACE could not start recording {row["id"]}: {type(error).__name__}; mapping unaffected', file=sys.stderr)
    token = CURRENT.set(recorder)
    cause_token = CAUSE.set(recorder.last if recorder else None)
    try:
        yield recorder
    finally:
        CURRENT.reset(token)
        CAUSE.reset(cause_token)


def observe(stage, function, data, *, graph_node=False):
    """Invoke exactly once, return the original result object, re-raise failures."""
    recorder = CURRENT.get()
    if recorder is None:
        return function()
    parents = None
    if graph_node:
        deps = [recorder.latest[k] for k in DEPENDENCIES.get(stage, ()) if k in recorder.latest]
        if stage in ALTERNATIVE_DEPS and deps:
            deps = [max(deps)]
        parents = [v[1] for v in deps if v[1]]
    payload = {}
    if stage == 'completion.replay':
        try:
            parents, handoff = recorder.completion_handoff(data)
            payload = {'upstream_handoff': handoff}
        except Exception as error:
            recorder.fail(error)
    round_value = data.get('review_round') if isinstance(data, dict) else None
    inp = data.get('reviewer_input') if isinstance(data, dict) else None
    if inp is not None:
        round_value = getattr(inp, 'review_round', None) or (inp.get('review_round') if isinstance(inp, dict) else None)
    first = recorder.emit('stage_started', payload, status='started', stage=stage, parents=parents,
        review_round=round_value, evidence_kind='observed_state', snapshot=data)
    tokens = (SPAN.set(first), STAGE.set(stage), ROUND.set(round_value or ROUND.get()))
    phase_token = INVENTORY_PHASE.set('completion' if stage == 'completion.replay' else
        'upstream' if stage == 'llt_selection_gate' else INVENTORY_PHASE.get())
    try:
        result = function()
    except BaseException as error:
        recorder.emit('stage_failed', {'error_type': type(error).__name__, 'message': str(error)},
                      stage=stage, status='failed', parents=[x for x in (first, CAUSE.get()) if x])
        raise
    else:
        last = recorder.emit('stage_completed', stage=stage, parents=[x for x in (first, CAUSE.get()) if x],
                             evidence_kind='observed_state', snapshot=result)
        if graph_node:
            with recorder.lock:
                recorder.latest[stage] = (recorder.sequence, last)
                if stage in UPSTREAM_OUTCOMES and last:
                    try:
                        recorder.upstream_outputs.append({'event_id': last, 'stage': stage,
                                                         'output': recorder.clean(result)})
                    except Exception as error:
                        recorder.fail(error)
        if stage == 'exact_matching' and getattr(result.get('exact_result'), 'found', False):
            for skipped in ('input_interpretation', 'retrieval_dispatch', 'fuzzy_matching', 'semantic_matching',
                            'umls_matching', 'aggregation', 'reviewer_input', 'reviewer_llm',
                            'llt_selection_gate', 'reviewer_validation', 'coordinator', 'completion'):
                recorder.emit('stage_skipped', {'reason': 'exact_match_bypass'}, stage=skipped,
                              status='skipped', parents=[last] if last else [])
        return result
    finally:
        INVENTORY_PHASE.reset(phase_token)
        SPAN.reset(tokens[0]); STAGE.reset(tokens[1]); ROUND.reset(tokens[2])


def node(stage, function):
    # A simple signature is intentional: LangGraph receives the unchanged state.
    def wrapped(state):
        return observe(stage, lambda: function(state), state, graph_node=True)
    return wrapped


def computation(stage):
    """Observe existing pure comparison calls; never recompute for logging."""
    def decorate(function):
        signature = inspect.signature(function)
        @wraps(function)
        def wrapped(*args, **kwargs):
            if CURRENT.get() is None:
                return function(*args, **kwargs)
            try:
                data = dict(signature.bind(*args, **kwargs).arguments)
                data = {k:v for k,v in data.items() if k not in ('ref', 'reference', 'lex', 'reranker')}
            except Exception:
                data = {'arguments_unavailable': True}
            return observe(stage, lambda: function(*args, **kwargs), data)
        return wrapped
    return decorate


def model_artifact(path, role, *, status='completed'):
    emit('model_' + role, {'artifact_role': role}, artifacts=[(path, role)], status=status,
         evidence_kind='model_reported' if role == 'response' else 'observed_execution')


def reference_inventory(candidates, *, input_term, pt_code, return_limit):
    """Observe the already-computed full reference ranking before slicing.

    Nothing is recomputed or returned to the decision path. Context distinguishes
    upstream expansion from completion, including discarded completion attempts.
    """
    if CURRENT.get() is not None:
        emit('reference_candidates_available',
             {'inventory_version': INVENTORY_VERSION, 'phase': INVENTORY_PHASE.get(),
              'input_term': input_term, 'pt_code': pt_code, 'return_limit': return_limit,
              'scope': 'actual_scored_reference_pool_before_return_limit; rank_within_PT_only'},
             snapshot={'candidates': candidates}, evidence_kind='observed_state')


def reported_json(raw, role, attempt):
    if CURRENT.get() is None:
        return
    try:
        value=json.loads(raw)
    except (ValueError, TypeError):
        emit('model_proposal_unparsed', {'role':role,'attempt':attempt,
             'missing':'structured proposal unavailable; original raw response remains in calls/'}, evidence_kind='model_reported')
    else:
        emit('original_model_proposal', {'role':role,'attempt':attempt},
             snapshot=value, evidence_kind='model_reported')


def prepare_run(directory, enabled):
    """Sidecar only. Never edit the immutable decision run contract."""
    path=Path(directory)/'traces'/'recorder.json'
    if path.exists() and not enabled:
        raise ValueError('This run enabled recording; resume with --record-decisions')
    if not enabled:
        return
    expected={'schema_version': VERSION, 'run_id': 'TRACE-'+digest_bytes((Path(directory)/'contract.json').read_bytes())[:24],
              'retrieval_inventory_version': INVENTORY_VERSION,
              'policy': 'observational; skipped inputs are never replayed or backfilled'}
    if path.exists():
        if json.loads(path.read_text(encoding='utf-8')) != expected:
            raise ValueError('Changed recorder binding')
    else:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open('x',encoding='utf-8') as f:
                json.dump(expected,f,indent=2)
        except OSError as error:
            print(f'TRACE recorder setup failed: {type(error).__name__}; mapping unaffected; traces may be missing',file=sys.stderr)


def skipped_attempt(directory, term_id):
    """Separate append-only inventory; do not append fictional stages to a term."""
    try:
        base=Path(directory)
        detail={'term_id':term_id,'utc_timestamp':utc(),'event_type':'resume_preserved_attempt',
            'record_exists':(base/'records'/(term_id+'.json')).exists(),
            'trace_exists':(base/'traces'/'events'/(term_id+'.jsonl')).exists(),
            'action':'not_reexecuted_or_backfilled'}
        with (base/'traces'/'resume.jsonl').open('a',encoding='utf-8') as f:
            f.write(encoded(detail).decode()+'\n')
    except Exception as error:
        print(f'TRACE resume inventory error: {type(error).__name__}',file=sys.stderr)
