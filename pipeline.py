"""Fresh term processing with immutable per-term records and frozen threshold T."""
import copy
import json
import os
import time
import traceback
import decision_recorder as trace
from contextlib import contextmanager
from pathlib import Path
from runtime_resources import ROOT, bootstrap, configuration, implementation, resource, save_new, load, sha
bootstrap()
SETTINGS = {'model': 'qwen3:8b', 'temperature': 0.0, 'seed': 42, 'think': False, 'num_ctx': 16384, 'max_validation_attempts': 3, 'max_review_rounds': 3, 'fuzzy_top_k': 5, 'semantic_top_k': 5}

def model_host():
    return os.environ.get('OLLAMA_REVIEWER_HOST', os.environ.get('OLLAMA_HOST', 'http://127.0.0.1:11434')).rstrip('/')

def jsonable(value):
    if hasattr(value, 'model_dump'):
        return value.model_dump(mode='json')
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value

def prediction(outcome):
    p = jsonable(outcome) or {}
    selected = p.get('selected_concepts') or []
    pts = selected + (p.get('resolved_pt_concepts') or [])
    return {'pt': sorted({str(c['pt_code']) for c in pts if c.get('pt_code')}), 'llt': sorted({str(c['llt_code']) for c in selected if c.get('llt_code')})}

@contextmanager
def workflow():
    from exact_matching import create_exact_matching_node
    from input_interpretation import OllamaInputInterpreter
    from reviewer_llm import OllamaReviewer
    from retriever_adapters import create_real_retriever_adapters
    from llt_reranking import PTConditionedLLTReranker
    from acceptance_gate import create_development_gate_node
    import graph
    settings = {k: SETTINGS[k] for k in ('model', 'temperature', 'seed', 'think', 'num_ctx', 'max_validation_attempts')}
    settings['host'] = model_host()
    reviewer = OllamaReviewer(**settings, retry_diagnostics=True)
    interpreter = OllamaInputInterpreter(**settings)
    reranker = PTConditionedLLTReranker(resource('meddra_exact_reference.csv'))
    factory = graph.create_llt_selection_gate_node
    graph.create_llt_selection_gate_node = create_development_gate_node
    try:
        with create_real_retriever_adapters(meddra_reference_path=resource('meddra_exact_reference.csv'), vector_database_path=resource('meddra_vector_db'), vector_config_path=resource('meddra_vector_db_config.json')) as a:
            compiled = graph.create_standardization_graph(exact_matching_node=create_exact_matching_node(resource('meddra_exact_reference.csv')), input_interpretation_node=interpreter, reviewer_node=reviewer, fuzzy_retriever=a.fuzzy, semantic_retriever=a.semantic, umls_retriever=a.umls, llt_reranker=reranker, fuzzy_top_k=5, semantic_top_k=5, max_review_rounds=3)
            yield (compiled, reranker)
    finally:
        graph.create_llt_selection_gate_node = factory

def run(inputs, directory, *, resume=False, env_file=None, dataset_binding=None, record_decisions=False):
    """Inputs contain only id/text/frequency. No benchmark labels are accepted."""
    from schemas import TermInput
    from acceptance_policy import replay
    import ollama
    rows = []
    for item in inputs:
        if set(item) - {'id', 'text', 'frequency'}:
            raise ValueError('Runtime input has unsupported fields')
        if not item['text'].strip() or not item['id']:
            raise ValueError('Empty input/identity')
        if not all((ch.isalnum() or ch in '_-' for ch in item['id'])):
            raise ValueError('Unsafe input ID')
        rows.append({'id': item['id'], 'text': item['text'], 'frequency': int(item.get('frequency', 1))})
    if not rows or len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Empty or duplicate input cohort')
    directory = Path(directory).resolve()
    if directory.exists() and (not resume):
        raise FileExistsError('Use a fresh run directory or --resume')
    if resume and (not (directory / 'contract.json').exists()):
        raise ValueError('Missing resume contract')
    if env_file:
        from dotenv import load_dotenv
        load_dotenv(env_file, override=False)
    models = ollama.Client(host=model_host()).list().model_dump(mode='json')['models']
    matches = [m for m in models if m.get('model', m.get('name')) == SETTINGS['model']]
    if len(matches) != 1:
        raise ValueError('Install the configured qwen3:8b model first; no automatic download')
    if matches[0]['digest'] != load(ROOT / 'model.lock.json')['digest']:
        raise ValueError('Installed model digest differs from the frozen release model')
    vector = resource('meddra_vector_db')
    vector_hashes = {str(p.relative_to(vector)): sha(p) for p in sorted(vector.rglob('*')) if p.is_file() and p.name != '.lock'}
    if not vector_hashes:
        raise ValueError('Existing populated vector store required')
    extra = {name: sha(resource(name)) for name in ('meddra_vector_db_config.json', 'umls_synonyms.csv') if resource(name).is_file()}
    contract = {'version': 'ASP-MT-1.0.0', 'inputs': rows, 'settings': SETTINGS, 'policy': 'threshold_T', 'implementation': implementation(), 'model_digest': matches[0]['digest'], 'model_host': model_host(), 'vector_hashes': vector_hashes, 'retrieval_resources': extra, 'max_calls': 12 * len(rows), 'dataset_binding': dataset_binding}
    if resume:
        if load(directory / 'contract.json') != contract:
            raise ValueError('Changed inputs, model, source, policy or resources')
    else:
        directory.mkdir(parents=True, exist_ok=False)
        save_new(directory / 'contract.json', contract)
        save_new(directory / 'fixed_policy.json', {'calibration': {'threshold': {'chosen': {'cutoff': 0.0}, 'assessment_ids': [r['id'] for r in rows], 'calibration_ids': []}}})
    for name in ('records', 'reservations', 'calls'):
        (directory / name).mkdir(exist_ok=True)
    trace.prepare_run(directory, record_decisions)
    original_chat = ollama.Client.chat
    calls = len(list((directory / 'calls').glob('*_request.json')))
    active = None

    def logged(client, *args, **kwargs):
        nonlocal calls
        if calls >= contract['max_calls']:
            raise RuntimeError('Run model-call budget exhausted')
        calls += 1
        stem = directory / 'calls' / f'{calls:06d}'
        save_new(str(stem) + '_request.json', {'id': active, 'args': jsonable(args), 'kwargs': jsonable(kwargs)})
        request_event = trace.emit('model_request', {'call_number': calls}, status='started',
                                   artifacts=[(str(stem) + '_request.json', 'request')], evidence_kind='observed_execution')
        start = time.perf_counter()
        try:
            response = original_chat(client, *args, **kwargs)
            save_new(str(stem) + '_response.json', {'seconds': time.perf_counter() - start, 'response': jsonable(response)})
            trace.emit('model_response', {'call_number': calls}, parents=[request_event] if request_event else None,
                       artifacts=[(str(stem) + '_response.json', 'response')], evidence_kind='model_reported')
            return response
        except Exception as e:
            save_new(str(stem) + '_exception.json', {'type': type(e).__name__, 'message': str(e), 'seconds': time.perf_counter() - start})
            trace.emit('model_exception', {'call_number': calls, 'error_type': type(e).__name__}, status='failed',
                       parents=[request_event] if request_event else None, artifacts=[(str(stem) + '_exception.json', 'exception')])
            raise
    ollama.Client.chat = logged
    try:
        with workflow() as (engine, reranker):
            for n, row in enumerate(rows, 1):
                active = row['id']
                out = directory / 'records' / (active + '.json')
                reservation = directory / 'reservations' / (active + '.json')
                if out.exists() or reservation.exists():
                    if record_decisions:
                        trace.skipped_attempt(directory, active)
                    print(f'{n}/{len(rows)} {active}: preserved existing attempt', flush=True)
                    continue
                save_new(reservation, row)
                start = time.perf_counter()
                record = {**row, 'status': 'error', 'prediction': {'pt': [], 'llt': []}}
                with trace.term_recording(directory, row, record_decisions) as recorder:
                    try:
                        state = engine.invoke({'term_id': active, 'term_input': TermInput(term=row['text'], frequency=row['frequency'])}, config={'recursion_limit': 100})
                        record['upstream_state'] = jsonable(state)
                        if not state.get('workflow_outcome'):
                            raise RuntimeError('Pipeline produced no final outcome')
                        try:
                            final = trace.observe('completion.replay', lambda: replay(state, directory / 'fixed_policy.json', reranker), state)
                        except Exception as e:
                            final = state
                            record['policy_replay_failure'] = {'type': type(e).__name__, 'message': str(e), 'traceback': traceback.format_exc()}
                        outcome = jsonable(final['workflow_outcome'])
                        record.update(status='completed', prediction=prediction(outcome), outcome=outcome, final_decision=jsonable(final.get('reviewer_decision')), final_gate=jsonable(final.get('llt_selection_gate_result')))
                        if outcome.get('status') == 'technical_failure':
                            record['status'] = 'error'
                    except Exception as e:
                        record['exception'] = {'type': type(e).__name__, 'message': str(e), 'traceback': traceback.format_exc()}
                        trace.emit('execution_failed', {'error_type': type(e).__name__, 'message': str(e)}, status='failed')
                    record['seconds'] = time.perf_counter() - start
                    save_new(out, record)
                    if recorder is not None:
                        recorder.finish(out)
                print(f"{n}/{len(rows)} {active}: {record['status']}; PT {record['prediction']['pt']}; LLT {record['prediction']['llt']}", flush=True)
    finally:
        ollama.Client.chat = original_chat
    failures = 0
    for r in rows:
        f = directory / 'records' / (r['id'] + '.json')
        if not f.exists():
            failures += 1
            continue
        saved = load(f)
        failures += bool(saved['status'] != 'completed' or saved.get('policy_replay_failure'))
    print(f'Total terms: {len(rows)} | Failed/unexecuted: {failures} | Model attempts: {calls}')
    return int(bool(failures))
