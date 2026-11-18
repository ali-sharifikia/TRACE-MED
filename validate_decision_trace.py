"""Offline TRACE event validator; never executes or re-scores the pipeline."""
import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from decision_recorder import (VERSION, SUPPORTED_VERSIONS, UPSTREAM_OUTCOMES,
    DEPENDENCIES, ALTERNATIVE_DEPS, output_authority, encoded, digest_bytes, identities)


class Identity(BaseModel):
    model_config=ConfigDict(extra='forbid',strict=True)
    candidate_id:str
    llt_code:str
    llt_name:str
    pt_code:str
    pt_name:str


class CandidateReference(BaseModel):
    model_config=ConfigDict(extra='forbid',strict=True)
    candidate_id:str
    resolution:Literal['observed_identity','unresolved_reported_reference']


class CandidateView(BaseModel):
    model_config=ConfigDict(extra='forbid',strict=True)
    candidate_id:str
    observed_identity:Identity
    scope:Literal['model_claim_not_identity_authority','comparison_or_unverified_identity_variant']


class Artifact(BaseModel):
    model_config=ConfigDict(extra='forbid',strict=True)
    path:str
    sha256:str=Field(pattern='^[0-9a-f]{64}$')
    bytes:int=Field(ge=0)
    role:str


def saved_record_shape(record):
    """Minimal structural safety before inspecting an untrusted JSON record."""
    if not isinstance(record,dict) or not isinstance(record.get('status'),str):return False
    pred=record.get('prediction')
    if not isinstance(pred,dict) or any(not isinstance(pred.get(k),list) or
        any(not isinstance(v,str) for v in pred[k]) for k in ('pt','llt')):return False
    if any(record.get(k) is not None and not isinstance(record[k],dict) for k in ('outcome','final_gate','upstream_state')):return False
    out=record.get('outcome') or {}
    return all(isinstance(out.get(k,[]),list) and all(isinstance(c,dict) for c in out.get(k,[]))
               for k in ('selected_concepts','resolved_pt_concepts'))


class TraceEvent(BaseModel):
    model_config=ConfigDict(extra='forbid',strict=True)
    schema_version:Literal['trace.decision-events/1','trace.decision-events/2']
    run_id:str
    term_id:str
    event_id:str
    sequence:int=Field(ge=1)
    utc_timestamp:str
    stage:str
    event_type:str
    execution_status:Literal['started','completed','failed','skipped']
    review_round:int|None
    component_ids:list[str]
    parent_event_ids:list[str]
    evidence_kind:Literal['computed','observed_state','observed_execution','model_reported']
    candidate_definitions:list[Identity]
    candidate_views:list[CandidateView]=Field(default_factory=list)
    candidate_references:list[CandidateReference]
    payload:dict[str,Any]
    artifact_refs:list[Artifact]


class EvidenceError(ValueError):
    def __init__(self, kind, path):
        self.kind, self.path = kind, path
        super().__init__(kind + ':' + str(path))


def resolve_artifact(root, ref):
    """Shared read-only integrity resolver for validator and renderer."""
    root = Path(root).resolve()
    try:
        target = (root / ref['path']).resolve()
        target.relative_to(root)
    except (KeyError, ValueError, OSError):
        raise EvidenceError('unsafe_artifact', ref.get('path'))
    try:
        data = target.read_bytes()
    except FileNotFoundError:
        raise EvidenceError('missing_artifact', ref['path'])
    except OSError:
        raise EvidenceError('unreadable_artifact', ref['path'])
    if hashlib.sha256(data).hexdigest() != ref['sha256'] or len(data) != ref['bytes']:
        raise EvidenceError('corrupt_artifact', ref['path'])
    return data


def read_events(directory, term_id):
    """Read the valid prefix, preserving explicit unsupported/truncated findings."""
    events, errors = [], []
    path = Path(directory) / 'traces' / 'events' / (term_id + '.jsonl')
    if not path.exists():
        return events, ['missing_event_file']
    try:
        with path.open(encoding='utf-8') as f:
            for n, line in enumerate(f, 1):
                try:
                    event = json.loads(line)
                except (ValueError, TypeError):
                    errors.append(f'line_{n}:invalid_or_interrupted_json'); break
                if isinstance(event, dict) and event.get('schema_version') not in SUPPORTED_VERSIONS:
                    errors.append(f'line_{n}:unsupported_schema_version'); break
                try:
                    TraceEvent.model_validate(event)
                except ValidationError:
                    errors.append(f'line_{n}:invalid_event_schema'); break
                events.append(event)
    except (OSError, UnicodeError) as e:
        errors.append('unreadable_events:' + type(e).__name__)
    return events, errors


def handoff_errors(event, seen, snapshots):
    """Require the recorded producer and its actual output to feed completion."""
    inp = snapshots.get(event['event_id'])
    if not isinstance(inp, dict):
        return ['completion_input_unavailable']
    producers = [e for e in seen.values() if e['stage'] in UPSTREAM_OUTCOMES
                 and e['event_type'] == 'stage_completed']
    matching = []
    for p in producers:
        out = snapshots.get(p['event_id'])
        if isinstance(out, dict) and out and all(k in inp and inp[k] == v for k,v in out.items()):
            matching.append(p)
    h = event['payload'].get('upstream_handoff')
    if event['schema_version'].endswith('/1') and not h:
        return ['legacy_completion_dependency_gap'] if not (
            len(matching) == 1 and matching[0]['event_id'] in event['parent_event_ids']) else []
    if not isinstance(h, dict):
        return ['missing_completion_handoff']
    errors = []
    if h.get('consumed_state_sha256') != digest_bytes(encoded(inp)) or h.get('workflow_outcome_sha256') != digest_bytes(encoded(inp.get('workflow_outcome'))):
        errors.append('completion_state_binding_mismatch')
    if h.get('available_producer_event_ids') != [p['event_id'] for p in producers]:
        errors.append('completion_producer_inventory_mismatch')
    if h.get('status') != 'observed':
        errors.append('completion_upstream_observation_gap:' + str(h.get('status')))
        if len(matching) == 1:
            errors.append('completion_observed_producer_not_linked')
        gaps = [seen[p] for p in event['parent_event_ids'] if p in seen and seen[p]['event_type'] == 'observation_gap']
        if not gaps or any(g['payload'].get('upstream_handoff') != h for g in gaps):
            errors.append('missing_explicit_handoff_gap')
    elif len(matching) != 1 or h.get('producer_event_id') != matching[0]['event_id']:
        errors.append('completion_producer_output_mismatch')
    else:
        producer = matching[0]
        if producer['event_id'] not in event['parent_event_ids']:
            errors.append('completion_upstream_dependency_missing')
        if h.get('producer_stage') != producer['stage'] or h.get('matched_output_keys') != sorted(snapshots[producer['event_id']]):
            errors.append('completion_producer_binding_mismatch')
    return errors


def validate_term(directory, term_id, *, require_summary=True):
    root=Path(directory).resolve()
    if not term_id or not all(c.isalnum() or c in '_-' for c in term_id):
        raise ValueError('Unsafe term ID')
    path=root/'traces'/'events'/(term_id+'.jsonl')
    errors=[]; events=[]; known={}; seen={}; open_stages={}; finished=set(); snapshots={}
    required={'schema_version','run_id','term_id','event_id','sequence','utc_timestamp','stage','event_type',
              'execution_status','parent_event_ids','payload','artifact_refs','candidate_definitions',
              'candidate_references','evidence_kind','component_ids','review_round'}
    if not path.exists():
        return {'term_id':term_id,'trace_completeness':'missing','errors':['missing_event_file'],'events':0}
    events, errors = read_events(root, term_id)
    run_id='TRACE-'+hashlib.sha256((root/'contract.json').read_bytes()).hexdigest()[:24] if (root/'contract.json').exists() else None
    recorder_config={}
    try:
        recorder_config=json.loads((root/'traces'/'recorder.json').read_text(encoding='utf-8'))
        if recorder_config.get('run_id')!=run_id or recorder_config.get('schema_version') not in SUPPORTED_VERSIONS:
            errors.append('recorder_configuration_binding')
    except Exception:
        errors.append('missing_or_invalid_recorder_configuration')
    for n,e in enumerate(events,1):
        prefix=f'event_{n}:'
        if not isinstance(e,dict) or required-set(e):
            errors.append(prefix+'missing_required_fields'); continue
        if e['schema_version'] not in SUPPORTED_VERSIONS or e['term_id']!=term_id or e['run_id']!=run_id or e['schema_version'] != recorder_config.get('schema_version'):
            errors.append(prefix+'schema_or_identity_binding')
        if e['sequence']!=n or e['event_id']!=f'{term_id}:{n:06d}' or e['event_id'] in seen:
            errors.append(prefix+'sequence_or_event_id')
        try:
            if not e['utc_timestamp'].endswith('Z') or datetime.fromisoformat(e['utc_timestamp'].replace('Z','+00:00')).utcoffset().total_seconds()!=0:
                raise ValueError()
        except Exception:
            errors.append(prefix+'invalid_utc_timestamp')
        if e['execution_status'] not in ('started','completed','failed','skipped') or not isinstance(e['payload'],dict):
            errors.append(prefix+'invalid_status_or_payload')
        if any(p not in seen for p in e['parent_event_ids']):
            errors.append(prefix+'missing_or_forward_parent')
        if n>1 and not e['parent_event_ids']:
            errors.append(prefix+'missing_dependency')
        for ref in e['artifact_refs']:
            try:
                data=resolve_artifact(root,ref)
                if ref['role']=='snapshot':
                    snapshots[e['event_id']]=json.loads(data)
            except EvidenceError as error:
                errors.append(prefix+error.kind+':'+ref['path'])
            except Exception:
                errors.append(prefix+'invalid_artifact_json:'+ref['path'])
        for identity in e['candidate_definitions']:
            if set(('candidate_id','llt_code','llt_name','pt_code','pt_name'))-set(identity) or identity['candidate_id']!='LLT:'+identity['llt_code']:
                errors.append(prefix+'invalid_candidate_identity'); continue
            observed=identities([e['payload'],snapshots.get(e['event_id'])])
            if identity not in observed:
                errors.append(prefix+'candidate_definition_not_in_evidence')
            if identity['candidate_id'] in known and identity not in known[identity['candidate_id']]:
                errors.append(prefix+'candidate_identity_redefined')
            known.setdefault(identity['candidate_id'],[]).append(identity)
        for view in e.get('candidate_views',[]):
            if view['candidate_id']!=view['observed_identity']['candidate_id']:
                errors.append(prefix+'candidate_view_identity_mismatch')
        for ref in e['candidate_references']:
            if ref.get('resolution')=='observed_identity' and ref.get('candidate_id') not in known:
                errors.append(prefix+'unresolved_candidate_reference')
            elif ref.get('resolution') not in ('observed_identity','unresolved_reported_reference'):
                errors.append(prefix+'invalid_candidate_reference_status')
        if e['event_type']=='stage_started':
            open_stages[e['event_id']]=e['stage']
            if e['stage']=='completion.replay':
                errors.extend(prefix+x for x in handoff_errors(e,seen,snapshots))
            if e['stage'] in DEPENDENCIES:
                deps=DEPENDENCIES[e['stage']]
                parent_stages={seen[p]['stage'] for p in e['parent_event_ids'] if p in seen}
                if e['stage'] in ALTERNATIVE_DEPS:
                    valid=bool(parent_stages & set(deps))
                else:
                    valid=set(deps)<=parent_stages
                if not valid:
                    errors.append(prefix+'workflow_dependencies_missing')
                if any(seen[p]['stage'] in deps and seen[p]['event_type'] not in ('stage_completed','term_started')
                       for p in e['parent_event_ids'] if p in seen):
                    errors.append(prefix+'workflow_dependency_not_completed')
        if e['event_type'] in ('stage_completed','stage_failed'):
            matched=[p for p in e['parent_event_ids'] if open_stages.get(p)==e['stage']]
            if len(matched)!=1:
                errors.append(prefix+'unmatched_stage_terminal')
            else:
                del open_stages[matched[0]]
            if e['event_type']=='stage_completed':
                finished.add(e['stage'])
        seen[e['event_id']]=e
    if not events or events[0]['event_type']!='term_started':
        errors.append('missing_term_start')
    if open_stages:
        errors.append('unfinished_stages')
    finals=[e for e in events if e.get('event_type')=='term_final']
    record_path=root/'records'/(term_id+'.json')
    record=None
    if record_path.exists():
        try:
            record=json.loads(record_path.read_text(encoding='utf-8'))
            if not saved_record_shape(record):
                errors.append('invalid_saved_record_structure')
                record=None
        except (ValueError,OSError,UnicodeError):
            pass
    if len(finals)!=1 or record is None:
        errors.append('missing_final_event_or_saved_record')
    else:
        final=finals[0]
        if events[-1] is not final:
            errors.append('events_after_final')
        if final['payload'].get('prediction')!=record.get('prediction') or final['payload'].get('status')!=record.get('status'):
            errors.append('final_output_disagrees_with_saved_record')
        if final['payload'].get('authority')!=output_authority(record):
            errors.append('final_authority_disagrees_with_saved_record')
        ancestry=set(final['parent_event_ids']); pending=list(ancestry)
        while pending:
            for parent in seen.get(pending.pop(),{}).get('parent_event_ids',[]):
                if parent not in ancestry:
                    ancestry.add(parent); pending.append(parent)
        completions=[e for e in events if e['stage']=='completion.replay' and e['event_type'] in ('stage_completed','stage_failed')]
        if completions and not any(e['event_id'] in ancestry for e in completions):
            errors.append('final_completion_dependency_missing')
        if not any(ref['path']==f'records/{term_id}.json' and ref['role']=='saved_record' for ref in final['artifact_refs']):
            errors.append('missing_final_record_reference')
        outcome=record.get('outcome') or {}
        gate=record.get('final_gate') or {}
        if gate.get('decision_authority') and gate['decision_authority']!=outcome.get('decision_authority'):
            errors.append('saved_gate_outcome_authority_inconsistent')
        selected=outcome.get('selected_concepts') or []
        parents=selected+(outcome.get('resolved_pt_concepts') or [])
        actual={'pt':sorted({str(c['pt_code']) for c in parents if c.get('pt_code')}),
                'llt':sorted({str(c['llt_code']) for c in selected if c.get('llt_code')})}
        if outcome and actual!=record.get('prediction'):
            errors.append('saved_outcome_prediction_inconsistent')
        for c in selected:
            cid=c.get('candidate_id') or 'LLT:'+str(c.get('llt_code'))
            if not any(all(str(c.get(k))==i[k] for k in ('llt_code','llt_name','pt_code','pt_name')) for i in known.get(cid,())):
                errors.append('final_candidate_missing_or_inconsistent:'+cid)
        if record.get('status')=='completed':
            exact=(record.get('upstream_state') or {}).get('exact_result') or {}
            required_stages={'normalization','exact_matching'}
            if exact.get('found'):
                required_stages.add('exact_match_outcome')
                skipped={e['stage'] for e in events if e.get('event_type')=='stage_skipped' and e['payload'].get('reason')=='exact_match_bypass'}
                if not {'input_interpretation','reviewer_llm','completion'}<=skipped:
                    errors.append('exact_bypass_not_recorded')
            else:
                required_stages.update(('input_interpretation','fuzzy_matching','semantic_matching','umls_matching',
                                        'aggregation','reviewer_llm','reviewer_validation','coordinator'))
            if not required_stages<=finished:
                errors.append('missing_executed_stage_evidence:'+','.join(sorted(required_stages-finished)))
            if not exact.get('found') and 'completion.replay' not in finished and not record.get('policy_replay_failure'):
                errors.append('missing_completion_replay_evidence')
            if not any(e.get('event_type')=='stage_completed' and isinstance(snapshots.get(e['event_id']),dict)
                       and snapshots[e['event_id']].get('workflow_outcome')==outcome for e in events):
                errors.append('no_observed_final_outcome')
        # A recorded call must have a terminal response or exception. Failed
        # validation is legitimate evidence, not a reason to hide the attempt.
        for call in (e for e in events if e['event_type']=='model_request'):
            if not any(e['event_type'] in ('model_response','model_exception') and call['event_id'] in e['parent_event_ids'] for e in events):
                errors.append('unfinished_model_request:'+call['event_id'])
    if (root/'traces'/'recorder_errors.jsonl').exists():
        try:
            if any(json.loads(line).get('term_id')==term_id for line in (root/'traces'/'recorder_errors.jsonl').read_text().splitlines()):
                errors.append('recorder_reported_failure')
        except Exception:
            errors.append('recorder_error_log_unreadable')
    summary_path=root/'traces'/'summaries'/(term_id+'.json')
    if summary_path.exists():
        try:
            summary=json.loads(summary_path.read_text())
            if summary.get('term_id')!=term_id or summary.get('run_id')!=run_id or summary.get('schema_version') not in SUPPORTED_VERSIONS:
                errors.append('summary_identity_binding')
            if finals and (summary.get('final_event_id')!=finals[0]['event_id'] or summary.get('event_count')!=len(events)):
                errors.append('summary_event_binding')
            if finals and (summary['prediction']!=finals[0]['payload']['prediction'] or summary['acceptance_authority']!=finals[0]['payload']['authority']):
                errors.append('summary_final_inconsistency')
            if errors and summary.get('trace_completeness')=='complete':
                errors.append('summary_incorrectly_claims_completeness')
        except Exception:
            errors.append('invalid_summary')
    elif require_summary:
        errors.append('missing_summary')
    return {'term_id':term_id,'trace_completeness':'complete' if not errors else 'incomplete',
            'errors':errors,'events':len(events),'unknown_candidate_claims':sum(
                r.get('resolution')=='unresolved_reported_reference' for e in events for r in e.get('candidate_references',[]))}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory',type=Path,required=True)
    p.add_argument('--term-id')
    p.add_argument('--output',type=Path)
    a=p.parse_args()
    contract=json.loads((a.directory/'contract.json').read_text(encoding='utf-8'))
    ids=[r['id'] for r in contract['inputs']]
    if a.term_id:
        if a.term_id not in ids:
            p.error('Term not in run contract')
        ids=[a.term_id]
    results=[validate_term(a.directory,i) for i in ids]
    summary={'schema_version':VERSION,'terms':len(results),'complete':sum(r['trace_completeness']=='complete' for r in results),'results':results}
    if a.output:
        a.output.parent.mkdir(parents=True,exist_ok=True)
        with a.output.open('x',encoding='utf-8') as f:
            json.dump(summary,f,indent=2)
    print(json.dumps({'terms':summary['terms'],'complete':summary['complete'],'incomplete_or_missing':summary['terms']-summary['complete']}))
    return int(summary['complete']!=summary['terms'])


if __name__=='__main__':
    raise SystemExit(main())
