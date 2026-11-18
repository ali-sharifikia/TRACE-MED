"""Read-only trace loader and deterministic presentation model. No pipeline calls."""
import hashlib
import json
from pathlib import Path

from validate_decision_trace import read_events, resolve_artifact, EvidenceError, validate_term, saved_record_shape


def walk(value, pointer=''):
    yield pointer, value
    if isinstance(value, dict):
        for k, v in value.items():
            yield from walk(v, pointer + '/' + str(k).replace('~', '~0').replace('/', '~1'))
    elif isinstance(value, list):
        for i, v in enumerate(value):
            yield from walk(v, pointer + '/' + str(i))


def short(value, limit=400):
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, sort_keys=True)
    return text if len(text) <= limit else text[:limit] + ' [excerpt; full value in evidence]'


class TraceBundle:
    def __init__(self, directory, term_id):
        self.root = Path(directory).resolve()
        if not term_id or not all(c.isalnum() or c in '_-' for c in term_id):
            raise ValueError('Unsafe term ID')
        self.term_id = term_id
        self.validation = validate_term(self.root, term_id)
        self.events, self.load_errors = read_events(self.root, term_id)
        self.by_id = {e['event_id']: e for e in self.events}
        self.artifacts, self.values, self.snapshots = {}, {}, {}
        self.refs_by_event = {}
        self.findings = list(self.validation['errors'])
        for e in self.events:
            refs = []
            for ref in e['artifact_refs']:
                key = (ref['path'], ref['sha256'])
                if key not in self.artifacts:
                    aid = 'A%04d' % (len(self.artifacts) + 1)
                    entry = {'id': aid, **ref, 'status': 'verified'}
                    try:
                        raw = resolve_artifact(self.root, ref)
                        self.values[aid] = json.loads(raw)
                    except EvidenceError as error:
                        entry['status'] = error.kind
                    except (ValueError, UnicodeError):
                        entry['status'] = 'invalid_artifact_json'
                    self.artifacts[key] = entry
                entry = self.artifacts[key]
                refs.append(entry['id'])
                if ref['role'] == 'snapshot' and entry['status'] == 'verified':
                    self.snapshots[e['event_id']] = self.values.get(entry['id'])
            self.refs_by_event[e['event_id']] = refs
        self.summary = None
        path = self.root / 'traces' / 'summaries' / (term_id + '.json')
        self.summary_sha256 = None
        if path.exists():
            try:
                raw = path.read_bytes()
                self.summary_sha256 = hashlib.sha256(raw).hexdigest()
                self.summary = json.loads(raw)
            except (OSError, ValueError, UnicodeError):
                self.findings.append('unreadable_summary')
        else:
            self.findings.append('missing_summary')
        # Never use an intermediate outcome or an unverified record as final.
        self.final = next((e for e in reversed(self.events) if e['event_type'] == 'term_final'), None)
        self.record = None
        if self.final:
            for ref in self.final['artifact_refs']:
                if ref['role'] == 'saved_record':
                    entry = self.artifacts[(ref['path'], ref['sha256'])]
                    if entry['status'] == 'verified':
                        value=self.values[entry['id']]
                        if saved_record_shape(value):self.record = value

    def sources(self, events):
        ids = [e['event_id'] for e in events]
        return {'event_ids': ids, 'artifact_ids': list(dict.fromkeys(
            a for eid in ids for a in self.refs_by_event[eid]))}


TITLES = {
    'run': 'Original input', 'normalization': 'Normalization', 'exact_matching': 'Exact-match check',
    'input_interpretation': 'Structured interpretation', 'retrieval_dispatch': 'Parallel retrieval dispatch',
    'fuzzy_matching': 'Fuzzy retrieval', 'semantic_matching': 'Semantic retrieval', 'umls_matching': 'UMLS retrieval',
    'aggregation': 'Aggregate and rank candidates', 'reviewer_input': 'Candidate set supplied to reviewer',
    'reviewer_llm': 'Reviewer proposal and repairs', 'llt_selection_gate': 'Upstream acceptance gate',
    'reviewer_validation': 'Validate reviewer decision', 'coordinator': 'Route next action',
    'evidence_request_executor': 'Additional evidence requests', 'evidence_reaggregation': 'Reaggregate evidence',
    'exact_match_outcome': 'Exact-match outcome', 'reviewer_final_outcome': 'Reviewer outcome',
    'human_review_outcome': 'Human-review outcome', 'technical_failure_outcome': 'Execution failure',
    'completion.replay': 'Deterministic completion', 'completion.handoff': 'Upstream observation gap',
    'final': 'FINAL SAVED RESULT',
}


def group_events(bundle):
    """Contract events -> execution groups; dependency contraction, not a timeline.

    Roots are actual graph executions and completion invocations. Nested calls
    follow their recorded parent. Parallel retrieval never uses interval overlap.
    """
    groups, owner, starts = [], {}, {}
    for e in bundle.events:
        eid, stage, kind = e['event_id'], e['stage'], e['event_type']
        terminal_start = next((p for p in e['parent_event_ids'] if p in starts and
                               bundle.by_id[p]['stage'] == stage), None)
        is_root = (kind in ('term_started', 'term_final') or
                   (kind == 'stage_started' and stage in TITLES) or
                   (kind == 'observation_gap'))
        if is_root:
            group = {'id': 'N%03d' % (len(groups)+1), 'stage': stage, 'title': TITLES.get(stage, stage),
                     'round': e['review_round'], 'events': [], 'execution_status': 'started',
                     'lines': [], 'detail_ids': []}
            groups.append(group)
        elif terminal_start:
            group = owner[terminal_start]
        else:
            # Latest parent listed is the direct nested observation when a
            # terminal also references its enclosing stage start.
            group = next((owner[p] for p in reversed(e['parent_event_ids']) if p in owner), None)
            if group is None:
                group = {'id': 'N%03d' % (len(groups)+1), 'stage': stage,
                         'title': 'Unattached recorded evidence', 'round': e['review_round'], 'events': [],
                         'execution_status': 'unknown', 'lines': [], 'detail_ids': []}
                groups.append(group)
        group['events'].append(e)
        owner[eid] = group
        if kind == 'stage_started':
            starts[eid] = True
        if kind in ('term_started', 'term_final') or (kind in ('stage_completed','stage_failed') and stage == group['stage']):
            group['execution_status'] = e['execution_status']
    edges = {}
    for e in bundle.events:
        for p in e['parent_event_ids']:
            if p not in owner:
                continue
            a,b = owner[p]['id'],owner[e['event_id']]['id']
            if a != b:
                key=(a,b)
                edges.setdefault(key, {'from':a, 'to':b, 'event_dependencies':[]})['event_dependencies'].append([p,e['event_id']])
    return groups, list(edges.values())


def select_fields(obj, fields):
    return {k:obj[k] for k in fields if k in obj and obj[k] is not None}


def readable(value):
    """Display labels only. Never infer a decision or reinterpret clinical text."""
    labels = {
        'matched': 'Automatic match', 'matched_single': 'Single LLT match',
        'matched_multiple': 'Multiple LLTs match', 'human_review': 'Human review',
        'matched_pt_llt_human_review': 'PT resolved; LLT needs human review',
        'no_supported_match': 'No supported match', 'single_concept': 'One clinical concept',
        'auto_accepted': 'Accepted automatically', 'deferred_disagreement': 'Deferred for disagreement',
        'coverage_policy_pt_resolved_llt_deferred': 'PT resolved; exact LLT wording remains unresolved',
        'deterministic_input_discrimination': 'Deterministic input comparison',
        'retained_gate': 'Retained acceptance gate',
    }
    return labels.get(value, str(value).replace('_', ' ')) if value is not None else 'Unavailable'


def reviewer_group_lines(reviewer_input):
    """Display the supplied hierarchy without deriving or changing any ranking.

    PT/LLT array positions are presentation order, not confidence ranks. Source
    ranks are copied only from evidence bound to that same LLT candidate.
    """
    groups=reviewer_input.get('pt_candidate_groups')
    if not isinstance(groups,list) or not groups:
        return ['The nested reviewer candidate groups were not recorded.']
    lines=['PT groups and LLTs in recorded presentation order; retrieval ranks are source-specific.']
    if not any(g.get('pt_rank') is not None or g.get('rank') is not None for g in groups):
        lines.append('No combined PT ranking was recorded in this reviewer input.')
    for position,group in enumerate(groups,1):
        pt_code=group.get('pt_code','?')
        heading=str(position)+'. PT: '+str(group.get('pt_name','Name unavailable'))+' (PT:'+str(pt_code)+')'
        rank=group.get('pt_rank',group.get('rank'))
        if rank is not None:heading+=' | recorded PT rank '+str(rank)
        lines.append(heading)
        candidates=group.get('llt_candidates')
        if not isinstance(candidates,list):
            lines.append('    LLT membership details unavailable.');continue
        for within,candidate in enumerate(candidates,1):
            cid=candidate.get('candidate_id')
            label=str(candidate.get('llt_name','Name unavailable'))+' ('+str(cid or 'LLT identity unavailable')+')'
            notes=[]
            if candidate.get('pt_code',pt_code)!=pt_code:
                notes.append('WARNING: recorded parent conflicts with this group')
            for source,score_key,score_label in (('fuzzy','fuzzy_score','score'),('semantic','cosine_similarity','cosine'),('umls',None,None)):
                evidence=candidate.get(source+'_evidence')
                if not isinstance(evidence,dict):continue
                if not cid or evidence.get('candidate_id')!=cid:
                    notes.append(source+' rank/score withheld: evidence identity mismatch');continue
                fields=[]
                if evidence.get('rank') is not None:fields.append('rank '+str(evidence['rank']))
                if score_key and isinstance(evidence.get(score_key),(int,float)):
                    fields.append(score_label+' '+format(evidence[score_key],'.6g'))
                if fields:notes.append(source.capitalize()+' '+', '.join(fields))
            if not notes:notes.append('retrieval rank unavailable')
            lines.append('    '+str(position)+'.'+str(within)+' '+label+' — '+'; '.join(notes))
        if not candidates:lines.append('    No LLTs listed in this recorded group.')
    return lines


def reviewer_source_panels(reviewer_input):
    """Separate views of recorded source ranks, never a new combined ranking.

    A candidate with both sources appears in both views. Ties retain supplied
    order, unknown ranks follow known ranks, and unrelated scores never transfer.
    """
    panels=[{'source':source,'title':title,'entries':[]} for source,title in
            (('fuzzy','Fuzzy retrieval'),('semantic','Semantic retrieval'))]
    groups=reviewer_input.get('pt_candidate_groups')
    other=[]
    if not isinstance(groups,list):
        for panel in panels:panel['empty_message']='Reviewer candidate evidence unavailable.'
        return panels,other
    for gi,group in enumerate(groups):
        for ci,candidate in enumerate(group.get('llt_candidates') or []):
            cid=candidate.get('candidate_id')
            identity={'candidate_id':cid,'llt_name':candidate.get('llt_name','Name unavailable'),
                      'pt_code':group.get('pt_code'),'pt_name':group.get('pt_name','Name unavailable'),
                      'supplied_group_position':gi,'supplied_llt_position':ci,
                      'json_pointer':f'/reviewer_input/pt_candidate_groups/{gi}/llt_candidates/{ci}'}
            present=False
            for panel in panels:
                source=panel['source']; evidence=candidate.get(source+'_evidence')
                if not isinstance(evidence,dict) and source not in (candidate.get('evidence_sources') or []):continue
                present=True;entry={**identity,'rank':None,'score':None,'ranking_score':None,'notes':[],
                                    'evidence_pointer':identity['json_pointer']+'/'+source+'_evidence',
                                    'query':candidate.get(source+'_query_term')}
                if candidate.get('pt_code',group.get('pt_code'))!=group.get('pt_code'):
                    entry['notes'].append('Recorded parent conflicts with supplied PT group.')
                if not isinstance(evidence,dict):
                    entry['notes'].append('Source rank and score were not recorded.')
                elif not cid or evidence.get('candidate_id')!=cid:
                    entry['notes'].append('Rank and score withheld: source evidence identity mismatch.')
                else:
                    rank=evidence.get('rank')
                    if isinstance(rank,int) and not isinstance(rank,bool) and rank>0:entry['rank']=rank
                    value=evidence.get('fuzzy_score' if source=='fuzzy' else 'cosine_similarity')
                    if isinstance(value,(int,float)) and not isinstance(value,bool):entry['score']=value
                    # Copy the recorded fuzzy ordering score separately. Never
                    # reconstruct it or substitute raw similarity when absent.
                    if source=='fuzzy':
                        value=evidence.get('ranking_score')
                        if isinstance(value,(int,float)) and not isinstance(value,bool):entry['ranking_score']=value
                panel['entries'].append(entry)
            if not present:other.append(identity)
    for panel in panels:
        # Display order only. Do not mutate source candidates or break rank ties
        # with a new confidence rule.
        panel['entries'].sort(key=lambda e:(e['rank'] is None,e['rank'] or 0))
        panel['empty_message']='No candidates with this source in the recorded reviewer input.'
    return panels,other


def overview_content(bundle, node, identities):
    """Concise evidence-derived prose; full diagnostics stay in cards/manifest.

    No clinical matching, policy execution, or parsing of model narratives here.
    Names come from observed identities; narratives are explicitly quoted claims.
    """
    events=node['events']; stage=node['stage']; lines=[]
    outputs=[bundle.snapshots[e['event_id']] for e in events
             if e['stage']==stage and e['event_type']=='stage_completed'
             and isinstance(bundle.snapshots.get(e['event_id']),dict)]

    def names(ids):
        return '; '.join((identities.get(cid,{}).get('llt_name') or 'Name unavailable')+
                         ' ('+cid+')' for cid in ids) or 'None recorded'

    def proposal(decision, label):
        text=label+': '+readable(decision.get('decision_type'))
        if decision.get('selected_candidate_ids'):text+=' — '+names(decision['selected_candidate_ids'])
        return text

    def number(value):
        return format(value,'.6g') if isinstance(value,(int,float)) else 'unavailable'

    if stage=='run':
        lines=['Original input: '+str(events[0]['payload'].get('text','Unavailable'))]
    elif stage=='normalization':
        for s in outputs:
            lines.append('Normalized text: '+str(s.get('normalization',{}).get('normalized_term','Unavailable')))
    elif stage=='exact_matching':
        for s in outputs:
            result=s.get('exact_result',{})
            found=result.get('found')
            lines.append('Exact match found.' if found is True else 'No exact match found.' if found is False else 'Exact-match result unavailable.')
            if found and result.get('exact_match_provenance'):
                lines.append('Recorded match source: '+readable(result['exact_match_provenance']))
    elif stage=='input_interpretation':
        for s in outputs:
            interpretation=s.get('input_interpretation',{})
            lines.append('Returned interpretation: '+readable(interpretation.get('structure_type')))
            components=interpretation.get('asserted_components',[])
            for c in components[:3]:
                lines.append('Component: '+str(c.get('component_text','Unavailable'))+
                             (' [reported negated]' if c.get('is_negated') else ''))
            if len(components)>3:lines.append(str(len(components)-3)+' further components indexed in the manifest.')
    elif stage=='retrieval_dispatch':
        lines=['Dispatch recorded retrieval branches in parallel.']
    elif stage in ('fuzzy_matching','semantic_matching','umls_matching'):
        for s in outputs:
            result=s.get(stage.replace('_matching','_result'),{})
            lines.append(str(len(result.get('candidates',[])))+' candidate results')
            query=result.get('retrieval_query_term') or result.get('normalized_input_term')
            if query:lines.append('Query: '+str(query))
            if result.get('query_variants'):
                variants=result['query_variants']
                if isinstance(variants,list) and all(isinstance(v,str) for v in variants):
                    lines.extend('Query variant: '+v for v in variants[:2])
                    if len(variants)>2:lines.append('Further query variants indexed in the manifest.')
            if result.get('error_message'):lines.append('Recorded retrieval error: '+short(result['error_message'],180))
    elif stage in ('aggregation','evidence_reaggregation','reviewer_input'):
        for s in outputs:
            ri=s.get('reviewer_input',{})
            agg=s.get('aggregation_result') or ri.get('aggregation_result',{})
            candidates=agg.get('candidates',[])
            groups=agg.get('pt_groups') or ri.get('pt_candidate_groups',[])
            lines.append(str(len(candidates))+' LLT candidates in '+str(len(groups))+' recorded PT groups.')
            lines.append('Candidate order and source evidence retained.' if stage!='reviewer_input' else
                         'This recorded candidate set was supplied to the reviewer.')
            if stage=='reviewer_input':
                node['reviewer_hierarchy_lines']=reviewer_group_lines(ri)
                node['candidate_source_panels'],other=reviewer_source_panels(ri)
                node['candidates_without_fuzzy_semantic_evidence']=other
                lines.append('Fuzzy rank uses a combined lexical score, with possible parent-PT similarity boosts; '
                             'it can differ from raw fuzzy similarity. Semantic rank uses cosine similarity. ')
                if other:
                    lines.append('Other supplied candidates (outside these two sources): '+
                                 '; '.join(c['llt_name']+' ('+str(c['candidate_id'])+'), PT '+str(c['pt_code']) for c in other)+'.')
    elif stage=='reviewer_llm':
        for e in events:
            snapshot=bundle.snapshots.get(e['event_id'])
            kind=e['event_type']; payload=e['payload']
            if kind=='original_model_proposal' and isinstance(snapshot,dict):
                attempt=payload.get('attempt','?')
                lines.append(proposal(snapshot,'Original model proposal, attempt '+str(attempt)))
                rationale=snapshot.get('decision_rationale') or snapshot.get('human_review_reason')
                if rationale:
                    # Excerpts are explicitly labelled; original text remains in
                    # the snapshot and detail card. Never endorse a misnamed LLT.
                    excerpt=str(rationale)
                    if len(excerpt)>260:excerpt=excerpt[:260].rstrip()+'… [excerpt]'
                    lines.append('Recorded model justification (claim): “'+excerpt+'”')
            elif kind=='deterministic_repair':
                actions=payload.get('repair_actions') or payload.get('reasons') or []
                if isinstance(actions,str):actions=[actions]
                lines.extend('Deterministic repair: '+readable(a) for a in actions[:3] if isinstance(a,str))
                if not actions:lines.append('Deterministic repair recorded; diagnostics are in the manifest.')
                if len(actions)>3:lines.append(str(len(actions)-3)+' further repair actions indexed in the manifest.')
            elif kind in ('validation_failed','model_proposal_unparsed','model_exception'):
                lines.append('Attempt '+str(payload.get('attempt','?'))+': '+readable(kind)+'.')
        for s in outputs:
            if s.get('reviewer_decision'):lines.append(proposal(s['reviewer_decision'],'Stage returned after repairs'))
    elif stage=='llt_selection_gate':
        for s in outputs:
            gate=s.get('llt_selection_gate_result',{})
            if gate:lines.append('Deterministic gate: '+readable(gate.get('action')))
            reason=gate.get('reason') or gate.get('gate_reason')
            if reason:lines.append('Recorded reason: '+readable(reason))
            if s.get('reviewer_decision'):lines.append(proposal(s['reviewer_decision'],'Result after gate'))
    elif stage=='reviewer_validation':
        for s in outputs:
            if s.get('reviewer_decision'):lines.append(proposal(s['reviewer_decision'],'Decision returned by validation'))
        lines.append('Validation completion alone is not mapping acceptance.')
    elif stage=='coordinator':
        for s in outputs:
            route=s.get('coordinator_result',{})
            lines.append('Next action: '+readable(route.get('route')))
    elif stage in ('exact_match_outcome','human_review_outcome','reviewer_final_outcome','technical_failure_outcome'):
        for s in outputs:
            outcome=s.get('workflow_outcome',{})
            lines.append('Outcome at this stage: '+readable(outcome.get('status')))
            for c in outcome.get('resolved_pt_concepts',[]):
                lines.append('Resolved PT: '+str(c.get('pt_name','Name unavailable'))+' (PT:'+str(c.get('pt_code','?'))+')')
            for c in outcome.get('selected_concepts',[]):
                lines.append('Selected LLT: '+str(c.get('llt_name','Name unavailable'))+' (LLT:'+str(c.get('llt_code','?'))+')')
            lines.append('This is not the final saved result.')
    elif stage=='completion.replay':
        start=next(e for e in events if e['event_type']=='stage_started' and e['stage']==stage)
        handoff=start['payload'].get('upstream_handoff',{})
        if handoff.get('status')!='observed':
            lines.append('DEPENDENCY GAP: consumed upstream state is not linked to a uniquely observed producer.')
        for e in events:
            if e['event_type']!='stage_completed' or e['stage'] not in ('completion.retained','completion.J','completion.P','completion.T'):continue
            snapshot=bundle.snapshots.get(e['event_id'])
            decision=snapshot[0] if isinstance(snapshot,list) and snapshot else (
                snapshot.get('reviewer_decision',{}) if isinstance(snapshot,dict) else {})
            branch=e['stage'].split('.')[-1]
            label='Retained policy' if branch=='retained' else 'Completion branch '+branch
            lines.append(proposal(decision,label) if isinstance(decision,dict) and decision else label+': result diagnostics unavailable.')
        if bundle.record:
            authority=bundle.final['payload'].get('authority') or {}
            branch=authority.get('branch')
            lines.append('Saved acceptance branch: '+readable(branch))
            audits=(bundle.record.get('final_gate') or {}).get('coverage_acceptance',{})
            for value in audits.values():
                if not isinstance(value,dict):continue
                audit=value.get('audit',{})
                if not isinstance(audit,dict) or not audit.get('accepted'):continue
                numeric=audit.get('numeric',{})
                if 'group_margin' in numeric:
                    lines.append('Accepted equivalence group: score '+number(numeric.get('score'))+
                                 '; group margin '+number(numeric['group_margin'])+
                                 '; original candidate margin '+number(numeric.get('ordinary_margin'))+'.')
                elif numeric:
                    lines.append('Accepted comparison: score '+number(numeric.get('score'))+
                                 ('; outside-rival margin '+number(numeric['margin']) if 'margin' in numeric else '')+
                                 ('; strongest outside score '+number(numeric['outside_score']) if 'outside_score' in numeric else '')+'.')
            thresholds=authority.get('applicable_completion_thresholds')
            if thresholds:
                floor=thresholds.get('margin_floor')
                epsilon=thresholds.get('strict_positive_margin_epsilon')
                lines.append('Applicable completion requirements: score ≥ '+number(thresholds.get('score_floor'))+
                             ('; margin > '+number(epsilon)+'.' if epsilon is not None else '; margin ≥ '+number(floor)+'.'))
            else:lines.append('Active completion thresholds are not explicitly identified in this trace.')
            old=(authority.get('gate_thresholds') or {}).get('top_margin')
            if old is not None:lines.append('Earlier gate margin: '+number(old)+'; kept separate, not substituted for completion diagnostics.')
    elif stage=='completion.handoff':
        lines=['DEPENDENCY GAP: upstream observation unavailable; no execution is reconstructed.']
    elif stage=='final':
        if bundle.record:
            record=bundle.record; outcome=record.get('outcome') or {}; pred=record.get('prediction') or {}
            lines.append('FINAL mapping: '+readable(outcome.get('status',record.get('status'))))
            selected=outcome.get('selected_concepts',[]); resolved=outcome.get('resolved_pt_concepts',[])
            for level in ('pt','llt'):
                for code in pred.get(level,[]):
                    matches=[c for c in selected+resolved if c.get(level+'_code')==code]
                    name=next((c.get(level+'_name') for c in matches if c.get(level+'_name')),None)
                    lines.append(level.upper()+': '+(name or 'Name unavailable')+' ('+level.upper()+':'+code+')')
            if not pred.get('llt'):lines.append('No automatic LLT selected.')
            authority=bundle.final['payload'].get('authority') or {}
            if any('authority' in f or 'final_inconsistency' in f for f in bundle.findings):
                lines.append('WARNING: declared authority is inconsistent with saved evidence.')
            lines.append('Recorded final authority: '+readable(authority.get('decision_authority') or authority.get('kind'))+
                         '; branch '+readable(authority.get('branch'))+'.')
        else:lines=['Final saved record is missing or unverified. No intermediate result is substituted.']
    if not lines:lines=['Recorded stage '+readable(node['execution_status'])+'. Detailed diagnostics are unavailable in this overview.']
    if node.get('skips'):
        skipped=[TITLES.get(s.split(':',1)[0],readable(s.split(':',1)[0]).replace('completion.','Completion ')) for s in node['skips']]
        lines.append('Explicitly skipped: '+', '.join(skipped)+'.')
    return list(dict.fromkeys(lines))


def build_model(bundle):
    groups,edges = group_events(bundle)
    identities, views = {}, []
    for e in bundle.events:
        invalid=any(f.startswith(f"event_{e['sequence']}:") and 'candidate_' in f for f in bundle.findings)
        if not invalid:
            for i in e['candidate_definitions']:
                identities.setdefault(i['candidate_id'], i)
        for v in e.get('candidate_views', []):
            views.append({**v, **bundle.sources([e])})
    cards=[]

    def card(title, lines, events, node=None, pointer='', kind='computed'):
        if node and title.startswith(('Computed ', 'Ranking comparison')):
            prior=next((c for c in cards if c['id'] in node['detail_ids'] and c['title']==title and c['lines']==lines),None)
            if prior:
                src=bundle.sources(events)
                for key in ('event_ids','artifact_ids'):
                    prior[key]=list(dict.fromkeys(prior[key]+src[key]))
                prior['repeated_identical_observations']=prior.get('repeated_identical_observations',1)+1
                return prior
        c={'id':'D%03d'%(len(cards)+1),'title':title,'lines':lines or ['Detailed diagnostics unavailable.'],
           **bundle.sources(events),'json_pointer':pointer,'kind':kind}
        cards.append(c)
        if node:
            node['detail_ids'].append(c['id'])
        return c

    def label(cid):
        i=identities.get(cid)
        return f"{cid} | {i['llt_name']}" if i else str(cid)+' | identity unavailable'

    def decision_lines(d, prefix):
        lines=[prefix+': '+str(d.get('decision_type','unavailable'))]
        for k in ('selected_candidate_ids','human_review_candidate_ids'):
            if d.get(k): lines += [k+': '+', '.join(d[k])]
        for k in ('decision_rationale','human_review_reason'):
            if d.get(k): lines += [('Recorded model justification: ' if prefix.startswith('Original model') else k+': ')+short(d[k],950)]
        return lines

    for g in groups:
        es=g['events']; stage=g['stage']
        main=[e for e in es if e['stage']==stage and e['event_type']=='stage_completed']
        outputs=[(e,bundle.snapshots[e['event_id']]) for e in main if e['event_id'] in bundle.snapshots]
        if stage=='run':
            g['lines']=['Original: '+short(es[0]['payload'].get('text','Unavailable'),200)]
            card('Original input', [short(es[0]['payload'],1200)], [es[0]],g,kind='observed')
        elif stage=='normalization':
            for e,s in outputs:
                norm=s.get('normalization',{})
                g['lines']=['Normalized: '+short(norm.get('normalized_term','unavailable'),180)]
                card('Normalization and applied operations',[k+': '+short(v,700) for k,v in norm.items()],[e],g,'/normalization')
        elif stage=='exact_matching':
            for e,s in outputs:
                ex=s.get('exact_result',{})
                g['lines']=['Match found: '+str(ex.get('found','unavailable')),
                            'Provenance: '+str(ex.get('exact_match_provenance') or 'unavailable')]
                card('Exact-match evidence',[k+': '+short(v,650) for k,v in ex.items() if v not in (None,[],{})],[e],g,'/exact_result')
        elif stage=='input_interpretation':
            for e,s in outputs:
                inp=s.get('input_interpretation',{})
                g['lines']=['Structure: '+str(inp.get('structure_type','unavailable')),
                    'Components: '+str(len(inp.get('asserted_components',[])))]
                card('Returned interpretation (may include repairs)',
                     [k+': '+short(v,1400) for k,v in inp.items() if v not in (None,[],{})], [e],g,'/input_interpretation')
        elif stage in ('fuzzy_matching','semantic_matching','umls_matching'):
            for e,s in outputs:
                r=s.get(stage.replace('_matching','_result'),{})
                g['lines']=['Candidates: '+str(len(r.get('candidates',[]))),
                    'Query: '+short(r.get('retrieval_query_term') or r.get('normalized_input_term','unavailable'),150)]
                card('Retrieval queries and source result',
                     [k+': '+short(r[k],1800) for k in ('retrieval_query_term','query_variants','lookup_status','error_message','embedding_model') if k in r]+[
                     'Candidate order: '+', '.join(c.get('candidate_id','unknown') for c in r.get('candidates',[]))], [e],g)
        elif stage in ('aggregation','evidence_reaggregation','reviewer_input'):
            for e,s in outputs:
                ri=s.get('reviewer_input',{})
                agg=s.get('aggregation_result') or ri.get('aggregation_result',{})
                cs=agg.get('candidates',[])
                expanded=[c['candidate_id'] for c in cs if any(isinstance(v,dict) and v.get('introduced_by_pt_expansion')
                          for _,v in walk(c))]
                g['lines']=['LLT candidates: '+str(len(cs)), 'PT groups: '+str(len(agg.get('pt_groups',[])))]
                card('Candidate membership and ordering',[
                    'Order: '+', '.join(c['candidate_id'] for c in cs),
                    'Recorded PT-expansion flags (including nested source evidence): '+(', '.join(expanded) or 'Unavailable in this snapshot; no absence of expansion is inferred.'),
                    'Group diagnostics: '+short(select_fields(agg,('unique_llt_count','unique_pt_count','total_source_candidates')),700),
                    'Full candidate attributes and group membership: referenced snapshot.'],[e],g)
        elif stage=='reviewer_llm':
            proposals=[e for e in es if e['event_type']=='original_model_proposal']
            g['lines']=['Original proposals: '+str(len(proposals)),
                'Repairs: '+str(sum(e['event_type']=='deterministic_repair' for e in es)),
                'Validation failures: '+str(sum(e['event_type']=='validation_failed' for e in es))]
            for p in proposals:
                original=bundle.snapshots.get(p['event_id'],{})
                g['lines'].append('Model proposed: '+str(original.get('decision_type','unavailable'))+' '+', '.join(original.get('selected_candidate_ids',[])))
            for e,s in outputs:
                d=s.get('reviewer_decision',{})
                card('Reviewer stage returned decision (after any repairs)', decision_lines(d,'Returned decision'),[e],g,'/reviewer_decision')
        elif stage in ('llt_selection_gate','reviewer_validation','coordinator'):
            for e,s in outputs:
                gate=s.get('llt_selection_gate_result',{})
                d=s.get('reviewer_decision',{})
                route=s.get('coordinator_result',{})
                if gate:g['lines']=['Gate action: '+str(gate.get('action',gate.get('gate_action','unavailable'))),
                                    'Recorded reason: '+short(gate.get('reason',gate.get('gate_reason','unavailable')),200)]
                elif route:g['lines']=['Route: '+str(route.get('route','unavailable')),
                                      'From decision: '+str(route.get('source_decision_type','unavailable'))]
                elif d:g['lines']=['Validated decision: '+str(d.get('decision_type','unavailable')),
                                   'Selected: '+', '.join(d.get('selected_candidate_ids',[]))]
        elif stage in ('exact_match_outcome','human_review_outcome','reviewer_final_outcome','technical_failure_outcome'):
            for e,s in outputs:
                o=s.get('workflow_outcome',{})
                g['lines']=['INTERMEDIATE: '+str(o.get('status','unavailable')),'May change during completion.']
                card('Intermediate outcome, not the final result',[
                    'Status: '+str(o.get('status')),'Selected: '+short(o.get('selected_concepts',[]),1600),
                    'PT-only: '+short(o.get('resolved_pt_concepts',[]),900)], [e],g,'/workflow_outcome')
        elif stage=='completion.replay':
            start=next(e for e in es if e['event_type']=='stage_started' and e['stage']==stage)
            h=start['payload'].get('upstream_handoff')
            g['lines']=['Handoff: '+str(h.get('status') if h else 'legacy dependency not recorded'),
                'Branches: '+', '.join(e['stage'].split('.')[-1] for e in es if e['event_type']=='stage_started' and e['stage'] in ('completion.retained','completion.J','completion.P','completion.T'))]
            card('Upstream to completion handoff', [short(h,1800) if h else 'Legacy trace: upstream-to-completion data dependency was not explicitly bound. No causal edge is reconstructed.'],[start],g)
            if bundle.record:
                g['lines'].append('Saved authority branch: '+str((bundle.final['payload'].get('authority') or {}).get('branch','unavailable')))
                for ns, value in (bundle.record.get('final_gate') or {}).get('coverage_acceptance',{}).items():
                    if isinstance(value,dict) and isinstance(value.get('audit'),dict) and value['audit'].get('accepted'):
                        num=value['audit'].get('numeric',{})
                        if num: g['lines'].append(ns+': '+short(num,230))
        elif stage=='completion.handoff':
            g['lines']=['GAP: upstream execution unavailable','Consumed state is recorded, not reconstructed.']
        elif stage=='final':
            if bundle.record:
                record=bundle.record; outcome=record.get('outcome') or {}
                g['lines']=['Mapping: '+str(outcome.get('status') or record.get('status','unavailable')),
                    'PT: '+', '.join(record.get('prediction',{}).get('pt',[])),
                    'LLT: '+', '.join(label('LLT:'+cid) for cid in record.get('prediction',{}).get('llt',[]))]
                a=es[-1]['payload'].get('authority',{})
                lines=['Saved mapping status: '+str(outcome.get('status')),
                    'Execution status: '+str(record.get('status')),
                    'Declared final authority: '+short(select_fields(a,('kind','branch','decision_authority','gate_authority')),1000),
                    'decision_origin is recorded provenance, NOT independent model authorization: '+str(outcome.get('decision_origin'))]
                if any('authority' in f or 'final_inconsistency' in f for f in bundle.findings):
                    lines.insert(0,'WARNING: final authority is inconsistent with saved evidence; do not treat the declaration below as verified.')
                for c in outcome.get('selected_concepts',[]):
                    lines.append('Selected LLT '+str(c.get('llt_code'))+' '+str(c.get('llt_name'))+' -> PT '+str(c.get('pt_code'))+' '+str(c.get('pt_name')))
                for c in outcome.get('resolved_pt_concepts',[]):
                    lines.append('Resolved PT only: '+str(c.get('pt_code'))+' '+str(c.get('pt_name')))
                lines += ['Applicable completion thresholds: '+short(a.get('applicable_completion_thresholds'),1000),
                          'Retained gate fields (NOT automatically active completion criteria): '+short(a.get('gate_thresholds'),1100)]
                card('Final saved result and authority',lines,[es[-1]],g,kind='final')
                gate=record.get('final_gate') or {}
                for namespace,v in (gate.get('coverage_acceptance') or {}).items():
                    if isinstance(v,dict) and isinstance(v.get('audit'),dict):
                        au=v['audit']
                        card('Final policy evidence: '+namespace,
                            [k+': '+short(au[k],1400) for k in ('accepted','blockers','group','representative','numeric','threshold_policy','proof_priority') if k in au]+[
                            'Source namespace retained verbatim; numerical values are scores, not probabilities.'],[es[-1]],g,
                            '/final_gate/coverage_acceptance/'+namespace+'/audit')
            else:
                g['lines']=['Final record missing or unverified','No intermediate result promoted to final.']
        if not g['lines']:
            g['lines']=['Recorded execution: '+g['execution_status']]

        # Distinct original proposals, retry attempts, repairs and computations.
        for e in es:
            kind=e['event_type']; s=bundle.snapshots.get(e['event_id'])
            if kind=='original_model_proposal' and isinstance(s,dict):
                role=e['payload'].get('role','model'); attempt=e['payload'].get('attempt','?')
                if role=='reviewer':
                    lines=decision_lines(s,'Original model proposal')
                    assessments=s.get('candidate_assessments',[])
                    lines+=['Assessed candidate IDs: '+', '.join(a.get('candidate_id','?') for a in assessments),
                            'Assessment narratives for consequential candidates appear below; all narratives remain in this snapshot.']
                else:
                    lines=[k+': '+short(v,1400) for k,v in s.items()]
                card('Original '+role+' proposal | attempt '+str(attempt),lines,[e],g,kind='model')
            elif kind in ('deterministic_repair','validation_failed','model_proposal_unparsed','model_exception','execution_failed','observation_gap'):
                card(kind.replace('_',' ').capitalize(),[short(e['payload'],2200)], [e],g,kind='repair' if kind=='deterministic_repair' else 'warning')
            elif kind=='model_request':
                card('Model attempt request', ['Attempt event '+e['event_id']+'; exact prompt/schema/candidate presentation in linked request.'],[e],g,kind='model')
            elif kind=='stage_skipped':
                # compact list, never a fake executed node
                g.setdefault('skips',[]).append(e['stage']+': '+str(e['payload'].get('reason','unavailable')))
            elif kind=='stage_completed' and e['stage'].startswith('comparison.') and isinstance(s,dict):
                card('Computed '+e['stage'],[k+': '+short(s[k],1400) for k in (
                    'accepted','branch','blockers','old_reasons','group','anchor','representative','numeric',
                    'threshold_policy','proof_priority') if k in s]+['Other per-rule diagnostics remain in the linked snapshot.'],[e],g)
            elif kind=='stage_completed' and e['stage']=='ranking.event_comparison' and isinstance(s,dict):
                ranks=s.get('pt_rankings',[])
                card('Ranking comparison and reference groups',[
                    'PT ranking order: '+', '.join(str(r.get('pt_code')) for r in ranks),
                    'Top recorded rankings: '+short([select_fields(r,('pt_code','pt_name','best_candidate_id','original_score','score','rank')) for r in ranks[:4]],1600),
                    'Current-reference group sizes: '+short({p:len(cs) for p,cs in s.get('groups',{}).items()},800),
                    'Full LLT identities, ordered PT comparisons, ties and configuration are in the referenced snapshot.'],[e],g)
            elif kind=='stage_completed' and e['stage'].startswith('completion.') and e['stage']!='completion.replay':
                ds=s[0] if isinstance(s,list) and s and isinstance(s[0],dict) else (s or {}).get('reviewer_decision',{}) if isinstance(s,dict) else {}
                lines=[]
                if ds: lines+=decision_lines(ds,'Deterministic returned decision')
                if isinstance(s,dict):
                    for key in ('reviewer_validation','reviewer_validation_result','coordinator_result','workflow_outcome'):
                        if key in s:
                            value=s[key]
                            lines+=[key+': '+short(select_fields(value,('status','is_valid','valid','route','reasons','errors','selected_concepts','resolved_pt_concepts')) if isinstance(value,dict) else value,1500)]
                card(e['stage']+' | executed result (not necessarily acceptance)',lines,[e],g)
            elif kind=='stage_completed' and e['stage'] in ('llt_selection_gate','reviewer_validation','coordinator','evidence_request_executor') and isinstance(s,dict):
                for key,value in s.items():
                    if not isinstance(value,dict): continue
                    fields=select_fields(value,('action','decision_type','selected_candidate_ids','route','status','is_valid','valid','errors','reason','reasons','messages','gate_reason','gate_reasons','minimum_score','minimum_margin','top_score','top_margin'))
                    card(e['stage']+' | '+key, [k+': '+short(v,1300) for k,v in fields.items()]+[
                        'Full returned diagnostic fields: '+', '.join(value)], [e],g,'/'+key)
        if g.get('skips'):
            card('Branches explicitly skipped',g['skips'],[e for e in es if e['event_type']=='stage_skipped'],g)
        g.update(bundle.sources(es))

    # Consequential candidates: final, model selections, proven groups and closest
    # rivals. Sorting is display-only, never a new ranking or clinical preference.
    relevant=set()
    if bundle.record:
        relevant.update('LLT:'+c for c in bundle.record.get('prediction',{}).get('llt',[]))
    for e in bundle.events:
        s=bundle.snapshots.get(e['event_id'])
        if isinstance(s,dict) and (e['event_type']=='original_model_proposal' or e['stage'].startswith('comparison.')):
            for k in ('selected_candidate_ids','group','top_set'):
                relevant.update(x for x in s.get(k,[]) if isinstance(x,str))
            num=s.get('numeric',{})
            for k in ('strongest_outside','ordinary_closest','outside_rivals'):
                relevant.update(x for x in num.get(k,[]) if isinstance(x,str))
    shown=sorted(relevant)[:16]
    if not shown: shown=sorted(identities)[:6]
    for cid in shown:
        i=identities.get(cid)
        facts=[]; source_events=[]; associations=[]; dedup=set()
        for e in bundle.events:
            if e['event_type'] not in ('stage_completed','original_model_proposal'):continue
            s=bundle.snapshots.get(e['event_id'])
            if e['stage'].startswith('comparison.') and isinstance(s,dict):
                for field in ('input_support','comparisons'):
                    support=s.get(field,{}).get(cid) if isinstance(s.get(field),dict) else None
                    if isinstance(support,dict):
                        f=select_fields(support,('positive','score','clinical_guards','unrepresented_input_information',
                            'existing_wording_support','saved_retrieval_conflict','conflicts','unsupported_specificity','blockers'))
                        key=(e['stage'],e['review_round'],json.dumps(f,sort_keys=True))
                        if key not in dedup:
                            dedup.add(key);facts.append('Computed input support: '+short(f,900));source_events.append(e)
                            associations.append({'event_id':e['event_id'],'json_pointer':'/'+field+'/'+cid})
            for ptr,obj in walk(s):
                if not isinstance(obj,dict) or obj.get('candidate_id')!=cid:continue
                if e['event_type']=='original_model_proposal':
                    keys=('verdict','rationale'); pre='Recorded model assessment (claim): '
                elif e['stage'] in ('fuzzy_matching','semantic_matching','umls_matching','aggregation'):
                    keys=('rank','fuzzy_score','ranking_score','cosine_similarity','evidence_sources','retrieval_methods','conflict_flags','directionality_conflict','negation_conflict','unsupported_specificity','introduced_by_pt_expansion');pre=e['stage']+': '
                else:continue
                f=select_fields(obj,keys)
                if not f:continue
                key=(e['stage'],e['review_round'],json.dumps(f,sort_keys=True))
                if key in dedup:continue
                dedup.add(key)
                facts.append(pre+short(f,900))
                source_events.append(e);associations.append({'event_id':e['event_id'],'json_pointer':ptr})
        name=label(cid)
        lines=[name, ('Parent: '+i['pt_code']+' | '+i['pt_name']) if i else 'Identity unresolved.']
        lines+=facts[:10]
        if len(facts)>10:lines.append(f'{len(facts)-10} additional observations condensed; see candidate evidence references.')
        lines+=['No unrecorded per-rule rejection is inferred.']
        c=card('Candidate evidence | '+cid,lines,list({e['event_id']:e for e in source_events}.values()),pointer='',kind='candidate')
        c['candidate_id']=cid;c['associations']=associations
    if views:
        card('Comparison views and model-claimed identities',[
            v['candidate_id']+' | '+v['observed_identity']['llt_name']+' | '+v['scope'] for v in views[:24]]+[
            f'{max(0,len(views)-24)} additional views omitted; all are indexed in the manifest. Observed identity is not renamed.'],
            [bundle.by_id[eid] for v in views[:24] for eid in v['event_ids']],kind='warning')
    omitted=sorted(set(identities)-set(shown))
    card('Candidate detail coverage', [f'{len(shown)} candidate detail panels; {len(omitted)} other observed identities condensed.',
        'Omitted detail IDs: '+', '.join(omitted),
        'All observed identities, source events and original candidate sets are indexed in the manifest. No candidates were removed from the pipeline.'],
        [e for e in bundle.events if e['candidate_definitions']],kind='observed')
    if bundle.findings:
        card('Trace gaps and integrity findings',bundle.findings,[],kind='warning')
    for g in groups:
        g['overview_lines']=overview_content(bundle,g,identities)
        overview_events=list(g['events'])
        if g['stage']=='completion.replay' and bundle.record:
            overview_events.append(bundle.final)
        g['overview_sources']=bundle.sources(overview_events)
        g.pop('events')
    authority = bundle.final['payload'].get('authority') if bundle.final else None
    return {'presentation_version':'trace.static-view/1','term_id':bundle.term_id,
        'run_id':bundle.events[0]['run_id'] if bundle.events else None,
        'schema_versions':sorted({e['schema_version'] for e in bundle.events}),
        'validation':bundle.validation,'findings':bundle.findings,
        'original_input':next((e['payload'].get('text') for e in bundle.events if e['event_type']=='term_started'),None),
        'synthetic':bundle.term_id.startswith('SYNTHETIC'),
        'mapping_status':(bundle.record.get('outcome') or {}).get('status',bundle.record.get('status')) if bundle.record else 'unavailable',
        'prediction':bundle.record.get('prediction') if bundle.record else None,
        'authority':authority,'authority_consistent':not any('authority' in f or 'final_inconsistency' in f for f in bundle.findings),
        'historical_summary_completeness':bundle.summary.get('trace_completeness') if isinstance(bundle.summary,dict) else None,
        'summary_sha256':bundle.summary_sha256,
        'nodes':groups,'edges':edges,'cards':cards,
        'candidate_identities':identities,'candidate_views':views,'condensed_candidate_ids':omitted,
        'artifacts':list(bundle.artifacts.values()),
        'limitations':['Arrows contract only recorded dependencies; absent edges are never reconstructed.',
            'Execution completed does not mean accepted. Complete trace does not mean clinically correct.',
            'Model justifications are recorded claims, not endorsed terminology evidence. Scores are not probabilities.']}
