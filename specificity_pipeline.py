"""Opt-in original-specificity extension of the existing graph, not a new ranker.

No service clients are constructed here. Retrieval uses injected existing tools.
No positive mappings are manufactured. All existing numerical gates remain.
"""
from __future__ import annotations
from runtime_resources import resource as asp_resource
import csv
import hashlib
from pathlib import Path
from pydantic import Field
from schemas import ReviewerInput, ReviewerDecisionType, HumanReviewCandidate, ExactMatchResult, EvidenceRequest, EvidenceSource, EvidenceRequestAction, EvidenceExecutionStatus, LLTSelectionGateAction
from clinical_specificity import VERSION, interpret, compare

class SpecificityReviewerInput(ReviewerInput):
    clinical_specificity: dict = Field(default_factory=dict)

class SpecificityPolicy:

    def __init__(self, reference_path, loinc_directory=None):
        self.path = Path(reference_path)
        self.sha256 = hashlib.sha256(self.path.read_bytes()).hexdigest()
        with self.path.open(encoding='utf-8-sig', newline='') as stream:
            self.reference = {}
            for row in csv.DictReader(stream):
                key = 'LLT:' + row['llt_code']
                identity = {k: row[k] for k in ('llt_code', 'llt_name', 'pt_code', 'pt_name')}
                if key in self.reference and self.reference[key] != identity:
                    raise ValueError('Conflicting reference identity: ' + key)
                self.reference[key] = identity
        self.components = None
        if loinc_directory is not None:
            from specificity_sources import ComponentSpans
            self.components = ComponentSpans(loinc_directory)

    def identity(self, candidate):
        row = self.reference.get(candidate.candidate_id)
        return bool(candidate.is_current and row and all((str(getattr(candidate, key)) == value for key, value in row.items())))

    def candidate(self, original, candidate):
        evidence = compare(original, candidate.llt_name)
        evidence.update(candidate_id=candidate.candidate_id, llt_code=candidate.llt_code, llt_name=candidate.llt_name, pt_code=candidate.pt_code, pt_name=candidate.pt_name, current_identity_verified=self.identity(candidate), reference_sha256=self.sha256, parent_comparison=compare(original, candidate.pt_name))
        if self.components:
            evidence['source_component_spans'] = {'input': self.components.spans(original), 'candidate': self.components.spans(candidate.llt_name), 'limitation': 'Components are alternatives, not asserted specimen/property/method. No semantic acceptance inferred.'}
            evidence['input_frame']['source_analyte_candidates'] = evidence['source_component_spans']['input']
            evidence['candidate_frame']['source_analyte_candidates'] = evidence['source_component_spans']['candidate']
        if not evidence['current_identity_verified']:
            evidence['full_equivalence_blocked'] = True
            evidence['unresolved'].append('current_identity_or_parent_unverified')
        return evidence

    def evidence(self, original, candidates):
        return dict(version=VERSION, original=original, input_frame=interpret(original), reference_sha256=self.sha256, reference_release='unknown', candidates={c.candidate_id: self.candidate(original, c) for c in candidates}, instructions='Treat spans as explicit wording evidence, not inferred clinical facts. Unrepresented wording is unresolved implicit meaning, not proven absence. Do not accept full equivalence with unsupported additions, contradictions or unresolved substantive information loss. Shared PT is not LLT equivalence. Broader possibilities remain review. SPECIALIST normalization is retrieval-only.')

    def reviewer_input(self, state):
        from reviewer_input import reviewer_input_node
        update = reviewer_input_node(state)
        inp = update['reviewer_input']
        payload = inp.model_dump(mode='python')
        payload['clinical_specificity'] = self.evidence(inp.original_term, inp.aggregation_result.candidates)
        payload['clinical_specificity']['normalization_comparison'] = compare(inp.original_term, inp.normalized_term)
        payload['clinical_specificity']['interpreter_component_texts'] = [c.component_text for c in inp.input_interpretation.asserted_components]
        update['reviewer_input'] = SpecificityReviewerInput.model_validate(payload)
        update['specificity_evidence'] = payload['clinical_specificity']
        return update

    def decorate_reviewer(self, node):

        def wrapped(state):
            update = self.reviewer_input(state)
            return {**update, **node({**state, **update})}
        return wrapped

    def exact(self, node):

        def wrapped(state):
            update = node(state)
            result = update['exact_result']
            original = state['term_input'].term
            evidence = self.evidence(original, [result.concept] if result.found else result.ambiguous_candidates)
            if update.get('specificity_evidence', {}).get('normalized_exact_veto'):
                evidence['normalized_exact_veto'] = True
            if result.found and evidence['candidates'][result.concept.candidate_id]['full_equivalence_blocked']:
                payload = result.model_dump(mode='python')
                payload.update(found=False, concept=None, match_method=None, ambiguous_candidates=[result.concept], ambiguity_reason='Original clinical specificity is unresolved; normalized exact is not semantic authority.')
                update['exact_result'] = ExactMatchResult.model_validate(payload)
                evidence['normalized_exact_veto'] = True
            update['specificity_evidence'] = evidence
            return update
        return wrapped

    def aggregation(self, fuzzy, semantic, umls):
        from aggregation import aggregation_node
        from evidence_request_executor import EvidenceRequestExecutor
        from evidence_reaggregation import _build_combined_source_results, _apply_evidence_queries, _merge_provenance
        executor = EvidenceRequestExecutor(fuzzy_retriever=fuzzy, semantic_retriever=semantic, umls_retriever=umls)

        def wrapped(state):
            update = aggregation_node(state)
            original = state['term_input'].term
            initial = update['aggregation_result']
            evidence = self.evidence(original, initial.candidates)
            executions = []
            already = state.get('specificity_retrieval', {})
            failures = [e for e in evidence['candidates'].values() if e['full_equivalence_blocked']]
            interpretation = state.get('input_interpretation')
            single = interpretation is not None and interpretation.structure_type.value == 'single_concept'
            missing_qualified_pool = not initial.candidates and bool(interpret(original)['attributes'])
            if (failures or missing_qualified_pool) and single and (not already.get('attempted')):
                for source in (EvidenceSource.FUZZY, EvidenceSource.SEMANTIC):
                    request = EvidenceRequest(request_id='specificity_' + source.value, source=source, action=EvidenceRequestAction.FOCUSED_SEARCH, query_term=original, requested_top_k=20, reason='Restore original qualifications and avoid unsupported additions; full-expression retrieval, not acceptance evidence.')
                    executions.append(executor.execute_request(term_id=state['term_id'], review_round=state.get('review_round', 1), normalized_input_term=state['normalization'].normalized_term, request=request))
                completed = [e for e in executions if e.status == EvidenceExecutionStatus.COMPLETED]
                if completed:
                    f, s, u, maps = _build_combined_source_results(normalized_input_term=state['normalization'].normalized_term, initial_fuzzy_result=state['fuzzy_result'], initial_semantic_result=state['semantic_result'], initial_umls_result=state['umls_result'], executions=completed)
                    sources = dict(fuzzy_result=f, semantic_result=s, umls_result=u)
                    merged = aggregation_node({**state, **sources})['aggregation_result']
                    update.update(sources)
                    update['aggregation_result'] = _apply_evidence_queries(aggregation_result=merged, query_maps=maps)
                    update['retrieval_provenance'] = _merge_provenance(existing_provenance=state.get('retrieval_provenance', []), executions=completed)
            update['specificity_retrieval'] = already if already.get('attempted') else dict(attempted=bool(executions), budget=2, top_k=20, failed_checks=[e['candidate_id'] for e in failures], missing_qualified_pool=missing_qualified_pool, executions=[e.model_dump(mode='json') for e in executions], interpretation_single_concept=single)
            update['specificity_evidence'] = self.evidence(original, update['aggregation_result'].candidates)
            return update
        return wrapped

    def validate(self, node):

        def wrapped(state):
            update = node(state)
            current = {**state, **update}
            decision = current['reviewer_decision']
            gate = current.get('llt_selection_gate_result')
            inp = current['reviewer_input']
            candidates = {c.candidate_id: c for c in inp.aggregation_result.candidates}
            if gate:
                candidates.update({c.candidate_id: c for c in gate.ranked_candidates + gate.human_review_candidates})
            evidence = self.evidence(inp.original_term, list(candidates.values()))
            selected = decision.selected_candidate_ids
            blocked = [i for i in selected if i not in evidence['candidates'] or evidence['candidates'][i]['full_equivalence_blocked']]
            if selected and decision.decision_type == ReviewerDecisionType.MATCHED_MULTIPLE:
                from compound_validation import assess_decomposition
                scope = assess_decomposition(inp.original_term, inp.input_interpretation.asserted_components)
                effective = {c['component_id']: c['effective_text'] for c in scope['components']}
                evidence['component_comparisons'] = {}
                blocked = []
                for match in decision.component_matches:
                    text = effective.get(match.component_id)
                    if text is None or match.candidate_id not in candidates:
                        blocked.append(match.candidate_id)
                        continue
                    e = self.candidate(text, candidates[match.candidate_id])
                    evidence['component_comparisons'][match.component_id] = e
                    if e['full_equivalence_blocked']:
                        blocked.append(match.candidate_id)
                covered = {(a['attribute'], a['value']) for text in effective.values() for a in interpret(text)['attributes']}
                uncovered = [a for a in interpret(inp.original_term)['attributes'] if (a['attribute'], a['value']) not in covered]
                evidence['unscoped_compound_attributes'] = uncovered
                if uncovered:
                    blocked = list(selected)
            pt_name = decision.resolved_pt_name or (gate.resolved_pt_name if gate else None)
            unsafe_pt = False
            if decision.decision_type == ReviewerDecisionType.MATCHED_PT_LLT_HUMAN_REVIEW and pt_name:
                parent = compare(inp.original_term, pt_name)
                evidence['retained_pt_comparison'] = parent
                unsafe_pt = bool(parent['unsupported_additions'] or parent['contradictions'])
            if blocked or unsafe_pt:
                from llt_selection_gate import _deferred_decision
                originals = decision.model_dump(mode='json')
                shortlist = [HumanReviewCandidate(**{k: getattr(c, k) for k in ('candidate_id', 'llt_code', 'llt_name', 'pt_code', 'pt_name', 'is_current')}) for i, c in candidates.items() if i in (selected or decision.human_review_candidate_ids)]
                pt_ok = bool(not unsafe_pt and gate and gate.resolved_pt_code and (len({c.pt_code for c in shortlist}) == 1))
                if pt_ok:
                    parent = compare(inp.original_term, gate.resolved_pt_name)
                    pt_ok = not (parent['unsupported_additions'] or parent['contradictions'])
                decision = _deferred_decision(reviewer_input=inp, reviewer_decision=decision, human_review_candidates=shortlist, pt_resolved=pt_ok, resolved_pt_code=gate.resolved_pt_code if pt_ok else None, resolved_pt_name=gate.resolved_pt_name if pt_ok else None, decision_rationale='Deterministic original-specificity veto; original reviewer retained in audit.', human_review_reason='Full equivalence has unsupported or unresolved clinical specificity. Broader possibilities require review.')
                evidence.update(blocked_selection=blocked, original_reviewer_decision=originals, authority='deterministic_specificity_veto', pt_preserved=pt_ok)
                if gate:
                    action = LLTSelectionGateAction.DEFERRED_DISAGREEMENT
                    reason = 'Original-specificity evidence blocks full equivalence.'
                    evidence['prior_gate'] = gate.model_dump(mode='json')
                    payload = gate.model_dump(mode='python')
                    payload.update(action=action, gate_action=action, reason=reason, gate_reason=reason, decision_authority='deterministic_specificity_veto', human_review_candidates=shortlist, human_review_pt_groups=[], coverage_acceptance={}, resolved_pt_code=gate.resolved_pt_code if pt_ok else None, resolved_pt_name=gate.resolved_pt_name if pt_ok else None)
                    gate = type(gate).model_validate(payload)
                    update['llt_selection_gate_result'] = gate
                update['reviewer_decision'] = decision
                node({**current, **update})
            update['specificity_evidence'] = evidence
            return update
        return wrapped

    def outcome(self, node):

        def wrapped(state):
            update = node(state)
            result = update['workflow_outcome']
            config = dict(result.policy_configuration)
            config['clinical_specificity'] = dict(version=VERSION, reference_sha256=self.sha256, evidence=state.get('specificity_evidence'), retrieval=state.get('specificity_retrieval'))
            update['workflow_outcome'] = result.model_copy(update={'policy_configuration': config})
            return update
        return wrapped
