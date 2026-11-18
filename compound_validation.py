"""Source-grounded compound assessment, independent of benchmark labels.

This is a conservative assertion check, not a clinical-language parser. Unknown
scope is preserved as uncertainty; a model's multiple_concepts label is not proof.
"""
from __future__ import annotations
from runtime_resources import resource as asp_resource
import re
from dataclasses import replace
COMPOUND_POLICY_VERSION = '1.3.2'
ATTRIBUTES = frozenset('worsening worsened worse progression progressive progressed exacerbation exacerbated aggravation aggravated deterioration deteriorated severe severity mild moderate acute chronic recurrent recurrence persistent increased decreased high low bilateral left right associated related'.split())
ANATOMY = frozenset('prostate prostatic renal kidney hepatic liver lung pulmonary breast breastbone tongue oral mouth abdominal abdomen gastric stomach cardiac heart knee hip arm leg eye ear skin bone cerebral brain'.split())
FUNCTION_WORDS = frozenset('the a an of in on with to at both and or after before following during developed occurred reported was were is are patient patients'.split())
GENERIC_ENTITY = frozenset('tumor tumors tumour tumours neoplasm neoplasms mass masses growth growths'.split())
MALIGNANCY = frozenset('cancer cancers carcinoma carcinomas malignant malignancy metastasis metastases metastatic sarcoma lymphoma leukemia leukaemia'.split())
SITE_EQUIVALENTS = {'prostatic': 'prostate', 'renal': 'kidney', 'hepatic': 'liver', 'pulmonary': 'lung', 'oral': 'mouth', 'abdominal': 'abdomen', 'gastric': 'stomach', 'cardiac': 'heart', 'cerebral': 'brain'}

def tokens(text):
    return re.findall('[a-z]+', text.casefold())

def assess_decomposition(original, components):
    """Return auditable source spans, event cores, and justified shared scope."""
    components = [c.model_dump() if hasattr(c, 'model_dump') else c for c in components]
    positive = [c for c in components if not c.get('is_negated', False)]
    source = ' '.join(original.casefold().split())
    rows, reasons = ([], [])
    prefix = source.split(':', 1)[0] if ':' in source else ''
    prefix_words = set(tokens(prefix))
    shared_scope = prefix if len(positive) > 1 and prefix_words and (prefix_words <= ANATOMY | ATTRIBUTES | FUNCTION_WORDS) else ''
    context_match = re.match('^(?:after|during|following)\\s+(?:the\\s+)?(?:treatment|infusion|surgery|dose)\\s*[:,]', source)
    shared_context = context_match.group(0).rstrip(':,').strip() if context_match else ''
    for component in positive:
        text = ' '.join(component['component_text'].casefold().split())
        words = set(tokens(text))
        core = words - ATTRIBUTES - ANATOMY - FUNCTION_WORDS
        start = source.find(text)
        findings = []
        if start < 0:
            findings.append('component_not_grounded_in_source')
        if not core:
            findings.append('attribute_or_anatomy_without_independent_event')
        rows.append({'component_id': component['component_id'], 'component_text': component['component_text'], 'source_span': [start, start + len(text)], 'event_core': sorted(core), 'qualifiers': sorted(words & (ANATOMY | ATTRIBUTES)), 'effective_text': (shared_scope + ' ' + text).strip() if shared_scope else text, 'shared_scope': shared_scope or None, 'reasons': findings})
    if len(positive) < 2:
        return {'version': COMPOUND_POLICY_VERSION, 'status': 'not_multiple', 'reasons': [], 'components': rows}
    from clinical_features import interpreted_features
    source_features = interpreted_features(original)
    scoped_features = [interpreted_features(row['effective_text']) for row in rows]
    if any((bound not in [b for f in scoped_features for b in f['bounds']] for bound in source_features['bounds'])):
        reasons.append('numerical_constraint_not_scoped_to_component')
    if set(source_features['units']) - {u for f in scoped_features for u in f['units']}:
        reasons.append('measurement_unit_not_scoped_to_component')
    ordered = sorted(rows, key=lambda r: r['source_span'])
    connectors = []
    for first, second in zip(ordered, ordered[1:]):
        if first['source_span'][1] > second['source_span'][0]:
            reasons.append('overlapping_component_spans')
        between = source[first['source_span'][1]:second['source_span'][0]]
        connectors.append(between.strip())
        if re.search('\\b(or|versus|vs)\\b|/', between):
            reasons.append('alternative_not_simultaneous_assertion')
        elif not re.search('\\b(and|also|plus|followed by|subsequently|then|developed|occurred|reported)\\b|[;,]', between):
            reasons.append('independent_assertion_relationship_unresolved')
        first_sites = {SITE_EQUIVALENTS.get(t, t) for t in first['qualifiers'] if t in ANATOMY}
        second_sites = {SITE_EQUIVALENTS.get(t, t) for t in second['qualifiers'] if t in ANATOMY}
        distinct_sites = bool(first_sites and second_sites and first_sites.isdisjoint(second_sites))
        if set(first['event_core']) & set(second['event_core']) and (not distinct_sites):
            reasons.append('overlapping_event_cores_require_redundancy_review')
    for row in rows:
        reasons.extend(row['reasons'])
    covered = set()
    for row in rows:
        if row['source_span'][0] >= 0:
            covered.update(range(*row['source_span']))
    remainder = ''.join((' ' if i in covered else c for i, c in enumerate(source)))
    remaining = set(tokens(remainder)) - FUNCTION_WORDS - {'also', 'plus', 'subsequently', 'then', 'by', 'followed'}
    if shared_scope:
        remaining -= prefix_words
    if shared_context:
        remaining -= set(tokens(shared_context))
    if remaining & (ANATOMY | ATTRIBUTES):
        reasons.append('unassigned_qualifier_scope')
    if remaining - ANATOMY - ATTRIBUTES:
        reasons.append('unrepresented_source_content')
    qualifier_sets = [set(row['qualifiers']) & ANATOMY for row in rows]
    if any(qualifier_sets) and (not all(qualifier_sets)) and (not shared_scope):
        symptom_heads = {'pain', 'swelling', 'tenderness', 'redness', 'stiffness', 'weakness'}
        if any((set(row['event_core']) <= symptom_heads and (not set(row['qualifiers']) & ANATOMY) for row in rows)):
            reasons.append('shared_anatomy_scope_unresolved')
    reasons = sorted(set(reasons))
    rejected = {'attribute_or_anatomy_without_independent_event', 'overlapping_component_spans', 'alternative_not_simultaneous_assertion', 'component_not_grounded_in_source'}
    return {'version': COMPOUND_POLICY_VERSION, 'status': 'rejected' if rejected & set(reasons) else 'unresolved' if reasons else 'accepted', 'reasons': reasons or ['distinct_source_grounded_events_with_assertion_relationship'], 'components': rows, 'assertion_connectors': connectors, 'shared_episode_context': shared_context or None, 'scope_authority': 'explicit_shared_prefix' if shared_scope else 'local_source_spans'}

def component_semantic_issues(text, candidate_text):
    """Unsupported specificity is uncertainty, not an invented opposite diagnosis."""
    source, target = (set(tokens(text)), set(tokens(candidate_text)))
    issues = []
    if source & GENERIC_ENTITY and (not source & MALIGNANCY) and target & MALIGNANCY:
        issues.append('generic_neoplasm_does_not_establish_malignancy')
    if (target & ANATOMY) - (source & ANATOMY):
        if not source & ANATOMY:
            issues.append('anatomical_qualifier_not_scoped_to_component')
    return issues

def assess_multiple_acceptance(reviewer_input, decision, config, *, llt_rankings=None, reference_complete=True, component_rankings=None):
    """Shared final acceptance policy. Missing evidence is never a passing score.

LLT ranks must be computed per component over the selected current PT's full
reference. Saved partial pools may diagnose rejection, but cannot prove acceptance.
"""
    from adaptive_review_policy import prepare_policy_evidence, _ambiguity
    from clinical_features import automatic_mapping_uncertainties
    from pt_reranking import build_component_pt_rankings, explicit_clinical_conflicts
    from llt_selection_gate import _explicit_candidate_conflicts
    from schemas import HumanReviewCandidate
    interpretation = reviewer_input.input_interpretation
    decomposition = assess_decomposition(reviewer_input.normalized_term, interpretation.asserted_components)
    ranks = component_rankings if component_rankings is not None else build_component_pt_rankings(reviewer_input)
    by_component = {r['component_id']: r['pt_rankings'] for r in ranks}
    by_id = {c.candidate_id: c for c in reviewer_input.aggregation_result.candidates}
    matches = list(decision.component_matches)
    global_reasons = []
    global_reasons.extend(automatic_mapping_uncertainties(reviewer_input.normalized_term))
    if decomposition['status'] != 'accepted':
        global_reasons.append('decomposition_' + decomposition['status'])
    components = decomposition['components']
    component_ids = {c['component_id'] for c in components}
    if {m.component_id for m in matches} != component_ids:
        global_reasons.append('mapping_does_not_cover_exactly_asserted_components')
    if sorted((m.candidate_id for m in matches)) != sorted(decision.selected_candidate_ids):
        global_reasons.append('selected_candidates_and_component_mappings_disagree')
    if len({m.pt_code for m in matches}) != len(matches):
        global_reasons.append('redundant_parent_coding_requires_review')
    audits = []
    for component in components:
        cid, text = (component['component_id'], component['effective_text'])
        selections = [m for m in matches if m.component_id == cid]
        reasons, conflicts, ambiguities = ([], [], [])
        row = {'component_id': cid, 'component_text': component['component_text'], 'effective_text': text, 'selected_candidate_ids': [m.candidate_id for m in selections], 'pt_score': None, 'pt_margin': None, 'llt_score': None, 'llt_margin': None, 'current_parent_verified': False, 'retrieval_conflict_findings': [], 'ranking_conflict_findings': []}
        if len(selections) != 1:
            reasons.append('component_requires_one_unique_mapping')
        else:
            match = selections[0]
            candidate = by_id.get(match.candidate_id)
            pt_ranks = by_component.get(cid, [])
            selected_rank = next((i for i, p in enumerate(pt_ranks, 1) if p['pt_code'] == match.pt_code), None)
            row.update(selected_pt_code=match.pt_code, selected_pt_rank=selected_rank)
            if selected_rank is not None:
                row['pt_score'] = pt_ranks[selected_rank - 1]['reranking_score']
                rivals = [p['reranking_score'] for p in pt_ranks if p['pt_code'] != match.pt_code]
                row['pt_margin'] = round(row['pt_score'] - max(rivals), 6) if rivals else 1.0
            if selected_rank != 1:
                reasons.append('selected_pt_not_unique_deterministic_leader')
            if row['pt_score'] is None or row['pt_score'] < config.pt_minimum_score:
                reasons.append('pt_score_below_threshold_or_missing')
            if row['pt_margin'] is None or row['pt_margin'] < config.pt_minimum_margin:
                reasons.append('pt_margin_below_threshold_or_missing')
            if candidate is None or not candidate.is_current or candidate.pt_code != match.pt_code:
                reasons.append('candidate_identity_or_current_parent_unverified')
            else:
                row['retrieval_conflict_findings'] = _explicit_candidate_conflicts(candidate)
                if selected_rank is not None:
                    selected_pt = pt_ranks[selected_rank - 1]
                    row['ranking_conflict_findings'] = selected_pt.get('score_breakdown', {}).get('modifier_conflict_reasons', [])
                    leader = by_id.get(selected_pt.get('best_candidate_id'))
                    if leader and leader.candidate_id != candidate.candidate_id:
                        row['retrieval_conflict_findings'].extend(_explicit_candidate_conflicts(leader))
                if row['retrieval_conflict_findings'] or row['ranking_conflict_findings']:
                    reasons.append('existing_pt_compatibility_conflict_guard_failed')
                conflicts = sorted(set(explicit_clinical_conflicts(text, candidate.llt_name)) | set(explicit_clinical_conflicts(text, candidate.pt_name)))
                reasons.extend(component_semantic_issues(text, candidate.llt_name + ' ' + candidate.pt_name))
                entries = list((llt_rankings or {}).get(cid, []))
                found = next((e for e in entries if e.candidate_id == match.candidate_id), None)
                row['current_parent_verified'] = bool(found and found.llt_code == candidate.llt_code and (found.pt_code == candidate.pt_code) and (found.llt_name == candidate.llt_name) and (found.pt_name == candidate.pt_name))
                if not row['current_parent_verified'] or not reference_complete:
                    reasons.append('complete_current_llt_reference_evidence_missing')
                if found:
                    row['llt_score'] = found.score
                    rival_scores = [e.score for e in entries if e.candidate_id != found.candidate_id]
                    row['llt_margin'] = round(found.score - max(rival_scores), 6) if rival_scores else 1.0
                    if not entries or entries[0].candidate_id != match.candidate_id:
                        reasons.append('reviewer_llt_reranker_disagreement')
                    supplemental = tuple((HumanReviewCandidate(candidate_id=e.candidate_id, llt_code=e.llt_code, llt_name=e.llt_name, pt_code=e.pt_code, pt_name=e.pt_name, is_current=True, was_retrieved=e.was_retrieved, retrieval_source_count=e.retrieval_source_count, reviewer_selected=e.candidate_id == match.candidate_id, reranker_shortlisted=True) for e in entries))
                    evidence = prepare_policy_evidence(reviewer_input, (match.candidate_id,), supplemental=supplemental)
                    options = {o.candidate.candidate_id: o for o in evidence.options}
                    options = {e.candidate_id: replace(options[e.candidate_id], score=e.score, conflicts=tuple(explicit_clinical_conflicts(text, e.llt_name) + explicit_clinical_conflicts(text, e.pt_name))) for e in entries if e.candidate_id in options}
                    if match.candidate_id in options:
                        ambiguities = [a for i, o in options.items() if i != match.candidate_id if (a := _ambiguity(text, options[match.candidate_id], o, config))]
                if row['llt_score'] is None or row['llt_score'] < config.llt_minimum_score:
                    reasons.append('llt_score_below_threshold_or_missing')
                if row['llt_margin'] is None or row['llt_margin'] < config.llt_minimum_margin:
                    reasons.append('llt_margin_below_threshold_or_missing')
                if row['pt_score'] is None or row['pt_score'] < max(0.9, config.pt_minimum_score):
                    reasons.append('llt_compatibility_stage_pt_floor_not_met')
        if conflicts:
            reasons.append('explicit_clinical_conflict')
        reasons.extend(automatic_mapping_uncertainties(text))
        if ambiguities:
            reasons.append('same_pt_llt_specificity_ambiguous')
        row.update(conflicts=conflicts, ambiguity=ambiguities, reasons=sorted(set(reasons)), accepted=not reasons)
        audits.append(row)
    if any((not a['accepted'] for a in audits)):
        global_reasons.append('one_or_more_components_failed_acceptance')
    return {'version': COMPOUND_POLICY_VERSION, 'authority': 'deterministic_component_acceptance', 'term_id': reviewer_input.term_id, 'input_text': reviewer_input.normalized_term, 'selected_candidate_ids': list(decision.selected_candidate_ids), 'decomposition': decomposition, 'component_mappings': [{'component_id': m.component_id, 'candidate_id': m.candidate_id, 'pt_code': m.pt_code} for m in matches], 'components': audits, 'accepted': bool(audits) and (not global_reasons), 'reasons': global_reasons or ['all_asserted_components_passed'], 'configuration': config.to_dict(), 'llt_compatibility_pt_floor': max(0.9, config.pt_minimum_score), 'reference_scope': 'full_current_pt_reference' if reference_complete else 'saved_available_evidence_only', 'partial_automatic_outcome_supported': False}
