"""Offline, failure-inclusive retrieval availability evaluation; never runs the pipeline.

Two passes: freeze label-free observed pools, then join the existing eligibility/gold.
Saved training and test records are pooled, not deduplicated across partitions.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from runtime_resources import ROOT, bootstrap, configuration, load, resource, save_new, sha

VERSION = 'trace.retrieval-evaluation/2'
SOURCES = ('fuzzy', 'semantic', 'umls')
CHECKPOINTS = (
    'initial_fuzzy', 'initial_semantic', 'initial_umls', 'initial_combined',
    'first_reviewer', 'after_evidence', 'before_completion', 'after_completion',
)
RANKED = {'initial_fuzzy', 'initial_semantic'}
IDENTITY = ('candidate_id', 'llt_code', 'llt_name', 'pt_code', 'pt_name', 'is_current')
METHODS = {
    'population': 'All prepared training and test rows; partition IDs retained; no cross-partition deduplication or annotation merging. Frequency is not a weight.',
    'exclusion': 'Initial exact_result from saved execution; only missing exact observations use the unchanged normalization/exact matcher, without labels.',
    'eligibility': 'evaluation.dataset primary_pt_eligible / primary_llt_eligible after initial-exact exclusions. No LLT parent fallback or lexical-equivalence credit.',
    'ordering': 'Fuzzy/semantic: recorded source ranks, LLT duplicates removed at first occurrence; PT rank is first occurrence of each distinct verified parent. No cross-source score fusion. Other checkpoints are unordered availability sets, even if subset/within-PT ranks are saved.',
    'cutoffs': 'Recall@5/@10 use up to k available distinct codes, without fetching extra candidates or changing limits.',
    'recall': 'Macro average of |reference intersection pool| / |reference|; complete-set rate requires all reference codes, permits extras.',
    'mrr': 'Only eligible rows with exactly one distinct reference code at that level. Absent reference = zero. No reciprocal-rank average for unordered pools.',
    'missing': 'All eligible failures/missing observations stay in primary denominators. Only positively observed verified codes earn credit; missing pools contribute zero. Partial pools give observed lower bounds, not an assertion of empty/complete retrieval.',
    'rank_gaps': 'Missing or invalid source ranks contribute zero in ranked-method primary metrics and are counted separately; unordered-checkpoint rank metrics are null.',
    'later': 'Cumulative observed availability: carry prior pools; read validated recorded stage histories, all reference-ranking inventories, and completion attempt inputs/outputs when present. Separate evidence retrieval, upstream expansion, and completion. Legacy final snapshots alone do not reconstruct discarded attempts.',
    'attribution_gaps': 'Set differences are newly observed availability only. They are confirmed stage additions only when the preceding checkpoint is fully observed; otherwise prior availability is unknown, not absent.',
    'interpretation': 'Candidate availability is not accepted mapping accuracy or automatic coverage. These are source-pool diagnostics of the same combined executions, not independent single-method pipeline runs.',
}


def json_line(handle, value):
    handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False) + '\n')


def lines(path):
    with Path(path).open(encoding='utf-8') as handle:
        for line in handle:
            yield json.loads(line)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def runtime_rows(partition, terms):
    """Read only runtime columns; gold/eligibility are not inspected in this pass."""
    prefix = 'TAC_TEST_EVAL_' if partition == 'test' else 'TAC_EVAL_'
    with Path(terms).open(encoding='utf-8-sig', newline='') as handle:
        return [{'id': prefix + f'{i:07d}', 'text': r['Term'].strip(), 'frequency': int(r['Frequency'])}
                for i, r in enumerate(csv.DictReader(handle), 1)]


class EvidenceReader:
    """Hash each read artifact once; never rewrite a source artifact."""
    def __init__(self, directory):
        self.directory = Path(directory).resolve()
        self.hashes = {}
        self.findings = []

    def read_bytes(self, relative):
        path = (self.directory / relative).resolve()
        if not path.is_relative_to(self.directory):
            raise ValueError('Unsafe evidence path')
        if not path.is_file():
            return None
        data = path.read_bytes()
        key = str(path.relative_to(self.directory))
        bound = hashlib.sha256(data).hexdigest()
        if key in self.hashes and self.hashes[key] != bound:
            raise ValueError('Evidence changed after first read: ' + key)
        self.hashes[key] = bound
        return data

    def read(self, relative):
        data = self.read_bytes(relative)
        if data is None:
            return None
        try:
            return json.loads(data.decode('utf-8-sig'))
        except (ValueError, UnicodeError) as exc:
            self.findings.append({'path': relative, 'kind': 'corrupt_json', 'detail': str(exc)})
            return None

    def verify_unchanged(self):
        return [name for name, bound in self.hashes.items()
                if not (self.directory / name).is_file() or sha(self.directory / name) != bound]


def reviewer_payload(request):
    """Extract actual delimited evidence, not an inferred prompt reconstruction."""
    for message in request.get('kwargs', {}).get('messages', []):
        content = message.get('content', '')
        marker = 'BEGIN_REVIEWER_EVIDENCE\n'
        if message.get('role') == 'user' and marker in content:
            return json.loads(content.split(marker, 1)[1].split('\nEND_REVIEWER_EVIDENCE', 1)[0])
    return None


def index_requests(reader, rows):
    """Only small locations/hashes in memory; request bodies stay on disk."""
    by_id = {r['id']: r for r in rows}
    index = defaultdict(list)
    for path in sorted((reader.directory / 'calls').glob('*_request.json')):
        rel = str(path.relative_to(reader.directory))
        request = reader.read(rel)
        if request is None:
            continue
        try:
            payload = reviewer_payload(request)
            if payload is None:
                continue
            term = payload['term']
            identifier = request['id']
            if identifier not in by_id:
                continue
            if term['term_id'] != identifier or term['original_term'] != by_id[identifier]['text']:
                raise ValueError('Reviewer request/input identity mismatch')
            index[identifier].append({'path': rel, 'round': int(term['review_round']),
                                     'payload_sha256': digest(payload)})
        except (KeyError, TypeError, ValueError) as exc:
            reader.findings.append({'path': rel, 'kind': 'invalid_reviewer_request', 'detail': str(exc)})
    return index


def request_candidates(payload):
    result = []
    for group in payload['pt_candidate_groups']:
        for candidate in group['llt_candidates']:
            result.append({**candidate, 'pt_code': group['pt_code'], 'pt_name': group['pt_name']})
    return result


def observed_scores(candidate):
    """Keep recorded rank/score scopes without treating them as a unified ranking."""
    keys = ('rank', 'ranking_score', 'fuzzy_score', 'cosine_similarity', 'reranker_rank',
            'reranker_score', 'score', 'direct_pt_match_score', 'introduced_by_pt_expansion',
            'introduction_query_variant_ids', 'conflict_flags', 'retrieval_conflict',
            'mapping_method', 'umls_cui', 'umls_source_code', 'umls_term_type',
            'fuzzy_query_term', 'semantic_query_term', 'umls_query_term')
    result = {k: candidate[k] for k in keys if k in candidate}
    for source in SOURCES:
        evidence = candidate.get(source + '_evidence')
        if evidence:
            result[source + '_evidence'] = {k: evidence[k] for k in keys if k in evidence}
    return result


class Candidates:
    def __init__(self, reference):
        self.reference = reference
        self.registry = {}
        self.invalid = []

    def add(self, candidates, evidence):
        codes = []
        for position, candidate in enumerate(candidates):
            ref = {**evidence, 'position': position}
            if not isinstance(candidate, dict):
                self.invalid.append({'evidence': ref, 'reason': 'invalid_candidate_structure'})
                continue
            code = str(candidate.get('llt_code') or str(candidate.get('candidate_id', '')).removeprefix('LLT:'))
            identity = self.reference.identity(code)
            reasons = []
            if not identity.get('current'):
                reasons.append('missing_or_noncurrent_identity')
            if candidate.get('candidate_id', 'LLT:' + code) != 'LLT:' + code:
                reasons.append('candidate_id_mismatch')
            for field in ('llt_name', 'pt_code', 'pt_name'):
                if field in candidate and str(candidate[field]) != str(identity.get(field)):
                    reasons.append(field + '_mismatch')
            if candidate.get('is_current', True) is not True:
                reasons.append('noncurrent_candidate_claim')
            for source in SOURCES:
                nested = candidate.get(source + '_evidence')
                if nested and nested.get('candidate_id', 'LLT:' + code) != 'LLT:' + code:
                    reasons.append('nested_evidence_identity_mismatch')
            if reasons:
                self.invalid.append({'evidence': ref, 'recorded_candidate': {k: candidate.get(k) for k in IDENTITY}, 'reasons': reasons})
                continue
            entry = self.registry.setdefault(code, {
                **{k: identity[k] for k in ('llt_code', 'llt_name', 'pt_code', 'pt_name')},
                'candidate_id': 'LLT:' + code, 'identity_evidence': identity['identity_evidence'],
                'observations': [],
            })
            entry['observations'].append({**ref, 'recorded_scores_and_provenance': observed_scores(candidate)})
            if code not in codes:
                codes.append(code)
        return codes


def pool(codes=(), *, status='observed', ranked=False, metadata=None, issues=()):
    return {'llt_codes': list(dict.fromkeys(codes)), 'evidence_status': status,
            'ranked': ranked, 'metadata': metadata or {}, 'issues': list(issues)}


def union_pool(pools, **kwargs):
    codes = list(dict.fromkeys(c for p in pools for c in p['llt_codes']))
    incomplete = any(p['evidence_status'] != 'observed' for p in pools)
    status = 'partial' if incomplete and codes else 'missing' if incomplete else 'observed'
    return pool(codes, status=kwargs.pop('status', status), **kwargs)


def source_pool(result, source, registry, evidence):
    if not isinstance(result, dict) or not isinstance(result.get('candidates'), list):
        return pool(status='missing', ranked=source != 'umls', issues=['source_result_not_recorded'])
    meta = {k: v for k, v in result.items() if k != 'candidates'}
    before = len(registry.invalid)
    candidates = result['candidates']
    ranked = source != 'umls'
    issues = []
    if ranked:
        ranks = [c.get('rank') for c in candidates]
        valid_ranks = all(type(r) is int and r >= 1 for r in ranks) and len(set(ranks)) == len(ranks)
        if valid_ranks:
            candidates = sorted(candidates, key=lambda c: c['rank'])
        else:
            ranked = False
            issues.append('recorded_source_ranks_missing_or_conflicting')
    codes = registry.add(candidates, evidence)
    failed = result.get('lookup_status') == 'lookup_failed' or bool(result.get('error_message'))
    invalid = len(registry.invalid) != before
    # Invalid identities must not promote following candidates into unsupported ranks.
    if invalid:
        ranked = False
        issues.append('invalid_candidate_identity_withheld')
    status = 'partial' if (failed or invalid) and codes else 'failed' if failed else 'partial' if invalid else 'observed'
    return pool(codes, status=status, ranked=ranked, metadata=meta, issues=issues)


def gate_candidates(gate, registry, evidence):
    """Only explicitly persisted candidate inventories, never parent expansion here."""
    codes = []
    if not isinstance(gate, dict):
        return codes
    def add(values, path):
        if isinstance(values, dict):
            values = list(values.values())
        if isinstance(values, list):
            codes.extend(registry.add(values, {**evidence, 'pointer': evidence['pointer'] + '/' + path}))
    for field in ('ranked_candidates', 'human_review_candidates'):
        add(gate.get(field), field)
    cov = gate.get('coverage_acceptance') or {}
    add(cov.get('assessment_scope_v1_7_1', {}).get('universe'), 'coverage_acceptance/assessment_scope_v1_7_1/universe')
    add(cov.get('joint_ranking', {}).get('candidates'), 'coverage_acceptance/joint_ranking/candidates')
    # Accepted completion authorities persist their exact compared case inventory.
    for name, value in cov.items():
        if not isinstance(value, dict):
            continue
        authority = value.get('authority')
        if not isinstance(authority, dict):
            continue
        case = authority.get('request', {}).get('case', {})
        add(case.get('candidates'), 'coverage_acceptance/' + name + '/authority/request/case/candidates')
    return list(dict.fromkeys(codes))


def extract(row, record, request_locations, reader, reference, exact_fallback):
    """Gold-free extraction. No candidate generation, re-ranking, or acceptance."""
    result = {'id': row['id'], 'input': row['text'], 'frequency': row['frequency'],
              'source_record': 'records/' + row['id'] + '.json', 'execution_status': 'missing_record',
              'issues': [], 'reviewer_requests': request_locations}
    if record and (record.get('id') != row['id'] or record.get('text') != row['text']):
        raise ValueError('Saved record/input mismatch: ' + row['id'])
    state = dict((record or {}).get('upstream_state') or {})
    from retrieval_trace_evidence import load_history
    history_trace = load_history(reader, row['id'])
    result['recorded_history'] = history_trace['audit']
    for key, observed in history_trace['initial'].items():
        if key not in state:
            state[key] = observed['value']
        elif state[key] != observed['value']:
            raise ValueError('Trace and saved initial state disagree: ' + row['id'] + ':' + key)
    if record:
        result['execution_status'] = record.get('status', 'unknown')
        result['policy_replay_failure'] = bool(record.get('policy_replay_failure'))
    exact = state.get('exact_result')
    if not isinstance(exact, dict) or type(exact.get('found')) is not bool:
        exact = exact_fallback(row)
        result['exact_provenance'] = 'unchanged_local_exact_matcher_recomputed_without_labels'
    else:
        result['exact_provenance'] = 'observed_trace_initial_exact_result' if 'exact_result' in history_trace['initial'] else 'observed_upstream_initial_exact_result'
    if exact['found']:
        concept = exact.get('concept') or {}
        verified = reference.identity(str(concept.get('llt_code', '')))
        if not verified.get('current') or any(str(concept.get(k)) != str(verified.get(k)) for k in ('llt_name', 'pt_code', 'pt_name')):
            raise ValueError('Exact exclusion has invalid terminology identity: ' + row['id'])
    result.update(exact_excluded=bool(exact['found']), exact_result=exact)
    if exact['found']:
        result.update(checkpoints={}, candidates={}, invalid_candidates=[])
        return result
    registry = Candidates(reference)
    checkpoints = {}
    for source in SOURCES:
        name = 'initial_' + source
        checkpoints[name] = source_pool(state.get(source + '_result'), source, registry,
            {**history_trace['initial'].get(source + '_result', {}).get('evidence',
                {'artifact': result['source_record'], 'pointer': '/upstream_state/' + source + '_result'}),
             'stage': name, 'candidate_pointer_suffix': '/candidates'})
    checkpoints['initial_combined'] = union_pool([checkpoints['initial_' + s] for s in SOURCES],
                                                metadata={'ordering': 'Unordered LLT union of the three initial source results.'})
    reviewer_pools = []
    first_pools = []
    payload_hashes = defaultdict(set)
    for location in request_locations:
        request = reader.read(location['path'])
        if request is None:
            result['issues'].append('missing_or_corrupt_reviewer_request:' + location['path'])
            continue
        payload = reviewer_payload(request)
        if digest(payload) != location['payload_sha256']:
            raise ValueError('Reviewer request changed during extraction')
        before = len(registry.invalid)
        codes = registry.add(request_candidates(payload), {'artifact': location['path'], 'pointer': '/kwargs/messages', 'embedded_json_marker': 'BEGIN_REVIEWER_EVIDENCE', 'embedded_pointer': '/pt_candidate_groups', 'stage': 'reviewer_round_' + str(location['round'])})
        observed = pool(codes, status='partial' if len(registry.invalid) != before else 'observed',
                        metadata={'round': location['round'], 'retrieval_provenance': payload.get('retrieval_provenance', [])})
        reviewer_pools.append(observed)
        payload_hashes[location['round']].add(digest(sorted(codes)))
        if location['round'] == 1:
            first_pools.append(observed)
    if first_pools:
        checkpoints['first_reviewer'] = first_pools[0]
        if len(payload_hashes[1]) != 1:
            checkpoints['first_reviewer']['issues'].append('candidate_pool_changed_between_round1_attempts; earliest_saved_attempt_used')
    else:
        # A saved ReviewerInput alone is not proof that a model request was issued.
        checkpoints['first_reviewer'] = pool(status='missing', issues=['no_recorded_round1_model_request'])
    evidence_pools = [checkpoints['initial_combined']] + reviewer_pools
    # Runtime history contains batches; older fixtures/exports may contain
    # individual execution entries. Preserve each actual request in either form.
    execution_history = [execution for entry in state.get('evidence_execution_history') or []
                         for execution in (entry['executions'] if 'executions' in entry else [entry])]
    batch = state.get('evidence_execution_batch') or {}
    history = {(x.get('review_round'), x.get('request', {}).get('request_id')): x for x in execution_history}
    execution_refs = {}
    for observation in history_trace['executions']:
        x = observation['value']
        key = (x.get('review_round'), x.get('request', {}).get('request_id'))
        if key in history and history[key] != x:
            raise ValueError('Trace and saved retrieval execution disagree')
        history[key] = x
        execution_refs[key] = observation['evidence']
    for x in batch.get('executions', []):
        history[(x.get('review_round'), x.get('request', {}).get('request_id'))] = x
    retrieval_failures = []
    for i, execution in enumerate(history.values()):
        req = execution.get('request', {})
        source = req.get('source')
        if source not in SOURCES:
            result['issues'].append('unknown_evidence_source')
            continue
        observed = source_pool(execution.get(source + '_result'), source, registry,
            {**execution_refs.get((execution.get('review_round'), req.get('request_id')),
                {'artifact': result['source_record'], 'pointer': '/upstream_state'}),
             'execution_locator': {'review_round': execution.get('review_round'), 'request_id': req.get('request_id'), 'containers': ['evidence_execution_history', 'evidence_execution_batch/executions']}, 'stage': 'additional_evidence', 'request': req})
        evidence_pools.append(observed)
        if execution.get('status') != 'completed':
            retrieval_failures.append(req.get('request_id'))
    late = union_pool(evidence_pools, metadata={'additional_executions': len(history), 'failed_requests': retrieval_failures})
    if not state.get('workflow_outcome'):
        late['evidence_status'] = 'partial' if late['llt_codes'] else 'missing'
        late['issues'].append('upstream_execution_not_fully_saved')
    if state.get('review_round', 1) > 1 and not history and not history_trace['audit']['complete']:
        late['evidence_status'] = 'partial'
        late['issues'].append('later_round_without_complete_retrieval_execution_history')
    checkpoints['after_evidence'] = late
    up_gate = state.get('llt_selection_gate_result')
    upstream_codes = gate_candidates(up_gate, registry, {'artifact': result['source_record'], 'pointer': '/upstream_state/llt_selection_gate_result', 'stage': 'upstream_reference_expansion'})
    traced_codes = {'upstream': [], 'completion': []}
    for observation in history_trace['gates']:
        traced_codes[observation['phase']].extend(gate_candidates(observation['value'], registry,
            {**observation['evidence'], 'stage': observation['phase'] + '_gate_history'}))
    for observation in history_trace['inventories']:
        traced_codes[observation['phase']].extend(registry.add(observation['candidates'],
            {**observation['evidence'], 'stage': observation['phase'] + '_reference_inventory'}))
    upstream_codes.extend(traced_codes['upstream'])
    checkpoints['before_completion'] = union_pool([late, pool(upstream_codes)],
        metadata={'ordering': 'Cumulative observed candidates plus saved upstream gate inventories; subset/within-PT ranks are not a total order.'})
    if not up_gate and not history_trace['audit']['complete']:
        checkpoints['before_completion']['evidence_status'] = 'partial' if late['llt_codes'] else 'missing'
        checkpoints['before_completion']['issues'].append('upstream_gate_not_saved')
    if state.get('review_round', 1) > 1 and not history_trace['audit']['complete']:
        checkpoints['before_completion']['evidence_status'] = 'partial'
        checkpoints['before_completion']['issues'].append('only_latest_upstream_gate_inventory_saved; earlier_round_expansions_not_reconstructed')
    final_codes = gate_candidates((record or {}).get('final_gate'), registry,
                                 {'artifact': result['source_record'], 'pointer': '/final_gate', 'stage': 'deterministic_completion'})
    final_codes.extend(traced_codes['completion'])
    final = union_pool([checkpoints['before_completion'], pool(final_codes)])
    upstream_selected = (state.get('workflow_outcome') or {}).get('selected_concepts', [])
    exhaustive = (history_trace['audit']['complete'] if history_trace['audit']['available'] else
                  bool(upstream_selected and not (record or {}).get('policy_replay_failure')))
    if not exhaustive or (record or {}).get('policy_replay_failure'):
        final['evidence_status'] = 'partial' if final['llt_codes'] else 'missing'
        final['issues'].append('saved_final_inventories_only; discarded_or_failed_completion_attempt_pools_not_reconstructed')
    final['metadata']['completion_history_exhaustive'] = exhaustive
    checkpoints['after_completion'] = final
    for name, previous in (('after_evidence', 'initial_combined'), ('before_completion', 'after_evidence'), ('after_completion', 'before_completion')):
        new = sorted(set(checkpoints[name]['llt_codes']) - set(checkpoints[previous]['llt_codes']))
        known_before = checkpoints[previous]['evidence_status'] == 'observed'
        checkpoints[name]['newly_observed_llt_codes'] = new
        checkpoints[name]['new_llt_codes'] = new if known_before else []
        checkpoints[name]['prior_availability_unknown_llt_codes'] = [] if known_before else new
        checkpoints[name]['addition_baseline'] = previous
    # Withhold suspect identities and disclose the impact on every pool containing them.
    if registry.invalid:
        result['issues'].append('invalid_candidate_observations_withheld')
        for name in ('before_completion', 'after_completion'):
            checkpoints[name]['evidence_status'] = 'partial'
    if history_trace['audit']['findings']:
        for name in ('before_completion', 'after_completion'):
            checkpoints[name]['evidence_status'] = 'partial'
            checkpoints[name]['issues'].append('recorded_history_integrity_findings')
    result.update(checkpoints=checkpoints, candidates=registry.registry, invalid_candidates=registry.invalid,
                  initial_retrieval_failures=[s for s in SOURCES if checkpoints['initial_' + s]['evidence_status'] == 'failed' or checkpoints['initial_' + s]['metadata'].get('error_message')],
                  additional_retrieval_failures=retrieval_failures)
    return result


def contribution(reference_codes, candidate_codes, *, eligible=True, ranked=True, rank_metric_defined=True):
    """Pure strict code metrics; duplicate labels and candidates never add weight."""
    gold = set(str(c) for c in reference_codes)
    ordered = list(dict.fromkeys(str(c) for c in candidate_codes))
    if not eligible:
        return None
    if not gold:
        raise ValueError('Eligible input has no reference codes')
    hit = gold & set(ordered)
    singleton = len(gold) == 1
    mrr = (1.0 / (ordered.index(next(iter(gold))) + 1) if hit else 0.0) if singleton else None
    return {'required': len(gold), 'retrieved_required': len(hit), 'full_pool_recall': len(hit) / len(gold),
            'complete_set': int(gold <= set(ordered)),
            'recall_at_5': len(gold & set(ordered[:5])) / len(gold) if ranked and rank_metric_defined else 0.0 if rank_metric_defined else None,
            'recall_at_10': len(gold & set(ordered[:10])) / len(gold) if ranked and rank_metric_defined else 0.0 if rank_metric_defined else None,
            'reciprocal_rank': mrr if ranked and rank_metric_defined else 0.0 if rank_metric_defined and singleton else None,
            'mrr_eligible': singleton, 'rank_observed': ranked and rank_metric_defined}


def level_codes(checkpoint, registry, level):
    if level == 'LLT':
        return checkpoint['llt_codes']
    return list(dict.fromkeys(registry[c]['pt_code'] for c in checkpoint['llt_codes']))


class Metrics:
    def __init__(self, ranked):
        self.ranked = ranked
        self.counts = Counter()
        self.sums = Counter()

    def add(self, item, checkpoint):
        if item is None:
            return
        self.counts['eligible'] += 1
        self.counts[checkpoint['evidence_status']] += 1
        self.counts['observed_empty'] += checkpoint['evidence_status'] == 'observed' and not checkpoint['llt_codes']
        self.counts['mrr_eligible'] += item['mrr_eligible']
        self.counts['rank_unavailable'] += self.ranked and not item['rank_observed']
        for key in ('full_pool_recall', 'complete_set', 'recall_at_5', 'recall_at_10', 'reciprocal_rank'):
            if item[key] is not None:
                self.sums[key] += item[key]

    def summary(self):
        n = self.counts['eligible']
        single = self.counts['mrr_eligible']
        mean = lambda key: self.sums[key] / n if n else None
        return {'eligible': n, 'recall_at_5': mean('recall_at_5') if self.ranked else None,
                'recall_at_10': mean('recall_at_10') if self.ranked else None,
                'full_pool_recall': mean('full_pool_recall'), 'complete_set_retrieval_rate': mean('complete_set'),
                'mrr': self.sums['reciprocal_rank'] / single if single and self.ranked else None,
                'mrr_denominator': single, 'rank_metrics_defined': self.ranked, 'evidence_counts': dict(self.counts),
                'full_pool_interpretation': 'observed lower bound where execution evidence is incomplete'}


def format_summary(summary):
    p = summary['population']
    out = ['POOLED RETRIEVAL EVALUATION (strict codes; not mapping accuracy)',
           f"Total records: {p['total']} | Initial exact exclusions: {p['exact_excluded']} | Remaining: {p['nonexact']}",
           f"Eligible non-exact: PT {p['PT_eligible']} | LLT {p['LLT_eligible']}",
           f"Missing records: {p['missing_records']} | Execution errors: {p['execution_errors']} | Inputs with recorded retrieval failure: {p['retrieval_failure_inputs']}", '']
    pct = lambda x: 'N/A' if x is None else f'{100*x:.2f}%'
    for level in ('PT', 'LLT'):
        out += [level, f"{'Checkpoint':<22} {'N':>6} {'R@5':>8} {'R@10':>8} {'Full':>8} {'All codes':>10} {'MRR':>8} {'MRR N':>7} {'Gaps':>6}"]
        for name in CHECKPOINTS:
            m = summary['metrics'][name][level]
            gaps = m['eligible'] - m['evidence_counts'].get('observed', 0)
            out.append(f"{name:<22} {m['eligible']:>6} {pct(m['recall_at_5']):>8} {pct(m['recall_at_10']):>8} {pct(m['full_pool_recall']):>8} {pct(m['complete_set_retrieval_rate']):>10} {('N/A' if m['mrr'] is None else format(m['mrr'], '.4f')):>8} {m['mrr_denominator']:>7} {gaps:>6}")
        out.append('')
    out += ['N/A: no meaningful pool ranking or no eligible denominator.',
            'Gaps remain in denominators; unobserved codes earn no credit. Incomplete pools are lower bounds.',
            'Later rows are combined-workflow availability, not independent source runs or accepted outputs.',
            'No cross-partition deduplication. Read summary.json for evidence counts and limitations.']
    return '\n'.join(out) + '\n'


def score_frozen(availability, datasets, reference, output):
    from evaluation import dataset
    gold_rows = {}
    for partition, paths in datasets.items():
        for identifier, row in dataset(partition, paths['terms'], paths['cases']).items():
            gold_rows[(partition, identifier)] = row
    metrics = {name: {level: Metrics(name in RANKED) for level in ('PT', 'LLT')} for name in CHECKPOINTS}
    population = Counter({k: 0 for k in ('total', 'exact_excluded', 'nonexact', 'PT_eligible', 'LLT_eligible', 'missing_records', 'execution_errors', 'retrieval_failure_inputs', 'policy_errors', 'initial_retrieval_evidence_missing_inputs', 'invalid_candidate_inputs')})
    additions = {name: Counter() for name in ('after_evidence', 'before_completion', 'after_completion')}
    with Path(output).open('x', encoding='utf-8') as handle:
        for row in lines(availability):
            labelled = gold_rows[(row['partition'], row['id'])]
            if labelled['text'] != row['input']:
                raise ValueError('Annotation join changed original input')
            g = labelled['gold']
            population['total'] += 1
            population['exact_excluded'] += row['exact_excluded']
            population['missing_records'] += row['execution_status'] == 'missing_record'
            population['execution_errors'] += row['execution_status'] not in ('completed', 'missing_record')
            population['policy_errors'] += bool(row.get('policy_replay_failure'))
            population['retrieval_failure_inputs'] += bool(row.get('initial_retrieval_failures') or row.get('additional_retrieval_failures'))
            population['invalid_candidate_inputs'] += bool(row['invalid_candidates'])
            observed_history = row.get('recorded_history', {})
            population['recorded_histories_available'] += bool(observed_history.get('available'))
            population['recorded_histories_complete'] += bool(observed_history.get('complete'))
            population['initial_retrieval_evidence_missing_inputs'] += any(row.get('checkpoints', {}).get('initial_' + s, {}).get('evidence_status') == 'missing' for s in SOURCES)
            scored = {'partition': row['partition'], 'id': row['id'], 'input': row['input'],
                      'exact_excluded': row['exact_excluded'], 'gold': g, 'availability_reference': {'path': 'availability.jsonl', 'line': population['total']},
                      'reference_identity_findings': [], 'checkpoints': {}}
            if not row['exact_excluded']:
                population['nonexact'] += 1
                for level in ('PT', 'LLT'):
                    population[level + '_eligible'] += bool(g['primary_' + level.lower() + '_eligible'])
                for code in g['llt_codes']:
                    if not reference.identity(code).get('current'):
                        scored['reference_identity_findings'].append({'level': 'LLT', 'code': code, 'reason': 'missing_or_noncurrent_reference_label; denominator_retained'})
                for code in g['pt_codes']:
                    if code not in reference.groups:
                        scored['reference_identity_findings'].append({'level': 'PT', 'code': code, 'reason': 'parent_not_resolved_in_current_reference; denominator_retained'})
                for name in CHECKPOINTS:
                    checkpoint = row['checkpoints'][name]
                    values = {}
                    for level in ('PT', 'LLT'):
                        codes = level_codes(checkpoint, row['candidates'], level)
                        item = contribution(g[level.lower() + '_codes'], codes,
                            eligible=g['primary_' + level.lower() + '_eligible'], ranked=checkpoint['ranked'], rank_metric_defined=name in RANKED)
                        metrics[name][level].add(item, checkpoint)
                        values[level] = {'candidate_codes': codes, 'ranks': {code: i for i, code in enumerate(codes, 1)} if checkpoint['ranked'] else None, 'contribution': item}
                    scored['checkpoints'][name] = {'evidence_status': checkpoint['evidence_status'], **values}
                    if name in additions:
                        a = additions[name]
                        a['inputs_with_confirmed_stage_additions'] += bool(checkpoint['new_llt_codes'])
                        a['new_LLT_candidate_observations'] += len(checkpoint['new_llt_codes'])
                        a['newly_observed_LLTs_with_unknown_prior_availability'] += len(checkpoint['prior_availability_unknown_llt_codes'])
                        prior = checkpoint['addition_baseline']
                        for level in ('PT', 'LLT'):
                            before = scored['checkpoints'][prior][level]['contribution']
                            after = values[level]['contribution']
                            if after:
                                confirmed = row['checkpoints'][prior]['evidence_status'] == 'observed'
                                prefix = level + ('_confirmed_stage' if confirmed else '_newly_observed_unknown_prior')
                                a[prefix + '_additional_required_codes'] += after['retrieved_required'] - before['retrieved_required']
                                a[prefix + '_new_complete_reference_sets'] += after['complete_set'] - before['complete_set']
            json_line(handle, scored)
    return {'version': VERSION, 'population': dict(population),
            'metrics': {name: {level: metric.summary() for level, metric in levels.items()} for name, levels in metrics.items()},
            'stage_additions': {name: dict(value) for name, value in additions.items()}, 'methods': METHODS}


def run_evaluation(training_directory, test_directory, directory):
    bootstrap()
    directory = Path(directory).resolve()
    runs = ROOT / 'runs'
    if directory == runs or not directory.is_relative_to(runs):
        raise ValueError('Use a fresh child directory under TRACE runs/')
    if directory.exists():
        raise FileExistsError('Fresh retrieval evaluation directory required')
    readers = {p: EvidenceReader(d) for p, d in (('training', training_directory), ('test', test_directory))}
    if any(directory.is_relative_to(r.directory) or r.directory.is_relative_to(directory) for r in readers.values()):
        raise ValueError('Evaluation output must be separate from source runs')
    cfg = configuration()
    from lexical_equivalence_evaluation import Reference
    reference = Reference(resource('meddra_exact_reference.csv'), resource('llt.asc'), resource('pt.asc'))
    # Binding verification does not require importing any model or executing policy.
    locked = load(ROOT / 'resources.lock.json')
    for name, expected in locked.items():
        if sha(resource(name)) != expected:
            raise ValueError('Changed frozen reference/lexical resource: ' + name)
    source_hashes = {p.name: sha(p) for p in ROOT.glob('*.py')}
    datasets, inputs, contracts = {}, {}, {}
    for partition, reader in readers.items():
        contract = reader.read('contract.json')
        if not contract or contract.get('dataset_binding', {}).get('partition') != partition:
            raise ValueError('Missing/wrong source run contract: ' + partition)
        binding = contract['dataset_binding']
        datasets[partition] = {k: str(Path(cfg[partition + '_' + k])) for k in ('terms', 'cases')}
        for name, path in datasets[partition].items():
            if sha(path) != binding[name + '_sha256']:
                raise ValueError('Dataset differs from saved run: ' + partition + ':' + name)
        for name in ('meddra_exact_reference.csv', 'llt.asc', 'pt.asc'):
            if contract['implementation'].get('resource:' + name) != sha(resource(name)):
                raise ValueError('Saved terminology binding differs: ' + name)
        inputs[partition] = runtime_rows(partition, datasets[partition]['terms'])
        selected = {r['id']: r for r in contract['inputs']}
        if len(selected) != len(contract['inputs']):
            raise ValueError('Duplicate source-run input IDs')
        actual = {r['id']: r for r in inputs[partition]}
        if any(actual.get(k) != value for k, value in selected.items()):
            raise ValueError('Source contract inputs differ from dataset')
        contracts[partition] = {'contract_sha256': reader.hashes['contract.json'], 'dataset_binding': binding,
            'source_directory': str(reader.directory), 'saved_settings': contract.get('settings'),
            'historical_implementation': contract['implementation'],
            'changed_historical_source_files': [k for k, v in contract['implementation'].items() if k.endswith('.py') and source_hashes.get(k) != v],
            'new_source_files_since_run': sorted(set(source_hashes) - set(contract['implementation']))}
    directory.mkdir(parents=True, exist_ok=False)
    save_new(directory / 'methods.json', {'version': VERSION, 'methods': METHODS, 'scoring_started': False})
    exact_node = None
    def exact_fallback(row):
        nonlocal exact_node
        from exact_matching import create_exact_matching_node
        from normalization import normalize_term
        if exact_node is None:
            exact_node = create_exact_matching_node(resource('meddra_exact_reference.csv'))
        return exact_node({'normalization': normalize_term(row['text'])})['exact_result'].model_dump(mode='json')
    with (directory / 'availability.jsonl').open('x', encoding='utf-8') as handle:
        for partition, reader in readers.items():
            allowed = {r['id'] for r in load(reader.directory / 'contract.json')['inputs']}
            requests = index_requests(reader, [r for r in inputs[partition] if r['id'] in allowed])
            for n, row in enumerate(inputs[partition], 1):
                record = reader.read('records/' + row['id'] + '.json') if row['id'] in allowed else None
                item = extract(row, record, requests.get(row['id'], []), reader, reference, exact_fallback)
                json_line(handle, {'partition': partition, **item})
                if n % 200 == 0 or n == len(inputs[partition]):
                    print(f'Observed {partition} {n}/{len(inputs[partition])}', flush=True)
    frozen_hash = sha(directory / 'availability.jsonl')
    save_new(directory / 'availability_binding.json', {'version': VERSION, 'availability_sha256': frozen_hash,
        'labels_joined': False, 'contracts': contracts, 'reference_metadata': reference.metadata,
        'implementation': source_hashes, 'resource_hashes': locked})
    print('Label-free availability frozen; joining existing annotations for scoring.', flush=True)
    summary = score_frozen(directory / 'availability.jsonl', datasets, reference, directory / 'per_input.jsonl')
    changed = {p: r.verify_unchanged() for p, r in readers.items()}
    source_changes = [name for name, bound in source_hashes.items() if sha(ROOT / name) != bound]
    if any(changed.values()) or source_changes or sha(directory / 'availability.jsonl') != frozen_hash:
        raise RuntimeError('Evidence or implementation changed during evaluation')
    summary['preservation'] = {'all_used_artifacts_unchanged': True, 'artifacts_checked': sum(len(r.hashes) for r in readers.values()),
                               'existing_python_files_unchanged_during_run': True, 'availability_sha256': frozen_hash}
    summary['source_findings'] = {p: r.findings for p, r in readers.items()}
    summary['evidence_manifest'] = 'source_artifacts.json'
    save_new(directory / 'source_artifacts.json', {p: {'directory': str(r.directory), 'sha256': r.hashes} for p, r in readers.items()})
    save_new(directory / 'summary.json', summary)
    text = format_summary(summary)
    with (directory / 'summary.txt').open('x', encoding='utf-8') as handle:
        handle.write(text)
    print(text)
    print('Saved:', directory)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    score = sub.add_parser('score', help='OFFLINE: use saved executions; no model or retrieval calls')
    score.add_argument('--training-directory', type=Path, default=ROOT / 'runs/training_full_1')
    score.add_argument('--test-directory', type=Path, default=ROOT / 'runs/test_full_1')
    score.add_argument('--directory', type=Path, required=True, help='Fresh results directory under runs/')
    summary = sub.add_parser('summary')
    summary.add_argument('--directory', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'summary':
        print(format_summary(load(args.directory / 'summary.json')))
        return 0
    result = run_evaluation(args.training_directory, args.test_directory, args.directory)
    return int(bool(result['population']['missing_records'] or result['population']['execution_errors']))


if __name__ == '__main__':
    raise SystemExit(main())
