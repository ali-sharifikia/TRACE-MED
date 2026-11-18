"""Strict, independently lexical and missing-LLT-only parent-fallback scoring."""
import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from runtime_resources import bootstrap, configuration, resource, save_new, sha, load

def boolean(value):
    s = str(value).strip().lower()
    if s not in ('true', 'false', '1', '0'):
        raise ValueError('Invalid manifest boolean: ' + s)
    return s in ('true', '1')

def key(s):
    return ' '.join(s.casefold().split())

def dataset(partition, terms, cases):
    sources = defaultdict(list)
    with Path(cases).open(encoding='utf-8-sig', newline='') as f:
        for r in csv.DictReader(f):
            sources[key(r['input_term'])].append(r)
    rows = {}
    with Path(terms).open(encoding='utf-8-sig', newline='') as f:
        for n, r in enumerate(csv.DictReader(f), 1):
            id = ('TAC_TEST_EVAL_' if partition == 'test' else 'TAC_EVAL_') + f'{n:07d}'
            g = {k: json.loads(r['stable_current_gold_' + k]) for k in ('pt_codes', 'llt_codes')}
            g.update({k: boolean(r[k]) for k in ('primary_pt_eligible', 'primary_llt_eligible', 'manual_reconciliation_required', 'llt_context_ambiguous', 'pt_context_ambiguous')})
            for level in ('pt', 'llt'):
                if g['primary_' + level + '_eligible'] and (not g[level + '_codes']):
                    raise ValueError('Eligible row lacks gold: ' + id)
            src = sources[key(r['Term'])]
            fallback = not g['primary_llt_eligible'] and g['primary_pt_eligible'] and bool(g['pt_codes']) and (not g['llt_codes']) and (not any((g[k] for k in ('manual_reconciliation_required', 'llt_context_ambiguous', 'pt_context_ambiguous')))) and bool(src) and (not any((json.loads(c[k]) for c in src for k in ('original_gold_llt_codes', 'original_gold_llt_names'))))
            rows[id] = {'id': id, 'text': r['Term'].strip(), 'frequency': int(r['Frequency']), 'gold': g, 'parent_fallback_eligible': bool(fallback)}
    return rows

def metric(eligible, automatic, correct):
    return {'eligible': eligible, 'automatic': automatic, 'correct': correct, 'disagreements': automatic - correct, 'coverage': automatic / eligible if eligible else None, 'accuracy': correct / automatic if automatic else None, 'remaining': eligible - automatic}

def score_rows(rows, outputs, reference):
    bootstrap()
    from equivalence_group import score_sets
    counters = {k: [0, 0, 0] for k in ('PT', 'LLT', 'LLT_lexical', 'complete', 'complete_lexical', 'LLT_gold_or_parent', 'LLT_supported')}
    audit = []
    complete = 0
    outcomes = {k: 0 for k in ('automatic', 'human_review', 'unsupported', 'execution_errors', 'unexecuted', 'unknown')}
    for row in rows:
        out = outputs.get(row['id'], {'status': 'unexecuted', 'prediction': {'pt': [], 'llt': []}})
        p = out['prediction']
        g = row['gold']
        ok = out['status'] == 'completed'
        pt_auto = ok and bool(p['pt'])
        llt_auto = ok and bool(p['llt'])
        complete += bool(pt_auto and llt_auto)
        outcome_status = (out.get('outcome') or {}).get('status')
        if not ok:
            disposition = 'unexecuted' if out['status'] == 'unexecuted' else 'execution_errors'
        elif outcome_status == 'no_supported_match':
            disposition = 'unsupported'
        elif outcome_status == 'human_review':
            disposition = 'human_review'
        elif pt_auto and llt_auto:
            disposition = 'automatic'
        else:
            # Old prediction-only exports cannot distinguish review from unsupported.
            disposition = 'unknown'
        outcomes[disposition] += 1
        pt_correct = bool(pt_auto and set(p['pt']) == set(g['pt_codes']))
        llt_correct = bool(llt_auto and set(p['llt']) == set(g['llt_codes']))
        detail = score_sets(p['llt'], g['llt_codes'], row['text'], reference) if llt_auto and g['primary_llt_eligible'] else None
        lex_correct = bool(detail and detail['equivalence'])
        fallback = False
        if row['parent_fallback_eligible'] and llt_auto:
            ids = [reference.identity(c) for c in p['llt']]
            fallback = bool(len(p['llt']) == len(set(p['llt'])) == len(set(g['pt_codes'])) and all((i.get('current') for i in ids)) and ({i['pt_code'] for i in ids} == set(g['pt_codes'])))
        values = {'PT': (g['primary_pt_eligible'], pt_auto, pt_correct), 'LLT': (g['primary_llt_eligible'], llt_auto, llt_correct), 'LLT_lexical': (g['primary_llt_eligible'], llt_auto, lex_correct), 'complete': (g['primary_llt_eligible'] and g['primary_pt_eligible'], pt_auto and llt_auto, pt_correct and llt_correct), 'complete_lexical': (g['primary_llt_eligible'] and g['primary_pt_eligible'], pt_auto and llt_auto, pt_correct and lex_correct), 'LLT_gold_or_parent': (g['primary_llt_eligible'] or row['parent_fallback_eligible'], llt_auto, llt_correct if g['primary_llt_eligible'] else fallback)}
        values['LLT_supported'] = (g['primary_llt_eligible'] or row['parent_fallback_eligible'], llt_auto,
                                   (llt_correct or lex_correct) if g['primary_llt_eligible'] else fallback)
        for name, (eligible, auto, correct) in values.items():
            counters[name][0] += bool(eligible)
            counters[name][1] += bool(eligible and auto)
            counters[name][2] += bool(eligible and auto and correct)
        audit.append({'id': row['id'], 'input': row['text'], 'status': out['status'], 'disposition': disposition, 'prediction': p, 'gold': g, 'parent_fallback_eligible': row['parent_fallback_eligible'], 'parent_match': fallback, 'lexical': detail})
    return ({'total_terms': len(rows), 'metrics': {k: metric(*v) for k, v in counters.items()}, 'outcomes': outcomes,
             'policy_errors': sum(bool(outputs.get(r['id'], {}).get('policy_replay_failure')) for r in rows),
             'operational_incomplete': len(rows) - complete, 'failed_or_unexecuted': sum((outputs.get(r['id'], {}).get('status') != 'completed' or bool(outputs.get(r['id'], {}).get('policy_replay_failure')) for r in rows))}, audit)

def terminal(result):
    m = result['metrics']
    pct = lambda n: 'N/A' if n is None else f'{n * 100:.2f}%'
    total = result['total_terms']
    print(f'\nEVALUATION SUMMARY - {result.get("partition", "evaluation").upper()}')
    print(f'Total terms evaluated: {total}')
    print(f'{"Measure":<30} {"Coverage":>10} {"Accuracy/agreement":>19} {"Disagreements":>15}')
    for label, name in (('PT - exact gold', 'PT'), ('LLT - exact gold', 'LLT'), ('LLT - supported agreement', 'LLT_supported')):
        item = m.get(name)
        if item is None:
            print(f'{label:<30} {"N/A":>10} {"rescore required":>19}')
        else:
            print(f'{label:<30} {pct(item["coverage"]):>10} {pct(item["accuracy"]):>19} {item["disagreements"]:>15}')
    outcomes = result.get('outcomes')
    if outcomes is not None:
        for label, name in (('Automatically matched (PT + LLT)', 'automatic'), ('Human review', 'human_review'), ('Unsupported (no supported match)', 'unsupported')):
            count = outcomes[name]
            print(f'{label}: {count} ({pct(count / total if total else None)})')
        print(f'Execution errors: {outcomes["execution_errors"]} | Policy errors: {result.get("policy_errors", 0)} | Unexecuted: {outcomes["unexecuted"]}')
        if outcomes['unknown']:
            print(f'Outcome unavailable in saved evidence: {outcomes["unknown"]}')
    else:
        print('Outcome breakdown unavailable in this older score; rescore saved records.')
    print('Supported agreement: exact/verified lexical; gold PT parent only when LLT gold is absent.')
    if 'overlap' in result:
        print('Training-wording overlap breakdown retained in the scoring JSON.')

def main(partition):
    p = argparse.ArgumentParser(description=f'ASP-MT {partition}: run (LIVE), score (OFFLINE), summary (OFFLINE)')
    sub = p.add_subparsers(dest='command', required=True)
    q = sub.add_parser('run')
    q.add_argument('--directory', type=Path, required=True)
    q.add_argument('--terms', type=Path)
    q.add_argument('--cases', type=Path)
    q.add_argument('--limit', type=int)
    q.add_argument('--resume', action='store_true')
    q.add_argument('--record-decisions', action='store_true', help='Write observational TRACE decision events; metrics unchanged')
    q.add_argument('--env-file', type=Path)
    q = sub.add_parser('score')
    q.add_argument('--directory', type=Path, required=True)
    q.add_argument('--output', type=Path, required=True)
    q = sub.add_parser('summary')
    q.add_argument('--score', type=Path, required=True)
    a = p.parse_args()
    if a.command == 'summary':
        terminal(load(a.score))
        return 0
    if a.command == 'run':
        c = configuration()
        terms = a.terms or Path(c[partition + '_terms'])
        cases = a.cases or Path(c[partition + '_cases'])
        rows = dataset(partition, terms, cases)
        selected = list(rows.values())
        if a.limit is not None:
            if a.limit < 1:
                p.error('--limit must be positive')
            selected = sorted(selected, key=lambda r: (hashlib.sha256(r['text'].encode()).hexdigest(), r['id']))[:a.limit]
        binding = {'partition': partition, 'terms': str(terms.resolve()), 'cases': str(cases.resolve()), 'terms_sha256': sha(terms), 'cases_sha256': sha(cases)}
        if partition == 'test' and c.get('training_terms'):
            binding.update(training_terms=c['training_terms'], training_terms_sha256=sha(c['training_terms']))
        from pipeline import run
        run([{k: r[k] for k in ('id', 'text', 'frequency')} for r in selected], a.directory, resume=a.resume, env_file=a.env_file, dataset_binding=binding, record_decisions=a.record_decisions)
        # Both evaluators score and print the same concise summary after generation.
        a.output = a.directory / 'evaluation.json'
        suffix = 2
        while a.output.exists() or a.output.with_name(a.output.stem + '_audit.json').exists():
            a.output = a.directory / f'evaluation_{suffix}.json'
            suffix += 1
    if a.output.exists() or a.output.with_name(a.output.stem + '_audit.json').exists():
        raise FileExistsError('Fresh scoring output required')
    contract = load(a.directory / 'contract.json')
    b = contract['dataset_binding']
    if b['partition'] != partition:
        raise ValueError('Wrong evaluation partition')
    for name in ('terms', 'cases', 'training_terms'):
        if name in b and sha(b[name]) != b[name + '_sha256']:
            raise ValueError('Changed benchmark binding: ' + name)
    all_rows = dataset(partition, b['terms'], b['cases'])
    rows = [all_rows[r['id']] for r in contract['inputs']]
    if any(({k: r[k] for k in ('id', 'text', 'frequency')} != s for r, s in zip(rows, contract['inputs']))):
        raise ValueError('Changed evaluation inputs')
    from runtime_resources import implementation
    if contract['implementation'] != implementation():
        raise ValueError('Changed scoring/runtime/reference bindings')
    bootstrap()
    from lexical_equivalence_evaluation import Reference
    ref = Reference(resource('meddra_exact_reference.csv'), resource('llt.asc'), resource('pt.asc'))
    outputs = {}
    for r in rows:
        f = a.directory / 'records' / (r['id'] + '.json')
        if f.exists():
            x = load(f)
            if x['id'] != r['id'] or x['text'] != r['text']:
                raise ValueError('Wrong per-input record')
            outputs[r['id']] = {'status': x['status'], 'prediction': x['prediction'], 'outcome': x.get('outcome'), 'policy_replay_failure': x.get('policy_replay_failure')}
    result, audit = score_rows(rows, outputs, ref)
    result.update(partition=partition, policy='threshold_T', bindings=b, run_contract_sha256=sha(a.directory / 'contract.json'))
    if 'training_terms' in b:
        with Path(b['training_terms']).open(encoding='utf-8-sig', newline='') as f:
            seen = {key(r['Term']) for r in csv.DictReader(f)}
        result['overlap'] = {label: score_rows([r for r in rows if (key(r['text']) in seen) == flag], outputs, ref)[0] for label, flag in (('training-wording-overlap', True), ('non-overlapping-wording', False))}
    save_new(a.output.with_name(a.output.stem + '_audit.json'), audit)
    save_new(a.output, result)
    terminal(result)
    print('Saved:', a.output)
    return int(bool(result['failed_or_unexecuted']))
