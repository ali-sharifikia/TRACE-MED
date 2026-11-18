"""Opt-in phrase views and input-bound clinical-proof transport, without labels."""
from runtime_resources import resource as asp_resource
import copy
import types
from contextvars import ContextVar
from functools import lru_cache
from pathlib import Path
import expression_context_recovery as baseline
import phrase_lexicon as phrases
from clinical_specificity import interpret
prior = baseline.prior
previous = baseline.previous
reconciliation = baseline.reconciliation
proof_engine = baseline.proof_engine
control = baseline.control
MODES = ('F', 'P', 'T', 'PT')
ACTIVE = ContextVar('phrase_evidence_consistency', default=None)
PHRASE_SOURCE = phrases.DEFAULT_SOURCE

@lru_cache(maxsize=1)
def default_normalizer():
    return phrases.default_normalizer()

class ComparisonLex:

    def __init__(self, lex, mode, context=None, normalizer=None):
        if mode not in MODES:
            raise ValueError('Unknown phrase evidence condition')
        self.lex, self.mode, self.context = (lex, mode, context)
        self.normalizer = normalizer

    def __getattr__(self, name):
        return getattr(self.lex, name)

    def view(self, text):
        original = text
        transformations = []
        context = self.context
        if self.mode in ('T', 'PT') and context and context.get('resolved') and (text == context['original']):
            if not context.get('witnesses') or not context.get('execution'):
                raise ValueError('Missing source/context execution authority')
            v = baseline.abbreviation.substitute(text, context['span'], context['expansion'], overlap=True)
            text = v['view']
            transformations.append({**v['transformation'], 'witnesses': context['witnesses'], 'context_execution': context['execution'], 'authority': 'source-witnessed, model-assessed original context'})
        phrase = None
        if self.mode in ('P', 'PT'):
            phrase = (self.normalizer or default_normalizer()).view(text)
            text = phrase['view']
            transformations.extend(phrase['transformations'])
        result = baseline.ComparisonLex(self.lex, 'F', None).view(text)
        return {**result, 'original': original, 'transformations': transformations + result['transformations'], 'new_transformations': transformations + result['new_transformations'], 'transported_transformations': transformations, 'phrase_view': phrase, 'experimental_condition': self.mode}

def lex_for(lex, mode, context=None):
    return ComparisonLex(control.control.lex_for(lex, 'S'), mode, context if context is not None else ACTIVE.get())

def baseline_compare(text, label):
    """Exactly the established F attachment repair; no new source substitution."""
    old = copy.deepcopy(control.control.control.aligned_compare(text, label))
    p = baseline.semantic_engine().proof(text, label)
    a, b = (baseline.semantic_view(text), baseline.semantic_view(label))
    if p['positive'] and (not a['unresolved']) and (not b['unresolved']) and (prior.norm(a['view']) == prior.norm(b['view'])) and (not old['unrepresented']) and (not old['unsupported_additions']) and (not old['contradictions']) and (old['unresolved'] == ['full_expression_attachment_unverified']):
        old.update(unresolved=[], full_equivalence_blocked=False, relationship='bounded_full_expression_attachment', full_expression_witness=p)
    return old

def _expansions(context):
    result = set()
    for row in context.get('all_alternatives', []):
        if isinstance(row, str):
            result.add(prior.norm(row))
            continue
        if isinstance(row, dict) and row.get('expansion'):
            result.add(prior.norm(row['expansion']))
        elif isinstance(row, dict) and len(row.get('fields', [])) >= 5:
            result.add(prior.norm(row['fields'][4]))
    return result

def _remove_attribute_spans(text, attributes):
    spans = sorted({tuple(a['span']) for a in attributes}, reverse=True)
    for a, b in spans:
        text = text[:a] + text[b:]
    return prior.norm(text)

def introduced_attribute_proof(raw, context):
    """A source expansion may explain intrinsic attributes, not choose a narrower sense silently."""
    additions = raw['unsupported_additions']
    out = {'proved': not additions, 'explained': [], 'narrowing_alternatives': []}
    if not additions:
        return out
    if not context or not context.get('resolved') or context.get('original') != raw['original']:
        return out
    span = context['span']
    a, b = span['span']
    if raw['original'][a:b] != span['text'] or not context.get('witnesses') or (not context.get('execution')):
        return out
    expansion = context['expansion']
    expansion_attributes = interpret(expansion)['attributes']
    for added in additions:
        matches = [x for x in expansion_attributes if (x['attribute'], x['value']) == (added['attribute'], added['value'])]
        if not matches:
            return out
        core = _remove_attribute_spans(expansion, matches)
        for alternative in _expansions(context):
            if alternative == prior.norm(expansion):
                continue
            if prior.norm(alternative) == core:
                out['narrowing_alternatives'].append({'added': added, 'selected_expansion': expansion, 'less_specific_source_alternative': alternative})
        out['explained'].append({'added': added, 'expansion_attributes': matches, 'source_witnesses': context['witnesses'], 'original_abbreviation_span': span})
    out['proved'] = len(out['explained']) == len(additions) and (not out['narrowing_alternatives'])
    return out

def transported_compare(text, label, wrapper, enabled=True):
    raw = baseline_compare(text, label)
    if not enabled:
        return raw
    a, b = (wrapper.view(text), wrapper.view(label))
    if not (a['transported_transformations'] or b['transported_transformations']):
        return raw
    after = copy.deepcopy(control.control.control.aligned_compare(a['view'], b['view']))
    proof = proof_engine.input_proof(a['view'], b['view'])
    additions = introduced_attribute_proof(raw, wrapper.context)
    safe = bool(proof['positive'] and (not after['full_equivalence_blocked']) and additions['proved'] and (not raw['unrepresented']) and (not raw['contradictions']) and (not set(raw['unresolved']) - {'full_expression_attachment_unverified'}) and (not (a.get('phrase_view') or {}).get('unresolved')) and (not (b.get('phrase_view') or {}).get('unresolved')))
    evidence = {'proved': safe, 'raw': copy.deepcopy(raw), 'views': {'input': a, 'candidate': b}, 'full_expression_proof': proof, 'introduced_attribute_proof': additions, 'view_comparison': after, 'authority': 'same bound source/context/phrase evidence, not omission of a raw guard'}
    if not safe:
        return {**raw, 'phrase_evidence_transport': evidence}
    return {**after, 'original': text, 'candidate': label, 'relationship': 'bound_full_expression_evidence_transport', 'phrase_evidence_transport': evidence}

def analyze(case, ref, sem, lex, *, raw=None, mode='F', context=None):
    if mode not in MODES:
        raise ValueError('Unknown phrase evidence condition')
    if mode == 'F':
        return baseline.analyze(case, ref, sem, lex, raw=raw, mode='F', context=None)
    context = context if context is not None else ACTIVE.get()
    wrapper = ComparisonLex(lex, mode, context)
    transport = mode in ('T', 'PT')
    if not any((wrapper.view(text)['transported_transformations'] for text in [case['input']] + [r['llt_name'] for r in case['candidates']])):
        result = baseline.analyze(case, ref, sem, lex, raw=raw, mode='F', context=None)
        result.update(experimental_condition=mode, input_view=wrapper.view(case['input']), context_resolution=copy.deepcopy(context))
        return result
    if case['baseline']['pt']:
        if len(case['baseline']['pt']) != 1 or case['baseline']['llt']:
            return baseline.analyze(case, ref, sem, lex, raw=raw, mode='F', context=None)
        k = reconciliation.M.propose(case, ref, sem, wrapper, 'K')
        engine = previous.loader.isolated('_phrase_evidence_reconciliation', reconciliation.__file__)
        engine.compare = lambda text, label: transported_compare(text, label, wrapper, transport)
        result = engine.proposal(case, k, raw or {}, 'RW')
        result.update(branch='accepted_PT', comparisons=k['comparisons'], original_group_comparison=k, clinical_transport={cid: engine.compare(case['input'], next((r['llt_name'] for r in case['candidates'] if r['candidate_id'] == cid))) for cid in k.get('group', [])})
    else:
        source = control.control
        engine = previous.loader.isolated('_phrase_evidence_hierarchy', proof_engine.__file__)
        engine.specificity = source.control.aligned_compare
        spelling = source.Spelling(wrapper)

        def pair(text, left, right, reference):
            old = source.pair(text, left, right, reference, spelling)
            if old['status'] != 'unresolved' or old.get('retained_differences'):
                return old
            x, y = (reference.identity(left), reference.identity(right))
            if not x.get('current') or not y.get('current') or x['pt_code'] != y['pt_code']:
                return old
            a, b = (wrapper.view(x['llt_name']), wrapper.view(y['llt_name']))
            if not (a['new_transformations'] or b['new_transformations']):
                return old
            g = prior.group.wording_proof(a['view'], b['view'])
            cp = prior.ranker.comparison(a['view'], b['view'])
            clinical = [engine.specificity(p, q) for p, q in ((a['view'], b['view']), (b['view'], a['view']))]
            if not g or not cp['equivalent'] or cp['protected_differences'] or (not cp['source_frame']['safe']) or (not cp['candidate_frame']['safe']) or any((c['full_equivalence_blocked'] for c in clinical)):
                return old
            return {**old, 'status': 'verified_equivalent', 'route': 'bound_phrase_views', 'rule': 'source_orthography_full_phrase', 'reasons': [], 'retained_differences': [], 'witness': {'left': a, 'right': b, 'grammar': g, 'comparison': cp, 'clinical': clinical}}
        group = types.SimpleNamespace(**{**vars(prior.group), 'compare_pair': pair})
        engine.prior = types.SimpleNamespace(**{**vars(prior), 'group': group})
        result = engine.analyze(case, ref, sem, lex_for(lex, mode, context), raw=raw, reconcile=False)
        result['cross_PT'] = previous.parent_contrasts(case, result)
        if transport:
            checks = {}
            for cid in result.get('group', []):
                name = result['comparisons'][cid]['candidate']['llt_name']
                checks[cid] = transported_compare(case['input'], name, wrapper, True)
                if checks[cid].get('phrase_evidence_transport') and checks[cid]['full_equivalence_blocked']:
                    result['blockers'].append('raw_clinical_transport_unestablished:' + cid)
            result['clinical_transport'] = checks
            result['blockers'] = sorted(set(result['blockers']))
            result['accepted'] = not result['blockers']
    result.update(experimental_condition=mode, input_view=wrapper.view(case['input']), context_resolution=copy.deepcopy(context))
    return result
