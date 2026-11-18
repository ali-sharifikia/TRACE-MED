"""Default-off semantic views; original contexts, rivals and fixed gates retained."""
from runtime_resources import resource as asp_resource
import copy
import types
from contextvars import ContextVar
from pathlib import Path
from functools import lru_cache
import wording_scope_consistency as control
import abbreviation_context_proof as abbreviation
prior = control.prior
previous = control.previous
reconciliation = control.reconciliation
proof_engine = control.proof_engine
MODES = ('W', 'F', 'A', 'AO', 'FAO')
ACTIVE = ContextVar('expression_context_recovery', default=None)
SOURCE_N = Path(asp_resource('lexicon_nominalizations.json'))
SOURCE_A = Path(asp_resource('lexicon_agreement.json'))

@lru_cache(maxsize=1)
def semantic_engine():
    import full_expression_source as semantic
    return semantic.FullExpressionProof(abbreviation.load(SOURCE_N)['records']['LEX/LRNOM'], {str(p): abbreviation.inventory.sha(p) for p in (SOURCE_N, SOURCE_A)}, abbreviation.load(SOURCE_A)['records']['LEX/LRAGR'])

def semantic_view(text):
    return semantic_engine().view(text)

class ComparisonLex:

    def __init__(self, lex, mode, context=None):
        if mode not in MODES:
            raise ValueError('Unknown experimental condition')
        self.lex, self.mode, self.context = (lex, mode, context)

    def __getattr__(self, name):
        return getattr(self.lex, name)

    def view(self, text):
        original = text
        changes = []
        context = self.context
        if self.mode in ('A', 'AO', 'FAO') and context and context.get('resolved') and (context['original'] == text):
            v = abbreviation.substitute(text, context['span'], context['expansion'], overlap=self.mode != 'A')
            text = v['view']
            changes.append({**v['transformation'], 'witnesses': context['witnesses'], 'authority': 'source-witnessed, model-assessed original context', 'context_execution': context['execution']})
        if self.mode in ('F', 'FAO'):
            v = semantic_view(text)
            text = v['view']
            changes.extend(v['transformations'])
        old = control.WordingLex(self.lex).view(text)
        return {**old, 'original': original, 'transformations': changes + old['transformations'], 'new_transformations': changes, 'experimental_condition': self.mode}

def lex_for(lex, mode, context=None):
    if mode == 'W':
        return control.lex_for(lex, 'W')
    return ComparisonLex(control.control.lex_for(lex, 'S'), mode, context if context is not None else ACTIVE.get())

def analyze(case, ref, sem, lex, *, raw=None, mode='W', context=None):
    context = context if context is not None else ACTIVE.get()
    wrapper = ComparisonLex(lex, mode, context)
    if case['baseline']['pt'] and mode in ('F', 'FAO') and (len(case['baseline']['pt']) == 1) and (not case['baseline']['llt']):
        k = reconciliation.M.propose(case, ref, sem, wrapper, 'K')
        engine = previous.loader.isolated('_expression_context_reconciliation', reconciliation.__file__)

        def compared(text, label):
            old = copy.deepcopy(control.control.control.aligned_compare(text, label))
            p = semantic_engine().proof(text, label)
            va, vb = (semantic_view(text), semantic_view(label))
            unambiguous = not va['unresolved'] and (not vb['unresolved']) and (prior.norm(va['view']) == prior.norm(vb['view']))
            if p['positive'] and unambiguous and (not old['unrepresented']) and (not old['unsupported_additions']) and (not old['contradictions']) and (old['unresolved'] == ['full_expression_attachment_unverified']):
                old.update(unresolved=[], full_equivalence_blocked=False, relationship='bounded_full_expression_attachment', full_expression_witness=p)
            return old
        engine.compare = compared
        result = engine.proposal(case, k, raw or {}, 'RW')
        result.update(branch='accepted_PT', comparisons=k['comparisons'], original_group_comparison=k)
    elif mode == 'W' or (mode in ('A', 'AO') and (not (context and context.get('resolved')))) or case['baseline']['pt']:
        result = control.control.analyze(case, ref, sem, wrapper, raw=raw, mode='S')
    else:
        source = control.control
        engine = previous.loader.isolated('_expression_context_hierarchy', proof_engine.__file__)
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
            return {**old, 'status': 'verified_equivalent', 'route': 'bounded_full_expression_views', 'rule': 'source_role_preserving_pair', 'reasons': [], 'retained_differences': [], 'witness': {'left': a, 'right': b, 'grammar': g, 'comparison': cp, 'clinical': clinical}, 'previous': old}
        group = types.SimpleNamespace(**{**vars(prior.group), 'compare_pair': pair})
        engine.prior = types.SimpleNamespace(**{**vars(prior), 'group': group})
        result = engine.analyze(case, ref, sem, lex_for(lex, mode, context), raw=raw, reconcile=False)
        result['cross_PT'] = previous.parent_contrasts(case, result)
    result['experimental_condition'] = mode
    result['input_view'] = wrapper.view(case['input'])
    result['context_resolution'] = copy.deepcopy(context)
    return result
