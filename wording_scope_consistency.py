"""Opt-in transport of existing bounded exclusion spelling evidence; no labels."""
from runtime_resources import resource as asp_resource
import copy
import re
from functools import lru_cache
import source_pair_consistency as control
from clinical_features import NEGATABLE_MODIFIERS
prior = control.prior
previous = control.previous
reconciliation = control.reconciliation
proof_engine = control.proof_engine
MODES = ('W',)
PREFIX = re.compile('\\bnon(?P<separator>[-\\u2010-\\u2015 ]?)(?P<modifier>' + '|'.join(sorted(NEGATABLE_MODIFIERS, key=lambda w: (-len(w), w))) + ')\\b', re.I)

@lru_cache(maxsize=32768)
def view(text):
    changes = []

    def replace(m):
        if not m['separator']:
            return m[0]
        canonical = 'non' + m['modifier'].casefold()
        changes.append({'rule': 'existing_bounded_non_modifier_spelling', 'authority': 'clinical_features.NEGATABLE_MODIFIERS; prefix retained', 'original': m[0], 'span': list(m.span()), 'canonical': canonical, 'modifier': m['modifier'].casefold()})
        return canonical
    return {'original': text, 'view': PREFIX.sub(replace, text), 'transformations': changes}

class WordingLex:

    def __init__(self, lex):
        self.lex = lex

    def __getattr__(self, name):
        return getattr(self.lex, name)

    def view(self, text):
        prefix = copy.deepcopy(view(text))
        old = self.lex.view(prefix['view'])
        return {**old, 'original': text, 'scope_spelling': prefix, 'transformations': prefix['transformations'] + old['transformations']}

def lex_for(lex, mode):
    if mode != 'W':
        raise ValueError('Unknown wording consistency configuration')
    return control.lex_for(WordingLex(lex), 'S')

def analyze(case, ref, sem, lex, *, raw=None, mode='W'):
    if mode != 'W':
        raise ValueError('Unknown wording consistency configuration')
    result = control.analyze(case, ref, sem, WordingLex(lex), raw=raw, mode='S')
    result['wording_scope_view'] = copy.deepcopy(view(case['input']))
    return result
