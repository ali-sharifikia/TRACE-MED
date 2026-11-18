from runtime_resources import resource as asp_resource
import re
from types import SimpleNamespace
from runtime_resources import load, sha
inventory = SimpleNamespace(sha=sha)
HEADS = frozenset('count counts level levels concentration concentrations activity activities rate rates percentage percentages fraction fractions volume volumes interval intervals'.split())

def substitute(text, span, expansion, *, overlap=False):
    a, b = span['span']
    if text[a:b] != span['text'] or not expansion.strip():
        raise ValueError('Invalid original span or empty expansion')
    transform = {'rule': 'literal_source_span_substitution', 'original_span': [a, b], 'original_text': text[a:b], 'expansion': expansion, 'authority': 'possible_sense_not_context_resolution'}
    right = text[b:]
    end = re.search('\\b([A-Za-z]+)$', expansion)
    next_word = re.match('\\s+([A-Za-z]+)\\b', right)
    if overlap and end and next_word and (end[1].casefold() == next_word[1].casefold()) and (end[1].casefold() in HEADS):
        transform.update(rule='source_expansion_shared_measurement_head', shared_head=end[1], original_head_span=[b + next_word.start(1), b + next_word.end(1)], explanation='The adjacent original measurement head supplies the identical expansion head.')
        expansion = expansion[:end.start()].rstrip()
    return {'original': text, 'view': text[:a] + expansion + right, 'transformation': transform}
