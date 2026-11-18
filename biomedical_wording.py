"""Conservative biomedical language rules; no dataset or terminology codes."""
from __future__ import annotations
from runtime_resources import resource as asp_resource
import re
from functools import lru_cache
WORDING_VERSION = '1.0.1'
CONFLICT_RULES_VERSION = '1.3.1'
GENERIC_CONTEXT_VERSION = '1.0.0'
_GENERIC = '(?:(?:adverse|side)\\s+)?(?:events?|reactions?|effects?)'
_SUFFIX = re.compile(f'^(?P<clinical>.+?)\\s+(?:adverse\\s+(?:events?|reactions?|effects?)|side\\s+effects?)$', re.I)
_RELATED_SUFFIX = re.compile(f'^(?P<clinical>.+?)[\\s-]+(?:associated|related)\\s+{_GENERIC}$', re.I)
_COMPOUND = re.compile('\\s+(?:and|plus|with)\\s+|[;,/&]', re.I)

def generic_context_base(value: str) -> str | None:
    """Return the named concept only when a generic suffix is contextual.

    Coordination is deliberately left to the component interpreter. A site
    qualifier is retained (renal-related reactions -> renal), never invented
    into an independently asserted disease.
    """
    value = value.strip().rstrip('.')
    match = _RELATED_SUFFIX.fullmatch(value) or _SUFFIX.fullmatch(value)
    if not match:
        return None
    clinical = match.group('clinical').strip()
    if not clinical or _COMPOUND.search(clinical):
        return None
    return clinical

def is_generic_event(value: str) -> bool:
    return bool(re.fullmatch(_GENERIC, value.strip(), re.I))

@lru_cache(maxsize=65536)
def canonical_biomedical_wording(value: str) -> str:
    """Spelling and tightly scoped derivations, not unrestricted stemming."""
    text = value.casefold().replace('≥', '>=').replace('≤', '<=')
    text = re.sub('[\\u2010-\\u2015\\u2212]', '-', text)
    text = re.sub('\\bhaemato', 'hemato', text)
    text = re.sub('\\bhaemo', 'hemo', text)
    text = re.sub('\\bfoetal\\b', 'fetal', text)
    text = re.sub('\\bhaemat(?:ologic|ological|ology)\\b', 'hematologic', text)
    text = re.sub('\\bhemat(?:ologic|ological|ology)\\b', 'hematologic', text)
    prefixes = {'hemato': 'hematologic', 'hepato': 'hepatic', 'nephro': 'renal', 'cardio': 'cardiac', 'neuro': 'neurologic', 'oto': 'otic'}
    text = re.sub('\\b(hemato|hepato|nephro|cardio|neuro|oto)[- ]?(toxicity|toxicities|toxic)\\b', lambda m: prefixes[m[1]] + ' toxicity', text)
    equivalents = {'toxicities': 'toxicity', 'toxic': 'toxicity', 'neurological': 'neurologic', 'oedema': 'edema', 'anaemia': 'anemia', 'tumour': 'tumor', 'tumours': 'tumor', 'toes': 'toe', 'fingers': 'finger', 'reactions': 'reaction', 'events': 'event', 'effects': 'effect', 'infections': 'infection', 'values': 'value', 'levels': 'level', 'counts': 'count'}
    text = re.sub('\\b[a-z]+\\b', lambda m: equivalents.get(m[0], m[0]), text)
    return ' '.join(text.split())

def repair_generic_context_payload(payload: dict) -> dict:
    """Collapse spurious suffix components while retaining original text."""
    text = str(payload.get('normalized_term', ''))
    if generic_context_base(text) is None:
        return payload
    components = payload.get('asserted_components', [])
    if len(components) <= 1:
        return payload
    if any((component.get('is_negated') for component in components)):
        return payload
    return {**payload, 'structure_type': 'single_concept', 'asserted_components': [{'component_id': 'COMPONENT_1', 'component_text': text, 'is_negated': False}], 'interpretation_rationale': str(payload.get('interpretation_rationale', '')) + ' Deterministic generic-context repair: the event/reaction suffix qualifies one named concept.', 'deterministic_repair_reasons': list(payload.get('deterministic_repair_reasons', [])) + ['generic_context_suffix_is_not_an_independent_event']}
