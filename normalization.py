from runtime_resources import resource as asp_resource
import re
import unicodedata
from schemas import NormalizationResult
from state import StandardizationState
from clinical_features import preserve_numeric_notation
MULTIPLE_WHITESPACE_PATTERN = re.compile('\\s+')
UNICODE_DASH_PATTERN = re.compile('[\\u2010\\u2011\\u2012\\u2013\\u2014\\u2015\\u2212]')
HYPHEN_WHITESPACE_PATTERN = re.compile('\\s*-\\s*')
TRAILING_PUNCTUATION_PATTERN = re.compile('[\\s.,;:!?]+$')
COMMON_PUNCTUATION_PATTERN = re.compile('(?<!\\d)[.,](?!\\d)|[;:!?()\\[\\]{}]+')
TOKEN_PATTERN = re.compile('[a-z0-9]+(?:-[a-z0-9]+)*')

def _singularize_token(token: str) -> str | None:
    """Return a conservative grammatical singular candidate.

    This is only a lookup variant. Exact matching accepts it only when the
    complete variant resolves to one current MedDRA identity.
    """
    if len(token) >= 5 and token.endswith('ies'):
        return token[:-3] + 'y'
    if len(token) >= 5 and token.endswith(('ches', 'shes', 'sses', 'xes', 'zes')):
        return token[:-2]
    if len(token) >= 4 and token.endswith('s') and (not token.endswith(('ss', 'us', 'is'))):
        return token[:-1]
    return None

def normalization_variants(term: str) -> list[tuple[str, str]]:
    """Build ordered lookup candidates while preserving numerical constraints.

    The operation labels make exact-path behavior auditable. Plural variants
    are deliberately generated but never accepted without an index-level
    uniqueness check in ``exact_matching``.
    """
    normalized = normalize_term(term).normalized_term
    variants: list[tuple[str, str]] = [(normalized, 'normalized')]

    def add(value: str, operation: str) -> None:
        value = MULTIPLE_WHITESPACE_PATTERN.sub(' ', value).strip()
        if value and value not in {item[0] for item in variants}:
            variants.append((value, operation))
    hyphen_spaced = re.sub('(?<=[a-z])-(?=[a-z])', ' ', normalized)
    add(hyphen_spaced, 'hyphen_to_space')
    add(COMMON_PUNCTUATION_PATTERN.sub(' ', normalized), 'punctuation_spacing')
    add(COMMON_PUNCTUATION_PATTERN.sub(' ', hyphen_spaced), 'punctuation_and_hyphen_to_space')
    tokens = TOKEN_PATTERN.findall(hyphen_spaced)
    singular_tokens = list(tokens)
    changed = False
    for index, token in enumerate(tokens):
        singular = _singularize_token(token)
        if singular is not None:
            singular_tokens[index] = singular
            changed = True
    if changed:
        replacements = dict(zip(tokens, singular_tokens))
        add(TOKEN_PATTERN.sub(lambda match: replacements.get(match[0], match[0]), hyphen_spaced), 'unambiguous_plural_candidate')
    return variants

def normalize_term(term: str) -> NormalizationResult:
    """
    Apply conservative text normalization to one term.
    """
    original_term = term
    normalized_term = preserve_numeric_notation(term)
    applied_operations: list[str] = []
    if normalized_term != term:
        applied_operations.append('preserved_numeric_notation')
    unicode_normalized = unicodedata.normalize('NFKC', normalized_term)
    if unicode_normalized != normalized_term:
        applied_operations.append('unicode_normalization')
    normalized_term = unicode_normalized
    dash_normalized = UNICODE_DASH_PATTERN.sub('-', normalized_term)
    if dash_normalized != normalized_term:
        applied_operations.append('normalized_dashes')
    normalized_term = dash_normalized
    stripped_term = normalized_term.strip()
    if stripped_term != normalized_term:
        applied_operations.append('trimmed_whitespace')
    normalized_term = stripped_term
    collapsed_term = MULTIPLE_WHITESPACE_PATTERN.sub(' ', normalized_term)
    if collapsed_term != normalized_term:
        applied_operations.append('collapsed_whitespace')
    normalized_term = collapsed_term
    hyphen_normalized = HYPHEN_WHITESPACE_PATTERN.sub('-', normalized_term)
    if hyphen_normalized != normalized_term:
        applied_operations.append('normalized_hyphen_spacing')
    normalized_term = hyphen_normalized
    punctuation_trimmed = TRAILING_PUNCTUATION_PATTERN.sub('', normalized_term)
    if punctuation_trimmed != normalized_term:
        applied_operations.append('trimmed_terminal_punctuation')
    normalized_term = punctuation_trimmed
    casefolded_term = normalized_term.casefold()
    if casefolded_term != normalized_term:
        applied_operations.append('casefolded')
    normalized_term = casefolded_term
    if not normalized_term:
        raise ValueError('Normalization produced an empty term.')
    return NormalizationResult(original_term=original_term, normalized_term=normalized_term, applied_operations=applied_operations)

def normalization_node(state: StandardizationState) -> dict[str, NormalizationResult]:
    """
    Normalize the input term and add the result to graph state.
    """
    normalization_result = normalize_term(state['term_input'].term)
    return {'normalization': normalization_result}
