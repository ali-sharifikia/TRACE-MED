from runtime_resources import resource as asp_resource
from functools import lru_cache
from runtime_resources import resource
REF = resource('meddra_exact_reference.csv')

@lru_cache(maxsize=2)
def verifier_reranker(path, hash_value):
    from llt_reranking import PTConditionedLLTReranker
    from runtime_resources import sha
    if sha(path) != hash_value:
        raise ValueError('Changed terminology reference')
    return PTConditionedLLTReranker(path)
