from runtime_resources import resource as asp_resource
import importlib.util
from pathlib import Path

def isolated(name, path):
    spec = importlib.util.spec_from_file_location(name, Path(path))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m

def semantic_observations(inp):
    return {c.candidate_id: c.semantic_evidence.cosine_similarity for c in inp.aggregation_result.candidates if c.semantic_evidence}
