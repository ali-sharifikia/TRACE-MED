"""Versioned, gold-independent policy configuration for runtime and replay."""
from runtime_resources import resource as asp_resource
from dataclasses import asdict, dataclass
PIPELINE_POLICY_VERSION = '1.3.3'

@dataclass(frozen=True)
class PolicyConfig:
    version: str = PIPELINE_POLICY_VERSION
    pt_minimum_score: float = 0.88
    pt_minimum_margin: float = 0.1
    llt_minimum_score: float = 0.9
    llt_minimum_margin: float = 0.1
    base_pt_groups: int = 4
    maximum_pt_rank: int = 10
    llts_per_leading_pt: int = 4
    maximum_candidates: int = 12
    score_delta: float = 0.2
    strong_semantic: float = 0.85
    ambiguity_score_delta: float = 0.22
    ambiguity_semantic: float = 0.95
    ambiguity_surface: float = 0.6
    ambiguity_enabled: bool = True

    def __post_init__(self):
        for name in ('pt_minimum_score', 'pt_minimum_margin', 'llt_minimum_score', 'llt_minimum_margin', 'score_delta', 'strong_semantic', 'ambiguity_score_delta', 'ambiguity_semantic', 'ambiguity_surface'):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f'{name} must be between zero and one.')
        if min(self.base_pt_groups, self.maximum_pt_rank, self.llts_per_leading_pt, self.maximum_candidates) < 1:
            raise ValueError('Candidate limits must be positive.')
        if self.maximum_pt_rank < self.base_pt_groups:
            raise ValueError('maximum_pt_rank must cover the base PT groups.')

    def to_dict(self) -> dict:
        return asdict(self)
DEFAULT_POLICY = PolicyConfig()
