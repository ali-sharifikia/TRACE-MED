from __future__ import annotations
from runtime_resources import resource as asp_resource
import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from schemas import SemanticCandidate

def _canonical_label(value: str) -> str:
    return ' '.join(re.findall('[a-z0-9]+', value.casefold()))

def _is_current(value: object) -> bool:
    return str(value).strip().casefold() in {'1', 'true', 'yes', 'y'}

@dataclass(frozen=True)
class PTExpansionLLT:
    llt_code: str
    llt_name: str
    pt_code: str
    pt_name: str

class SemanticPTIndex:
    """Current PT labels and a bounded LLT expansion for semantic retrieval."""

    def __init__(self, reference_path: str | Path, *, llts_per_pt: int=2) -> None:
        if llts_per_pt < 1:
            raise ValueError('llts_per_pt must be at least 1.')
        rows_by_pt: dict[str, list[PTExpansionLLT]] = {}
        pt_names: dict[str, str] = {}
        with Path(reference_path).open('r', encoding='utf-8-sig', newline='') as handle:
            for row in csv.DictReader(handle):
                if not _is_current(row.get('is_current', 'true')):
                    continue
                llt_code = str(row.get('llt_code', '')).strip()
                llt_name = str(row.get('llt_name', '')).strip()
                pt_code = str(row.get('pt_code', '')).strip()
                pt_name = str(row.get('pt_name', '')).strip()
                if not all((llt_code, llt_name, pt_code, pt_name)):
                    continue
                pt_names.setdefault(pt_code, pt_name)
                rows_by_pt.setdefault(pt_code, []).append(PTExpansionLLT(llt_code=llt_code, llt_name=llt_name, pt_code=pt_code, pt_name=pt_name))
        ordered_pts = sorted(pt_names.items(), key=lambda item: (_canonical_label(item[1]), item[0]))
        self.pt_codes = tuple((pt_code for pt_code, _ in ordered_pts))
        self.pt_names = tuple((pt_name for _, pt_name in ordered_pts))
        self._expansions: dict[str, tuple[PTExpansionLLT, ...]] = {}
        for pt_code, pt_name in ordered_pts:
            unique = {item.llt_code: item for item in rows_by_pt.get(pt_code, [])}
            ordered_llts = sorted(unique.values(), key=lambda item: (_canonical_label(item.llt_name) != _canonical_label(pt_name), _canonical_label(item.llt_name), item.llt_code))
            self._expansions[pt_code] = tuple(ordered_llts[:llts_per_pt])

    def expand_ranked_pts(self, ranked_indices_and_scores: Iterable[tuple[int, float]]) -> list[SemanticCandidate]:
        candidates: list[SemanticCandidate] = []
        rank = 1
        for pt_index, score in ranked_indices_and_scores:
            pt_code = self.pt_codes[pt_index]
            for item in self._expansions[pt_code]:
                candidates.append(SemanticCandidate(candidate_id=f'LLT:{item.llt_code}', rank=rank, llt_code=item.llt_code, llt_name=item.llt_name, pt_code=item.pt_code, pt_name=item.pt_name, is_current=True, cosine_similarity=max(-1.0, min(1.0, float(score))), direct_pt_match_score=max(-1.0, min(1.0, float(score))), introduced_by_pt_expansion=True))
                rank += 1
        return candidates
