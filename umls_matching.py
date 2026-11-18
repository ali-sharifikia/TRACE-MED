from __future__ import annotations
from runtime_resources import resource as asp_resource
import os
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit
import pandas as pd
import requests
from dotenv import load_dotenv
from schemas import UMLSCandidate, UMLSLookupStatus, UMLSMappingMethod, UMLSMatchResult
from state import StandardizationState

class UMLSRequestError(RuntimeError):
    """
    Raised when the UMLS API request cannot be
    completed successfully.
    """

class UMLSMatcher:
    """
    LangGraph-compatible UMLS-to-MedDRA lookup node.

    The node does not make final matching decisions.
    It only returns locally validated MedDRA
    candidates supported by UMLS.
    """
    BASE_URL = 'https://uts-ws.nlm.nih.gov/rest'

    def __init__(self, meddra_reference_path: Path, api_key: str | None=None, umls_version: str='current', source_abbreviation: str='MDR', timeout_seconds: float=30.0, max_retries: int=3, retry_backoff_seconds: float=1.0, request_delay_seconds: float=0.05, page_size: int=200) -> None:
        self.meddra_reference_path = Path(meddra_reference_path)
        self.umls_version = umls_version
        self.source_abbreviation = source_abbreviation
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.retry_backoff_seconds = retry_backoff_seconds
        self.request_delay_seconds = request_delay_seconds
        self.page_size = min(page_size, 200)
        self._load_api_key(api_key)
        self._session = requests.Session()
        self._session.headers.update({'Accept': 'application/json', 'User-Agent': 'MedDRA-Standardization-Workflow/0.1'})
        self._llt_by_code, self._llts_by_pt_code = self._load_meddra_reference()
        self._cache: dict[str, UMLSMatchResult] = {}

    def _load_api_key(self, supplied_api_key: str | None) -> None:
        project_root = Path(__file__).resolve().parent
        load_dotenv(dotenv_path=project_root / '.env')
        api_key = supplied_api_key or os.getenv('UMLS_API_KEY')
        if not api_key or not api_key.strip():
            raise ValueError('UMLS_API_KEY was not found. Add it to the project .env file.')
        self._api_key = api_key.strip()

    def _load_meddra_reference(self) -> tuple[dict[str, dict[str, str]], dict[str, list[dict[str, str]]]]:
        if not self.meddra_reference_path.exists():
            raise FileNotFoundError(f'MedDRA reference file was not found: {self.meddra_reference_path}')
        reference_df = pd.read_csv(self.meddra_reference_path, dtype=str, keep_default_na=False)
        required_columns = {'llt_code', 'llt_name', 'pt_code', 'pt_name'}
        missing_columns = required_columns - set(reference_df.columns)
        if missing_columns:
            raise ValueError(f'MedDRA reference is missing columns: {sorted(missing_columns)}')
        llt_by_code: dict[str, dict[str, str]] = {}
        llts_by_pt_code: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in reference_df.itertuples(index=False):
            record = {'llt_code': str(getattr(row, 'llt_code')).strip(), 'llt_name': str(getattr(row, 'llt_name')).strip(), 'pt_code': str(getattr(row, 'pt_code')).strip(), 'pt_name': str(getattr(row, 'pt_name')).strip()}
            if not all(record.values()):
                continue
            existing = llt_by_code.get(record['llt_code'])
            if existing is not None:
                if existing != record:
                    raise ValueError(f"Conflicting records found for LLT code {record['llt_code']}.")
                continue
            llt_by_code[record['llt_code']] = record
            llts_by_pt_code[record['pt_code']].append(record)
        if not llt_by_code:
            raise ValueError('The MedDRA reference contains no valid LLT records.')
        return (llt_by_code, dict(llts_by_pt_code))

    @staticmethod
    def _extract_source_code(code_value: Any) -> str | None:
        """
        UMLS may return either a plain source code
        or a URL ending in that code.
        """
        if code_value is None:
            return None
        code_text = str(code_value).strip()
        if not code_text:
            return None
        parsed = urlsplit(code_text)
        if parsed.scheme and parsed.path:
            code_text = parsed.path
        code_text = unquote(code_text.rstrip('/').rsplit('/', maxsplit=1)[-1])
        if code_text.upper().startswith('MDR:'):
            code_text = code_text.split(':', maxsplit=1)[1]
        code_text = code_text.strip()
        return code_text or None

    def _resolve_current_meddra_records(self, source_code: str) -> list[dict[str, str]]:
        """
        Convert a UMLS MedDRA source code into one
        or more current local LLT records.

        Direct LLT codes are preferred. If the code
        is a PT code, only its same-named preferred
        LLT is returned. Arbitrary child LLTs are
        not introduced.
        """
        direct_llt = self._llt_by_code.get(source_code)
        if direct_llt is not None:
            return [direct_llt]
        pt_llts = self._llts_by_pt_code.get(source_code, [])
        if not pt_llts:
            return []
        preferred_llts = [record for record in pt_llts if record['llt_code'] == record['pt_code']]
        if preferred_llts:
            return preferred_llts
        same_name_llts = [record for record in pt_llts if record['llt_name'].casefold() == record['pt_name'].casefold()]
        return same_name_llts

    def _retry_delay(self, attempt: int, retry_after: str | None=None) -> float:
        if retry_after:
            try:
                return float(retry_after)
            except ValueError:
                pass
        return self.retry_backoff_seconds * 2 ** (attempt - 1)

    def _get_json(self, endpoint: str, params: dict[str, Any] | None=None) -> dict[str, Any]:
        url = f"{self.BASE_URL}/{endpoint.lstrip('/')}"
        request_params = dict(params or {})
        request_params['apiKey'] = self._api_key
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self._session.get(url, params=request_params, timeout=self.timeout_seconds)
            except requests.RequestException as exc:
                if attempt == self.max_retries:
                    raise UMLSRequestError('Could not connect to the UMLS API.') from exc
                time.sleep(self._retry_delay(attempt))
                continue
            if response.status_code == 404:
                return {}
            if response.status_code in {401, 403}:
                raise UMLSRequestError('UMLS authentication failed. Check UMLS_API_KEY.')
            if response.status_code == 429 or response.status_code >= 500:
                if attempt == self.max_retries:
                    raise UMLSRequestError(f'UMLS API remained unavailable after all retry attempts. Status: {response.status_code}')
                time.sleep(self._retry_delay(attempt, response.headers.get('Retry-After')))
                continue
            try:
                response.raise_for_status()
            except requests.HTTPError as exc:
                raise UMLSRequestError(f'UMLS request failed with status {response.status_code}.') from exc
            try:
                response_data = response.json()
            except ValueError as exc:
                raise UMLSRequestError('UMLS returned an invalid JSON response.') from exc
            if self.request_delay_seconds > 0:
                time.sleep(self.request_delay_seconds)
            if not isinstance(response_data, dict):
                raise UMLSRequestError('UMLS returned an unexpected response structure.')
            return response_data
        raise UMLSRequestError('UMLS request failed unexpectedly.')

    @staticmethod
    def _search_results(response_data: dict[str, Any]) -> list[dict[str, Any]]:
        results = response_data.get('result', {}).get('results', [])
        if not isinstance(results, list):
            return []
        return [result for result in results if isinstance(result, dict)]

    def _direct_meddra_search(self, normalized_term: str) -> list[UMLSCandidate]:
        """
        Search for an exact MedDRA source term.

        No partial, fuzzy or word-based matching is
        performed here.
        """
        response_data = self._get_json(endpoint=f'search/{self.umls_version}', params={'string': normalized_term, 'sabs': self.source_abbreviation, 'returnIdType': 'code', 'searchType': 'exact', 'pageSize': self.page_size})
        candidates: list[UMLSCandidate] = []
        for result in self._search_results(response_data):
            result_ui = result.get('ui')
            if not result_ui or result_ui == 'NONE':
                continue
            source_code = self._extract_source_code(result_ui)
            if source_code is None:
                continue
            local_records = self._resolve_current_meddra_records(source_code)
            for record in local_records:
                candidates.append(UMLSCandidate(candidate_id=f"LLT:{record['llt_code']}", llt_code=record['llt_code'], llt_name=record['llt_name'], pt_code=record['pt_code'], pt_name=record['pt_name'], umls_cui=None, umls_term_name=str(result.get('name') or normalized_term), umls_source_code=source_code, umls_source=str(result.get('rootSource') or self.source_abbreviation), umls_term_type=None, mapping_method=UMLSMappingMethod.DIRECT_MEDDRA_SEARCH))
        return self._deduplicate_candidates(candidates)

    def _find_best_cui(self, normalized_term: str) -> tuple[str, str | None] | None:
        """
        Find a UMLS CUI containing an exact synonym
        for the unchanged normalized input term.
        """
        response_data = self._get_json(endpoint=f'search/{self.umls_version}', params={'string': normalized_term, 'searchType': 'exact', 'pageSize': self.page_size})
        for result in self._search_results(response_data):
            cui = str(result.get('ui') or '').strip()
            if not re.fullmatch('C\\d+', cui):
                continue
            cui_name = str(result.get('name') or '').strip()
            return (cui, cui_name or None)
        return None

    def _get_meddra_atoms(self, cui: str) -> list[dict[str, Any]]:
        response_data = self._get_json(endpoint=f'content/{self.umls_version}/CUI/{cui}/atoms', params={'sabs': self.source_abbreviation, 'ttys': 'LLT,PT', 'language': 'ENG', 'pageSize': self.page_size})
        atoms = response_data.get('result', [])
        if not isinstance(atoms, list):
            return []
        return [atom for atom in atoms if isinstance(atom, dict)]

    def _crosswalk_cui_to_meddra(self, cui: str) -> list[UMLSCandidate]:
        candidates: list[UMLSCandidate] = []
        for atom in self._get_meddra_atoms(cui):
            source_code = self._extract_source_code(atom.get('code'))
            if source_code is None:
                continue
            local_records = self._resolve_current_meddra_records(source_code)
            for record in local_records:
                atom_name = str(atom.get('name') or record['llt_name']).strip()
                candidates.append(UMLSCandidate(candidate_id=f"LLT:{record['llt_code']}", llt_code=record['llt_code'], llt_name=record['llt_name'], pt_code=record['pt_code'], pt_name=record['pt_name'], umls_cui=cui, umls_term_name=atom_name, umls_source_code=source_code, umls_source=str(atom.get('rootSource') or self.source_abbreviation), umls_term_type=str(atom.get('termType')).strip() or None, mapping_method=UMLSMappingMethod.CUI_ATOM_CROSSWALK))
        return self._deduplicate_candidates(candidates)

    @staticmethod
    def _deduplicate_candidates(candidates: list[UMLSCandidate]) -> list[UMLSCandidate]:
        """
        Keep one result per LLT code.

        Sorting by LLT code only makes output
        reproducible; it is not a candidate ranking.
        """
        unique_candidates: dict[str, UMLSCandidate] = {}
        for candidate in candidates:
            unique_candidates.setdefault(candidate.llt_code, candidate)
        return sorted(unique_candidates.values(), key=lambda candidate: candidate.llt_code)

    def lookup(self, normalized_term: str) -> UMLSMatchResult:
        normalized_term = str(normalized_term).strip()
        if not normalized_term:
            raise ValueError('The UMLS lookup term cannot be empty.')
        cached_result = self._cache.get(normalized_term)
        if cached_result is not None:
            return cached_result.model_copy(deep=True)
        try:
            direct_candidates = self._direct_meddra_search(normalized_term)
            if direct_candidates:
                result = UMLSMatchResult(normalized_input_term=normalized_term, lookup_status=UMLSLookupStatus.CANDIDATES_FOUND, searched_cui=None, searched_cui_name=None, candidates=direct_candidates, error_message=None)
                self._cache[normalized_term] = result
                return result.model_copy(deep=True)
            cui_result = self._find_best_cui(normalized_term)
            if cui_result is None:
                result = UMLSMatchResult(normalized_input_term=normalized_term, lookup_status=UMLSLookupStatus.NO_SUPPORTED_MATCH, searched_cui=None, searched_cui_name=None, candidates=[], error_message=None)
                self._cache[normalized_term] = result
                return result.model_copy(deep=True)
            cui, cui_name = cui_result
            crosswalk_candidates = self._crosswalk_cui_to_meddra(cui)
            lookup_status = UMLSLookupStatus.CANDIDATES_FOUND if crosswalk_candidates else UMLSLookupStatus.NO_SUPPORTED_MATCH
            result = UMLSMatchResult(normalized_input_term=normalized_term, lookup_status=lookup_status, searched_cui=cui, searched_cui_name=cui_name, candidates=crosswalk_candidates, error_message=None)
            self._cache[normalized_term] = result
            return result.model_copy(deep=True)
        except UMLSRequestError as exc:
            return UMLSMatchResult(normalized_input_term=normalized_term, lookup_status=UMLSLookupStatus.LOOKUP_FAILED, searched_cui=None, searched_cui_name=None, candidates=[], error_message=str(exc))

    def __call__(self, state: StandardizationState) -> dict[str, UMLSMatchResult]:
        normalization_result = state.get('normalization')
        if normalization_result is None:
            raise ValueError('UMLS matching requires the normalization result.')
        exact_result = state.get('exact_result')
        if exact_result is None:
            raise ValueError('UMLS matching requires the exact-matching result.')
        if exact_result.found:
            raise ValueError('An exact-matched term should not be sent to UMLS matching.')
        umls_result = self.lookup(normalization_result.normalized_term)
        return {'umls_result': umls_result}

    def close(self) -> None:
        self._session.close()

    def __enter__(self) -> UMLSMatcher:
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()
