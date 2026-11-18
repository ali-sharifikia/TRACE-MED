from __future__ import annotations
from runtime_resources import resource as asp_resource
import decision_recorder as trace
from copy import deepcopy
import os
from time import perf_counter
from pathlib import Path
from typing import Any
import json
import traceback
from dotenv import load_dotenv
from pydantic import ValidationError
from reviewer_prompt import build_reviewer_messages
from reviewer_validation import ReviewerDecisionValidationError, validate_reviewer_decision, validate_selected_clinical_mappings, ClinicalMappingValidationError
from schemas import REVIEWER_DECISION_ADAPTER, ReviewerDecision, ReviewerInput, EvidenceRequest, EvidenceRequestAction, EvidenceSource, ReviewerDecisionType, InputStructureType, CandidateReviewVerdict
from reviewer_ambiguity import apply_same_pt_ambiguity_gate
from reviewer_pt_disagreement import apply_pt_disagreement_gate
from state import StandardizationState
PROJECT_ROOT = Path(__file__).resolve().parent

class ReviewerLLMError(RuntimeError):
    """Base exception for reviewer LLM failures."""

class ReviewerLLMConnectionError(ReviewerLLMError):
    """Raised when the local Ollama request fails."""

class ReviewerLLMOutputError(ReviewerLLMError):
    """
    Raised when the model repeatedly returns a decision that
    fails schema or deterministic validation.
    """

class OllamaReviewer:
    """
    Local Ollama-backed reviewer and LangGraph-compatible node.

    The reviewer uses a discriminated ReviewerDecision schema
    for structured generation, parses the response with
    Pydantic, and applies the deterministic validation rules.
    """

    def __init__(self, model: str | None=None, host: str | None=None, *, temperature: float=0.0, seed: int=42, think: bool=False, max_validation_attempts: int=2, retry_diagnostics: bool=True, num_ctx: int=16384, pt_disagreement_gate: bool=True, pt_disagreement_top_k: int=5, same_pt_ambiguity_gate: bool=True, same_pt_ambiguity_margin: float=0.02, same_pt_meaningful_modifier_margin: float=0.04, same_pt_min_support_score: float=0.65, client: Any=None, reliability_repair: bool=False) -> None:
        if client is None:
            load_dotenv(PROJECT_ROOT / '.env')
        self.model = model or os.getenv('OLLAMA_REVIEWER_MODEL', 'qwen3:8b')
        self.host = host or os.getenv('OLLAMA_REVIEWER_HOST', 'http://127.0.0.1:11434')
        if max_validation_attempts < 1:
            raise ValueError('max_validation_attempts must be at least 1.')
        if num_ctx < 4096:
            raise ValueError('num_ctx must be at least 4096 for the reviewer prompt and evidence package.')
        if pt_disagreement_top_k < 1:
            raise ValueError('pt_disagreement_top_k must be at least 1.')
        if not 0.0 <= same_pt_ambiguity_margin <= 1.0:
            raise ValueError('same_pt_ambiguity_margin must be between 0 and 1.')
        if not 0.0 <= same_pt_min_support_score <= 1.0:
            raise ValueError('same_pt_min_support_score must be between 0 and 1.')
        self.temperature = temperature
        self.seed = seed
        self.think = think
        self.max_validation_attempts = max_validation_attempts
        self.num_ctx = num_ctx
        self.pt_disagreement_gate = pt_disagreement_gate
        self.pt_disagreement_top_k = pt_disagreement_top_k
        self.same_pt_ambiguity_gate = same_pt_ambiguity_gate
        self.same_pt_ambiguity_margin = same_pt_ambiguity_margin
        self.same_pt_meaningful_modifier_margin = same_pt_meaningful_modifier_margin
        self.same_pt_min_support_score = same_pt_min_support_score
        self.request_timeout = 600.0
        self.keep_alive = '30m'
        self.debug = True
        self.retry_diagnostics = retry_diagnostics
        self.reliability_repair = reliability_repair
        self.last_reliability_responses = []
        self.last_repair_applied = False
        self.last_repair_type: str | None = None
        self.last_repair_actions: list[str] = []
        self.last_attempt_count = 0
        self.last_retry_diagnostics: list[dict[str, Any]] = []
        if client is None:
            from ollama import Client
            client = Client(host=self.host, timeout=self.request_timeout)
        self.client = client
        self.last_validation_fallback: dict[str, Any] = {}

    @staticmethod
    def _build_structured_output_schema(reviewer_input: ReviewerInput) -> dict[str, Any]:
        """
        Build an input-specific structured-output schema.

        A final decision must assess every supplied
        candidate. The candidate IDs and candidate count
        are known only at runtime, so those constraints
        are added before sending the schema to Ollama.
        """
        schema = deepcopy(REVIEWER_DECISION_ADAPTER.json_schema())
        definitions = schema['$defs']
        available_candidate_ids = [candidate.candidate_id for candidate in reviewer_input.aggregation_result.candidates]
        if len(available_candidate_ids) != len(set(available_candidate_ids)):
            raise ValueError('ReviewerInput contains duplicate candidate IDs.')
        candidate_count = len(available_candidate_ids)
        positive_component_ids = [component.component_id for component in reviewer_input.input_interpretation.asserted_components if not component.is_negated]
        if len(positive_component_ids) != len(set(positive_component_ids)):
            raise ValueError('InputInterpretation contains duplicate positive component IDs.')
        if available_candidate_ids:
            assessment_id_schema = definitions['CandidateAssessment']['properties']['candidate_id']
            assessment_id_schema['enum'] = available_candidate_ids
            request_id_schema = definitions['EvidenceRequest']['properties']['candidate_ids_of_interest']['items']
            request_id_schema['enum'] = available_candidate_ids
        if positive_component_ids:
            component_match_definition = definitions['ComponentCandidateMatch']
            component_match_definition['properties']['component_id']['enum'] = positive_component_ids
            if available_candidate_ids:
                component_match_definition['properties']['candidate_id']['enum'] = available_candidate_ids
        decision_definition_names = ('MatchedSingleReviewerDecision', 'MatchedMultipleReviewerDecision', 'NoSupportedMatchReviewerDecision', 'RequestMoreEvidenceReviewerDecision', 'HumanReviewReviewerDecision')
        for definition_name in decision_definition_names:
            definition = definitions[definition_name]
            properties = definition['properties']
            properties['term_id']['const'] = reviewer_input.term_id
            properties['review_round']['const'] = reviewer_input.review_round
            if available_candidate_ids:
                selected_items = properties['selected_candidate_ids']['items']
                selected_items['enum'] = available_candidate_ids
                human_review_items = properties['human_review_candidate_ids']['items']
                human_review_items['enum'] = available_candidate_ids
        assessment_required_definition_names = ('MatchedSingleReviewerDecision', 'MatchedMultipleReviewerDecision', 'NoSupportedMatchReviewerDecision', 'HumanReviewReviewerDecision')
        for definition_name in assessment_required_definition_names:
            definition = definitions[definition_name]
            assessment_schema = definition['properties']['candidate_assessments']
            assessment_schema.pop('default', None)
            assessment_schema['minItems'] = candidate_count
            assessment_schema['maxItems'] = candidate_count
            required_fields = definition.setdefault('required', [])
            if 'candidate_assessments' not in required_fields:
                required_fields.append('candidate_assessments')
        matched_component_requirements = {'MatchedSingleReviewerDecision': 1, 'MatchedMultipleReviewerDecision': len(positive_component_ids)}
        for definition_name, required_match_count in matched_component_requirements.items():
            definition = definitions[definition_name]
            component_match_schema = definition['properties']['component_matches']
            component_match_schema.pop('default', None)
            component_match_schema['minItems'] = required_match_count
            component_match_schema['maxItems'] = required_match_count
            required_fields = definition.setdefault('required', [])
            if 'component_matches' not in required_fields:
                required_fields.append('component_matches')
        nonmatched_definition_names = ('NoSupportedMatchReviewerDecision', 'RequestMoreEvidenceReviewerDecision', 'HumanReviewReviewerDecision')
        for definition_name in nonmatched_definition_names:
            component_match_schema = definitions[definition_name]['properties']['component_matches']
            component_match_schema['maxItems'] = 0
        if reviewer_input.input_interpretation.structure_type == InputStructureType.AMBIGUOUS_CONCEPT:
            union_key = None
            if 'oneOf' in schema:
                union_key = 'oneOf'
            elif 'anyOf' in schema:
                union_key = 'anyOf'
            if union_key is None:
                raise ValueError('Could not locate the ReviewerDecision union in the generated JSON schema.')
            human_review_ref = '#/$defs/HumanReviewReviewerDecision'
            schema[union_key] = [option for option in schema[union_key] if option.get('$ref') == human_review_ref]
            if not schema[union_key]:
                raise ValueError('The generated ReviewerDecision schema does not contain the HumanReviewReviewerDecision definition.')
            discriminator = schema.get('discriminator')
            if discriminator is not None:
                discriminator['mapping'] = {ReviewerDecisionType.HUMAN_REVIEW.value: human_review_ref}
        requires_focused_component_retrieval = reviewer_input.input_interpretation.structure_type == InputStructureType.MULTIPLE_CONCEPTS and reviewer_input.review_round == 1 and (candidate_count < len(positive_component_ids))
        if requires_focused_component_retrieval:
            union_key = None
            if 'oneOf' in schema:
                union_key = 'oneOf'
            elif 'anyOf' in schema:
                union_key = 'anyOf'
            if union_key is None:
                raise ValueError('Could not locate the ReviewerDecision union in the generated JSON schema.')
            request_more_evidence_ref = '#/$defs/RequestMoreEvidenceReviewerDecision'
            schema[union_key] = [option for option in schema[union_key] if option.get('$ref') == request_more_evidence_ref]
            if not schema[union_key]:
                raise ValueError('The generated ReviewerDecision schema does not contain the RequestMoreEvidenceReviewerDecision definition.')
            discriminator = schema.get('discriminator')
            if discriminator is not None:
                discriminator['mapping'] = {ReviewerDecisionType.REQUEST_MORE_EVIDENCE.value: request_more_evidence_ref}
        return schema

    def _call_ollama(self, messages: list[dict[str, str]], reviewer_input: ReviewerInput) -> str:
        """
        Call Ollama once and report timing information.
        """
        structured_schema = self._build_structured_output_schema(reviewer_input)
        started_at = perf_counter()
        candidate_count = len(reviewer_input.aggregation_result.candidates)
        max_generated_tokens = min(4096, max(1024, 320 + 220 * candidate_count))
        try:
            response = self.client.chat(model=self.model, messages=messages, format=structured_schema, options={'temperature': self.temperature, 'seed': self.seed, 'num_predict': max_generated_tokens, 'num_ctx': self.num_ctx}, think=self.think, stream=False, keep_alive=self.keep_alive)
        except Exception as exc:
            elapsed = perf_counter() - started_at
            raise ReviewerLLMConnectionError(f'The local Ollama reviewer could not complete the request.\nTerm: {reviewer_input.term_id}\nElapsed: {elapsed:.2f} seconds\nUnderlying error: {type(exc).__name__}: {exc}') from exc
        elapsed = perf_counter() - started_at
        if self.debug:
            total_duration = (getattr(response, 'total_duration', 0) or 0) / 1000000000
            load_duration = (getattr(response, 'load_duration', 0) or 0) / 1000000000
            prompt_duration = (getattr(response, 'prompt_eval_duration', 0) or 0) / 1000000000
            generation_duration = (getattr(response, 'eval_duration', 0) or 0) / 1000000000
            prompt_tokens = getattr(response, 'prompt_eval_count', None)
            generated_tokens = getattr(response, 'eval_count', None)
            print(f'\n[OLLAMA REVIEW]\nTerm: {reviewer_input.term_id}\nWall time: {elapsed:.2f}s\nOllama total: {total_duration:.2f}s\nModel loading: {load_duration:.2f}s\nPrompt evaluation: {prompt_duration:.2f}s\nGeneration: {generation_duration:.2f}s\nPrompt tokens: {prompt_tokens}\nGenerated tokens: {generated_tokens}\n')
        content = response.message.content
        if not content or not content.strip():
            raise ReviewerLLMOutputError(f"Ollama returned an empty reviewer response for {reviewer_input.term_id}. Done reason: {getattr(response, 'done_reason', None)}. Generated tokens: {getattr(response, 'eval_count', None)}.")
        return content

    @staticmethod
    def _validation_feedback(error: Exception | None) -> str:
        """Create compact feedback for a correction attempt."""
        if error is None:
            return 'No validation details were available.'
        return str(error).strip()[:3000]

    @classmethod
    def _build_correction_message(cls, reviewer_input: ReviewerInput, error: Exception) -> str:
        """
        Provide concise, error-specific correction
        instructions without repeating the system prompt.
        """
        validation_feedback = cls._validation_feedback(error)
        candidate_ids = ', '.join((candidate.candidate_id for candidate in reviewer_input.aggregation_result.candidates))
        component_lines = '\n'.join((f'- {component.component_id}: {component.component_text}' for component in reviewer_input.input_interpretation.asserted_components if not component.is_negated))
        targeted_instruction = 'Correct the fields identified by the validation error.'
        if 'NO_SUPPORTED_MATCH is premature' in validation_feedback:
            targeted_instruction = 'Return request_more_evidence with expand_candidates. Do not return no_supported_match again.'
        elif 'AMBIGUOUS_CONCEPT requires' in validation_feedback:
            targeted_instruction = 'Return human_review. Assess every plausible interpretation as uncertain and include its ID in human_review_candidate_ids.'
        elif 'does not cover every positively asserted' in validation_feedback:
            targeted_instruction = 'Return request_more_evidence with focused_search for the missing component. selected_candidate_ids and component_matches must be empty.'
        elif 'Multiple-concept inputs must use focused_search' in validation_feedback:
            targeted_instruction = 'Replace expand_candidates with focused_search. query_term must be the missing asserted component, not the complete input.'
        elif 'UMLS CUIs that were not supplied' in validation_feedback:
            targeted_instruction = 'Remove every UMLS CUI that is absent from the supplied evidence.'
        return f'The preceding structured decision failed deterministic validation.\n\nVALIDATION ERROR\n{validation_feedback}\n\nREQUIRED CORRECTION\n{targeted_instruction}\n\nAVAILABLE CANDIDATE IDS\n{candidate_ids}\n\nPOSITIVELY ASSERTED COMPONENTS\n{component_lines}\n\nAlways obey these consistency rules:\n- Preserve term_id and review_round.\n- Use only available candidate IDs.\n- For a matched decision, selected IDs, supported assessments and component_matches candidate IDs must be identical.\n- Select at most one synonymous LLT from the same PT for one clinical concept.\n- Final and human-review decisions must assess every supplied candidate.\n- Non-matched decisions must have empty component_matches.\n- Do not invent LLT codes, PT codes, CUIs, scores or clinical facts.\n\nReturn the complete corrected structured ReviewerDecision only.'

    def _record_deterministic_repair(self, *, repair_type: str, repair_actions: list[str]) -> None:
        """
        Record deterministic structural repairs for
        reviewer metrics and optional diagnostics.
        """
        if not repair_actions:
            return
        trace.emit('deterministic_repair', {'repair_type': repair_type, 'repair_actions': repair_actions, 'attempt': self.last_attempt_count})
        self.last_repair_applied = True
        existing_types = set()
        if self.last_repair_type:
            existing_types.update(self.last_repair_type.split('+'))
        existing_types.add(repair_type)
        self.last_repair_type = '+'.join(sorted(existing_types))
        for repair_action in repair_actions:
            if repair_action not in self.last_repair_actions:
                self.last_repair_actions.append(repair_action)
        if not self.retry_diagnostics:
            return
        print('\n' + '-' * 80)
        print('DETERMINISTIC STRUCTURAL REPAIR')
        print('-' * 80)
        print(f'Repair type: {repair_type}')
        for repair_action in repair_actions:
            print(f'- {repair_action}')
        print('-' * 80)

    @staticmethod
    def _repair_raw_matched_decision(reviewer_input: ReviewerInput, raw_content: str) -> tuple[str, list[str]]:
        """
        Repair duplicated, omitted, or inconsistent
        candidate assessments before Pydantic parsing.

        For final matched decisions,
        selected_candidate_ids is treated as the
        authoritative final selection.

        This repair:
        - deduplicates candidate assessments;
        - reconstructs omitted assessments;
        - marks selected candidates as supported;
        - marks unselected candidates as not_supported;
        - preserves unknown candidate IDs so validation
          can reject them.
        """
        try:
            payload = json.loads(raw_content)
        except (json.JSONDecodeError, TypeError):
            return (raw_content, [])
        if not isinstance(payload, dict):
            return (raw_content, [])
        matched_decision_types = {ReviewerDecisionType.MATCHED_SINGLE.value, ReviewerDecisionType.MATCHED_MULTIPLE.value}
        if payload.get('decision_type') not in matched_decision_types:
            return (raw_content, [])
        selected_candidate_ids_value = payload.get('selected_candidate_ids')
        assessments_value = payload.get('candidate_assessments')
        if not isinstance(selected_candidate_ids_value, list):
            return (raw_content, [])
        if not isinstance(assessments_value, list):
            return (raw_content, [])
        selected_candidate_ids = {candidate_id for candidate_id in selected_candidate_ids_value if isinstance(candidate_id, str)}
        candidates = reviewer_input.aggregation_result.candidates
        available_candidate_ids = [candidate.candidate_id for candidate in candidates]
        available_candidate_id_set = set(available_candidate_ids)
        assessment_by_id: dict[str, dict[str, Any]] = {}
        unrecognized_assessments: list[Any] = []
        repair_actions: list[str] = []
        for assessment_value in assessments_value:
            if not isinstance(assessment_value, dict):
                unrecognized_assessments.append(assessment_value)
                continue
            candidate_id = assessment_value.get('candidate_id')
            if not isinstance(candidate_id, str) or candidate_id not in available_candidate_id_set:
                unrecognized_assessments.append(assessment_value)
                continue
            assessment_copy = dict(assessment_value)
            existing_assessment = assessment_by_id.get(candidate_id)
            if existing_assessment is None:
                assessment_by_id[candidate_id] = assessment_copy
                continue
            repair_actions.append(f'Removed duplicate candidate assessment for {candidate_id}.')
            existing_rationale = str(existing_assessment.get('rationale', '')).strip()
            new_rationale = str(assessment_copy.get('rationale', '')).strip()
            if len(new_rationale) > len(existing_rationale):
                assessment_by_id[candidate_id] = assessment_copy
        repaired_assessments: list[dict[str, Any]] = []
        for candidate_id in available_candidate_ids:
            expected_verdict = CandidateReviewVerdict.SUPPORTED.value if candidate_id in selected_candidate_ids else CandidateReviewVerdict.NOT_SUPPORTED.value
            assessment = assessment_by_id.get(candidate_id)
            if assessment is None:
                repair_actions.append(f'Reconstructed missing candidate assessment for {candidate_id}.')
                if expected_verdict == CandidateReviewVerdict.SUPPORTED.value:
                    rationale = 'This candidate appears in selected_candidate_ids. Its omitted supported assessment was reconstructed deterministically.'
                else:
                    rationale = 'This candidate was not selected in the final reviewer decision. Its omitted not-supported assessment was reconstructed deterministically.'
                assessment = {'assessment_authority': 'deterministic_reviewer_repair', 'candidate_id': candidate_id, 'verdict': expected_verdict, 'rationale': rationale}
            elif assessment.get('verdict') != expected_verdict:
                previous_verdict = assessment.get('verdict')
                repair_actions.append(f'Corrected candidate assessment for {candidate_id}: {previous_verdict} -> {expected_verdict}.')
                assessment['verdict'] = expected_verdict
                assessment['assessment_authority'] = 'deterministic_reviewer_repair'
                if expected_verdict == CandidateReviewVerdict.SUPPORTED.value:
                    assessment['rationale'] = 'This candidate was selected as the final supported LLT. The verdict was aligned deterministically with selected_candidate_ids.'
                else:
                    assessment['rationale'] = 'This candidate was not selected as the final representative LLT. The verdict was aligned deterministically with selected_candidate_ids.'
            repaired_assessments.append(assessment)
        repaired_assessments.extend(unrecognized_assessments)
        payload['candidate_assessments'] = repaired_assessments
        if not repair_actions:
            return (raw_content, [])
        return (json.dumps(payload, ensure_ascii=False), repair_actions)

    @staticmethod
    def _canonicalize_matched_decision(reviewer_input: ReviewerInput, reviewer_decision: ReviewerDecision) -> tuple[ReviewerDecision, list[str]]:
        """
        Repair redundant structural fields in a final
        matched decision.

        The LLM chooses selected_candidate_ids and maps
        them to input components. Deterministic code then:

        1. Resolves the authoritative PT for each selected
           LLT from ReviewerInput.
        2. Makes candidate-assessment verdicts consistent
           with selected_candidate_ids.

        This method does not repair unknown candidate IDs,
        missing assessments, invalid component IDs, multiple
        same-PT selections, or other clinical decision errors.
        Those remain validation failures.
        """
        matched_decision_types = {ReviewerDecisionType.MATCHED_SINGLE, ReviewerDecisionType.MATCHED_MULTIPLE}
        if reviewer_decision.decision_type not in matched_decision_types:
            return (reviewer_decision, [])
        candidate_by_id = {candidate.candidate_id: candidate for candidate in reviewer_input.aggregation_result.candidates}
        selected_candidate_ids = set(reviewer_decision.selected_candidate_ids)
        if selected_candidate_ids - set(candidate_by_id):
            return (reviewer_decision, [])
        repair_actions: list[str] = []
        repaired_component_matches = []
        for component_match in reviewer_decision.component_matches:
            candidate = candidate_by_id.get(component_match.candidate_id)
            if candidate is None:
                repaired_component_matches.append(component_match)
                continue
            expected_pt_code = str(candidate.pt_code)
            provided_pt_code = str(component_match.pt_code)
            if provided_pt_code != expected_pt_code:
                repair_actions.append(f'Corrected component-match PT for {component_match.candidate_id}: {provided_pt_code} -> {expected_pt_code}')
                component_match = component_match.model_copy(update={'pt_code': expected_pt_code})
            repaired_component_matches.append(component_match)
        repaired_assessments = []
        for assessment in reviewer_decision.candidate_assessments:
            if assessment.candidate_id in selected_candidate_ids:
                expected_verdict = CandidateReviewVerdict.SUPPORTED
            else:
                expected_verdict = CandidateReviewVerdict.NOT_SUPPORTED
            if assessment.verdict != expected_verdict:
                repair_actions.append(f'Corrected assessment verdict for {assessment.candidate_id}: {assessment.verdict.value} -> {expected_verdict.value}')
                assessment = assessment.model_copy(update={'verdict': expected_verdict})
            repaired_assessments.append(assessment)
        if not repair_actions:
            return (reviewer_decision, [])
        repaired_decision = reviewer_decision.model_copy(update={'component_matches': repaired_component_matches, 'candidate_assessments': repaired_assessments})
        return (repaired_decision, repair_actions)

    @staticmethod
    def _repair_premature_no_match(reviewer_input: ReviewerInput, reviewer_decision: ReviewerDecision, validation_error: Exception) -> ReviewerDecision | None:
        """
        Convert a premature no-supported-match
        decision into a deterministic candidate
        expansion request.

        This repair is applied only when the
        validation layer has explicitly identified
        that expansion is required.
        """
        error_message = str(validation_error)
        if reviewer_decision.decision_type != ReviewerDecisionType.NO_SUPPORTED_MATCH:
            return None
        if 'NO_SUPPORTED_MATCH is premature' not in error_message:
            return None
        aggregation_result = reviewer_input.aggregation_result
        conflicting_candidate_ids = [candidate.candidate_id for candidate in aggregation_result.candidates if candidate.fuzzy_evidence is not None and candidate.fuzzy_evidence.directionality_conflict]
        fuzzy_count = aggregation_result.fuzzy_candidate_count
        semantic_count = aggregation_result.semantic_candidate_count
        if conflicting_candidate_ids and fuzzy_count < 20:
            source = EvidenceSource.FUZZY
            current_source_count = fuzzy_count
        elif semantic_count < 20:
            source = EvidenceSource.SEMANTIC
            current_source_count = semantic_count
        else:
            return None
        requested_top_k = min(20, max(5, current_source_count + 1))
        evidence_request = EvidenceRequest(request_id=f'AUTO_EXPAND_{source.value.upper()}_R{reviewer_input.review_round}', source=source, action=EvidenceRequestAction.EXPAND_CANDIDATES, query_term=None, requested_top_k=requested_top_k, candidate_ids_of_interest=conflicting_candidate_ids, reason='The current candidates do not preserve the positively asserted directionality of the input. Additional candidates are required before finalizing no supported match.')
        return ReviewerDecision(term_id=reviewer_decision.term_id, review_round=reviewer_decision.review_round, decision_type=ReviewerDecisionType.REQUEST_MORE_EVIDENCE, selected_candidate_ids=[], component_matches=[], candidate_assessments=reviewer_decision.candidate_assessments, evidence_requests=[evidence_request], human_review_candidate_ids=[], decision_rationale='The reviewer rejected the current candidate because it contradicts the input directionality. The deterministic review policy converted the premature no-supported-match decision into a candidate-expansion request.', multiple_match_explanation=None, human_review_reason=None)

    def _record_retry_diagnostic(self, *, reviewer_input: ReviewerInput, attempt_number: int, stage: str, error: Exception, reviewer_decision: ReviewerDecision | None=None, raw_content: str | None=None) -> None:
        """
        Record and optionally print why a reviewer
        attempt failed.

        This method does not modify the decision or
        validation process.
        """
        diagnostic: dict[str, Any] = {'term_id': reviewer_input.term_id, 'review_round': reviewer_input.review_round, 'attempt_number': attempt_number, 'stage': stage, 'error_type': type(error).__name__, 'traceback': ''.join(traceback.format_exception(type(error), error, error.__traceback__)), 'error_message': self._validation_feedback(error)}
        if reviewer_decision is not None:
            diagnostic.update({'decision_type': reviewer_decision.decision_type.value, 'selected_candidate_ids': list(reviewer_decision.selected_candidate_ids), 'component_matches': [{'component_id': component_match.component_id, 'candidate_id': component_match.candidate_id} for component_match in reviewer_decision.component_matches], 'candidate_assessments': [{'candidate_id': assessment.candidate_id, 'verdict': assessment.verdict.value} for assessment in reviewer_decision.candidate_assessments], 'evidence_requests': [{'request_id': request.request_id, 'source': request.source.value, 'action': request.action.value, 'query_term': request.query_term} for request in reviewer_decision.evidence_requests], 'human_review_candidate_ids': list(reviewer_decision.human_review_candidate_ids)})
        elif raw_content:
            diagnostic['raw_response_preview'] = raw_content.strip()[:1500]
        self.last_retry_diagnostics.append(diagnostic)
        trace.emit('validation_failed', {k:v for k,v in diagnostic.items() if k!='raw_response_preview'}, status='failed')
        if not self.retry_diagnostics:
            return
        print('\n' + '-' * 80)
        print('REVIEWER RETRY DIAGNOSTIC')
        print('-' * 80)
        print(f'Term: {reviewer_input.term_id}')
        print(f'Review round: {reviewer_input.review_round}')
        print(f'Failed attempt: {attempt_number}/{self.max_validation_attempts}')
        print(f'Failure stage: {stage}')
        print(f'Error type: {type(error).__name__}')
        print('Validation error:')
        print(self._validation_feedback(error))
        if reviewer_decision is not None:
            print(f'Attempted decision type: {reviewer_decision.decision_type.value}')
            print(f'Selected candidates: {reviewer_decision.selected_candidate_ids}')
            supported_ids = [assessment.candidate_id for assessment in reviewer_decision.candidate_assessments if assessment.verdict.value == 'supported']
            print(f'Candidates assessed as supported: {supported_ids}')
            component_candidate_ids = [match.candidate_id for match in reviewer_decision.component_matches]
            print(f'Candidates in component_matches: {component_candidate_ids}')
        elif raw_content:
            print('Raw-response preview:')
            print(raw_content.strip()[:1500])
        print('-' * 80)

    def review(self, reviewer_input: ReviewerInput) -> ReviewerDecision:
        """Generate and validate one reviewer decision."""
        self.last_attempt_count = 0
        self.last_retry_diagnostics = []
        self.last_repair_applied = False
        self.last_repair_type = None
        self.last_repair_actions = []
        self.last_validation_fallback = {}
        clinical_rejections = []
        self.last_reliability_responses = []
        messages = [dict(message) for message in build_reviewer_messages(reviewer_input)]
        last_error: Exception | None = None
        for attempt_number in range(1, self.max_validation_attempts + 1):
            self.last_attempt_count = attempt_number
            reviewer_decision: ReviewerDecision | None = None
            raw_content = None
            if self.debug:
                print(f'\nReviewer term: {reviewer_input.term_id}\nValidation attempt: {attempt_number}/{self.max_validation_attempts}')
            try:
                raw_content = self._call_ollama(messages, reviewer_input)
                trace.reported_json(raw_content, 'reviewer', attempt_number)
            except ReviewerLLMOutputError as exc:
                last_error = exc
                self._record_retry_diagnostic(reviewer_input=reviewer_input, attempt_number=attempt_number, stage='empty_response', error=exc)
                if attempt_number >= self.max_validation_attempts:
                    break
                messages.append({'role': 'user', 'content': 'Your previous response was empty. Return one complete structured ReviewerDecision object conforming to the supplied JSON schema. Do not return explanatory text, Markdown, or an empty response.'})
                continue
            try:
                duplicate_edits = []
                if self.reliability_repair:
                    from reviewer_reliability import consolidate_review_assessments
                    response_record = {'attempt': attempt_number, 'original_raw_response': raw_content}
                    self.last_reliability_responses.append(response_record)
                    raw_content, duplicate_edits = consolidate_review_assessments(reviewer_input, raw_content)
                    response_record.update(duplicate_consolidations=duplicate_edits, post_consolidation_raw_response=raw_content)
                    if duplicate_edits:
                        self._record_deterministic_repair(repair_type='consistent_unresolved_duplicate_assessments', repair_actions=['Consolidated consistent duplicate assessments for ' + x['candidate_id'] + '; retained all original rationales.' for x in duplicate_edits])
                raw_content, raw_repair_actions = self._repair_raw_matched_decision(reviewer_input=reviewer_input, raw_content=raw_content)
                if raw_repair_actions:
                    self._record_deterministic_repair(repair_type='raw_candidate_assessments', repair_actions=raw_repair_actions)
                reviewer_decision = REVIEWER_DECISION_ADAPTER.validate_json(raw_content)
                parsed_assessments = {a.candidate_id: a for a in reviewer_decision.candidate_assessments}
                if self.reliability_repair:
                    from assessment_scope import original_validated
                    response_record['assessment_scope_check'] = {'required_ids': sorted((c.candidate_id for c in reviewer_input.aggregation_result.candidates)), 'parsed_ids': sorted(parsed_assessments), 'missing_ids': sorted({c.candidate_id for c in reviewer_input.aggregation_result.candidates} - set(parsed_assessments)), 'authority': 'existing_original_scope_contract; no assessment completion'}
                    original_validated(reviewer_input, reviewer_decision)
                reviewer_decision, structural_repairs = self._canonicalize_matched_decision(reviewer_input=reviewer_input, reviewer_decision=reviewer_decision)
                if structural_repairs:
                    self._record_deterministic_repair(repair_type='matched_decision_consistency', repair_actions=structural_repairs)
                validate_selected_clinical_mappings(reviewer_input, reviewer_decision)
                if self.pt_disagreement_gate:
                    reviewer_decision, pt_disagreement_repairs = apply_pt_disagreement_gate(reviewer_input=reviewer_input, reviewer_decision=reviewer_decision, candidate_pt_top_k=self.pt_disagreement_top_k)
                    if pt_disagreement_repairs:
                        self._record_deterministic_repair(repair_type='pt_reranker_disagreement', repair_actions=pt_disagreement_repairs)
                if self.same_pt_ambiguity_gate:
                    reviewer_decision, ambiguity_repairs = apply_same_pt_ambiguity_gate(reviewer_input=reviewer_input, reviewer_decision=reviewer_decision, score_margin=self.same_pt_ambiguity_margin, meaningful_modifier_margin=self.same_pt_meaningful_modifier_margin, minimum_support_score=self.same_pt_min_support_score)
                    if ambiguity_repairs:
                        self._record_deterministic_repair(repair_type='same_pt_llt_ambiguity', repair_actions=ambiguity_repairs)
                validated_decision = validate_reviewer_decision(reviewer_input, reviewer_decision)
                if duplicate_edits:
                    self.last_validation_fallback = {'version': '1.9.reliability.1', 'authority': 'deterministic_duplicate_assessment_review', 'failure_category': 'consistent_duplicate_assessments', 'force_human_review': True, 'retry_exhausted': False, 'attempt_count': self.last_attempt_count, 'consolidations': duplicate_edits, 'original_response_preserved': True}
                return validated_decision.model_copy(update={'candidate_assessments': [a.model_copy(update={'assessment_authority': 'deterministic_reviewer_repair'}) if a != parsed_assessments.get(a.candidate_id) else a for a in validated_decision.candidate_assessments]})
            except (ValidationError, ReviewerDecisionValidationError) as exc:
                last_error = exc
                if isinstance(exc, ClinicalMappingValidationError):
                    clinical_rejections.append({'attempt': attempt_number, 'candidate_ids': exc.candidate_ids, 'reasons': exc.reasons})
                failure_stage = 'schema_validation' if isinstance(exc, ValidationError) else 'deterministic_validation'
                self._record_retry_diagnostic(reviewer_input=reviewer_input, attempt_number=attempt_number, stage=failure_stage, error=exc, reviewer_decision=reviewer_decision, raw_content=raw_content)
                if reviewer_decision is not None:
                    repaired_decision = self._repair_premature_no_match(reviewer_input, reviewer_decision, exc)
                    if repaired_decision is not None:
                        validated_repaired_decision = validate_reviewer_decision(reviewer_input, repaired_decision)
                        self._record_deterministic_repair(repair_type='premature_no_supported_match', repair_actions=['Converted a premature no_supported_match decision into request_more_evidence.'])
                        return validated_repaired_decision
                if attempt_number >= self.max_validation_attempts:
                    break
                messages.append({'role': 'user', 'content': self._build_correction_message(reviewer_input, exc)})
        # Recover only an already-unresolved response after the unchanged retry
        # budget. Never discard conflicting assessments or create match support.
        from reviewer_reliability import consolidate_review_assessments, terminal_clinical_sequence
        if raw_content:
            try:
                repaired_raw, edits = consolidate_review_assessments(reviewer_input, raw_content)
                if edits:
                    recovered = REVIEWER_DECISION_ADAPTER.validate_json(repaired_raw)
                    recovered = validate_reviewer_decision(reviewer_input, recovered)
                    # The gate requires full assessment scope even where the
                    # reviewer validator permits omission. Use the existing
                    # deterministic review-only shortlist builder; no missing
                    # model assessment is invented or promoted to supported.
                    from dataclasses import replace
                    from adaptive_review_policy import prepare_policy_evidence, apply_policy
                    from llt_selection_gate import _deferred_decision
                    missing = sorted({c.candidate_id for c in reviewer_input.aggregation_result.candidates}
                                     - {a.candidate_id for a in recovered.candidate_assessments})
                    shortlist = apply_policy(replace(prepare_policy_evidence(reviewer_input, ()), force_human_review=True))
                    fallback = _deferred_decision(reviewer_input=reviewer_input, reviewer_decision=recovered,
                        human_review_candidates=shortlist.candidates, pt_resolved=False,
                        decision_rationale='Reviewer requested human review; repeated assessment entries consolidated after retry exhaustion. No mapping is authorized.',
                        human_review_reason='Review unresolved candidates. Missing model assessments remain identified in the recovery diagnostics.')
                    fallback = validate_reviewer_decision(reviewer_input, fallback)
                    self.last_validation_fallback = {
                        'version': 'terminal-review-recovery/1', 'retry_exhausted': True,
                        'force_human_review': True, 'attempt_count': self.last_attempt_count,
                        'authority': 'deterministic_duplicate_assessment_review',
                        'failure_category': 'consistent_duplicate_assessments',
                        'missing_model_assessment_ids': missing,
                        'assessment_completion_authority': 'existing_deterministic_review_only_shortlist',
                        'consolidations': edits, 'original_response_preserved': True,
                    }
                    self._record_deterministic_repair(repair_type='terminal_consistent_unresolved_duplicates',
                        repair_actions=['Consolidated consistent duplicate assessments after retry exhaustion; human review only.'])
                    return fallback
            except (ValidationError, ReviewerDecisionValidationError):
                pass  # An inconsistent/incomplete response remains an execution error.
        terminal_clinical = terminal_clinical_sequence(clinical_rejections, self.last_attempt_count)
        if len(clinical_rejections) == self.max_validation_attempts and len(clinical_rejections) >= 2 or terminal_clinical:
            from dataclasses import replace
            from adaptive_review_policy import prepare_policy_evidence, apply_policy
            from llt_selection_gate import _deferred_decision
            evidence = replace(prepare_policy_evidence(reviewer_input, ()), force_human_review=True)
            shortlist = apply_policy(evidence)
            fallback = _deferred_decision(reviewer_input=reviewer_input, reviewer_decision=reviewer_decision, human_review_candidates=shortlist.candidates, pt_resolved=False, decision_rationale='Repeated deterministic clinical validation rejection; no automatic mapping is authorized.', human_review_reason='Reviewer validation exhausted. Review compatible alternatives or leave unresolved if none remain.')
            fallback = validate_reviewer_decision(reviewer_input, fallback)
            self.last_validation_fallback = {'version': '1.3.1', 'retry_exhausted': True, 'authority': 'deterministic_clinical_validation_fallback', 'attempt_count': self.last_attempt_count, 'rejected_candidate_ids': sorted({i for r in clinical_rejections for i in r['candidate_ids']}), 'validation_reasons': sorted({i for r in clinical_rejections for i in r['reasons']}), 'rejections': clinical_rejections, 'unresolved_without_candidates': not shortlist.candidates}
            if terminal_clinical:
                self.last_validation_fallback.update(version='1.9.reliability.1', failure_category='terminal_repeated_clinical_rejection', eligibility_rule='two_consecutive_final_structurally_valid_clinical_rejections', preceding_failure_types=[{'attempt': d['attempt_number'], 'stage': d['stage'], 'error_type': d['error_type']} for d in self.last_retry_diagnostics])
            return fallback
        raise ReviewerLLMOutputError(f'The reviewer model did not produce a decision that satisfied the schema and deterministic validation rules after {self.max_validation_attempts} attempts. Last validation error: {self._validation_feedback(last_error)}')

    def __call__(self, state: StandardizationState) -> dict[str, Any]:
        """Run as a LangGraph-compatible reviewer node."""
        reviewer_input = state.get('reviewer_input')
        if reviewer_input is None:
            raise ValueError("The reviewer LLM node requires state['reviewer_input'].")
        if not isinstance(reviewer_input, ReviewerInput):
            reviewer_input = ReviewerInput.model_validate(reviewer_input)
        decision = self.review(reviewer_input)
        return {'reviewer_decision': decision, 'reviewer_failure_diagnostics': self.last_validation_fallback}
