from __future__ import annotations
from runtime_resources import resource as asp_resource
import decision_recorder as trace
from copy import deepcopy
import os
from pathlib import Path
from typing import Any
from dotenv import load_dotenv
from pydantic import ValidationError
from schemas import InputInterpretation
from state import StandardizationState
PROJECT_ROOT = Path(__file__).resolve().parent
INPUT_INTERPRETATION_PROMPT = '\nYou are the input-interpretation component of a\nMedDRA standardization workflow.\n\nYour only task is to identify the clinical structure\nof the supplied input term before MedDRA candidates\nare reviewed.\n\nYou do not select MedDRA candidates and do not\nproduce LLT or PT codes.\n\nSTRUCTURE TYPES\n\nUse single_concept when the input expresses one\nclinical event, condition, symptom, investigation\nresult or other clinical concept.\n\nA plural expression can still be one concept. A\nbroad category can still be one concept. Several\nwords describing one event do not make it multiple\nconcepts.\n\nUse multiple_concepts only when the input explicitly\nasserts two or more separate clinical concepts as\nsimultaneously present.\n\nMultiple possible interpretations of one expression\nare not multiple_concepts.\n\nUse negated_non_event when the complete input states\nthat an event did not occur, was absent, was denied,\nwas ruled out or was not reported.\n\nUse ambiguous_concept only when ambiguity is explicit\nfrom the wording of the input itself.\n\nIf the input contains one context-free expression\nwhose alternative meanings would become apparent\nonly after examining terminology candidates, classify\nit as single_concept. The later candidate reviewer is\nresponsible for deciding whether those competing\ncandidate interpretations require human review.\n\nThe input interpreter identifies the number and\nassertion status of explicit input components. It\ndoes not perform the final clinical disambiguation.\n\nBefore selecting single_concept, perform an explicit\nambiguity check.\n\nAsk whether the complete context-free expression\ncould commonly refer to different clinical concepts,\nsuch as:\n\n- a disease versus a symptom or sensation;\n- a clinical condition versus a procedure;\n- an anatomical concept versus a clinical event; or\n- two different established medical meanings.\n\nIf two or more such interpretations remain plausible,\nuse ambiguous_concept even if one interpretation is\nmore common.\n\nDo not silently choose the most common interpretation.\nDo not add clinical context that is absent from the\ninput.\n\nUse single_concept when the expression is broad or\nnonspecific but still refers to one clinical concept.\nBroadness alone is not ambiguity. Candidate variation\nalone is also not ambiguity.\n\nCOMPONENT RULES\n\nA negated modifier is not a negated event: non-pathological fracture still\nasserts a fracture. Preserve the full non-/not scope and never equate different\nnegated modifiers. Preserve decimal values, comparison operators, exponents and\nunits. A numeric bound alone does not establish an abnormal increase/decrease.\nAcronym meanings require surrounding clinical/measurement context; counts,\npercentages and antibody measurements are distinct.\n\nGeneric contextual suffixes such as adverse events, adverse reactions and\nside effects do not assert a second independent event after a named clinical\nconcept. An X-associated/X-related event phrase describes X in context.\nKeep true coordinated concepts (conjunctions or semicolons) separate.\n\nFor every component:\n\n- use IDs COMPONENT_1, COMPONENT_2 and so on;\n- component_text must be copied exactly from the\n  normalized input;\n- do not paraphrase component_text;\n- do not use individual adjectives, modifiers or\n  isolated words as separate clinical components;\n- set is_negated=true only when that component is\n  explicitly negated.\n\nFor single_concept, return exactly one component.\n\nFor multiple_concepts, return at least two components,\none for each separately asserted clinical concept.\n\nFor ambiguous_concept, return exactly one component\ncontaining the complete ambiguous expression. Do not\nlist its possible interpretations as separately\nasserted components.\n\nThe interpretation_rationale must identify at least\ntwo plausible meanings and explain why the input\ndoes not resolve them.\n\nDo not classify an input as multiple_concepts merely\nbecause:\n\n- it is plural;\n- it describes a broad category;\n- it contains several words;\n- several MedDRA candidates might represent it;\n- it contains a modifier and a clinical noun; or\n- it could have several possible interpretations.\n\nA disease and its worsening, progression, severity or anatomical qualifier are\nnot independently asserted events just because each can retrieve a MedDRA term.\nExplain the assertion relationship using the original text. Do not silently\ndistribute a shared anatomical qualifier across components; identify its scope.\nGeneric tumor/neoplasm wording does not establish malignancy. Telegraphic wording\nwith unresolved attachment should remain ambiguous, not be expanded into several\nasserted diagnoses. Conjunctions are useful evidence but are neither necessary\nnor sufficient: independently asserted clauses and explicit shared context can\nalso support genuine compounds. Deterministic decomposition diagnostics are\nrecomputed from the source; never supply them as model-derived proof.\n\nOUTPUT RULES\n\nReturn exactly one InputInterpretation object.\n\nCopy term_id and normalized_term exactly. Do not\ninclude Markdown or text outside the structured\nobject.\n'.strip()

class InputInterpretationError(RuntimeError):
    """Base input-interpretation error."""

class InputInterpretationConnectionError(InputInterpretationError):
    """Raised when Ollama cannot complete the request."""

class InputInterpretationOutputError(InputInterpretationError):
    """
    Raised when Ollama repeatedly produces an invalid
    interpretation.
    """

class OllamaInputInterpreter:
    """
    Local Ollama-backed input interpreter and
    LangGraph-compatible node.
    """

    def __init__(self, model: str | None=None, host: str | None=None, *, temperature: float=0.0, seed: int=42, think: bool=False, max_validation_attempts: int=3, num_ctx: int=16384, client: Any=None) -> None:
        if client is None:
            load_dotenv(PROJECT_ROOT / '.env')
        self.model = model or os.getenv('OLLAMA_INTERPRETER_MODEL') or os.getenv('OLLAMA_REVIEWER_MODEL', 'qwen3:8b')
        self.host = host or os.getenv('OLLAMA_REVIEWER_HOST', 'http://127.0.0.1:11434')
        if max_validation_attempts < 1:
            raise ValueError('max_validation_attempts must be at least 1.')
        if num_ctx < 4096:
            raise ValueError('num_ctx must be at least 4096.')
        self.temperature = temperature
        self.seed = seed
        self.think = think
        self.num_ctx = num_ctx
        self.max_validation_attempts = max_validation_attempts
        if client is None:
            from ollama import Client
            client = Client(host=self.host)
        self.client = client

    @staticmethod
    def _structured_schema(*, term_id: str, normalized_term: str) -> dict[str, Any]:
        """
        Build an input-specific JSON schema that fixes
        term_id and normalized_term.
        """
        schema = deepcopy(InputInterpretation.model_json_schema())
        schema['properties']['term_id']['const'] = term_id
        schema['properties']['normalized_term']['const'] = normalized_term
        return schema

    @staticmethod
    def _messages(*, term_id: str, original_term: str, normalized_term: str) -> list[dict[str, str]]:
        user_message = f'Interpret the following input term.\n\nterm_id: {term_id}\noriginal_term: {original_term}\nnormalized_term: {normalized_term}\n\nReturn exactly one structured InputInterpretation object.'
        return [{'role': 'system', 'content': INPUT_INTERPRETATION_PROMPT}, {'role': 'user', 'content': user_message}]

    def _call_ollama(self, *, messages: list[dict[str, str]], term_id: str, normalized_term: str) -> str:
        schema = self._structured_schema(term_id=term_id, normalized_term=normalized_term)
        try:
            response = self.client.chat(model=self.model, messages=messages, format=schema, options={'temperature': self.temperature, 'seed': self.seed, 'num_ctx': self.num_ctx}, think=self.think, stream=False)
        except Exception as error:
            raise InputInterpretationConnectionError(f"The local Ollama input interpreter could not complete the request. Confirm that Ollama is running and that model '{self.model}' is installed.") from error
        content = response.message.content
        if content is None or not content.strip():
            raise InputInterpretationOutputError('Ollama returned an empty input interpretation.')
        return content

    @staticmethod
    def _validate_expected_values(interpretation: InputInterpretation, *, term_id: str, normalized_term: str) -> None:
        if interpretation.term_id != term_id:
            raise ValueError('InputInterpretation term_id does not match the state.')
        if interpretation.normalized_term != normalized_term:
            raise ValueError('InputInterpretation normalized_term does not match the normalization result.')

    @staticmethod
    def _correction_message(*, error: Exception, term_id: str, normalized_term: str) -> str:
        return f'Your preceding InputInterpretation failed validation.\n\nValidation error:\n{str(error)[:2000]}\n\nReturn a corrected complete object.\nterm_id must be exactly: {term_id}\nnormalized_term must be exactly: {normalized_term}\nEvery component_text must be copied from normalized_term.\nUse multiple_concepts only for separately asserted clinical concepts.\nA plural or multi-word expression is not automatically multiple concepts.\nBefore using single_concept, check whether the context-free expression has multiple established and mutually exclusive clinical meanings. Do not assume the most common meaning when the input provides no disambiguating context.'

    def interpret(self, *, term_id: str, original_term: str, normalized_term: str) -> InputInterpretation:
        messages = self._messages(term_id=term_id, original_term=original_term, normalized_term=normalized_term)
        last_error: Exception | None = None
        for attempt_number in range(1, self.max_validation_attempts + 1):
            raw_content = self._call_ollama(messages=messages, term_id=term_id, normalized_term=normalized_term)
            trace.reported_json(raw_content, 'interpreter', attempt_number)
            try:
                interpretation = InputInterpretation.model_validate_json(raw_content)
                self._validate_expected_values(interpretation, term_id=term_id, normalized_term=normalized_term)
                return interpretation
            except (ValidationError, ValueError) as error:
                trace.emit('validation_failed', {'attempt': attempt_number, 'error_type': type(error).__name__, 'message': str(error)}, status='failed')
                last_error = error
                if attempt_number >= self.max_validation_attempts:
                    break
                messages.extend([{'role': 'assistant', 'content': raw_content}, {'role': 'user', 'content': self._correction_message(error=error, term_id=term_id, normalized_term=normalized_term)}])
        raise InputInterpretationOutputError(f'The input-interpreter model did not produce a valid interpretation after {self.max_validation_attempts} attempts. Last validation error: {str(last_error)[:2000]}')

    def __call__(self, state: StandardizationState) -> dict[str, InputInterpretation]:
        term_input = state.get('term_input')
        if term_input is None:
            raise ValueError("Input interpretation requires state['term_input'].")
        normalization_result = state.get('normalization')
        if normalization_result is None:
            raise ValueError("Input interpretation requires state['normalization'].")
        interpretation = self.interpret(term_id=state['term_id'], original_term=term_input.term, normalized_term=normalization_result.normalized_term)
        return {'input_interpretation': interpretation}
