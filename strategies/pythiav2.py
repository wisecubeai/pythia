import json
import sys
import traceback
from typing import Union, List, Optional, Dict, Collection, Callable

import litellm

from _evaluators import SimpleEvaluator, BaseEvaluator, Evaluation, SingleClaimEvaluation
from models import HostedModel
from prompts import pythiav2, pythiav1
from extractors import PythiaV1Extractor
from prompts.pythiav2 import CheckingResponse, ExtractionResponse
from template import label_entailment, label_contradiction, label_neutral
from operator import and_, or_
import utils


class PythiaV2Evaluator(BaseEvaluator):
    AND_FUN = and_
    OR_FUN = or_

    def __init__(self, model: HostedModel, mode: Callable[[bool, bool], bool] = AND_FUN):
        self._model = model
        self.mode = mode

    @property
    def model(self) -> HostedModel:
        return self._model

    @staticmethod
    def _create_summary_extraction_call(text: str, question: Optional[str] = None) -> List[Dict[str, str]]:
        if question is None:
            prompt = pythiav1.GPT4_TRIPLET_EXTRACTION_PROMPT.format(text=text)
        else:
            prompt = pythiav1.GPT4_TRIPLET_EXTRACTION_PROMPT_Q.format(text=text, question=question)
        return [{
            "role": "user",
            "content": prompt
        }]

    @staticmethod
    def _create_summary_check_call(
            summary_triples: ExtractionResponse,
            reference_triples: ExtractionResponse
    ) -> List[Dict[str, str]]:
        summary_str = "\n".join(f'("{t.subject}", "{t.predicate}", "{t.object}")' for t in summary_triples.triples)
        reference_str = "\n".join(f'("{t.subject}", "{t.predicate}", "{t.object}")' for t in reference_triples.triples)
        checking_system = pythiav2.CHECKING_SYSTEM
        checking_message = pythiav2.CHECKING_TEMPLATE.format(
            response_triples=summary_str, reference_triples=reference_str)
        return [{
            "role": "system",
            "content": checking_system
        }, {
            "role": "user",
            "content": checking_message
        }]

    @staticmethod
    def _parse_claims(response: str) -> List[List[str]]:
        if not (response and len(response)):
            return []
        if '###' in response:
            kg_str = response[:response.index('###')]
        else:
            kg_str = response
        triplets = PythiaV1Extractor._parse_claim_triplets(kg_str)
        return triplets

    @staticmethod
    def _process_summary_check_response(response: CheckingResponse) -> Evaluation:
        metrics = {}
        claims = []
        for claim in response.claims:
            metrics[claim.category] = metrics.get(claim.category, 0) + 1
            cat = claim.category
            single_claim = SingleClaimEvaluation(
                claim=[claim.claim.subject, claim.claim.predicate, claim.claim.object],
                className=str(cat),
                reasoning=claim.reasoning,
                category=str(cat)

            )
            claims.append(single_claim)
        metrics["accuracy"] = utils.calc_accuracy(metrics)
        verdict = "PASS" if metrics["accuracy"] >= 0.9 else "FAIL"
        return Evaluation(verdict=verdict, reasoning=response, metrics=metrics, claims=claims)

    def evaluate_summary(
            self,
            summary: str,
            reference: Union[str, List[str]],
            question: Optional[str] = None,
            **kwargs) -> Evaluation:
        if isinstance(reference, list):
            reference = "\n\n".join(reference)
        summary_triples, reference_triples = utils.parallel_model_call(
            [
                PythiaV2Evaluator._create_summary_extraction_call(summary),
                PythiaV2Evaluator._create_summary_extraction_call(summary)
            ],
            self.model,
            response_format=pythiav2.ExtractionResponse,
            temperature=0.0,
            seed=123,
            **kwargs)
        summary_triples = pythiav2.ExtractionResponse(**json.loads(summary_triples))
        reference_triples = pythiav2.ExtractionResponse(**json.loads(reference_triples))
        result = litellm.completion(
            model=self.model.model,
            api_key=self.model.api_key,
            api_base=self.model.api_base,
            messages=PythiaV2Evaluator._create_summary_check_call(summary_triples, reference_triples),
            response_format=pythiav2.CheckingResponse,
            temperature=0.0,
            seed=123,
            **kwargs
        )
        result = result.choices[0].message.content
        result = pythiav2.CheckingResponse(**json.loads(result))
        return self._process_summary_check_response(result)

    def batch_summary(
            self,
            summaries: Collection[str],
            references: Collection[List[str]],
            questions: Optional[Collection[Optional[str]]] = None,
            **kwargs) -> List[Evaluation]:
        summary_calls = [PythiaV2Evaluator._create_summary_extraction_call(s) for s in summaries]
        reference_calls = [PythiaV2Evaluator._create_summary_extraction_call(r) for r in references]
        summary_triple_strs = utils.parallel_model_call(
            summary_calls, self.model, response_format=ExtractionResponse, **kwargs)
        reference_triple_strs = utils.parallel_model_call(
            reference_calls, self.model, response_format=ExtractionResponse, **kwargs)
        summary_triple_sets = [pythiav2.ExtractionResponse(**json.loads(t)) for t in summary_triple_strs]
        reference_triple_sets = [pythiav2.ExtractionResponse(**json.loads(t)) for t in reference_triple_strs]
        triple_sets = zip(summary_triple_sets, reference_triple_sets)
        check_calls = [PythiaV2Evaluator._create_summary_check_call(s, r) for s, r in triple_sets]
        responses = utils.parallel_model_call(
            check_calls, self.model, response_format=pythiav2.CheckingResponse, **kwargs)
        parsed_responses = []
        for r in responses:
            try:
                parsed_responses.append(pythiav2.CheckingResponse(**json.loads(r)))
            except json.decoder.JSONDecodeError as jde:
                print(r, file=sys.stderr)
                print(traceback.format_exc(), file=sys.stderr)
                continue
        evaluations = [self._process_summary_check_response(r) for r in parsed_responses]
        return evaluations

    @staticmethod
    def _create_qa_qa_call(answer: str, question: str):
        qa_system = pythiav2.QA_SYSTEM
        qa_message = pythiav2.QA_TEMPLATE.format(
            answer=answer,
            question=question,
        )
        return [{
            "role": "system",
            "content": qa_system
        }, {
            "role": "user",
            "content": qa_message
        }]

    @staticmethod
    def _create_qa_ta_call(answer: str, context: str):
        ta_system = pythiav2.TA_SYSTEM
        ta_message = pythiav2.TA_TEMPLATE.format(
            answer=answer,
            context=context,
        )
        return [{
            "role": "system",
            "content": ta_system
        }, {
            "role": "user",
            "content": ta_message
        }]

    def _process_qa_response(self, qa_response: str, ta_response: Optional[str] = None):
        ta_verdict = ta_response is None or "True." in ta_response
        qa_verdict = "True." in qa_response
        verdict = "PASS" if self.mode(ta_verdict, qa_verdict) else "FAIL"
        return Evaluation(
            verdict=verdict,
            reasoning=f"QA reasoning: {qa_response}\nTA reasoning: {ta_response}",
            metrics={"score": 1.0 if verdict == "PASS" else 0.0}
        )

    def evaluate_qa(
            self,
            answer: str,
            question: str,
            context: Optional[Union[str, List[str]]] = None,
            **kwargs) -> Evaluation:
        if isinstance(context, list):
            context = "\n\n".join(context)
        calls = []
        if context is not None:
            calls.append(PythiaV2Evaluator._create_qa_ta_call(answer, context))
        calls.append(PythiaV2Evaluator._create_qa_qa_call(answer, question))
        results = utils.batch_model_call(calls, self.model)
        ta_result = None
        if context is not None:
            ta_result, qa_result = results
        else:
            qa_result, = results
        return self._process_qa_response(qa_result, ta_result)

    def batch_qa(
            self,
            answers: Collection[str],
            questions: Collection[str],
            contexts: Optional[Collection[Optional[List[str]]]] = None,
            **kwargs):
        assert len(answers) == len(questions)
        assert contexts is None or len(answers) == len(contexts)
        if contexts is not None:
            contexts = ["\n\n".join(cs) for cs in contexts]
            ta_calls = [PythiaV2Evaluator._create_qa_ta_call(a, c) for a, c in zip(answers, contexts)]
            ta_results = utils.batch_model_call(ta_calls, self.model)
        else:
            ta_results = [None] * len(answers)
        qa_calls = [PythiaV2Evaluator._create_qa_qa_call(a, q) for a, q in zip(answers, questions)]
        qa_results = utils.batch_model_call(qa_calls, self.model)
        evaluations = []
        for qar, tar in zip(qa_results, ta_results):
            evaluations.append(self._process_qa_response(qar, tar))
        return evaluations
