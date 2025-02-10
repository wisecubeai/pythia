"""
This module contains the code for the different kinds of AI hallucination evaluation implementations
"""
from abc import ABC, abstractmethod
from typing import Union, Optional, Dict, List, TypeVar, Generic, Sized, Collection

import litellm
from pydantic import BaseModel, Field

import utils
from models import HostedModel


############################################
# Evaluation classes
############################################


class SingleClaimEvaluation(BaseModel):
    """This is the dataclass for a single claim evaluation"""
    claim: List[str] = Field(default_factory=list)
    category: str = Field(default="N/A")
    reasoning: Optional[Union[str, List[str]]] = Field(default="")


class Evaluation(BaseModel):
    """This is the base class for the evaluation dataclass"""
    metrics: Dict[str, float] = Field(default_factory=dict)
    claims: List[SingleClaimEvaluation] = Field(default_factory=list)
    verdict: str = Field(default="N/A")
    reasoning: Optional[Union[str, List[str]]] = Field(default="")


############################################
# Evaluation classes
############################################


class SingleClaimEvaluation(BaseModel):
    class Config:
        populate_by_name = True  # Allows using 'classType' in input
        use_enum_values = True
    """This is the dataclass for a single claim evaluation"""
    claim: List[str] = Field(default_factory=list)
    category: str = Field(default="N/A")
    reasoning: Optional[Union[str, List[str]]] = Field(default="")
    className: str = Field(alias="class", default="N/A")




class Evaluation(BaseModel):
    """This is the base class for the evaluation dataclass"""
    metrics: Dict[str, float] = Field(default_factory=dict)
    claims: List[SingleClaimEvaluation] = Field(default_factory=list)
    verdict: str = Field(default="N/A")
    validatorsResults: List[Dict[str, dict]] = Field(default_factory=list)

    class Config:
        populate_by_name = True
        use_enum_values = True


############################################
# Base Evaluator classes
############################################


class BaseEvaluator(ABC):
    """This is the base class for evaluators"""

    @abstractmethod
    def evaluate_summary(
            self,
            summary: str,
            reference: Union[str, List[str]],
            question: Optional[str] = None,
            **kwargs) -> Evaluation:
        """
        This method evaluates a generated summary.

        :param summary: The generated text of the summary
        :param reference: The text or texts be summarized. This can be either a `str` or `List[str]`.
        :param question: This is an optional question that can be used as a focus of the hallucination evaluation. For
            example, if this is a summary of a set of patient documents then you can focus the evaluation with the
            question "What drugs have been prescribed?"
        :param kwargs: these arguments can be used to provide arguments for functions and methods used to evaluate. For
            example, if an evaluator uses a call to an LLM, the temperature can be supplied by the kwargs
        :returns: This method returns an evaluation object
        """
        pass

    @abstractmethod
    def batch_summary(
            self,
            summaries: Collection[str],
            references: Collection[List[str]],
            questions: Optional[Collection[Optional[str]]] = None,
            **kwargs) -> List[Evaluation]:
        """
        This method evaluates a batch of generated summaries.

        :param summaries: The generated texts of the summaries
        :param references: The texts summarized.
        :param questions: These are optional questions that can be used as a focus of the hallucination evaluation. For
            example, if there is a summary of a set of patient documents then you can focus the evaluation with the
            question "What drugs have been prescribed?"
        :param kwargs: these arguments can be used to provide arguments for functions and methods used to evaluate. For
            example, if an evaluator uses a call to an LLM, the temperature can be supplied by the kwargs
        :returns: This method returns a list of evaluation objects
        """
        pass

    @abstractmethod
    def evaluate_qa(
            self,
            answer: str,
            question: str,
            context: Optional[Union[str, List[str]]] = None,
            **kwargs) -> Evaluation:
        """
        This method evaluates an answer from a RAG QA system.

        :param answer: The generated text of the answer
        :param question: The question posed to the system
        :param context: The optional text or texts be used as context. This can be either a `str` or `List[str]`. If
            evaluating the output of a RAG-based QA system, the retrieved text(s) should be passed here.
        :param kwargs: these arguments can be used to provide arguments for functions and methods used to evaluate. For
            example, if an evaluator uses a call to an LLM, the temperature can be supplied by the kwargs
        :returns: This method returns an evaluation object
        """
        pass

    @abstractmethod
    def batch_qa(
            self,
            answers: Collection[str],
            questions: Collection[str],
            contexts: Optional[Collection[Optional[List[str]]]] = None,
            **kwargs):
        """
        This method evaluates an answer from a RAG QA system.

        :param answers: The generated texts of the answers
        :param questions: The questions posed to the system
        :param contexts: The optional texts be used as context. If evaluating the output of a RAG-based QA system, the
            retrieved text(s) should be passed here.
        :param kwargs: these arguments can be used to provide arguments for functions and methods used to evaluate. For
            example, if an evaluator uses a call to an LLM, the temperature can be supplied by the kwargs
        :returns: This method returns a list of evaluation objects
        """
        pass


class SimpleEvaluator(BaseEvaluator):
    @property
    @abstractmethod
    def model(self) -> HostedModel:
        pass

    @abstractmethod
    def _create_summary_call(
            self,
            summary: str,
            reference: Union[str, List[str]],
            question: Optional[str] = None) -> List[Dict[str, str]]:
        """
        This creates a call to an LLM to evaluate a generated summary.

        :param summary: The generated text of the summary
        :param reference: The text or texts be summarized. This can be either a `str` or `List[str]`.
        :param question: This is an optional question that can be used as a focus of the hallucination evaluation. For
            example, if this is a summary of a set of patient documents then you can focus the evaluation with the
            question "What drugs have been prescribed?"
        :returns: This method returns an list of messages for performing the summary evaluation
        """
        pass

    @abstractmethod
    def _create_qa_call(
            self,
            answer: str,
            question: str,
            context: Optional[Union[str, List[str]]] = None) -> List[Dict[str, str]]:
        """
        This creates a call to an LLM to evaluate an answer from a RAG QA system.

        :param answer: The generated text of the answer
        :param question: The question posed to the system
        :param context: The optional text or texts be used as context. This can be either a `str` or `List[str]`. If
            evaluating the output of a RAG-based QA system, the retrieved text(s) should be passed here.
        :returns: This method returns an list of messages for performing the QA evaluation
        """
        pass

    @abstractmethod
    def _process_summary_response(self, response: str) -> Evaluation:
        """
        This method processes the response from a call to the LLM, turning the summary response into an Evaluation

        :param response: the LLM response to be processed
        :return: the summary Evaluation (SimpleEvaluation)
        """
        pass

    @abstractmethod
    def _process_qa_response(self, response: str) -> Evaluation:
        """
        This method processes the response from a call to the LLM, turning the QA response into an Evaluation

        :param response: the LLM response to be processed
        :return: the summary Evaluation (SimpleEvaluation)
        """
        pass

    def evaluate_summary(
            self,
            summary: str,
            reference: Union[str, List[str]],
            question: Optional[str] = None,
            **kwargs) -> Evaluation:
        messages = self._create_summary_call(summary, reference, question)
        result = litellm.completion(
            model=self.model.model,
            api_key=self.model.api_key,
            api_base=self.model.api_base,
            messages=messages,
            **kwargs
        )
        evaluation = self._process_summary_response(result.choices[0].message.content)
        return evaluation

    def batch_summary(
            self,
            summaries: Collection[str],
            references: Collection[List[str]],
            questions: Optional[Collection[str]] = None,
            **kwargs) -> List[Evaluation]:
        assert len(summaries) == len(references)
        if questions is not None:
            data = (summaries, ["\n\n".join(rs) for rs in references], questions)
        else:
            data = (summaries, references)
        assert questions is None or len(summaries) == len(questions)
        list_of_messages = [self._create_summary_call(*elements) for elements in zip(*data)]
        results = utils.parallel_batch_model_call(list_of_messages, self.model, **kwargs)
        evaluations = [self._process_summary_response(r) for r in results]
        return evaluations

    def evaluate_qa(
            self,
            answer: str,
            question: str,
            context: Optional[Union[str, List[str]]] = None,
            **kwargs) -> Evaluation:
        messages = self._create_qa_call(answer, question, context)
        result = litellm.completion(
            model=self.model.model,
            api_key=self.model.api_key,
            api_base=self.model.api_base,
            messages=messages,
            **kwargs
        )
        evaluation = self._process_qa_response(result.choices[0].message.content)
        return evaluation

    def batch_qa(
            self,
            answers: Collection[str],
            questions: Collection[str],
            contexts: Optional[Collection[List[str]]] = None,
            **kwargs):
        assert len(answers) == len(questions)
        assert contexts is None or len(answers) == len(contexts)
        if contexts is not None:
            data = (answers, questions, ["\n\n".join(cs) for cs in contexts])
        else:
            data = (answers, questions)
        list_of_messages = [self._create_qa_call(*elements) for elements in zip(*data)]
        results = utils.parallel_batch_model_call(list_of_messages, self.model, **kwargs)
        evaluations = [self._process_qa_response(r) for r in results]
        return evaluations


class ClaimEvaluator(BaseEvaluator):
    """This class is for the claim-based evaluators"""

    @abstractmethod
    def extract(self, text: str, question: Optional[str], **kwargs) -> List[List[str]]:
        """
        This method extracts claims from the text.

        :param text: The generated text of the answer
        :param question: This is an optional question that can be used as a focus of the claim extraction. For example,
            if this is a patient record then you can focus the extraction with the question "What drugs have been
            prescribed?"
        :param kwargs: these arguments can be used to provide arguments for functions and methods used to evaluate. For
            example, if an evaluator uses a call to an LLM, the temperature can be supplied by the kwargs
        :returns: This method returns an evaluation object
        """
        pass

    @property
    @abstractmethod
    def model(self) -> HostedModel:
        pass


class RedirectEvaluator(BaseEvaluator):
    def __init__(self, model: HostedModel, summary_evaluator: BaseEvaluator, qa_evaluator: BaseEvaluator):
        self.model = model
        self.summary_evaluator = summary_evaluator
        self.qa_evaluator = qa_evaluator

    def evaluate_summary(
            self,
            summary: str,
            reference: Union[str, List[str]],
            question: Optional[str] = None,
            **kwargs) -> Evaluation:
        return self.summary_evaluator.evaluate_summary(summary, reference, question, **kwargs)

    def batch_summary(
            self,
            summaries: Collection[str],
            references: Collection[List[str]],
            questions: Optional[Collection[Optional[str]]] = None,
            **kwargs) -> List[Evaluation]:
        return self.summary_evaluator.batch_summary(summaries, references, questions, **kwargs)

    def evaluate_qa(
            self,
            answer: str,
            question: str,
            context: Optional[Union[str, List[str]]] = None,
            **kwargs) -> Evaluation:
        return self.qa_evaluator.evaluate_qa(answer, question, context, **kwargs)

    def batch_qa(
            self,
            answers: Collection[str],
            questions: Collection[str],
            contexts: Optional[Collection[Optional[List[str]]]] = None,
            **kwargs):
        return self.qa_evaluator.batch_qa(answers, questions, contexts, **kwargs)


__all__ = ["SingleClaimEvaluation", "Evaluation", "BaseEvaluator", "SimpleEvaluator", "ClaimEvaluator",
           "RedirectEvaluator"]
