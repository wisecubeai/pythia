"""
This module contains the common elements for the evaluators.
"""
import abc
import json
import math
from collections import Counter
from collections.abc import Iterable
from enum import Enum
from typing import Optional, Any, List, Dict, Union, Set, Iterator

import litellm
from litellm.types.utils import ModelResponse
from pydantic import BaseModel, field_validator, Field
from tqdm import trange


class DefaultFields(str, Enum):
    """
    This class represents the default field names.
    """
    QUESTION = "question"
    ANSWER = "answer"
    CONTEXT = "context"
    SUMMARY = "summary"
    REFERENCE = "reference"
    MESSAGES = "messages"
    RESPONSE = "response"
    EVALUATION = "evaluation"
    EXTRACTION = "extraction"


class Verdict(str, Enum):
    """
    This class represents the different verdicts that the evaluations can have.
    """
    PASS = "PASS"
    FAIL = "FAIL"
    INCOMPLETE = "INCOMPLETE"
    N_A = "N/A"


class ExtractedClaim(BaseModel):
    """This is the dataclass for an extracted claim"""
    subject: str
    predicate: str
    object: str

    def __str__(self):
        return f"{self.subject} {self.predicate} {self.object}"

    def __repr__(self):
        return f"({self.subject}, {self.predicate}, {self.object})"


class Extraction(BaseModel, Iterable):
    """This is the dataclass for a set of claims"""
    claims: List[ExtractedClaim]

    def __iter__(self) -> Iterator[ExtractedClaim]:
        return iter(self.claims)

    def __str__(self):
        return "\n".join([str(claim) for claim in self.claims])


class Category(str, Enum):
    NA = "NA"
    ENTAILMENT = "entailment"
    CONTRADICTION = "contradiction"
    NEUTRAL = "neutral"
    RELIABLE = "reliable"


class SingleClaimEvaluation(BaseModel):
    """This is the dataclass for a single claim evaluation"""
    claim: ExtractedClaim
    category: Category = Category.NA
    reasoning: Optional[Union[str,List[str]]] = None


class CollectionClaimEvaluation(BaseModel):
    """This is the dataclass for a collection of claim evaluations"""
    claims: List[SingleClaimEvaluation]

    def __iter__(self) -> Iterator[SingleClaimEvaluation]:
        return iter(self.claims)


class Evaluation(BaseModel):
    """This is the dataclass for an evaluation"""
    metrics: Dict[str, float]
    verdict: Verdict
    claims: Optional[CollectionClaimEvaluation] = None
    reasoning: Optional[str] = None


def simple_factual_accuracy(claims: List[ExtractedClaim], reference: List[str]) -> Dict[str, float]:
    """
    Calculate the factual accuracy of a list of claims against a reference.
    """
    correct = 0
    for claim in claims:
        if claim in reference:
            correct += 1
    return {"accuracy": correct / len(claims)}


def categorized_factual_accuracy(
        claim_evaluations: List[SingleClaimEvaluation],
        included_categories: Set[Category] = frozenset([Category.ENTAILMENT, Category.CONTRADICTION]),
        penalized_categories: Set[Category] = frozenset([Category.CONTRADICTION]),
        weights: Optional[Dict[Category, float]] = None,
        eps: float = 1e-10
) -> Dict[str, float]:
    """
    Calculate the factual accuracy of a list of claim evaluations.
    """
    if weights is None:
        weights = {}
        for cat in included_categories:
            weights[cat] = 1.0
    counts = Counter([c.category for c in claim_evaluations])
    for cat in included_categories:
        if cat not in counts:
            counts[cat] = 0
    total = max(sum([counts[c] for c in included_categories]), 1)
    rates: Dict[str, float] = {cat: counts[cat] / total for cat in included_categories}
    for cat in penalized_categories:
        rates[cat] = 1 - rates[cat]
    numerator = sum([weights.get(cat, 1.0) for cat in included_categories])
    denominator = sum([weights.get(cat, 1.0) / (rates[cat] + eps) for cat in included_categories])
    return {"accuracy": round(numerator / denominator, ndigits=round(math.log10(1 / eps))-1)}


class Task(str, Enum):
    """
    This class represents the different tasks that the LLM API can perform.
    """
    SUMMARIZATION = "summarization"
    SUMMARIZATION_W_Q = "summarization_w_q"
    QA_ZERO_CONTEXT = "qa_zero_context"
    RAG_QA = "rag_qa"

    @classmethod
    def determine_task(cls, input_args: Dict[str, Any]) -> "Task":
        if DefaultFields.QUESTION in input_args and DefaultFields.ANSWER in input_args and DefaultFields.CONTEXT in input_args:
            return Task.RAG_QA
        elif DefaultFields.QUESTION in input_args and DefaultFields.ANSWER in input_args:
            return Task.QA_ZERO_CONTEXT
        elif DefaultFields.SUMMARY in input_args and DefaultFields.REFERENCE in input_args and DefaultFields.QUESTION in input_args:
            return Task.SUMMARIZATION_W_Q
        elif DefaultFields.SUMMARY in input_args and DefaultFields.REFERENCE in input_args:
            return Task.SUMMARIZATION
        else:
            raise ValueError("Invalid task")

    @classmethod
    def validate_task(cls, input_args: Dict[str, Any], task: "Task") -> bool:
        if task == Task.RAG_QA:
            return DefaultFields.QUESTION in input_args and DefaultFields.ANSWER in input_args and DefaultFields.CONTEXT in input_args
        elif task == Task.QA_ZERO_CONTEXT:
            return DefaultFields.QUESTION in input_args and DefaultFields.ANSWER in input_args
        elif task == Task.SUMMARIZATION_W_Q:
            return DefaultFields.SUMMARY in input_args and DefaultFields.REFERENCE in input_args and DefaultFields.QUESTION in input_args
        elif task == Task.SUMMARIZATION:
            return DefaultFields.SUMMARY in input_args and DefaultFields.REFERENCE in input_args
        else:
            raise ValueError("Invalid task")


class LLM(BaseModel):
    """
    This class is a wrapper for the LLM API.
    """
    model: str
    api_key: Optional[str] = Field(default=None, exclude=True)
    api_base_url: Optional[str] = Field(default=None, exclude=True)

    def __call__(self, messages: List[Dict[str, str]], **kwargs) -> ModelResponse:
        return litellm.completion(
            model=self.model,
            api_key=self.api_key,
            api_base=self.api_base_url,
            messages=messages,
            **kwargs
        )

    def batch(self, messages: List[List[Dict[str, str]]], batch_size: int, verbose=False, **kwargs) -> List[ModelResponse]:
        results = []
        if verbose:
            r = trange(0, len(messages), batch_size)
        else:
            r = range(0, len(messages), batch_size)
        for i in r:
            results.extend(litellm.batch_completion(
                model=self.model,
                api_key=self.api_key,
                api_base=self.api_base_url,
                messages=messages[i:i + batch_size],
                **kwargs))
        else:
            if verbose:
                print("Batch completion done. # of results:", len(results))
        return results


class EvalStep(BaseModel, abc.ABC):
    """
    This class represents a single evaluation step.
    """
    description: str

    @property
    @abc.abstractmethod
    def input_keys(self) -> List[str]:
        pass

    @property
    @abc.abstractmethod
    def output_keys(self) -> List[str]:
        pass

    @abc.abstractmethod
    def __call__(self, input_arg: Dict[str, Any]) -> Dict[str, Any]:
        pass

    def batch(self, input_args: List[Dict[str, Any]], batch_size: int) -> List[Dict[str, Any]]:
        return [self(input_arg) for input_arg in input_args]

class PrepareLLMCall(EvalStep):
    """
    This class creates calls to the LLM API.
    """
    prompt_template: str
    template_field_map: Dict[str, str]
    system_prompt: Optional[str] = None
    messages_field: str = DefaultFields.MESSAGES

    @property
    def input_keys(self) -> List[str]:
        return list(self.template_field_map.values())

    @property
    def output_keys(self) -> List[str]:
        return [self.messages_field]

    def __call__(self, input_args: Dict[str, Any]) -> Dict[str, Any]:
        messages = []
        if self.system_prompt is not None:
            messages.append({"role": "system", "content": self.system_prompt})
        user_message = self.prompt_template.format(**{tmpl_f: input_args[in_f] for in_f, tmpl_f in self.template_field_map.items()})
        messages.append({"role": "user", "content": user_message})
        input_args.update({
            self.messages_field: messages
        })
        return input_args


class CallLLM(EvalStep):
    """
    This class calls the LLM API.
    """
    llm: LLM
    response_format: Optional[Any] = None
    kwargs: Optional[Dict[str, Any]] = None
    messages_field: str = DefaultFields.MESSAGES
    response_field: str = DefaultFields.RESPONSE

    @property
    def input_keys(self) -> List[str]:
        return [self.messages_field]

    @property
    def output_keys(self) -> List[str]:
        return [self.response_field]

    def __call__(self, input_args: Dict[str, Any]) -> Dict[str, Any]:
        if self.response_format is not None:
            response: ModelResponse = self.llm(messages=input_args[self.messages_field],
                                               response_format=self.response_format, **self.kwargs)
            response: dict = json.loads(response.choices[0].message.content)
            response = self.response_format(**response)
        else:
            response = self.llm(messages=input_args[self.messages_field], **self.kwargs)
            response = response.choices[0].message.content
        input_args.update({
            self.response_field: response
        })
        return input_args

    def batch(self, input_args: List[Dict[str, Any]], batch_size: int) -> List[Dict[str, Any]]:
        kwargs = dict(**self.kwargs)
        verbose = "verbose" in kwargs and kwargs["verbose"]
        if verbose:
            del kwargs["verbose"]
        if self.response_format is not None:
            responses = self.llm.batch(
                messages=[input_arg[self.messages_field] for input_arg in input_args],
                batch_size=batch_size,
                verbose=verbose,
                response_format=self.response_format,
                **kwargs
            )
            responses = [self.response_format(**json.loads(response.choices[0].message.content)) for response in responses]
        else:
            responses = self.llm.batch(
                messages=[input_arg[self.messages_field] for input_arg in input_args],
                batch_size=batch_size,
                verbose=verbose,
                **kwargs
            )
            responses = [response.choices[0].message.content for response in responses]
        for i, input_arg in enumerate(input_args):
            input_arg.update({
                self.response_field: responses[i]
            })
        return input_args


class EvalChain(BaseModel):
    """
    This class represents a chain of evaluation steps.
    """
    steps: List[EvalStep]

    def __call__(self, input_arg: Dict[str, Any]) -> Dict[str, Any]:
        for step in self.steps:
            input_arg = step(input_arg)
        return input_arg

    def batch(self, input_args: List[Dict[str, Any]], batch_size: int) -> List[Dict[str, Any]]:
        for step in self.steps:
            input_args = step.batch(input_args, batch_size)
        return input_args

    def __add__(self, other):
        if isinstance(other, EvalStep):
            return EvalChain(steps=self.steps + [other])
        elif isinstance(other, EvalChain):
            return EvalChain(steps=self.steps + other.steps)
        else:
            raise TypeError("unsupported operand type(s) for +: 'EvalChain' and '{}'".format(type(other)))


class ChainStep(EvalStep):
    """
    This class represents a single step in a chain of evaluation steps.
    """
    chain: EvalChain

    @property
    def input_keys(self) -> List[str]:
        return self.chain.steps[0].input_keys

    @property
    def output_keys(self) -> List[str]:
        return self.chain.steps[-1].output_keys

    def __call__(self, input_arg: Dict[str, Any]) -> Dict[str, Any]:
        input_arg = self.chain(input_arg)
        return input_arg

    def batch(self, input_args: List[Dict[str, Any]], batch_size: int) -> List[Dict[str, Any]]:
        return self.chain.batch(input_args, batch_size)


class IteratedStep(EvalStep):
    """
    This class represents a single step that is iterated over a list of input arguments.
    """
    step: EvalStep
    iterated_field: str
    result_field: str
    batch_size: int = 1

    @property
    def input_keys(self) -> List[str]:
        return self.step.input_keys

    @property
    def output_keys(self) -> List[str]:
        return self.step.output_keys

    def __call__(self, input_arg: Dict[str, Any]) -> Dict[str, Any]:
        results = []
        if self.batch_size == 1:
            for arg in input_arg[self.iterated_field]:
                curr = dict(**input_arg)
                del curr[self.iterated_field]
                curr[self.iterated_field] = arg
                results.append(self.step(curr))
        else:
            for i in range(0, len(input_arg[self.iterated_field]), self.batch_size):
                curr = dict(**input_arg)
                del curr[self.iterated_field]
                curr[self.iterated_field] = input_arg[self.iterated_field][i:i + self.batch_size]
                results.extend(self.step.batch([curr], self.batch_size))
        input_arg.update({self.result_field: [r[self.result_field] for r in  results]})
        return input_arg

    def batch(self, input_args: List[Dict[str, Any]], batch_size: int) -> List[Dict[str, Any]]:
        batches = []
        if self.batch_size == 1:
            for input_arg in input_args:
                for arg in input_arg[self.iterated_field]:
                    curr = dict(**input_arg)
                    del curr[self.iterated_field]
                    curr[self.iterated_field] = arg
                    batches.append(curr)
        else:
            for input_arg in input_args:
                for i in range(0, len(input_arg[self.iterated_field]), self.batch_size):
                    curr = dict(**input_arg)
                    del curr[self.iterated_field]
                    curr[self.iterated_field] = input_arg[self.iterated_field][i:i + self.batch_size]
                    batches.append(curr)
        results = self.step.batch(batches, batch_size)
        for i, input_arg in enumerate(input_args):
            input_arg.update({self.result_field: [r[self.result_field] for r in results[i::len(input_args)]]})
        return input_args


class Strategy(BaseModel):
    """
    This class represents a strategy for evaluating a given set of tasks
    """
    name: str
    description: str
    task_chains: Dict[Task, EvalChain]

    @field_validator("task_chains", mode="before")
    def _validate_task_chains(cls, value: Dict[Task, EvalChain]) -> Dict[Task, EvalChain]:
        if len(value) == 0:
            raise ValueError("Strategy must have at least one task chain")
        return value

    def __call__(self, input_arg: Dict[str, Any], task: Optional[Task]=None) -> Dict[str, Any]:
        if task is None:
            task = Task.determine_task(input_arg)
        return self.task_chains[task](input_arg)

    def batch(self, input_args: List[Dict[str, Any]], batch_size: int, task: Optional[Task]=None) -> List[Dict[str, Any]]:
        if task is None:
            task = Task.determine_task(input_args[0])
        return self.task_chains[task].batch(input_args, batch_size)
