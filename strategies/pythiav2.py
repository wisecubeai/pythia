from _evaluators import Task, LLM, DefaultFields, EvalChain, PrepareLLMCall, CallLLM, CollectionClaimEvaluation, \
    Verdict, EvalStep, ChainStep, Strategy, Evaluation, ExtractedClaim, categorized_factual_accuracy, Extraction
from prompts.pythiav2 import *

from collections import Counter
from typing import List, Dict, Any

EXT_PFX = "extraction_"
CHK_PFX = "check_"


class PythiaV2Evaluation(EvalStep):
    """
    This class is a simple evaluation step that calculates the accuracy of the claims
    """
    threshold: float = 0.9
    response_field: str = DefaultFields.RESPONSE
    evaluation_field: str = DefaultFields.EVALUATION

    @property
    def input_keys(self) -> List[str]:
        return [self.response_field]

    @property
    def output_keys(self) -> List[str]:
        return [self.evaluation_field]

    def __call__(self, input_arg: Dict[str, Any]) -> Dict[str, Any]:
        claim_evals = input_arg[self.response_field]
        category_counts = Counter([e.category for e in claim_evals.claims])
        total = sum(category_counts.values())
        metrics: Dict[str, float] = {cat: count / max(1, total) for cat, count in category_counts.items()}
        metrics.update(categorized_factual_accuracy(claim_evals.claims))
        verdict = Verdict.PASS if metrics["accuracy"] >= self.threshold else Verdict.FAIL
        evaluation = Evaluation(metrics=metrics, claims=claim_evals, verdict=verdict)
        input_arg.update({self.evaluation_field: evaluation})
        return input_arg


def get_pythiav2_extract_chain(task: Task, llm: LLM, messages_field: str = EXT_PFX + DefaultFields.MESSAGES,
                               extraction_field: str = DefaultFields.EXTRACTION, **kwargs) -> EvalChain:
    if task == Task.QA_ZERO_CONTEXT:
        raise ValueError(f"Unsupported task: {task}")
    elif task == Task.RAG_QA:
        response_fields = {DefaultFields.ANSWER: "text"}
        context_fields = {DefaultFields.CONTEXT: "text"}
        output_pfx = DefaultFields.ANSWER + "_"
        context_pfx = DefaultFields.CONTEXT + "_"
    elif task == Task.SUMMARIZATION_W_Q or task == Task.SUMMARIZATION:
        response_fields = {DefaultFields.SUMMARY: "text"}
        context_fields = {DefaultFields.REFERENCE: "text"}
        output_pfx = DefaultFields.SUMMARY + "_"
        context_pfx = DefaultFields.REFERENCE + "_"
    else:
        raise ValueError(f"Task {task} is not supported for pythiav2.")
    extraction_system = PYTHIAV2_EXTRACTION_SYSTEM
    extraction_template = PYTHIAV2_EXTRACTION_TEMPLATE
    return EvalChain(steps=[
        PrepareLLMCall(
            description="Prepare output extraction call",
            prompt_template=extraction_template,
            template_field_map=response_fields,
            system_prompt=extraction_system,
            messages_field= output_pfx + messages_field,
        ),
        PrepareLLMCall(
            description="Prepare context extraction call",
            prompt_template=extraction_template,
            template_field_map=context_fields,
            system_prompt=extraction_system,
            messages_field=context_pfx + messages_field,
        ),
        CallLLM(
            description="Call LLM to extract claims",
            llm=llm,
            response_format=Extraction,
            messages_field=output_pfx + messages_field,
            response_field=output_pfx + extraction_field,
            kwargs=kwargs,
        ),
        CallLLM(
            description="Call LLM to extract claims",
            llm=llm,
            response_format=Extraction,
            messages_field=context_pfx + messages_field,
            response_field=context_pfx + extraction_field,
            kwargs=kwargs,
        )
    ])


def get_pythiav2_check_chain(task: Task, llm: LLM, messages_field: str = CHK_PFX + DefaultFields.MESSAGES,
                             extraction_field_suffix: str = DefaultFields.EXTRACTION, **kwargs) -> EvalChain:
    if task == Task.QA_ZERO_CONTEXT:
        raise ValueError(f"Unsupported task: {task}")
    elif task == Task.RAG_QA:
        output_extraction_field = DefaultFields.ANSWER + "_" + extraction_field_suffix
        context_extraction_field = DefaultFields.CONTEXT + "_" + extraction_field_suffix
    elif task == Task.SUMMARIZATION_W_Q or task == Task.SUMMARIZATION:
        output_extraction_field = DefaultFields.SUMMARY + "_" + extraction_field_suffix
        context_extraction_field = DefaultFields.REFERENCE + "_" + extraction_field_suffix
    else:
        raise ValueError(f"Task {task} is not supported for pythiav2.")
    checking_fields = {output_extraction_field: "response_triples", context_extraction_field: "reference_triples"}
    checking_system = PYTHIAV2_CHECKING_SYSTEM
    checking_template = PYTHIAV2_CHECKING_TEMPLATE
    return EvalChain(steps=[
        PrepareLLMCall(
            description="Prepare checking call",
            prompt_template=checking_template,
            template_field_map=checking_fields,
            system_prompt=checking_system,
            messages_field=messages_field,
        ),
        CallLLM(
            description="Call LLM to check claims",
            llm=llm,
            response_format=CollectionClaimEvaluation,
            messages_field=messages_field,
            response_field=DefaultFields.RESPONSE,
            kwargs=kwargs,
        ),
        PythiaV2Evaluation(
            description="Evaluate the claims",
            threshold=0.9,
            response_field=DefaultFields.RESPONSE,
            evaluation_field=DefaultFields.EVALUATION
        )
    ])


def get_pythiav2_chain(task: Task, llm: LLM, messages_field: str = DefaultFields.MESSAGES, **kwargs) -> EvalChain:
    extraction_chain = get_pythiav2_extract_chain(task, llm, messages_field=EXT_PFX + messages_field, **kwargs)
    check_chain = get_pythiav2_check_chain(task, llm, messages_field=CHK_PFX + messages_field, **kwargs)
    return EvalChain(steps=[
        ChainStep(
            description="Extract claims",
            chain=extraction_chain
        ),
        ChainStep(
            description="Check claims",
            chain=check_chain
        )
    ])


def get_pythiav2_strategy(llm: LLM, messages_field: str = DefaultFields.MESSAGES, **kwargs) -> Strategy:
    return Strategy(
        name="pythiav2",
        description="PythiaV2 strategy",
        task_chains={
            Task.RAG_QA: get_pythiav2_chain(Task.RAG_QA, llm, messages_field, **kwargs),
            Task.SUMMARIZATION: get_pythiav2_chain(Task.SUMMARIZATION, llm, messages_field, **kwargs),
            Task.SUMMARIZATION_W_Q: get_pythiav2_chain(Task.SUMMARIZATION_W_Q, llm, messages_field, **kwargs),
        }
    )