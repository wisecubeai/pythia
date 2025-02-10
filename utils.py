"""
This contains the utility functions for the `evaluators` modules
"""
import concurrent
import json
import multiprocessing
import re
import time
from typing import List, Dict, Optional, Set, Any

from litellm import batch_completion
from openai import APIError as OpenAIAPIError
from openai import APITimeoutError as OpenAITimeoutError
from openai import RateLimitError as OpenAIRateLimitError
from tqdm import tqdm

from models import HostedModel
from template import label_entailment, label_contradiction, label_neutral, label_reliable


def parallel_batch_model_call(
        message_list: List[List[Dict[str, str]]], model: HostedModel, batch_size: int = 16, verbose: bool = True,
        **kwargs) -> List[str]:
    """
    This method does parallel batch calling using LiteLLM
    :param message_list: the list of lists of messages that make up the batches
    :param model: the model to be used
    :param batch_size: the size of batches to be sent in parallel calls
    :param verbose: the flag determines whether to display progress bar
    :param kwargs: additional keyword arguments intended to be passed to LiteLLM
    :return: the list of results
    """
    import litellm
    litellm.suppress_debug_info = True
    def run_batch(batch):
        return batch_completion(
            model=model.model,
            api_base=model.api_base,
            api_key=model.api_key,
            messages=batch,
            temperature=0.0,
            seed=123
            **kwargs
        )

    if not message_list:
        return []

    num_cpus = multiprocessing.cpu_count()
    max_workers = min(num_cpus, len(message_list))
    batches = []
    i = 0
    while i < len(message_list):
        batches.append(message_list[i:i+batch_size])
        i += batch_size

    while True:
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(run_batch, batch) for batch in batches]
            if verbose:
                pbar = tqdm(total=len(batches), desc='batched calls')
                for _ in concurrent.futures.as_completed(futures):
                    pbar.update(n=1)
            results_raw = [future.result() for future in futures]
            results = [item for sublist in results_raw for item in sublist]
            results = [r.choices[0].message.content if hasattr(r, "choices") else None for r in results]
            for i in range(len(results)):
                if not results[i] or len(results[i]) == 0:
                    results[i] = json.dumps({"status": "ERROR", "message": message_list[i]})
            return results
        except OpenAIRateLimitError as e:
            print(f"{e} [sleep 10 seconds]")
            time.sleep(10)
            continue
        except OpenAIAPIError as e:
            print(f"{e} [sleep 10 seconds]")
            time.sleep(10)
            continue
        except OpenAITimeoutError as e:
            print(f"{e} [sleep 10 seconds]")
            time.sleep(10)
            continue


def parallel_model_call(
        message_list: List[List[Dict[str, str]]], model: HostedModel, verbose: bool = True,
        **kwargs) -> List[str]:
    """
    This method does parallel (non-batched) calling using LiteLLM
    :param message_list: the list of lists of messages that make up the batches
    :param model: the model to be used
    :param verbose: the flag determines whether to display progress bar
    :param kwargs: additional keyword arguments intended to be passed to LiteLLM
    :return: the list of results
    """
    import litellm
    litellm.suppress_debug_info = True
    def run_batch(message):
        return litellm.completion(
            model=model.model,
            api_base=model.api_base,
            api_key=model.api_key,
            messages=message,
            **kwargs
        )

    if not message_list:
        return []

    num_cpus = multiprocessing.cpu_count()
    max_workers = min(num_cpus, len(message_list))

    while True:
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(run_batch, message) for message in message_list]
            if verbose:
                pbar = tqdm(total=len(message_list), desc='batched calls')
                for _ in concurrent.futures.as_completed(futures):
                    pbar.update(n=1)
            results_raw = [future.result() for future in futures]
            results = [result for result in results_raw]
            results = [r.choices[0].message.content if hasattr(r, "choices") else None for r in results]
            for i in range(len(results)):
                if not results[i] or len(results[i]) == 0:
                    results[i] = json.dumps({"status": "ERROR", "message": message_list[i]})
            return results
        except OpenAIRateLimitError as e:
            print(f"{e} [sleep 10 seconds]")
            time.sleep(10)
            continue
        except OpenAIAPIError as e:
            print(f"{e} [sleep 10 seconds]")
            time.sleep(10)
            continue
        except OpenAITimeoutError as e:
            print(f"{e} [sleep 10 seconds]")
            time.sleep(10)
            continue


def calc_accuracy(
        counts: Dict[str, int],
        penalized: Set[str] = frozenset([label_contradiction]),
        weights: Optional[Dict[str, float]] = None) -> float:
    """
    This method is used for calculating the accuracy metric(s)
    :param counts: the number of occurrences of each category of claims
    :param penalized: the categories that should be penalized
    :param weights: the weights for categories when doing the weighted harmonic sum
    :return: the accuracy metric
    """
    eps = 1e-10
    if weights is None:
        weights = {label_entailment: 1.0 , label_contradiction: 1.0, label_neutral: 0.0, label_reliable: 0.0}
    numerator = 0.0
    total = 0
    for m, wt in weights.items():
        if m in counts:
            numerator += wt
            total += counts[m]
    rates = {}
    total = max(total, 1)
    for m, wt in weights.items():
        if m in penalized:
            rates[m] = 1 - counts.get(m, 0) / total
        else:
            rates[m] = counts.get(m, 0) / total
    denominator = sum([wt / (rates[m] + eps) for m, wt in weights.items()])
    accuracy = numerator / denominator
    return round(accuracy, ndigits=9)


def parse_json_result(result: str) -> Dict[str, Any]:
    raw_result = result
    result = result.strip().strip("`").strip()
    try:
        result = result[result.index("{"):result.rindex("}") + 1]

        reasoning_start = result.index(":", result.rfind('"reasoning"')) + 1
        verdict_start = result.index(":", result.rfind('"verdict"')) + 1
        reasoning_end = result.rindex(",", 0, verdict_start)
        verdict_end = result.rindex("}")

        reasoning = result[reasoning_start:reasoning_end].strip().strip('"')
        verdict = result[verdict_start:verdict_end].strip().strip('"')
    except ValueError:
        verdict = re.search("([Vv][Ee][Rr][Dd][Ii][Cc][Tt].*:.*)([ABCDF])", result).group(2)
        reasoning = result

    return {"reasoning": reasoning, "verdict": verdict}


__all__ = ["parallel_batch_model_call", "parallel_model_call", "calc_accuracy"]