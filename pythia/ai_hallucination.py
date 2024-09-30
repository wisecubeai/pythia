from __future__ import annotations

import concurrent
import json
import logging
import multiprocessing
import os
import base64
import tqdm
from typing import List, Iterable, Tuple
import numpy as np
from openai import OpenAI
import time
from collections import Counter
from litellm import batch_completion
from openai import RateLimitError as OpenAIRateLimitError
from openai import APIError as OpenAIAPIError
from openai import Timeout as OpenAITimeout
from tqdm import tqdm
from pythia.prompt_gpt import LLM_CHECKING_PROMPT, LLM_CHECKING_PROMPT_Q
from pythia.llm_extractor import llm_extractor
from pythia.template import labels, label_contradiction, label_entailment, label_neutral
from pythia.validator_call import ValidatorCall
from wisecube_sdk.client import WisecubeClient
import pandas as pd


api_key = os.getenv("OPENAI_API_KEY", "ollama")
base_url = os.getenv("MODEL_BASE_URL")


client = OpenAI(api_key=api_key, base_url=base_url)
wisecube_client = WisecubeClient(os.getenv("API_KEY")).client
model_name = os.getenv("MODEL_NAME", "gpt-4o")
print("OpenAI host: {} and model name: {}".format(base_url, model_name))

def get_model_batch_response(prompts, max_new_tokens=500, temperature=0, model=model_name):
    if not prompts or len(prompts) == 0:
        raise ValueError("Invalid input.")

    message_list = []
    for prompt in prompts:
        if len(prompt) == 0:
            raise ValueError("Invalid prompt.")
        if isinstance(prompt, str):
            messages = [{
                'role': 'user',
                'content': prompt
            }]
        elif isinstance(prompt, list):
            messages = prompt
        else:
            raise ValueError("Invalid prompt type.")
        message_list.append(messages)
    import litellm
    litellm.suppress_debug_info = True
    while True:
        try:
            responses = batch_completion(
                model=model,
                messages=message_list,
                temperature=temperature,
                n=1,
                max_tokens=max_new_tokens
            )
            response_list = [r.choices[0].message.content for r in responses]
            for r in response_list:
                if not r or len(r) == 0:
                    raise ValueError(f'{model} API returns None or empty string')
            return response_list
        except Exception as e:
            if isinstance(e, OpenAIRateLimitError) or isinstance(e, OpenAIAPIError) or isinstance(e, OpenAITimeout):
                print(f"{e} [sleep 10 seconds]")
                time.sleep(10)
                continue
            print(e)
            return None


def llm_check(claims, reference, question=None, batch_size=16, temperature=0.0, model=model_name):
    ret_labels = []
    prompt_list = []

    for claim in claims:
        claim = f"({claim[0]}, {claim[1]}, {claim[2]})"
        if question is None:
            prompt = LLM_CHECKING_PROMPT.format(
                reference=reference,
                claim=claim
            )
        else:
            prompt = LLM_CHECKING_PROMPT_Q.format(
                question=question,
                reference=reference,
                claim=claim
            )
        prompt_list.append(prompt)

    for i in tqdm(range(0, len(prompt_list), batch_size)):
        batch_prompts = prompt_list[i:i + batch_size]
        llm_responses = get_model_batch_response(
            prompts=batch_prompts,
            temperature=temperature,
            model=model,
            max_new_tokens=10,
        )
        for llm_response in llm_responses:
            if llm_response and len(llm_response):
                label = None
                if label_contradiction in llm_response.lower():
                    label = label_contradiction
                elif label_entailment in llm_response.lower():
                    label = label_entailment
                else:
                    label = label_neutral
                ret_labels.append(label)
            else:
                raise 'API returns None or empty string'
    return ret_labels


def parallel_llm_check(claims, references: List[str], question=None, batch_size=16, temperature=0.0, model=model_name):
    num_cpus = multiprocessing.cpu_count()
    max_workers = min(num_cpus, len(references))
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(llm_check, claims, reference, question, batch_size, temperature, model)
                   for reference
                   in references]

    results = [future.result() for future in futures]
    merged_list = [item for sublist in results for item in sublist]
    return merged_list


def calc_accuracy(e, c, r=None, we=1, wc=1, wr=0):
    eps = 1e-10
    if r is None or wr == 0:
        r = 1.0
        wr = 0
    accuracy = (we + wc + wr) / (we * (1 / (e + eps)) + wc * (1 / (1 - c + eps)) + wr * (1 / (r + eps)))
    return round(accuracy, ndigits=9)


# call each validator using the name from the input
def call_method(validator, obj, method_name, *args, **kwargs):
    method = getattr(obj, method_name, None)
    if callable(method):
        return method(validator=validator, **kwargs)
    else:
        raise AttributeError(f"Method {method_name} not found in {obj}")


from concurrent.futures import ThreadPoolExecutor, as_completed


def call_validators(input_reference, input_response, question=None, validators_list=None):
    if validators_list is None:
        return None
    if len(validators_list) == 0:
        return None
    validators_data = []
    validator_class = ValidatorCall()

    def call_validator(validator):
        validator_name = validator["name"]
        print("Execute Validator {}".format(validator_name))
        try:
            validator_data = call_method(
                validator, validator_class, validator_name,
                input_reference=input_reference,
                input_response=input_response,
                question=question
            )
            return validator_data
        except Exception as e:
            print("Fail to execute validation for validator {}. Exception {}".format(validator_name, e))
            return [{
                "validatedField": "",
                "validator": validator,
                "isValid": False,
                "errorMessage": str(e),
                "riskScore": 1
            }]

    num_cpus = multiprocessing.cpu_count()
    max_workers = min(num_cpus, len(validators_list))
    print("Execute validators using {} workers".format(max_workers))
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(call_validator, validator) for validator in validators_list]
        for future in concurrent.futures.as_completed(futures):
            validators_data.extend(future.result())
    print("Validators executed ...")

    return validators_data


def ask_pythia_method(input_reference: List[str], input_response, question=None):
    if not isinstance(input_reference, list):
        raise TypeError("Argument [input_reference] must be a list of strings")
    try:
        claims = llm_extractor(input_response, question=question)
        classes = parallel_llm_check(claims, input_reference, question=question)
        metrics = Counter(classes)
        metrics.update({label: 0 for label in labels})
        metrics = {c: n / max(len(classes), 1) for c, n in metrics.items()}
        metrics["accuracy"] = calc_accuracy(metrics[label_entailment], metrics[label_contradiction])
        triples = []
        for claim, clazz in zip(claims, classes):
            triples.append({
                "claim": claim,
                "class": clazz
            })

        return {
            "claims": triples,
            "metrics": metrics
        }
    except Exception as e:
        print("Fail to calculate Metrics {}".format(e))
    return None


def _ensure_list_of_strings_references(variable):
    if isinstance(variable, list):
        if all(isinstance(item, str) for item in variable):
            return variable
        else:
            return [str(item) for item in variable]
    else:
        if isinstance(variable, str):
            return [variable]
        else:
            return [str(variable)]


def ask_pythia(input_reference, input_response, question=None, validators_list=None):
    input_reference = _ensure_list_of_strings_references(input_reference)
    if not isinstance(input_reference, list):
        raise TypeError("Argument [input_reference] must be a list of strings")
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        future1 = executor.submit(
            ask_pythia_method, input_reference, input_response, question)
        future2 = executor.submit(call_validators, input_reference, input_response, question, validators_list)

        pythia_results = future1.result()
        validator_results = future2.result()
    if validator_results is None:
        return pythia_results
    pythia_results["validatorsResults"] = validator_results
    return pythia_results

def chat_gpt(question):
    prompt = f"""
    Given the following scientific question, extract only the biological process or disease terms depending on question!

    Question:
    "{question}"
    The response will be returned as text separated by |:
    """
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()


def search_qids(question):
    list_of_words = chat_gpt(question).split("|")
    qids = []
    for word in list_of_words:
        search_text = wisecube_client.search_text(word)
        try:
            search_results = search_text["data"]["searchAsYouType"]["data"]["searchLabels"]
            exact_qid = None
            for result in search_results:
                if result["text"].lower() == word.strip().lower():
                    exact_qid = result["qid"]
                    break

            if exact_qid:
                qids.append(exact_qid)
            else:
                qids.append(search_results[0]["qid"])
        except Exception as e:
            print(e)
    return qids


EXACT = "EXACT"
PREFIX = "PREFIX"
SUFFIX = "SUFFIX"
CONTAINS = "CONTAINS"
FUZZY = "FUZZY"
MATCHING_STRATEGIES = set([EXACT, PREFIX, SUFFIX, CONTAINS, FUZZY])

def entity_search(name: str, ignore_case: bool|None=None, matching_strategy: str|None=None, limit: int|None=None):
      """
      This function searches Wikidata for a particular entity using the MediaWiki API
      :param name: the name for searching
      :param ignore_case: if this flag is true then it will be a case insensitive search (default: False)
      "param matching_strategy: this is how the name is being used to match to candidate entities (default: "FUZZY")
      EXACT - exact match of label and name
      PREFIX - do labels of candidate entities begin with name
      SUFFIX - do labels of candidate entities end with name
      CONTAINS - do labels of candidate entities contain name
      FUZZY - do not filter labels after MediaWiki retrieval
      """
      case_clause = f"""BIND(?label AS ?matchLabel)"""
      if ignore_case:
        name = name.lower()
        case_clause = f"""BIND(LCASE(?label) AS ?matchLabel)"""
      match_clause: str
      if matching_strategy == EXACT:
        match_clause = f"""FILTER(?matchLabel = "{name}"@en)"""
      elif matching_strategy == PREFIX:
        match_clause = f"""FILTER(STRSTARTS(?matchLabel, "{name}"))"""
      elif matching_strategy == SUFFIX:
        match_clause = f"""FILTER(STRENDS(?matchLabel, "{name}"))"""
      elif matching_strategy == CONTAINS:
        match_clause = f"""FILTER(CONTAINS(?matchLabel, "{name}"))"""
      elif matching_strategy == FUZZY:
        match_clause = ""
      else:
        raise ValueError(f"unrecognized matching strategy [{matching_strategy}]")
      query = f"""
      PREFIX bd: <http://www.bigdata.com/rdf#>
      PREFIX mwapi: <https://www.mediawiki.org/ontology#API/>
      PREFIX schema: <http://schema.org/>
      PREFIX wikibase: <http://wikiba.se/ontology#>
      SELECT DISTINCT ?item ?label ?description
      WHERE
      {{
        SERVICE wikibase:mwapi {{
          bd:serviceParam wikibase:endpoint "www.wikidata.org" ;
                          wikibase:api "EntitySearch" ;
                          mwapi:search "{name}" ;
                          mwapi:language "en" ;
                          mwapi:limit "{limit}" .
            ?item wikibase:apiOutputItem mwapi:item.
            ?num wikibase:apiOrdinal true.
        }}
        FILTER(LANG(?label)="en")
        # excluding documents
        MINUS {{
          ?item wdt:P31/wdt:P279* wd:Q49848 .
        }}
        # excluding research projects (e.g. clinical trials)
        MINUS {{
          ?item wdt:P31/wdt:P279* wd:Q1298668 .
        }}
        ?item rdfs:label ?label ;
              schema:description ?description .
        FILTER(LANG(?description)="en")
        {case_clause}
        {match_clause}
      }}
      ORDER BY ?num
      """
      try:
        return wisecube_client.advance_search(query)
      except KeyError as ke:
        return


def predicate_search(name: str, ignore_case: bool|None=None, matching_strategy: str|None=None, limit: int|None=None):
      """
      This function searches Wikidata for a particular predicate
      :param name: the name for searching
      :param ignore_case: if this flag is true then it will be a case insensitive search (default: True)
      "param matching_strategy: this is how the name is being used to match to candidate entities (default: "CONTAINS")
      EXACT - exact match of label and name
      PREFIX - do labels of candidate entities begin with name
      SUFFIX - do labels of candidate entities end with name
      CONTAINS - do labels of candidate entities contain name
      """
      case_clause = f"""BIND(?label AS ?matchLabel)"""
      if ignore_case:
        name = name.lower()
        case_clause = f"""BIND(LCASE(?label) AS ?matchLabel)"""
      match_clause: str
      if matching_strategy == EXACT:
        match_clause = f"""FILTER(?matchLabel = "{name}"@en)"""
      elif matching_strategy == PREFIX:
        match_clause = f"""FILTER(STRSTARTS(?matchLabel, "{name}"))"""
      elif matching_strategy == SUFFIX:
        match_clause = f"""FILTER(STRENDS(?matchLabel, "{name}"))"""
      elif matching_strategy == CONTAINS:
        match_clause = f"""FILTER(CONTAINS(?matchLabel, "{name}"))"""
      elif matching_strategy == FUZZY:
        match_clause = ""
      else:
        raise ValueError(f"unrecognized matching strategy [{matching_strategy}]")
      # this is very inefficient, every search tries to match against every predicate name
      # we should replace with a MediaWiki search once we can get it to search for predicates
      terms = "".join([c if c.isalnum() or c.isspace() or c in "'-" else " " for c in name]).split()
      query = f"""
      PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
      PREFIX schema: <http://schema.org/>
      PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
      PREFIX wikibase: <http://wikiba.se/ontology#>
      SELECT DISTINCT ?pred ?label ?description
      WHERE
      {{
        VALUES ?term {{ "{'" "'.join(terms)}" }}
        ?wdPred wikibase:directClaim ?pred ;
                rdfs:label ?label ;
                schema:description ?description .
        FILTER(CONTAINS(LCASE(?description), ?term) || CONTAINS(LCASE(?label), ?term))
        BIND(IF(CONTAINS(?label, ?term), 1, 0) AS ?labelhit)
        BIND(0 AS ?althit)
        OPTIONAL {{
          ?wdPred skos:altLabel ?altLabel .
          BIND(IF(CONTAINS(?altLabel, ?term), 1, 0) AS ?althit)
        }}
        BIND(IF(CONTAINS(?description, ?term), 1, 0) AS ?deschit)
        FILTER(LANG(?label)="en")
        FILTER(LANG(?description)="en")
        {case_clause}
        {match_clause}
      }}
      GROUP BY ?pred ?label ?description
      ORDER BY DESC(SUM(?labelhit)) DESC(SUM(?althit)) DESC(SUM(?deschit))
      LIMIT {limit}
      """
      try:
        return wisecube_client.advance_search(query)
      except KeyError as ke:
        return

