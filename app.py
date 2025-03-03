import concurrent
import multiprocessing
import os

from strategies.pythiav2 import get_pythiav2_strategy
from _evaluators import LLM, DefaultFields, Evaluation
import json
import time
import boto3

client = boto3.client(
    'sagemaker-runtime',
    region_name='us-east-2'
)
llm = LLM(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))
strategy = get_pythiav2_strategy(llm, verbose=True)


def sagemaker_validator_2(text):
    custom_attributes = "c000b4f9-df62-4c85-a0bf-7c525f9104a4"  # An example of a trace ID.
    endpoint_name = "detect-pii-serverless"  # Your endpoint name.
    content_type = 'application/json'  # The MIME type of the input data in the request body.
    accept = 'text/plain'  # The desired MIME type of the inference in the response.
    payload = {
        "text": text
    }  # Payload for inference.
    print("Start the sagemaker Call...")
    st = time.time()
    response = client.invoke_endpoint(
        EndpointName=endpoint_name,

        ContentType=content_type,
        Accept=accept,
        Body=json.dumps(payload)
    )

    jsonContent = json.loads(response['Body'].read())

    end = time.time()
    print(jsonContent)

    print("duration {}".format(end - st))
    return jsonContent


def _execute_sagemaker_validator(text, endpoint_name):
    try:
        content_type = 'application/json'  # The MIME type of the input data in the request body.
        accept = 'text/plain'  # The desired MIME type of the inference in the response.
        payload = {
            "text": text
        }
        print("Start the sagemaker Call...")
        st = time.time()
        response = client.invoke_endpoint(
            EndpointName=endpoint_name.replace("_", "-"),

            ContentType=content_type,
            Accept=accept,
            Body=json.dumps(payload)
        )
        json_content = json.loads(response['Body'].read())
        end = time.time()
        print("duration {}".format(end - st))
        return json_content
    except Exception as e:
        return {
            "isValid": False,
            "errorMessage": str(e),
            "riskScore": 1
        }


def _execute_validator(text, validator):
    validator_response = _execute_sagemaker_validator(text, validator["name"])
    try:
        validator_response["validatedField"] = validator["input"] if "input" in validator else validator["output"]
    except Exception as e:
        validator_response["validatedField"] = 'input_response'
    validator_response["validator"] = validator
    return validator_response


def process_validators(text, validatos):
    num_cpus = multiprocessing.cpu_count()
    max_workers = min(num_cpus, len(validatos))
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_execute_validator, text, validator)
                   for validator
                   in validatos]

    results = [future.result() for future in futures]
    return results


def handler(event, context):
    # Validate 'response'
    if not event["response"] or not isinstance(event["response"], str) or event["response"].strip() == "":
        raise ValueError("The 'response' parameter cannot be empty or null.")

    # Validate 'reference'
    if not event["reference"] or not isinstance(event["reference"], list):
        raise ValueError("The 'reference' parameter must be a non-empty list.")

    # Check that no element in 'reference' is empty or whitespace
    if any(not item or not isinstance(item, str) or item.strip() == "" for item in event["reference"]):
        raise ValueError("The 'reference' list cannot contain empty or whitespace-only strings.")

    strategy_input = {
        DefaultFields.QUESTION: event["question"] if "question" in event else None,
        DefaultFields.CONTEXT: event["reference"] if "reference" in event else None,
        DefaultFields.ANSWER: event["response"] if "response" in event else None,
    }
    result = strategy(strategy_input)
    evaluation: Evaluation = result["evaluation"]

    claims_list = []
    for claim in evaluation.claims.claims:
        claim_obj = claim.model_dump(by_alias=True)
        claim_list = {
            "claim": [
                claim_obj["claim"]["subject"],
                claim_obj["claim"]["predicate"],
                claim_obj["claim"]["object"]
            ],
            "category": claim_obj["category"],
            "reasoning": claim_obj["reasoning"],
            "class":  claim_obj["category"]

        }
        claims_list.append(claim_list)

    response = {
        "claims": claims_list,
        "metrics": evaluation.metrics,
        "verdict": evaluation.verdict,
        "validatorsResults": []
    }
    if "validators" in event:
        validators_response = process_validators(text=event["response"], validatos=event["validators"])
        response["validatorsResults"] = validators_response
    return response


