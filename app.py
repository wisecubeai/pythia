import concurrent
import multiprocessing

from strategies.pythiav2 import PythiaV2Evaluator
from models import HostedModel
import json
import time
import boto3

client = boto3.client(
    'sagemaker-runtime',
    region_name='us-east-2'
)
model = HostedModel(model="gpt-4o-mini")
evaluator = PythiaV2Evaluator(model)


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
    if not event["response"] or not isinstance(event["response"], str) or event["response"].strip() == "":
        raise ValueError("The 'response' parameter cannot be empty or null.")
    if not event["reference"] or not isinstance(event["reference"], str) or event["reference"].strip() == "":
        raise ValueError("The 'reference' parameter cannot be empty or null.")

    evaluation_result = evaluator.evaluate_summary(event["response"], event["reference"])

    if "validators" in event:
        validators_response = process_validators(text=event["response"], validatos=event["validators"])
        evaluation_result.validatorsResults = validators_response
    return json.loads(evaluation_result.json(by_alias=True))

