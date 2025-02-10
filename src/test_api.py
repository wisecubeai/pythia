import json
import time

import boto3

access_key = 'AWS_ACCESS'
secret_key = 'AWS_SECRET'

client = boto3.client(
    'sagemaker-runtime',
    region_name='us-east-2',
    aws_access_key_id=access_key,
    aws_secret_access_key=secret_key
)

question = "Diabetes mellitus, password=sk-121D#a2often = known simply as diabetes diseases characterized by sustained high blood sugar levels.[11][12] Diabetes is due to either the pancreas not producing enough insulin, or the cells of the body becoming unresponsive to the hormone's effects.[13] Classic symptoms include thirst, polyuria, weight loss, and blurred vision. If left untreated, the disease can lead to various health complications, including disorders of the cardiovascular system, eye, kidney, and nerves.[3] Untreated or poorly treated diabetes accounts for approximately 1.5 million deaths every year."
custom_attributes = "c000b4f9-df62-4c85-a0bf-7c525f9104a4"  # An example of a trace ID.
endpoint_name = "detect-toxicity"  # Your endpoint name.
content_type = 'application/json'  # The MIME type of the input data in the request body.
accept = 'text/plain'  # The desired MIME type of the inference in the response.
payload = {
    "text": question
}  # Payload for inference.

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
