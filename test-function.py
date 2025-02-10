import boto3
import json

# Initialize the Lambda client
client = boto3.client('lambda', region_name='us-east-2')

# Invoke the Lambda function
response = client.invoke(
    FunctionName='pythia-docker',
    InvocationType='RequestResponse',  # Can also use 'Event' for async invocation
    Payload=json.dumps({
        "references": [
            "2-D ECHOCARDIOGRAM,Multiple views of the heart and great vessels reveal normal intracardiac and great vessel relationships. Cardiac function is normal.  There is no significant chamber enlargement or hypertrophy.  There is no pericardial effusion or vegetations seen.  Doppler interrogation, including color flow imaging, reveals systemic venous return to the right atrium with normal tricuspid inflow. Pulmonary outflow is normal at the valve.  Pulmonary venous return is to the left atrium.  The interatrial septum is intact.  Mitral inflow and ascending aorta flow are normal.  The aortic valve is trileaflet.  The coronary arteries appear to be normal in their origins.  The aortic arch is left-sided and patent with normal descending aorta pulsatility."],
        "response": "65 year old female presented with pT3 pN1a moderately differentiated adenocarcinoma of rectum on 9/15/2021.",
        "question": "What was theTNM stage of the cancer?",
        "validators": ["detect-bias", "detect-gibberish", "detect-pii-serverless", "detect-toxicity",
                       "detect-prompt-injection", "detect-secrets-serverless"]
    })
)

print(response)
# Parse the response
response_payload = json.load(response['Payload'])
print(response_payload)
