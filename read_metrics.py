import os
import random
import time

import requests

service_host = os.getenv('JAEGER_SERVICE_HOST')
service_port = int(os.getenv('JAEGER_SERVICE_PORT'))

JAEGER_TRACES_ENDPOINT = f"http://{service_host}:{service_port}/jaeger/ui/api/traces"
print(JAEGER_TRACES_ENDPOINT)
processed_trace_ids = set()


def get_traces(service_name,lookback_seconds=10):
    """
    Returns list of all traces for a service
    """

    end_time = int(time.time() * 1000 * 1000)
    start_time = end_time - (lookback_seconds * 1000 * 1000)

    params = {
        "service": service_name,
        "start": start_time,
        "end": end_time
    }

    try:
        response = requests.get(JAEGER_TRACES_ENDPOINT, params=params)
        response.raise_for_status()
        traces = response.json().get("data", [])
        print(f"Traces found: {len(traces)}")

        # processed_trace_ids.clear()
        #filter traces already processed
        new_traces = [trace for trace in traces if trace['traceID'] not in processed_trace_ids]
        # print(new_traces)
        #update list with processed ids
        processed_trace_ids.update([trace['traceID'] for trace in new_traces])

        return new_traces
    except requests.exceptions.RequestException as err:
        print(f"Error: {err}")
        return []


def extract_prompt_and_completion(traces):
    responses = []
    current_prompt = None
    system_message_list = []
    user_prompt = None
    completion = None

    for trace in traces:
        for span in trace.get("spans", []):
            for log in span.get("logs", []):
                fields = {field["key"]: field["value"] for field in log.get("fields", [])}

                if 'gen_ai.prompt' in fields:
                    current_prompt = fields['gen_ai.prompt']

                if 'gen_ai.completion' in fields and current_prompt:
                    completion = fields['gen_ai.completion']
                    responses.append({"prompt": current_prompt, "completion": completion})

                    # Reset the current prompt after pairing
                    current_prompt = None

    for x in responses:
        prompt = x.get("prompt")
        system_message, user_prompt = prompt.split('\nuser: ', 1)

        system_message = system_message.replace('system: ', '').strip()
        user_prompt = user_prompt.strip()
        completion = x.get("completion")

        system_message_list.append(system_message)

    return system_message_list, user_prompt, completion

# def extract_prompt_and_completion_pairs(response):
#     prompt = None
#     completion = None
#     for x in response:
#         prompt = x.get("prompt")
#         completion = x.get("completion")
#     return prompt, completion

# service_name = "default"
# traces = get_traces(service_name)
# # print(traces)
# system_message, user_prompt,completion = extract_prompt_and_completion(traces)
#
# print("Reference: ",system_message)
# print("Question: ",user_prompt)
# print("Completion: ",completion)
# # abstract_list = ["2-D ECHOCARDIOGRAM,Multiple views of the heart and great vessels reveal normal intracardiac and great vessel relationships. Cardiac function is normal.  There is no significant chamber enlargement or hypertrophy.  There is no pericardial effusion or vegetations seen."]
# claim = ask_pythia(system_message,completion,user_prompt)
# print(claim)