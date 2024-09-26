import json
import os

from fastapi import HTTPException
from read_metrics import get_traces, extract_prompt_and_completion

from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor

import openlit as openlit
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from pythia.ai_hallucination import ask_pythia, search_qids, \
    entity_search, predicate_search
from pythia.validator import ValidatorPool

import logging
from starlette_prometheus import PrometheusMiddleware, metrics
from prometheus_client import Histogram
from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
import threading
from opentelemetry.trace.status import Status, StatusCode

# Prometheus Histogram for each metric in the dictionary
accuracy_metric = Histogram('model_accuracy', 'Accuracy of the model')
entailment_metric = Histogram('model_entailment', 'Entailment score of the model')
contradiction_metric = Histogram('model_contradiction', 'Contradiction score of the model')
neutral_metric = Histogram('model_neutral', 'Neutral score of the model')

app = FastAPI()
app.add_middleware(PrometheusMiddleware)
app.add_route("/metrics", metrics)

logging.basicConfig(level=logging.INFO)


@app.post('/ask-pythia')
async def ask_pythia_api(request: Request):
    data = await request.json()
    question = None
    input_reference = None
    input_response = None
    validators = None
    try:
        input_response = data["response"]
    except Exception as e:
        raise HTTPException(status_code=400, detail="Invalid JSON passed, must include a 'response' key {}".format(e))
    try:
        input_reference = data["reference"]
    except Exception as e:
        raise HTTPException(status_code=400, detail="Invalid JSON passed, must include a 'reference' key {}".format(e))
    try:
        question = data["question"]
    except Exception as e:
        print("No question set")

    try:
        validators = data["validators"]
    except KeyError:
        print("No validators set")

    clam_checker_data = ask_pythia(input_reference=input_reference,
                                   input_response=input_response,
                                   question=question,
                                   validators_list=validators)
    return clam_checker_data


@app.post('/search-qids')
async def search_qids_api(request: Request):
    data = await request.json()
    try:
        question = data["question"]
    except KeyError:
        raise HTTPException(status_code=400, detail="Invalid JSON passed, must include a 'question' key")

    try:
        qids = search_qids(question)
        return qids
    except Exception as e:
        raise HTTPException(status_code=500, detail="Invalid JSON passed")


@app.post('/search_entity')
async def search_entity_api(request: Request):
    data = await request.json()
    try:
        name = data["name"]
    except KeyError:
        raise HTTPException(status_code=400, detail="Invalid JSON passed, must include a 'name' key")

    ignore_case = data.get("ignore_case", False)
    matching_strategy = data.get("matching_strategy", "FUZZY")
    limit = data.get("limit", 10)
    try:

        search_results = entity_search(
            name=name,
            ignore_case=ignore_case,
            matching_strategy=matching_strategy,
            limit=limit
        )
        return search_results, 200

    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Error processing request: {e}')


@app.post('/search_predicate')
async def search_predicate_api(request: Request):
    data = await request.json()
    try:
        name = data["name"]
    except KeyError:
        raise HTTPException(status_code=400, detail='Invalid JSON passed, must include a "name" key')

    ignore_case = data.get("ignore_case", True)
    matching_strategy = data.get("matching_strategy", "CONTAINS")
    limit = data.get("limit", 10)
    try:

        search_results = predicate_search(
            name=name,
            ignore_case=ignore_case,
            matching_strategy=matching_strategy,
            limit=limit
        )
        return search_results, 200

    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Error processing request: {e}')


@app.get("/", response_class=HTMLResponse)
async def orpheus_pythia():
    return '''
    <h1> Orpheus Pythia Application </h1>
    <h4> V1.0.0 </h4>
    '''


def get_model_metrics():
    service_name = os.getenv("JAEGER_SERVICE_NAME")
    traces = get_traces(service_name)
    for trace_obj in traces:
        print("Process Trace with id: {}".format(trace_obj["traceID"]))
        system_message, user_prompt, completion = extract_prompt_and_completion(trace_obj)
        if system_message is None or user_prompt is None or completion is None:
            return None

        validators = ValidatorPool().enabled_validators
        claim = ask_pythia(input_reference=system_message,
                           input_response=completion,
                           question=user_prompt,
                           validators_list=validators)

        print(json.dumps(claim))
        trace_pythia_response(claim)


# Update Prometheus metrics with the values from the dictionary
READ_INTERVAL = os.getenv("READ_INTERVAL", "10")


def update_metrics_job():
    print("Read Metrics ....")
    data = get_model_metrics()
    if data is not None:
        # Set the Prometheus metrics with values from the dictionary
        accuracy_metric.observe(data['accuracy'])
        entailment_metric.observe(data['entailment'])
        contradiction_metric.observe(data['contradiction'])
        neutral_metric.observe(data['neutral'])
        print("Metrics updated")

    threading.Timer(int(READ_INTERVAL), update_metrics_job).start()


JAEGER_HOST = os.getenv("JAEGER_HOST", "jaeger")
JAEGER_PORT = os.getenv("JAEGER_PORT", "6831")


def trace_pythia_response(pythia_response):
    # 1. Create Jaeger exporter
    jaeger_exporter = JaegerExporter(
        agent_host_name=JAEGER_HOST,  # Jaeger agent host (default localhost)
        agent_port=int(JAEGER_PORT)  # Jaeger agent port (default 6831)
    )
    tracer_provider = TracerProvider(
        resource=Resource.create({"service.name": "pythia-service"})
    )
    # 2. Set the TracerProvider with resource information
    trace.set_tracer_provider(tracer_provider)
    span_processor = BatchSpanProcessor(jaeger_exporter)
    tracer_provider.add_span_processor(span_processor)

    # 4. Get a tracer instance
    tracer = trace.get_tracer(__name__)

    # 5. Create a new trace/span and add dictionary values as attributes
    with tracer.start_as_current_span("ask-pythia") as span:
        print("Create new Trace ...")
        for key, value in pythia_response.get("metrics").items():
            span.set_attribute(key, value)

        for validator in pythia_response.get("validatorsResults"):
            try:
                span.set_attribute("{}.isValid".format(validator["validator"]["name"]),
                                   validator.get("isValid"))
                span.set_attribute("{}.riskScore".format(validator["validator"]["name"]),
                                   validator.get("riskScore"))
            except Exception as e:
                pass

        span.set_status(Status(status_code=StatusCode.OK))

    print("Tracing complete.")


update_metrics_job()
if __name__ == '__main__':
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
