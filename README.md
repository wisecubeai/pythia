
# Pythia

![Pythia AI Observability Architecture](pythia-architecture-diagram.png)


### Before you start 
#### LLM Model
##### Default Model Enpoint
The Pythia Stack provide a default model as docker and to use it make sure to have this set in the `.env` file.

```
#Pythia
OPENAI_API_KEY=default
MODEL_BASE_URL=http://pythia-model:8080/v1
```

##### Custom Model Enpoint
To use a model that is compatible with OpenAi Client set the `OPENAI_API_KEY` and `MODEL_BASE_URL` as needed in the `.env` file.

For example using OpenAi
```bash
OPENAI_API_KEY=sk-********
MODEL_BASE_URL=https://api.openai.com/v1
```

#### Model Name
If the model endpoint supports multiple models you can change the model name using the `MODEL_NAME` parameter in the `.env` file.

```
MODEL_NAME=gpt-4o
```

### Custum service Name
If you want to change where and from were pythia writes and reads traces you can set the `JAEGER_SERVICE_NAME` in the .env file and also set the `service_name` for the Pythia traces to your value. The default value is `default`.

```
JAEGER_SERVICE_NAME=default
```

```python
## Pythia Trace Collector with custom service name
traces = PythiaTraces(endpoint="http://localhost:4318", service_name="test2")
traces.init()
```



### Run the docker compose 
```
docker compose up --force-recreate --remove-orphans --detach
```


### Add the Pythia Trace collector to your app 

#### Install pythia sdk

# pythia-sdk

## As a library
You can install the project as a library from github
```commandline
pip install git+https://github.com/wisecubeai/pythia-sdk
```
or if you don't have the git configure use the token
```
pip install git+https://<GIT_TOKEN>@github.com/wisecubeai/pythia-sdk
```

```python
import os
from fastapi import FastAPI
from openai import OpenAI

app = FastAPI()
from pythia.traces.pythia_wrapper import PythiaTraces

# take the correct port for the otl collector
traces = PythiaTraces(endpoint="http://localhost:63714")
traces.init()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


@app.get("/")
def metric_method():
    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "system",
             "content": "There is no significant chamber enlargement or hypertrophy.  There is no pericardial effusion or vegetations seen.  Doppler interrogation, including color flow imaging, reveals systemic venous return to the right atrium with normal tricuspid inflow. Pulmonary outflow is normal at the valve.  Pulmonary venous return is to the left atrium.  The interatrial septum is intact.  Mitral inflow and ascending aorta flow are normal.  The aortic valve is trileaflet.  The coronary arteries appear to be normal in their origins.  The aortic arch is left-sided and patent with normal descending aorta pulsatility."
             },
            {"role": "user", "content": "What was theTNM stage of the cancer?"}
        ],
        model="gpt-4o",
    )
    return chat_completion

```


