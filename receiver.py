import os

from fastapi import FastAPI
from openai import OpenAI
app = FastAPI()

from pythia.traces.pythia_wrapper import PythiaTraces
# take the correct port for the otl collector

traces = PythiaTraces(endpoint="http://localhost:4318", service_name="test-pythia")
traces.init()


client = OpenAI(api_key="demo", base_url="http://localhost:8080/v1")

@app.get("/")
def metric_method():
    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "system", "content": "2-D ECHOCARDIOGRAM,Multiple views of the heart and great vessels reveal normal intracardiac and great vessel relationships. Cardiac function is normal.  There is no significant chamber enlargement or hypertrophy.  There is no pericardial effusion or vegetations seen.  Doppler interrogation, including color flow imaging, reveals systemic venous return to the right atrium with normal tricuspid inflow. Pulmonary outflow is normal at the valve.  Pulmonary venous return is to the left atrium.  The interatrial septum is intact.  Mitral inflow and ascending aorta flow are normal.  The aortic valve is trileaflet.  The coronary arteries appear to be normal in their origins.  The aortic arch is left-sided and patent with normal descending aorta pulsatility."
             },
            {"role": "user", "content": "What was theTNM stage of the cancer?"}
        ],
        model="gpt-4o",
    )
    print(chat_completion)
    return chat_completion
