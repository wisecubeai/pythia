
### Build Pythia for AWS Lambda
```
docker build -t wisecube-pythia .
```
```
docker tag wisecube-pythia:latest 224544181397.dkr.ecr.us-east-2.amazonaws.com/wisecube-pythia:lambda
```
```
docker push 224544181397.dkr.ecr.us-east-2.amazonaws.com/wisecube-pythia:lambda
```



### Test Data

```json
{
  "references": ["2-D ECHOCARDIOGRAM,Multiple views of the heart and great vessels reveal normal intracardiac and great vessel relationships. Cardiac function is normal.  There is no significant chamber enlargement or hypertrophy.  There is no pericardial effusion or vegetations seen.  Doppler interrogation, including color flow imaging, reveals systemic venous return to the right atrium with normal tricuspid inflow. Pulmonary outflow is normal at the valve.  Pulmonary venous return is to the left atrium.  The interatrial septum is intact.  Mitral inflow and ascending aorta flow are normal.  The aortic valve is trileaflet.  The coronary arteries appear to be normal in their origins.  The aortic arch is left-sided and patent with normal descending aorta pulsatility."],
  "response": "65 year old female presented with pT3 pN1a moderately differentiated adenocarcinoma of rectum on 9/15/2021.",
  "question": "What was theTNM stage of the cancer?",
  "validators": ["detect-bias", "detect-gibberish", "detect-pii-serverless"]
}
```