## Create and push new validator to ECR
### 1. Build the image
```shell
 docker build -t validators-base-image .
```

### 2. Tag the image with the name of the validator

```shell
docker tag validators-base-image:latest 224544181397.dkr.ecr.us-east-2.amazonaws.com/validators-base-image:<TAG>
```
### 3.Push 
```shell
docker push 224544181397.dkr.ecr.us-east-2.amazonaws.com/validators-base-image:<Tag>
```

Validators updates:
 - [x] bias
 - [x] gibberish
 - [x] pii
 - [x] prompt injection
 - [x] secrets
 - [x] toxicity
 - [] factual_consistency
 - [] ban_substrings
 - [] relevance
