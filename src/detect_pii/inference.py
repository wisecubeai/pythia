from llm_guard.vault import Vault
from llm_guard.input_scanners import Anonymize
from llm_guard.input_scanners.anonymize_helpers import BERT_LARGE_NER_CONF
import concurrent
import os


def load_model(model_dir: str):
    """
    Load the model from the specified directory.
    """
    return "Load Model"


def _validate(text):
    try:
        max_length = 500

        vault = Vault()
        scanner = Anonymize(vault, recognizer_conf=BERT_LARGE_NER_CONF, language="en")
        sanitized_prompt, is_valid, risk_score = scanner.scan(text)
        truncated_string = f"{sanitized_prompt[:max_length]}{'...' if len(sanitized_prompt) > max_length else ''}"

        return {
            "isValid": is_valid,
            "errorMessage": None if is_valid else "Error, the text sent has sensitive data -> {}".format(
                truncated_string),
            "riskScore": risk_score

        }
    except Exception as e:
        print(e)
        return {
            "isValid": False,
            "errorMessage": str(e),
            "riskScore": 1
        }


TIME_OUT = os.getenv("VALIDATOR_TIME_OUT", "250")


def validate(text):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(_validate, text)

        try:
            # Wait for a maximum of 5 seconds for the task to complete
            return future.result(timeout=int(TIME_OUT))
        except concurrent.futures.TimeoutError:
            print("Task detect_pii timed out and was interrupted! after {} seconds".format(TIME_OUT))
            return None


def predict(body: dict, model) -> dict:
    return validate(body["text"])
