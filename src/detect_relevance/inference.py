from llm_guard.output_scanners import Bias
from llm_guard.output_scanners.bias import MatchType
import concurrent
import os


def load_model(model_dir: str):
    """
    Load the model from the specified directory.
    """
    return "Load Model"


def _validate(text):
    try:
        scanner = Bias(threshold=0.5, match_type=MatchType.FULL)
        sanitized_output, is_valid, risk_score = scanner.scan(None, text)

        return {
            "isValid": is_valid,
            "errorMessage": None if is_valid else "Error,bias detected ",
            "riskScore": risk_score

        }
    except Exception as e:
        print(e)
        return {
            "isValid": False,
            "errorMessage": str(e),
            "riskScore": 1
        }


TIME_OUT = os.getenv("VALIDATOR_TIME_OUT", "30")


def validate(text):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(_validate, text)

        try:
            # Wait for a maximum of 5 seconds for the task to complete
            return future.result(timeout=int(TIME_OUT))
        except concurrent.futures.TimeoutError:
            print("Task detect_bias timed out and was interrupted! after {} seconds".format(TIME_OUT))
            return None


def predict(body: dict, model) -> dict:
    return validate(body["text"])
