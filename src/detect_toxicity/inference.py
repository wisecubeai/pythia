from llm_guard.input_scanners.toxicity import MatchType, Toxicity
import concurrent
import os


def _validate(text):
    try:
        max_length = 500
        scanner = Toxicity(threshold=0.5, match_type=MatchType.SENTENCE)
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


TIME_OUT = os.getenv("VALIDATOR_TIME_OUT", "120")


def validate(text):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(_validate, text)

        try:
            # Wait for a maximum of 5 seconds for the task to complete
            return future.result(timeout=int(TIME_OUT))
        except concurrent.futures.TimeoutError:
            print("Task detect_toxicity timed out and was interrupted! after {} seconds".format(TIME_OUT))
            return None

def load_model(model_dir: str):
    """
    Load the model from the specified directory.
    """
    return "Load Model"




def predict(body: dict, model) -> dict:
    return validate(body["text"])
