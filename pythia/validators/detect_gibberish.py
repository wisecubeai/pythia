from llm_guard.input_scanners.gibberish import Gibberish,MatchType

def validate(text):
    try:
        max_length = 500
        scanner = Gibberish(match_type=MatchType.FULL, threshold=0.8)
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
