from llm_guard.input_scanners import Secrets


def validate(text):
    try:
        scanner = Secrets()
        sanitized_prompt, is_valid, risk_score = scanner.scan(text)
        return {
            "isValid": is_valid,
            "errorMessage": None if is_valid else sanitized_prompt,
            "riskScore": risk_score

        }
    except Exception as e:
        print(e)
        return {
            "isValid": False,
            "errorMessage": str(e),
            "riskScore": 1
        }

