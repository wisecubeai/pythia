from llm_guard.output_scanners import Relevance


def validate(text, original_prompt):
    try:

        scanner = Relevance(threshold=0.5)
        sanitized_output, is_valid, risk_score = scanner.scan(text, original_prompt)
        return {
            "isValid": is_valid,
            "errorMessage": None if is_valid else sanitized_output,
            "riskScore": risk_score

        }
    except Exception as e:
        print(e)
        return {
            "isValid": False,
            "errorMessage": str(e),
            "riskScore": 1
        }
