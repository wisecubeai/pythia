from llm_guard.output_scanners import FactualConsistency


def validate(text, documents):
    try:
        scanner = FactualConsistency(minimum_score=0.7)
        sanitized_output, is_valid, risk_score = scanner.scan(documents[0], text)

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
