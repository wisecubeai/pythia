from pythia.validators import detect_pii as detect_pii_method
from pythia.validators import detect_toxicity, detect_gibberish, detect_prompt_injection, detect_secrets
from llm_guard.input_scanners import Gibberish, Toxicity, PromptInjection, Anonymize, Secrets, BanSubstrings, BanTopics
from llm_guard.input_scanners.ban_substrings import MatchType
from pythia.validators import detect_pii as detect_pii_method
from pythia.validators import secrets_present as secrets_present_method
from pythia.validators import qa_relevance as qa_relevance_method
from pythia.validators import saliency_check as saliency_check_method

from llm_guard.input_scanners.toxicity import MatchType
from llm_guard.input_scanners.gibberish import MatchType
from llm_guard.input_scanners.prompt_injection import MatchType
from llm_guard.input_scanners.anonymize_helpers import BERT_LARGE_NER_CONF
from llm_guard.vault import Vault

vault = Vault()


def format_response(is_valid, error_response, risk_score):
    return {
        "isValid": is_valid,
        "errorMessage": error_response,
        "riskScore": risk_score
    }


def get_input_data(validator, kwargs):
    try:
        if "input" not in validator:
            return ""
        if validator["input"] is None:
            return ""
        return ", ".join(kwargs[validator["input"]])
    except Exception as e:
        return ""


def get_output_data(validator, kwargs):
    try:
        if "output" not in validator:
            return ""
        if validator["output"] is None:
            return ""
        return ", ".join(kwargs[validator["output"]])
    except Exception as e:
        return ""


def get_question(kwargs):
    try:
        if "question" not in kwargs:
            return ""
        if kwargs["question"] is None:
            return ""
        return kwargs["question"]
    except Exception as e:
        return ""


class ValidatorCall:
    def detect_pii(self, validator, **kwargs):
        question = get_question(kwargs=kwargs)
        input_data_reference = get_input_data(validator=validator, kwargs=kwargs)
        output_data_response = get_output_data(validator=validator, kwargs=kwargs)
        validator_results = []
        if input_data_reference:
            validator_result = detect_pii_method.validate(text=input_data_reference)
            validator_result["validatedField"] = validator["input"]
            validator_result["validator"] = validator
            validator_results.append(validator_result)
        if output_data_response:
            validator_result = detect_pii_method.validate(text=output_data_response)
            validator_result["validatedField"] = validator["output"]
            validator_result["validator"] = validator
            validator_results.append(validator_result)
        return validator_results

    def detect_gibberish(self, validator, **kwargs):
        input_data_reference = get_input_data(validator=validator, kwargs=kwargs)
        output_data_response = get_output_data(validator=validator, kwargs=kwargs)
        validator_results = []
        if input_data_reference:
            validator_result = detect_gibberish.validate(text=input_data_reference)
            validator_result["validatedField"] = validator["input"]
            validator_result["validator"] = validator
            validator_results.append(validator_result)
        if output_data_response:
            validator_result = detect_gibberish.validate(text=output_data_response)
            validator_result["validatedField"] = validator["output"]
            validator_result["validator"] = validator
            validator_results.append(validator_result)
        return validator_results

    # def detect_prompt_injection(self, **kwargs):
    #     print("detect prompt injection called with", kwargs)
    #     results = []
    #     input_data = kwargs["input"]
    #     try:
    #         scanner = PromptInjection(threshold=0.5, match_type=MatchType.FULL)
    #         if isinstance(input_data, list):
    #             for item in input_data:
    #                 sanitized_prompt, is_valid, risk_score = scanner.scan(item)
    #                 results.append({
    #                     "item": item,
    #                     "is_valid": is_valid,
    #                     "sanitized_prompt": sanitized_prompt,
    #                     "risk_score": risk_score
    #                 })
    #         else:
    #             sanitized_prompt, is_valid, risk_score = scanner.scan(input_data)
    #             results.append({
    #                 "item": input_data,
    #                 "is_valid": is_valid,
    #                 "sanitized_prompt": sanitized_prompt,
    #                 "risk_score": risk_score
    #             })
    #         return results
    #     except Exception as e:
    #         if isinstance(input_data, list):
    #             for item in input_data:
    #                 results.append({
    #                     "item": item,
    #                     "is_valid": False,
    #                     "sanitized_prompt": str(e),
    #                     "risk_score": 1
    #                 })
    #         else:
    #             results.append({
    #                 "item": input_data,
    #                 "is_valid": False,
    #                 "sanitized_prompt": str(e),
    #                 "risk_score": 1
    #             })
    #         return results

    def detect_toxicity(self, validator, **kwargs):
        input_data_reference = get_input_data(validator=validator, kwargs=kwargs)
        output_data_response = get_output_data(validator=validator, kwargs=kwargs)
        validator_results = []
        if input_data_reference:
            validator_result = detect_toxicity.validate(text=input_data_reference)
            validator_result["validatedField"] = validator["input"]
            validator_result["validator"] = validator
            validator_results.append(validator_result)
        if output_data_response:
            validator_result = detect_toxicity.validate(text=output_data_response)
            validator_result["validatedField"] = validator["output"]
            validator_result["validator"] = validator
            validator_results.append(validator_result)
        return validator_results

    def detect_prompt_injection(self, validator, **kwargs):
        input_data_reference = get_input_data(validator=validator, kwargs=kwargs)
        output_data_response = get_output_data(validator=validator, kwargs=kwargs)
        validator_results = []
        if input_data_reference:
            validator_result = detect_prompt_injection.validate(text=input_data_reference)
            validator_result["validatedField"] = validator["input"]
            validator_result["validator"] = validator
            validator_results.append(validator_result)
        if output_data_response:
            validator_result = detect_prompt_injection.validate(text=output_data_response)
            validator_result["validatedField"] = validator["output"]
            validator_result["validator"] = validator
            validator_results.append(validator_result)
        return validator_results

    def ban_substrings(self, **kwargs):
        print("ban substrings called with", kwargs)
        try:
            competitors_names = [
                "finances"
            ]
            scanner = BanSubstrings(
                substrings=competitors_names,
                match_type=MatchType.STR,
                case_sensitive=False,
                redact=False,
                contains_all=False,
            )
            sanitized_prompt, is_valid, risk_score = scanner.scan(kwargs["input_response"])
            return format_response(is_valid, sanitized_prompt, risk_score)
        except Exception as e:
            # print(e)
            return format_response(False, str(e), 1)

    # def ban_topics(self, **kwargs):
    #     print("ban topics called with", kwargs)
    #     try:
    #         scanner = BanTopics(topics=["drugs"], threshold=0.5)
    #         sanitized_prompt, is_valid, risk_score = scanner.scan(kwargs["input_response"])
    #         return format_response(is_valid, sanitized_prompt, risk_score)
    #     except Exception as e:
    #         # print(e)
    #         return format_response(False, str(e), 1)
    def detect_secrets(self, validator, **kwargs):
        question = get_question(kwargs=kwargs)
        input_data_reference = get_input_data(validator=validator, kwargs=kwargs)
        output_data_response = get_output_data(validator=validator, kwargs=kwargs)
        validator_results = []
        if input_data_reference:
            validator_result = secrets_present_method.validate(text=input_data_reference)
            validator_result["validatedField"] = validator["input"]
            validator_result["validator"] = validator
            validator_results.append(validator_result)
        if output_data_response:
            validator_result = secrets_present_method.validate(text=output_data_response)
            validator_result["validatedField"] = validator["output"]
            validator_result["validator"] = validator
            validator_results.append(validator_result)
        return validator_results

    def detect_relevance(self, validator, **kwargs):
        question = get_question(kwargs=kwargs)
        input_data_reference = get_input_data(validator=validator, kwargs=kwargs)
        output_data_response = get_output_data(validator=validator, kwargs=kwargs)
        validator_results = []
        if input_data_reference:
            validator_result = qa_relevance_method.validate(text=input_data_reference,
                                                            original_prompt=output_data_response)
            validator_result["validatedField"] = validator["input"]
            validator_result["validator"] = validator
            validator_results.append(validator_result)
        if output_data_response:
            validator_result = qa_relevance_method.validate(text=input_data_reference,
                                                            original_prompt=output_data_response)
            validator_result["validatedField"] = validator["output"]
            validator_result["validator"] = validator
            validator_results.append(validator_result)
        return validator_results

    def detect_factual_consistency(self, validator, **kwargs):
        question = get_question(kwargs=kwargs)
        input_data_reference = get_input_data(validator=validator, kwargs=kwargs)
        output_data_response = get_output_data(validator=validator, kwargs=kwargs)
        validator_results = []
        if input_data_reference:
            validator_result = saliency_check_method.validate(text=output_data_response, documents=input_data_reference)
            validator_result["validatedField"] = validator["input"]
            validator_result["validator"] = validator
            validator_results.append(validator_result)
        if output_data_response:
            validator_result = saliency_check_method.validate(text=output_data_response, documents=input_data_reference)
            validator_result["validatedField"] = validator["output"]
            validator_result["validator"] = validator
            validator_results.append(validator_result)
        return validator_results
