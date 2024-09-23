# from guardrails.hub import WikiProvenance
# from guardrails import Guard
#
#
# def validate(text, topic_name):
#     try:
#         guard = Guard().use(
#             WikiProvenance,
#             topic_name=topic_name,
#             validation_method="sentence",
#             llm_callable="gpt-3.5-turbo",
#             on_fail="exception"
#         )
#         response = guard.validate(text, metadata={"pass_on_invalid": True})
#         return {
#             "validation_passed": True,
#             "response": response.validated_output
#         }
#     except Exception as e:
#         print(e)
#         return {
#             "validation_passed": False,
#             "response": str(e)
#         }
