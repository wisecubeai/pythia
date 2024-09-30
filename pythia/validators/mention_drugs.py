# from guardrails import Guard
# from guardrails.hub import MentionsDrugs
#
#
# def validate(text):
#     try:
#         guard = Guard().use(MentionsDrugs, on_fail="exception")
#         response = guard.validate(text)
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
