# # Import Guard and Validator
# from guardrails.hub import ExtractedSummarySentencesMatch
# from guardrails import Guard
#
# # Initialize Validator
# val = ExtractedSummarySentencesMatch(
#     threshold=0.8,
#     filepaths="/Users/cosminprata/Work/Wisecube/wisecube-pythia/pythia/validators/docs/doc1.txt"
# )
#
# # Setup Guard
# guard = Guard.from_string(validators=[val])
#
# guard("Summarized text")  # Validator passes
# guard("Inaccurate summary")  # Validator fails
