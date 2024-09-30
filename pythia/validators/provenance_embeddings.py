# from guardrails.hub import ProvenanceEmbeddings
# from guardrails import Guard
# import numpy as np
#
# try:
#     from sentence_transformers import SentenceTransformer
# except ImportError:
#     raise ImportError(
#         "This example requires the `sentence-transformers` package. "
#         "Install it with `pip install sentence-transformers`, and try again."
#     )
#
# MODEL = SentenceTransformer("paraphrase-MiniLM-L6-v2")
#
#
# # Create embed function
# def embed_function(sources: list[str]) -> np.array:
#     return MODEL.encode(sources)
#
#
# def validate(text, sources):
#     try:
#         guard = Guard().use(
#             ProvenanceEmbeddings,
#             threshold=0.2,  # Lower the threshold to make the validator stricter
#             validation_method="sentence",
#             on_fail="exception",
#         )
#
#         response = guard.validate(text,
#                                   metadata={"sources": sources, "embed_function": embed_function,
#                                             "pass_on_invalid": True},
#                                   )
#         return {
#             "validation_passed": True,
#             "response": response
#         }
#     except Exception as e:
#         print(e)
#         return {
#             "validation_passed": False,
#             "response": e
#         }
