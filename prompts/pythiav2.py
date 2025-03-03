"""
This module contains the prompts for Pythia V2
"""

PYTHIAV2_EXTRACTION_SYSTEM = ("You are an AI assistant made to claim extraction on text and return the claims "
                              "formatted as triples. Each triple should be formatted like (entity, relationship, "
                              "property) and should represent a claim in the text. Entities will usually be proper "
                              "nouns and/or specific things. Do not try to fill in blanks, instead blanks should be "
                              "treated as N/A.")

PYTHIAV2_EXTRACTION_TEMPLATE = """Extract all triples from this document:\n{text}"""

PYTHIAV2_CHECKING_SYSTEM = ("You are an AI assistant made to check if triples extracted from a summary are consistent "
                            "with the information in the corresponding document. All triples should be formated as "
                            "{entity, relationship, property}. For each triple in the summary triples, label it as "
                            "entailment if it is supported by the document, neutral if it is neither supported nor "
                            "refuted, and contradiction if there is information in the document triples that "
                            "contradicts the given summary triple. Give a small explanation for the choice of label.  "
                            "Do not try to fill in blanks or evaluate triples with N/A in them, these triples should "
                            "be labeled as Incomplete.")

PYTHIAV2_CHECKING_TEMPLATE = ("The summary triples are: {response_triples} and the document triples are: "
                              "{reference_triples}")

