"""
This module contains the prompts for Pythia V2
"""
from typing import List, Literal, Optional, Union, Tuple

from pydantic import BaseModel

EXTRACTION_SYSTEM = ("You are an AI assistant made to claim extraction on text and return the claims formatted as "
                     "triples. Each triple should be formatted like (entity, relationship, property) and should "
                     "represent a claim in the text. Entities will usually be proper nouns and/or specific things. "
                     "Do not try to fill in blanks, instead blanks should be treated as N/A.")

EXTRACTION_TEMPLATE = """Extract all triples from this document:\n{text}"""


class ResponseTriple(BaseModel):
    subject: str
    predicate: str
    object: str


class ExtractionResponse(BaseModel):
    triples: List[ResponseTriple]


CHECKING_SYSTEM = ("You are an AI assistant made to check if triples extracted from a summary are consistent with "
                   "the information in the corresponding document. All triples should be formated as {entity, "
                   "relationship, property}. For each triple in the summary triples, label it as entailment if it is "
                   "supported by the document, neutral if it is neither supported nor refuted, and contradiction if "
                   "there is information in the document triples that contradicts the given summary triple. Give a "
                   "small explanation for the choice of label.  Do not try to fill in blanks or evaluate triples "
                   "with N/A in them, these triples should be labeled as Incomplete.")

CHECKING_TEMPLATE = "The summary triples are: {response_triples} and the document triples are: {reference_triples}"


class ResponseClaimCheck(BaseModel):
    claim: ResponseTriple
    category: Literal["entailment", "contradiction", "neutral"]
    reasoning: str


class CheckingResponse(BaseModel):
    claims: List[ResponseClaimCheck]


QA_SYSTEM = ("You are an AI assistant made to check if an answer is an acceptable response for the given question. "
             "Return either True or False plus a brief one sentence explanation.")

QA_TEMPLATE = """question: {question}
answer: {answer}"""

TA_SYSTEM = ("You are an AI assistant made to check if an answer is supported by and faithful to the provided text. "
             "This means no information outside of that found in the text should be present. The answer does not need "
             "to be complete or comprehensive to be supported by the text. Return either True or False plus a brief "
             "one sentence explanation.")

TA_TEMPLATE = """context: {context}
answer: {answer}"""