"""
This contains the evaluation prompt from the Lynx paper
https://arxiv.org/abs/2407.08488
Lynx: An Open Source Hallucination Evaluation Model
Selvan Sunitha Ravi, Bartosz Mielczarek, Anand Kannappan, Douwe Kiela, Rebecca Qian

the prompt has been slightly modified
- separating into system and user prompts
- lowercasing the keys in the output JSON
"""


ORIGINAL_TEMPLATE = """Given the following QUESTION, DOCUMENT
and ANSWER you must analyze the provided
answer and determine whether it is
faithful to the contents of the DOCUMENT.
The ANSWER must not offer new information
beyond the context provided in the DOCUMENT.
The ANSWER also must not contradict
information provided in the DOCUMENT.
Output your final verdict by strictly
following this format: "PASS" if the
answer is faithful to the DOCUMENT
and "FAIL" if the answer is not
faithful to the DOCUMENT. Show your
reasoning.

--
QUESTION (THIS DOES NOT COUNT
AS BACKGROUND INFORMATION):
{question}

--
DOCUMENT:
{context}

--
ANSWER:
{answer}

--

Your output should be in JSON FORMAT with
the keys "REASONING" and "SCORE":
{{"REASONING": <your reasoning as
bullet points>, "SCORE": <your final score>}}"""


SIMPLE_QA_SYSTEM = """Given a QUESTION, DOCUMENT
and ANSWER you must analyze the provided
answer and determine whether it is
faithful to the contents of the DOCUMENT.
The ANSWER must not offer new information
beyond the context provided in the DOCUMENT.
The ANSWER also must not contradict
information provided in the DOCUMENT.
Output your final verdict by strictly
following this format: "PASS" if the
answer is faithful to the DOCUMENT
and "FAIL" if the answer is not
faithful to the DOCUMENT. Show your
reasoning.

Your output should be in JSON FORMAT with
the keys "reasoning" and "verdict":
{"reasoning": <your reasoning as bullet points>, "verdict": <your final verdict>}"""

SIMPLE_QA_TEMPLATE = """
QUESTION (THIS DOES NOT COUNT
AS BACKGROUND INFORMATION):
{question}

--
DOCUMENT:
{context}

--
ANSWER:
{answer}"""
