"""
These are the prompts for hallucination detection as done by a simple grading system
"""


GRADING_SUMMARY_SYSTEM = """
Your job is to grade a summary of a provided reference. 
Use A-F school grades.
The summary should be factually consistent with the reference. 
The less consistent the summary is, the worse the grade it should receive.

Your output should be in JSON FORMAT with
the keys "reasoning" and "verdict":
{"reasoning": <your reasoning as bullet points>, "verdict": <your final grade>}"""


GRADING_SUMMARY_TEMPLATE = """
reference: 
{reference}

summary: 
{summary}
"""


GRADING_SUMMARY_W_Q_SYSTEM = """
Your job is to grade a summary of a provided reference focusing on things relevant to the question. 
Use A-F school grades.
The summary should be factually consistent with the reference with special focus on facts relevant to the question.
The less consistent the summary is, the worse the grade it should receive.

Your output should be in JSON FORMAT with
the keys "reasoning" and "verdict":
{"reasoning": <your reasoning as bullet points>, "verdict": <your final grade>}"""


GRADING_SUMMARY_W_Q_TEMPLATE = """
reference: 
{reference}

question: 
{question}

summary: 
{summary}
"""


GRADING_QA_SYSTEM = """
Your job is to grade an answer to a given question with a provided context. 
Use A-F school grades.
The answer should be factually consistent with the context. 
The less consistent the answer is, the worse the grade it should receive.

Your output should be in JSON FORMAT with
the keys "reasoning" and "verdict":
{"reasoning": <your reasoning as bullet points>, "verdict": <your final grade>}
"""


GRADING_QA_TEMPLATE = """
context: 
{context}

question: 
{question}

answer: 
{answer}
"""


GRADING_NO_CONTEXT_QA_SYSTEM = """
Your job is to grade the answer to a given question. 
Use A-F school grades.
The answer should be factually accurate. 
The less accurate it is, the worse the grade it should receive.

Your output should be in JSON FORMAT with
the keys "reasoning" and "verdict":
{"reasoning": <your reasoning as bullet points>, "verdict": <your final grade>}"""


GRADING_NO_CONTEXT_QA_TEMPLATE = """
question: 
{question}

answer: 
{answer}
"""