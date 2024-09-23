from pythia.prompt_gpt import GPT4_TRIPLET_EXTRACTION_PROMPT, GPT4_TRIPLET_EXTRACTION_PROMPT_Q
import re
from openai import OpenAI
import os

api_key = os.getenv("OPENAI_API_KEY", "ollama")
base_url = os.getenv("MODEL_BASE_URL")
print("OpenAI host {}".format(base_url))

client = OpenAI(api_key=api_key, base_url=base_url)
model_name = os.getenv("MODEL_NAME", "gpt-4o")


def parse_triplets(pattern, text, triple_length=3):
    triplets = []
    matches = re.findall(pattern, text)
    for m in matches:
        try:
            t = eval(m)
        except:
            t = m.split(', ')
            if t[0].startswith('('):
                t[0] = t[0][1:]
            if t[-1].endswith(')'):
                t[-1] = t[-1][:-1]
        if len(t) != triple_length:
            continue
        if any([not isinstance(e, str) for e in t]):
            continue
        if any([len(e) == 0 for e in t]):
            continue
        triplets.append(list(t))
    return triplets


def parse_claim_triplets(text):
    ret = []
    patterns = [
        r'\(".*", ".*", ".*"\)',
        r'\(".*", ".*", \'.*\'\)',
        r'\(".*", \'.*\', ".*"\)',
        r'\(\'.*\', ".*", ".*"\)',
        r'\(".*", \'.*\', \'.*\'\)',
        r'\(\'.*\', ".*", \'.*\'\)',
        r'\(\'.*\', \'.*\', ".*"\)',
        r'\(\'.*\', \'.*\', \'.*\'\)'
    ]
    for p in patterns:
        triplets = parse_triplets(p, text, triple_length=3)
        if triplets:
            ret += triplets

    # deduplication
    final_triple_set = []
    for t in ret:
        if tuple(t) not in final_triple_set:
            final_triple_set.append(tuple(t))

    return [list(t) for t in final_triple_set]


def llm_extractor(text, question=None, max_new_tokens=2000, temperature=0.0, model=model_name):
    if question is None:
        prompt = GPT4_TRIPLET_EXTRACTION_PROMPT.format(
            input_text=text
        )
    else:
        prompt = GPT4_TRIPLET_EXTRACTION_PROMPT_Q.format(
            q=question,
            a=text
        )
    messages = [{"role": "user", "content": prompt}]
    choices = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        n=1,
        max_tokens=max_new_tokens
    ).choices
    if choices:
        response = choices[0].message.content
    else:
        return []
    if response and len(response):
        kg_str = None
        if '###' in response:
            kg_str = response[:response.index('###')]
        else:
            kg_str = response
        triplets = parse_claim_triplets(kg_str)
        return triplets
    return []
