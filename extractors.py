import re
from abc import ABC, abstractmethod
from typing import Optional, List

import litellm

from models import HostedModel
from prompts import pythiav1

class BaseExtractor(ABC):
    @abstractmethod
    def extract(self, text: str, question: Optional[str], **kwargs) -> List[List[str]]:
        pass


class PythiaV1Extractor(BaseExtractor):
    """
    This class implements a simple A-F grade-based evaluator
    """
    def __init__(self, model: HostedModel):
        self.model = model

    @staticmethod
    def _process_extraction(extraction: str):
        extraction = extraction.strip()
        extraction = PythiaV1Extractor._parse_claim_triplets(extraction)
        return extraction

    @staticmethod
    def _parse_triplets(pattern: str, text: str, triple_length: int = 3) -> List[List[str]]:
        triplets = []
        matches = re.findall(pattern, text)
        for m in matches:
            try:
                t = eval(m)
            except SyntaxError:
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

    @staticmethod
    def _parse_claim_triplets(text: str) -> List[List[str]]:
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
            triplets = PythiaV1Extractor._parse_triplets(p, text, triple_length=3)
            if triplets:
                ret += triplets

        # deduplication
        final_triple_set = []
        for t in ret:
            if tuple(t) not in final_triple_set:
                final_triple_set.append(tuple(t))

        return [list(t) for t in final_triple_set]

    def extract(self, text: str, question: Optional[str], **kwargs) -> List[List[str]]:
        if question is None:
            prompt = pythiav1.GPT4_TRIPLET_EXTRACTION_PROMPT.format(text=text)
        else:
            prompt = pythiav1.GPT4_TRIPLET_EXTRACTION_PROMPT_Q.format(text=text, question=question)
        extraction = litellm.completion(
            model=self.model.model,
            api_key=self.model.api_key,
            api_base=self.model.api_base,
            messages=[{
                "role": "user",
                "content": prompt
            }],
            **kwargs
        )
        if not extraction.choices:
            return []
        extraction = extraction.choices[0].message.content
        if not (extraction and len(extraction)):
            return []
        if '###' in extraction:
            kg_str = extraction[:extraction.index('###')]
        else:
            kg_str = extraction
        triplets = PythiaV1Extractor._parse_claim_triplets(kg_str)
        return triplets


__all__ = ["BaseExtractor", "PythiaV1Extractor"]
