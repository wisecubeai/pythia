"""
This module contains the classes and functions to use models in evaluators
"""



from pydantic import BaseModel
from typing import Optional


class HostedModel(BaseModel):
    """This is the dataclass for calling hosted models"""
    model: str
    api_key: Optional[str] = None
    api_base: Optional[str] = None
