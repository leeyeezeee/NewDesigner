from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Union, Optional

from GDesigner.llm.format import Message


@dataclass(frozen=True)
class TokenLogProb:
    token: str
    logprob: Optional[float]
    probability: Optional[float] = None
    bytes: Optional[List[int]] = None


@dataclass(frozen=True)
class LLMGeneration:
    content: str
    token_logprobs: List[TokenLogProb]


class LLM(ABC):
    DEFAULT_MAX_TOKENS = 1000
    DEFAULT_TEMPERATURE = 0.2
    DEFUALT_NUM_COMPLETIONS = 1

    @abstractmethod
    async def agen(
        self,
        messages: List[Message],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        num_comps: Optional[int] = None,
        return_logprobs: bool = False,
        top_logprobs: Optional[int] = None,
        ) -> Union[List[str], str, List[LLMGeneration], LLMGeneration]:

        pass

    @abstractmethod
    def gen(
        self,
        messages: List[Message],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        num_comps: Optional[int] = None,
        return_logprobs: bool = False,
        top_logprobs: Optional[int] = None,
        ) -> Union[List[str], str, List[LLMGeneration], LLMGeneration]:

        pass
