import aiohttp
import math
from typing import List, Union, Optional
from tenacity import retry, wait_random_exponential, stop_after_attempt
from typing import Dict, Any
from dotenv import load_dotenv
import os
from openai import AsyncOpenAI, OpenAI

from GDesigner.llm.format import Message
from GDesigner.llm.price import cost_count
from GDesigner.llm.llm import LLM, LLMGeneration, TokenLogProb
from GDesigner.llm.llm_registry import LLMRegistry


OPENAI_API_KEYS = ['']
BASE_URL = ''
_OPENAI_COMPATIBLE_TIMEOUT_SECONDS = 1200.0

load_dotenv()


def _agent_base_url() -> str:
    return os.getenv("AGENT_BASE_URL") or os.getenv("BASE_URL") or ""


def _is_openai_compatible(base_url: str) -> bool:
    api_type = os.getenv("AGENT_API_TYPE") or os.getenv("LLM_API_TYPE") or ""
    if api_type.lower() in {"openai", "vllm", "openai-compatible"}:
        return True
    normalized = base_url.rstrip("/")
    return normalized.endswith("/v1") or "/v1/" in normalized


def _agent_api_key(base_url: str) -> str:
    api_key = os.getenv("AGENT_API_KEY") or os.getenv("API_KEY") or ""
    if base_url and not api_key and _is_openai_compatible(base_url):
        return "EMPTY"
    return api_key


def _optional_bool_env(name: str) -> Optional[bool]:
    value = os.getenv(name)
    if value is None or value == "":
        return None
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _chat_completion_extra_body(model: str) -> Dict[str, Any]:
    if "qwen" not in model.lower():
        return {}
    enable_thinking = _optional_bool_env("QWEN_ENABLE_THINKING")
    if enable_thinking is None:
        enable_thinking = False
    return {
        "enable_thinking": enable_thinking,
        "chat_template_kwargs": {
            "enable_thinking": enable_thinking,
        }
    }


@retry(wait=wait_random_exponential(max=100), stop=stop_after_attempt(3))
async def custom_achat(
    model: str,
    msg: List[Dict],
    return_logprobs: bool = False,):
    if return_logprobs:
        raise NotImplementedError(
            "Token logprobs are only implemented for OpenAI-compatible agent backends."
        )
    request_url = _agent_base_url()
    authorization_key = _agent_api_key(request_url)
    headers = {
        'Content-Type': 'application/json',
        'authorization': authorization_key
    }
    data = {
        "name": model,
        "inputs": {
            "stream": False,
            "msg": repr(msg),
        }
    }
    async with aiohttp.ClientSession() as session:
        async with session.post(request_url, headers=headers ,json=data) as response:
            response_data = await response.json()
            prompt = "".join([item['content'] for item in msg])
            cost_count(prompt,response_data['data'],model)
            return response_data['data']


def _message_dicts(messages: List[Message]) -> List[Dict[str, Any]]:
    if isinstance(messages, str):
        return [{"role": "user", "content": messages}]
    return [
        message if isinstance(message, dict) else {"role": message.role, "content": message.content}
        for message in messages
    ]


def _openai_client_kwargs(base_url: str) -> Dict[str, Any]:
    client_kwargs = {
        "api_key": _agent_api_key(base_url),
        "timeout": _OPENAI_COMPATIBLE_TIMEOUT_SECONDS,
    }
    if base_url:
        client_kwargs["base_url"] = base_url
    return client_kwargs


def _get_attr_or_key(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _probability_from_logprob(logprob: Optional[float]) -> Optional[float]:
    if logprob is None:
        return None
    if logprob < -745:
        return 0.0
    return math.exp(logprob)


def _choice_token_logprobs(choice: Any) -> List[TokenLogProb]:
    logprobs = _get_attr_or_key(choice, "logprobs")
    content = _get_attr_or_key(logprobs, "content", []) if logprobs else []
    if not content:
        return []

    token_logprobs = []
    for token_info in content:
        token = _get_attr_or_key(token_info, "token", "")
        logprob = _get_attr_or_key(token_info, "logprob")
        token_bytes = _get_attr_or_key(token_info, "bytes")
        top_logprobs = []
        for top_info in _get_attr_or_key(token_info, "top_logprobs", []) or []:
            top_token = _get_attr_or_key(top_info, "token", "")
            top_logprob = _get_attr_or_key(top_info, "logprob")
            top_bytes = _get_attr_or_key(top_info, "bytes")
            if top_logprob is not None:
                top_logprob = float(top_logprob)
            top_logprobs.append(
                {
                    "token": str(top_token),
                    "logprob": top_logprob,
                    "probability": _probability_from_logprob(top_logprob),
                    "bytes": top_bytes,
                }
            )
        if logprob is not None:
            logprob = float(logprob)
        token_logprobs.append(
            TokenLogProb(
                token=str(token),
                logprob=logprob,
                probability=_probability_from_logprob(logprob),
                bytes=token_bytes,
                top_logprobs=top_logprobs,
            )
        )
    return token_logprobs


def _choice_content(choice: Any) -> str:
    message = _get_attr_or_key(choice, "message")
    return _get_attr_or_key(message, "content", "") or ""


async def openai_compatible_achat(
    model: str,
    msg: List[Dict],
    max_tokens: int,
    temperature: float,
    num_comps: int,
    return_logprobs: bool = False,
    top_logprobs: Optional[int] = None,
) -> Union[List[str], str, List[LLMGeneration], LLMGeneration]:
    base_url = _agent_base_url()
    request_kwargs = {
        "model": model,
        "messages": msg,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "n": num_comps,
    }
    if return_logprobs:
        request_kwargs["logprobs"] = True
        if top_logprobs is not None:
            request_kwargs["top_logprobs"] = top_logprobs
    extra_body = _chat_completion_extra_body(model)
    if extra_body:
        request_kwargs["extra_body"] = extra_body
    response = await AsyncOpenAI(**_openai_client_kwargs(base_url)).chat.completions.create(
        **request_kwargs,
    )
    outputs = [_choice_content(choice) for choice in response.choices]
    prompt = "".join([item["content"] for item in msg])
    for output in outputs:
        cost_count(prompt, output, model)
    if return_logprobs:
        generations = [
            LLMGeneration(
                content=output,
                token_logprobs=_choice_token_logprobs(choice),
            )
            for output, choice in zip(outputs, response.choices)
        ]
        return generations[0] if num_comps == 1 else generations
    return outputs[0] if num_comps == 1 else outputs


def openai_compatible_chat(
    model: str,
    msg: List[Dict],
    max_tokens: int,
    temperature: float,
    num_comps: int,
    return_logprobs: bool = False,
    top_logprobs: Optional[int] = None,
) -> Union[List[str], str, List[LLMGeneration], LLMGeneration]:
    base_url = _agent_base_url()
    request_kwargs = {
        "model": model,
        "messages": msg,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "n": num_comps,
    }
    if return_logprobs:
        request_kwargs["logprobs"] = True
        if top_logprobs is not None:
            request_kwargs["top_logprobs"] = top_logprobs
    extra_body = _chat_completion_extra_body(model)
    if extra_body:
        request_kwargs["extra_body"] = extra_body
    response = OpenAI(**_openai_client_kwargs(base_url)).chat.completions.create(
        **request_kwargs,
    )
    outputs = [_choice_content(choice) for choice in response.choices]
    prompt = "".join([item["content"] for item in msg])
    for output in outputs:
        cost_count(prompt, output, model)
    if return_logprobs:
        generations = [
            LLMGeneration(
                content=output,
                token_logprobs=_choice_token_logprobs(choice),
            )
            for output, choice in zip(outputs, response.choices)
        ]
        return generations[0] if num_comps == 1 else generations
    return outputs[0] if num_comps == 1 else outputs

@LLMRegistry.register('GPTChat')
class GPTChat(LLM):

    def __init__(self, model_name: str):
        self.model_name = model_name

    async def agen(
        self,
        messages: List[Message],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        num_comps: Optional[int] = None,
        return_logprobs: bool = False,
        top_logprobs: Optional[int] = None,
        ) -> Union[List[str], str, List[LLMGeneration], LLMGeneration]:

        if max_tokens is None:
            max_tokens = self.DEFAULT_MAX_TOKENS
        if temperature is None:
            temperature = self.DEFAULT_TEMPERATURE
        if num_comps is None:
            num_comps = self.DEFUALT_NUM_COMPLETIONS
        
        messages = _message_dicts(messages)
        base_url = _agent_base_url()
        if _is_openai_compatible(base_url):
            return await openai_compatible_achat(
                self.model_name,
                messages,
                max_tokens,
                temperature,
                num_comps,
                return_logprobs=return_logprobs,
                top_logprobs=top_logprobs,
            )
        if return_logprobs:
            raise NotImplementedError(
                "Token logprobs are only implemented for OpenAI-compatible agent backends."
            )
        return await custom_achat(self.model_name, messages, return_logprobs=return_logprobs)
    
    def gen(
        self,
        messages: List[Message],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        num_comps: Optional[int] = None,
        return_logprobs: bool = False,
        top_logprobs: Optional[int] = None,
    ) -> Union[List[str], str, List[LLMGeneration], LLMGeneration]:
        if max_tokens is None:
            max_tokens = self.DEFAULT_MAX_TOKENS
        if temperature is None:
            temperature = self.DEFAULT_TEMPERATURE
        if num_comps is None:
            num_comps = self.DEFUALT_NUM_COMPLETIONS

        messages = _message_dicts(messages)
        base_url = _agent_base_url()
        if _is_openai_compatible(base_url):
            return openai_compatible_chat(
                self.model_name,
                messages,
                max_tokens,
                temperature,
                num_comps,
                return_logprobs=return_logprobs,
                top_logprobs=top_logprobs,
            )
        raise NotImplementedError("Synchronous generation is only implemented for OpenAI-compatible agent backends.")
