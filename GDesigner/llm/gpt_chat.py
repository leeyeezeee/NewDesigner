import aiohttp
import httpx
import math
from typing import Any, Awaitable, Callable, Dict, List, Optional, TypeVar, Union
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_random_exponential,
)
from dotenv import load_dotenv
import os
from openai import (
    APIConnectionError,
    APITimeoutError,
    AsyncOpenAI,
    NotFoundError,
    OpenAI,
)

from GDesigner.llm.format import Message
from GDesigner.llm.price import cost_count, remote_token_usage_or_raise
from GDesigner.llm.llm import LLM, LLMGeneration, TokenLogProb
from GDesigner.llm.llm_registry import LLMRegistry


OPENAI_API_KEYS = ['']
BASE_URL = ''
_OPENAI_COMPATIBLE_TIMEOUT_SECONDS = 1200.0
_OPENAI_CONNECT_TIMEOUT_SECONDS = 20.0
_OPENAI_POOL_TIMEOUT_SECONDS = 20.0
_OPENAI_WRITE_TIMEOUT_SECONDS = 120.0
_ASYNC_OPENAI_CLIENTS: Dict[tuple[str, str], AsyncOpenAI] = {}
_DIAGNOSTIC_TEXT_LIMIT = 500
_NETWORK_RETRY_ATTEMPTS = 5
_NETWORK_RETRY_MAX_WAIT_SECONDS = 30
_T = TypeVar("_T")

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
        "max_retries": 0,
    }
    if base_url:
        client_kwargs["base_url"] = base_url
    return client_kwargs


def _get_async_openai_client(base_url: str) -> AsyncOpenAI:
    """Reuse one bounded keep-alive pool per OpenAI-compatible backend."""
    api_key = _agent_api_key(base_url)
    cache_key = (base_url, api_key)
    client = _ASYNC_OPENAI_CLIENTS.get(cache_key)
    if client is not None:
        return client

    timeout = httpx.Timeout(
        timeout=_OPENAI_COMPATIBLE_TIMEOUT_SECONDS,
        connect=_OPENAI_CONNECT_TIMEOUT_SECONDS,
        pool=_OPENAI_POOL_TIMEOUT_SECONDS,
        read=_OPENAI_COMPATIBLE_TIMEOUT_SECONDS,
        write=_OPENAI_WRITE_TIMEOUT_SECONDS,
    )
    http_client = httpx.AsyncClient(
        timeout=timeout,
        limits=httpx.Limits(
            max_connections=64,
            max_keepalive_connections=32,
            keepalive_expiry=60.0,
        ),
    )
    client_kwargs: Dict[str, Any] = {
        "api_key": api_key,
        "timeout": timeout,
        "max_retries": 0,
        "http_client": http_client,
    }
    if base_url:
        client_kwargs["base_url"] = base_url
    client = AsyncOpenAI(**client_kwargs)
    _ASYNC_OPENAI_CLIENTS[cache_key] = client
    return client


def _should_retry_openai_request(exception: BaseException) -> bool:
    return isinstance(
        exception,
        (APIConnectionError, APITimeoutError, NotFoundError),
    )


@retry(
    wait=wait_random_exponential(multiplier=1, max=_NETWORK_RETRY_MAX_WAIT_SECONDS),
    stop=stop_after_attempt(_NETWORK_RETRY_ATTEMPTS),
    retry=retry_if_exception(_should_retry_openai_request),
    reraise=True,
)
async def _async_openai_request(operation: Callable[[], Awaitable[_T]]) -> _T:
    """Retry transport failures, timeouts, and HTTP 404 responses."""
    return await operation()


@retry(
    wait=wait_random_exponential(multiplier=1, max=_NETWORK_RETRY_MAX_WAIT_SECONDS),
    stop=stop_after_attempt(_NETWORK_RETRY_ATTEMPTS),
    retry=retry_if_exception(_should_retry_openai_request),
    reraise=True,
)
def _sync_openai_request(operation: Callable[[], _T]) -> _T:
    """Synchronous counterpart of :func:`_async_openai_request`."""
    return operation()


def _get_attr_or_key(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


class EmptyChatCompletionError(RuntimeError):
    """An OpenAI-compatible request succeeded but returned no usable content."""


def _diagnostic_preview(value: Any) -> str:
    if value is None:
        return "None"
    text = str(value)
    suffix = "..." if len(text) > _DIAGNOSTIC_TEXT_LIMIT else ""
    return repr(text[:_DIAGNOSTIC_TEXT_LIMIT] + suffix)


def _usage_field(usage: Any, name: str) -> Any:
    return _get_attr_or_key(usage, name) if usage is not None else None


def _choice_contents_or_raise(
    response: Any,
    *,
    requested_model: str,
    base_url: str,
    request_kwargs: Dict[str, Any],
) -> List[str]:
    outputs: List[str] = []
    for choice_position, choice in enumerate(response.choices):
        message = _get_attr_or_key(choice, "message")
        content = _get_attr_or_key(message, "content", "") or ""
        if str(content).strip():
            outputs.append(str(content))
            continue

        usage = _get_attr_or_key(response, "usage")
        reasoning_content = _get_attr_or_key(message, "reasoning_content")
        reasoning = _get_attr_or_key(message, "reasoning")
        completion_details = _usage_field(usage, "completion_tokens_details")
        chat_template_kwargs = (
            request_kwargs.get("extra_body", {}).get("chat_template_kwargs", {})
        )
        raise EmptyChatCompletionError(
            "OpenAI-compatible chat completion returned blank message.content.\n"
            f"requested_model: {requested_model!r}\n"
            f"response_model: {_get_attr_or_key(response, 'model')!r}\n"
            f"base_url: {base_url!r}\n"
            f"response_id: {_get_attr_or_key(response, 'id')!r}\n"
            f"choice_position: {choice_position}\n"
            f"choice_index: {_get_attr_or_key(choice, 'index')!r}\n"
            f"finish_reason: {_get_attr_or_key(choice, 'finish_reason')!r}\n"
            f"content: {_diagnostic_preview(content)}\n"
            f"reasoning_content_length: {len(str(reasoning_content or ''))}\n"
            f"reasoning_content_preview: {_diagnostic_preview(reasoning_content)}\n"
            f"reasoning_length: {len(str(reasoning or ''))}\n"
            f"reasoning_preview: {_diagnostic_preview(reasoning)}\n"
            f"refusal: {_diagnostic_preview(_get_attr_or_key(message, 'refusal'))}\n"
            f"prompt_tokens: {_usage_field(usage, 'prompt_tokens')!r}\n"
            f"completion_tokens: {_usage_field(usage, 'completion_tokens')!r}\n"
            f"total_tokens: {_usage_field(usage, 'total_tokens')!r}\n"
            f"completion_tokens_details: {completion_details!r}\n"
            f"request_max_tokens: {request_kwargs.get('max_tokens')!r}\n"
            f"request_temperature: {request_kwargs.get('temperature')!r}\n"
            f"request_n: {request_kwargs.get('n')!r}\n"
            f"request_enable_thinking: {chat_template_kwargs.get('enable_thinking')!r}"
        )
    return outputs


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
        if logprob is not None:
            logprob = float(logprob)
        token_logprobs.append(
            TokenLogProb(
                token=str(token),
                logprob=logprob,
                probability=_probability_from_logprob(logprob),
                bytes=token_bytes,
            )
        )
    return token_logprobs


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
    client = _get_async_openai_client(base_url)
    response = await _async_openai_request(
        lambda: client.chat.completions.create(**request_kwargs)
    )
    outputs = _choice_contents_or_raise(
        response,
        requested_model=model,
        base_url=base_url,
        request_kwargs=request_kwargs,
    )
    prompt = "".join([item["content"] for item in msg])
    prompt_tokens, completion_tokens = remote_token_usage_or_raise(
        response,
        requested_model=model,
        base_url=base_url,
    )
    cost_count(
        prompt,
        "\n".join(outputs),
        model,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )
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
    client = OpenAI(**_openai_client_kwargs(base_url))
    response = _sync_openai_request(
        lambda: client.chat.completions.create(**request_kwargs)
    )
    outputs = _choice_contents_or_raise(
        response,
        requested_model=model,
        base_url=base_url,
        request_kwargs=request_kwargs,
    )
    prompt = "".join([item["content"] for item in msg])
    prompt_tokens, completion_tokens = remote_token_usage_or_raise(
        response,
        requested_model=model,
        base_url=base_url,
    )
    cost_count(
        prompt,
        "\n".join(outputs),
        model,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )
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
