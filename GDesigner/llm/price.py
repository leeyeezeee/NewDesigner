import os
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Dict, Iterator, Optional, Tuple

from GDesigner.utils.globals import Cost, PromptTokens, CompletionTokens
import tiktoken
# GPT-4:  https://platform.openai.com/docs/models/gpt-4-and-gpt-4-turbo
# GPT3.5: https://platform.openai.com/docs/models/gpt-3-5
# DALL-E: https://openai.com/pricing

def cal_token(model:str, text:str):
    try:
        encoder = tiktoken.encoding_for_model(model)
    except KeyError:
        encoder = tiktoken.get_encoding("cl100k_base")
    num_tokens = len(encoder.encode(text))
    return num_tokens

_ACTIVE_GRAPH_TOKEN_USAGE: ContextVar[Optional[Dict[str, Any]]] = ContextVar(
    "active_graph_token_usage",
    default=None,
)


class MissingRemoteTokenUsageError(RuntimeError):
    """Raised when an OpenAI-compatible response omits required token usage."""


def remote_token_usage_or_raise(
    response: Any,
    *,
    requested_model: str,
    base_url: str,
) -> Tuple[int, int]:
    usage = getattr(response, "usage", None)
    prompt_tokens = getattr(usage, "prompt_tokens", None)
    completion_tokens = getattr(usage, "completion_tokens", None)
    if usage is None or prompt_tokens is None or completion_tokens is None:
        raise MissingRemoteTokenUsageError(
            "OpenAI-compatible response did not include required remote token "
            "usage fields. Local tokenization and fallback estimation are "
            "disabled. "
            f"requested_model={requested_model!r}, base_url={base_url!r}, "
            f"response_id={getattr(response, 'id', None)!r}, "
            f"usage={usage!r}."
        )
    prompt_tokens = int(prompt_tokens)
    completion_tokens = int(completion_tokens)
    if prompt_tokens < 0 or completion_tokens < 0:
        raise MissingRemoteTokenUsageError(
            "OpenAI-compatible response returned invalid negative token usage. "
            f"prompt_tokens={prompt_tokens}, "
            f"completion_tokens={completion_tokens}."
        )
    return prompt_tokens, completion_tokens


@contextmanager
def track_graph_token_usage() -> Iterator[Dict[str, Any]]:
    """Track one asynchronously executed graph without mixing concurrent graphs."""
    usage: Dict[str, Any] = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "request_count": 0,
        "token_source": "remote_response_usage",
    }
    context_token = _ACTIVE_GRAPH_TOKEN_USAGE.set(usage)
    try:
        yield usage
    finally:
        _ACTIVE_GRAPH_TOKEN_USAGE.reset(context_token)


def cost_count(
    prompt,
    response,
    model_name,
    *,
    prompt_tokens: Optional[int] = None,
    completion_tokens: Optional[int] = None,
):
    branch: str
    prompt_len: int
    completion_len: int
    price: float

    graph_usage = _ACTIVE_GRAPH_TOKEN_USAGE.get()
    if prompt_tokens is None or completion_tokens is None:
        if graph_usage is not None:
            raise MissingRemoteTokenUsageError(
                "Graph token tracking requires remote prompt_tokens and "
                "completion_tokens. Local tokenization and fallback estimation "
                "are disabled."
            )
        prompt_len = cal_token(model_name, prompt)
        completion_len = cal_token(model_name, response)
    else:
        prompt_len = int(prompt_tokens)
        completion_len = int(completion_tokens)
    custom_input_price = os.getenv("LOCAL_MODEL_INPUT_PRICE_PER_1K")
    custom_output_price = os.getenv("LOCAL_MODEL_OUTPUT_PRICE_PER_1K")
    if custom_input_price is not None and custom_output_price is not None:
        branch = "custom"
        price = prompt_len * float(custom_input_price) / 1000 + \
            completion_len * float(custom_output_price) / 1000
    elif "gpt-4" in model_name and model_name in OPENAI_MODEL_INFO["gpt-4"]:
        branch = "gpt-4"
        price = prompt_len * OPENAI_MODEL_INFO[branch][model_name]["input"] /1000 + \
                completion_len * OPENAI_MODEL_INFO[branch][model_name]["output"] /1000
    elif "gpt-3.5" in model_name and model_name in OPENAI_MODEL_INFO["gpt-3.5"]:
        branch = "gpt-3.5"
        price = prompt_len * OPENAI_MODEL_INFO[branch][model_name]["input"] /1000 + \
            completion_len * OPENAI_MODEL_INFO[branch][model_name]["output"] /1000
    elif "dall-e" in model_name:
        branch = "dall-e"
        price = 0.0
        prompt_len = 0
        completion_len = 0
    else:
        branch = "other"
        price = 0.0

    Cost.instance().value += price
    PromptTokens.instance().value += prompt_len
    CompletionTokens.instance().value += completion_len

    if graph_usage is not None:
        graph_usage["prompt_tokens"] += prompt_len
        graph_usage["completion_tokens"] += completion_len
        graph_usage["total_tokens"] += prompt_len + completion_len
        graph_usage["request_count"] += 1

    # print(f"Prompt Tokens: {prompt_len}, Completion Tokens: {completion_len}")
    return price, prompt_len, completion_len

OPENAI_MODEL_INFO ={
    "gpt-4": {
        "current_recommended": "gpt-4-1106-preview",
        "gpt-4-0125-preview": {
            "context window": 128000, 
            "training": "Jan 2024", 
            "input": 0.01, 
            "output": 0.03
        },      
        "gpt-4-1106-preview": {
            "context window": 128000, 
            "training": "Apr 2023", 
            "input": 0.01, 
            "output": 0.03
        },
        "gpt-4-vision-preview": {
            "context window": 128000, 
            "training": "Apr 2023", 
            "input": 0.01, 
            "output": 0.03
        },
        "gpt-4": {
            "context window": 8192, 
            "training": "Sep 2021", 
            "input": 0.03, 
            "output": 0.06
        },
        "gpt-4-0314": {
            "context window": 8192, 
            "training": "Sep 2021", 
            "input": 0.03, 
            "output": 0.06
        },
        "gpt-4-32k": {
            "context window": 32768, 
            "training": "Sep 2021", 
            "input": 0.06, 
            "output": 0.12
        },
        "gpt-4-32k-0314": {
            "context window": 32768, 
            "training": "Sep 2021", 
            "input": 0.06, 
            "output": 0.12
        },
        "gpt-4-0613": {
            "context window": 8192, 
            "training": "Sep 2021", 
            "input": 0.06, 
            "output": 0.12
        },
        "gpt-4o": {
            "context window": 128000, 
            "training": "Jan 2024", 
            "input": 0.005, 
            "output": 0.015
        }, 
    },
    "gpt-3.5": {
        "current_recommended": "gpt-3.5-turbo-1106",
        "gpt-3.5-turbo-0125": {
            "context window": 16385, 
            "training": "Jan 2024", 
            "input": 0.0010, 
            "output": 0.0020
        },
        "gpt-3.5-turbo-1106": {
            "context window": 16385, 
            "training": "Sep 2021", 
            "input": 0.0010, 
            "output": 0.0020
        },
        "gpt-3.5-turbo-instruct": {
            "context window": 4096, 
            "training": "Sep 2021", 
            "input": 0.0015, 
            "output": 0.0020
        },
        "gpt-3.5-turbo": {
            "context window": 4096, 
            "training": "Sep 2021", 
            "input": 0.0015, 
            "output": 0.0020
        },
        "gpt-3.5-turbo-0301": {
            "context window": 4096, 
            "training": "Sep 2021", 
            "input": 0.0015, 
            "output": 0.0020
        },
        "gpt-3.5-turbo-0613": {
            "context window": 16384, 
            "training": "Sep 2021", 
            "input": 0.0015, 
            "output": 0.0020
        },
        "gpt-3.5-turbo-16k-0613": {
            "context window": 16384, 
            "training": "Sep 2021", 
            "input": 0.0015, 
            "output": 0.0020
        }
    },
    "dall-e": {
        "current_recommended": "dall-e-3",
        "dall-e-3": {
            "release": "Nov 2023",
            "standard": {
                "1024×1024": 0.040,
                "1024×1792": 0.080,
                "1792×1024": 0.080
            },
            "hd": {
                "1024×1024": 0.080,
                "1024×1792": 0.120,
                "1792×1024": 0.120
            }
        },
        "dall-e-2": {
            "release": "Nov 2022",
            "1024×1024": 0.020,
            "512×512": 0.018,
            "256×256": 0.016
        }
    }
}



