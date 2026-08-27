import os
from typing import Any, Optional, Tuple

from GDesigner.utils.globals import Cost, PromptTokens, CompletionTokens, LLMCalls
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

class MissingRemoteTokenUsageError(RuntimeError):
    """Raised when an OpenAI-compatible response omits required token usage."""


def _optional_price_per_1k(name: str) -> Optional[float]:
    """Read an optional non-negative per-1K-token price from the environment."""
    raw_value = os.getenv(name)
    if raw_value is None or not raw_value.strip():
        return None
    try:
        value = float(raw_value)
    except ValueError as exc:
        raise ValueError(
            f"{name} must be a number or left unset; received {raw_value!r}."
        ) from exc
    if value < 0:
        raise ValueError(f"{name} must be non-negative; received {value}.")
    return value


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

    if prompt_tokens is None or completion_tokens is None:
        prompt_len = cal_token(model_name, prompt)
        completion_len = cal_token(model_name, response)
    else:
        prompt_len = int(prompt_tokens)
        completion_len = int(completion_tokens)
    custom_input_price = _optional_price_per_1k(
        "LOCAL_MODEL_INPUT_PRICE_PER_1K"
    )
    custom_output_price = _optional_price_per_1k(
        "LOCAL_MODEL_OUTPUT_PRICE_PER_1K"
    )
    if (custom_input_price is None) != (custom_output_price is None):
        raise ValueError(
            "LOCAL_MODEL_INPUT_PRICE_PER_1K and "
            "LOCAL_MODEL_OUTPUT_PRICE_PER_1K must either both be numeric or "
            "both be unset/blank."
        )
    if custom_input_price is not None and custom_output_price is not None:
        branch = "custom"
        price = prompt_len * custom_input_price / 1000 + \
            completion_len * custom_output_price / 1000
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
    LLMCalls.instance().value += 1

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



