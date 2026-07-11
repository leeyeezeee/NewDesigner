import os
from argparse import ArgumentParser, Namespace


def add_agent_backend_args(parser: ArgumentParser) -> None:
    parser.add_argument(
        "--agent_base_url",
        type=str,
        default=None,
        help=(
            "Override AGENT_BASE_URL/BASE_URL from .env for the main agent LLM. "
            "For local vLLM, use a URL like http://localhost:8005/v1."
        ),
    )
    parser.add_argument(
        "--agent_api_type",
        type=str,
        default=None,
        help="Override AGENT_API_TYPE/LLM_API_TYPE from .env, e.g. vllm or openai-compatible.",
    )
    parser.add_argument(
        "--agent_api_key",
        type=str,
        default=None,
        help=(
            "Override AGENT_API_KEY/API_KEY from .env for the main agent LLM. "
            "If --agent_base_url is set and this is omitted, EMPTY is used."
        ),
    )


def _set_env(name: str, value: str | None) -> None:
    if value is None:
        return
    os.environ[name] = value


def apply_agent_backend_args(args: Namespace) -> None:
    base_url = getattr(args, "agent_base_url", None)
    api_type = getattr(args, "agent_api_type", None)
    api_key = getattr(args, "agent_api_key", None)

    _set_env("AGENT_BASE_URL", base_url)
    if base_url is not None:
        os.environ["BASE_URL"] = base_url

    _set_env("AGENT_API_TYPE", api_type)
    if api_type is not None:
        os.environ["LLM_API_TYPE"] = api_type

    if api_key is not None:
        os.environ["AGENT_API_KEY"] = api_key
        os.environ["API_KEY"] = api_key
    elif base_url is not None:
        os.environ["AGENT_API_KEY"] = "EMPTY"
