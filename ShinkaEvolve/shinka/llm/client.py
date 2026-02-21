from typing import Any, Tuple
import os
from pathlib import Path

import anthropic
from dotenv import load_dotenv
from google import genai

# instructor is used for Anthropic structured output too, so keep it required.
import instructor

from .providers.pricing import get_provider

env_path = Path(__file__).parent.parent.parent / ".env"
load_dotenv(dotenv_path=env_path, override=True)

TIMEOUT = 600


def _require_openai():
    """
    Lazily import openai so Bedrock/Anthropic-only users don't need the package
    (or OPENAI_API_KEY) at import-time.
    """
    try:
        import openai  # type: ignore
    except ImportError as e:
        raise ImportError(
            "OpenAI provider requested but the `openai` package is not installed. "
            "Install it (pip install openai) or switch to a non-OpenAI provider."
        ) from e
    return openai


def get_client_llm(model_name: str, structured_output: bool = False) -> Tuple[Any, str]:
    provider = get_provider(model_name)

    if provider == "anthropic":
        client = anthropic.Anthropic(timeout=TIMEOUT)
        if structured_output:
            client = instructor.from_anthropic(
                client, mode=instructor.mode.Mode.ANTHROPIC_JSON
            )

    elif provider == "bedrock":
        client = anthropic.AnthropicBedrock(
            aws_access_key=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            aws_region=os.getenv("AWS_REGION_NAME"),
            timeout=TIMEOUT,
        )
        if structured_output:
            client = instructor.from_anthropic(
                client, mode=instructor.mode.Mode.ANTHROPIC_JSON
            )

    elif provider == "openai":
        openai = _require_openai()
        client = openai.OpenAI(timeout=TIMEOUT)
        if structured_output:
            client = instructor.from_openai(client, mode=instructor.Mode.TOOLS_STRICT)

    elif model_name.startswith("azure-"):
        openai = _require_openai()
        model_name = model_name.split("azure-")[-1]
        client = openai.AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_API_ENDPOINT") + "openai/v1/",
            timeout=TIMEOUT,
        )
        if structured_output:
            client = instructor.from_openai(client, mode=instructor.Mode.TOOLS_STRICT)

    elif provider == "deepseek":
        openai = _require_openai()
        client = openai.OpenAI(
            api_key=os.environ["DEEPSEEK_API_KEY"],
            base_url="https://api.deepseek.com",
            timeout=TIMEOUT,
        )
        if structured_output:
            client = instructor.from_openai(client, mode=instructor.Mode.MD_JSON)

    elif provider == "google":
        client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        if structured_output:
            # This path relies on instructor's OpenAI-style wrapper.
            # If you want *zero* OpenAI-package dependency, keep structured_output=False for google.
            client = instructor.from_openai(client, mode=instructor.Mode.GEMINI_JSON)

    elif provider == "fugu":
        openai = _require_openai()
        client = openai.OpenAI(
            api_key=os.environ["FUGU_API_KEY"],
            base_url=os.environ["FUGU_BASE_URL"],
            timeout=TIMEOUT,
        )
        if structured_output:
            raise ValueError("Fugu does not support structured output.")

    elif provider == "openrouter":
        openai = _require_openai()
        client = openai.OpenAI(
            api_key=os.environ["OPENROUTER_API_KEY"],
            base_url="https://openrouter.ai/api/v1",
            timeout=TIMEOUT,
        )
        if structured_output:
            client = instructor.from_openai(client, mode=instructor.Mode.MD_JSON)

    else:
        raise ValueError(f"Model {model_name} not supported.")

    return client, model_name

def get_async_client_llm(model_name: str, structured_output: bool = False) -> Tuple[Any, str]:
    provider = get_provider(model_name)

    if provider == "anthropic":
        client = anthropic.AsyncAnthropic(timeout=TIMEOUT)
        if structured_output:
            client = instructor.from_anthropic(
                client, mode=instructor.mode.Mode.ANTHROPIC_JSON
            )

    elif provider == "bedrock":
        client = anthropic.AsyncAnthropicBedrock(
            aws_access_key=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            aws_region=os.getenv("AWS_REGION_NAME"),
            timeout=TIMEOUT,
        )
        if structured_output:
            client = instructor.from_anthropic(
                client, mode=instructor.mode.Mode.ANTHROPIC_JSON
            )

    elif provider == "openai":
        openai = _require_openai()
        client = openai.AsyncOpenAI(timeout=TIMEOUT)
        if structured_output:
            client = instructor.from_openai(client, mode=instructor.Mode.TOOLS_STRICT)

    elif model_name.startswith("azure-"):
        openai = _require_openai()
        model_name = model_name.split("azure-")[-1]
        client = openai.AsyncAzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_API_ENDPOINT"),
            timeout=TIMEOUT,
        )
        if structured_output:
            client = instructor.from_openai(client, mode=instructor.Mode.TOOLS_STRICT)

    elif provider == "deepseek":
        openai = _require_openai()
        client = openai.AsyncOpenAI(
            api_key=os.environ["DEEPSEEK_API_KEY"],
            base_url="https://api.deepseek.com",
            timeout=TIMEOUT,
        )
        if structured_output:
            client = instructor.from_openai(client, mode=instructor.Mode.MD_JSON)

    elif provider == "google":
        client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        if structured_output:
            raise ValueError("Gemini does not support structured output.")

    elif provider == "fugu":
        openai = _require_openai()
        client = openai.AsyncOpenAI(
            api_key=os.environ["FUGU_API_KEY"],
            base_url=os.environ["FUGU_BASE_URL"],
            timeout=TIMEOUT,
        )
        if structured_output:
            raise ValueError("Fugu does not support structured output.")

    elif provider == "openrouter":
        openai = _require_openai()
        client = openai.AsyncOpenAI(
            api_key=os.environ["OPENROUTER_API_KEY"],
            base_url="https://openrouter.ai/api/v1",
            timeout=TIMEOUT,
        )
        if structured_output:
            client = instructor.from_openai(client, mode=instructor.Mode.MD_JSON)

    else:
        raise ValueError(f"Model {model_name} not supported.")

    return client, model_name