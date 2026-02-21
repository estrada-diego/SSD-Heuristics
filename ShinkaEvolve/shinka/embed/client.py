from typing import Any, Tuple
import os
from pathlib import Path

import boto3
import botocore
from dotenv import load_dotenv
from google import genai

from .providers.pricing import get_provider

env_path = Path(__file__).parent.parent.parent / ".env"
load_dotenv(dotenv_path=env_path, override=True)

def _require_openai():
    try:
        import openai  # type: ignore
    except ImportError as e:
        raise ImportError(
            "Embedding provider requested OpenAI/Azure, but `openai` is not installed. "
            "Install it (pip install openai) or switch embedding_model to a non-OpenAI provider."
        ) from e
    return openai

env_path = Path(__file__).parent.parent.parent / ".env"
load_dotenv(dotenv_path=env_path, override=True)


def get_client_embed(model_name: str) -> Tuple[Any, str]:
    provider = get_provider(model_name)

    if provider == "openai":
        openai = _require_openai()
        client = openai.OpenAI(timeout=600)

    elif provider == "azure":
        openai = _require_openai()
        model_name = model_name.split("azure-")[-1]
        client = openai.AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_API_ENDPOINT"),
            timeout=600,
        )

    elif provider == "google":
        client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

    elif provider == "bedrock":
        # Bedrock Runtime client (used by embedding.py to invoke the model)
        region = os.getenv("AWS_REGION_NAME") or os.getenv("AWS_REGION") or "us-east-1"
        # credentials are optional here if you use standard AWS env/instance profile
        client = boto3.client(
            "bedrock-runtime",
            region_name=region,
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            config=botocore.config.Config(read_timeout=300),
        )

    else:
        raise ValueError(f"Embedding model {model_name} not supported.")

    return client, model_name

def get_async_client_embed(model_name: str) -> Tuple[Any, str]:
    provider = get_provider(model_name)

    if provider == "openai":
        openai = _require_openai()
        client = openai.AsyncOpenAI()

    elif provider == "azure":
        openai = _require_openai()
        model_name = model_name.split("azure-")[-1]
        client = openai.AsyncAzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_API_ENDPOINT"),
        )

    elif provider == "google":
        client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

    elif provider == "bedrock":
        region = os.getenv("AWS_REGION_NAME") or os.getenv("AWS_REGION") or "us-east-1"
        client = boto3.client(
            "bedrock-runtime",
            region_name=region,
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            config=botocore.config.Config(read_timeout=300),
        )

    else:
        raise ValueError(f"Embedding model {model_name} not supported.")

    return client, model_name