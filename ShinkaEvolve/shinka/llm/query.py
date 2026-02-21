from typing import List, Optional, Dict
from pydantic import BaseModel
from .client import get_client_llm, get_async_client_llm
from .providers.pricing import get_provider
from .providers.result import QueryResult  
import logging

logger = logging.getLogger(__name__)


def query(
    model_name: str,
    msg: str,
    system_msg: str,
    msg_history: List = [],
    output_model: Optional[BaseModel] = None,
    model_posteriors: Optional[Dict[str, float]] = None,
    **kwargs,
) -> QueryResult:
    client, model_name = get_client_llm(
        model_name, structured_output=output_model is not None
    )
    provider = get_provider(model_name)

    if provider in ("anthropic", "bedrock") or "anthropic" in model_name:
        from .providers.anthropic import query_anthropic as query_fn
    elif provider in ("openai", "fugu", "openrouter"):
        from .providers.openai import query_openai as query_fn
    elif provider == "deepseek":
        from .providers.deepseek import query_deepseek as query_fn
    elif provider == "google":
        from .providers.gemini import query_gemini as query_fn
    else:
        raise ValueError(f"Model {model_name} not supported.")

    return query_fn(
        client,
        model_name,
        msg,
        system_msg,
        msg_history,
        output_model,
        model_posteriors=model_posteriors,
        **kwargs,
    )


async def query_async(
    model_name: str,
    msg: str,
    system_msg: str,
    msg_history: List = [],
    output_model: Optional[BaseModel] = None,
    model_posteriors: Optional[Dict[str, float]] = None,
    **kwargs,
) -> QueryResult:
    client, model_name = get_async_client_llm(
        model_name, structured_output=output_model is not None
    )
    provider = get_provider(model_name)

    if provider in ("anthropic", "bedrock") or "anthropic" in model_name:
        from .providers.anthropic import query_anthropic_async as query_fn
    elif provider in ("openai", "fugu", "openrouter"):
        from .providers.openai import query_openai_async as query_fn
    elif provider == "deepseek":
        from .providers.deepseek import query_deepseek_async as query_fn
    elif provider == "google":
        from .providers.gemini import query_gemini_async as query_fn
    else:
        raise ValueError(f"Model {model_name} not supported.")

    return await query_fn(
        client,
        model_name,
        msg,
        system_msg,
        msg_history,
        output_model,
        model_posteriors=model_posteriors,
        **kwargs,
    )