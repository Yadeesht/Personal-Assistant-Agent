from langchain_openai import AzureChatOpenAI

from config.settings import (
    AZURE_AI_CREDENTIAL,
    AZURE_AI_ENDPOINT,
    AZURE_API_VERSION,
    MODEL_NAME,
    MAX_RETRIES,
    REQUEST_TIMEOUT,
)


def _build_llm(model: str = None):
    return AzureChatOpenAI(
        azure_deployment=model or MODEL_NAME,
        azure_endpoint=AZURE_AI_ENDPOINT,
        api_key=AZURE_AI_CREDENTIAL,
        api_version=AZURE_API_VERSION,
        max_retries=MAX_RETRIES,
        timeout=REQUEST_TIMEOUT,
    )


def build_llm_with_tools(tools, model: str = None, **bind_kwargs):
    return _build_llm(model).bind_tools(tools, **bind_kwargs)


def build_llm(model: str = None):
    return _build_llm(model)
