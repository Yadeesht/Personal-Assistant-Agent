from langchain_openai import AzureChatOpenAI

from config.settings import (
    AZURE_AI_CREDENTIAL,
    AZURE_AI_ENDPOINT,
    AZURE_API_VERSION,
    MODEL_NAME,
    MODEL_TEMPERATURE,
    MODEL_TOOL_REASONING_EFFORT,
    MAX_RETRIES,
    REQUEST_TIMEOUT,
)
from utils.helper import setup_logger

logger = setup_logger(__name__)


def _endpoint() -> str:
    """The Azure resource endpoint.

    A full request URL copied from the Azure portal
    (.../openai/deployments/<name>/chat/completions?api-version=...) pins every call to
    that deployment and silently ignores MODEL_NAME, so it is cut back to the resource
    address.
    """
    endpoint = AZURE_AI_ENDPOINT or ""
    if "/openai/" in endpoint:
        logger.warning(
            "AZURE_AI_ENDPOINT is a full request URL; using only the resource address so "
            "MODEL_NAME picks the deployment. Set it to https://<resource>.../ in .env."
        )
        endpoint = endpoint.split("/openai/", 1)[0].rstrip("/") + "/"
    return endpoint


def _build_llm(model: str = None, with_tools: bool = False):
    options = {}
    if MODEL_TEMPERATURE is not None:
        options["temperature"] = MODEL_TEMPERATURE
    if with_tools and MODEL_TOOL_REASONING_EFFORT:
        options["reasoning_effort"] = MODEL_TOOL_REASONING_EFFORT
    return AzureChatOpenAI(
        azure_deployment=model or MODEL_NAME,
        azure_endpoint=_endpoint(),
        api_key=AZURE_AI_CREDENTIAL,
        api_version=AZURE_API_VERSION,
        max_retries=MAX_RETRIES,
        timeout=REQUEST_TIMEOUT,
        **options,
    )


def build_llm_with_tools(tools, model: str = None, **bind_kwargs):
    return _build_llm(model, with_tools=True).bind_tools(tools, **bind_kwargs)


def build_llm(model: str = None):
    return _build_llm(model)
