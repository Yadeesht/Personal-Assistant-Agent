import os
from typing import Dict

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from utils.helper import setup_logger
import dotenv

dotenv.load_dotenv()

logger = setup_logger(__name__)


SCOPES = {
    "gmail": [
        "https://mail.google.com/",
        "https://www.googleapis.com/auth/gmail.modify",
        "https://www.googleapis.com/auth/gmail.settings.basic",
        "https://www.googleapis.com/auth/gmail.settings.sharing",
    ],
    "calendar": [
        "https://www.googleapis.com/auth/calendar",
        "https://www.googleapis.com/auth/calendar.readonly",
        "https://www.googleapis.com/auth/calendar.events",
    ],
    "gchat": [
        "https://www.googleapis.com/auth/chat.messages.readonly",
        "https://www.googleapis.com/auth/chat.messages",
        "https://www.googleapis.com/auth/chat.spaces",
    ],
    "gdrive": [
        "https://www.googleapis.com/auth/drive",
        "https://www.googleapis.com/auth/drive.file",
    ],
    "tasks": [
        "https://www.googleapis.com/auth/tasks",
        "https://www.googleapis.com/auth/tasks.readonly",
    ],
    "slides": [
        "https://www.googleapis.com/auth/presentations",
        "https://www.googleapis.com/auth/presentations.readonly",
    ],
    "sheets": [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/spreadsheets.readonly",
    ],
    "forms": [
        "https://www.googleapis.com/auth/forms.body",
        "https://www.googleapis.com/auth/forms.responses.readonly",
    ],
    "docs": [
        "https://www.googleapis.com/auth/documents",
        "https://www.googleapis.com/auth/documents.readonly",
    ],
}

_service_cache: Dict[str, any] = {}


def get_google_service(
    service_type: str,
    scope_key: str,
    token_path: str,
    creds_path: str,
    force_refresh: bool = False,
):
    """
    Get or create Google API service with proper authentication.

    Args:
        service_type: Type of service ('gmail', 'calendar', etc.)
        scope_key: Key for scopes in SCOPES dict
        token_path: Path to token file (service-specific)
        creds_path: Path to credentials file
        force_refresh: Force creation of new service

    Returns:
        Authenticated Google API service
    """

    # Custom Search uses API key, not OAuth
    if service_type == "customsearch":
        cache_key = "customsearch_api_key"
        if not force_refresh and cache_key in _service_cache:
            logger.info(f"Returning cached service for {cache_key}")
            return _service_cache[cache_key]

        api_key = os.environ.get("GOOGLE_PSE_API_KEY")
        if not api_key:
            raise ValueError(
                "GOOGLE_PSE_API_KEY environment variable not set. "
                "Get your API key from https://console.cloud.google.com/apis/credentials"
            )

        logger.info("Building Custom Search service with API key")
        service = build("customsearch", "v1", developerKey=api_key)
        _service_cache[cache_key] = service
        logger.info("Custom Search service created and cached successfully")
        return service

    cache_key = f"{service_type}_{scope_key}"
    if not force_refresh and cache_key in _service_cache:
        logger.info(f"Returning cached service for {cache_key}")
        return _service_cache[cache_key]

    creds = None
    scopes = SCOPES.get(scope_key, [])

    logger.info(f"Authenticating {service_type} with scopes: {scope_key}")

    # Load existing credentials from service-specific token file
    if os.path.exists(token_path):
        logger.info(f"Loading credentials from {token_path}")
        creds = Credentials.from_authorized_user_file(token_path, scopes)

    # If there are no (valid) credentials available, let the user log in
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            logger.info("Refreshing expired credentials")
            try:
                creds.refresh(Request())
            except Exception as e:
                logger.warning(f"Token refresh failed: {e}. Starting new auth flow.")
                creds = None

        if not creds:
            logger.info(f"Starting new authentication flow using {creds_path}")
            flow = InstalledAppFlow.from_client_secrets_file(creds_path, scopes)
            creds = flow.run_local_server(port=0)

        # Save the credentials for the next run
        logger.info(f"Saving credentials to {token_path}")
        os.makedirs(os.path.dirname(token_path), exist_ok=True)
        with open(token_path, "w") as token:
            token.write(creds.to_json())

    api_name_overrides = {"gchat": "chat", "gdrive": "drive"}
    api_service_name = api_name_overrides.get(service_type, service_type)
    version = (
        "v1"
        if service_type in ["gmail", "gchat", "tasks", "slides", "forms", "docs"]
        else "v4"
        if service_type in ["sheets"]
        else "v3"
    )

    logger.info(f"Building {api_service_name} service version {version}")
    service = build(api_service_name, version, credentials=creds)

    # Cache the service
    _service_cache[cache_key] = service
    logger.info(f"Service {cache_key} created and cached successfully")

    return service


def clear_service_cache():
    """Clear the service cache"""
    global _service_cache
    _service_cache.clear()
    logger.info("Service cache cleared")
