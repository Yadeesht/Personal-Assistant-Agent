"""
Google Custom Search (PSE) Tools

This module provides tools for interacting with Google Programmable Search Engine.
"""

import asyncio
import os
from typing import Optional, List, Literal
from pathlib import Path

from app_tools.auth.service_decoder import get_google_service
from app_tools.core.server_init import supervisor_server
from utils.helper import setup_logger

logger = setup_logger(__name__)


def get_service():
    """Get Gmail service using shared authentication."""
    base_dir = Path(__file__).parent.parent.parent
    token_path = str(base_dir / "cred" / "gsearch_token.json")
    creds_path = str(base_dir / "cred" / "setup_cred.json")

    return get_google_service(
        service_type="customsearch",
        scope_key="customsearch",
        token_path=token_path,
        creds_path=creds_path,
    )


@supervisor_server.tool()
async def search_custom(
    q: str,
    num: int = 10,
    start: int = 1,
    safe: Literal["active", "moderate", "off"] = "off",
    search_type: Optional[Literal["image"]] = None,
    site_search: Optional[str] = None,
    site_search_filter: Optional[Literal["e", "i"]] = None,
    date_restrict: Optional[str] = None,
    file_type: Optional[str] = None,
    language: Optional[str] = None,
    country: Optional[str] = None,
) -> str:
    """
    Performs a search using Google Custom Search JSON API.

    Args:
        q (str): The search query. Required.
        num (int): Number of results to return (1-10). Defaults to 10.
        start (int): The index of the first result to return (1-based). Defaults to 1.
        safe (Literal["active", "moderate", "off"]): Safe search level. Defaults to "off".
        search_type (Optional[Literal["image"]]): Search for images if set to "image".
        site_search (Optional[str]): Restrict search to a specific site/domain.
        site_search_filter (Optional[Literal["e", "i"]]): Exclude ("e") or include ("i") site_search results.
        date_restrict (Optional[str]): Restrict results by date (e.g., "d5" for past 5 days, "m3" for past 3 months).
        file_type (Optional[str]): Filter by file type (e.g., "pdf", "doc").
        language (Optional[str]): Language code for results (e.g., "lang_en").
        country (Optional[str]): Country code for results (e.g., "countryUS").

    Returns:
        str: Formatted search results including title, link, and snippet for each result.
    """
    service = get_service()
    # Get API key and search engine ID from environment
    api_key = os.environ.get("GOOGLE_PSE_API_KEY")
    if not api_key:
        raise ValueError(
            "GOOGLE_PSE_API_KEY environment variable not set. Please set it to your Google Custom Search API key."
        )

    cx = os.environ.get("GOOGLE_PSE_ENGINE_ID")
    if not cx:
        raise ValueError(
            "GOOGLE_PSE_ENGINE_ID environment variable not set. Please set it to your Programmable Search Engine ID."
        )

    logger.info(f"[search_custom] Invoked. Query: '{q}', CX: '{cx}'")

    # Build the request parameters
    params = {
        "key": api_key,
        "cx": cx,
        "q": q,
        "num": num,
        "start": start,
        "safe": safe,
    }

    # Add optional parameters
    if search_type:
        params["searchType"] = search_type
    if site_search:
        params["siteSearch"] = site_search
    if site_search_filter:
        params["siteSearchFilter"] = site_search_filter
    if date_restrict:
        params["dateRestrict"] = date_restrict
    if file_type:
        params["fileType"] = file_type
    if language:
        params["lr"] = language
    if country:
        params["cr"] = country

    # Execute the search request
    result = await asyncio.to_thread(service.cse().list(**params).execute)

    # Extract search information
    search_info = result.get("searchInformation", {})
    total_results = search_info.get("totalResults", "0")
    search_time = search_info.get("searchTime", 0)

    # Extract search results
    items = result.get("items", [])

    # Format the response
    confirmation_message = f"""Search Results:
- Query: "{q}"
- Search Engine ID: {cx}
- Total Results: {total_results}
- Search Time: {search_time:.3f} seconds
- Results Returned: {len(items)} (showing {start} to {start + len(items) - 1})

"""

    if items:
        confirmation_message += "Results:\n"
        for i, item in enumerate(items, start):
            title = item.get("title", "No title")
            link = item.get("link", "No link")
            snippet = item.get("snippet", "No description available").replace("\n", " ")

            confirmation_message += f"\n{i}. {title}\n"
            confirmation_message += f"   URL: {link}\n"
            confirmation_message += f"   Snippet: {snippet}\n"

            # Add additional metadata if available
            if "pagemap" in item:
                pagemap = item["pagemap"]
                if "metatags" in pagemap and pagemap["metatags"]:
                    metatag = pagemap["metatags"][0]
                    if "og:type" in metatag:
                        confirmation_message += f"   Type: {metatag['og:type']}\n"
                    if "article:published_time" in metatag:
                        confirmation_message += (
                            f"   Published: {metatag['article:published_time'][:10]}\n"
                        )
    else:
        confirmation_message += "\nNo results found."

    # Add information about pagination
    queries = result.get("queries", {})
    if "nextPage" in queries:
        next_start = queries["nextPage"][0].get("startIndex", 0)
        confirmation_message += (
            f"\n\nTo see more results, search again with start={next_start}"
        )

    logger.info("Search completed successfully")
    return confirmation_message


@supervisor_server.tool()
async def get_search_engine_info() -> str:
    """
    Retrieves metadata about a Programmable Search Engine.

    Returns:
        str: Information about the search engine including its configuration and available refinements.
    """
    service = get_service()
    # Get API key and search engine ID from environment
    api_key = os.environ.get("GOOGLE_PSE_API_KEY")
    if not api_key:
        raise ValueError(
            "GOOGLE_PSE_API_KEY environment variable not set. Please set it to your Google Custom Search API key."
        )

    cx = os.environ.get("GOOGLE_PSE_ENGINE_ID")
    if not cx:
        raise ValueError(
            "GOOGLE_PSE_ENGINE_ID environment variable not set. Please set it to your Programmable Search Engine ID."
        )

    logger.info(f"[get_search_engine_info] Invoked. CX: '{cx}'")

    # Perform a minimal search to get the search engine context
    params = {
        "key": api_key,
        "cx": cx,
        "q": "test",  # Minimal query to get metadata
        "num": 1,
    }

    result = await asyncio.to_thread(service.cse().list(**params).execute)

    # Extract context information
    context = result.get("context", {})
    title = context.get("title", "Unknown")

    confirmation_message = f"""Search Engine Information:
- Search Engine ID: {cx}
- Title: {title}
"""

    # Add facet information if available
    if "facets" in context:
        confirmation_message += "\nAvailable Refinements:\n"
        for facet in context["facets"]:
            for item in facet:
                label = item.get("label", "Unknown")
                anchor = item.get("anchor", "Unknown")
                confirmation_message += f"  - {label} (anchor: {anchor})\n"

    # Add search information
    search_info = result.get("searchInformation", {})
    if search_info:
        total_results = search_info.get("totalResults", "Unknown")
        confirmation_message += "\nSearch Statistics:\n"
        confirmation_message += f"  - Total indexed results: {total_results}\n"

    logger.info("Search engine info retrieved successfully")
    return confirmation_message


@supervisor_server.tool()
async def search_custom_siterestrict(
    q: str,
    sites: List[str],
    num: int = 10,
    start: int = 1,
    safe: Literal["active", "moderate", "off"] = "off",
) -> str:
    """
    Performs a search restricted to specific sites using Google Custom Search.

    Args:
        q (str): The search query. Required.
        sites (List[str]): List of sites/domains to search within.
        num (int): Number of results to return (1-10). Defaults to 10.
        start (int): The index of the first result to return (1-based). Defaults to 1.
        safe (Literal["active", "moderate", "off"]): Safe search level. Defaults to "off".

    Returns:
        str: Formatted search results from the specified sites.
    """
    service = get_service()
    logger.info(f"[search_custom_siterestrict] Invoked. Query: '{q}', Sites: {sites}")

    # Build site restriction query
    site_query = " OR ".join([f"site:{site}" for site in sites])
    full_query = f"{q} ({site_query})"

    # Use the main search function with the modified query
    return await search_custom(
        service=service,
        q=full_query,
        num=num,
        start=start,
        safe=safe,
    )
