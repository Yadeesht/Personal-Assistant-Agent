"""
Google Slides Tools

This module provides tools for interacting with Google Slides API.
"""

import asyncio
from typing import List, Dict, Any
from pathlib import Path

from app_tools.auth.service_decoder import get_google_service
from app_tools.core.server_init import content_server
from app_tools.tools.google.workspace_comment_base import create_comment_tools
from app_tools.helper.pydantic_models import (
    CreatePresentationRequest,
    CreatePresentationResponse,
    GetPresentationRequest,
    GetPresentationResponse,
    BatchUpdatePresentationRequest,
    BatchUpdatePresentationResponse,
    GetPageRequest,
    GetPageResponse,
    GetPageThumbnailRequest,
    GetPageThumbnailResponse,
)
from utils.helper import setup_logger

logger = setup_logger(__name__)


def get_service():
    """Get Gmail service using shared authentication."""
    base_dir = Path(__file__).parent.parent.parent
    token_path = str(base_dir / "cred" / "gslide_token.json")
    creds_path = str(base_dir / "cred" / "setup_cred.json")

    return get_google_service(
        service_type="slides",
        scope_key="slides",
        token_path=token_path,
        creds_path=creds_path,
    )


@content_server.tool()
async def create_presentation(title: str = "Untitled Presentation") -> str:
    """
    Create a new Google Slides presentation.

    Args:
        title (str): The title for the new presentation. Defaults to "Untitled Presentation".

    Returns:
        str: Details about the created presentation including ID and URL.
    """
    # Validate input parameters
    try:
        request = CreatePresentationRequest(title=title)
    except Exception as e:
        error_msg = f"Invalid parameters: {str(e)}"
        logger.error(f"[create_presentation] {error_msg}")
        return CreatePresentationResponse(
            status="error", message="", error=error_msg
        ).model_dump_json(indent=2)

    service = get_service()
    logger.info(f"[create_presentation] Invoked. Title: '{request.title}'")

    body = {"title": request.title}

    try:
        result = await asyncio.to_thread(
            service.presentations().create(body=body).execute
        )

        presentation_id = result.get("presentationId")
        presentation_url = (
            f"https://docs.google.com/presentation/d/{presentation_id}/edit"
        )

        confirmation_message = f"""Presentation Created Successfully:
- Title: {request.title}
- Presentation ID: {presentation_id}
- URL: {presentation_url}
- Slides: {len(result.get("slides", []))} slide(s) created"""

        logger.info("Presentation created successfully")
        return CreatePresentationResponse(
            status="success",
            message=confirmation_message,
            presentation_id=presentation_id,
        ).model_dump_json(indent=2)

    except Exception as error:
        error_msg = f"Error creating presentation: {str(error)}"
        logger.error(f"[create_presentation] {error_msg}")
        return CreatePresentationResponse(
            status="error", message="", error=error_msg
        ).model_dump_json(indent=2)


@content_server.tool()
async def get_presentation(presentation_id: str) -> str:
    """
    Get details about a Google Slides presentation.

    Args:
        presentation_id (str): The ID of the presentation to retrieve.

    Returns:
        str: Details about the presentation including title, slides count, and metadata.
    """
    # Validate input parameters
    try:
        request = GetPresentationRequest(presentation_id=presentation_id)
    except Exception as e:
        error_msg = f"Invalid parameters: {str(e)}"
        logger.error(f"[get_presentation] {error_msg}")
        return GetPresentationResponse(
            status="error", message="", error=error_msg
        ).model_dump_json(indent=2)

    service = get_service()
    logger.info(f"[get_presentation] Invoked. ID: '{request.presentation_id}'")

    result = await asyncio.to_thread(
        service.presentations().get(presentationId=request.presentation_id).execute
    )

    title = result.get("title", "Untitled")
    slides = result.get("slides", [])
    page_size = result.get("pageSize", {})

    slides_info = []
    for i, slide in enumerate(slides, 1):
        slide_id = slide.get("objectId", "Unknown")
        page_elements = slide.get("pageElements", [])

        # Collect text from the slide whose JSON structure is very complicated
        # https://googleapis.github.io/google-api-python-client/docs/dyn/slides_v1.presentations.html#get
        slide_text = ""
        try:
            texts_from_elements = []
            for page_element in slide.get("pageElements", []):
                shape = page_element.get("shape", None)
                if shape and shape.get("text", None):
                    text = shape.get("text", None)
                    if text:
                        text_elements_in_shape = []
                        for text_element in text.get("textElements", []):
                            text_run = text_element.get("textRun", None)
                            if text_run:
                                content = text_run.get("content", None)
                                if content:
                                    start_index = text_element.get("startIndex", 0)
                                    text_elements_in_shape.append(
                                        (start_index, content)
                                    )

                        if text_elements_in_shape:
                            # Sort text elements within a single shape
                            text_elements_in_shape.sort(key=lambda item: item[0])
                            full_text_from_shape = "".join(
                                [item[1] for item in text_elements_in_shape]
                            )
                            texts_from_elements.append(full_text_from_shape)

            # cleanup text we collected
            slide_text = "\n".join(texts_from_elements)
            slide_text_rows = slide_text.split("\n")
            slide_text_rows = [row for row in slide_text_rows if len(row.strip()) > 0]
            if slide_text_rows:
                slide_text_rows = ["    > " + row for row in slide_text_rows]
                slide_text = "\n" + "\n".join(slide_text_rows)
            else:
                slide_text = ""
        except Exception as e:
            logger.warning(f"Failed to extract text from the slide {slide_id}: {e}")
            slide_text = f"<failed to extract text: {type(e)}, {e}>"

        slides_info.append(
            f"  Slide {i}: ID {slide_id}, {len(page_elements)} element(s), text: {slide_text if slide_text else 'empty'}"
        )

    confirmation_message = f"""Presentation Details:
- Title: {title}
- Presentation ID: {request.presentation_id}
- URL: https://docs.google.com/presentation/d/{request.presentation_id}/edit
- Total Slides: {len(slides)}
- Page Size: {page_size.get("width", {}).get("magnitude", "Unknown")} x {page_size.get("height", {}).get("magnitude", "Unknown")} {page_size.get("width", {}).get("unit", "")}

Slides Breakdown:
{chr(10).join(slides_info) if slides_info else "  No slides found"}"""

    logger.info("Presentation retrieved successfully")
    return GetPresentationResponse(
        status="success", message=confirmation_message
    ).model_dump_json(indent=2)


@content_server.tool()
async def batch_update_presentation(
    presentation_id: str,
    requests: List[Dict[str, Any]],
) -> str:
    """
    Apply batch updates to a Google Slides presentation.

    Args:
        presentation_id (str): The ID of the presentation to update.
        requests (List[Dict[str, Any]]): List of update requests to apply.

    Returns:
        str: Details about the batch update operation results.
    """
    # Validate input parameters
    try:
        request = BatchUpdatePresentationRequest(
            presentation_id=presentation_id, requests=requests
        )
    except Exception as e:
        error_msg = f"Invalid parameters: {str(e)}"
        logger.error(f"[batch_update_presentation] {error_msg}")
        return BatchUpdatePresentationResponse(
            status="error", message="", error=error_msg
        ).model_dump_json(indent=2)

    service = get_service()
    logger.info(
        f"[batch_update_presentation] Invoked. ID: '{request.presentation_id}', Requests: {len(request.requests)}"
    )

    body = {"requests": request.requests}

    try:
        result = await asyncio.to_thread(
            service.presentations()
            .batchUpdate(presentationId=request.presentation_id, body=body)
            .execute
        )

        replies = result.get("replies", [])

        confirmation_message = f"""Batch Update Completed:
- Presentation ID: {request.presentation_id}
- URL: https://docs.google.com/presentation/d/{request.presentation_id}/edit
- Requests Applied: {len(request.requests)}
- Replies Received: {len(replies)}"""

        if replies:
            confirmation_message += "\n\nUpdate Results:"
            for i, reply in enumerate(replies, 1):
                if "createSlide" in reply:
                    slide_id = reply["createSlide"].get("objectId", "Unknown")
                    confirmation_message += (
                        f"\n  Request {i}: Created slide with ID {slide_id}"
                    )
                elif "createShape" in reply:
                    shape_id = reply["createShape"].get("objectId", "Unknown")
                    confirmation_message += (
                        f"\n  Request {i}: Created shape with ID {shape_id}"
                    )
                else:
                    confirmation_message += f"\n  Request {i}: Operation completed"

        logger.info("Batch update completed successfully")
        return BatchUpdatePresentationResponse(
            status="success", message=confirmation_message
        ).model_dump_json(indent=2)

    except Exception as error:
        error_msg = f"Error updating presentation: {str(error)}"
        logger.error(f"[batch_update_presentation] {error_msg}")
        return BatchUpdatePresentationResponse(
            status="error", message="", error=error_msg
        ).model_dump_json(indent=2)


@content_server.tool()
async def get_page(presentation_id: str, page_object_id: str) -> str:
    """
    Get details about a specific page (slide) in a presentation.

    Args:
        presentation_id (str): The ID of the presentation.
        page_object_id (str): The object ID of the page/slide to retrieve.

    Returns:
        str: Details about the specific page including elements and layout.
    """
    # Validate input parameters
    try:
        request = GetPageRequest(
            presentation_id=presentation_id, page_object_id=page_object_id
        )
    except Exception as e:
        error_msg = f"Invalid parameters: {str(e)}"
        logger.error(f"[get_page] {error_msg}")
        return GetPageResponse(
            status="error", message="", error=error_msg
        ).model_dump_json(indent=2)

    service = get_service()
    logger.info(
        f"[get_page] Invoked. Presentation: '{request.presentation_id}', Page: '{request.page_object_id}'"
    )

    try:
        result = await asyncio.to_thread(
            service.presentations()
            .pages()
            .get(
                presentationId=request.presentation_id,
                pageObjectId=request.page_object_id,
            )
            .execute
        )

        page_type = result.get("pageType", "Unknown")
        page_elements = result.get("pageElements", [])

        elements_info = []
        for element in page_elements:
            element_id = element.get("objectId", "Unknown")
            if "shape" in element:
                shape_type = element["shape"].get("shapeType", "Unknown")
                elements_info.append(f"  Shape: ID {element_id}, Type: {shape_type}")
            elif "table" in element:
                table = element["table"]
                rows = table.get("rows", 0)
                cols = table.get("columns", 0)
                elements_info.append(f"  Table: ID {element_id}, Size: {rows}x{cols}")
            elif "line" in element:
                line_type = element["line"].get("lineType", "Unknown")
                elements_info.append(f"  Line: ID {element_id}, Type: {line_type}")
            else:
                elements_info.append(f"  Element: ID {element_id}, Type: Unknown")

        confirmation_message = f"""Page Details:
- Presentation ID: {request.presentation_id}
- Page ID: {request.page_object_id}
- Page Type: {page_type}
- Total Elements: {len(page_elements)}

Page Elements:
{chr(10).join(elements_info) if elements_info else "  No elements found"}"""

        logger.info("Page retrieved successfully")
        return GetPageResponse(
            status="success", message=confirmation_message
        ).model_dump_json(indent=2)

    except Exception as error:
        error_msg = f"Error retrieving page: {str(error)}"
        logger.error(f"[get_page] {error_msg}")
        return GetPageResponse(
            status="error", message="", error=error_msg
        ).model_dump_json(indent=2)


@content_server.tool()
async def get_page_thumbnail(
    presentation_id: str,
    page_object_id: str,
    thumbnail_size: str = "MEDIUM",
) -> str:
    """
    Generate a thumbnail URL for a specific page (slide) in a presentation.

    Args:
        presentation_id (str): The ID of the presentation.
        page_object_id (str): The object ID of the page/slide.
        thumbnail_size (str): Size of thumbnail ("LARGE", "MEDIUM", "SMALL"). Defaults to "MEDIUM".

    Returns:
        str: URL to the generated thumbnail image.
    """
    # Validate input parameters
    try:
        request = GetPageThumbnailRequest(
            presentation_id=presentation_id,
            page_object_id=page_object_id,
            thumbnail_size=thumbnail_size,
        )
    except Exception as e:
        error_msg = f"Invalid parameters: {str(e)}"
        logger.error(f"[get_page_thumbnail] {error_msg}")
        return GetPageThumbnailResponse(
            status="error", message="", error=error_msg
        ).model_dump_json(indent=2)

    logger.info(
        f"[get_page_thumbnail] Invoked. Presentation: '{request.presentation_id}', Page: '{request.page_object_id}', Size: '{request.thumbnail_size}'"
    )
    service = get_service()

    try:
        result = await asyncio.to_thread(
            service.presentations()
            .pages()
            .getThumbnail(
                presentationId=request.presentation_id,
                pageObjectId=request.page_object_id,
                thumbnailProperties_thumbnailSize=request.thumbnail_size,
                thumbnailProperties_mimeType="PNG",
            )
            .execute
        )

        thumbnail_url = result.get("contentUrl", "")

        confirmation_message = f"""Thumbnail Generated:
- Presentation ID: {request.presentation_id}
- Page ID: {request.page_object_id}
- Thumbnail Size: {request.thumbnail_size}
- Thumbnail URL: {thumbnail_url}

You can view or download the thumbnail using the provided URL."""

        logger.info("Thumbnail generated successfully")
        return GetPageThumbnailResponse(
            status="success",
            message=confirmation_message,
            thumbnail_url=thumbnail_url,
        ).model_dump_json(indent=2)

    except Exception as error:
        error_msg = f"Error generating thumbnail: {str(error)}"
        logger.error(f"[get_page_thumbnail] {error_msg}")
        return GetPageThumbnailResponse(
            status="error", message="", error=error_msg
        ).model_dump_json(indent=2)


# Create comment management tools for slides
_comment_tools = create_comment_tools("presentation", "presentation_id")
read_presentation_comments = _comment_tools["read_comments"]
create_presentation_comment = _comment_tools["create_comment"]
reply_to_presentation_comment = _comment_tools["reply_to_comment"]
resolve_presentation_comment = _comment_tools["resolve_comment"]

# Aliases for backwards compatibility and intuitive naming
read_slide_comments = read_presentation_comments
create_slide_comment = create_presentation_comment
reply_to_slide_comment = reply_to_presentation_comment
resolve_slide_comment = resolve_presentation_comment
