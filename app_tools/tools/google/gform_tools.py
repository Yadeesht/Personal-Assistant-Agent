"""
Google Forms Tools

This module provides tools for interacting with Google Forms API.
"""

import asyncio
from typing import Optional, Dict, Any
from pathlib import Path

from app_tools.auth.service_decoder import get_google_service
from app_tools.core.server_init import content_server
from app_tools.helper.pydantic_models import (
    CreateFormRequest,
    CreateFormResponse,
    GetFormRequest,
    GetFormResponse,
    SetPublishSettingsRequest,
    SetPublishSettingsResponse,
    GetFormResponseRequest,
    GetFormResponseResponse,
    ListFormResponsesRequest,
    ListFormResponsesResponse,
)
from utils.helper import setup_logger

logger = setup_logger(__name__)


def get_service():
    """Get Gmail service using shared authentication."""
    base_dir = Path(__file__).parent.parent.parent
    token_path = str(base_dir / "cred" / "gform_token.json")
    creds_path = str(base_dir / "cred" / "setup_cred.json")

    return get_google_service(
        service_type="forms",
        scope_key="forms",
        token_path=token_path,
        creds_path=creds_path,
    )


@content_server.tool()
async def create_form(
    title: str,
    description: Optional[str] = None,
    document_title: Optional[str] = None,
) -> str:
    """
    Create a new form using the title given in the provided form message in the request.

    Args:
        title (str): The title of the form.
        description (Optional[str]): The description of the form.
        document_title (Optional[str]): The document title (shown in browser tab).

    Returns:
        str: Confirmation message with form ID and edit URL.
    """
    # Validate input parameters
    try:
        request = CreateFormRequest(
            title=title,
            description=description,
            document_title=document_title,
        )
    except Exception as e:
        error_msg = f"Invalid parameters: {str(e)}"
        logger.error(f"[create_form] {error_msg}")
        return CreateFormResponse(
            status="error", message="", error=error_msg
        ).model_dump_json(indent=2)

    service = get_service()
    logger.info(f"[create_form] Invoked. Title: {request.title}")

    try:
        form_body: Dict[str, Any] = {"info": {"title": request.title}}

        if request.description:
            form_body["info"]["description"] = request.description

        if request.document_title:
            form_body["info"]["document_title"] = request.document_title

        created_form = await asyncio.to_thread(
            service.forms().create(body=form_body).execute
        )

        form_id = created_form.get("formId")
        edit_url = f"https://docs.google.com/forms/d/{form_id}/edit"
        responder_url = created_form.get(
            "responderUri", f"https://docs.google.com/forms/d/{form_id}/viewform"
        )

        message = f"Successfully created form '{created_form.get('info', {}).get('title', request.title)}'. Form ID: {form_id}. Edit URL: {edit_url}. Responder URL: {responder_url}"
        logger.info(f"Form created successfully. ID: {form_id}")

        return CreateFormResponse(
            status="success",
            message=message,
            form_id=form_id,
            edit_url=edit_url,
            responder_url=responder_url,
        ).model_dump_json(indent=2)

    except Exception as error:
        error_msg = f"Error creating form: {str(error)}"
        logger.error(f"[create_form] {error_msg}")
        return CreateFormResponse(
            status="error", message="", error=error_msg
        ).model_dump_json(indent=2)


@content_server.tool()
async def get_form(user_google_email: str, form_id: str) -> str:
    """
    Get a form.

    Args:
        form_id (str): The ID of the form to retrieve.

    Returns:
        str: Form details including title, description, questions, and URLs.
    """
    # Validate input parameters
    try:
        request = GetFormRequest(
            user_google_email=user_google_email,
            form_id=form_id,
        )
    except Exception as e:
        error_msg = f"Invalid parameters: {str(e)}"
        logger.error(f"[get_form] {error_msg}")
        return GetFormResponse(
            status="error", message="", error=error_msg
        ).model_dump_json(indent=2)

    logger.info(f"[get_form] Invoked. Form ID: {request.form_id}")

    try:
        service = get_service()
        form = await asyncio.to_thread(
            service.forms().get(formId=request.form_id).execute
        )

        form_info = form.get("info", {})
        title = form_info.get("title", "No Title")
        description = form_info.get("description", "No Description")
        document_title = form_info.get("documentTitle", title)

        edit_url = f"https://docs.google.com/forms/d/{request.form_id}/edit"
        responder_url = form.get(
            "responderUri",
            f"https://docs.google.com/forms/d/{request.form_id}/viewform",
        )

        items = form.get("items", [])
        questions_summary = []
        for i, item in enumerate(items, 1):
            item_title = item.get("title", f"Question {i}")
            item_type = (
                item.get("questionItem", {}).get("question", {}).get("required", False)
            )
            required_text = " (Required)" if item_type else ""
            questions_summary.append(f"  {i}. {item_title}{required_text}")

        questions_text = (
            "\n".join(questions_summary)
            if questions_summary
            else "  No questions found"
        )

        message = f"""Form Details:
- Title: "{title}"
- Description: "{description}"
- Document Title: "{document_title}"
- Form ID: {request.form_id}
- Edit URL: {edit_url}
- Responder URL: {responder_url}
- Questions ({len(items)} total):
{questions_text}"""

        logger.info(f"Successfully retrieved form. ID: {request.form_id}")
        return GetFormResponse(status="success", message=message).model_dump_json(
            indent=2
        )

    except Exception as error:
        error_msg = f"Error getting form: {str(error)}"
        logger.error(f"[get_form] {error_msg}")
        return GetFormResponse(
            status="error", message="", error=error_msg
        ).model_dump_json(indent=2)


@content_server.tool()
async def set_publish_settings(
    form_id: str,
    publish_as_template: bool = False,
    require_authentication: bool = False,
) -> str:
    """
    Updates the publish settings of a form.

    Args:
        form_id (str): The ID of the form to update publish settings for.
        publish_as_template (bool): Whether to publish as a template. Defaults to False.
        require_authentication (bool): Whether to require authentication to view/submit. Defaults to False.

    Returns:
        str: Confirmation message of the successful publish settings update.
    """
    # Validate input parameters
    try:
        request = SetPublishSettingsRequest(
            form_id=form_id,
            publish_as_template=publish_as_template,
            require_authentication=require_authentication,
        )
    except Exception as e:
        error_msg = f"Invalid parameters: {str(e)}"
        logger.error(f"[set_publish_settings] {error_msg}")
        return SetPublishSettingsResponse(
            status="error", message="", error=error_msg
        ).model_dump_json(indent=2)

    service = get_service()
    logger.info(f"[set_publish_settings] Invoked. Form ID: {request.form_id}")

    try:
        settings_body = {
            "publishAsTemplate": request.publish_as_template,
            "requireAuthentication": request.require_authentication,
        }

        await asyncio.to_thread(
            service.forms()
            .setPublishSettings(formId=request.form_id, body=settings_body)
            .execute
        )

        message = f"Successfully updated publish settings for form {request.form_id}. Publish as template: {request.publish_as_template}, Require authentication: {request.require_authentication}"
        logger.info(
            f"Publish settings updated successfully. Form ID: {request.form_id}"
        )

        return SetPublishSettingsResponse(
            status="success", message=message
        ).model_dump_json(indent=2)

    except Exception as error:
        error_msg = f"Error setting publish settings: {str(error)}"
        logger.error(f"[set_publish_settings] {error_msg}")
        return SetPublishSettingsResponse(
            status="error", message="", error=error_msg
        ).model_dump_json(indent=2)


@content_server.tool()
async def get_form_response(form_id: str, response_id: str) -> str:
    """
    Get one response from the form.

    Args:
        form_id (str): The ID of the form.
        response_id (str): The ID of the response to retrieve.

    Returns:
        str: Response details including answers and metadata.
    """
    # Validate input parameters
    try:
        request = GetFormResponseRequest(
            form_id=form_id,
            response_id=response_id,
        )
    except Exception as e:
        error_msg = f"Invalid parameters: {str(e)}"
        logger.error(f"[get_form_response] {error_msg}")
        return GetFormResponseResponse(
            status="error", message="", error=error_msg
        ).model_dump_json(indent=2)

    service = get_service()
    logger.info(
        f"[get_form_response] Invoked. Form ID: {request.form_id}, Response ID: {request.response_id}"
    )

    try:
        response = await asyncio.to_thread(
            service.forms()
            .responses()
            .get(formId=request.form_id, responseId=request.response_id)
            .execute
        )

        response_id = response.get("responseId", "Unknown")
        create_time = response.get("createTime", "Unknown")
        last_submitted_time = response.get("lastSubmittedTime", "Unknown")

        answers = response.get("answers", {})
        answer_details = []
        for question_id, answer_data in answers.items():
            question_response = answer_data.get("textAnswers", {}).get("answers", [])
            if question_response:
                answer_text = ", ".join(
                    [ans.get("value", "") for ans in question_response]
                )
                answer_details.append(f"  Question ID {question_id}: {answer_text}")
            else:
                answer_details.append(
                    f"  Question ID {question_id}: No answer provided"
                )

        answers_text = (
            "\n".join(answer_details) if answer_details else "  No answers found"
        )

        message = f"""Form Response Details:
- Form ID: {request.form_id}
- Response ID: {response_id}
- Created: {create_time}
- Last Submitted: {last_submitted_time}
- Answers:
{answers_text}"""

        logger.info(
            f"Successfully retrieved response. Response ID: {request.response_id}"
        )
        return GetFormResponseResponse(
            status="success", message=message
        ).model_dump_json(indent=2)

    except Exception as error:
        error_msg = f"Error getting form response: {str(error)}"
        logger.error(f"[get_form_response] {error_msg}")
        return GetFormResponseResponse(
            status="error", message="", error=error_msg
        ).model_dump_json(indent=2)


@content_server.tool()
async def list_form_responses(
    form_id: str,
    page_size: int = 10,
    page_token: Optional[str] = None,
) -> str:
    """
    List a form's responses.

    Args:
        form_id (str): The ID of the form.
        page_size (int): Maximum number of responses to return. Defaults to 10.
        page_token (Optional[str]): Token for retrieving next page of results.

    Returns:
        str: List of responses with basic details and pagination info.
    """
    # Validate input parameters
    try:
        request = ListFormResponsesRequest(
            form_id=form_id,
            page_size=page_size,
            page_token=page_token,
        )
    except Exception as e:
        error_msg = f"Invalid parameters: {str(e)}"
        logger.error(f"[list_form_responses] {error_msg}")
        return ListFormResponsesResponse(
            status="error", message="", error=error_msg
        ).model_dump_json(indent=2)

    service = get_service()
    logger.info(f"[list_form_responses] Invoked. Form ID: {request.form_id}")

    try:
        params = {"formId": request.form_id, "pageSize": request.page_size}
        if request.page_token:
            params["pageToken"] = request.page_token

        responses_result = await asyncio.to_thread(
            service.forms().responses().list(**params).execute
        )

        responses = responses_result.get("responses", [])
        next_page_token = responses_result.get("nextPageToken")

        if not responses:
            message = f"No responses found for form {request.form_id}."
            return ListFormResponsesResponse(
                status="success", message=message
            ).model_dump_json(indent=2)

        response_details = []
        for i, response in enumerate(responses, 1):
            response_id = response.get("responseId", "Unknown")
            create_time = response.get("createTime", "Unknown")
            last_submitted_time = response.get("lastSubmittedTime", "Unknown")

            answers_count = len(response.get("answers", {}))
            response_details.append(
                f"  {i}. Response ID: {response_id} | Created: {create_time} | Last Submitted: {last_submitted_time} | Answers: {answers_count}"
            )

        pagination_info = (
            f"\nNext page token: {next_page_token}"
            if next_page_token
            else "\nNo more pages."
        )

        message = f"""Form Responses:
- Form ID: {request.form_id}
- Total responses returned: {len(responses)}
- Responses:
{chr(10).join(response_details)}{pagination_info}"""

        logger.info(
            f"Successfully retrieved {len(responses)} responses. Form ID: {request.form_id}"
        )
        return ListFormResponsesResponse(
            status="success", message=message
        ).model_dump_json(indent=2)

    except Exception as error:
        error_msg = f"Error listing form responses: {str(error)}"
        logger.error(f"[list_form_responses] {error_msg}")
        return ListFormResponsesResponse(
            status="error", message="", error=error_msg
        ).model_dump_json(indent=2)
