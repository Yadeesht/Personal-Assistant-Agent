"""Refactored Gmail API Tools - Best Practices Version"""

import asyncio
import base64
import webbrowser
from datetime import datetime, timedelta
from email import message_from_bytes
from email.header import decode_header
from email.message import EmailMessage
from pathlib import Path
from typing import Any

from googleapiclient.errors import HttpError

from app_tools.auth.service_decoder import get_google_service
from app_tools.core.server_init import communication_server
from utils.helper import clean_email_body

from app_tools.helper.pydantic_models import (
    EmailResponse,
    SendEmailRequest,
    SendEmailResponse,
    UnreadEmailsRequest,
    UnreadEmailsResponse,
    LabelRequest,
    LabelResponse,
    ListLabelsResponse,
    ApplyLabelRequest,
    StandardResponse,
    DraftRequest,
    DraftResponse,
    ListDraftsResponse,
    BatchArchiveRequest,
    BatchArchiveResponse,
    EmailIdRequest,
    ReadEmailResponse,
    ListFiltersResponse,
    FilterIdRequest,
    SearchEmailsRequest,
    SearchEmailsResponse,
    FolderRequest,
    FolderResponse,
    MoveToFolderRequest,
    ListFoldersResponse,
    RenameLabelRequest,
    ListArchivedRequest,
    SearchByLabelRequest,
    SearchByLabelResponse,
)
from utils.helper import setup_logger

logger = setup_logger(__name__)


def get_service():
    """Get Gmail service using shared authentication."""
    base_dir = Path(__file__).parent.parent.parent
    token_path = str(base_dir / "cred" / "gmail_token.json")
    creds_path = str(base_dir / "cred" / "setup_cred.json")

    return get_google_service(
        service_type="gmail",
        scope_key="gmail",
        token_path=token_path,
        creds_path=creds_path,
    )


def decode_mime_header(header: str) -> str:
    """Decode MIME-encoded email headers."""
    decoded_parts = decode_header(header)
    decoded_string = ""
    for part, encoding in decoded_parts:
        if isinstance(part, bytes):
            decoded_string += part.decode(encoding or "utf-8")
        else:
            decoded_string += part
    return decoded_string


async def get_user_email(service) -> str:
    """Get the authenticated user's email address."""
    profile = await asyncio.to_thread(service.users().getProfile(userId="me").execute)
    return profile.get("emailAddress", "")


@communication_server.tool()
async def send_email(recipient_id: str, subject: str, message: str) -> dict[str, Any]:
    """Send an email via Gmail.

    Args:
        recipient_id: Recipient's email address
        subject: Email subject line
        message: Email body content (plain text or HTML)

    Returns:
        Dict with 'success' boolean, 'message_id' if successful or error message
    """
    try:
        request = SendEmailRequest(
            recipient_id=recipient_id, subject=subject, message=message
        )
    except Exception as e:
        return SendEmailResponse(
            success=False, error=f"Invalid parameters: {str(e)}"
        ).model_dump()

    service = get_service()

    try:
        service = get_service()
        user_email = await get_user_email(service)

        message_obj = EmailMessage()
        message_obj.set_content(request.message)
        message_obj["To"] = request.recipient_id
        message_obj["From"] = user_email
        message_obj["Subject"] = request.subject

        encoded_message = base64.urlsafe_b64encode(message_obj.as_bytes()).decode()
        create_message = {"raw": encoded_message}

        send_message = await asyncio.to_thread(
            service.users().messages().send(userId="me", body=create_message).execute
        )

        logger.info(f"Email sent successfully: {send_message['id']}")
        return SendEmailResponse(
            success=True, message_id=send_message["id"]
        ).model_dump()

    except HttpError as error:
        logger.error(f"Gmail API error: {error}")
        return SendEmailResponse(success=False, error=str(error)).model_dump()


@communication_server.tool()
async def open_email(email_id: str) -> dict[str, Any]:
    """Open a specific email in Gmail web interface.

    Args:
        email_id: The unique Gmail message ID

    Returns:
        Dict with 'success' boolean or error message
    """
    try:
        request = EmailIdRequest(email_id=email_id)
    except Exception as e:
        return EmailResponse(
            success=False, error=f"Invalid email ID: {str(e)}"
        ).model_dump()

    try:
        url = f"https://mail.google.com/#all/{request.email_id}"
        webbrowser.open(url, new=0, autoraise=True)
        return EmailResponse(success=True).model_dump()
    except Exception as error:
        logger.error(f"Open email error: {error}")
        return EmailResponse(success=False, error=str(error)).model_dump()


@communication_server.tool()
async def get_unread_emails(date: int = 7, max_results: int = 20) -> dict[str, Any]:
    """Fetch recent unread emails from the primary inbox.

    Args:
        date: Look back this many days (default: 7)
        max_results: Maximum emails to return (default: 20)

    Returns:
        Dict with 'count' and 'emails' list containing email metadata

    Note:
        Only searches the primary inbox category to avoid clutter.
    """

    try:
        request = UnreadEmailsRequest(date=date, max_results=max_results)
    except Exception as e:
        return UnreadEmailsResponse(
            count=0, error=f"Invalid parameters: {str(e)}"
        ).model_dump()

    try:
        service = get_service()
        after_date = (datetime.now() - timedelta(days=request.date)).strftime(
            "%Y/%m/%d"
        )
        query = f"in:inbox is:unread category:primary after:{after_date}"

        response = await asyncio.to_thread(
            service.users()
            .messages()
            .list(userId="me", q=query, maxResults=request.max_results)
            .execute
        )

        messages = []
        if "messages" in response:
            for msg in response["messages"]:
                email_id = msg["id"]

                full_msg = await asyncio.to_thread(
                    service.users()
                    .messages()
                    .get(
                        userId="me",
                        id=email_id,
                        format="metadata",
                        metadataHeaders=["Subject", "From", "Date", "To"],
                    )
                    .execute
                )

                headers = full_msg.get("payload", {}).get("headers", [])

                email_details = {
                    "id": email_id,
                    "thread_id": msg["threadId"],
                    "snippet": clean_email_body(full_msg.get("snippet", "")),
                    # "labels": full_msg.get("labelIds", []),
                    # "size": full_msg.get("sizeEstimate", 0),
                    # "internal_date": full_msg.get("internalDate"),
                    "subject": next(
                        (
                            clean_email_body(h["value"])
                            for h in headers
                            if h["name"] == "Subject"
                        ),
                        "No Subject",
                    ),
                    "from": next(
                        (h["value"] for h in headers if h["name"] == "From"),
                        "Unknown",
                    ),
                    "date": next(
                        (h["value"] for h in headers if h["name"] == "Date"), ""
                    ),
                    "to": next((h["value"] for h in headers if h["name"] == "To"), ""),
                }
                messages.append(email_details)

        return UnreadEmailsResponse(count=len(messages), emails=messages).model_dump()
    except HttpError as error:
        logger.error(f"Failed to fetch unread emails: {error}")
        return EmailResponse(success=False, error=str(error)).model_dump()
    except Exception as error:
        logger.error(f"Unexpected error: {error}")
        return EmailResponse(success=False, error=str(error)).model_dump()


@communication_server.tool()
async def read_email(email_id: str) -> dict[str, Any]:
    """Read the full content of a specific email and mark it as read. Use when user asks for complete details.

    Args:
        email_id: The unique Gmail message ID

    Returns:
        Dict with email content, subject, from, to, date fields or error message

    Note:
        Automatically marks the email as read after fetching.
    """
    try:
        request = EmailIdRequest(email_id=email_id)
    except Exception as e:
        return ReadEmailResponse(
            content="",
            subject="",
            from_="",
            to="",
            date="",
            error=f"Invalid email ID: {str(e)}",
        ).model_dump()

    try:
        service = get_service()
        msg = await asyncio.to_thread(
            service.users()
            .messages()
            .get(userId="me", id=request.email_id, format="raw")
            .execute
        )

        raw_data = msg["raw"]
        decoded_data = base64.urlsafe_b64decode(raw_data)

        mime_message = message_from_bytes(decoded_data)

        body = None
        if mime_message.is_multipart():
            for part in mime_message.walk():
                if part.get_content_type() == "text/plain":
                    body = part.get_payload(decode=True).decode()
                    break
        else:
            body = mime_message.get_payload(decode=True).decode()

        logger.info(f"Email read: {request.email_id}")

        # Mark as read
        await asyncio.to_thread(
            service.users()
            .messages()
            .modify(
                userId="me", id=request.email_id, body={"removeLabelIds": ["UNREAD"]}
            )
            .execute
        )

        return ReadEmailResponse(
            content=clean_email_body(body) or "",
            subject=decode_mime_header(mime_message.get("subject", "")),
            from_=mime_message.get("from", ""),
            to=mime_message.get("to", ""),
            date=mime_message.get("date", ""),
        ).model_dump()
    except HttpError as error:
        logger.error(f"Failed to read email {request.email_id}: {error}")
        return ReadEmailResponse(
            content="", subject="", from_="", to="", date="", error=str(error)
        ).model_dump()
    except Exception as error:
        logger.error(f"Unexpected error: {error}")
        return ReadEmailResponse(
            content="", subject="", from_="", to="", date="", error=str(error)
        ).model_dump()


@communication_server.tool()  # testing done till here date : 15/12/23
async def trash_email(email_id: str) -> dict[str, Any]:
    """Move an email to trash.

    Args:
        email_id: The unique Gmail message ID

    Returns:
        Dict with 'success' boolean or 'error' message

    Warning:
        Always confirm with user before trashing emails.
    """
    try:
        request = EmailIdRequest(email_id=email_id)
    except Exception as e:
        return StandardResponse(
            success=False, error=f"Invalid email ID: {str(e)}"
        ).model_dump()

    try:
        service = get_service()
        await asyncio.to_thread(
            service.users().messages().trash(userId="me", id=request.email_id).execute
        )
        logger.info(f"Email trashed: {request.email_id}")
        return StandardResponse(success=True).model_dump()
    except HttpError as error:
        logger.error(f"Failed to trash email: {error}")
        return StandardResponse(success=False, error=str(error)).model_dump()
    except Exception as error:
        logger.error(f"Unexpected error: {error}")
        return StandardResponse(success=False, error=str(error)).model_dump()


@communication_server.tool()
async def mark_email_as_read(email_id: str) -> dict[str, Any]:
    """Mark a specific email as read to remove the unread status.

    Args:
        email_id: The unique Gmail message ID to mark as read

    Returns:
        Dict with 'success' boolean or 'error' message
    """
    try:
        request = EmailIdRequest(email_id=email_id)
    except Exception as e:
        return StandardResponse(
            success=False, error=f"Invalid email ID: {str(e)}"
        ).model_dump()

    try:
        service = get_service()
        await asyncio.to_thread(
            service.users()
            .messages()
            .modify(
                userId="me", id=request.email_id, body={"removeLabelIds": ["UNREAD"]}
            )
            .execute
        )
        logger.info(f"Email marked as read: {request.email_id}")
        return StandardResponse(success=True).model_dump()
    except HttpError as error:
        logger.error(f"Failed to mark as read: {error}")
        return StandardResponse(success=False, error=str(error)).model_dump()
    except Exception as error:
        logger.error(f"Unexpected error: {error}")
        return StandardResponse(success=False, error=str(error)).model_dump()


@communication_server.tool()
async def create_draft(recipient_id: str, subject: str, message: str) -> dict[str, Any]:
    """Create a draft email without sending it.

    Args:
        recipient_id: Recipient's email address (e.g., user@example.com)
        subject: Email subject line
        message: Email body content (plain text or HTML)

    Returns:
        Dict with 'success' boolean, 'draft_id' if successful or error message

    Note:
        Use this when user wants to draft an email for later review.
    """
    try:
        request = DraftRequest(
            recipient_id=recipient_id, subject=subject, message=message
        )
    except Exception as e:
        return DraftResponse(
            success=False, error=f"Invalid parameters: {str(e)}"
        ).model_dump()

    try:
        service = get_service()
        user_email = await get_user_email(service)

        message_obj = EmailMessage()
        message_obj.set_content(request.message)
        message_obj["To"] = request.recipient_id
        message_obj["From"] = user_email
        message_obj["Subject"] = request.subject

        encoded_message = base64.urlsafe_b64encode(message_obj.as_bytes()).decode()
        create_message = {"raw": encoded_message}

        draft = await asyncio.to_thread(
            service.users()
            .drafts()
            .create(userId="me", body={"message": create_message})
            .execute
        )

        logger.info(f"Draft created: {draft['id']}")
        return DraftResponse(success=True, draft_id=draft["id"]).model_dump()
    except HttpError as error:
        logger.error(f"Failed to create draft: {error}")
        return DraftResponse(success=False, error=str(error)).model_dump()
    except Exception as error:
        logger.error(f"Unexpected error: {error}")
        return DraftResponse(success=False, error=str(error)).model_dump()


@communication_server.tool()
async def list_drafts() -> dict[str, Any]:
    """List all draft emails in the user's mailbox.

    Returns:
        Dict with 'count' and 'drafts' list containing draft metadata
    """
    try:
        service = get_service()
        results = await asyncio.to_thread(
            service.users().drafts().list(userId="me").execute
        )
        drafts = results.get("drafts", [])

        draft_list = []
        for draft in drafts:
            draft_id = draft["id"]
            draft_data = await asyncio.to_thread(
                service.users().drafts().get(userId="me", id=draft_id).execute
            )

            message = draft_data.get("message", {})
            headers = message.get("payload", {}).get("headers", [])

            subject = next(
                (h["value"] for h in headers if h["name"].lower() == "subject"),
                "No Subject",
            )
            to = next(
                (h["value"] for h in headers if h["name"].lower() == "to"),
                "No Recipient",
            )

            draft_list.append({"id": draft_id, "subject": subject, "to": to})

        return ListDraftsResponse(count=len(draft_list), drafts=draft_list).model_dump()
    except HttpError as error:
        logger.error(f"Failed to list drafts: {error}")
        return ListDraftsResponse(count=0, drafts=[], error=str(error)).model_dump()
    except Exception as error:
        logger.error(f"Unexpected error: {error}")
        return ListDraftsResponse(count=0, drafts=[], error=str(error)).model_dump()


@communication_server.tool()
async def list_labels() -> dict[str, Any]:
    """List all labels (tags/folders) in the user's Gmail mailbox.

    Returns:
        Dict with 'count' and 'labels' list containing id, name, and type

    Note:
        Includes both system labels (INBOX, SENT) and user-created labels.
    """
    try:
        service = get_service()
        results = await asyncio.to_thread(
            service.users().labels().list(userId="me").execute
        )
        labels = results.get("labels", [])

        label_list = [
            {
                "id": label["id"],
                "name": label["name"],
                "type": label.get("type", "user"),
            }
            for label in labels
        ]

        return ListLabelsResponse(count=len(label_list), labels=label_list).model_dump()
    except HttpError as error:
        logger.error(f"Failed to list labels: {error}")
        return ListLabelsResponse(count=0, labels=[], error=str(error)).model_dump()
    except Exception as error:
        logger.error(f"Unexpected error: {error}")
        return ListLabelsResponse(count=0, labels=[], error=str(error)).model_dump()


@communication_server.tool()
async def create_label(name: str) -> dict[str, Any]:
    """Create a new label (tag/folder) in Gmail for organizing emails.

    Args:
        name: The name of the new label to create

    Returns:
        Dict with 'success' boolean, 'label_id' and 'name' if successful or error message
    """
    try:
        request = LabelRequest(name=name)
    except Exception as e:
        return LabelResponse(
            success=False, error=f"Invalid label name: {str(e)}"
        ).model_dump()

    try:
        service = get_service()
        label_object = {
            "name": request.name,
            "labelListVisibility": "labelShow",
            "messageListVisibility": "show",
        }

        created_label = await asyncio.to_thread(
            service.users().labels().create(userId="me", body=label_object).execute
        )

        logger.info(f"Label created: {created_label['id']}")
        return LabelResponse(
            success=True, label_id=created_label["id"], name=created_label["name"]
        ).model_dump()
    except HttpError as error:
        logger.error(f"Failed to create label: {error}")
        return LabelResponse(success=False, error=str(error)).model_dump()
    except Exception as error:
        logger.error(f"Unexpected error: {error}")
        return LabelResponse(success=False, error=str(error)).model_dump()


@communication_server.tool()
async def apply_label(email_id: str, label_id: str) -> dict[str, Any]:
    """Apply a label (tag) to a specific email for organization.

    Args:
        email_id: The unique Gmail message ID
        label_id: The label ID to apply (from list_labels_tool)

    Returns:
        Dict with 'success' boolean or 'error' message
    """
    try:
        request = ApplyLabelRequest(email_id=email_id, label_id=label_id)
    except Exception as e:
        return StandardResponse(
            success=False, error=f"Invalid parameters: {str(e)}"
        ).model_dump()

    try:
        service = get_service()
        await asyncio.to_thread(
            service.users()
            .messages()
            .modify(
                userId="me",
                id=request.email_id,
                body={"addLabelIds": [request.label_id]},
            )
            .execute
        )

        logger.info(f"Label {request.label_id} applied to email {request.email_id}")
        return StandardResponse(success=True).model_dump()
    except HttpError as error:
        logger.error(f"Failed to apply label: {error}")
        return StandardResponse(success=False, error=str(error)).model_dump()
    except Exception as error:
        logger.error(f"Unexpected error: {error}")
        return StandardResponse(success=False, error=str(error)).model_dump()


@communication_server.tool()
async def remove_label(email_id: str, label_id: str) -> dict[str, Any]:
    """Remove a label (tag) from a specific email.

    Args:
        email_id: The unique Gmail message ID
        label_id: The label ID to remove (from list_labels_tool)

    Returns:
        Dict with 'success' boolean or 'error' message
    """
    try:
        request = ApplyLabelRequest(email_id=email_id, label_id=label_id)
    except Exception as e:
        return StandardResponse(
            success=False, error=f"Invalid parameters: {str(e)}"
        ).model_dump()

    try:
        service = get_service()
        await asyncio.to_thread(
            service.users()
            .messages()
            .modify(
                userId="me",
                id=request.email_id,
                body={"removeLabelIds": [request.label_id]},
            )
            .execute
        )

        logger.info(f"Label {request.label_id} removed from email {request.email_id}")
        return StandardResponse(success=True).model_dump()
    except HttpError as error:
        logger.error(f"Failed to remove label: {error}")
        return StandardResponse(success=False, error=str(error)).model_dump()
    except Exception as error:
        logger.error(f"Unexpected error: {error}")
        return StandardResponse(success=False, error=str(error)).model_dump()


@communication_server.tool()
async def search_by_label(label_id: str) -> dict[str, Any]:
    """Search for all emails that have a specific label applied.

    Args:
        label_id: The label ID to search for (from list_labels_tool)

    Returns:
        List of message IDs with the label
    """
    try:
        request = SearchByLabelRequest(label_id=label_id)
    except Exception as e:
        return SearchByLabelResponse(
            count=0, messages=[], error=f"Invalid label ID: {str(e)}"
        ).model_dump()

    try:
        service = get_service()
        query = f"label:{request.label_id}"

        response = await asyncio.to_thread(
            service.users().messages().list(userId="me", q=query).execute
        )

        messages = []
        if "messages" in response:
            messages.extend(response["messages"])

        while "nextPageToken" in response:
            page_token = response["nextPageToken"]
            response = await asyncio.to_thread(
                service.users()
                .messages()
                .list(userId="me", q=query, pageToken=page_token)
                .execute
            )
            messages.extend(response["messages"])

        return SearchByLabelResponse(
            count=len(messages), messages=messages
        ).model_dump()
    except HttpError as error:
        logger.error(f"Search by label failed: {error}")
        return SearchByLabelResponse(
            count=0, messages=[], error=str(error)
        ).model_dump()
    except Exception as error:
        logger.error(f"Unexpected error: {error}")
        return SearchByLabelResponse(
            count=0, messages=[], error=str(error)
        ).model_dump()


"""This tool is not working with the Gmail API but there is a way using the google.auth"""
# @communication_server.tool()
# async def create_filter_tool(**kwargs):
#     """
#     name="create-filter"
#     description="Creates a new email filter"
#     schema={
#         "type": "object",
#         "properties": {
#             "from_email": {"type": "string", "description": "Filter emails from this sender"},
#             "to_email": {"type": "string", "description": "Filter emails to this recipient"},
#             "subject": {"type": "string", "description": "Filter emails with this subject"},
#             "query": {"type": "string", "description": "Filter emails matching this query"},
#             "has_attachment": {"type": "boolean", "description": "Filter emails with attachments"},
#             "exclude_chats": {"type": "boolean", "description": "Exclude chats from filter"},
#             "size_comparison": {"type": "string", "description": "Size comparison ('larger' or 'smaller')"},
#             "size": {"type": "integer", "description": "Size in bytes for comparison"},
#             "add_label_ids": {"type": "array", "items": {"type": "string"}, "description": "Labels to add to matching emails"},
#             "remove_label_ids": {"type": "array", "items": {"type": "string"}, "description": "Labels to remove from matching emails"},
#             "forward_to": {"type": "string", "description": "Email address to forward matching emails to"},
#         },
#     }
#     """
#     create_filter_response = await gmail_service.create_filter(**kwargs)
#     return create_filter_response


@communication_server.tool()
async def list_filters() -> dict[str, Any]:
    """List all email filters configured in Gmail.

    Returns:
        Dict with 'filters' list or 'error' message

    Note:
        Filters are rules that automatically organize incoming emails.
    """
    try:
        service = get_service()
        results = await asyncio.to_thread(
            service.users().settings().filters().list(userId="me").execute
        )
        filters = results.get("filters", [])
        return ListFiltersResponse(count=len(filters), filters=filters).model_dump()
    except HttpError as error:
        logger.error(f"Failed to list filters: {error}")
        return ListFiltersResponse(count=0, filters=[], error=str(error)).model_dump()
    except Exception as error:
        logger.error(f"Unexpected error: {error}")
        return ListFiltersResponse(count=0, filters=[], error=str(error)).model_dump()


@communication_server.tool()
async def get_filter(filter_id: str) -> dict[str, Any]:
    """Get detailed information about a specific email filter by its ID.

    Args:
        filter_id: The filter ID to retrieve (from list_filters)
    Returns:
        Filter configuration details or error message
    """
    try:
        request = FilterIdRequest(filter_id=filter_id)
    except Exception as e:
        return {"error": f"Invalid filter ID: {str(e)}"}

    try:
        service = get_service()
        filter_data = await asyncio.to_thread(
            service.users()
            .settings()
            .filters()
            .get(userId="me", id=request.filter_id)
            .execute
        )
        return filter_data
    except HttpError as error:
        logger.error(f"Failed to get filter: {error}")
        return {"error": str(error)}
    except Exception as error:
        logger.error(f"Unexpected error: {error}")
        return {"error": str(error)}


@communication_server.tool()
async def delete_filter_tool(filter_id: str) -> dict[str, Any]:
    """Delete a specific email filter by its ID.

    Args:
        filter_id: The filter ID to delete (from list_filters_tool)

    Returns:
        Dict with 'success' boolean or 'error' message
    """
    try:
        request = FilterIdRequest(filter_id=filter_id)
    except Exception as e:
        return StandardResponse(
            success=False, error=f"Invalid filter ID: {str(e)}"
        ).model_dump()

    try:
        service = get_service()
        await asyncio.to_thread(
            service.users()
            .settings()
            .filters()
            .delete(userId="me", id=request.filter_id)
            .execute
        )
        logger.info(f"Filter deleted: {request.filter_id}")
        return StandardResponse(success=True).model_dump()
    except HttpError as error:
        logger.error(f"Failed to delete filter: {error}")
        return StandardResponse(success=False, error=str(error)).model_dump()
    except Exception as error:
        logger.error(f"Unexpected error: {error}")
        return StandardResponse(success=False, error=str(error)).model_dump()


@communication_server.tool()
async def search_emails(query: str, max_results: int | None = None) -> dict[str, Any]:
    """Search emails using Gmail's query syntax.

    Args:
        query: Gmail search query (supports from:, subject:, after:, has:attachment, etc.)
        max_results: Maximum number of results to return (optional)

    Returns:
        Dict with 'count' and 'emails' list

    Examples:
        - "from:boss@company.com subject:urgent"
        - "has:attachment after:2024/01/01"
        - "is:unread from:newsletter@site.com"

    See: https://support.google.com/mail/answer/7190
    """
    try:
        request = SearchEmailsRequest(query=query, max_results=max_results)
    except Exception as e:
        return SearchEmailsResponse(
            count=0, emails=[], error=f"Invalid parameters: {str(e)}"
        ).model_dump()

    try:
        service = get_service()
        response = await asyncio.to_thread(
            service.users()
            .messages()
            .list(userId="me", q=request.query, maxResults=request.max_results)
            .execute
        )

        messages = []
        if "messages" in response:
            messages.extend(response["messages"])

        # Get metadata for each message
        result_messages = []
        for msg in messages:
            msg_data = await asyncio.to_thread(
                service.users()
                .messages()
                .get(
                    userId="me",
                    id=msg["id"],
                    format="metadata",
                    metadataHeaders=["Subject", "From", "Date"],
                )
                .execute
            )

            headers = msg_data.get("payload", {}).get("headers", [])

            subject = next(
                (h["value"] for h in headers if h["name"].lower() == "subject"),
                "No Subject",
            )
            sender = next(
                (h["value"] for h in headers if h["name"].lower() == "from"),
                "Unknown Sender",
            )
            date = next(
                (h["value"] for h in headers if h["name"].lower() == "date"),
                "",
            )

            result_messages.append(
                {
                    "id": msg["id"],
                    "thread_id": msg["threadId"],
                    "subject": subject,
                    "from": sender,
                    "date": date,
                    "snippet": msg_data.get("snippet", ""),
                }
            )

        return SearchEmailsResponse(
            count=len(result_messages), emails=result_messages
        ).model_dump()
    except HttpError as error:
        logger.error(f"Search failed for query '{request.query}': {error}")
        return SearchEmailsResponse(count=0, emails=[], error=str(error)).model_dump()
    except Exception as error:
        logger.error(f"Unexpected error: {error}")
        return SearchEmailsResponse(count=0, emails=[], error=str(error)).model_dump()


@communication_server.tool()
async def create_folder(name: str) -> dict[str, Any]:
    """Create a new folder in Gmail.

    Args:
        name: The name of the new folder to create

    Returns:
        Dict with 'success' boolean, 'folder_id' and 'name' if successful or error message

    Note:
        In Gmail, folders are implemented as labels with special handling.
    """
    try:
        request = FolderRequest(name=name)
    except Exception as e:
        return FolderResponse(
            success=False, error=f"Invalid folder name: {str(e)}"
        ).model_dump()

    try:
        service = get_service()
        label_object = {
            "name": request.name,
            "labelListVisibility": "labelShow",
            "messageListVisibility": "show",
            "type": "user",
        }

        created_label = await asyncio.to_thread(
            service.users().labels().create(userId="me", body=label_object).execute
        )

        logger.info(f"Folder created: {created_label['id']}")
        return FolderResponse(
            success=True,
            folder_id=created_label["id"],
            name=created_label["name"],
        ).model_dump()
    except HttpError as error:
        logger.error(f"Failed to create folder: {error}")
        return FolderResponse(success=False, error=str(error)).model_dump()
    except Exception as error:
        logger.error(f"Unexpected error: {error}")
        return FolderResponse(success=False, error=str(error)).model_dump()


@communication_server.tool()
async def move_to_folder(email_id: str, folder_id: str) -> dict[str, Any]:
    """Move an email to a specific folder.

    Args:
        email_id: The unique Gmail message ID
        folder_id: The folder ID (label ID) to move the email to

    Returns:
        Dict with 'success' boolean or 'error' message

    Note:
        This applies folder label and removes from inbox.
    """
    try:
        request = MoveToFolderRequest(email_id=email_id, folder_id=folder_id)
    except Exception as e:
        return StandardResponse(
            success=False, error=f"Invalid parameters: {str(e)}"
        ).model_dump()

    try:
        service = get_service()
        await asyncio.to_thread(
            service.users()
            .messages()
            .modify(
                userId="me",
                id=request.email_id,
                body={"addLabelIds": [request.folder_id], "removeLabelIds": ["INBOX"]},
            )
            .execute
        )

        logger.info(f"Email {request.email_id} moved to folder {request.folder_id}")
        return StandardResponse(success=True).model_dump()
    except HttpError as error:
        logger.error(f"Failed to move to folder: {error}")
        return StandardResponse(success=False, error=str(error)).model_dump()
    except Exception as error:
        logger.error(f"Unexpected error: {error}")
        return StandardResponse(success=False, error=str(error)).model_dump()


@communication_server.tool()
async def list_folders() -> dict[str, Any]:
    """List all user-created folders in Gmail.

    Returns:
        Dict with 'count' and 'folders' list containing user-created labels
    """
    try:
        service = get_service()
        results = await asyncio.to_thread(
            service.users().labels().list(userId="me").execute
        )
        labels = results.get("labels", [])

        # Filter to only include user-created labels
        folders = [
            {"id": label["id"], "name": label["name"]}
            for label in labels
            if label.get("type") == "user"
        ]

        return ListFoldersResponse(count=len(folders), folders=folders).model_dump()
    except HttpError as error:
        logger.error(f"Failed to list folders: {error}")
        return ListFoldersResponse(count=0, folders=[], error=str(error)).model_dump()
    except Exception as error:
        logger.error(f"Unexpected error: {error}")
        return ListFoldersResponse(count=0, folders=[], error=str(error)).model_dump()


@communication_server.tool()
async def rename_label(label_id: str, new_name: str) -> dict[str, Any]:
    """Rename an existing label to a new name.

    Args:
        label_id: The label ID to rename (from list_labels_tool)
        new_name: The new name for the label

    Returns:
        Dict with 'success' boolean, 'label_id' and 'name' if successful or error message
    """
    try:
        request = RenameLabelRequest(label_id=label_id, new_name=new_name)
    except Exception as e:
        return LabelResponse(
            success=False, error=f"Invalid parameters: {str(e)}"
        ).model_dump()

    try:
        service = get_service()
        # Get current label
        label = await asyncio.to_thread(
            service.users().labels().get(userId="me", id=request.label_id).execute
        )

        # Update name
        label["name"] = request.new_name

        # Update the label
        updated_label = await asyncio.to_thread(
            service.users()
            .labels()
            .update(userId="me", id=request.label_id, body=label)
            .execute
        )

        logger.info(f"Label renamed: {request.label_id} to {request.new_name}")
        return LabelResponse(
            success=True, label_id=updated_label["id"], name=updated_label["name"]
        ).model_dump()
    except HttpError as error:
        logger.error(f"Failed to rename label: {error}")
        return LabelResponse(success=False, error=str(error)).model_dump()
    except Exception as error:
        logger.error(f"Unexpected error: {error}")
        return LabelResponse(success=False, error=str(error)).model_dump()


@communication_server.tool()
async def delete_label(label_id: str) -> dict[str, Any]:
    """Permanently delete a label from Gmail.

    Args:
        label_id: The label ID to delete (from list_labels)
    Returns:
        Dict with 'success' boolean or 'error' message

    Warning:
        This action cannot be undone.
    """
    try:
        request = LabelRequest(name=label_id)  # Reusing LabelRequest for ID validation
    except Exception as e:
        return StandardResponse(
            success=False, error=f"Invalid label ID: {str(e)}"
        ).model_dump()

    try:
        service = get_service()
        await asyncio.to_thread(
            service.users().labels().delete(userId="me", id=request.label_id).execute
        )

        logger.info(f"Label deleted: {label_id}")
        return StandardResponse(success=True).model_dump()
    except HttpError as error:
        logger.error(f"Failed to delete label: {error}")
        return StandardResponse(success=False, error=str(error)).model_dump()
    except Exception as error:
        logger.error(f"Unexpected error: {error}")
        return StandardResponse(success=False, error=str(error)).model_dump()


@communication_server.tool()
async def archive_email(email_id: str) -> dict[str, Any]:
    """Archive a single email.

    Args:
        email_id: The unique Gmail message ID to archive

    Returns:
        Dict with 'success' boolean or 'error' message

    Note:
        Removes from inbox without deleting, keeps in 'All Mail'.
    """
    try:
        request = EmailIdRequest(email_id=email_id)
    except Exception as e:
        return StandardResponse(
            success=False, error=f"Invalid email ID: {str(e)}"
        ).model_dump()

    try:
        service = get_service()
        await asyncio.to_thread(
            service.users()
            .messages()
            .modify(
                userId="me", id=request.email_id, body={"removeLabelIds": ["INBOX"]}
            )
            .execute
        )

        logger.info(f"Email archived: {request.email_id}")
        return StandardResponse(success=True).model_dump()
    except HttpError as error:
        logger.error(f"Failed to archive email: {error}")
        return StandardResponse(success=False, error=str(error)).model_dump()
    except Exception as error:
        logger.error(f"Unexpected error: {error}")
        return StandardResponse(success=False, error=str(error)).model_dump()


@communication_server.tool()
async def batch_archive(query: str, max_emails: int = 100) -> dict[str, Any]:
    """Archive multiple emails matching a Gmail search query.

    Args:
        query: Gmail search query to find emails to archive (e.g., 'from:sender older_than:30d')
        max_emails: Maximum number of emails to archive (default: 100)

    Returns:
        Dict with 'success' boolean, 'archived_count'

    Note:
        Use Gmail search syntax to specify which emails to archive.
    """
    try:
        request = BatchArchiveRequest(query=query, max_emails=max_emails)
    except Exception as e:
        return BatchArchiveResponse(
            success=False, error=f"Invalid parameters: {str(e)}"
        ).model_dump()

    try:
        service = get_service()
        user_id = "me"

        response = await asyncio.to_thread(
            service.users()
            .messages()
            .list(userId=user_id, q=request.query, maxResults=request.max_emails)
            .execute
        )

        messages = []
        if "messages" in response:
            messages.extend(response["messages"])

        if not messages:
            return BatchArchiveResponse(
                success=True,
                archived_count=0,
                total_found=0,
                message="No emails found matching the query.",
            ).model_dump()

        # Archive each email in the batch
        archived_count = 0
        for msg in messages:
            try:
                await asyncio.to_thread(
                    service.users()
                    .messages()
                    .modify(
                        userId="me",
                        id=msg["id"],
                        body={"removeLabelIds": ["INBOX"]},
                    )
                    .execute
                )
                archived_count += 1
            except Exception as e:
                logger.error(f"Error archiving email {msg['id']}: {e}")

        logger.info(f"Batch archived {archived_count} emails")
        return BatchArchiveResponse(
            success=True, archived_count=archived_count, total_found=len(messages)
        ).model_dump()
    except HttpError as error:
        logger.error(f"Batch archive failed: {error}")
        return BatchArchiveResponse(success=False, error=str(error)).model_dump()
    except Exception as error:
        logger.error(f"Unexpected error: {error}")
        return BatchArchiveResponse(success=False, error=str(error)).model_dump()


@communication_server.tool()
async def list_archived(max_results: int = 100) -> dict[str, Any]:
    """List archived emails.

    Args:
        max_results: Maximum number of archived emails to return (default: 100)

    Returns:
        Dict with 'count' and 'emails' list or 'error' message

    Note:
        Emails not in inbox but still in 'All Mail'.
    """
    try:
        request = ListArchivedRequest(max_results=max_results)
    except Exception as e:
        return SearchEmailsResponse(
            count=0, emails=[], error=f"Invalid parameters: {str(e)}"
        ).model_dump()

    try:
        query = "-in:inbox"
        return await search_emails(query, request.max_results)
    except Exception as error:
        logger.error(f"Failed to list archived emails: {error}")
        return SearchEmailsResponse(count=0, emails=[], error=str(error)).model_dump()


@communication_server.tool()
async def restore_to_inbox(email_id: str) -> dict[str, Any]:
    """Restore an archived email back to the inbox.

    Args:
        email_id: The unique Gmail message ID to restore to inbox

    Returns:
        Dict with 'success' boolean or 'error' message
    """
    try:
        request = EmailIdRequest(email_id=email_id)
    except Exception as e:
        return StandardResponse(
            success=False, error=f"Invalid email ID: {str(e)}"
        ).model_dump()

    try:
        service = get_service()
        await asyncio.to_thread(
            service.users()
            .messages()
            .modify(userId="me", id=request.email_id, body={"addLabelIds": ["INBOX"]})
            .execute
        )

        logger.info(f"Email restored to inbox: {request.email_id}")
        return StandardResponse(success=True).model_dump()
    except HttpError as error:
        logger.error(f"Failed to restore to inbox: {error}")
        return StandardResponse(success=False, error=str(error)).model_dump()
    except Exception as error:
        logger.error(f"Unexpected error: {error}")
        return StandardResponse(success=False, error=str(error)).model_dump()
