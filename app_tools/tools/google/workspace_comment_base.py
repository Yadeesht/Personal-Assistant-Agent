"""
Core Comments Module

This module provides reusable comment management functions for Google Workspace applications.
Google Docs and Sheets both use the Drive API for comment operations.
"""

import asyncio
from pathlib import Path


from app_tools.auth.service_decoder import get_google_service
from app_tools.core.server_init import content_server
from utils.helper import setup_logger

logger = setup_logger(__name__)


def get_service():
    """Get Gmail service using shared authentication."""
    base_dir = Path(__file__).parent.parent.parent
    token_path = str(base_dir / "cred" / "gdrive_token.json")
    creds_path = str(base_dir / "cred" / "setup_cred.json")

    return get_google_service(
        service_type="gdrive",
        scope_key="gdrive",
        token_path=token_path,
        creds_path=creds_path,
    )


def create_comment_tools(app_name: str, file_id_param: str):
    """
    Factory function to create comment management tools for a specific Google Workspace app.

    Args:
        app_name: Name of the app ("document" or "spreadsheet")
        file_id_param: Parameter name for the file ID ("document_id" or "spreadsheet_id")

    Returns:
        Dict containing the four comment management functions with unique names
    """

    # Create unique function names based on the app type
    read_func_name = f"read_{app_name}_comments"
    create_func_name = f"create_{app_name}_comment"
    reply_func_name = f"reply_to_{app_name}_comment"
    resolve_func_name = f"resolve_{app_name}_comment"

    # Create functions without decorators first, then apply decorators with proper names
    if file_id_param == "document_id":

        async def read_comments(document_id: str) -> str:
            """Read all comments from a Google Document."""
            service = get_service()
            return await _read_comments_impl(service, app_name, document_id)

        async def create_comment(document_id: str, comment_content: str) -> str:
            """Create a new comment on a Google Document."""
            service = get_service()
            return await _create_comment_impl(
                service, app_name, document_id, comment_content
            )

        async def reply_to_comment(
            document_id: str,
            comment_id: str,
            reply_content: str,
        ) -> str:
            """Reply to a specific comment in a Google Document."""
            service = get_service()
            return await _reply_to_comment_impl(
                service, app_name, document_id, comment_id, reply_content
            )

        async def resolve_comment(document_id: str, comment_id: str) -> str:
            """Resolve a comment in a Google Document."""
            service = get_service()
            return await _resolve_comment_impl(
                service, app_name, document_id, comment_id
            )

    elif file_id_param == "spreadsheet_id":

        async def read_comments(spreadsheet_id: str) -> str:
            """Read all comments from a Google Spreadsheet."""
            service = get_service()
            return await _read_comments_impl(service, app_name, spreadsheet_id)

        async def create_comment(spreadsheet_id: str, comment_content: str) -> str:
            """Create a new comment on a Google Spreadsheet."""
            service = get_service()
            return await _create_comment_impl(
                service, app_name, spreadsheet_id, comment_content
            )

        async def reply_to_comment(
            spreadsheet_id: str,
            comment_id: str,
            reply_content: str,
        ) -> str:
            """Reply to a specific comment in a Google Spreadsheet."""
            service = get_service()
            return await _reply_to_comment_impl(
                service, app_name, spreadsheet_id, comment_id, reply_content
            )

        async def resolve_comment(spreadsheet_id: str, comment_id: str) -> str:
            """Resolve a comment in a Google Spreadsheet."""
            service = get_service()
            return await _resolve_comment_impl(
                service, app_name, spreadsheet_id, comment_id
            )

    # Set the proper function names and register with server
    read_comments.__name__ = read_func_name
    create_comment.__name__ = create_func_name
    reply_to_comment.__name__ = reply_func_name
    resolve_comment.__name__ = resolve_func_name

    # Register tools with the server using the proper names
    content_server.tool()(read_comments)
    content_server.tool()(create_comment)
    content_server.tool()(reply_to_comment)
    content_server.tool()(resolve_comment)

    return {
        "read_comments": read_comments,
        "create_comment": create_comment,
        "reply_to_comment": reply_to_comment,
        "resolve_comment": resolve_comment,
    }


async def _read_comments_impl(service, app_name: str, file_id: str) -> str:
    """Implementation for reading comments from any Google Workspace file."""
    logger.info(f"[read_{app_name}_comments] Reading comments for {app_name} {file_id}")

    response = await asyncio.to_thread(
        service.comments()
        .list(
            fileId=file_id,
            fields="comments(id,content,author,createdTime,modifiedTime,resolved,replies(content,author,id,createdTime,modifiedTime))",
        )
        .execute
    )

    comments = response.get("comments", [])

    if not comments:
        return f"No comments found in {app_name} {file_id}"

    output = [f"Found {len(comments)} comments in {app_name} {file_id}:\n"]

    for comment in comments:
        author = comment.get("author", {}).get("displayName", "Unknown")
        content = comment.get("content", "")
        created = comment.get("createdTime", "")
        resolved = comment.get("resolved", False)
        comment_id = comment.get("id", "")
        status = " [RESOLVED]" if resolved else ""

        output.append(f"Comment ID: {comment_id}")
        output.append(f"Author: {author}")
        output.append(f"Created: {created}{status}")
        output.append(f"Content: {content}")

        # Add replies if any
        replies = comment.get("replies", [])
        if replies:
            output.append(f"  Replies ({len(replies)}):")
            for reply in replies:
                reply_author = reply.get("author", {}).get("displayName", "Unknown")
                reply_content = reply.get("content", "")
                reply_created = reply.get("createdTime", "")
                reply_id = reply.get("id", "")
                output.append(f"    Reply ID: {reply_id}")
                output.append(f"    Author: {reply_author}")
                output.append(f"    Created: {reply_created}")
                output.append(f"    Content: {reply_content}")

        output.append("")  # Empty line between comments

    return "\n".join(output)


async def _create_comment_impl(
    service, app_name: str, file_id: str, comment_content: str
) -> str:
    """Implementation for creating a comment on any Google Workspace file."""
    logger.info(f"[create_{app_name}_comment] Creating comment in {app_name} {file_id}")

    body = {"content": comment_content}

    comment = await asyncio.to_thread(
        service.comments()
        .create(
            fileId=file_id,
            body=body,
            fields="id,content,author,createdTime,modifiedTime",
        )
        .execute
    )

    comment_id = comment.get("id", "")
    author = comment.get("author", {}).get("displayName", "Unknown")
    created = comment.get("createdTime", "")

    return f"Comment created successfully!\nComment ID: {comment_id}\nAuthor: {author}\nCreated: {created}\nContent: {comment_content}"


async def _reply_to_comment_impl(
    service, app_name: str, file_id: str, comment_id: str, reply_content: str
) -> str:
    """Implementation for replying to a comment on any Google Workspace file."""
    logger.info(
        f"[reply_to_{app_name}_comment] Replying to comment {comment_id} in {app_name} {file_id}"
    )

    body = {"content": reply_content}

    reply = await asyncio.to_thread(
        service.replies()
        .create(
            fileId=file_id,
            commentId=comment_id,
            body=body,
            fields="id,content,author,createdTime,modifiedTime",
        )
        .execute
    )

    reply_id = reply.get("id", "")
    author = reply.get("author", {}).get("displayName", "Unknown")
    created = reply.get("createdTime", "")

    return f"Reply posted successfully!\nReply ID: {reply_id}\nAuthor: {author}\nCreated: {created}\nContent: {reply_content}"


async def _resolve_comment_impl(
    service, app_name: str, file_id: str, comment_id: str
) -> str:
    """Implementation for resolving a comment on any Google Workspace file."""
    logger.info(
        f"[resolve_{app_name}_comment] Resolving comment {comment_id} in {app_name} {file_id}"
    )

    body = {"content": "This comment has been resolved.", "action": "resolve"}

    reply = await asyncio.to_thread(
        service.replies()
        .create(
            fileId=file_id,
            commentId=comment_id,
            body=body,
            fields="id,content,author,createdTime,modifiedTime",
        )
        .execute
    )

    reply_id = reply.get("id", "")
    author = reply.get("author", {}).get("displayName", "Unknown")
    created = reply.get("createdTime", "")

    return f"Comment {comment_id} has been resolved successfully.\nResolve reply ID: {reply_id}\nAuthor: {author}\nCreated: {created}"
