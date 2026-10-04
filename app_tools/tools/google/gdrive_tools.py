import asyncio
import io
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse
from urllib.request import url2pathname

import httpx
from app_tools.auth.service_decoder import get_google_service

from app_tools.core.server_init import content_server
from app_tools.helper.utils import extract_office_xml_text
from app_tools.helper.pydantic_models import (
    SearchDriveFilesRequest,
    SearchDriveFilesResponse,
    GetDriveFileContentRequest,
    GetDriveFileContentResponse,
    ListDriveItemsRequest,
    ListDriveItemsResponse,
    CreateDriveFileRequest,
    CreateDriveFileResponse,
    GetDriveFilePermissionsRequest,
    GetDriveFilePermissionsResponse,
    CheckDriveFilePublicAccessRequest,
    CheckDriveFilePublicAccessResponse,
    UpdateDriveFileRequest,
    UpdateDriveFileResponse,
)
from googleapiclient.http import MediaIoBaseDownload, MediaIoBaseUpload
from utils.helper import setup_logger

logger = setup_logger(__name__)


DOWNLOAD_CHUNK_SIZE_BYTES = 256 * 1024  # 256 KB
UPLOAD_CHUNK_SIZE_BYTES = 5 * 1024 * 1024  # 5 MB (Google recommended minimum)


# Auth
def get_service():
    """Get Gmail service using shared auth"""
    base_dir = Path(__file__).parent.parent.parent
    token_path = str(base_dir / "cred" / "gdrive_token.json")
    creds_path = str(base_dir / "cred" / "setup_cred.json")

    return get_google_service(
        service_type="gdrive",
        scope_key="gdrive",
        token_path=token_path,
        creds_path=creds_path,
    )


# helper fucntions for the GDrive


def check_public_link_permission(permissions: List[Dict[str, Any]]) -> bool:
    """
    Check if file has 'anyone with the link' permission.

    Args:
        permissions: List of permission objects from Google Drive API

    Returns:
        bool: True if file has public link sharing enabled
    """
    return any(
        p.get("type") == "anyone" and p.get("role") in ["reader", "writer", "commenter"]
        for p in permissions
    )


def format_public_sharing_error(file_name: str, file_id: str) -> str:
    """
    Format error message for files without public sharing.

    Args:
        file_name: Name of the file
        file_id: Google Drive file ID

    Returns:
        str: Formatted error message
    """
    return (
        f"❌ Permission Error: '{file_name}' not shared publicly. "
        f"Set 'Anyone with the link' → 'Viewer' in Google Drive sharing. "
        f"File: https://drive.google.com/file/d/{file_id}/view"
    )


def get_drive_image_url(file_id: str) -> str:
    """
    Get the correct Drive URL format for publicly shared images.

    Args:
        file_id: Google Drive file ID

    Returns:
        str: URL for embedding Drive images
    """
    return f"https://drive.google.com/uc?export=view&id={file_id}"


# Precompiled regex patterns for Drive query detection
DRIVE_QUERY_PATTERNS = [
    re.compile(r'\b\w+\s*(=|!=|>|<)\s*[\'"].*?[\'"]', re.IGNORECASE),  # field = 'value'
    re.compile(r"\b\w+\s*(=|!=|>|<)\s*\d+", re.IGNORECASE),  # field = number
    re.compile(r"\bcontains\b", re.IGNORECASE),  # contains operator
    re.compile(r"\bin\s+parents\b", re.IGNORECASE),  # in parents
    re.compile(r"\bhas\s*\{", re.IGNORECASE),  # has {properties}
    re.compile(r"\btrashed\s*=\s*(true|false)\b", re.IGNORECASE),  # trashed=true/false
    re.compile(r"\bstarred\s*=\s*(true|false)\b", re.IGNORECASE),  # starred=true/false
    re.compile(
        r'[\'"][^\'"]+[\'"]\s+in\s+parents', re.IGNORECASE
    ),  # 'parentId' in parents
    re.compile(r"\bfullText\s+contains\b", re.IGNORECASE),  # fullText contains
    re.compile(r"\bname\s*(=|contains)\b", re.IGNORECASE),  # name = or name contains
    re.compile(r"\bmimeType\s*(=|!=)\b", re.IGNORECASE),  # mimeType operators
]


def build_drive_list_params(
    query: str,
    page_size: int,
    drive_id: Optional[str] = None,
    include_items_from_all_drives: bool = True,
    corpora: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Helper function to build common list parameters for Drive API calls.

    Args:
        query: The search query string
        page_size: Maximum number of items to return
        drive_id: Optional shared drive ID
        include_items_from_all_drives: Whether to include items from all drives
        corpora: Optional corpus specification

    Returns:
        Dictionary of parameters for Drive API list calls
    """
    list_params = {
        "q": query,
        "pageSize": page_size,
        "fields": "nextPageToken, files(id, name, mimeType, webViewLink, iconLink, modifiedTime, size)",
        "supportsAllDrives": True,
        "includeItemsFromAllDrives": include_items_from_all_drives,
    }

    if drive_id:
        list_params["driveId"] = drive_id
        if corpora:
            list_params["corpora"] = corpora
        else:
            list_params["corpora"] = "drive"
    elif corpora:
        list_params["corpora"] = corpora

    return list_params


SHORTCUT_MIME_TYPE = "application/vnd.google-apps.shortcut"
FOLDER_MIME_TYPE = "application/vnd.google-apps.folder"
BASE_SHORTCUT_FIELDS = (
    "id, mimeType, parents, shortcutDetails(targetId, targetMimeType)"
)


async def resolve_drive_item(
    service,
    file_id: str,
    *,
    extra_fields: Optional[str] = None,
    max_depth: int = 5,
) -> Tuple[str, Dict[str, Any]]:
    """
    Resolve a Drive shortcut so downstream callers operate on the real item.

    Returns the resolved file ID and its metadata. Raises if shortcut targets loop
    or exceed max_depth to avoid infinite recursion.
    """
    current_id = file_id
    depth = 0
    fields = BASE_SHORTCUT_FIELDS
    if extra_fields:
        fields = f"{fields}, {extra_fields}"

    while True:
        metadata = await asyncio.to_thread(
            service.files()
            .get(fileId=current_id, fields=fields, supportsAllDrives=True)
            .execute
        )
        mime_type = metadata.get("mimeType")
        if mime_type != SHORTCUT_MIME_TYPE:
            return current_id, metadata

        shortcut_details = metadata.get("shortcutDetails") or {}
        target_id = shortcut_details.get("targetId")
        if not target_id:
            raise Exception(f"Shortcut '{current_id}' is missing target details.")

        depth += 1
        if depth > max_depth:
            raise Exception(
                f"Shortcut resolution exceeded {max_depth} hops starting from '{file_id}'."
            )
        current_id = target_id


async def resolve_folder_id(
    service,
    folder_id: str,
    *,
    max_depth: int = 5,
) -> str:
    """
    Resolve a folder ID that might be a shortcut and ensure the final target is a folder.
    """
    resolved_id, metadata = await resolve_drive_item(
        service,
        folder_id,
        max_depth=max_depth,
    )
    mime_type = metadata.get("mimeType")
    if mime_type != FOLDER_MIME_TYPE:
        raise Exception(
            f"Resolved ID '{resolved_id}' (from '{folder_id}') is not a folder; mimeType={mime_type}."
        )
    return resolved_id


"""
Google Drive Tools

This module provides tools for interacting with Google Drive API.
"""


@content_server.tool()
async def search_drive_files(
    query: str,
    page_size: int = 10,
    drive_id: Optional[str] = None,
    include_items_from_all_drives: bool = True,
    corpora: Optional[str] = None,
) -> str:
    """
    Searches for files and folders within a user's Google Drive, including shared drives.

    Args:
        query (str): The search query string. Supports Google Drive search operators.
        page_size (int): The maximum number of files to return. Defaults to 10.
        drive_id (Optional[str]): ID of the shared drive to search. If None, behavior depends on `corpora` and `include_items_from_all_drives`.
        include_items_from_all_drives (bool): Whether shared drive items should be included in results. Defaults to True. This is effective when not specifying a `drive_id`.
        corpora (Optional[str]): Bodies of items to query (e.g., 'user', 'domain', 'drive', 'allDrives').
                                 If 'drive_id' is specified and 'corpora' is None, it defaults to 'drive'.
                                 Otherwise, Drive API default behavior applies. Prefer 'user' or 'drive' over 'allDrives' for efficiency.

    Returns:
        str: A formatted list of found files/folders with their details (ID, name, type, size, modified time, link).
    """
    # Validate input parameters
    try:
        request = SearchDriveFilesRequest(
            query=query,
            page_size=page_size,
            drive_id=drive_id,
            include_items_from_all_drives=include_items_from_all_drives,
            corpora=corpora,
        )
    except Exception as e:
        error_msg = f"Invalid parameters: {str(e)}"
        logger.error(f"[search_drive_files] {error_msg}")
        return SearchDriveFilesResponse(
            status="error", message="", error=error_msg
        ).model_dump_json(indent=2)

    service = get_service()

    try:
        # Check if the query looks like a structured Drive query or free text
        # Look for Drive API operators and structured query patterns
        is_structured_query = any(
            pattern.search(request.query) for pattern in DRIVE_QUERY_PATTERNS
        )

        if is_structured_query:
            final_query = request.query
            logger.info(
                f"[search_drive_files] Using structured query as-is: '{final_query}'"
            )
        else:
            # For free text queries, wrap in fullText contains
            escaped_query = request.query.replace("'", "\\'")
            final_query = f"fullText contains '{escaped_query}'"
            logger.info(
                f"[search_drive_files] Reformatting free text query '{request.query}' to '{final_query}'"
            )

        list_params = build_drive_list_params(
            query=final_query,
            page_size=request.page_size,
            drive_id=request.drive_id,
            include_items_from_all_drives=request.include_items_from_all_drives,
            corpora=request.corpora,
        )

        results = await asyncio.to_thread(service.files().list(**list_params).execute)
        files = results.get("files", [])
        if not files:
            message = f"No files found for '{request.query}'."
            return SearchDriveFilesResponse(
                status="success", message=message
            ).model_dump_json(indent=2)

        formatted_files_text_parts = [
            f"Found {len(files)} files matching '{request.query}':"
        ]
        for item in files:
            size_str = f", Size: {item.get('size', 'N/A')}" if "size" in item else ""
            formatted_files_text_parts.append(
                f'- Name: "{item["name"]}" (ID: {item["id"]}, Type: {item["mimeType"]}{size_str}, Modified: {item.get("modifiedTime", "N/A")}) Link: {item.get("webViewLink", "#")}'
            )
        message = "\n".join(formatted_files_text_parts)

        return SearchDriveFilesResponse(
            status="success", message=message
        ).model_dump_json(indent=2)

    except Exception as error:
        error_msg = f"Error searching drive files: {str(error)}"
        logger.error(f"[search_drive_files] {error_msg}")
        return SearchDriveFilesResponse(
            status="error", message="", error=error_msg
        ).model_dump_json(indent=2)


@content_server.tool()
async def get_drive_file_content(
    file_id: str,
) -> str:
    """
    Retrieves the content of a specific Google Drive file by ID, supporting files in shared drives.

    • Native Google Docs, Sheets, Slides → exported as text / CSV.
    • Office files (.docx, .xlsx, .pptx) → unzipped & parsed with std-lib to
      extract readable text.
    • Any other file → downloaded; tries UTF-8 decode, else notes binary.

    Args:
        user_google_email: The user’s Google email address.
        file_id: Drive file ID.

    Returns:
        str: The file content as plain text with metadata header.
    """
    # Validate input parameters
    try:
        request = GetDriveFileContentRequest(file_id=file_id)
    except Exception as e:
        error_msg = f"Invalid parameters: {str(e)}"
        logger.error(f"[get_drive_file_content] {error_msg}")
        return GetDriveFileContentResponse(
            status="error", message="", error=error_msg
        ).model_dump_json(indent=2)

    logger.info(f"[get_drive_file_content] Invoked. File ID: '{request.file_id}'")
    service = get_service()

    try:
        resolved_file_id, file_metadata = await resolve_drive_item(
            service,
            request.file_id,
            extra_fields="name, webViewLink",
        )
        file_id = resolved_file_id
        mime_type = file_metadata.get("mimeType", "")
        file_name = file_metadata.get("name", "Unknown File")
        export_mime_type = {
            "application/vnd.google-apps.document": "text/plain",
            "application/vnd.google-apps.spreadsheet": "text/csv",
            "application/vnd.google-apps.presentation": "text/plain",
        }.get(mime_type)

        request_obj = (
            service.files().export_media(fileId=file_id, mimeType=export_mime_type)
            if export_mime_type
            else service.files().get_media(fileId=file_id)
        )
        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fh, request_obj)
        loop = asyncio.get_event_loop()
        done = False
        while not done:
            status, done = await loop.run_in_executor(None, downloader.next_chunk)

        file_content_bytes = fh.getvalue()

        # Attempt Office XML extraction only for actual Office XML files
        office_mime_types = {
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "application/vnd.openxmlformats-officedocument.presentationml.presentation",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        }

        if mime_type in office_mime_types:
            office_text = extract_office_xml_text(file_content_bytes, mime_type)
            if office_text:
                body_text = office_text
            else:
                # Fallback: try UTF-8; otherwise flag binary
                try:
                    body_text = file_content_bytes.decode("utf-8")
                except UnicodeDecodeError:
                    body_text = (
                        f"[Binary or unsupported text encoding for mimeType '{mime_type}' - "
                        f"{len(file_content_bytes)} bytes]"
                    )
        else:
            # For non-Office files (including Google native files), try UTF-8 decode directly
            try:
                body_text = file_content_bytes.decode("utf-8")
            except UnicodeDecodeError:
                body_text = (
                    f"[Binary or unsupported text encoding for mimeType '{mime_type}' - "
                    f"{len(file_content_bytes)} bytes]"
                )

        # Assemble response
        header = (
            f'File: "{file_name}" (ID: {file_id}, Type: {mime_type})\n'
            f"Link: {file_metadata.get('webViewLink', '#')}\n\n--- CONTENT ---\n"
        )
        message = header + body_text

        return GetDriveFileContentResponse(
            status="success", message=message
        ).model_dump_json(indent=2)

    except Exception as error:
        error_msg = f"Error getting drive file content: {str(error)}"
        logger.error(f"[get_drive_file_content] {error_msg}")
        return GetDriveFileContentResponse(
            status="error", message="", error=error_msg
        ).model_dump_json(indent=2)


@content_server.tool()
async def list_drive_items(
    folder_id: str = "root",
    page_size: int = 100,
    drive_id: Optional[str] = None,
    include_items_from_all_drives: bool = True,
    corpora: Optional[str] = None,
) -> str:
    """
    Lists files and folders, supporting shared drives.
    If `drive_id` is specified, lists items within that shared drive. `folder_id` is then relative to that drive (or use drive_id as folder_id for root).
    If `drive_id` is not specified, lists items from user's "My Drive" and accessible shared drives (if `include_items_from_all_drives` is True).

    Args:
        folder_id (str): The ID of the Google Drive folder. Defaults to 'root'. For a shared drive, this can be the shared drive's ID to list its root, or a folder ID within that shared drive.
        page_size (int): The maximum number of items to return. Defaults to 100.
        drive_id (Optional[str]): ID of the shared drive. If provided, the listing is scoped to this drive.
        include_items_from_all_drives (bool): Whether items from all accessible shared drives should be included if `drive_id` is not set. Defaults to True.
        corpora (Optional[str]): Corpus to query ('user', 'drive', 'allDrives'). If `drive_id` is set and `corpora` is None, 'drive' is used. If None and no `drive_id`, API defaults apply.

    Returns:
        str: A formatted list of files/folders in the specified folder.
    """
    # Validate input parameters
    try:
        request = ListDriveItemsRequest(
            folder_id=folder_id,
            page_size=page_size,
            drive_id=drive_id,
            include_items_from_all_drives=include_items_from_all_drives,
            corpora=corpora,
        )
    except Exception as e:
        error_msg = f"Invalid parameters: {str(e)}"
        logger.error(f"[list_drive_items] {error_msg}")
        return ListDriveItemsResponse(
            status="error", message="", error=error_msg
        ).model_dump_json(indent=2)

    service = get_service()
    logger.info(f"[list_drive_items] Invoked. Folder ID: '{request.folder_id}'")

    try:
        resolved_folder_id = await resolve_folder_id(service, request.folder_id)
        final_query = f"'{resolved_folder_id}' in parents and trashed=false"

        list_params = build_drive_list_params(
            query=final_query,
            page_size=request.page_size,
            drive_id=request.drive_id,
            include_items_from_all_drives=request.include_items_from_all_drives,
            corpora=request.corpora,
        )

        results = await asyncio.to_thread(service.files().list(**list_params).execute)
        files = results.get("files", [])
        if not files:
            message = f"No items found in folder '{request.folder_id}'."
            return ListDriveItemsResponse(
                status="success", message=message
            ).model_dump_json(indent=2)

        formatted_items_text_parts = [
            f"Found {len(files)} items in folder '{request.folder_id}' :"
        ]
        for item in files:
            size_str = f", Size: {item.get('size', 'N/A')}" if "size" in item else ""
            formatted_items_text_parts.append(
                f'- Name: "{item["name"]}" (ID: {item["id"]}, Type: {item["mimeType"]}{size_str}, Modified: {item.get("modifiedTime", "N/A")}) Link: {item.get("webViewLink", "#")}'
            )
        message = "\n".join(formatted_items_text_parts)

        return ListDriveItemsResponse(
            status="success", message=message
        ).model_dump_json(indent=2)

    except Exception as error:
        error_msg = f"Error listing drive items: {str(error)}"
        logger.error(f"[list_drive_items] {error_msg}")
        return ListDriveItemsResponse(
            status="error", message="", error=error_msg
        ).model_dump_json(indent=2)


@content_server.tool()
async def create_drive_file(
    file_name: str,
    content: Optional[str] = None,
    folder_id: str = "root",
    mime_type: str = "text/plain",
    fileUrl: Optional[str] = None,
) -> str:
    """
    Creates a new file in Google Drive, supporting creation within shared drives.
    Accepts either direct content or a fileUrl to fetch the content from.

    Args:
        file_name (str): The name for the new file.
        content (Optional[str]): If provided, the content to write to the file.
        folder_id (str): The ID of the parent folder. Defaults to 'root'.
        mime_type (str): The MIME type of the file. Defaults to 'text/plain'.
        fileUrl (Optional[str]): If provided, fetches the file content from this URL.

    Returns:
        str: Confirmation message of the successful file creation with file link.
    """
    # Validate input parameters
    try:
        request = CreateDriveFileRequest(
            file_name=file_name,
            content=content,
            folder_id=folder_id,
            mime_type=mime_type,
            fileUrl=fileUrl,
        )
    except Exception as e:
        error_msg = f"Invalid parameters: {str(e)}"
        logger.error(f"[create_drive_file] {error_msg}")
        return CreateDriveFileResponse(
            status="error", message="", error=error_msg
        ).model_dump_json(indent=2)

    service = get_service()
    logger.info(
        f"[create_drive_file] Invoked. File: {request.file_name}, URL: {request.fileUrl}"
    )

    try:
        if not request.content and not request.fileUrl:
            raise Exception("You must provide either 'content' or 'fileUrl'.")

        resolved_folder_id = await resolve_folder_id(service, request.folder_id)

        file_metadata = {
            "name": request.file_name,
            "parents": [resolved_folder_id],
            "mimeType": request.mime_type,
        }

        # Handle fileUrl if provided
        if request.fileUrl:
            parsed_url = urlparse(request.fileUrl)

        # Handle file:// URLs
        if parsed_url.scheme == "file":
            logger.info("[create_drive_file] Reading from local filesystem")

            # Convert file:// URL to local path
            raw_path = parsed_url.path or ""
            netloc = parsed_url.netloc
            if netloc and netloc.lower() != "localhost":
                raw_path = f"//{netloc}{raw_path}"
            file_path = url2pathname(raw_path)

            path_obj = Path(file_path)
            if not path_obj.exists():
                raise Exception(f"Local file does not exist: {file_path}")
            if not path_obj.is_file():
                raise Exception(f"Path is not a file: {file_path}")

            # Read and upload
            file_data = await asyncio.to_thread(path_obj.read_bytes)
            logger.info(
                f"[create_drive_file] Read {len(file_data)} bytes from local file"
            )

            media = MediaIoBaseUpload(
                io.BytesIO(file_data),
                mimetype=request.mime_type,
                resumable=True,
                chunksize=UPLOAD_CHUNK_SIZE_BYTES,
            )

            created_file = await asyncio.to_thread(
                service.files()
                .create(
                    body=file_metadata,
                    media_body=media,
                    fields="id, name, webViewLink",
                    supportsAllDrives=True,
                )
                .execute
            )

        # Handle HTTP/HTTPS URLs
        elif parsed_url.scheme in ("http", "https"):
            logger.info(f"[create_drive_file] Fetching from URL: {request.fileUrl}")

            # Always use in-memory approach (works in both modes)
            async with httpx.AsyncClient(follow_redirects=True, timeout=60.0) as client:
                resp = await client.get(request.fileUrl)
                if resp.status_code != 200:
                    raise Exception(
                        f"Failed to fetch file from URL: {request.fileUrl} (status {resp.status_code})"
                    )

                file_data = await resp.aread()
                total_bytes = len(file_data)
                logger.info(
                    f"[create_drive_file] Downloaded {total_bytes} bytes from URL"
                )

                # Try to get MIME type from Content-Type header
                content_type = resp.headers.get("Content-Type")
                if content_type and content_type != "application/octet-stream":
                    request.mime_type = content_type
                    file_metadata["mimeType"] = request.mime_type
                    logger.info(
                        f"[create_drive_file] Using MIME type: {request.mime_type}"
                    )

            # Upload using in-memory buffer
            media = MediaIoBaseUpload(
                io.BytesIO(file_data),
                mimetype=request.mime_type,
                resumable=True,
                chunksize=UPLOAD_CHUNK_SIZE_BYTES,
            )

            logger.info("[create_drive_file] Starting upload to Google Drive...")
            created_file = await asyncio.to_thread(
                service.files()
                .create(
                    body=file_metadata,
                    media_body=media,
                    fields="id, name, webViewLink",
                    supportsAllDrives=True,
                )
                .execute
            )
        else:
            if not parsed_url.scheme:
                raise Exception(
                    "fileUrl is missing a URL scheme. Use file:// (local only), http://, or https://."
                )
            raise Exception(
                f"Unsupported URL scheme '{parsed_url.scheme}'. "
                f"Only file:// (local environment), http://, and https:// are supported."
            )

        # Handle direct content
        if request.content:
            file_data = request.content.encode("utf-8")
            media = MediaIoBaseUpload(
                io.BytesIO(file_data),
                mimetype=request.mime_type,
                resumable=True,
                chunksize=UPLOAD_CHUNK_SIZE_BYTES,
            )

            created_file = await asyncio.to_thread(
                service.files()
                .create(
                    body=file_metadata,
                    media_body=media,
                    fields="id, name, webViewLink",
                    supportsAllDrives=True,
                )
                .execute
            )

        link = created_file.get("webViewLink", "No link available")
        logger.info(f"✅ Successfully created file. Link: {link}")

        message = (
            f"✅ Successfully created file '{created_file.get('name', request.file_name)}' "
            f"(ID: {created_file.get('id', 'N/A')}) in folder '{request.folder_id}' "
            f" Link: {link}"
        )

        return CreateDriveFileResponse(
            status="success", message=message
        ).model_dump_json(indent=2)

    except Exception as error:
        error_msg = f"Error creating drive file: {str(error)}"
        logger.error(f"[create_drive_file] {error_msg}")
        return CreateDriveFileResponse(
            status="error", message="", error=error_msg
        ).model_dump_json(indent=2)


@content_server.tool()
async def get_drive_file_permissions(
    file_id: str,
) -> str:
    """
    Gets detailed metadata about a Google Drive file including sharing permissions.

    Args:
        file_id (str): The ID of the file to check permissions for.

    Returns:
        str: Detailed file metadata including sharing status and URLs.
    """
    # Validate input parameters
    try:
        request = GetDriveFilePermissionsRequest(file_id=file_id)
    except Exception as e:
        error_msg = f"Invalid parameters: {str(e)}"
        logger.error(f"[get_drive_file_permissions] {error_msg}")
        return GetDriveFilePermissionsResponse(
            status="error", message="", error=error_msg
        ).model_dump_json(indent=2)

    service = get_service()

    logger.info(f"[get_drive_file_permissions] Checking file {request.file_id} ")

    try:
        resolved_file_id, _ = await resolve_drive_item(service, request.file_id)
        file_id = resolved_file_id
        # Get comprehensive file metadata including permissions
        file_metadata = await asyncio.to_thread(
            service.files()
            .get(
                fileId=file_id,
                fields="id, name, mimeType, size, modifiedTime, owners, permissions, "
                "webViewLink, webContentLink, shared, sharingUser, viewersCanCopyContent",
                supportsAllDrives=True,
            )
            .execute
        )

        # Format the response
        output_parts = [
            f"File: {file_metadata.get('name', 'Unknown')}",
            f"ID: {file_id}",
            f"Type: {file_metadata.get('mimeType', 'Unknown')}",
            f"Size: {file_metadata.get('size', 'N/A')} bytes",
            f"Modified: {file_metadata.get('modifiedTime', 'N/A')}",
            "",
            "Sharing Status:",
            f"  Shared: {file_metadata.get('shared', False)}",
        ]

        # Add sharing user if available
        sharing_user = file_metadata.get("sharingUser")
        if sharing_user:
            output_parts.append(
                f"  Shared by: {sharing_user.get('displayName', 'Unknown')} ({sharing_user.get('emailAddress', 'Unknown')})"
            )

        # Process permissions
        permissions = file_metadata.get("permissions", [])
        if permissions:
            output_parts.append(f"  Number of permissions: {len(permissions)}")
            output_parts.append("  Permissions:")
            for perm in permissions:
                perm_type = perm.get("type", "unknown")
                role = perm.get("role", "unknown")

                if perm_type == "anyone":
                    output_parts.append(f"    - Anyone with the link ({role})")
                elif perm_type == "user":
                    email = perm.get("emailAddress", "unknown")
                    output_parts.append(f"    - User: {email} ({role})")
                elif perm_type == "domain":
                    domain = perm.get("domain", "unknown")
                    output_parts.append(f"    - Domain: {domain} ({role})")
                elif perm_type == "group":
                    email = perm.get("emailAddress", "unknown")
                    output_parts.append(f"    - Group: {email} ({role})")
                else:
                    output_parts.append(f"    - {perm_type} ({role})")
        else:
            output_parts.append("  No additional permissions (private file)")

        # Add URLs
        output_parts.extend(
            [
                "",
                "URLs:",
                f"  View Link: {file_metadata.get('webViewLink', 'N/A')}",
            ]
        )

        # webContentLink is only available for files that can be downloaded
        web_content_link = file_metadata.get("webContentLink")
        if web_content_link:
            output_parts.append(f"  Direct Download Link: {web_content_link}")

        has_public_link = check_public_link_permission(permissions)

        if has_public_link:
            output_parts.extend(
                [
                    "",
                    "✅ This file is shared with 'Anyone with the link' - it can be inserted into Google Docs",
                ]
            )
        else:
            output_parts.extend(
                [
                    "",
                    "❌ This file is NOT shared with 'Anyone with the link' - it cannot be inserted into Google Docs",
                    "   To fix: Right-click the file in Google Drive → Share → Anyone with the link → Viewer",
                ]
            )

        message = "\n".join(output_parts)

        return GetDriveFilePermissionsResponse(
            status="success", message=message
        ).model_dump_json(indent=2)

    except Exception as error:
        error_msg = f"Error getting file permissions: {str(error)}"
        logger.error(f"[get_drive_file_permissions] {error_msg}")
        return GetDriveFilePermissionsResponse(
            status="error", message="", error=error_msg
        ).model_dump_json(indent=2)


@content_server.tool()
async def check_drive_file_public_access(
    file_name: str,
) -> str:
    """
    Searches for a file by name and checks if it has public link sharing enabled.

    Args:
        file_name (str): The name of the file to check.

    Returns:
        str: Information about the file's sharing status and whether it can be used in Google Docs.
    """
    # Validate input parameters
    try:
        request = CheckDriveFilePublicAccessRequest(file_name=file_name)
    except Exception as e:
        error_msg = f"Invalid parameters: {str(e)}"
        logger.error(f"[check_drive_file_public_access] {error_msg}")
        return CheckDriveFilePublicAccessResponse(
            status="error", message="", error=error_msg
        ).model_dump_json(indent=2)

    service = get_service()

    logger.info(f"[check_drive_file_public_access] Searching for {request.file_name}")

    try:
        # Search for the file
        escaped_name = request.file_name.replace("'", "\\'")
        query = f"name = '{escaped_name}'"

        list_params = {
            "q": query,
            "pageSize": 10,
            "fields": "files(id, name, mimeType, webViewLink)",
            "supportsAllDrives": True,
            "includeItemsFromAllDrives": True,
        }

        results = await asyncio.to_thread(service.files().list(**list_params).execute)

        files = results.get("files", [])
        if not files:
            message = f"No file found with name '{request.file_name}'"
            return CheckDriveFilePublicAccessResponse(
                status="success", message=message
            ).model_dump_json(indent=2)

        if len(files) > 1:
            output_parts = [
                f"Found {len(files)} files with name '{request.file_name}':"
            ]
            for f in files:
                output_parts.append(f"  - {f['name']} (ID: {f['id']})")
            output_parts.append("\nChecking the first file...")
            output_parts.append("")
        else:
            output_parts = []

        # Check permissions for the first file
        file_id = files[0]["id"]
        resolved_file_id, _ = await resolve_drive_item(service, file_id)
        file_id = resolved_file_id

        # Get detailed permissions
        file_metadata = await asyncio.to_thread(
            service.files()
            .get(
                fileId=file_id,
                fields="id, name, mimeType, permissions, webViewLink, webContentLink, shared",
                supportsAllDrives=True,
            )
            .execute
        )

        permissions = file_metadata.get("permissions", [])
        has_public_link = check_public_link_permission(permissions)

        output_parts.extend(
            [
                f"File: {file_metadata['name']}",
                f"ID: {file_id}",
                f"Type: {file_metadata['mimeType']}",
                f"Shared: {file_metadata.get('shared', False)}",
                "",
            ]
        )

        if has_public_link:
            output_parts.extend(
                [
                    "✅ PUBLIC ACCESS ENABLED - This file can be inserted into Google Docs",
                    f"Use with insert_doc_image_url: {get_drive_image_url(file_id)}",
                ]
            )
        else:
            output_parts.extend(
                [
                    "❌ NO PUBLIC ACCESS - Cannot insert into Google Docs",
                    "Fix: Drive → Share → 'Anyone with the link' → 'Viewer'",
                ]
            )

        message = "\n".join(output_parts)

        return CheckDriveFilePublicAccessResponse(
            status="success", message=message
        ).model_dump_json(indent=2)

    except Exception as error:
        error_msg = f"Error checking drive file public access: {str(error)}"
        logger.error(f"[check_drive_file_public_access] {error_msg}")
        return CheckDriveFilePublicAccessResponse(
            status="error", message="", error=error_msg
        ).model_dump_json(indent=2)


@content_server.tool()
async def update_drive_file(
    file_id: str,
    # File metadata updates
    name: Optional[str] = None,
    description: Optional[str] = None,
    mime_type: Optional[str] = None,
    # Folder organization
    add_parents: Optional[str] = None,  # Comma-separated folder IDs to add
    remove_parents: Optional[str] = None,  # Comma-separated folder IDs to remove
    # File status
    starred: Optional[bool] = None,
    trashed: Optional[bool] = None,
    # Sharing and permissions
    writers_can_share: Optional[bool] = None,
    copy_requires_writer_permission: Optional[bool] = None,
    # Custom properties
    properties: Optional[dict] = None,  # User-visible custom properties
) -> str:
    """
    Updates metadata and properties of a Google Drive file.

    Args:
        file_id (str): The ID of the file to update. Required.
        name (Optional[str]): New name for the file.
        description (Optional[str]): New description for the file.
        mime_type (Optional[str]): New MIME type (note: changing type may require content upload).
        add_parents (Optional[str]): Comma-separated folder IDs to add as parents.
        remove_parents (Optional[str]): Comma-separated folder IDs to remove from parents.
        starred (Optional[bool]): Whether to star/unstar the file.
        trashed (Optional[bool]): Whether to move file to/from trash.
        writers_can_share (Optional[bool]): Whether editors can share the file.
        copy_requires_writer_permission (Optional[bool]): Whether copying requires writer permission.
        properties (Optional[dict]): Custom key-value properties for the file.

    Returns:
        str: Confirmation message with details of the updates applied.
    """
    # Validate input parameters
    try:
        request = UpdateDriveFileRequest(
            file_id=file_id,
            name=name,
            description=description,
            mime_type=mime_type,
            add_parents=add_parents,
            remove_parents=remove_parents,
            starred=starred,
            trashed=trashed,
            writers_can_share=writers_can_share,
            copy_requires_writer_permission=copy_requires_writer_permission,
            properties=properties,
        )
    except Exception as e:
        error_msg = f"Invalid parameters: {str(e)}"
        logger.error(f"[update_drive_file] {error_msg}")
        return UpdateDriveFileResponse(
            status="error", message="", error=error_msg
        ).model_dump_json(indent=2)

    service = get_service()

    logger.info(f"[update_drive_file] Updating file {request.file_id}")

    try:
        current_file_fields = (
            "name, description, mimeType, parents, starred, trashed, webViewLink, "
            "writersCanShare, copyRequiresWriterPermission, properties"
        )
        resolved_file_id, current_file = await resolve_drive_item(
            service,
            file_id,
            extra_fields=current_file_fields,
        )
        file_id = resolved_file_id

        # Build the update body with only specified fields
        update_body = {}
        if name is not None:
            update_body["name"] = name
        if description is not None:
            update_body["description"] = description
        if mime_type is not None:
            update_body["mimeType"] = mime_type
        if starred is not None:
            update_body["starred"] = starred
        if trashed is not None:
            update_body["trashed"] = trashed
        if writers_can_share is not None:
            update_body["writersCanShare"] = writers_can_share
        if copy_requires_writer_permission is not None:
            update_body["copyRequiresWriterPermission"] = (
                copy_requires_writer_permission
            )
        if properties is not None:
            update_body["properties"] = properties

        async def _resolve_parent_arguments(parent_arg: Optional[str]) -> Optional[str]:
            if not parent_arg:
                return None
            parent_ids = [
                part.strip() for part in parent_arg.split(",") if part.strip()
            ]
            if not parent_ids:
                return None

            resolved_ids = []
            for parent in parent_ids:
                resolved_parent = await resolve_folder_id(service, parent)
                resolved_ids.append(resolved_parent)
            return ",".join(resolved_ids)

        resolved_add_parents = await _resolve_parent_arguments(add_parents)
        resolved_remove_parents = await _resolve_parent_arguments(remove_parents)

        # Build query parameters for parent changes
        query_params = {
            "fileId": file_id,
            "supportsAllDrives": True,
            "fields": "id, name, description, mimeType, parents, starred, trashed, webViewLink, writersCanShare, copyRequiresWriterPermission, properties",
        }

        if resolved_add_parents:
            query_params["addParents"] = resolved_add_parents
        if resolved_remove_parents:
            query_params["removeParents"] = resolved_remove_parents

        # Only include body if there are updates
        if update_body:
            query_params["body"] = update_body

        # Perform the update
        updated_file = await asyncio.to_thread(
            service.files().update(**query_params).execute
        )

        # Build response message
        output_parts = [
            f"✅ Successfully updated file: {updated_file.get('name', current_file['name'])}"
        ]
        output_parts.append(f"   File ID: {file_id}")

        # Report what changed
        changes = []
        if name is not None and name != current_file.get("name"):
            changes.append(f"   • Name: '{current_file.get('name')}' → '{name}'")
        if description is not None:
            old_desc_value = current_file.get("description")
            new_desc_value = description
            should_report_change = (old_desc_value or "") != (new_desc_value or "")
            if should_report_change:
                old_desc_display = (
                    old_desc_value if old_desc_value not in (None, "") else "(empty)"
                )
                new_desc_display = (
                    new_desc_value if new_desc_value not in (None, "") else "(empty)"
                )
                changes.append(
                    f"   • Description: {old_desc_display} → {new_desc_display}"
                )
        if add_parents:
            changes.append(f"   • Added to folder(s): {add_parents}")
        if remove_parents:
            changes.append(f"   • Removed from folder(s): {remove_parents}")
        current_starred = current_file.get("starred")
        if starred is not None and starred != current_starred:
            star_status = "starred" if starred else "unstarred"
            changes.append(f"   • File {star_status}")
        current_trashed = current_file.get("trashed")
        if trashed is not None and trashed != current_trashed:
            trash_status = "moved to trash" if trashed else "restored from trash"
            changes.append(f"   • File {trash_status}")
        current_writers_can_share = current_file.get("writersCanShare")
        if (
            writers_can_share is not None
            and writers_can_share != current_writers_can_share
        ):
            share_status = "can" if writers_can_share else "cannot"
            changes.append(f"   • Writers {share_status} share the file")
        current_copy_requires_writer_permission = current_file.get(
            "copyRequiresWriterPermission"
        )
        if (
            copy_requires_writer_permission is not None
            and copy_requires_writer_permission
            != current_copy_requires_writer_permission
        ):
            copy_status = (
                "requires" if copy_requires_writer_permission else "doesn't require"
            )
            changes.append(f"   • Copying {copy_status} writer permission")
        if properties:
            changes.append(f"   • Updated custom properties: {properties}")

        if changes:
            output_parts.append("")
            output_parts.append("Changes applied:")
            output_parts.extend(changes)
        else:
            output_parts.append("   (No changes were made)")

        output_parts.append("")
        output_parts.append(f"View file: {updated_file.get('webViewLink', '#')}")

        return "\n".join(output_parts)

    except Exception as e:
        error_msg = f"Error updating drive file: {str(e)}"
        logger.error(f"[update_drive_file] {error_msg}")
        return UpdateDriveFileResponse(
            status="error", message="", error=error_msg
        ).model_dump_json(indent=2)
