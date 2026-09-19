"""
Google Docs Tools

This module provides tools for interacting with Google Docs API and managing Google Docs via Drive.
"""

import asyncio
import io
from typing import List, Dict, Any
from pathlib import Path

from googleapiclient.http import MediaIoBaseDownload, MediaIoBaseUpload

# Auth & server utilities
from app_tools.auth.service_decoder import get_google_service
from app_tools.helper.utils import extract_office_xml_text
from app_tools.core.server_init import content_server
from app_tools.tools.google.workspace_comment_base import create_comment_tools

# Import Pydantic models
from app_tools.helper.pydantic_models import (
    SearchDocsRequest,
    SearchDocsResponse,
    DocInfo,
    GetDocContentRequest,
    GetDocContentResponse,
    ListDocsInFolderRequest,
    ListDocsInFolderResponse,
    CreateDocRequest,
    CreateDocResponse,
    ModifyDocTextRequest,
    ModifyDocTextResponse,
    FindAndReplaceDocRequest,
    FindAndReplaceDocResponse,
    InsertDocElementsRequest,
    InsertDocElementsResponse,
    InsertDocImageRequest,
    InsertDocImageResponse,
    UpdateDocHeadersFootersRequest,
    UpdateDocHeadersFootersResponse,
    BatchUpdateDocRequest,
    BatchUpdateDocResponse,
    InspectDocStructureRequest,
    InspectDocStructureResponse,
    CreateTableWithDataRequest,
    CreateTableWithDataResponse,
    DebugTableStructureRequest,
    DebugTableStructureResponse,
    ExportDocToPdfRequest,
    ExportDocToPdfResponse,
)

# Import helper functions for document operations
from app_tools.helper.docs_helper import (
    create_insert_text_request,
    create_delete_range_request,
    create_format_text_request,
    create_find_replace_request,
    create_insert_table_request,
    create_insert_page_break_request,
    create_insert_image_request,
    create_bullet_list_request,
    parse_document_structure,
    find_tables,
    analyze_document_complexity,
    extract_table_as_data,
)

# Import operation managers for complex business logic
from app_tools.helper.docs_managers import (
    TableOperationManager,
    HeaderFooterManager,
    ValidationManager,
    BatchOperationManager,
)
from utils.helper import setup_logger

logger = setup_logger(__name__)


def get_service():
    """Get Gmail service using shared authentication."""
    base_dir = Path(__file__).parent.parent.parent
    token_path = str(base_dir / "cred" / "gdocs_token.json")
    creds_path = str(base_dir / "cred" / "setup_cred.json")

    return get_google_service(
        service_type="docs",
        scope_key="docs",
        token_path=token_path,
        creds_path=creds_path,
    )


def drive_get_service():
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


@content_server.tool()
async def search_docs(
    query: str,
    page_size: int = 10,
) -> dict:
    """
    Searches for Google Docs by name using Drive API (mimeType filter).

    Args:
        query (str): The search query string. Required.
        page_size (int): The number of results to return. Optional.

    Returns:
        dict: A dictionary containing count, docs list, and query.
    """
    try:
        request = SearchDocsRequest(query=query, page_size=page_size)
    except Exception as e:
        return SearchDocsResponse(
            count=0,
            docs=[],
            query=query,
            error=f"Invalid parameters: {str(e)}",
        ).model_dump()

    try:
        service = get_service()
        logger.info(f"[search_docs] Query='{query}'")

        escaped_query = request.query.replace("'", "\\'")

        response = await asyncio.to_thread(
            service.files()
            .list(
                q=f"name contains '{escaped_query}' and mimeType='application/vnd.google-apps.document' and trashed=false",
                pageSize=request.page_size,
                fields="files(id, name, createdTime, modifiedTime, webViewLink)",
                supportsAllDrives=True,
                includeItemsFromAllDrives=True,
            )
            .execute
        )
        files = response.get("files", [])
        if not files:
            return SearchDocsResponse(
                count=0, docs=[], query=request.query
            ).model_dump()

        doc_list = [
            DocInfo(
                id=f.get("id", ""),
                name=f.get("name", "Unnamed Document"),
                created_time=f.get("createdTime", ""),
                modified_time=f.get("modifiedTime", ""),
                web_view_link=f.get("webViewLink", ""),
            )
            for f in files
        ]

        logger.info(f"Found {len(doc_list)} docs matching '{request.query}'")
        return SearchDocsResponse(
            count=len(doc_list), docs=doc_list, query=request.query
        ).model_dump()

    except Exception as error:
        logger.error(f"Failed to search docs: {error}")
        return SearchDocsResponse(
            count=0, docs=[], query=query, error=str(error)
        ).model_dump()


@content_server.tool()
async def get_doc_content(
    document_id: str,
) -> dict:
    """
    Retrieves content of a Google Doc or a Drive file (like .docx) identified by document_id.
    - Native Google Docs: Fetches content via Docs API.
    - Office files (.docx, etc.) stored in Drive: Downloads via Drive API and extracts text.

    Args:
        document_id: ID of the Google Doc or Drive file to retrieve content from. Required.

    Returns:
        dict: A dictionary containing document metadata and content.
    """
    try:
        request = GetDocContentRequest(document_id=document_id)
    except Exception as e:
        return GetDocContentResponse(
            document_id=document_id,
            name="",
            mime_type="",
            content="",
            web_view_link="",
            error=f"Invalid parameters: {str(e)}",
        ).model_dump()

    try:
        drive_service = drive_get_service()
        docs_service = get_service()
        logger.info(
            f"[get_doc_content] Invoked. Document/File ID: '{request.document_id}'"
        )

        # Step 2: Get file metadata from Drive
        file_metadata = await asyncio.to_thread(
            drive_service.files()
            .get(
                fileId=request.document_id,
                fields="id, name, mimeType, webViewLink",
                supportsAllDrives=True,
            )
            .execute
        )
        mime_type = file_metadata.get("mimeType", "")
        file_name = file_metadata.get("name", "Unknown File")
        web_view_link = file_metadata.get("webViewLink", "#")

        logger.info(
            f"[get_doc_content] File '{file_name}' (ID: {request.document_id}) has mimeType: '{mime_type}'"
        )

        body_text = ""  # Initialize body_text

        # Step 3: Process based on mimeType
        if mime_type == "application/vnd.google-apps.document":
            logger.info("[get_doc_content] Processing as native Google Doc.")
            doc_data = await asyncio.to_thread(
                docs_service.documents()
                .get(documentId=request.document_id, includeTabsContent=True)
                .execute
            )
            # Tab header format constant
            TAB_HEADER_FORMAT = "\n--- TAB: {tab_name} ---\n"

            def extract_text_from_elements(elements, tab_name=None, depth=0):
                """Extract text from document elements (paragraphs, tables, etc.)"""
                # Prevent infinite recursion by limiting depth
                if depth > 5:
                    return ""
                text_lines = []
                if tab_name:
                    text_lines.append(TAB_HEADER_FORMAT.format(tab_name=tab_name))

                for element in elements:
                    if "paragraph" in element:
                        paragraph = element.get("paragraph", {})
                        para_elements = paragraph.get("elements", [])
                        current_line_text = ""
                        for pe in para_elements:
                            text_run = pe.get("textRun", {})
                            if text_run and "content" in text_run:
                                current_line_text += text_run["content"]
                        if current_line_text.strip():
                            text_lines.append(current_line_text)
                    elif "table" in element:
                        # Handle table content
                        table = element.get("table", {})
                        table_rows = table.get("tableRows", [])
                        for row in table_rows:
                            row_cells = row.get("tableCells", [])
                            for cell in row_cells:
                                cell_content = cell.get("content", [])
                                cell_text = extract_text_from_elements(
                                    cell_content, depth=depth + 1
                                )
                                if cell_text.strip():
                                    text_lines.append(cell_text)
                return "".join(text_lines)

            def process_tab_hierarchy(tab, level=0):
                """Process a tab and its nested child tabs recursively"""
                tab_text = ""

                if "documentTab" in tab:
                    props = tab.get("tabProperties", {})
                    tab_title = props.get("title", "Untitled Tab")
                    tab_id = props.get("tabId", "Unknown ID")
                    # Add indentation for nested tabs to show hierarchy
                    if level > 0:
                        tab_title = "    " * level + f"{tab_title} ( ID: {tab_id})"
                    tab_body = (
                        tab.get("documentTab", {}).get("body", {}).get("content", [])
                    )
                    tab_text += extract_text_from_elements(tab_body, tab_title)

                # Process child tabs (nested tabs)
                child_tabs = tab.get("childTabs", [])
                for child_tab in child_tabs:
                    tab_text += process_tab_hierarchy(child_tab, level + 1)

                return tab_text

            processed_text_lines = []

            # Process main document body
            body_elements = doc_data.get("body", {}).get("content", [])
            main_content = extract_text_from_elements(body_elements)
            if main_content.strip():
                processed_text_lines.append(main_content)

            # Process all tabs
            tabs = doc_data.get("tabs", [])
            for tab in tabs:
                tab_content = process_tab_hierarchy(tab)
                if tab_content.strip():
                    processed_text_lines.append(tab_content)

            body_text = "".join(processed_text_lines)
        else:
            logger.info(
                f"[get_doc_content] Processing as Drive file (e.g., .docx, other). MimeType: {mime_type}"
            )

            export_mime_type_map = {
                # Example: "application/vnd.google-apps.spreadsheet"z: "text/csv",
                # Native GSuite types that are not Docs would go here if this function
                # was intended to export them. For .docx, direct download is used.
            }
            effective_export_mime = export_mime_type_map.get(mime_type)

            request_obj = (
                drive_service.files().export_media(
                    fileId=document_id,
                    mimeType=effective_export_mime,
                    supportsAllDrives=True,
                )
                if effective_export_mime
                else drive_service.files().get_media(
                    fileId=document_id, supportsAllDrives=True
                )
            )

            fh = io.BytesIO()
            downloader = MediaIoBaseDownload(fh, request_obj)
            loop = asyncio.get_event_loop()
            done = False
            while not done:
                status, done = await loop.run_in_executor(None, downloader.next_chunk)

            file_content_bytes = fh.getvalue()

            office_text = extract_office_xml_text(file_content_bytes, mime_type)
            if office_text:
                body_text = office_text
            else:
                try:
                    body_text = file_content_bytes.decode("utf-8")
                except UnicodeDecodeError:
                    body_text = (
                        f"[Binary or unsupported text encoding for mimeType '{mime_type}' - "
                        f"{len(file_content_bytes)} bytes]"
                    )

        logger.info(f"Successfully retrieved content from {document_id}")
        return GetDocContentResponse(
            document_id=document_id,
            name=file_name,
            mime_type=mime_type,
            content=body_text,
            web_view_link=web_view_link,
        ).model_dump()

    except Exception as error:
        logger.error(f"Failed to get doc content for {document_id}: {error}")
        return GetDocContentResponse(
            document_id=document_id,
            name="",
            mime_type="",
            content="",
            web_view_link="",
            error=str(error),
        ).model_dump()


@content_server.tool()
async def list_docs_in_folder(folder_id: str = "root", page_size: int = 100) -> dict:
    """
    Lists Google Docs within a specific Drive folder.

    Args:
        folder_id: ID of the Drive folder to list Docs from (default is 'root' for My Drive)
        page_size: Number of results to return (default is 100)

    Returns:
        dict: A dictionary containing count, folder_id, and docs list.
    """
    try:
        request = ListDocsInFolderRequest(folder_id=folder_id, page_size=page_size)
    except Exception as e:
        return ListDocsInFolderResponse(
            count=0,
            folder_id=folder_id,
            docs=[],
            error=f"Invalid parameters: {str(e)}",
        ).model_dump()

    try:
        service = get_service()
        logger.info(f"[list_docs_in_folder] Invoked. Folder ID: '{folder_id}'")

        rsp = await asyncio.to_thread(
            service.files()
            .list(
                q=f"'{request.folder_id}' in parents and mimeType='application/vnd.google-apps.document' and trashed=false",
                pageSize=request.page_size,
                fields="files(id, name, modifiedTime, webViewLink)",
                supportsAllDrives=True,
                includeItemsFromAllDrives=True,
            )
            .execute
        )
        items = rsp.get("files", [])
        if not items:
            return ListDocsInFolderResponse(
                count=0, folder_id=request.folder_id, docs=[]
            ).model_dump()

        doc_list = [
            DocInfo(
                id=f.get("id", ""),
                name=f.get("name", "Unnamed Document"),
                created_time="",
                modified_time=f.get("modifiedTime", ""),
                web_view_link=f.get("webViewLink", ""),
            )
            for f in items
        ]

        logger.info(f"Listed {len(doc_list)} docs in folder '{request.folder_id}'")
        return ListDocsInFolderResponse(
            count=len(doc_list), folder_id=request.folder_id, docs=doc_list
        ).model_dump()

    except Exception as error:
        logger.error(f"Failed to list docs in folder {folder_id}: {error}")
        return ListDocsInFolderResponse(
            count=0, folder_id=folder_id, docs=[], error=str(error)
        ).model_dump()


@content_server.tool()
async def create_doc(
    title: str,
    content: str = "",
) -> dict:
    """
    Creates a new Google Doc and optionally inserts initial content.
    Args:
        title: Title of the new document. Required.
        content: Initial text content to insert into the document (optional)
    Returns:
        dict: A dictionary containing success status, document_id, title, and web_view_link.
    """
    try:
        request = CreateDocRequest(title=title, content=content)
    except Exception as e:
        return CreateDocResponse(
            status="failed", title=title, error=f"Invalid parameters: {str(e)}"
        ).model_dump()

    try:
        service = get_service()
        logger.info(f"[create_doc] Invoked. Title='{title}'")

        doc = await asyncio.to_thread(
            service.documents().create(body={"title": request.title}).execute
        )
        doc_id = doc.get("documentId")
        if request.content:
            requests = [
                {"insertText": {"location": {"index": 1}, "text": request.content}}
            ]
            await asyncio.to_thread(
                service.documents()
                .batchUpdate(documentId=doc_id, body={"requests": requests})
                .execute
            )
        link = f"https://docs.google.com/document/d/{doc_id}/edit"
        logger.info(
            f"Successfully created Google Doc '{request.title}' (ID: {doc_id}). Link: {link}"
        )
        return CreateDocResponse(
            status="success",
            document_id=doc_id,
            title=request.title,
            web_view_link=link,
        ).model_dump()

    except Exception as error:
        logger.error(f"Failed to create doc '{title}': {error}")
        return CreateDocResponse(
            status="failed", title=title, error=str(error)
        ).model_dump()


@content_server.tool()
async def modify_doc_text(
    document_id: str,
    start_index: int,
    end_index: int = None,
    text: str = None,
    bold: bool = None,
    italic: bool = None,
    underline: bool = None,
    font_size: int = None,
    font_family: str = None,
    text_color: Any = None,
    background_color: Any = None,
) -> dict:
    """
    Modifies text in a Google Doc - can insert/replace text and/or apply formatting in a single operation.

    Args:
        document_id: ID of the document to update
        start_index: Start position for operation (0-based)
        end_index: End position for text replacement/formatting (if not provided with text, text is inserted)
        text: New text to insert or replace with (optional - can format existing text without changing it)
        bold: Whether to make text bold (True/False/None to leave unchanged)
        italic: Whether to make text italic (True/False/None to leave unchanged)
        underline: Whether to underline text (True/False/None to leave unchanged)
        font_size: Font size in points
        font_family: Font family name (e.g., "Arial", "Times New Roman")
        text_color: Foreground text color as a hex string (#RRGGBB)
        background_color: Background/highlight color as a hex string (#RRGGBB)

    Returns:
        dict: A dictionary containing success status, document_id, operation, and web_view_link.
    """
    try:
        request = ModifyDocTextRequest(
            document_id=document_id,
            start_index=start_index,
            end_index=end_index,
            text=text,
            bold=bold,
            italic=italic,
            underline=underline,
            font_size=font_size,
            font_family=font_family,
            text_color=text_color,
            background_color=background_color,
        )
    except Exception as e:
        return ModifyDocTextResponse(
            status="failed",
            document_id=document_id,
            operations=[],
            web_view_link="",
            error=f"Invalid parameters: {str(e)}",
        ).model_dump()

    try:
        service = get_service()
        logger.info(
            f"[modify_doc_text] Doc={request.document_id}, start={request.start_index}, end={request.end_index}, text={request.text is not None}, "
            f"formatting={any([request.bold, request.italic, request.underline, request.font_size, request.font_family, request.text_color, request.background_color])}"
        )

        # Input validation
        validator = ValidationManager()

        is_valid, error_msg = validator.validate_document_id(request.document_id)
        if not is_valid:
            return ModifyDocTextResponse(
                status="failed",
                document_id=request.document_id,
                operations=[],
                web_view_link="",
                error=f"Error: {error_msg}",
            ).model_dump()

        # Validate that we have something to do
        if request.text is None and not any(
            [
                request.bold is not None,
                request.italic is not None,
                request.underline is not None,
                request.font_size,
                request.font_family,
                request.text_color,
                request.background_color,
            ]
        ):
            return ModifyDocTextResponse(
                status="failed",
                document_id=request.document_id,
                operations=[],
                web_view_link="",
                error="Must provide either 'text' to insert/replace, or formatting parameters (bold, italic, underline, font_size, font_family, text_color, background_color).",
            ).model_dump()

        # Validate text formatting params if provided
        if any(
            [
                request.bold is not None,
                request.italic is not None,
                request.underline is not None,
                request.font_size,
                request.font_family,
                request.text_color,
                request.background_color,
            ]
        ):
            is_valid, error_msg = validator.validate_text_formatting_params(
                request.bold,
                request.italic,
                request.underline,
                request.font_size,
                request.font_family,
                request.text_color,
                request.background_color,
            )
            if not is_valid:
                return ModifyDocTextResponse(
                    status="failed",
                    document_id=request.document_id,
                    operations=[],
                    web_view_link="",
                    error=f"Error: {error_msg}",
                ).model_dump()

            # For formatting, we need end_index
            if request.end_index is None:
                return ModifyDocTextResponse(
                    status="failed",
                    document_id=request.document_id,
                    operations=[],
                    web_view_link="",
                    error="'end_index' is required when applying formatting.",
                ).model_dump()

            is_valid, error_msg = validator.validate_index_range(
                request.start_index, request.end_index
            )
            if not is_valid:
                return ModifyDocTextResponse(
                    status="failed",
                    document_id=request.document_id,
                    operations=[],
                    web_view_link="",
                    error=f"Error: {error_msg}",
                ).model_dump()

        requests = []
        operations = []

        # Handle text insertion/replacement
        if request.text is not None:
            if (
                request.end_index is not None
                and request.end_index > request.start_index
            ):
                # Text replacement
                if request.start_index == 0:
                    # Special case: Cannot delete at index 0 (first section break)
                    # Instead, we insert new text at index 1 and then delete the old text
                    requests.append(create_insert_text_request(1, request.text))
                    adjusted_end = request.end_index + len(request.text)
                    requests.append(
                        create_delete_range_request(1 + len(request.text), adjusted_end)
                    )
                    operations.append(
                        f"Replaced text from index {request.start_index} to {request.end_index}"
                    )
                else:
                    # Normal replacement: delete old text, then insert new text
                    requests.extend(
                        [
                            create_delete_range_request(
                                request.start_index, request.end_index
                            ),
                            create_insert_text_request(
                                request.start_index, request.text
                            ),
                        ]
                    )
                    operations.append(
                        f"Replaced text from index {request.start_index} to {request.end_index}"
                    )
            else:
                # Text insertion
                actual_index = 1 if request.start_index == 0 else request.start_index
                requests.append(create_insert_text_request(actual_index, request.text))
                operations.append(f"Inserted text at index {request.start_index}")

        # Handle formatting
        if any(
            [
                request.bold is not None,
                request.italic is not None,
                request.underline is not None,
                request.font_size,
                request.font_family,
                request.text_color,
                request.background_color,
            ]
        ):
            # Adjust range for formatting based on text operations
            format_start = request.start_index
            format_end = request.end_index

            if request.text is not None:
                if request.end_index is not None and request.end_index > request.start_index:
                    # Text was replaced - format the new text
                    format_end = request.start_index + len(request.text)
                else:
                    # Text was inserted - format the inserted text
                    actual_index = 1 if request.start_index == 0 else request.start_index
                    format_start = actual_index
                    format_end = actual_index + len(request.text)

            # Handle special case for formatting at index 0
            if format_start == 0:
                format_start = 1
            if format_end is not None and format_end <= format_start:
                format_end = format_start + 1

            requests.append(
                create_format_text_request(
                    format_start,
                    format_end,
                    request.bold,
                    request.italic,
                    request.underline,
                    request.font_size,
                    request.font_family,
                    request.text_color,
                    request.background_color,
                )
            )

            format_details = []
            if request.bold is not None:
                format_details.append(f"bold={request.bold}")
            if request.italic is not None:
                format_details.append(f"italic={request.italic}")
            if request.underline is not None:
                format_details.append(f"underline={request.underline}")
            if request.font_size:
                format_details.append(f"font_size={request.font_size}")
            if request.font_family:
                format_details.append(f"font_family={request.font_family}")
            if request.text_color:
                format_details.append(f"text_color={request.text_color}")
            if request.background_color:
                format_details.append(f"background_color={request.background_color}")

            operations.append(
                f"Applied formatting ({', '.join(format_details)}) to range {format_start}-{format_end}"
            )

        await asyncio.to_thread(
            service.documents()
            .batchUpdate(documentId=request.document_id, body={"requests": requests})
            .execute
        )

        link = f"https://docs.google.com/document/d/{request.document_id}/edit"
        operation_summary = "; ".join(operations)
        logger.info(f"Successfully modified doc {document_id}: {operation_summary}")
        return ModifyDocTextResponse(
            status="success",
            document_id=document_id,
            operations=operations,
            web_view_link=link,
        ).model_dump()

    except Exception as error:
        logger.error(f"Failed to modify doc text for {document_id}: {error}")
        return ModifyDocTextResponse(
            status="failed",
            document_id=document_id,
            operations=[],
            web_view_link="",
            error=str(error),
        ).model_dump()


@content_server.tool()
async def find_and_replace_doc(
    document_id: str,
    find_text: str,
    replace_text: str,
    match_case: bool = False,
) -> dict:
    """
    Finds and replaces text throughout a Google Doc.

    Args:
        document_id: ID of the document to update
        find_text: Text to search for
        replace_text: Text to replace with
        match_case: Whether to match case exactly

    Returns:
        dict: A dictionary containing success status, replacement count, and details.
    """
    try:
        request = FindAndReplaceDocRequest(
            document_id=document_id,
            find_text=find_text,
            replace_text=replace_text,
            match_case=match_case,
        )
    except Exception as e:
        return FindAndReplaceDocResponse(
            status="failed",
            document_id=document_id,
            replacements=0,
            find_text=find_text,
            replace_text=replace_text,
            web_view_link="",
            error=f"Invalid parameters: {str(e)}",
        ).model_dump()

    try:
        service = get_service()
        logger.info(
            f"[find_and_replace_doc] Doc={document_id}, find='{find_text}', replace='{replace_text}'"
        )

        requests = [
            create_find_replace_request(
                request.find_text, request.replace_text, request.match_case
            )
        ]

        result = await asyncio.to_thread(
            service.documents()
            .batchUpdate(documentId=request.document_id, body={"requests": requests})
            .execute
        )

        # Extract number of replacements from response
        replacements = 0
        if "replies" in result and result["replies"]:
            reply = result["replies"][0]
            if "replaceAllText" in reply:
                replacements = reply["replaceAllText"].get("occurrencesChanged", 0)

        link = f"https://docs.google.com/document/d/{request.document_id}/edit"
        logger.info(f"Replaced {replacements} occurrences in doc {request.document_id}")
        return FindAndReplaceDocResponse(
            status="success",
            document_id=request.document_id,
            replacements=replacements,
            find_text=request.find_text,
            replace_text=request.replace_text,
            web_view_link=link,
        ).model_dump()

    except Exception as error:
        logger.error(f"Failed to find and replace in doc {document_id}: {error}")
        return FindAndReplaceDocResponse(
            status="failed",
            document_id=document_id,
            replacements=0,
            find_text=find_text,
            replace_text=replace_text,
            web_view_link="",
            error=str(error),
        ).model_dump()


@content_server.tool()
async def insert_doc_elements(
    document_id: str,
    element_type: str,
    index: int,
    rows: int = None,
    columns: int = None,
    list_type: str = None,
    text: str = None,
) -> str:
    """
    Inserts structural elements like tables, lists, or page breaks into a Google Doc.

    Args:
        document_id: ID of the document to update
        element_type: Type of element to insert ("table", "list", "page_break")
        index: Position to insert element (0-based)
        rows: Number of rows for table (required for table)
        columns: Number of columns for table (required for table)
        list_type: Type of list ("UNORDERED", "ORDERED") (required for list)
        text: Initial text content for list items

    Returns:
        str: Confirmation message with insertion details
    """

    try:
        request = InsertDocElementsRequest(
            document_id=document_id,
            element_type=element_type,
            index=index,
            rows=rows,
            columns=columns,
            list_type=list_type,
            text=text,
        )
    except Exception as e:
        return InsertDocElementsResponse(
            status="error", error=f"Invalid parameters: {str(e)}"
        ).model_dump()

    try:
        service = get_service()

        logger.info(
            f"[insert_doc_elements] Doc={document_id}, type={element_type}, index={index}"
        )

        # Handle the special case where we can't insert at the first section break
        # If index is 0, bump it to 1 to avoid the section break
        if index == 0:
            logger.debug("Adjusting index from 0 to 1 to avoid first section break")
            index = 1

        requests = []
        description = ""

        if request.element_type == "table":
            if not request.rows or not request.columns:
                return InsertDocElementsResponse(
                    status="error",
                    document_id=request.document_id,
                    element_type=request.element_type,
                    index=index,
                    web_view_link="",
                    error="'rows' and 'columns' parameters are required for table insertion.",
                ).model_dump()

            requests.append(
                create_insert_table_request(index, request.rows, request.columns)
            )
            description = f"table ({request.rows}x{request.columns})"

        elif request.element_type == "list":
            if not request.list_type:
                return InsertDocElementsResponse(
                    status="error",
                    document_id=request.document_id,
                    element_type=request.element_type,
                    index=index,
                    web_view_link="",
                    error="'list_type' parameter is required for list insertion ('UNORDERED' or 'ORDERED').",
                ).model_dump()

            list_text = request.text if request.text else "List item"

            # Insert text first, then create list
            requests.extend(
                [
                    create_insert_text_request(index, list_text + "\n"),
                    create_bullet_list_request(
                        index, index + len(list_text), request.list_type
                    ),
                ]
            )
            description = f"{request.list_type.lower()} list"

        elif request.element_type == "page_break":
            requests.append(create_insert_page_break_request(index))
            description = "page break"
        else:
            return InsertDocElementsResponse(
                status="error",
                document_id=request.document_id,
                element_type=request.element_type,
                index=index,
                web_view_link="",
                error=f"Unsupported element_type '{request.element_type}'. Supported types are 'table', 'list', 'page_break'.",
            ).model_dump()

        await asyncio.to_thread(
            service.documents()
            .batchUpdate(documentId=document_id, body={"requests": requests})
            .execute
        )

        link = f"https://docs.google.com/document/d/{request.document_id}/edit"
        logger.info(f"Successfully inserted {description} in doc {request.document_id}")
        return InsertDocElementsResponse(
            status="success",
            document_id=request.document_id,
            element_type=request.element_type,
            index=index,
            web_view_link=link,
        ).model_dump()

    except Exception as error:
        logger.error(f"Failed to insert element in doc {document_id}: {error}")
        return InsertDocElementsResponse(
            status="error",
            document_id=document_id,
            element_type=element_type,
            index=index,
            web_view_link="",
            error=str(error),
        ).model_dump()


@content_server.tool()
async def insert_doc_image(
    document_id: str,
    image_source: str,
    index: int,
    width: int = 0,
    height: int = 0,
) -> dict:
    """
    Inserts an image into a Google Doc from Drive or a URL.

    Args:
        document_id: ID of the document to update
        image_source: Drive file ID or public image URL
        index: Position to insert image (0-based)
        width: Image width in points (optional)
        height: Image height in points (optional)

    Returns:
        dict: A dictionary containing status, document_id, and operation details.
    """
    try:
        request = InsertDocImageRequest(
            document_id=document_id,
            image_source=image_source,
            index=index,
            width=width,
            height=height,
        )
    except Exception as e:
        return InsertDocImageResponse(
            status="error",
            document_id=document_id,
            image_source=image_source,
            index=index,
            web_view_link="",
            error=f"Invalid parameters: {str(e)}",
        ).model_dump()

    try:
        docs_service = get_service()
        drive_service = drive_get_service()

        logger.info(
            f"[insert_doc_image] Doc={document_id}, source={image_source}, index={index}"
        )

        # Handle the special case where we can't insert at the first section break
        # If index is 0, bump it to 1 to avoid the section break
        if index == 0:
            logger.debug("Adjusting index from 0 to 1 to avoid first section break")
            index = 1

        # Determine if source is a Drive file ID or URL
        is_drive_file = not (
            request.image_source.startswith("http://")
            or request.image_source.startswith("https://")
        )

        if is_drive_file:
            # Verify Drive file exists and get metadata
            try:
                file_metadata = await asyncio.to_thread(
                    drive_service.files()
                    .get(
                        fileId=request.image_source,
                        fields="id, name, mimeType",
                        supportsAllDrives=True,
                    )
                    .execute
                )
                mime_type = file_metadata.get("mimeType", "")
                if not mime_type.startswith("image/"):
                    return InsertDocImageResponse(
                        status="error",
                        document_id=request.document_id,
                        image_source=request.image_source,
                        index=request.index,
                        web_view_link="",
                        error=f"File {request.image_source} is not an image (MIME type: {mime_type}).",
                    ).model_dump()

                image_uri = f"https://drive.google.com/uc?id={request.image_source}"
                source_description = (
                    f"Drive file {file_metadata.get('name', request.image_source)}"
                )
            except Exception as e:
                return InsertDocImageResponse(
                    status="error",
                    document_id=request.document_id,
                    image_source=request.image_source,
                    index=request.index,
                    web_view_link="",
                    error=f"Could not access Drive file {request.image_source}: {str(e)}",
                ).model_dump()
        else:
            image_uri = request.image_source
            source_description = "URL image"

        # Use helper to create image request
        requests = [
            create_insert_image_request(index, image_uri, request.width, request.height)
        ]

        await asyncio.to_thread(
            docs_service.documents()
            .batchUpdate(documentId=request.document_id, body={"requests": requests})
            .execute
        )

        link = f"https://docs.google.com/document/d/{request.document_id}/edit"
        logger.info(f"Successfully inserted image in doc {request.document_id}")
        return InsertDocImageResponse(
            status="success",
            document_id=request.document_id,
            image_source=request.image_source,
            index=index,
            web_view_link=link,
        ).model_dump()

    except Exception as error:
        logger.error(f"Failed to insert image in doc {document_id}: {error}")
        return InsertDocImageResponse(
            status="error",
            document_id=document_id,
            image_source=image_source,
            index=index,
            web_view_link="",
            error=str(error),
        ).model_dump()


@content_server.tool()
async def update_doc_headers_footers(
    document_id: str,
    section_type: str,
    content: str,
    header_footer_type: str = "DEFAULT",
) -> dict:
    """
    Updates headers or footers in a Google Doc.

    Args:
        document_id: ID of the document to update
        section_type: Type of section to update ("header" or "footer")
        content: Text content for the header/footer
        header_footer_type: Type of header/footer ("DEFAULT", "FIRST_PAGE_ONLY", "EVEN_PAGE")

    Returns:
        dict: A dictionary containing status, document_id, and operation details.
    """
    try:
        request = UpdateDocHeadersFootersRequest(
            document_id=document_id,
            section_type=section_type,
            content=content,
            header_footer_type=header_footer_type,
        )
    except Exception as e:
        return UpdateDocHeadersFootersResponse(
            status="error",
            document_id=document_id,
            section_type=section_type,
            header_footer_type=header_footer_type,
            web_view_link="",
            error=f"Invalid parameters: {str(e)}",
        ).model_dump()

    try:
        service = get_service()
        logger.info(
            f"[update_doc_headers_footers] Doc={document_id}, type={section_type}"
        )

        # Use HeaderFooterManager to handle the complex logic
        header_footer_manager = HeaderFooterManager(service)

        success, message = await header_footer_manager.update_header_footer_content(
            request.document_id,
            request.section_type,
            request.content,
            request.header_footer_type,
        )

        link = f"https://docs.google.com/document/d/{request.document_id}/edit"
        if success:
            logger.info(
                f"Successfully updated header/footer in doc {request.document_id}"
            )
            return UpdateDocHeadersFootersResponse(
                status="success",
                document_id=request.document_id,
                section_type=request.section_type,
                header_footer_type=request.header_footer_type,
                web_view_link=link,
            ).model_dump()
        else:
            return UpdateDocHeadersFootersResponse(
                status="error",
                document_id=request.document_id,
                section_type=request.section_type,
                header_footer_type=request.header_footer_type,
                web_view_link="",
                error=message,
            ).model_dump()

    except Exception as error:
        logger.error(f"Failed to update header/footer in doc {document_id}: {error}")
        return UpdateDocHeadersFootersResponse(
            status="error",
            document_id=document_id,
            section_type=section_type,
            header_footer_type=header_footer_type,
            web_view_link="",
            error=str(error),
        ).model_dump()


@content_server.tool()
async def batch_update_doc(
    document_id: str,
    operations: List[Dict[str, Any]],
) -> dict:
    """
    Executes multiple document operations in a single atomic batch update.

    Args:
        document_id: ID of the document to update
        operations: List of operation dictionaries. Each operation should contain:
                   - type: Operation type ('insert_text', 'delete_text', 'replace_text', 'format_text', 'insert_table', 'insert_page_break')
                   - Additional parameters specific to each operation type

    Example operations:
        [
            {"type": "insert_text", "index": 1, "text": "Hello World"},
            {"type": "format_text", "start_index": 1, "end_index": 12, "bold": true},
            {"type": "insert_table", "index": 20, "rows": 2, "columns": 3}
        ]

    Returns:
        dict: A dictionary containing status, document_id, and operation results.
    """
    try:
        request = BatchUpdateDocRequest(
            document_id=document_id,
            operations=operations,
        )
    except Exception as e:
        return BatchUpdateDocResponse(
            status="error",
            document_id=document_id,
            operations_count=len(operations),
            web_view_link="",
            error=f"Invalid parameters: {str(e)}",
        ).model_dump()

    try:
        service = get_service()
        logger.debug(
            f"[batch_update_doc] Doc={document_id}, operations={len(operations)}"
        )

        # Use BatchOperationManager to handle the complex logic
        batch_manager = BatchOperationManager(service)

        success, message, metadata = await batch_manager.execute_batch_operations(
            request.document_id, request.operations
        )

        link = f"https://docs.google.com/document/d/{request.document_id}/edit"
        if success:
            logger.info(
                f"Successfully executed batch operations on doc {request.document_id}"
            )
            return BatchUpdateDocResponse(
                status="success",
                document_id=request.document_id,
                operations_count=len(request.operations),
                web_view_link=link,
            ).model_dump()
        else:
            return BatchUpdateDocResponse(
                status="error",
                document_id=request.document_id,
                operations_count=len(request.operations),
                web_view_link="",
                error=message,
            ).model_dump()
    except Exception as error:
        logger.error(f"Failed to batch update doc {document_id}: {error}")
        return BatchUpdateDocResponse(
            status="error",
            document_id=document_id,
            operations_count=len(operations),
            web_view_link="",
            error=str(error),
        ).model_dump()


@content_server.tool()
async def inspect_doc_structure(
    document_id: str,
    detailed: bool = False,
) -> str:
    """
    Essential tool for finding safe insertion points and understanding document structure.

    USE THIS FOR:
    - Finding the correct index for table insertion
    - Understanding document layout before making changes
    - Locating existing tables and their positions
    - Getting document statistics and complexity info

    CRITICAL FOR TABLE OPERATIONS:
    ALWAYS call this BEFORE creating tables to get a safe insertion index.

    WHAT THE OUTPUT SHOWS:
    - total_elements: Number of document elements
    - total_length: Maximum safe index for insertion
    - tables: Number of existing tables
    - table_details: Position and dimensions of each table

    WORKFLOW:
    Step 1: Call this function
    Step 2: Note the "total_length" value
    Step 3: Use an index < total_length for table insertion
    Step 4: Create your table

    Args:
        document_id: ID of the document to inspect
        detailed: Whether to return detailed structure information

    Returns:
        str: JSON string containing document structure and safe insertion indices
    """
    try:
        request = InspectDocStructureRequest(
            document_id=document_id,
            detailed=detailed,
        )
    except Exception as error:
        return InspectDocStructureResponse(
            status="error",
            document_id=request.document_id,
            structure={},
            error=str(error),
        ).model_dump()

    try:
        service = get_service()
        logger.debug(f"[inspect_doc_structure] Doc={document_id}, detailed={detailed}")

        # Get the document
        doc = await asyncio.to_thread(
            service.documents().get(documentId=request.document_id).execute
        )

        if request.detailed:
            # Return full parsed structure
            structure = parse_document_structure(doc)

        # Simplify for JSON serialization
        result = {
            "title": structure["title"],
            "total_length": structure["total_length"],
            "statistics": {
                "elements": len(structure["body"]),
                "tables": len(structure["tables"]),
                "paragraphs": sum(
                    1 for e in structure["body"] if e.get("type") == "paragraph"
                ),
                "has_headers": bool(structure["headers"]),
                "has_footers": bool(structure["footers"]),
            },
            "elements": [],
        }

        # Add element summaries
        for element in structure["body"]:
            elem_summary = {
                "type": element["type"],
                "start_index": element["start_index"],
                "end_index": element["end_index"],
            }

            if element["type"] == "table":
                elem_summary["rows"] = element["rows"]
                elem_summary["columns"] = element["columns"]
                elem_summary["cell_count"] = len(element.get("cells", []))
            elif element["type"] == "paragraph":
                elem_summary["text_preview"] = element.get("text", "")[:100]

            result["elements"].append(elem_summary)

        # Add table details
        if structure["tables"]:
            result["tables"] = []
            for i, table in enumerate(structure["tables"]):
                table_data = extract_table_as_data(table)
                result["tables"].append(
                    {
                        "index": i,
                        "position": {
                            "start": table["start_index"],
                            "end": table["end_index"],
                        },
                        "dimensions": {
                            "rows": table["rows"],
                            "columns": table["columns"],
                        },
                        "preview": table_data[:3] if table_data else [],  # First 3 rows
                    }
                )

        else:
            # Return basic analysis
            result = analyze_document_complexity(doc)

            # Add table information
            tables = find_tables(doc)
            if tables:
                result["table_details"] = []
                for i, table in enumerate(tables):
                    result["table_details"].append(
                        {
                            "index": i,
                            "rows": table["rows"],
                            "columns": table["columns"],
                            "start_index": table["start_index"],
                            "end_index": table["end_index"],
                        }
                    )

            logger.info(
                f"Successfully inspected structure of doc {request.document_id}"
            )
            return InspectDocStructureResponse(
                status="success",
                document_id=request.document_id,
                structure=result,
            ).model_dump()

    except Exception as error:
        logger.error(
            f"Failed to inspect doc structure for {request.document_id}: {error}"
        )
        return InspectDocStructureResponse(
            status="error",
            document_id=request.document_id,
            structure={},
            error=str(error),
        ).model_dump()


@content_server.tool()
async def create_table_with_data(
    document_id: str,
    table_data: List[List[str]],
    index: int,
    bold_headers: bool = True,
) -> str:
    """
    Creates a table and populates it with data in one reliable operation.

    CRITICAL: YOU MUST CALL inspect_doc_structure FIRST TO GET THE INDEX!

    MANDATORY WORKFLOW - DO THESE STEPS IN ORDER:

    Step 1: ALWAYS call inspect_doc_structure first
    Step 2: Use the 'total_length' value from inspect_doc_structure as your index
    Step 3: Format data as 2D list: [["col1", "col2"], ["row1col1", "row1col2"]]
    Step 4: Call this function with the correct index and data

    EXAMPLE DATA FORMAT:
    table_data = [
        ["Header1", "Header2", "Header3"],    # Row 0 - headers
        ["Data1", "Data2", "Data3"],          # Row 1 - first data row
        ["Data4", "Data5", "Data6"]           # Row 2 - second data row
    ]

    CRITICAL INDEX REQUIREMENTS:
    - NEVER use index values like 1, 2, 10 without calling inspect_doc_structure first
    - ALWAYS get index from inspect_doc_structure 'total_length' field
    - Index must be a valid insertion point in the document

    DATA FORMAT REQUIREMENTS:
    - Must be 2D list of strings only
    - Each inner list = one table row
    - All rows MUST have same number of columns
    - Use empty strings "" for empty cells, never None
    - Use debug_table_structure after creation to verify results

    Args:
        document_id: ID of the document to update
        table_data: 2D list of strings - EXACT format: [["col1", "col2"], ["row1col1", "row1col2"]]
        index: Document position (MANDATORY: get from inspect_doc_structure 'total_length')
        bold_headers: Whether to make first row bold (default: true)

    Returns:
        str: Confirmation with table details and link
    """
    try:
        request = CreateTableWithDataRequest(
            document_id=document_id,
            table_data=table_data,
            index=index,
            bold_headers=bold_headers,
        )
    except Exception as e:
        logger.error(f"Request validation error: {e}")
        return CreateTableWithDataResponse(
            status="error",
            document_id=document_id,
            error=f"Invalid request parameters: {str(e)}",
        ).model_dump()

    try:
        service = get_service()
        logger.debug(
            f"[create_table_with_data] Doc={request.document_id}, index={request.index}"
        )

        # Use TableOperationManager to handle the complex logic
        table_manager = TableOperationManager(service)

        # Try to create the table, and if it fails due to index being at document end, retry with index-1
        success, message, metadata = await table_manager.create_and_populate_table(
            request.document_id, request.table_data, request.index, request.bold_headers
        )

        # If it failed due to index being at or beyond document end, retry with adjusted index
        if not success and "must be less than the end index" in message:
            logger.debug(
                f"Index {request.index} is at document boundary, retrying with index {request.index - 1}"
            )
            success, message, metadata = await table_manager.create_and_populate_table(
                request.document_id,
                request.table_data,
                request.index - 1,
                request.bold_headers,
            )

        link = f"https://docs.google.com/document/d/{request.document_id}/edit"
        rows = metadata.get("rows", len(request.table_data))
        columns = metadata.get(
            "columns", len(request.table_data[0]) if request.table_data else 0
        )

        if success:
            logger.info(f"Successfully created table in doc {request.document_id}")
            return CreateTableWithDataResponse(
                status="success",
                document_id=request.document_id,
                rows=rows,
                columns=columns,
                web_view_link=link,
            ).model_dump()
        else:
            return CreateTableWithDataResponse(
                status="error",
                document_id=request.document_id,
                rows=rows,
                columns=columns,
                web_view_link="",
                error=message,
            ).model_dump()

    except Exception as error:
        logger.error(f"Failed to create table in doc {request.document_id}: {error}")
        return CreateTableWithDataResponse(
            status="error",
            document_id=request.document_id,
            rows=0,
            columns=0,
            web_view_link="",
            error=str(error),
        ).model_dump()


@content_server.tool()
async def debug_table_structure(
    document_id: str,
    table_index: int = 0,
) -> str:
    """
    ESSENTIAL DEBUGGING TOOL - Use this whenever tables don't work as expected.

    USE THIS IMMEDIATELY WHEN:
    - Table population put data in wrong cells
    - You get "table not found" errors
    - Data appears concatenated in first cell
    - Need to understand existing table structure
    - Planning to use populate_existing_table

    WHAT THIS SHOWS YOU:
    - Exact table dimensions (rows × columns)
    - Each cell's position coordinates (row,col)
    - Current content in each cell
    - Insertion indices for each cell
    - Table boundaries and ranges

    HOW TO READ THE OUTPUT:
    - "dimensions": "2x3" = 2 rows, 3 columns
    - "position": "(0,0)" = first row, first column
    - "current_content": What's actually in each cell right now
    - "insertion_index": Where new text would be inserted in that cell

    WORKFLOW INTEGRATION:
    1. After creating table → Use this to verify structure
    2. Before populating → Use this to plan your data format
    3. After population fails → Use this to see what went wrong
    4. When debugging → Compare your data array to actual table structure

    Args:
        document_id: ID of the document to inspect
        table_index: Which table to debug (0 = first table, 1 = second table, etc.)

    Returns:
        str: Detailed JSON structure showing table layout, cell positions, and current content
    """
    try:
        request = DebugTableStructureRequest(
            document_id=document_id,
            table_index=table_index,
        )
    except Exception as e:
        logger.error(f"Request validation error: {e}")
        return DebugTableStructureResponse(
            status="error",
            document_id=document_id,
            table_index=table_index,
            structure={},
            error=f"Invalid request parameters: {str(e)}",
        ).model_dump()

    try:
        service = get_service()
        logger.debug(
            f"[debug_table_structure] Doc={request.document_id}, table_index={request.table_index}"
        )

        # Get the document
        doc = await asyncio.to_thread(
            service.documents().get(documentId=request.document_id).execute
        )

        # Find tables
        tables = find_tables(doc)
        if request.table_index >= len(tables):
            return DebugTableStructureResponse(
                status="error",
                document_id=request.document_id,
                table_index=request.table_index,
                structure={},
                error=f"Table index {request.table_index} not found. Document has {len(tables)} table(s).",
            ).model_dump()

        table_info = tables[request.table_index]

        # Extract detailed cell information
        debug_info = {
            "table_index": request.table_index,
            "dimensions": f"{table_info['rows']}x{table_info['columns']}",
            "table_range": f"[{table_info['start_index']}-{table_info['end_index']}]",
            "cells": [],
        }

        for row_idx, row in enumerate(table_info["cells"]):
            row_info = []
            for col_idx, cell in enumerate(row):
                cell_debug = {
                    "position": f"({row_idx},{col_idx})",
                    "range": f"[{cell['start_index']}-{cell['end_index']}]",
                    "insertion_index": cell.get("insertion_index", "N/A"),
                    "current_content": repr(cell.get("content", "")),
                    "content_elements_count": len(cell.get("content_elements", [])),
                }
                row_info.append(cell_debug)
            debug_info["cells"].append(row_info)

        logger.info(
            f"Successfully debugged table {request.table_index} in doc {request.document_id}"
        )
        return DebugTableStructureResponse(
            status="success",
            document_id=request.document_id,
            table_index=request.table_index,
            structure=debug_info,
        ).model_dump()

    except Exception as error:
        logger.error(
            f"Failed to debug table structure for doc {request.document_id}: {error}"
        )
        return DebugTableStructureResponse(
            status="error",
            document_id=request.document_id,
            table_index=request.table_index,
            structure={},
            error=str(error),
        ).model_dump()


@content_server.tool()
async def export_doc_to_pdf(
    document_id: str,
    pdf_filename: str = None,
    folder_id: str = None,
) -> dict:
    """
    Exports a Google Doc to PDF format and saves it to Google Drive.

    Args:
        document_id: ID of the Google Doc to export
        pdf_filename: Name for the PDF file (optional - if not provided, uses original name + "_PDF")
        folder_id: Drive folder ID to save PDF in (optional - if not provided, saves in root)

    Returns:
        dict: A dictionary containing status, document_id, and PDF details.
    """
    try:
        request = ExportDocToPdfRequest(
            document_id=document_id,
            pdf_filename=pdf_filename,
            folder_id=folder_id,
        )
    except Exception as e:
        return ExportDocToPdfResponse(
            status="error",
            document_id=document_id,
            pdf_filename=pdf_filename or "",
            error=f"Invalid parameters: {str(e)}",
        ).model_dump()

    try:
        service = get_service()
        logger.info(
            f"[export_doc_to_pdf] Doc={request.document_id}, pdf_filename={request.pdf_filename}, folder_id={request.folder_id}"
        )

        # Get file metadata first to validate it's a Google Doc
        try:
            file_metadata = await asyncio.to_thread(
                service.files()
                .get(
                    fileId=request.document_id,
                    fields="id, name, mimeType, webViewLink",
                    supportsAllDrives=True,
                )
                .execute
            )
        except Exception as e:
            return ExportDocToPdfResponse(
                status="error",
                document_id=request.document_id,
                pdf_filename=request.pdf_filename or "",
                error=f"Could not access document: {str(e)}",
            ).model_dump()

        mime_type = file_metadata.get("mimeType", "")
        original_name = file_metadata.get("name", "Unknown Document")
        web_view_link = file_metadata.get("webViewLink", "#")

        # Verify it's a Google Doc
        if mime_type != "application/vnd.google-apps.document":
            return ExportDocToPdfResponse(
                status="error",
                document_id=request.document_id,
                pdf_filename=request.pdf_filename or "",
                error=f"File '{original_name}' is not a Google Doc (MIME type: {mime_type}). Only native Google Docs can be exported to PDF.",
            ).model_dump()

        logger.info(f"[export_doc_to_pdf] Exporting '{original_name}' to PDF")

        # Export the document as PDF
        try:
            request_obj = service.files().export_media(
                fileId=request.document_id,
                mimeType="application/pdf",
                supportsAllDrives=True,
            )

            fh = io.BytesIO()
            downloader = MediaIoBaseDownload(fh, request_obj)

            done = False
            while not done:
                _, done = await asyncio.to_thread(downloader.next_chunk)

            pdf_content = fh.getvalue()
            pdf_size = len(pdf_content)

        except Exception as e:
            return ExportDocToPdfResponse(
                status="error",
                document_id=request.document_id,
                pdf_filename=request.pdf_filename or "",
                error=f"Failed to export document to PDF: {str(e)}",
            ).model_dump()

        # Determine PDF filename
        final_pdf_filename = request.pdf_filename
        if not final_pdf_filename:
            final_pdf_filename = f"{original_name}_PDF.pdf"
        elif not final_pdf_filename.endswith(".pdf"):
            final_pdf_filename += ".pdf"

        # Upload PDF to Drive
        try:
            # Reuse the existing BytesIO object by resetting to the beginning
            fh.seek(0)
            # Create media upload object
            media = MediaIoBaseUpload(fh, mimetype="application/pdf", resumable=True)

            # Prepare file metadata for upload
            file_metadata = {"name": final_pdf_filename, "mimeType": "application/pdf"}

            # Add parent folder if specified
            if request.folder_id:
                file_metadata["parents"] = [request.folder_id]

            # Upload the file
            uploaded_file = await asyncio.to_thread(
                service.files()
                .create(
                    body=file_metadata,
                    media_body=media,
                    fields="id, name, webViewLink, parents",
                    supportsAllDrives=True,
                )
                .execute
            )

            pdf_file_id = uploaded_file.get("id")
            pdf_web_link = uploaded_file.get("webViewLink", "#")

            logger.info(
                f"[export_doc_to_pdf] Successfully uploaded PDF to Drive: {pdf_file_id}"
            )

            return ExportDocToPdfResponse(
                status="success",
                document_id=request.document_id,
                pdf_id=pdf_file_id,
                pdf_filename=final_pdf_filename,
                web_view_link=pdf_web_link,
            ).model_dump()

        except Exception as e:
            return ExportDocToPdfResponse(
                status="error",
                document_id=request.document_id,
                pdf_filename=final_pdf_filename,
                error=f"Failed to upload PDF to Drive: {str(e)}. PDF was generated successfully ({pdf_size:,} bytes) but could not be saved to Drive.",
            ).model_dump()

    except Exception as error:
        logger.error(f"Failed to export doc to PDF for {request.document_id}: {error}")
        return ExportDocToPdfResponse(
            status="error",
            document_id=request.document_id,
            pdf_filename=request.pdf_filename or "",
            error=str(error),
        ).model_dump()


# Create comment management tools for documents
_comment_tools = create_comment_tools("document", "document_id")

# Extract and register the functions
read_doc_comments = _comment_tools["read_comments"]
create_doc_comment = _comment_tools["create_comment"]
reply_to_comment = _comment_tools["reply_to_comment"]
resolve_comment = _comment_tools["resolve_comment"]
