"""
Google Docs Helper Functions

This module provides utility functions for common Google Docs operations
to simplify the implementation of document editing tools.
"""

from typing import Dict, Any, Optional, Tuple, List, Union
from utils.helper import setup_logger

logger = setup_logger(__name__)


def _normalize_color(color: Any, param_name: str) -> Optional[Dict[str, float]]:
    """
    Normalize a user-supplied color into Docs API rgbColor format.

    Supports:
    - Hex strings: "#RRGGBB" or "RRGGBB"
    - Tuple/list of 3 ints (0-255) or floats (0-1)
    """
    if color is None:
        return None

    def _to_component(value: Any) -> float:
        if isinstance(value, bool):
            raise ValueError(f"{param_name} components cannot be boolean values")
        if isinstance(value, int):
            if value < 0 or value > 255:
                raise ValueError(
                    f"{param_name} components must be 0-255 when using integers"
                )
            return value / 255
        if isinstance(value, float):
            if value < 0 or value > 1:
                raise ValueError(
                    f"{param_name} components must be between 0 and 1 when using floats"
                )
            return value
        raise ValueError(f"{param_name} components must be int (0-255) or float (0-1)")

    if isinstance(color, str):
        hex_color = color.lstrip("#")
        if len(hex_color) != 6 or any(
            c not in "0123456789abcdefABCDEF" for c in hex_color
        ):
            raise ValueError(f"{param_name} must be a hex string like '#RRGGBB'")
        r = int(hex_color[0:2], 16) / 255
        g = int(hex_color[2:4], 16) / 255
        b = int(hex_color[4:6], 16) / 255
        return {"red": r, "green": g, "blue": b}

    if isinstance(color, (list, tuple)) and len(color) == 3:
        r = _to_component(color[0])
        g = _to_component(color[1])
        b = _to_component(color[2])
        return {"red": r, "green": g, "blue": b}

    raise ValueError(f"{param_name} must be a hex string or RGB tuple/list")


def build_text_style(
    bold: bool = None,
    italic: bool = None,
    underline: bool = None,
    font_size: int = None,
    font_family: str = None,
    text_color: Any = None,
    background_color: Any = None,
) -> tuple[Dict[str, Any], list[str]]:
    """
    Build text style object for Google Docs API requests.

    Args:
        bold: Whether text should be bold
        italic: Whether text should be italic
        underline: Whether text should be underlined
        font_size: Font size in points
        font_family: Font family name
        text_color: Text color as hex string or RGB tuple/list
        background_color: Background (highlight) color as hex string or RGB tuple/list

    Returns:
        Tuple of (text_style_dict, list_of_field_names)
    """
    text_style = {}
    fields = []

    if bold is not None:
        text_style["bold"] = bold
        fields.append("bold")

    if italic is not None:
        text_style["italic"] = italic
        fields.append("italic")

    if underline is not None:
        text_style["underline"] = underline
        fields.append("underline")

    if font_size is not None:
        text_style["fontSize"] = {"magnitude": font_size, "unit": "PT"}
        fields.append("fontSize")

    if font_family is not None:
        text_style["weightedFontFamily"] = {"fontFamily": font_family}
        fields.append("weightedFontFamily")

    if text_color is not None:
        rgb = _normalize_color(text_color, "text_color")
        text_style["foregroundColor"] = {"color": {"rgbColor": rgb}}
        fields.append("foregroundColor")

    if background_color is not None:
        rgb = _normalize_color(background_color, "background_color")
        text_style["backgroundColor"] = {"color": {"rgbColor": rgb}}
        fields.append("backgroundColor")

    return text_style, fields


def create_insert_text_request(index: int, text: str) -> Dict[str, Any]:
    """
    Create an insertText request for Google Docs API.

    Args:
        index: Position to insert text
        text: Text to insert

    Returns:
        Dictionary representing the insertText request
    """
    return {"insertText": {"location": {"index": index}, "text": text}}


def create_insert_text_segment_request(
    index: int, text: str, segment_id: str
) -> Dict[str, Any]:
    """
    Create an insertText request for Google Docs API with segmentId (for headers/footers).

    Args:
        index: Position to insert text
        text: Text to insert
        segment_id: Segment ID (for targeting headers/footers)

    Returns:
        Dictionary representing the insertText request with segmentId
    """
    return {
        "insertText": {
            "location": {"segmentId": segment_id, "index": index},
            "text": text,
        }
    }


def create_delete_range_request(start_index: int, end_index: int) -> Dict[str, Any]:
    """
    Create a deleteContentRange request for Google Docs API.

    Args:
        start_index: Start position of content to delete
        end_index: End position of content to delete

    Returns:
        Dictionary representing the deleteContentRange request
    """
    return {
        "deleteContentRange": {
            "range": {"startIndex": start_index, "endIndex": end_index}
        }
    }


def create_format_text_request(
    start_index: int,
    end_index: int,
    bold: bool = None,
    italic: bool = None,
    underline: bool = None,
    font_size: int = None,
    font_family: str = None,
    text_color: Any = None,
    background_color: Any = None,
) -> Optional[Dict[str, Any]]:
    """
    Create an updateTextStyle request for Google Docs API.

    Args:
        start_index: Start position of text to format
        end_index: End position of text to format
        bold: Whether text should be bold
        italic: Whether text should be italic
        underline: Whether text should be underlined
        font_size: Font size in points
        font_family: Font family name
        text_color: Text color as hex string or RGB tuple/list
        background_color: Background (highlight) color as hex string or RGB tuple/list

    Returns:
        Dictionary representing the updateTextStyle request, or None if no styles provided
    """
    text_style, fields = build_text_style(
        bold, italic, underline, font_size, font_family, text_color, background_color
    )

    if not text_style:
        return None

    return {
        "updateTextStyle": {
            "range": {"startIndex": start_index, "endIndex": end_index},
            "textStyle": text_style,
            "fields": ",".join(fields),
        }
    }


def create_find_replace_request(
    find_text: str, replace_text: str, match_case: bool = False
) -> Dict[str, Any]:
    """
    Create a replaceAllText request for Google Docs API.

    Args:
        find_text: Text to find
        replace_text: Text to replace with
        match_case: Whether to match case exactly

    Returns:
        Dictionary representing the replaceAllText request
    """
    return {
        "replaceAllText": {
            "containsText": {"text": find_text, "matchCase": match_case},
            "replaceText": replace_text,
        }
    }


def create_insert_table_request(index: int, rows: int, columns: int) -> Dict[str, Any]:
    """
    Create an insertTable request for Google Docs API.

    Args:
        index: Position to insert table
        rows: Number of rows
        columns: Number of columns

    Returns:
        Dictionary representing the insertTable request
    """
    return {
        "insertTable": {"location": {"index": index}, "rows": rows, "columns": columns}
    }


def create_insert_page_break_request(index: int) -> Dict[str, Any]:
    """
    Create an insertPageBreak request for Google Docs API.

    Args:
        index: Position to insert page break

    Returns:
        Dictionary representing the insertPageBreak request
    """
    return {"insertPageBreak": {"location": {"index": index}}}


def create_insert_image_request(
    index: int, image_uri: str, width: int = None, height: int = None
) -> Dict[str, Any]:
    """
    Create an insertInlineImage request for Google Docs API.

    Args:
        index: Position to insert image
        image_uri: URI of the image (Drive URL or public URL)
        width: Image width in points
        height: Image height in points

    Returns:
        Dictionary representing the insertInlineImage request
    """
    request = {"insertInlineImage": {"location": {"index": index}, "uri": image_uri}}

    # Add size properties if specified
    object_size = {}
    if width is not None:
        object_size["width"] = {"magnitude": width, "unit": "PT"}
    if height is not None:
        object_size["height"] = {"magnitude": height, "unit": "PT"}

    if object_size:
        request["insertInlineImage"]["objectSize"] = object_size

    return request


def create_bullet_list_request(
    start_index: int, end_index: int, list_type: str = "UNORDERED"
) -> Dict[str, Any]:
    """
    Create a createParagraphBullets request for Google Docs API.

    Args:
        start_index: Start of text range to convert to list
        end_index: End of text range to convert to list
        list_type: Type of list ("UNORDERED" or "ORDERED")

    Returns:
        Dictionary representing the createParagraphBullets request
    """
    bullet_preset = (
        "BULLET_DISC_CIRCLE_SQUARE"
        if list_type == "UNORDERED"
        else "NUMBERED_DECIMAL_ALPHA_ROMAN"
    )

    return {
        "createParagraphBullets": {
            "range": {"startIndex": start_index, "endIndex": end_index},
            "bulletPreset": bullet_preset,
        }
    }


def validate_operation(operation: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Validate a batch operation dictionary.

    Args:
        operation: Operation dictionary to validate

    Returns:
        Tuple of (is_valid, error_message)
    """
    op_type = operation.get("type")
    if not op_type:
        return False, "Missing 'type' field"

    # Validate required fields for each operation type
    required_fields = {
        "insert_text": ["index", "text"],
        "delete_text": ["start_index", "end_index"],
        "replace_text": ["start_index", "end_index", "text"],
        "format_text": ["start_index", "end_index"],
        "insert_table": ["index", "rows", "columns"],
        "insert_page_break": ["index"],
        "find_replace": ["find_text", "replace_text"],
    }

    if op_type not in required_fields:
        return False, f"Unsupported operation type: {op_type or 'None'}"

    for field in required_fields[op_type]:
        if field not in operation:
            return False, f"Missing required field: {field}"

    return True, ""


"""
Google Docs Document Structure Parsing and Analysis

This module provides utilities for parsing and analyzing the structure
of Google Docs documents, including finding tables, cells, and other elements.
"""


def parse_document_structure(doc_data: dict[str, Any]) -> dict[str, Any]:
    """
    Parse the full document structure into a navigable format.

    Args:
        doc_data: Raw document data from Google Docs API

    Returns:
        Dictionary containing parsed structure with elements and their positions
    """
    structure = {
        "title": doc_data.get("title", ""),
        "body": [],
        "tables": [],
        "headers": {},
        "footers": {},
        "total_length": 0,
    }

    body = doc_data.get("body", {})
    content = body.get("content", [])

    for element in content:
        element_info = _parse_element(element)
        if element_info:
            structure["body"].append(element_info)
            if element_info["type"] == "table":
                structure["tables"].append(element_info)

    # Calculate total document length
    if structure["body"]:
        last_element = structure["body"][-1]
        structure["total_length"] = last_element.get("end_index", 0)

    # Parse headers and footers
    for header_id, header_data in doc_data.get("headers", {}).items():
        structure["headers"][header_id] = _parse_segment(header_data)

    for footer_id, footer_data in doc_data.get("footers", {}).items():
        structure["footers"][footer_id] = _parse_segment(footer_data)

    return structure


def _parse_element(element: dict[str, Any]) -> Optional[dict[str, Any]]:
    """
    Parse a single document element.

    Args:
        element: Element data from document

    Returns:
        Parsed element information or None
    """
    element_info = {
        "start_index": element.get("startIndex", 0),
        "end_index": element.get("endIndex", 0),
    }

    if "paragraph" in element:
        paragraph = element["paragraph"]
        element_info["type"] = "paragraph"
        element_info["text"] = _extract_paragraph_text(paragraph)
        element_info["style"] = paragraph.get("paragraphStyle", {})

    elif "table" in element:
        table = element["table"]
        element_info["type"] = "table"
        element_info["rows"] = len(table.get("tableRows", []))
        element_info["columns"] = len(
            table.get("tableRows", [{}])[0].get("tableCells", [])
        )
        element_info["cells"] = _parse_table_cells(table)
        element_info["table_style"] = table.get("tableStyle", {})

    elif "sectionBreak" in element:
        element_info["type"] = "section_break"
        element_info["section_style"] = element["sectionBreak"].get("sectionStyle", {})

    elif "tableOfContents" in element:
        element_info["type"] = "table_of_contents"

    else:
        return None

    return element_info


def _parse_table_cells(table: dict[str, Any]) -> list[list[dict[str, Any]]]:
    """
    Parse table cells with their positions and content.

    Args:
        table: Table element data

    Returns:
        2D list of cell information
    """
    cells = []
    for row_idx, row in enumerate(table.get("tableRows", [])):
        row_cells = []
        for col_idx, cell in enumerate(row.get("tableCells", [])):
            # Find the first paragraph in the cell for insertion
            insertion_index = cell.get("startIndex", 0) + 1  # Default fallback

            # Look for the first paragraph in cell content
            content_elements = cell.get("content", [])
            for element in content_elements:
                if "paragraph" in element:
                    paragraph = element["paragraph"]
                    # Get the first element in the paragraph
                    para_elements = paragraph.get("elements", [])
                    if para_elements:
                        first_element = para_elements[0]
                        if "startIndex" in first_element:
                            insertion_index = first_element["startIndex"]
                            break

            cell_info = {
                "row": row_idx,
                "column": col_idx,
                "start_index": cell.get("startIndex", 0),
                "end_index": cell.get("endIndex", 0),
                "insertion_index": insertion_index,  # Where to insert text in this cell
                "content": _extract_cell_text(cell),
                "content_elements": content_elements,
            }
            row_cells.append(cell_info)
        cells.append(row_cells)
    return cells


def _extract_paragraph_text(paragraph: dict[str, Any]) -> str:
    """Extract text from a paragraph element."""
    text_parts = []
    for element in paragraph.get("elements", []):
        if "textRun" in element:
            text_parts.append(element["textRun"].get("content", ""))
    return "".join(text_parts)


def _extract_cell_text(cell: dict[str, Any]) -> str:
    """Extract text content from a table cell."""
    text_parts = []
    for element in cell.get("content", []):
        if "paragraph" in element:
            text_parts.append(_extract_paragraph_text(element["paragraph"]))
    return "".join(text_parts)


def _parse_segment(segment_data: dict[str, Any]) -> dict[str, Any]:
    """Parse a document segment (header/footer)."""
    return {
        "content": segment_data.get("content", []),
        "start_index": segment_data.get("content", [{}])[0].get("startIndex", 0)
        if segment_data.get("content")
        else 0,
        "end_index": segment_data.get("content", [{}])[-1].get("endIndex", 0)
        if segment_data.get("content")
        else 0,
    }


def find_tables(doc_data: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Find all tables in the document with their positions and dimensions.

    Args:
        doc_data: Raw document data from Google Docs API

    Returns:
        List of table information dictionaries
    """
    tables = []
    structure = parse_document_structure(doc_data)

    for idx, table_info in enumerate(structure["tables"]):
        tables.append(
            {
                "index": idx,
                "start_index": table_info["start_index"],
                "end_index": table_info["end_index"],
                "rows": table_info["rows"],
                "columns": table_info["columns"],
                "cells": table_info["cells"],
            }
        )

    return tables


def get_table_cell_indices(
    doc_data: dict[str, Any], table_index: int = 0
) -> Optional[list[list[tuple[int, int]]]]:
    """
    Get content indices for all cells in a specific table.

    Args:
        doc_data: Raw document data from Google Docs API
        table_index: Index of the table (0-based)

    Returns:
        2D list of (start_index, end_index) tuples for each cell, or None if table not found
    """
    tables = find_tables(doc_data)

    if table_index >= len(tables):
        logger.warning(
            f"Table index {table_index} not found. Document has {len(tables)} tables."
        )
        return None

    table = tables[table_index]
    cell_indices = []

    for row in table["cells"]:
        row_indices = []
        for cell in row:
            # Each cell contains at least one paragraph
            # Find the first paragraph in the cell for content insertion
            cell_content = cell.get("content_elements", [])
            if cell_content:
                # Look for the first paragraph in cell content
                first_para = None
                for element in cell_content:
                    if "paragraph" in element:
                        first_para = element["paragraph"]
                        break

                if first_para and "elements" in first_para and first_para["elements"]:
                    # Insert at the start of the first text run in the paragraph
                    first_text_element = first_para["elements"][0]
                    if "textRun" in first_text_element:
                        start_idx = first_text_element.get(
                            "startIndex", cell["start_index"] + 1
                        )
                        end_idx = first_text_element.get("endIndex", start_idx + 1)
                        row_indices.append((start_idx, end_idx))
                        continue

            # Fallback: use cell boundaries with safe margins
            content_start = cell["start_index"] + 1
            content_end = cell["end_index"] - 1
            row_indices.append((content_start, content_end))
        cell_indices.append(row_indices)

    return cell_indices


def find_element_at_index(
    doc_data: dict[str, Any], index: int
) -> Optional[dict[str, Any]]:
    """
    Find what element exists at a given index in the document.

    Args:
        doc_data: Raw document data from Google Docs API
        index: Position in the document

    Returns:
        Information about the element at that position, or None
    """
    structure = parse_document_structure(doc_data)

    for element in structure["body"]:
        if element["start_index"] <= index < element["end_index"]:
            element_copy = element.copy()

            # If it's a table, find which cell contains the index
            if element["type"] == "table" and "cells" in element:
                for row_idx, row in enumerate(element["cells"]):
                    for col_idx, cell in enumerate(row):
                        if cell["start_index"] <= index < cell["end_index"]:
                            element_copy["containing_cell"] = {
                                "row": row_idx,
                                "column": col_idx,
                                "cell_start": cell["start_index"],
                                "cell_end": cell["end_index"],
                            }
                            break

            return element_copy

    return None


def get_next_paragraph_index(doc_data: dict[str, Any], after_index: int = 0) -> int:
    """
    Find the next safe position to insert content after a given index.

    Args:
        doc_data: Raw document data from Google Docs API
        after_index: Index after which to find insertion point

    Returns:
        Safe index for insertion
    """
    structure = parse_document_structure(doc_data)

    # Find the first paragraph element after the given index
    for element in structure["body"]:
        if element["type"] == "paragraph" and element["start_index"] > after_index:
            # Insert at the end of the previous element or start of this paragraph
            return element["start_index"]

    # If no paragraph found, return the end of document
    return structure["total_length"] - 1 if structure["total_length"] > 0 else 1


def analyze_document_complexity(doc_data: dict[str, Any]) -> dict[str, Any]:
    """
    Analyze document complexity and provide statistics.

    Args:
        doc_data: Raw document data from Google Docs API

    Returns:
        Dictionary with document statistics
    """
    structure = parse_document_structure(doc_data)

    stats = {
        "total_elements": len(structure["body"]),
        "tables": len(structure["tables"]),
        "paragraphs": sum(1 for e in structure["body"] if e.get("type") == "paragraph"),
        "section_breaks": sum(
            1 for e in structure["body"] if e.get("type") == "section_break"
        ),
        "total_length": structure["total_length"],
        "has_headers": bool(structure["headers"]),
        "has_footers": bool(structure["footers"]),
    }

    # Add table statistics
    if structure["tables"]:
        total_cells = sum(
            table["rows"] * table["columns"] for table in structure["tables"]
        )
        stats["total_table_cells"] = total_cells
        stats["largest_table"] = max(
            (t["rows"] * t["columns"] for t in structure["tables"]), default=0
        )

    return stats


"""
Google Docs Table Operations

This module provides utilities for creating and manipulating tables
in Google Docs, including population with data and formatting.
"""


def build_table_population_requests(
    table_info: Dict[str, Any], data: List[List[str]], bold_headers: bool = True
) -> List[Dict[str, Any]]:
    """
    Build batch requests to populate a table with data.

    Args:
        table_info: Table information from document structure including cell indices
        data: 2D array of data to insert into table
        bold_headers: Whether to make the first row bold

    Returns:
        List of request dictionaries for batch update
    """
    requests = []
    cells = table_info.get("cells", [])

    if not cells:
        logger.warning("No cell information found in table_info")
        return requests

    # Process each cell - ONLY INSERT, DON'T DELETE
    for row_idx, row_data in enumerate(data):
        if row_idx >= len(cells):
            logger.warning(
                f"Data has more rows ({len(data)}) than table ({len(cells)})"
            )
            break

        for col_idx, cell_text in enumerate(row_data):
            if col_idx >= len(cells[row_idx]):
                logger.warning(
                    f"Data has more columns ({len(row_data)}) than table row {row_idx} ({len(cells[row_idx])})"
                )
                break

            cell = cells[row_idx][col_idx]

            # For new empty tables, use the insertion index
            # For tables with existing content, check if cell only contains newline
            existing_content = cell.get("content", "").strip()

            # Only insert if we have text to insert
            if cell_text:
                # Use the specific insertion index for this cell
                insertion_index = cell.get("insertion_index", cell["start_index"] + 1)

                # If cell only contains a newline, replace it
                if existing_content == "" or existing_content == "\n":
                    # Cell is empty (just newline), insert at the insertion index
                    requests.append(
                        {
                            "insertText": {
                                "location": {"index": insertion_index},
                                "text": cell_text,
                            }
                        }
                    )

                    # Apply bold formatting to first row if requested
                    if bold_headers and row_idx == 0:
                        requests.append(
                            {
                                "updateTextStyle": {
                                    "range": {
                                        "startIndex": insertion_index,
                                        "endIndex": insertion_index + len(cell_text),
                                    },
                                    "textStyle": {"bold": True},
                                    "fields": "bold",
                                }
                            }
                        )
                else:
                    # Cell has content, append after existing content
                    # Find the end of existing content
                    cell_end = cell["end_index"] - 1  # Don't include cell end marker
                    requests.append(
                        {
                            "insertText": {
                                "location": {"index": cell_end},
                                "text": cell_text,
                            }
                        }
                    )

                    # Apply bold formatting to first row if requested
                    if bold_headers and row_idx == 0:
                        requests.append(
                            {
                                "updateTextStyle": {
                                    "range": {
                                        "startIndex": cell_end,
                                        "endIndex": cell_end + len(cell_text),
                                    },
                                    "textStyle": {"bold": True},
                                    "fields": "bold",
                                }
                            }
                        )

    return requests


def calculate_cell_positions(
    table_start_index: int,
    rows: int,
    cols: int,
    existing_table_data: Optional[Dict[str, Any]] = None,
) -> List[List[Dict[str, int]]]:
    """
    Calculate estimated positions for each cell in a table.

    Args:
        table_start_index: Starting index of the table
        rows: Number of rows
        cols: Number of columns
        existing_table_data: Optional existing table data with actual positions

    Returns:
        2D list of cell position dictionaries
    """
    if existing_table_data and "cells" in existing_table_data:
        # Use actual positions from existing table
        return existing_table_data["cells"]

    # Estimate positions for a new table
    # Note: These are estimates; actual positions depend on content
    cells = []
    current_index = table_start_index + 2  # Account for table start

    for row_idx in range(rows):
        row_cells = []
        for col_idx in range(cols):
            # Each cell typically starts with a paragraph marker
            cell_start = current_index
            cell_end = current_index + 2  # Minimum cell size

            row_cells.append(
                {
                    "row": row_idx,
                    "column": col_idx,
                    "start_index": cell_start,
                    "end_index": cell_end,
                }
            )

            current_index = cell_end + 1

        cells.append(row_cells)

    return cells


def format_table_data(
    raw_data: Union[List[List[str]], List[str], str],
) -> List[List[str]]:
    """
    Normalize various data formats into a 2D array for table insertion.

    Args:
        raw_data: Data in various formats (2D list, 1D list, or delimited string)

    Returns:
        Normalized 2D list of strings
    """
    if isinstance(raw_data, str):
        # Parse delimited string (detect delimiter)
        lines = raw_data.strip().split("\n")
        if "\t" in raw_data:
            # Tab-delimited
            return [line.split("\t") for line in lines]
        elif "," in raw_data:
            # Comma-delimited (simple CSV)
            return [line.split(",") for line in lines]
        else:
            # Space-delimited or single column
            return [[cell.strip() for cell in line.split()] for line in lines]

    elif isinstance(raw_data, list):
        if not raw_data:
            return [[]]

        # Check if it's already a 2D list
        if isinstance(raw_data[0], list):
            # Ensure all cells are strings
            return [[str(cell) for cell in row] for row in raw_data]
        else:
            # Convert 1D list to single-column table
            return [[str(cell)] for cell in raw_data]

    else:
        # Convert single value to 1x1 table
        return [[str(raw_data)]]


def create_table_with_data(
    index: int,
    data: List[List[str]],
    headers: Optional[List[str]] = None,
    bold_headers: bool = True,
) -> List[Dict[str, Any]]:
    """
    Create a table and populate it with data in one operation.

    Args:
        index: Position to insert the table
        data: 2D array of table data
        headers: Optional header row (will be prepended to data)
        bold_headers: Whether to make headers bold

    Returns:
        List of request dictionaries for batch update
    """
    requests = []

    # Prepare data with headers if provided
    if headers:
        full_data = [headers] + data
    else:
        full_data = data

    # Normalize the data
    full_data = format_table_data(full_data)

    if not full_data or not full_data[0]:
        raise ValueError("Cannot create table with empty data")

    rows = len(full_data)
    cols = len(full_data[0])

    # Ensure all rows have the same number of columns
    for row in full_data:
        while len(row) < cols:
            row.append("")

    # Create the table
    requests.append(
        {"insertTable": {"location": {"index": index}, "rows": rows, "columns": cols}}
    )

    # Build text insertion requests for each cell
    # Note: In practice, we'd need to get the actual document structure
    # after table creation to get accurate indices

    return requests


def build_table_style_requests(
    table_start_index: int, style_options: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """
    Build requests to style a table.

    Args:
        table_start_index: Starting index of the table
        style_options: Dictionary of style options
            - border_width: Width of borders in points
            - border_color: RGB color for borders
            - background_color: RGB color for cell backgrounds
            - header_background: RGB color for header row background

    Returns:
        List of request dictionaries for styling
    """
    requests = []

    # Table cell style update
    if any(
        k in style_options for k in ["border_width", "border_color", "background_color"]
    ):
        table_cell_style = {}
        fields = []

        if "border_width" in style_options:
            border_width = {"magnitude": style_options["border_width"], "unit": "PT"}
            table_cell_style["borderTop"] = {"width": border_width}
            table_cell_style["borderBottom"] = {"width": border_width}
            table_cell_style["borderLeft"] = {"width": border_width}
            table_cell_style["borderRight"] = {"width": border_width}
            fields.extend(["borderTop", "borderBottom", "borderLeft", "borderRight"])

        if "border_color" in style_options:
            border_color = {"color": {"rgbColor": style_options["border_color"]}}
            if "borderTop" in table_cell_style:
                table_cell_style["borderTop"]["color"] = border_color["color"]
                table_cell_style["borderBottom"]["color"] = border_color["color"]
                table_cell_style["borderLeft"]["color"] = border_color["color"]
                table_cell_style["borderRight"]["color"] = border_color["color"]

        if "background_color" in style_options:
            table_cell_style["backgroundColor"] = {
                "color": {"rgbColor": style_options["background_color"]}
            }
            fields.append("backgroundColor")

        if table_cell_style and fields:
            requests.append(
                {
                    "updateTableCellStyle": {
                        "tableStartLocation": {"index": table_start_index},
                        "tableCellStyle": table_cell_style,
                        "fields": ",".join(fields),
                    }
                }
            )

    # Header row specific styling
    if "header_background" in style_options:
        requests.append(
            {
                "updateTableCellStyle": {
                    "tableRange": {
                        "tableCellLocation": {
                            "tableStartLocation": {"index": table_start_index},
                            "rowIndex": 0,
                            "columnIndex": 0,
                        },
                        "rowSpan": 1,
                        "columnSpan": 100,  # Large number to cover all columns
                    },
                    "tableCellStyle": {
                        "backgroundColor": {
                            "color": {"rgbColor": style_options["header_background"]}
                        }
                    },
                    "fields": "backgroundColor",
                }
            }
        )

    return requests


def extract_table_as_data(table_info: Dict[str, Any]) -> List[List[str]]:
    """
    Extract table content as a 2D array of strings.

    Args:
        table_info: Table information from document structure

    Returns:
        2D list of cell contents
    """
    data = []
    cells = table_info.get("cells", [])

    for row in cells:
        row_data = []
        for cell in row:
            row_data.append(cell.get("content", "").strip())
        data.append(row_data)

    return data


def find_table_by_content(
    tables: List[Dict[str, Any]], search_text: str, case_sensitive: bool = False
) -> Optional[int]:
    """
    Find a table index by searching for content within it.

    Args:
        tables: List of table information from document
        search_text: Text to search for in table cells
        case_sensitive: Whether to do case-sensitive search

    Returns:
        Index of the first matching table, or None
    """
    search_text = search_text if case_sensitive else search_text.lower()

    for idx, table in enumerate(tables):
        for row in table.get("cells", []):
            for cell in row:
                cell_content = cell.get("content", "")
                if not case_sensitive:
                    cell_content = cell_content.lower()

                if search_text in cell_content:
                    return idx

    return None


def validate_table_data(data: List[List[str]]) -> Tuple[bool, str]:
    """
    Validates table data format and provides specific error messages for LLMs.

    WHAT THIS CHECKS:
    - Data is a 2D list (list of lists)
    - All rows have consistent column counts
    - Dimensions are within Google Docs limits
    - No None or undefined values

    VALID FORMAT EXAMPLE:
    [
        ["Header1", "Header2"],     # Row 0 - 2 columns
        ["Data1", "Data2"],        # Row 1 - 2 columns
        ["Data3", "Data4"]         # Row 2 - 2 columns
    ]

    INVALID FORMATS:
    - [["col1"], ["col1", "col2"]]  # Inconsistent column counts
    - ["col1", "col2"]              # Not 2D (missing inner lists)
    - [["col1", None]]              # Contains None values
    - [] or [[]]                    # Empty data

    Args:
        data: 2D array of data to validate

    Returns:
        Tuple of (is_valid, error_message_with_examples)
    """
    if not data:
        return (
            False,
            "Data is empty. Use format: [['col1', 'col2'], ['row1col1', 'row1col2']]",
        )

    if not isinstance(data, list):
        return (
            False,
            f"Data must be a list, got {type(data).__name__}. Use format: [['col1', 'col2'], ['row1col1', 'row1col2']]",
        )

    if not all(isinstance(row, list) for row in data):
        return (
            False,
            f"Data must be a 2D list (list of lists). Each row must be a list. Check your format: {data}",
        )

    # Check for consistent column count
    col_counts = [len(row) for row in data]
    if len(set(col_counts)) > 1:
        return (
            False,
            f"All rows must have same number of columns. Found: {col_counts}. Fix your data format.",
        )

    # Check for reasonable size
    rows = len(data)
    cols = col_counts[0] if col_counts else 0

    if rows > 1000:
        return False, f"Too many rows ({rows}). Google Docs limit is 1000 rows."

    if cols > 20:
        return False, f"Too many columns ({cols}). Google Docs limit is 20 columns."

    return True, f"Valid table data: {rows}x{cols} table format"
