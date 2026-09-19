"""
Google Sheets Helper Functions

Shared utilities for Google Sheets operations including A1 parsing and
conditional formatting helpers.
"""

import asyncio
import json
import re
from typing import List, Optional, Union
from pathlib import Path
import copy
from app_tools.core.server_init import content_server
from app_tools.tools.google.workspace_comment_base import create_comment_tools
from app_tools.helper.utils import UserInputError
from app_tools.auth.service_decoder import get_google_service
from app_tools.helper.pydantic_models import (
    ListSpreadsheetsRequest,
    ListSpreadsheetsResponse,
    SpreadsheetInfo,
    GetSpreadsheetInfoRequest,
    GetSpreadsheetInfoResponse,
    ReadSheetValuesRequest,
    ReadSheetValuesResponse,
    ModifySheetValuesRequest,
    ModifySheetValuesResponse,
    FormatSheetRangeRequest,
    FormatSheetRangeResponse,
    AddConditionalFormattingRequest,
    AddConditionalFormattingResponse,
    UpdateConditionalFormattingRequest,
    UpdateConditionalFormattingResponse,
    DeleteConditionalFormattingRequest,
    DeleteConditionalFormattingResponse,
    CreateSpreadsheetRequest,
    CreateSpreadsheetResponse,
    CreateSheetRequest,
    CreateSheetResponse,
)
from utils.helper import setup_logger

logger = setup_logger(__name__)


A1_PART_REGEX = re.compile(r"^([A-Za-z]*)(\d*)$")


def get_service():
    """Get Gmail service using shared authentication."""
    base_dir = Path(__file__).parent.parent.parent
    token_path = str(base_dir / "cred" / "gsheet_token.json")
    creds_path = str(base_dir / "cred" / "setup_cred.json")

    return get_google_service(
        service_type="sheets",
        scope_key="sheets",
        token_path=token_path,
        creds_path=creds_path,
    )


def _column_to_index(column: str) -> Optional[int]:
    """Convert column letters (A, B, AA) to zero-based index."""
    if not column:
        return None
    result = 0
    for char in column.upper():
        result = result * 26 + (ord(char) - ord("A") + 1)
    return result - 1


def _parse_a1_part(
    part: str, pattern: re.Pattern[str] = A1_PART_REGEX
) -> tuple[Optional[int], Optional[int]]:
    """
    Parse a single A1 part like 'B2' or 'C' into zero-based column/row indexes.
    Supports anchors like '$A$1' by stripping the dollar signs.
    """
    clean_part = part.replace("$", "")
    match = pattern.match(clean_part)
    if not match:
        raise UserInputError(f"Invalid A1 range part: '{part}'.")
    col_letters, row_digits = match.groups()
    col_idx = _column_to_index(col_letters) if col_letters else None
    row_idx = int(row_digits) - 1 if row_digits else None
    return col_idx, row_idx


def _split_sheet_and_range(range_name: str) -> tuple[Optional[str], str]:
    """
    Split an A1 notation into (sheet_name, range_part), handling quoted sheet names.

    Examples:
    - "Sheet1!A1:B2" -> ("Sheet1", "A1:B2")
    - "'My Sheet'!$A$1:$B$10" -> ("My Sheet", "$A$1:$B$10")
    - "A1:B2" -> (None, "A1:B2")
    """
    if "!" not in range_name:
        return None, range_name

    if range_name.startswith("'"):
        closing = range_name.find("'!")
        if closing != -1:
            sheet_name = range_name[1:closing].replace("''", "'")
            a1_range = range_name[closing + 2 :]
            return sheet_name, a1_range

    sheet_name, a1_range = range_name.split("!", 1)
    return sheet_name.strip().strip("'"), a1_range


def _parse_a1_range(range_name: str, sheets: List[dict]) -> dict:
    """
    Convert an A1-style range (with optional sheet name) into a GridRange.

    Falls back to the first sheet if none is provided.
    """
    sheet_name, a1_range = _split_sheet_and_range(range_name)

    if not sheets:
        raise UserInputError("Spreadsheet has no sheets.")

    target_sheet = None
    if sheet_name:
        for sheet in sheets:
            if sheet.get("properties", {}).get("title") == sheet_name:
                target_sheet = sheet
                break
        if target_sheet is None:
            available_titles = [
                sheet.get("properties", {}).get("title", "Untitled") for sheet in sheets
            ]
            available_list = ", ".join(available_titles) if available_titles else "none"
            raise UserInputError(
                f"Sheet '{sheet_name}' not found in spreadsheet. Available sheets: {available_list}."
            )
    else:
        target_sheet = sheets[0]

    props = target_sheet.get("properties", {})
    sheet_id = props.get("sheetId")

    if not a1_range:
        raise UserInputError("A1-style range must not be empty (e.g., 'A1', 'A1:B10').")

    if ":" in a1_range:
        start, end = a1_range.split(":", 1)
    else:
        start = end = a1_range

    start_col, start_row = _parse_a1_part(start)
    end_col, end_row = _parse_a1_part(end)

    grid_range = {"sheetId": sheet_id}
    if start_row is not None:
        grid_range["startRowIndex"] = start_row
    if start_col is not None:
        grid_range["startColumnIndex"] = start_col
    if end_row is not None:
        grid_range["endRowIndex"] = end_row + 1
    if end_col is not None:
        grid_range["endColumnIndex"] = end_col + 1

    return grid_range


def _parse_hex_color(color: Optional[str]) -> Optional[dict]:
    """
    Convert a hex color like '#RRGGBB' to Sheets API color (0-1 floats).
    """
    if not color:
        return None

    trimmed = color.strip()
    if trimmed.startswith("#"):
        trimmed = trimmed[1:]

    if len(trimmed) != 6:
        raise UserInputError(f"Color '{color}' must be in format #RRGGBB or RRGGBB.")

    try:
        red = int(trimmed[0:2], 16) / 255
        green = int(trimmed[2:4], 16) / 255
        blue = int(trimmed[4:6], 16) / 255
    except ValueError as exc:
        raise UserInputError(f"Color '{color}' is not valid hex.") from exc

    return {"red": red, "green": green, "blue": blue}


def _index_to_column(index: int) -> str:
    """
    Convert a zero-based column index to column letters (0 -> A, 25 -> Z, 26 -> AA).
    """
    if index < 0:
        raise UserInputError(f"Column index must be non-negative, got {index}.")

    result = []
    index += 1  # Convert to 1-based for calculation
    while index:
        index, remainder = divmod(index - 1, 26)
        result.append(chr(ord("A") + remainder))
    return "".join(reversed(result))


def _color_to_hex(color: Optional[dict]) -> Optional[str]:
    """
    Convert a Sheets color object back to #RRGGBB hex string for display.
    """
    if not color:
        return None

    def _component(value: Optional[float]) -> int:
        try:
            # Clamp and round to nearest integer in 0-255
            return max(0, min(255, int(round(float(value or 0) * 255))))
        except (TypeError, ValueError):
            return 0

    red = _component(color.get("red"))
    green = _component(color.get("green"))
    blue = _component(color.get("blue"))
    return f"#{red:02X}{green:02X}{blue:02X}"


def _grid_range_to_a1(grid_range: dict, sheet_titles: dict[int, str]) -> str:
    """
    Convert a GridRange to an A1-like string using known sheet titles.
    Falls back to the sheet ID if the title is unknown.
    """
    sheet_id = grid_range.get("sheetId")
    sheet_title = sheet_titles.get(sheet_id, f"Sheet {sheet_id}")

    start_row = grid_range.get("startRowIndex")
    end_row = grid_range.get("endRowIndex")
    start_col = grid_range.get("startColumnIndex")
    end_col = grid_range.get("endColumnIndex")

    # If nothing is specified, treat as the whole sheet.
    if start_row is None and end_row is None and start_col is None and end_col is None:
        return sheet_title

    def row_label(idx: Optional[int]) -> str:
        return str(idx + 1) if idx is not None else ""

    def col_label(idx: Optional[int]) -> str:
        return _index_to_column(idx) if idx is not None else ""

    start_label = f"{col_label(start_col)}{row_label(start_row)}"
    # end indices in GridRange are exclusive; subtract 1 for display
    end_label = f"{col_label(end_col - 1 if end_col is not None else None)}{row_label(end_row - 1 if end_row is not None else None)}"

    if start_label and end_label:
        range_ref = (
            start_label if start_label == end_label else f"{start_label}:{end_label}"
        )
    elif start_label:
        range_ref = start_label
    elif end_label:
        range_ref = end_label
    else:
        range_ref = ""

    return f"{sheet_title}!{range_ref}" if range_ref else sheet_title


def _summarize_conditional_rule(
    rule: dict, index: int, sheet_titles: dict[int, str]
) -> str:
    """
    Produce a concise human-readable summary of a conditional formatting rule.
    """
    ranges = rule.get("ranges", [])
    range_labels = [_grid_range_to_a1(rng, sheet_titles) for rng in ranges] or [
        "(no range)"
    ]

    if "booleanRule" in rule:
        boolean_rule = rule["booleanRule"]
        condition = boolean_rule.get("condition", {})
        cond_type = condition.get("type", "UNKNOWN")
        cond_values = [
            val.get("userEnteredValue")
            for val in condition.get("values", [])
            if isinstance(val, dict) and "userEnteredValue" in val
        ]
        value_desc = f" values={cond_values}" if cond_values else ""

        fmt = boolean_rule.get("format", {})
        fmt_parts = []
        bg_hex = _color_to_hex(fmt.get("backgroundColor"))
        if bg_hex:
            fmt_parts.append(f"bg {bg_hex}")
        fg_hex = _color_to_hex(fmt.get("textFormat", {}).get("foregroundColor"))
        if fg_hex:
            fmt_parts.append(f"text {fg_hex}")
        fmt_desc = ", ".join(fmt_parts) if fmt_parts else "no format"

        return f"[{index}] {cond_type}{value_desc} -> {fmt_desc} on {', '.join(range_labels)}"

    if "gradientRule" in rule:
        gradient_rule = rule["gradientRule"]
        points = []
        for point_name in ("minpoint", "midpoint", "maxpoint"):
            point = gradient_rule.get(point_name)
            if not point:
                continue
            color_hex = _color_to_hex(point.get("color"))
            type_desc = point.get("type", point_name)
            value_desc = point.get("value")
            point_desc = type_desc
            if value_desc:
                point_desc += f":{value_desc}"
            if color_hex:
                point_desc += f" {color_hex}"
            points.append(point_desc)
        gradient_desc = " | ".join(points) if points else "gradient"
        return f"[{index}] gradient -> {gradient_desc} on {', '.join(range_labels)}"

    return f"[{index}] (unknown rule) on {', '.join(range_labels)}"


def _format_conditional_rules_section(
    sheet_title: str,
    rules: List[dict],
    sheet_titles: dict[int, str],
    indent: str = "  ",
) -> str:
    """
    Build a multi-line string describing conditional formatting rules for a sheet.
    """
    if not rules:
        return f'{indent}Conditional formats for "{sheet_title}": none.'

    lines = [f'{indent}Conditional formats for "{sheet_title}" ({len(rules)}):']
    for idx, rule in enumerate(rules):
        lines.append(
            f"{indent}  {_summarize_conditional_rule(rule, idx, sheet_titles)}"
        )
    return "\n".join(lines)


CONDITION_TYPES = {
    "NUMBER_GREATER",
    "NUMBER_GREATER_THAN_EQ",
    "NUMBER_LESS",
    "NUMBER_LESS_THAN_EQ",
    "NUMBER_EQ",
    "NUMBER_NOT_EQ",
    "TEXT_CONTAINS",
    "TEXT_NOT_CONTAINS",
    "TEXT_STARTS_WITH",
    "TEXT_ENDS_WITH",
    "TEXT_EQ",
    "DATE_BEFORE",
    "DATE_ON_OR_BEFORE",
    "DATE_AFTER",
    "DATE_ON_OR_AFTER",
    "DATE_EQ",
    "DATE_NOT_EQ",
    "DATE_BETWEEN",
    "DATE_NOT_BETWEEN",
    "NOT_BLANK",
    "BLANK",
    "CUSTOM_FORMULA",
    "ONE_OF_RANGE",
}

GRADIENT_POINT_TYPES = {"MIN", "MAX", "NUMBER", "PERCENT", "PERCENTILE"}


async def _fetch_sheets_with_rules(
    service, spreadsheet_id: str
) -> tuple[List[dict], dict[int, str]]:
    """
    Fetch sheets with titles and conditional format rules in a single request.
    """
    response = await asyncio.to_thread(
        service.spreadsheets()
        .get(
            spreadsheetId=spreadsheet_id,
            fields="sheets(properties(sheetId,title),conditionalFormats)",
        )
        .execute
    )
    sheets = response.get("sheets", []) or []
    sheet_titles: dict[int, str] = {}
    for sheet in sheets:
        props = sheet.get("properties", {})
        sid = props.get("sheetId")
        if sid is not None:
            sheet_titles[sid] = props.get("title", f"Sheet {sid}")
    return sheets, sheet_titles


def _select_sheet(sheets: List[dict], sheet_name: Optional[str]) -> dict:
    """
    Select a sheet by name, or default to the first sheet if name is not provided.
    """
    if not sheets:
        raise UserInputError("Spreadsheet has no sheets.")

    if sheet_name is None:
        return sheets[0]

    for sheet in sheets:
        if sheet.get("properties", {}).get("title") == sheet_name:
            return sheet

    available_titles = [
        sheet.get("properties", {}).get("title", "Untitled") for sheet in sheets
    ]
    raise UserInputError(
        f"Sheet '{sheet_name}' not found. Available sheets: {', '.join(available_titles)}."
    )


def _parse_condition_values(
    condition_values: Optional[Union[str, List[Union[str, int, float]]]],
) -> Optional[List[Union[str, int, float]]]:
    """
    Normalize and validate condition_values into a list of strings/numbers.
    """
    parsed = condition_values
    if isinstance(parsed, str):
        try:
            parsed = json.loads(parsed)
        except json.JSONDecodeError as exc:
            raise UserInputError(
                "condition_values must be a list or a JSON-encoded list (e.g., '[\"=$B2>1000\"]')."
            ) from exc

    if parsed is not None and not isinstance(parsed, list):
        parsed = [parsed]

    if parsed:
        for idx, val in enumerate(parsed):
            if not isinstance(val, (str, int, float)):
                raise UserInputError(
                    f"condition_values[{idx}] must be a string or number, got {type(val).__name__}."
                )

    return parsed


def _parse_gradient_points(
    gradient_points: Optional[Union[str, List[dict]]],
) -> Optional[List[dict]]:
    """
    Normalize gradient points into a list of dicts with type/value/color.
    Each point must have a 'type' (MIN, MAX, NUMBER, PERCENT, PERCENTILE) and a color.
    """
    if gradient_points is None:
        return None

    parsed = gradient_points
    if isinstance(parsed, str):
        try:
            parsed = json.loads(parsed)
        except json.JSONDecodeError as exc:
            raise UserInputError(
                "gradient_points must be a list or JSON-encoded list of points "
                '(e.g., \'[{"type":"MIN","color":"#ffffff"}, {"type":"MAX","color":"#ff0000"}]\').'
            ) from exc

    if not isinstance(parsed, list):
        raise UserInputError("gradient_points must be a list of point objects.")

    if len(parsed) < 2 or len(parsed) > 3:
        raise UserInputError("Provide 2 or 3 gradient points (min/max or min/mid/max).")

    normalized_points: List[dict] = []
    for idx, point in enumerate(parsed):
        if not isinstance(point, dict):
            raise UserInputError(
                f"gradient_points[{idx}] must be an object with type/color."
            )

        point_type = point.get("type")
        if not point_type or point_type.upper() not in GRADIENT_POINT_TYPES:
            raise UserInputError(
                f"gradient_points[{idx}].type must be one of {sorted(GRADIENT_POINT_TYPES)}."
            )
        color_raw = point.get("color")
        color_dict = (
            _parse_hex_color(color_raw)
            if not isinstance(color_raw, dict)
            else color_raw
        )
        if not color_dict:
            raise UserInputError(f"gradient_points[{idx}].color is required.")

        normalized = {"type": point_type.upper(), "color": color_dict}
        if "value" in point and point["value"] is not None:
            normalized["value"] = str(point["value"])
        normalized_points.append(normalized)

    return normalized_points


def _build_boolean_rule(
    ranges: List[dict],
    condition_type: str,
    condition_values: Optional[List[Union[str, int, float]]],
    background_color: Optional[str],
    text_color: Optional[str],
) -> tuple[dict, str]:
    """
    Build a Sheets boolean conditional formatting rule payload.
    Returns the rule and the normalized condition type.
    """
    if not background_color and not text_color:
        raise UserInputError(
            "Provide at least one of background_color or text_color for the rule format."
        )

    cond_type_normalized = condition_type.upper()
    if cond_type_normalized not in CONDITION_TYPES:
        raise UserInputError(
            f"condition_type must be one of {sorted(CONDITION_TYPES)}."
        )

    condition = {"type": cond_type_normalized}
    if condition_values:
        condition["values"] = [
            {"userEnteredValue": str(value)} for value in condition_values
        ]

    bg_color_parsed = _parse_hex_color(background_color)
    text_color_parsed = _parse_hex_color(text_color)

    format_obj = {}
    if bg_color_parsed:
        format_obj["backgroundColor"] = bg_color_parsed
    if text_color_parsed:
        format_obj["textFormat"] = {"foregroundColor": text_color_parsed}

    return (
        {
            "ranges": ranges,
            "booleanRule": {
                "condition": condition,
                "format": format_obj,
            },
        },
        cond_type_normalized,
    )


def _build_gradient_rule(
    ranges: List[dict],
    gradient_points: List[dict],
) -> dict:
    """
    Build a Sheets gradient conditional formatting rule payload.
    """
    rule_body: dict = {"ranges": ranges, "gradientRule": {}}
    if len(gradient_points) == 2:
        rule_body["gradientRule"]["minpoint"] = gradient_points[0]
        rule_body["gradientRule"]["maxpoint"] = gradient_points[1]
    else:
        rule_body["gradientRule"]["minpoint"] = gradient_points[0]
        rule_body["gradientRule"]["midpoint"] = gradient_points[1]
        rule_body["gradientRule"]["maxpoint"] = gradient_points[2]
    return rule_body


"""
Google Sheets Tools

This module provides tools for interacting with Google Sheets API.
"""


@content_server.tool()
async def list_spreadsheets(
    max_results: int = 25,
) -> str:
    """
    Lists spreadsheets from Google Drive that the user has access to.

    Args:
        max_results (int): Maximum number of spreadsheets to return. Defaults to 25.

    Returns:
        str: A formatted list of spreadsheet files (name, ID, modified time).
    """
    # Validate input parameters
    try:
        request = ListSpreadsheetsRequest(max_results=max_results)
    except Exception as e:
        error_msg = f"Invalid parameters: {str(e)}"
        logger.error(f"[list_spreadsheets] {error_msg}")
        return ListSpreadsheetsResponse(
            status="error", message="", count=0, spreadsheets=[], error=error_msg
        ).model_dump_json(indent=2)

    service = get_service()
    logger.info("[list_spreadsheets] Invoked.")

    try:
        files_response = await asyncio.to_thread(
            service.files()
            .list(
                q="mimeType='application/vnd.google-apps.spreadsheet'",
                pageSize=request.max_results,
                fields="files(id,name,modifiedTime,webViewLink)",
                orderBy="modifiedTime desc",
                supportsAllDrives=True,
                includeItemsFromAllDrives=True,
            )
            .execute
        )

        files = files_response.get("files", [])
        if not files:
            message = "No spreadsheets found."
            logger.info(message)
            return ListSpreadsheetsResponse(
                status="success", message=message, count=0, spreadsheets=[]
            ).model_dump_json(indent=2)

        # Convert to pydantic models
        spreadsheets = [
            SpreadsheetInfo(
                id=file["id"],
                name=file["name"],
                modified_time=file.get("modifiedTime", "Unknown"),
                web_view_link=file.get("webViewLink"),
            )
            for file in files
        ]

        spreadsheets_list = [
            f'- "{s.name}" (ID: {s.id}) | Modified: {s.modified_time} | Link: {s.web_view_link or "No link"}'
            for s in spreadsheets
        ]

        message = f"Successfully listed {len(files)} spreadsheets:\n" + "\n".join(
            spreadsheets_list
        )

        logger.info(f"Successfully listed {len(files)} spreadsheets.")
        return ListSpreadsheetsResponse(
            status="success",
            message=message,
            count=len(files),
            spreadsheets=spreadsheets,
        ).model_dump_json(indent=2)

    except Exception as error:
        error_msg = f"Error listing spreadsheets: {str(error)}"
        logger.error(f"[list_spreadsheets] {error_msg}")
        return ListSpreadsheetsResponse(
            status="error", message="", count=0, spreadsheets=[], error=error_msg
        ).model_dump_json(indent=2)


@content_server.tool()
async def get_spreadsheet_info(
    spreadsheet_id: str,
) -> str:
    """
    Gets information about a specific spreadsheet including its sheets.

    Args:
        spreadsheet_id (str): The ID of the spreadsheet to get info for. Required.

    Returns:
        str: Formatted spreadsheet information including title, locale, and sheets list.
    """
    # Validate input parameters
    try:
        request = GetSpreadsheetInfoRequest(spreadsheet_id=spreadsheet_id)
    except Exception as e:
        error_msg = f"Invalid parameters: {str(e)}"
        logger.error(f"[get_spreadsheet_info] {error_msg}")
        return GetSpreadsheetInfoResponse(
            status="error", message="", error=error_msg
        ).model_dump_json(indent=2)

    service = get_service()
    logger.info(
        f"[get_spreadsheet_info] Invoked. Spreadsheet ID: {request.spreadsheet_id}"
    )

    try:
        spreadsheet = await asyncio.to_thread(
            service.spreadsheets()
            .get(
                spreadsheetId=spreadsheet_id,
                fields="spreadsheetId,properties(title,locale),sheets(properties(title,sheetId,gridProperties(rowCount,columnCount)),conditionalFormats)",
            )
            .execute
        )

        properties = spreadsheet.get("properties", {})
        title = properties.get("title", "Unknown")
        locale = properties.get("locale", "Unknown")
        sheets = spreadsheet.get("sheets", [])

        sheet_titles = {}
        for sheet in sheets:
            sheet_props = sheet.get("properties", {})
            sid = sheet_props.get("sheetId")
            if sid is not None:
                sheet_titles[sid] = sheet_props.get("title", f"Sheet {sid}")

        sheets_info = []
        for sheet in sheets:
            sheet_props = sheet.get("properties", {})
            sheet_name = sheet_props.get("title", "Unknown")
            sheet_id = sheet_props.get("sheetId", "Unknown")
            grid_props = sheet_props.get("gridProperties", {})
            rows = grid_props.get("rowCount", "Unknown")
            cols = grid_props.get("columnCount", "Unknown")
            rules = sheet.get("conditionalFormats", []) or []

            sheets_info.append(
                f'  - "{sheet_name}" (ID: {sheet_id}) | Size: {rows}x{cols} | Conditional formats: {len(rules)}'
            )
            if rules:
                sheets_info.append(
                    _format_conditional_rules_section(
                        sheet_name, rules, sheet_titles, indent="    "
                    )
                )

            sheets_section = (
                "\n".join(sheets_info) if sheets_info else "  No sheets found"
            )
            message = "\n".join(
                [
                    f'Spreadsheet: "{title}" (ID: {request.spreadsheet_id}) | Locale: {locale}',
                    f"Sheets ({len(sheets)}):",
                    sheets_section,
                ]
            )

            logger.info(
                f"Successfully retrieved info for spreadsheet {request.spreadsheet_id}."
            )
            return GetSpreadsheetInfoResponse(
                status="success", message=message
            ).model_dump_json(indent=2)

    except Exception as error:
        error_msg = f"Error getting spreadsheet info: {str(error)}"
        logger.error(f"[get_spreadsheet_info] {error_msg}")
        return GetSpreadsheetInfoResponse(
            status="error", message="", error=error_msg
        ).model_dump_json(indent=2)


@content_server.tool()
async def read_sheet_values(
    spreadsheet_id: str,
    range_name: str = "A1:Z1000",
) -> str:
    """
    Reads values from a specific range in a Google Sheet.

    Args:
        spreadsheet_id (str): The ID of the spreadsheet. Required.
        range_name (str): The range to read (e.g., "Sheet1!A1:D10", "A1:D10"). Defaults to "A1:Z1000".

    Returns:
        str: The formatted values from the specified range.
    """
    # Validate input parameters
    try:
        request = ReadSheetValuesRequest(
            spreadsheet_id=spreadsheet_id, range_name=range_name
        )
    except Exception as e:
        error_msg = f"Invalid parameters: {str(e)}"
        logger.error(f"[read_sheet_values] {error_msg}")
        return ReadSheetValuesResponse(
            status="error", message="", error=error_msg
        ).model_dump_json(indent=2)

    logger.info(
        f"[read_sheet_values] Invoked. Spreadsheet: {request.spreadsheet_id}, Range: {request.range_name}"
    )
    service = get_service()

    try:
        result = await asyncio.to_thread(
            service.spreadsheets()
            .values()
            .get(spreadsheetId=request.spreadsheet_id, range=request.range_name)
            .execute
        )

        values = result.get("values", [])
        if not values:
            message = f"No data found in range '{request.range_name}'."
            logger.info(message)
            return ReadSheetValuesResponse(
                status="success", message=message
            ).model_dump_json(indent=2)

        # Resolve the actual sheet row number the range starts at, so labels
        # reflect real sheet coordinates instead of always starting at 1 —
        # e.g. reading "Sheet1!A10:D15" should label the first row "Row 10",
        # not "Row 1" (which would mislead a caller into writing back to the
        # wrong row).
        start_row_number = 1
        try:
            _, a1_range = _split_sheet_and_range(request.range_name)
            if a1_range:
                start_part = a1_range.split(":", 1)[0]
                _, start_row_idx = _parse_a1_part(start_part)
                if start_row_idx is not None:
                    start_row_number = start_row_idx + 1
        except UserInputError:
            pass

        # Format the output as a readable table
        formatted_rows = []
        for offset, row in enumerate(values):
            # Pad row with empty strings to show structure
            padded_row = (
                row + [""] * max(0, len(values[0]) - len(row)) if values else row
            )
            formatted_rows.append(f"Row {start_row_number + offset:2d}: {padded_row}")

        message = (
            f"Successfully read {len(values)} rows from range '{request.range_name}' in spreadsheet {request.spreadsheet_id}:\n"
            + "\n".join(formatted_rows[:50])  # Limit to first 50 rows for readability
            + (f"\n... and {len(values) - 50} more rows" if len(values) > 50 else "")
        )

        logger.info(f"Successfully read {len(values)} rows.")
        return ReadSheetValuesResponse(
            status="success", message=message
        ).model_dump_json(indent=2)

    except Exception as error:
        error_msg = f"Error reading sheet values: {str(error)}"
        logger.error(f"[read_sheet_values] {error_msg}")
        return ReadSheetValuesResponse(
            status="error", message="", error=error_msg
        ).model_dump_json(indent=2)


@content_server.tool()
async def modify_sheet_values(
    spreadsheet_id: str,
    range_name: str,
    values: Optional[Union[str, List[List[str]]]] = None,
    value_input_option: str = "USER_ENTERED",
    clear_values: bool = False,
) -> str:
    """
    Modifies values in a specific range of a Google Sheet - can write, update, or clear values.

    Args:
        spreadsheet_id (str): The ID of the spreadsheet. Required.
        range_name (str): The range to modify (e.g., "Sheet1!A1:D10", "A1:D10"). Required.
        values (Optional[Union[str, List[List[str]]]]): 2D array of values to write/update. Can be a JSON string or Python list. Required unless clear_values=True.
        value_input_option (str): How to interpret input values ("RAW" or "USER_ENTERED"). Defaults to "USER_ENTERED".
        clear_values (bool): If True, clears the range instead of writing values. Defaults to False.

    Returns:
        str: Confirmation message of the successful modification operation.
    """
    # Validate input parameters
    try:
        request = ModifySheetValuesRequest(
            spreadsheet_id=spreadsheet_id,
            range_name=range_name,
            values=values
            if isinstance(values, str)
            else json.dumps(values)
            if values
            else None,
            value_input_option=value_input_option,
            clear_values=clear_values,
        )
    except Exception as e:
        error_msg = f"Invalid parameters: {str(e)}"
        logger.error(f"[modify_sheet_values] {error_msg}")
        return ModifySheetValuesResponse(
            status="error", message="", error=error_msg
        ).model_dump_json(indent=2)

    service = get_service()
    operation = "clear" if request.clear_values else "write"
    logger.info(
        f"[modify_sheet_values] Invoked. Operation: {operation}, Spreadsheet: {request.spreadsheet_id}, Range: {request.range_name}"
    )

    try:
        # Parse values if it's a JSON string
        parsed_values = None
        if request.values is not None:
            try:
                parsed_values = (
                    json.loads(request.values)
                    if isinstance(request.values, str)
                    else request.values
                )
                if not isinstance(parsed_values, list):
                    raise ValueError(
                        f"Values must be a list, got {type(parsed_values).__name__}"
                    )
                for i, row in enumerate(parsed_values):
                    if not isinstance(row, list):
                        raise ValueError(
                            f"Row {i} must be a list, got {type(row).__name__}"
                        )
                logger.info(
                    f"[modify_sheet_values] Parsed values with {len(parsed_values)} rows"
                )
            except json.JSONDecodeError as e:
                raise UserInputError(f"Invalid JSON format for values: {e}")
            except ValueError as e:
                raise UserInputError(f"Invalid values structure: {e}")

        if not request.clear_values and not parsed_values:
            raise UserInputError(
                "Either 'values' must be provided or 'clear_values' must be True."
            )

        if request.clear_values:
            result = await asyncio.to_thread(
                service.spreadsheets()
                .values()
                .clear(spreadsheetId=request.spreadsheet_id, range=request.range_name)
                .execute
            )

            cleared_range = result.get("clearedRange", request.range_name)
            message = f"Successfully cleared range '{cleared_range}' in spreadsheet {request.spreadsheet_id}."
            logger.info(f"Successfully cleared range '{cleared_range}'.")
        else:
            body = {"values": parsed_values}

            result = await asyncio.to_thread(
                service.spreadsheets()
                .values()
                .update(
                    spreadsheetId=request.spreadsheet_id,
                    range=request.range_name,
                    valueInputOption=request.value_input_option,
                    body=body,
                )
                .execute
            )

            updated_cells = result.get("updatedCells", 0)
            updated_rows = result.get("updatedRows", 0)
            updated_columns = result.get("updatedColumns", 0)

            message = (
                f"Successfully updated range '{request.range_name}' in spreadsheet {request.spreadsheet_id}. "
                f"Updated: {updated_cells} cells, {updated_rows} rows, {updated_columns} columns."
            )
            logger.info(f"Successfully updated {updated_cells} cells.")

        return ModifySheetValuesResponse(
            status="success", message=message
        ).model_dump_json(indent=2)

    except UserInputError as e:
        error_msg = str(e)
        logger.error(f"[modify_sheet_values] {error_msg}")
        return ModifySheetValuesResponse(
            status="error", message="", error=error_msg
        ).model_dump_json(indent=2)
    except Exception as error:
        error_msg = f"Error modifying sheet values: {str(error)}"
        logger.error(f"[modify_sheet_values] {error_msg}")
        return ModifySheetValuesResponse(
            status="error", message="", error=error_msg
        ).model_dump_json(indent=2)


@content_server.tool()
async def format_sheet_range(
    spreadsheet_id: str,
    range_name: str,
    background_color: Optional[str] = None,
    text_color: Optional[str] = None,
    number_format_type: Optional[str] = None,
    number_format_pattern: Optional[str] = None,
) -> str:
    """
    Applies formatting to a range: background/text color and number/date formats.

    Colors accept hex strings (#RRGGBB). Number formats follow Sheets types
    (e.g., NUMBER, NUMBER_WITH_GROUPING, CURRENCY, DATE, TIME, DATE_TIME,
    PERCENT, TEXT, SCIENTIFIC). If no sheet name is provided, the first sheet
    is used.

    Args:
        spreadsheet_id (str): The ID of the spreadsheet. Required.
        range_name (str): A1-style range (optionally with sheet name). Required.
        background_color (Optional[str]): Hex background color (e.g., "#FFEECC").
        text_color (Optional[str]): Hex text color (e.g., "#000000").
        number_format_type (Optional[str]): Sheets number format type (e.g., "DATE").
        number_format_pattern (Optional[str]): Optional custom pattern for the number format.

    Returns:
        str: Confirmation of the applied formatting.
    """
    # Validate input parameters
    try:
        request = FormatSheetRangeRequest(
            spreadsheet_id=spreadsheet_id,
            range_name=range_name,
            background_color=background_color,
            text_color=text_color,
            number_format_type=number_format_type,
            number_format_pattern=number_format_pattern,
        )
    except Exception as e:
        error_msg = f"Invalid parameters: {str(e)}"
        logger.error(f"[format_sheet_range] {error_msg}")
        return FormatSheetRangeResponse(
            status="error", message="", error=error_msg
        ).model_dump_json(indent=2)

    service = get_service()
    logger.info(
        "[format_sheet_range] Invoked. Spreadsheet: %s, Range: %s",
        request.spreadsheet_id,
        request.range_name,
    )

    try:
        if not any(
            [request.background_color, request.text_color, request.number_format_type]
        ):
            raise UserInputError(
                "Provide at least one of background_color, text_color, or number_format_type."
            )

        bg_color_parsed = _parse_hex_color(request.background_color)
        text_color_parsed = _parse_hex_color(request.text_color)

        number_format = None
        if request.number_format_type:
            allowed_number_formats = {
                "NUMBER",
                "NUMBER_WITH_GROUPING",
                "CURRENCY",
                "PERCENT",
                "SCIENTIFIC",
                "DATE",
                "TIME",
                "DATE_TIME",
                "TEXT",
            }
            normalized_type = request.number_format_type.upper()
            if normalized_type not in allowed_number_formats:
                raise UserInputError(
                    f"number_format_type must be one of {sorted(allowed_number_formats)}."
                )
            number_format = {"type": normalized_type}
            if request.number_format_pattern:
                number_format["pattern"] = request.number_format_pattern

        metadata = await asyncio.to_thread(
            service.spreadsheets()
            .get(
                spreadsheetId=request.spreadsheet_id,
                fields="sheets(properties(sheetId,title))",
            )
            .execute
        )
        sheets = metadata.get("sheets", [])
        grid_range = _parse_a1_range(request.range_name, sheets)

        user_entered_format = {}
        fields = []
        if bg_color_parsed:
            user_entered_format["backgroundColor"] = bg_color_parsed
            fields.append("userEnteredFormat.backgroundColor")
        if text_color_parsed:
            user_entered_format["textFormat"] = {"foregroundColor": text_color_parsed}
            fields.append("userEnteredFormat.textFormat.foregroundColor")
        if number_format:
            user_entered_format["numberFormat"] = number_format
            fields.append("userEnteredFormat.numberFormat")

        if not user_entered_format:
            raise UserInputError(
                "No formatting applied. Verify provided colors or number format."
            )

        request_body = {
            "requests": [
                {
                    "repeatCell": {
                        "range": grid_range,
                        "cell": {"userEnteredFormat": user_entered_format},
                        "fields": ",".join(fields),
                    }
                }
            ]
        }

        await asyncio.to_thread(
            service.spreadsheets()
            .batchUpdate(spreadsheetId=request.spreadsheet_id, body=request_body)
            .execute
        )

        applied_parts = []
        if bg_color_parsed:
            applied_parts.append(f"background {request.background_color}")
        if text_color_parsed:
            applied_parts.append(f"text {request.text_color}")
        if number_format:
            nf_desc = number_format["type"]
            if request.number_format_pattern:
                nf_desc += f" (pattern: {request.number_format_pattern})"
            applied_parts.append(f"format {nf_desc}")

        summary = ", ".join(applied_parts)
        message = (
            f"Applied formatting to range '{request.range_name}' in spreadsheet {request.spreadsheet_id} "
            f": {summary}."
        )

        return FormatSheetRangeResponse(
            status="success", message=message
        ).model_dump_json(indent=2)

    except UserInputError as e:
        error_msg = str(e)
        logger.error(f"[format_sheet_range] {error_msg}")
        return FormatSheetRangeResponse(
            status="error", message="", error=error_msg
        ).model_dump_json(indent=2)
    except Exception as error:
        error_msg = f"Error formatting sheet range: {str(error)}"
        logger.error(f"[format_sheet_range] {error_msg}")
        return FormatSheetRangeResponse(
            status="error", message="", error=error_msg
        ).model_dump_json(indent=2)


@content_server.tool()
async def add_conditional_formatting(
    spreadsheet_id: str,
    range_name: str,
    condition_type: str,
    condition_values: Optional[Union[str, List[Union[str, int, float]]]] = None,
    background_color: Optional[str] = None,
    text_color: Optional[str] = None,
    rule_index: Optional[int] = None,
    gradient_points: Optional[Union[str, List[dict]]] = None,
) -> str:
    """
    Adds a conditional formatting rule to a range.

    Args:
        spreadsheet_id (str): The ID of the spreadsheet. Required.
        range_name (str): A1-style range (optionally with sheet name). Required.
        condition_type (str): Sheets condition type (e.g., NUMBER_GREATER, TEXT_CONTAINS, DATE_BEFORE, CUSTOM_FORMULA).
        condition_values (Optional[Union[str, List[Union[str, int, float]]]]): Values for the condition; accepts a list or a JSON string representing a list. Depends on condition_type.
        background_color (Optional[str]): Hex background color to apply when condition matches.
        text_color (Optional[str]): Hex text color to apply when condition matches.
        rule_index (Optional[int]): Optional position to insert the rule (0-based) within the sheet's rules.
        gradient_points (Optional[Union[str, List[dict]]]): List (or JSON list) of gradient points for a color scale. If provided, a gradient rule is created and boolean parameters are ignored.

    Returns:
        str: Confirmation of the added rule.
    """
    # Validate input parameters
    try:
        request = AddConditionalFormattingRequest(
            spreadsheet_id=spreadsheet_id,
            range_name=range_name,
            condition_type=condition_type,
            condition_values=json.dumps(condition_values)
            if isinstance(condition_values, list)
            else condition_values,
            background_color=background_color,
            text_color=text_color,
            rule_index=rule_index,
            gradient_points=json.dumps(gradient_points)
            if isinstance(gradient_points, list)
            else gradient_points,
        )
    except Exception as e:
        error_msg = f"Invalid parameters: {str(e)}"
        logger.error(f"[add_conditional_formatting] {error_msg}")
        return AddConditionalFormattingResponse(
            status="error", message="", error=error_msg
        ).model_dump_json(indent=2)

    service = get_service()
    logger.info(
        "[add_conditional_formatting] Invoked. Spreadsheet: %s, Range: %s, Type: %s, Values: %s",
        request.spreadsheet_id,
        request.range_name,
        request.condition_type,
        request.condition_values,
    )

    try:
        if request.rule_index is not None and (
            not isinstance(request.rule_index, int) or request.rule_index < 0
        ):
            raise UserInputError(
                "rule_index must be a non-negative integer when provided."
            )

        condition_values_list = _parse_condition_values(request.condition_values)
        gradient_points_list = _parse_gradient_points(request.gradient_points)

        sheets, sheet_titles = await _fetch_sheets_with_rules(
            service, request.spreadsheet_id
        )
        grid_range = _parse_a1_range(request.range_name, sheets)

        target_sheet = None
        for sheet in sheets:
            if sheet.get("properties", {}).get("sheetId") == grid_range.get("sheetId"):
                target_sheet = sheet
                break
        if target_sheet is None:
            raise UserInputError(
                "Target sheet not found while adding conditional formatting."
            )

        current_rules = target_sheet.get("conditionalFormats", []) or []

        insert_at = (
            request.rule_index if request.rule_index is not None else len(current_rules)
        )
        if insert_at > len(current_rules):
            raise UserInputError(
                f"rule_index {insert_at} is out of range for sheet '{target_sheet.get('properties', {}).get('title', 'Unknown')}' "
                f"(current count: {len(current_rules)})."
            )

        if gradient_points_list:
            new_rule = _build_gradient_rule([grid_range], gradient_points_list)
            rule_desc = "gradient"
            values_desc = ""
            applied_parts = [f"gradient points {len(gradient_points_list)}"]
        else:
            rule, cond_type_normalized = _build_boolean_rule(
                [grid_range],
                request.condition_type,
                condition_values_list,
                request.background_color,
                request.text_color,
            )
            new_rule = rule
            rule_desc = cond_type_normalized
            values_desc = ""
            if condition_values_list:
                values_desc = f" with values {condition_values_list}"
            applied_parts = []
            if request.background_color:
                applied_parts.append(f"background {request.background_color}")
            if request.text_color:
                applied_parts.append(f"text {request.text_color}")

        new_rules_state = copy.deepcopy(current_rules)
        new_rules_state.insert(insert_at, new_rule)

        add_rule_request = {"rule": new_rule}
        if request.rule_index is not None:
            add_rule_request["index"] = request.rule_index

        request_body = {"requests": [{"addConditionalFormatRule": add_rule_request}]}

        await asyncio.to_thread(
            service.spreadsheets()
            .batchUpdate(spreadsheetId=request.spreadsheet_id, body=request_body)
            .execute
        )

        format_desc = ", ".join(applied_parts) if applied_parts else "format applied"

        sheet_title = target_sheet.get("properties", {}).get("title", "Unknown")
        state_text = _format_conditional_rules_section(
            sheet_title, new_rules_state, sheet_titles, indent=""
        )

        message = "\n".join(
            [
                f"Added conditional format on '{request.range_name}' in spreadsheet {request.spreadsheet_id} "
                f": {rule_desc}{values_desc}; format: {format_desc}.",
                state_text,
            ]
        )

        return AddConditionalFormattingResponse(
            status="success", message=message
        ).model_dump_json(indent=2)

    except UserInputError as e:
        error_msg = str(e)
        logger.error(f"[add_conditional_formatting] {error_msg}")
        return AddConditionalFormattingResponse(
            status="error", message="", error=error_msg
        ).model_dump_json(indent=2)
    except Exception as error:
        error_msg = f"Error adding conditional formatting: {str(error)}"
        logger.error(f"[add_conditional_formatting] {error_msg}")
        return AddConditionalFormattingResponse(
            status="error", message="", error=error_msg
        ).model_dump_json(indent=2)


@content_server.tool()
async def update_conditional_formatting(
    spreadsheet_id: str,
    rule_index: int,
    range_name: Optional[str] = None,
    condition_type: Optional[str] = None,
    condition_values: Optional[Union[str, List[Union[str, int, float]]]] = None,
    background_color: Optional[str] = None,
    text_color: Optional[str] = None,
    sheet_name: Optional[str] = None,
    gradient_points: Optional[Union[str, List[dict]]] = None,
) -> str:
    """
    Updates an existing conditional formatting rule by index on a sheet.

    Args:
        spreadsheet_id (str): The ID of the spreadsheet. Required.
        range_name (Optional[str]): A1-style range to apply the updated rule (optionally with sheet name). If omitted, existing ranges are preserved.
        rule_index (int): Index of the rule to update (0-based).
        condition_type (Optional[str]): Sheets condition type. If omitted, the existing rule's type is preserved.
        condition_values (Optional[Union[str, List[Union[str, int, float]]]]): Values for the condition.
        background_color (Optional[str]): Hex background color when condition matches.
        text_color (Optional[str]): Hex text color when condition matches.
        sheet_name (Optional[str]): Sheet name to locate the rule when range_name is omitted. Defaults to first sheet.
        gradient_points (Optional[Union[str, List[dict]]]): If provided, updates the rule to a gradient color scale using these points.

    Returns:
        str: Confirmation of the updated rule and the current rule state.
    """
    # Validate input parameters
    try:
        request = UpdateConditionalFormattingRequest(
            spreadsheet_id=spreadsheet_id,
            rule_index=rule_index,
            range_name=range_name,
            condition_type=condition_type,
            condition_values=json.dumps(condition_values)
            if isinstance(condition_values, list)
            else condition_values,
            background_color=background_color,
            text_color=text_color,
            sheet_name=sheet_name,
            gradient_points=json.dumps(gradient_points)
            if isinstance(gradient_points, list)
            else gradient_points,
        )
    except Exception as e:
        error_msg = f"Invalid parameters: {str(e)}"
        logger.error(f"[update_conditional_formatting] {error_msg}")
        return UpdateConditionalFormattingResponse(
            status="error", message="", error=error_msg
        ).model_dump_json(indent=2)

    service = get_service()
    logger.info(
        "[update_conditional_formatting] Invoked. Spreadsheet: %s, Range: %s, Rule Index: %s",
        request.spreadsheet_id,
        request.range_name,
        request.rule_index,
    )

    try:
        if not isinstance(request.rule_index, int) or request.rule_index < 0:
            raise UserInputError("rule_index must be a non-negative integer.")

        condition_values_list = _parse_condition_values(request.condition_values)
        gradient_points_list = _parse_gradient_points(request.gradient_points)

        sheets, sheet_titles = await _fetch_sheets_with_rules(
            service, request.spreadsheet_id
        )

        target_sheet = None
        grid_range = None
        if request.range_name:
            grid_range = _parse_a1_range(request.range_name, sheets)
            for sheet in sheets:
                if sheet.get("properties", {}).get("sheetId") == grid_range.get(
                    "sheetId"
                ):
                    target_sheet = sheet
                    break
        else:
            target_sheet = _select_sheet(sheets, request.sheet_name)

        if target_sheet is None:
            raise UserInputError(
                "Target sheet not found while updating conditional formatting."
            )

        sheet_props = target_sheet.get("properties", {})
        sheet_id = sheet_props.get("sheetId")
        sheet_title = sheet_props.get("title", f"Sheet {sheet_id}")

        rules = target_sheet.get("conditionalFormats", []) or []
        if request.rule_index >= len(rules):
            raise UserInputError(
                f"rule_index {request.rule_index} is out of range for sheet '{sheet_title}' (current count: {len(rules)})."
            )

        existing_rule = rules[request.rule_index]
        ranges_to_use = existing_rule.get("ranges", [])
        if request.range_name:
            ranges_to_use = [grid_range]
        if not ranges_to_use:
            ranges_to_use = [{"sheetId": sheet_id}]

        new_rule = None
        rule_desc = ""
        values_desc = ""
        format_desc = ""

        if gradient_points_list is not None:
            new_rule = _build_gradient_rule(ranges_to_use, gradient_points_list)
            rule_desc = "gradient"
            format_desc = f"gradient points {len(gradient_points_list)}"
        elif "gradientRule" in existing_rule:
            if any(
                [
                    request.background_color,
                    request.text_color,
                    request.condition_type,
                    condition_values_list,
                ]
            ):
                raise UserInputError(
                    "Existing rule is a gradient rule. Provide gradient_points to update it, or omit formatting/condition parameters to keep it unchanged."
                )
            new_rule = {
                "ranges": ranges_to_use,
                "gradientRule": existing_rule.get("gradientRule", {}),
            }
            rule_desc = "gradient"
            format_desc = "gradient (unchanged)"
        else:
            existing_boolean = existing_rule.get("booleanRule", {})
            existing_condition = existing_boolean.get("condition", {})
            existing_format = copy.deepcopy(existing_boolean.get("format", {}))

            cond_type = (
                request.condition_type or existing_condition.get("type", "")
            ).upper()
            if not cond_type:
                raise UserInputError("condition_type is required for boolean rules.")
            if cond_type not in CONDITION_TYPES:
                raise UserInputError(
                    f"condition_type must be one of {sorted(CONDITION_TYPES)}."
                )

            if condition_values_list is not None:
                cond_values = [
                    {"userEnteredValue": str(val)} for val in condition_values_list
                ]
            else:
                cond_values = existing_condition.get("values")

            new_format = copy.deepcopy(existing_format) if existing_format else {}
            if request.background_color is not None:
                bg_color_parsed = _parse_hex_color(request.background_color)
                if bg_color_parsed:
                    new_format["backgroundColor"] = bg_color_parsed
                elif "backgroundColor" in new_format:
                    del new_format["backgroundColor"]
            if request.text_color is not None:
                text_color_parsed = _parse_hex_color(request.text_color)
                text_format = copy.deepcopy(new_format.get("textFormat", {}))
                if text_color_parsed:
                    text_format["foregroundColor"] = text_color_parsed
                elif "foregroundColor" in text_format:
                    del text_format["foregroundColor"]
                if text_format:
                    new_format["textFormat"] = text_format
                elif "textFormat" in new_format:
                    del new_format["textFormat"]

            if not new_format:
                raise UserInputError(
                    "At least one format option must remain on the rule."
                )

            new_rule = {
                "ranges": ranges_to_use,
                "booleanRule": {
                    "condition": {"type": cond_type},
                    "format": new_format,
                },
            }
            if cond_values:
                new_rule["booleanRule"]["condition"]["values"] = cond_values

            rule_desc = cond_type
            if condition_values_list:
                values_desc = f" with values {condition_values_list}"
            format_parts = []
            if "backgroundColor" in new_format:
                format_parts.append("background updated")
            if "textFormat" in new_format and new_format["textFormat"].get(
                "foregroundColor"
            ):
                format_parts.append("text color updated")
            format_desc = (
                ", ".join(format_parts) if format_parts else "format preserved"
            )

        new_rules_state = copy.deepcopy(rules)
        new_rules_state[request.rule_index] = new_rule

        request_body = {
            "requests": [
                {
                    "updateConditionalFormatRule": {
                        "index": request.rule_index,
                        "sheetId": sheet_id,
                        "rule": new_rule,
                    }
                }
            ]
        }

        await asyncio.to_thread(
            service.spreadsheets()
            .batchUpdate(spreadsheetId=request.spreadsheet_id, body=request_body)
            .execute
        )

        state_text = _format_conditional_rules_section(
            sheet_title, new_rules_state, sheet_titles, indent=""
        )

        message = "\n".join(
            [
                f"Updated conditional format at index {request.rule_index} on sheet '{sheet_title}' in spreadsheet {request.spreadsheet_id} "
                f": {rule_desc}{values_desc}; format: {format_desc}.",
                state_text,
            ]
        )

        return UpdateConditionalFormattingResponse(
            status="success", message=message
        ).model_dump_json(indent=2)

    except UserInputError as e:
        error_msg = str(e)
        logger.error(f"[update_conditional_formatting] {error_msg}")
        return UpdateConditionalFormattingResponse(
            status="error", message="", error=error_msg
        ).model_dump_json(indent=2)
    except Exception as error:
        error_msg = f"Error updating conditional formatting: {str(error)}"
        logger.error(f"[update_conditional_formatting] {error_msg}")
        return UpdateConditionalFormattingResponse(
            status="error", message="", error=error_msg
        ).model_dump_json(indent=2)


@content_server.tool()
async def delete_conditional_formatting(
    spreadsheet_id: str,
    rule_index: int,
    sheet_name: Optional[str] = None,
) -> str:
    """
    Deletes an existing conditional formatting rule by index on a sheet.

    Args:
        spreadsheet_id (str): The ID of the spreadsheet. Required.
        rule_index (int): Index of the rule to delete (0-based).
        sheet_name (Optional[str]): Name of the sheet that contains the rule. Defaults to the first sheet if not provided.

    Returns:
        str: Confirmation of the deletion and the current rule state.
    """
    # Validate input parameters
    try:
        request = DeleteConditionalFormattingRequest(
            spreadsheet_id=spreadsheet_id,
            rule_index=rule_index,
            sheet_name=sheet_name,
        )
    except Exception as e:
        error_msg = f"Invalid parameters: {str(e)}"
        logger.error(f"[delete_conditional_formatting] {error_msg}")
        return DeleteConditionalFormattingResponse(
            status="error", message="", error=error_msg
        ).model_dump_json(indent=2)

    service = get_service()
    logger.info(
        "[delete_conditional_formatting] Invoked. Spreadsheet: %s, Sheet: %s, Rule Index: %s",
        request.spreadsheet_id,
        request.sheet_name,
        request.rule_index,
    )

    try:
        if not isinstance(request.rule_index, int) or request.rule_index < 0:
            raise UserInputError("rule_index must be a non-negative integer.")

        sheets, sheet_titles = await _fetch_sheets_with_rules(
            service, request.spreadsheet_id
        )
        target_sheet = _select_sheet(sheets, request.sheet_name)

        sheet_props = target_sheet.get("properties", {})
        sheet_id = sheet_props.get("sheetId")
        target_sheet_name = sheet_props.get("title", f"Sheet {sheet_id}")
        rules = target_sheet.get("conditionalFormats", []) or []
        if request.rule_index >= len(rules):
            raise UserInputError(
                f"rule_index {request.rule_index} is out of range for sheet '{target_sheet_name}' (current count: {len(rules)})."
            )

        new_rules_state = copy.deepcopy(rules)
        del new_rules_state[request.rule_index]

        request_body = {
            "requests": [
                {
                    "deleteConditionalFormatRule": {
                        "index": request.rule_index,
                        "sheetId": sheet_id,
                    }
                }
            ]
        }

        await asyncio.to_thread(
            service.spreadsheets()
            .batchUpdate(spreadsheetId=request.spreadsheet_id, body=request_body)
            .execute
        )

        state_text = _format_conditional_rules_section(
            target_sheet_name, new_rules_state, sheet_titles, indent=""
        )

        message = "\n".join(
            [
                f"Deleted conditional format at index {request.rule_index} on sheet '{target_sheet_name}' in spreadsheet {request.spreadsheet_id}.",
                state_text,
            ]
        )

        return DeleteConditionalFormattingResponse(
            status="success", message=message
        ).model_dump_json(indent=2)

    except UserInputError as e:
        error_msg = str(e)
        logger.error(f"[delete_conditional_formatting] {error_msg}")
        return DeleteConditionalFormattingResponse(
            status="error", message="", error=error_msg
        ).model_dump_json(indent=2)
    except Exception as error:
        error_msg = f"Error deleting conditional formatting: {str(error)}"
        logger.error(f"[delete_conditional_formatting] {error_msg}")
        return DeleteConditionalFormattingResponse(
            status="error", message="", error=error_msg
        ).model_dump_json(indent=2)


@content_server.tool()
async def create_spreadsheet(
    title: str,
    sheet_names: Optional[List[str]] = None,
) -> str:
    """
    Creates a new Google Spreadsheet.

    Args:
        title (str): The title of the new spreadsheet. Required.
        sheet_names (Optional[List[str]]): List of sheet names to create. If not provided, creates one sheet with default name.

    Returns:
        str: Information about the newly created spreadsheet including ID, URL, and locale.
    """
    # Validate input parameters
    try:
        request = CreateSpreadsheetRequest(
            title=title,
            sheet_names=sheet_names,
        )
    except Exception as e:
        error_msg = f"Invalid parameters: {str(e)}"
        logger.error(f"[create_spreadsheet] {error_msg}")
        return CreateSpreadsheetResponse(
            status="error", message="", error=error_msg
        ).model_dump_json(indent=2)

    service = get_service()
    logger.info(f"[create_spreadsheet] Invoked. Title: {request.title}")

    try:
        spreadsheet_body = {"properties": {"title": request.title}}

        if request.sheet_names:
            spreadsheet_body["sheets"] = [
                {"properties": {"title": sheet_name}}
                for sheet_name in request.sheet_names
            ]

        spreadsheet = await asyncio.to_thread(
            service.spreadsheets()
            .create(
                body=spreadsheet_body,
                fields="spreadsheetId,spreadsheetUrl,properties(title,locale)",
            )
            .execute
        )

        properties = spreadsheet.get("properties", {})
        spreadsheet_id = spreadsheet.get("spreadsheetId")
        spreadsheet_url = spreadsheet.get("spreadsheetUrl")
        locale = properties.get("locale", "Unknown")

        message = (
            f"Successfully created spreadsheet '{request.title}'. "
            f"ID: {spreadsheet_id} | URL: {spreadsheet_url} | Locale: {locale}"
        )

        logger.info(f"Successfully created spreadsheet. ID: {spreadsheet_id}")
        return CreateSpreadsheetResponse(
            status="success", message=message
        ).model_dump_json(indent=2)

    except Exception as error:
        error_msg = f"Error creating spreadsheet: {str(error)}"
        logger.error(f"[create_spreadsheet] {error_msg}")
        return CreateSpreadsheetResponse(
            status="error", message="", error=error_msg
        ).model_dump_json(indent=2)


@content_server.tool()
async def create_sheet(
    spreadsheet_id: str,
    sheet_name: str,
) -> str:
    """
    Creates a new sheet within an existing spreadsheet.

    Args:
        spreadsheet_id (str): The ID of the spreadsheet. Required.
        sheet_name (str): The name of the new sheet. Required.

    Returns:
        str: Confirmation message of the successful sheet creation.
    """
    # Validate input parameters
    try:
        request = CreateSheetRequest(
            spreadsheet_id=spreadsheet_id,
            sheet_name=sheet_name,
        )
    except Exception as e:
        error_msg = f"Invalid parameters: {str(e)}"
        logger.error(f"[create_sheet] {error_msg}")
        return CreateSheetResponse(
            status="error", message="", error=error_msg
        ).model_dump_json(indent=2)

    service = get_service()
    logger.info(
        f"[create_sheet] Invoked. Spreadsheet: {request.spreadsheet_id}, Sheet: {request.sheet_name}"
    )

    try:
        request_body = {
            "requests": [{"addSheet": {"properties": {"title": request.sheet_name}}}]
        }

        response = await asyncio.to_thread(
            service.spreadsheets()
            .batchUpdate(spreadsheetId=request.spreadsheet_id, body=request_body)
            .execute
        )

        sheet_id = response["replies"][0]["addSheet"]["properties"]["sheetId"]

        message = f"Successfully created sheet '{request.sheet_name}' (ID: {sheet_id}) in spreadsheet {request.spreadsheet_id}."

        logger.info(f"Successfully created sheet. Sheet ID: {sheet_id}")
        return CreateSheetResponse(status="success", message=message).model_dump_json(
            indent=2
        )

    except Exception as error:
        error_msg = f"Error creating sheet: {str(error)}"
        logger.error(f"[create_sheet] {error_msg}")
        return CreateSheetResponse(
            status="error", message="", error=error_msg
        ).model_dump_json(indent=2)


# Create comment management tools for sheets
_comment_tools = create_comment_tools("spreadsheet", "spreadsheet_id")

# Extract and register the functions
read_sheet_comments = _comment_tools["read_comments"]
create_sheet_comment = _comment_tools["create_comment"]
reply_to_sheet_comment = _comment_tools["reply_to_comment"]
resolve_sheet_comment = _comment_tools["resolve_comment"]
