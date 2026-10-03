"""Pydantic models for Gmail API tools"""

from typing import Optional, List
from pydantic import BaseModel, EmailStr, Field, validator


class SendEmailRequest(BaseModel):
    """Send email request"""

    recipient_id: EmailStr = Field(
        ..., description="Recipient email", pattern=r"[^@]+@[^@]+\.[^@]+"
    )
    subject: str = Field(..., min_length=1, description="Email subject")
    message: str = Field(..., min_length=1, description="Email body")


class SendEmailResponse(BaseModel):
    """Send email response"""

    success: bool
    message_id: Optional[str] = None
    error: Optional[str] = None


class EmailAddress(BaseModel):
    """Email address with validation"""

    email: EmailStr


class EmailResponse(BaseModel):
    """Email response"""

    success: bool
    error: Optional[str] = None


class UnreadEmailsRequest(BaseModel):
    """Get unread emails request"""

    date: int = Field(10, ge=1, le=365, description="Days to look back")
    max_results: int = Field(20, ge=1, le=500, description="Max results")


class EmailMetadata(BaseModel):
    """Email metadata structure"""

    id: str
    thread_id: str
    snippet: str
    # labels: List[str]
    # size: int
    # internal_date: Optional[str]
    subject: str
    from_: str = Field(..., alias="from")
    date: str
    to: str

    class Config:
        populate_by_name = True


class UnreadEmailsResponse(BaseModel):
    """Unread emails response"""

    count: int
    emails: List[EmailMetadata]


class LabelRequest(BaseModel):
    """Create/rename label request"""

    name: str = Field(..., min_length=1, max_length=225, description="Label name")

    @validator("name")
    def validate_label_name(cls, v):
        if "/" in v and v.count("/") > 5:
            raise ValueError("Label name cannot have more than 5 levels")
        return v


class LabelResponse(BaseModel):
    """Label operation response"""

    success: bool
    label_id: Optional[str] = None
    name: Optional[str] = None
    error: Optional[str] = None


class LabelInfo(BaseModel):
    """Label information"""

    id: str
    name: str
    type: str


class ListLabelsResponse(BaseModel):
    """List labels response"""

    count: int
    labels: List[LabelInfo]
    error: Optional[str] = None


class ApplyLabelRequest(BaseModel):
    """Apply/remove label request"""

    email_id: str = Field(..., description="Message ID")
    label_id: str = Field(..., description="Label ID")


class DraftRequest(BaseModel):
    """Create draft request"""

    recipient_id: EmailStr
    subject: str = Field(..., min_length=1)
    message: str = Field(..., min_length=1)


class DraftInfo(BaseModel):
    """Draft information"""

    id: str
    subject: str
    to: str


class ListDraftsResponse(BaseModel):
    """List drafts response"""

    count: int
    drafts: List[DraftInfo]
    error: Optional[str] = None


class BatchArchiveRequest(BaseModel):
    """Batch archive request"""

    query: str = Field(..., description="Gmail search query")
    max_emails: int = Field(100, ge=1, le=500, description="Max emails to archive")


class BatchArchiveResponse(BaseModel):
    """Batch archive response"""

    success: bool
    archived_count: int = 0
    total_found: int = 0
    message: Optional[str] = None
    error: Optional[str] = None


class StandardResponse(BaseModel):
    """Standard success/error response"""

    success: bool
    error: Optional[str] = None


class EmailIdRequest(BaseModel):
    """Email ID request for single email operations"""

    email_id: str = Field(..., min_length=1, description="Gmail message ID")


class ReadEmailResponse(BaseModel):
    """Read email response"""

    content: str
    subject: str
    from_: str = Field(..., alias="from")
    to: str
    date: str
    error: Optional[str] = None

    class Config:
        populate_by_name = True


class DraftResponse(BaseModel):
    """Draft creation response"""

    success: bool
    draft_id: Optional[str] = None
    error: Optional[str] = None


class FilterInfo(
    BaseModel
):  # they are not directly used but use in other filers classes
    """Filter information structure"""

    id: str
    criteria: dict
    action: dict


class ListFiltersResponse(BaseModel):
    """List filters response"""

    count: int
    filters: List[FilterInfo]
    error: Optional[str] = None


class FilterIdRequest(BaseModel):
    """Filter ID request"""

    filter_id: str = Field(..., min_length=1, description="Filter ID")


class SearchEmailsRequest(BaseModel):
    """Search emails request"""

    query: str = Field(default="in:inbox", description="Gmail search query")
    max_results: Optional[int] = Field(None, ge=1, le=500, description="Max results")


class SearchEmailsResponse(BaseModel):
    """Search emails response"""

    count: int
    emails: List[dict]
    error: Optional[str] = None


class FolderRequest(BaseModel):
    """Create folder request"""

    name: str = Field(..., min_length=1, max_length=225, description="Folder name")


class FolderResponse(BaseModel):
    """Folder operation response"""

    success: bool
    folder_id: Optional[str] = None
    name: Optional[str] = None
    error: Optional[str] = None


class MoveToFolderRequest(BaseModel):
    """Move email to folder request"""

    email_id: str = Field(..., min_length=1, description="Gmail message ID")
    folder_id: str = Field(..., min_length=1, description="Folder/Label ID")


class FolderInfo(BaseModel):
    """Folder information"""

    id: str
    name: str


class ListFoldersResponse(BaseModel):
    """List folders response"""

    count: int
    folders: List[FolderInfo]
    error: Optional[str] = None


class RenameLabelRequest(BaseModel):
    """Rename label request"""

    label_id: str = Field(..., min_length=1, description="Label ID to rename")
    new_name: str = Field(
        ..., min_length=1, max_length=225, description="New label name"
    )

    @validator("new_name")
    def validate_label_name(cls, v):
        if "/" in v and v.count("/") > 5:
            raise ValueError("Label name cannot have more than 5 levels")
        return v


class ListArchivedRequest(BaseModel):
    """List archived emails request"""

    max_results: int = Field(100, ge=1, le=500, description="Max results")


class SearchByLabelRequest(BaseModel):
    """Search by label request"""

    label_id: str = Field(..., min_length=1, description="Label ID to search")


class SearchByLabelResponse(BaseModel):
    """Search by label response"""

    count: int
    messages: List[dict]
    error: Optional[str] = None


# Google Calendar Models


class CalendarInfo(BaseModel):
    """Calendar information"""

    id: str
    summary: str
    description: Optional[str] = None
    timeZone: str
    accessRole: str
    primary: bool = False


class ListCalendarsResponse(BaseModel):
    """List calendars response"""

    status: str
    count: int
    calendars: List[CalendarInfo]
    error: Optional[str] = None


class ReminderItem(BaseModel):
    """Calendar reminder item"""

    method: str = Field(..., pattern="^(popup|email)$", description="Reminder method")
    minutes: int = Field(..., ge=0, le=40320, description="Minutes before event")


class GetEventsRequest(BaseModel):
    """Get events request"""

    calendar_id: str = Field("primary", min_length=1, description="Calendar ID")
    event_id: Optional[str] = Field(None, description="Specific event ID")
    time_min: Optional[str] = Field(None, description="Start time (RFC3339)")
    time_max: Optional[str] = Field(None, description="End time (RFC3339)")
    max_results: int = Field(25, ge=1, le=2500, description="Max number of events")
    query: Optional[str] = Field(None, description="Search query")
    detailed: bool = Field(False, description="Return detailed information")
    include_attachments: bool = Field(False, description="Include attachment info")


class GetEventsResponse(BaseModel):
    """Get events response"""

    status: str
    message: str
    event_id: Optional[str] = None
    error: Optional[str] = None


class CreateEventRequest(BaseModel):
    """Create event request"""

    summary: str = Field(..., min_length=1, max_length=1024, description="Event title")
    start_time: str = Field(..., min_length=1, description="Start time in IST")
    end_time: str = Field(..., min_length=1, description="End time in IST")
    calendar_id: str = Field("primary", min_length=1, description="Calendar ID")
    description: Optional[str] = Field(
        None, max_length=8192, description="Event description"
    )
    location: Optional[str] = Field(None, max_length=1024, description="Event location")
    attendees: Optional[List[str]] = Field(None, description="Attendee emails")
    attachments: Optional[List[str]] = Field(None, description="Drive file URLs/IDs")
    add_google_meet: bool = Field(False, description="Add Google Meet link")
    reminders: Optional[List[ReminderItem]] = Field(
        None, max_items=5, description="Custom reminders"
    )
    use_default_reminders: bool = Field(True, description="Use default reminders")
    transparency: Optional[str] = Field(
        None, pattern="^(opaque|transparent)$", description="Busy/free status"
    )

    @validator("attendees")
    def validate_attendees(cls, v):
        if v and len(v) > 200:
            raise ValueError("Maximum 200 attendees allowed")
        return v


class CreateEventResponse(BaseModel):
    """Create event response"""

    status: str
    message: str
    error: Optional[str] = None


class ModifyEventRequest(BaseModel):
    """Modify event request"""

    event_id: str = Field(..., min_length=1, description="Event ID to modify")
    calendar_id: str = Field("primary", min_length=1, description="Calendar ID")
    summary: Optional[str] = Field(None, max_length=1024, description="New event title")
    start_time: Optional[str] = Field(None, description="New start time in IST")
    end_time: Optional[str] = Field(None, description="New end time in IST")
    description: Optional[str] = Field(
        None, max_length=8192, description="New description"
    )
    location: Optional[str] = Field(None, max_length=1024, description="New location")
    attendees: Optional[List[str]] = Field(None, description="New attendee emails")
    add_google_meet: Optional[bool] = Field(None, description="Add/remove Google Meet")
    reminders: Optional[List[ReminderItem]] = Field(
        None, max_items=5, description="New reminders"
    )
    use_default_reminders: Optional[bool] = Field(
        None, description="Use default reminders"
    )
    transparency: Optional[str] = Field(
        None, pattern="^(opaque|transparent)$", description="Busy/free status"
    )

    @validator("attendees")
    def validate_attendees(cls, v):
        if v and len(v) > 200:
            raise ValueError("Maximum 200 attendees allowed")
        return v


class ModifyEventResponse(BaseModel):
    """Modify event response"""

    status: str
    message: str
    event_id: str
    error: Optional[str] = None


class DeleteEventRequest(BaseModel):
    """Delete event request"""

    event_id: str = Field(..., min_length=1, description="Event ID to delete")
    calendar_id: str = Field("primary", min_length=1, description="Calendar ID")


class DeleteEventResponse(BaseModel):
    """Delete event response"""

    status: str
    message: str
    event_id: str
    error: Optional[str] = None


# ========================= Google Tasks Models =================================


class TaskListInfo(BaseModel):
    """Task list information"""

    id: str
    title: str
    updated: Optional[str] = None


class ListTaskListsRequest(BaseModel):
    """List task lists request"""

    max_results: int = Field(
        1000, ge=1, le=1000, description="Max number of task lists"
    )
    page_token: Optional[str] = Field(None, description="Token for pagination")


class ListTaskListsResponse(BaseModel):
    """List task lists response"""

    status: str
    count: int
    task_lists: List[TaskListInfo]
    next_page_token: Optional[str] = None
    error: Optional[str] = None


class GetTaskListRequest(BaseModel):
    """Get task list request"""

    task_list_id: str = Field(..., min_length=1, description="Task list ID")


class GetTaskListResponse(BaseModel):
    """Get task list response"""

    status: str
    message: str = ""
    task_list: Optional[TaskListInfo] = None
    error: Optional[str] = None


class CreateTaskListRequest(BaseModel):
    """Create task list request"""

    title: str = Field(
        ..., min_length=1, max_length=1024, description="Task list title"
    )


class CreateTaskListResponse(BaseModel):
    """Create task list response"""

    status: str
    message: str
    task_list_id: Optional[str] = None
    error: Optional[str] = None


class UpdateTaskListRequest(BaseModel):
    """Update task list request"""

    task_list_id: str = Field(..., min_length=1, description="Task list ID")
    title: str = Field(..., min_length=1, max_length=1024, description="New title")


class UpdateTaskListResponse(BaseModel):
    """Update task list response"""

    status: str
    message: str
    task_list_id: Optional[str] = None
    error: Optional[str] = None


class DeleteTaskListRequest(BaseModel):
    """Delete task list request"""

    task_list_id: str = Field(..., min_length=1, description="Task list ID to delete")


class DeleteTaskListResponse(BaseModel):
    """Delete task list response"""

    status: str
    message: str
    error: Optional[str] = None


class ListTasksRequest(BaseModel):
    """List tasks request"""

    task_list_id: str = Field(..., min_length=1, description="Task list ID")
    max_results: int = Field(20, ge=1, le=10000, description="Max number of tasks")
    page_token: Optional[str] = Field(None, description="Pagination token")
    show_completed: bool = Field(True, description="Include completed tasks")
    show_deleted: bool = Field(False, description="Include deleted tasks")
    show_hidden: bool = Field(False, description="Include hidden tasks")
    show_assigned: bool = Field(False, description="Include assigned tasks")
    completed_max: Optional[str] = Field(
        None, description="Upper bound for completion date"
    )
    completed_min: Optional[str] = Field(
        None, description="Lower bound for completion date"
    )
    due_max: Optional[str] = Field(None, description="Upper bound for due date")
    due_min: Optional[str] = Field(None, description="Lower bound for due date")
    updated_min: Optional[str] = Field(
        None, description="Lower bound for last modification"
    )


class ListTasksResponse(BaseModel):
    """List tasks response"""

    status: str
    message: str
    error: Optional[str] = None


class GetTaskRequest(BaseModel):
    """Get task request"""

    task_list_id: str = Field(..., min_length=1, description="Task list ID")
    task_id: str = Field(..., min_length=1, description="Task ID")


class GetTaskResponse(BaseModel):
    """Get task response"""

    status: str
    message: str
    error: Optional[str] = None


class CreateTaskRequest(BaseModel):
    """Create task request"""

    task_list_id: str = Field(..., min_length=1, description="Task list ID")
    title: str = Field(..., min_length=1, max_length=1024, description="Task title")
    notes: Optional[str] = Field(None, max_length=8192, description="Task notes")
    due: Optional[str] = Field(None, description="Due date (RFC 3339)")
    parent: Optional[str] = Field(None, description="Parent task ID for subtasks")
    previous: Optional[str] = Field(None, description="Previous sibling task ID")


class CreateTaskResponse(BaseModel):
    """Create task response"""

    status: str
    message: str
    task_id: Optional[str] = None
    error: Optional[str] = None


class UpdateTaskRequest(BaseModel):
    """Update task request"""

    task_list_id: str = Field(..., min_length=1, description="Task list ID")
    task_id: str = Field(..., min_length=1, description="Task ID to update")
    title: Optional[str] = Field(None, max_length=1024, description="New title")
    notes: Optional[str] = Field(None, max_length=8192, description="New notes")
    status: Optional[str] = Field(
        None, pattern="^(needsAction|completed)$", description="Task status"
    )
    due: Optional[str] = Field(None, description="New due date (RFC 3339)")


class UpdateTaskResponse(BaseModel):
    """Update task response"""

    status: str
    message: str
    task_id: Optional[str] = None
    error: Optional[str] = None


class DeleteTaskRequest(BaseModel):
    """Delete task request"""

    task_list_id: str = Field(..., min_length=1, description="Task list ID")
    task_id: str = Field(..., min_length=1, description="Task ID to delete")


class DeleteTaskResponse(BaseModel):
    """Delete task response"""

    status: str
    message: str
    error: Optional[str] = None


class MoveTaskRequest(BaseModel):
    """Move task request"""

    task_list_id: str = Field(..., min_length=1, description="Source task list ID")
    task_id: str = Field(..., min_length=1, description="Task ID to move")
    parent: Optional[str] = Field(None, description="New parent task ID")
    previous: Optional[str] = Field(None, description="New previous sibling ID")
    destination_task_list: Optional[str] = Field(
        None, description="Destination task list ID"
    )


class MoveTaskResponse(BaseModel):
    """Move task response"""

    status: str
    message: str
    error: Optional[str] = None


class ClearCompletedTasksRequest(BaseModel):
    """Clear completed tasks request"""

    task_list_id: str = Field(..., min_length=1, description="Task list ID")


class ClearCompletedTasksResponse(BaseModel):
    """Clear completed tasks response"""

    status: str
    message: str
    error: Optional[str] = None


# ========================= Google Sheets Models =================================


class ListSpreadsheetsRequest(BaseModel):
    """List spreadsheets request"""

    max_results: int = Field(
        25, ge=1, le=1000, description="Max number of spreadsheets"
    )


class SpreadsheetInfo(BaseModel):
    """Spreadsheet information"""

    id: str
    name: str
    modified_time: str
    web_view_link: Optional[str] = None


class ListSpreadsheetsResponse(BaseModel):
    """List spreadsheets response"""

    status: str
    message: str
    count: int
    spreadsheets: List[SpreadsheetInfo]
    error: Optional[str] = None


class GetSpreadsheetInfoRequest(BaseModel):
    """Get spreadsheet info request"""

    spreadsheet_id: str = Field(..., min_length=1, description="Spreadsheet ID")


class GetSpreadsheetInfoResponse(BaseModel):
    """Get spreadsheet info response"""

    status: str
    message: str
    error: Optional[str] = None


class ReadSheetValuesRequest(BaseModel):
    """Read sheet values request"""

    spreadsheet_id: str = Field(..., min_length=1, description="Spreadsheet ID")
    range_name: str = Field("A1:Z1000", min_length=1, description="A1 notation range")


class ReadSheetValuesResponse(BaseModel):
    """Read sheet values response"""

    status: str
    message: str
    error: Optional[str] = None


class ModifySheetValuesRequest(BaseModel):
    """Modify sheet values request"""

    spreadsheet_id: str = Field(..., min_length=1, description="Spreadsheet ID")
    range_name: str = Field(..., min_length=1, description="A1 notation range")
    values: Optional[str] = Field(None, description="Values as JSON string or list")
    value_input_option: str = Field(
        "USER_ENTERED",
        pattern="^(USER_ENTERED|RAW)$",
        description="How to interpret input",
    )
    clear_values: bool = Field(False, description="Clear values instead of updating")


class ModifySheetValuesResponse(BaseModel):
    """Modify sheet values response"""

    status: str
    message: str
    error: Optional[str] = None


class FormatSheetRangeRequest(BaseModel):
    """Format sheet range request"""

    spreadsheet_id: str = Field(..., min_length=1, description="Spreadsheet ID")
    range_name: str = Field(..., min_length=1, description="A1 notation range")
    background_color: Optional[str] = Field(
        None, description="Background color in hex format (#RRGGBB)"
    )
    text_color: Optional[str] = Field(
        None, description="Text color in hex format (#RRGGBB)"
    )
    number_format_type: Optional[str] = Field(
        None, description="Number format type (NUMBER, CURRENCY, etc.)"
    )
    number_format_pattern: Optional[str] = Field(
        None, description="Number format pattern"
    )


class FormatSheetRangeResponse(BaseModel):
    """Format sheet range response"""

    status: str
    message: str
    error: Optional[str] = None


class AddConditionalFormattingRequest(BaseModel):
    """Add conditional formatting request"""

    spreadsheet_id: str = Field(..., min_length=1, description="Spreadsheet ID")
    range_name: str = Field(..., min_length=1, description="A1 notation range")
    condition_type: str = Field(..., min_length=1, description="Condition type")
    condition_values: Optional[str] = Field(
        None, description="Condition values as JSON string or list"
    )
    background_color: Optional[str] = Field(
        None, description="Background color in hex format (#RRGGBB)"
    )
    text_color: Optional[str] = Field(
        None, description="Text color in hex format (#RRGGBB)"
    )
    rule_index: Optional[int] = Field(None, description="Rule index for insertion")
    gradient_points: Optional[str] = Field(
        None, description="Gradient points as JSON string or list"
    )


class AddConditionalFormattingResponse(BaseModel):
    """Add conditional formatting response"""

    status: str
    message: str
    error: Optional[str] = None


class UpdateConditionalFormattingRequest(BaseModel):
    """Update conditional formatting request"""

    spreadsheet_id: str = Field(..., min_length=1, description="Spreadsheet ID")
    rule_index: int = Field(..., ge=0, description="Rule index to update")
    range_name: Optional[str] = Field(None, description="A1 notation range")
    condition_type: Optional[str] = Field(None, description="Condition type")
    condition_values: Optional[str] = Field(
        None, description="Condition values as JSON string or list"
    )
    background_color: Optional[str] = Field(
        None, description="Background color in hex format (#RRGGBB)"
    )
    text_color: Optional[str] = Field(
        None, description="Text color in hex format (#RRGGBB)"
    )
    sheet_name: Optional[str] = Field(None, description="Sheet name")
    gradient_points: Optional[str] = Field(
        None, description="Gradient points as JSON string or list"
    )


class UpdateConditionalFormattingResponse(BaseModel):
    """Update conditional formatting response"""

    status: str
    message: str
    error: Optional[str] = None


class DeleteConditionalFormattingRequest(BaseModel):
    """Delete conditional formatting request"""

    spreadsheet_id: str = Field(..., min_length=1, description="Spreadsheet ID")
    rule_index: int = Field(..., ge=0, description="Rule index to delete")
    sheet_name: Optional[str] = Field(None, description="Sheet name")


class DeleteConditionalFormattingResponse(BaseModel):
    """Delete conditional formatting response"""

    status: str
    message: str
    error: Optional[str] = None


class CreateSpreadsheetRequest(BaseModel):
    """Create spreadsheet request"""

    title: str = Field(
        ..., min_length=1, max_length=400, description="Spreadsheet title"
    )
    sheet_names: Optional[List[str]] = Field(
        None, description="Optional list of sheet names"
    )


class CreateSpreadsheetResponse(BaseModel):
    """Create spreadsheet response"""

    status: str
    message: str
    spreadsheet_id: Optional[str] = None
    error: Optional[str] = None


class CreateSheetRequest(BaseModel):
    """Create sheet request"""

    spreadsheet_id: str = Field(..., min_length=1, description="Spreadsheet ID")
    sheet_name: str = Field(..., min_length=1, max_length=100, description="Sheet name")


class CreateSheetResponse(BaseModel):
    """Create sheet response"""

    status: str
    message: str
    error: Optional[str] = None


# ========================= Google Docs Models =================================


class SearchDocsRequest(BaseModel):
    """Search Google Docs request"""

    query: str = Field(..., min_length=1, description="Search query string")
    page_size: int = Field(10, ge=1, le=100, description="Number of results to return")


class DocInfo(BaseModel):
    """Google Doc information"""

    id: str
    name: str
    created_time: Optional[str] = None
    modified_time: Optional[str] = None
    web_view_link: str


class SearchDocsResponse(BaseModel):
    """Search Google Docs response"""

    count: int
    query: str
    docs: List[DocInfo]
    error: Optional[str] = None


class GetDocContentRequest(BaseModel):
    """Get document content request"""

    document_id: str = Field(..., min_length=1, description="Document ID")


class GetDocContentResponse(BaseModel):
    """Get document content response"""

    document_id: str
    name: str
    mime_type: str
    content: str
    web_view_link: str
    error: Optional[str] = None


class ListDocsInFolderRequest(BaseModel):
    """List docs in folder request"""

    folder_id: str = Field("root", min_length=1, description="Drive folder ID")
    page_size: int = Field(100, ge=1, le=1000, description="Number of results")


class ListDocsInFolderResponse(BaseModel):
    """List docs in folder response"""

    count: int
    folder_id: str
    docs: List[DocInfo]
    error: Optional[str] = None


class CreateDocRequest(BaseModel):
    """Create document request"""

    title: str = Field(..., min_length=1, max_length=255, description="Document title")
    content: str = Field("", description="Initial text content")


class CreateDocResponse(BaseModel):
    """Create document response"""

    status: str
    document_id: Optional[str] = None
    title: str
    web_view_link: Optional[str] = None
    error: Optional[str] = None


class ColorValue(BaseModel):
    """RGB color value"""

    red: float = Field(..., ge=0.0, le=1.0, description="Red component (0.0-1.0)")
    green: float = Field(..., ge=0.0, le=1.0, description="Green component (0.0-1.0)")
    blue: float = Field(..., ge=0.0, le=1.0, description="Blue component (0.0-1.0)")


class ModifyDocTextRequest(BaseModel):
    """Modify document text request"""

    document_id: str = Field(..., min_length=1, description="Document ID")
    start_index: int = Field(..., ge=0, description="Start position (0-based)")
    end_index: Optional[int] = Field(
        None, ge=0, description="End position for replacement"
    )
    text: Optional[str] = Field(None, description="Text to insert/replace")
    bold: Optional[bool] = Field(None, description="Make text bold")
    italic: Optional[bool] = Field(None, description="Make text italic")
    underline: Optional[bool] = Field(None, description="Underline text")
    font_size: Optional[int] = Field(
        None, ge=6, le=400, description="Font size in points"
    )
    font_family: Optional[str] = Field(
        None, max_length=100, description="Font family name"
    )
    text_color: Optional[str] = Field(
        None, pattern=r"^#[0-9A-Fa-f]{6}$", description="Text color (#RRGGBB)"
    )
    background_color: Optional[str] = Field(
        None, pattern=r"^#[0-9A-Fa-f]{6}$", description="Background color (#RRGGBB)"
    )


class ModifyDocTextResponse(BaseModel):
    """Modify document text response"""

    status: str
    document_id: str
    operations: List[str]
    web_view_link: str
    error: Optional[str] = None


class FindAndReplaceDocRequest(BaseModel):
    """Find and replace request"""

    document_id: str = Field(..., min_length=1, description="Document ID")
    find_text: str = Field(..., min_length=1, description="Text to search for")
    replace_text: str = Field(..., description="Text to replace with")
    match_case: bool = Field(False, description="Match case exactly")


class FindAndReplaceDocResponse(BaseModel):
    """Find and replace response"""

    status: str
    document_id: str
    replacements: int
    find_text: str
    replace_text: str
    web_view_link: str
    error: Optional[str] = None


class InsertDocElementsRequest(BaseModel):
    """Insert document elements request"""

    document_id: str = Field(..., min_length=1, description="Document ID")
    element_type: str = Field(
        ..., pattern=r"^(table|list|page_break)$", description="Element type"
    )
    index: int = Field(..., ge=0, description="Position to insert (0-based)")
    rows: Optional[int] = Field(
        None, ge=1, le=20, description="Number of rows (for table)"
    )
    columns: Optional[int] = Field(
        None, ge=1, le=20, description="Number of columns (for table)"
    )
    list_type: Optional[str] = Field(
        None, pattern=r"^(UNORDERED|ORDERED)$", description="List type"
    )
    text: Optional[str] = Field(None, description="Initial text for list items")

    @validator("rows")
    def validate_rows_for_table(cls, v, values):
        if values.get("element_type") == "table" and v is None:
            raise ValueError("rows is required for table element")
        return v

    @validator("columns")
    def validate_columns_for_table(cls, v, values):
        if values.get("element_type") == "table" and v is None:
            raise ValueError("columns is required for table element")
        return v

    @validator("list_type")
    def validate_list_type_for_list(cls, v, values):
        if values.get("element_type") == "list" and v is None:
            raise ValueError("list_type is required for list element")
        return v


class InsertDocElementsResponse(BaseModel):
    """Insert document elements response"""

    status: str
    document_id: str
    element_type: str
    index: int
    web_view_link: str
    error: Optional[str] = None


class InsertDocImageRequest(BaseModel):
    """Insert document image request"""

    document_id: str = Field(..., min_length=1, description="Document ID")
    image_source: str = Field(
        ..., min_length=1, description="Drive file ID or image URL"
    )
    index: int = Field(..., ge=0, description="Position to insert (0-based)")
    width: int = Field(0, ge=0, le=1584, description="Image width in points")
    height: int = Field(0, ge=0, le=1584, description="Image height in points")


class InsertDocImageResponse(BaseModel):
    """Insert document image response"""

    status: str
    document_id: str
    image_source: str
    index: int
    web_view_link: str
    error: Optional[str] = None


class UpdateDocHeadersFootersRequest(BaseModel):
    """Update headers/footers request"""

    document_id: str = Field(..., min_length=1, description="Document ID")
    section_type: str = Field(
        ..., pattern=r"^(header|footer)$", description="Section type"
    )
    content: str = Field(
        ..., min_length=1, max_length=1000, description="Header/footer content"
    )
    header_footer_type: str = Field(
        "DEFAULT",
        pattern=r"^(DEFAULT|FIRST_PAGE_ONLY|EVEN_PAGE)$",
        description="Header/footer type",
    )


class UpdateDocHeadersFootersResponse(BaseModel):
    """Update headers/footers response"""

    status: str
    document_id: str
    section_type: str
    header_footer_type: str
    web_view_link: str
    error: Optional[str] = None


class DocOperation(BaseModel):
    """Document operation for batch update"""

    type: str = Field(
        ...,
        pattern=r"^(insert_text|delete_text|replace_text|format_text|insert_table|insert_page_break)$",
        description="Operation type",
    )
    index: Optional[int] = Field(None, ge=0, description="Position for operation")
    start_index: Optional[int] = Field(None, ge=0, description="Start position")
    end_index: Optional[int] = Field(None, ge=0, description="End position")
    text: Optional[str] = Field(None, description="Text content")
    rows: Optional[int] = Field(None, ge=1, le=20, description="Table rows")
    columns: Optional[int] = Field(None, ge=1, le=20, description="Table columns")
    bold: Optional[bool] = None
    italic: Optional[bool] = None
    underline: Optional[bool] = None
    font_size: Optional[int] = Field(None, ge=6, le=400)
    font_family: Optional[str] = None


class BatchUpdateDocRequest(BaseModel):
    """Batch update document request"""

    document_id: str = Field(..., min_length=1, description="Document ID")
    operations: List[DocOperation] = Field(
        ..., min_items=1, max_items=500, description="List of operations"
    )


class BatchUpdateDocResponse(BaseModel):
    """Batch update document response"""

    status: str
    document_id: str
    operations_count: int
    web_view_link: str
    error: Optional[str] = None


class InspectDocStructureRequest(BaseModel):
    """Inspect document structure request"""

    document_id: str = Field(..., min_length=1, description="Document ID")
    detailed: bool = Field(False, description="Return detailed information")


class InspectDocStructureResponse(BaseModel):
    """Inspect document structure response"""

    status: str
    document_id: str
    structure: dict
    error: Optional[str] = None


class CreateTableWithDataRequest(BaseModel):
    """Create table with data request"""

    document_id: str = Field(..., min_length=1, description="Document ID")
    table_data: List[List[str]] = Field(
        ..., min_items=1, description="Table data (rows x columns)"
    )
    index: int = Field(..., ge=0, description="Position to insert (0-based)")
    bold_headers: bool = Field(True, description="Make first row bold")

    @validator("table_data")
    def validate_table_data(cls, v):
        if not v or not v[0]:
            raise ValueError("table_data must contain at least one row with data")
        col_count = len(v[0])
        for row in v:
            if len(row) != col_count:
                raise ValueError("All rows must have the same number of columns")
        return v


class CreateTableWithDataResponse(BaseModel):
    """Create table with data response"""

    status: str
    document_id: str
    rows: int
    columns: int
    web_view_link: str
    error: Optional[str] = None


class DebugTableStructureRequest(BaseModel):
    """Debug table structure request"""

    document_id: str = Field(..., min_length=1, description="Document ID")
    table_index: int = Field(0, ge=0, description="Table index")


class DebugTableStructureResponse(BaseModel):
    """Debug table structure response"""

    status: str
    document_id: str
    table_index: int
    structure: dict
    error: Optional[str] = None


class ExportDocToPdfRequest(BaseModel):
    """Export document to PDF request"""

    document_id: str = Field(..., min_length=1, description="Document ID")
    pdf_filename: Optional[str] = Field(
        None, max_length=255, description="PDF filename"
    )
    folder_id: Optional[str] = Field(None, description="Drive folder ID")


class ExportDocToPdfResponse(BaseModel):
    """Export document to PDF response"""

    status: str
    document_id: str
    pdf_id: Optional[str] = None
    pdf_filename: str
    web_view_link: Optional[str] = None
    error: Optional[str] = None


class ReadDocCommentsRequest(BaseModel):
    """Read document comments request"""

    document_id: str = Field(..., min_length=1, description="Document ID")


class CommentInfo(BaseModel):
    """Comment information"""

    id: str
    content: str
    author: str
    created_time: str
    resolved: bool
    replies: List[dict] = []


class ReadDocCommentsResponse(BaseModel):
    """Read document comments response"""

    status: str
    document_id: str
    count: int
    comments: List[CommentInfo]
    error: Optional[str] = None


class CreateDocCommentRequest(BaseModel):
    """Create document comment request"""

    document_id: str = Field(..., min_length=1, description="Document ID")
    content: str = Field(
        ..., min_length=1, max_length=2000, description="Comment content"
    )
    start_index: int = Field(..., ge=0, description="Start index for comment")
    end_index: int = Field(..., ge=0, description="End index for comment")

    @validator("end_index")
    def validate_end_index(cls, v, values):
        if "start_index" in values and v <= values["start_index"]:
            raise ValueError("end_index must be greater than start_index")
        return v


class CreateDocCommentResponse(BaseModel):
    """Create document comment response"""

    status: str
    comment_id: Optional[str] = None
    document_id: str
    error: Optional[str] = None


class ReplyToCommentRequest(BaseModel):
    """Reply to comment request"""

    document_id: str = Field(..., min_length=1, description="Document ID")
    comment_id: str = Field(..., min_length=1, description="Comment ID")
    reply_content: str = Field(
        ..., min_length=1, max_length=2000, description="Reply content"
    )


class ReplyToCommentResponse(BaseModel):
    """Reply to comment response"""

    status: str
    reply_id: Optional[str] = None
    comment_id: str
    document_id: str
    error: Optional[str] = None


class ResolveCommentRequest(BaseModel):
    """Resolve comment request"""

    document_id: str = Field(..., min_length=1, description="Document ID")
    comment_id: str = Field(..., min_length=1, description="Comment ID")


class ResolveCommentResponse(BaseModel):
    """Resolve comment response"""

    status: str
    comment_id: str
    document_id: str
    error: Optional[str] = None
    """Update Drive file response"""

    status: str
    message: str
    error: Optional[str] = None
    """Create sheet request"""

    spreadsheet_id: str = Field(..., min_length=1, description="Spreadsheet ID")
    sheet_name: str = Field(..., min_length=1, max_length=100, description="Sheet name")
