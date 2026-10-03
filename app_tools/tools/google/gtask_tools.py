"""
Google Tasks Tools

This module provides tools for interacting with Google Tasks API.
"""

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
from pathlib import Path

from googleapiclient.errors import HttpError

from app_tools.auth.service_decoder import get_google_service
from app_tools.core.server_init import planning_server
from app_tools.helper.pydantic_models import (
    TaskListInfo,
    ListTaskListsRequest,
    ListTaskListsResponse,
    GetTaskListRequest,
    GetTaskListResponse,
    CreateTaskListRequest,
    CreateTaskListResponse,
    UpdateTaskListRequest,
    UpdateTaskListResponse,
    DeleteTaskListRequest,
    DeleteTaskListResponse,
    ListTasksRequest,
    ListTasksResponse,
    GetTaskRequest,
    GetTaskResponse,
    CreateTaskRequest,
    CreateTaskResponse,
    UpdateTaskRequest,
    UpdateTaskResponse,
    DeleteTaskRequest,
    DeleteTaskResponse,
    MoveTaskRequest,
    MoveTaskResponse,
    ClearCompletedTasksRequest,
    ClearCompletedTasksResponse,
)
from utils.helper import setup_logger

logger = setup_logger(__name__)


LIST_TASKS_MAX_RESULTS_DEFAULT = 20
LIST_TASKS_MAX_RESULTS_MAX = 10_000
LIST_TASKS_MAX_POSITION = "99999999999999999999"


def get_service():
    """Get Gmail service using shared authentication."""
    base_dir = Path(__file__).parent.parent.parent
    token_path = str(base_dir / "cred" / "gtask_token.json")
    creds_path = str(base_dir / "cred" / "setup_cred.json")

    return get_google_service(
        service_type="tasks",
        scope_key="tasks",
        token_path=token_path,
        creds_path=creds_path,
    )


class StructuredTask:
    def __init__(self, task: Dict[str, str], is_placeholder_parent: bool) -> None:
        self.id = task["id"]
        self.title = task.get("title", None)
        self.status = task.get("status", None)
        self.due = task.get("due", None)
        self.notes = task.get("notes", None)
        self.updated = task.get("updated", None)
        self.completed = task.get("completed", None)
        self.is_placeholder_parent = is_placeholder_parent
        self.subtasks: List["StructuredTask"] = []

    def add_subtask(self, subtask: "StructuredTask") -> None:
        self.subtasks.append(subtask)

    def __repr__(self) -> str:
        return f"StructuredTask(title={self.title}, {len(self.subtasks)} subtasks)"


def _adjust_due_max_for_tasks_api(due_max: str) -> str:
    """
    Compensate for the Google Tasks API treating dueMax as an exclusive bound.

    The API stores due dates at day resolution and compares using < dueMax, so to
    include tasks due on the requested date we bump the bound by one day.
    """
    try:
        parsed = datetime.fromisoformat(due_max.replace("Z", "+00:00"))
    except ValueError:
        logger.warning(
            "[list_tasks] Unable to parse due_max '%s'; sending unmodified value",
            due_max,
        )
        return due_max

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)

    adjusted = parsed + timedelta(days=1)
    if adjusted.tzinfo == timezone.utc:
        return adjusted.isoformat().replace("+00:00", "Z")
    return adjusted.isoformat()


@planning_server.tool()
async def list_task_lists(
    max_results: int = 1000,
    page_token: Optional[str] = None,
) -> str:
    """
    List all task lists for the user.

    Args:
        max_results (int): Maximum number of task lists to return (default: 1000, max: 1000).
        page_token (Optional[str]): Token for pagination.

    Returns:
        str: List of task lists with their IDs, titles, and details.
    """
    # Validate input parameters
    try:
        request = ListTaskListsRequest(max_results=max_results, page_token=page_token)
    except Exception as e:
        error_msg = f"Invalid parameters: {str(e)}"
        logger.error(f"[list_task_lists] {error_msg}")
        return ListTaskListsResponse(
            status="error", count=0, task_lists=[], error=error_msg
        ).model_dump_json(indent=2)

    logger.info("[list_task_lists] Invoked.")
    service = get_service()

    try:
        params: Dict[str, Any] = {}
        if request.max_results is not None:
            params["maxResults"] = request.max_results
        if request.page_token:
            params["pageToken"] = request.page_token

        result = await asyncio.to_thread(service.tasklists().list(**params).execute)

        task_lists_raw = result.get("items", [])
        next_page_token = result.get("nextPageToken")

        # Convert to Pydantic models
        task_lists = [
            TaskListInfo(id=tl["id"], title=tl["title"], updated=tl.get("updated"))
            for tl in task_lists_raw
        ]

        logger.info(f"Found {len(task_lists)} task lists")
        return ListTaskListsResponse(
            status="success",
            count=len(task_lists),
            task_lists=task_lists,
            next_page_token=next_page_token,
        ).model_dump_json(indent=2)

    except HttpError as error:
        error_msg = f"API error: {error}"
        logger.error(f"[list_task_lists] {error_msg}")
        return ListTaskListsResponse(
            status="error", count=0, task_lists=[], error=error_msg
        ).model_dump_json(indent=2)
    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        logger.error(f"[list_task_lists] {error_msg}")
        return ListTaskListsResponse(
            status="error", count=0, task_lists=[], error=error_msg
        ).model_dump_json(indent=2)


@planning_server.tool()  # type: ignore
async def get_task_list(task_list_id: str) -> str:
    """
    Get details of a specific task list.

    Args:
        task_list_id (str): The ID of the task list to retrieve.

    Returns:
        str: Task list details including title, ID, and last updated time.
    """
    # Validate input parameters
    try:
        request = GetTaskListRequest(task_list_id=task_list_id)
    except Exception as e:
        error_msg = f"Invalid parameters: {str(e)}"
        logger.error(f"[get_task_list] {error_msg}")
        return GetTaskListResponse(
            status="error", message="", error=error_msg
        ).model_dump_json(indent=2)

    logger.info(f"[get_task_list] Invoked. Task List ID: {request.task_list_id}")
    service = get_service()
    try:
        task_list = await asyncio.to_thread(
            service.tasklists().get(tasklist=request.task_list_id).execute
        )

        message = f"""Task List Details:
- Title: {task_list["title"]}
- ID: {task_list["id"]}
- Updated: {task_list.get("updated", "N/A")}
- Self Link: {task_list.get("selfLink", "N/A")}"""

        logger.info(f"Retrieved task list '{task_list['title']}'")
        return GetTaskListResponse(status="success", message=message).model_dump_json(
            indent=2
        )

    except HttpError as error:
        error_msg = f"API error: {error}"
        logger.error(f"[get_task_list] {error_msg}")
        return GetTaskListResponse(
            status="error", message="", error=error_msg
        ).model_dump_json(indent=2)
    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        logger.error(f"[get_task_list] {error_msg}")
        return GetTaskListResponse(
            status="error", message="", error=error_msg
        ).model_dump_json(indent=2)


@planning_server.tool()
async def create_task_list(title: str) -> str:
    """
    Create a new task list.

    Args:
        title (str): The title of the new task list.

    Returns:
        str: Confirmation message with the new task list ID and details.
    """
    # Validate input parameters
    try:
        request = CreateTaskListRequest(title=title)
    except Exception as e:
        error_msg = f"Invalid parameters: {str(e)}"
        logger.error(f"[create_task_list] {error_msg}")
        return CreateTaskListResponse(
            status="error", message="", task_list_id=None, error=error_msg
        ).model_dump_json(indent=2)

    logger.info(f"[create_task_list] Invoked. Title: '{request.title}'")
    service = get_service()

    try:
        body = {"title": request.title}

        result = await asyncio.to_thread(service.tasklists().insert(body=body).execute)

        message = f"""Task List Created:
- Title: {result["title"]}
- ID: {result["id"]}
- Created: {result.get("updated", "N/A")}
- Self Link: {result.get("selfLink", "N/A")}"""

        logger.info(f"Created task list '{request.title}' with ID {result['id']}")
        return CreateTaskListResponse(
            status="success", message=message, task_list_id=result["id"]
        ).model_dump_json(indent=2)

    except HttpError as error:
        error_msg = f"API error: {error}"
        logger.error(f"[create_task_list] {error_msg}")
        return CreateTaskListResponse(
            status="error", message="", task_list_id=None, error=error_msg
        ).model_dump_json(indent=2)
    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        logger.error(f"[create_task_list] {error_msg}")
        return CreateTaskListResponse(
            status="error", message="", task_list_id=None, error=error_msg
        ).model_dump_json(indent=2)


@planning_server.tool()
async def update_task_list(task_list_id: str, title: str) -> str:
    """
    Update an existing task list.

    Args:
        task_list_id (str): The ID of the task list to update.
        title (str): The new title for the task list.

    Returns:
        str: Confirmation message with updated task list details.
    """
    # Validate input parameters
    try:
        request = UpdateTaskListRequest(task_list_id=task_list_id, title=title)
    except Exception as e:
        error_msg = f"Invalid parameters: {str(e)}"
        logger.error(f"[update_task_list] {error_msg}")
        return UpdateTaskListResponse(
            status="error", message="", error=error_msg
        ).model_dump_json(indent=2)

    logger.info(
        f"[update_task_list] Invoked. Task List ID: {request.task_list_id}, New Title: '{request.title}'"
    )
    service = get_service()

    try:
        body = {"id": request.task_list_id, "title": request.title}

        result = await asyncio.to_thread(
            service.tasklists().update(tasklist=request.task_list_id, body=body).execute
        )

        message = f"""Task List Updated:
- Title: {result["title"]}
- ID: {result["id"]}
- Updated: {result.get("updated", "N/A")}"""

        logger.info(
            f"Updated task list {request.task_list_id} with new title '{request.title}'"
        )
        return UpdateTaskListResponse(
            status="success", message=message, task_list_id=request.task_list_id
        ).model_dump_json(indent=2)

    except HttpError as error:
        error_msg = f"API error: {error}"
        logger.error(f"[update_task_list] {error_msg}")
        return UpdateTaskListResponse(
            status="error", message="", error=error_msg
        ).model_dump_json(indent=2)
    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        logger.error(f"[update_task_list] {error_msg}")
        return UpdateTaskListResponse(
            status="error", message="", error=error_msg
        ).model_dump_json(indent=2)


@planning_server.tool()
async def delete_task_list(task_list_id: str) -> str:
    """
    Delete a task list. Note: This will also delete all tasks in the list.

    Args:
        task_list_id (str): The ID of the task list to delete.

    Returns:
        str: Confirmation message.
    """
    # Validate input parameters
    try:
        request = DeleteTaskListRequest(task_list_id=task_list_id)
    except Exception as e:
        error_msg = f"Invalid parameters: {str(e)}"
        logger.error(f"[delete_task_list] {error_msg}")
        return DeleteTaskListResponse(
            status="error", message="", error=error_msg
        ).model_dump_json(indent=2)

    logger.info(f"[delete_task_list] Invoked. Task List ID: {request.task_list_id}")
    service = get_service()
    try:
        await asyncio.to_thread(
            service.tasklists().delete(tasklist=request.task_list_id).execute
        )

        message = f"Task list {request.task_list_id} has been deleted. All tasks in this list have also been deleted."

        logger.info(f"Deleted task list {request.task_list_id}")
        return DeleteTaskListResponse(
            status="success", message=message
        ).model_dump_json(indent=2)

    except HttpError as error:
        error_msg = f"API error: {error}"
        logger.error(f"[delete_task_list] {error_msg}")
        return DeleteTaskListResponse(
            status="error", message="", error=error_msg
        ).model_dump_json(indent=2)
    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        logger.error(f"[delete_task_list] {error_msg}")
        return DeleteTaskListResponse(
            status="error", message="", error=error_msg
        ).model_dump_json(indent=2)


@planning_server.tool()
async def list_tasks(
    task_list_id: str,
    max_results: int = LIST_TASKS_MAX_RESULTS_DEFAULT,
    page_token: Optional[str] = None,
    show_completed: bool = True,
    show_deleted: bool = False,
    show_hidden: bool = False,
    show_assigned: bool = False,
    completed_max: Optional[str] = None,
    completed_min: Optional[str] = None,
    due_max: Optional[str] = None,
    due_min: Optional[str] = None,
    updated_min: Optional[str] = None,
) -> str:
    """
    List all tasks in a specific task list.

    Args:
        task_list_id (str): The ID of the task list to retrieve tasks from.
        max_results (int): Maximum number of tasks to return. (default: 20, max: 10000).
        page_token (Optional[str]): Token for pagination.
        show_completed (bool): Whether to include completed tasks (default: True). Note that show_hidden must also be true to show tasks completed in first party clients, such as the web UI and Google's mobile apps.
        show_deleted (bool): Whether to include deleted tasks (default: False).
        show_hidden (bool): Whether to include hidden tasks (default: False).
        show_assigned (bool): Whether to include assigned tasks (default: False).
        completed_max (Optional[str]): Upper bound for completion date (RFC 3339 timestamp).
        completed_min (Optional[str]): Lower bound for completion date (RFC 3339 timestamp).
        due_max (Optional[str]): Upper bound for due date (RFC 3339 timestamp).
        due_min (Optional[str]): Lower bound for due date (RFC 3339 timestamp).
        updated_min (Optional[str]): Lower bound for last modification time (RFC 3339 timestamp).

    Returns:
        str: List of tasks with their details.
    """
    # Validate input parameters
    try:
        request = ListTasksRequest(
            task_list_id=task_list_id,
            max_results=max_results,
            page_token=page_token,
            show_completed=show_completed,
            show_deleted=show_deleted,
            show_hidden=show_hidden,
            show_assigned=show_assigned,
            completed_max=completed_max,
            completed_min=completed_min,
            due_max=due_max,
            due_min=due_min,
            updated_min=updated_min,
        )
    except Exception as e:
        error_msg = f"Invalid parameters: {str(e)}"
        logger.error(f"[list_tasks] {error_msg}")
        return ListTasksResponse(
            status="error", message="", error=error_msg
        ).model_dump_json(indent=2)

    logger.info(f"[list_tasks] Invoked. Task List ID: {request.task_list_id}")
    service = get_service()
    try:
        params: Dict[str, Any] = {"tasklist": request.task_list_id}
        if request.max_results is not None:
            params["maxResults"] = request.max_results
        if request.page_token:
            params["pageToken"] = request.page_token
        if request.show_completed is not None:
            params["showCompleted"] = request.show_completed
        if request.show_deleted is not None:
            params["showDeleted"] = request.show_deleted
        if request.show_hidden is not None:
            params["showHidden"] = request.show_hidden
        if request.show_assigned is not None:
            params["showAssigned"] = request.show_assigned
        if request.completed_max:
            params["completedMax"] = request.completed_max
        if request.completed_min:
            params["completedMin"] = request.completed_min
        if request.due_max:
            adjusted_due_max = _adjust_due_max_for_tasks_api(due_max)
            if adjusted_due_max != due_max:
                logger.info(
                    "[list_tasks] Adjusted due_max from '%s' to '%s' to include due date boundary",
                    due_max,
                    adjusted_due_max,
                )
            params["dueMax"] = adjusted_due_max
        if due_min:
            params["dueMin"] = due_min
        if updated_min:
            params["updatedMin"] = updated_min

        result = await asyncio.to_thread(service.tasks().list(**params).execute)

        tasks = result.get("items", [])
        next_page_token = result.get("nextPageToken")

        # In order to return a sorted and organized list of tasks all at once, we support retrieving more than a single
        # page from the Google tasks API.
        results_remaining = (
            min(max_results, LIST_TASKS_MAX_RESULTS_MAX)
            if max_results
            else LIST_TASKS_MAX_RESULTS_DEFAULT
        )
        results_remaining -= len(tasks)
        while results_remaining > 0 and next_page_token:
            params["pageToken"] = next_page_token
            params["maxResults"] = str(results_remaining)
            result = await asyncio.to_thread(service.tasks().list(**params).execute)
            more_tasks = result.get("items", [])
            next_page_token = result.get("nextPageToken")
            if len(more_tasks) == 0:
                # For some unexpected reason, no more tasks were returned. Break to avoid an infinite loop.
                break
            tasks.extend(more_tasks)
            results_remaining -= len(more_tasks)

        if not tasks:
            return f"No tasks found in task list {task_list_id}."

        structured_tasks = get_structured_tasks(tasks)

        response = f"Tasks in list {task_list_id}:\n"
        response += serialize_tasks(structured_tasks, 0)

        if next_page_token:
            response += f"Next page token: {next_page_token}\n"

        logger.info(f"Found {len(tasks)} tasks in list {task_list_id}")
        return response

    except HttpError as error:
        # Report like the other task tools instead of raising, and don't point
        # the model at an auth tool that doesn't exist.
        message = f"API error: {error}"
        logger.error(message, exc_info=True)
        return message
    except Exception as e:
        message = f"Unexpected error: {e}."
        logger.exception(message)
        return message


def get_structured_tasks(tasks: List[Dict[str, str]]) -> List[StructuredTask]:
    """
    Convert a flat list of task dictionaries into StructuredTask objects based on parent-child relationships sorted by position.

    Args:
        tasks: List of task dictionaries.

    Returns:
        list: Sorted list of top-level StructuredTask objects with nested subtasks.
    """
    tasks_by_id = {
        task["id"]: StructuredTask(task, is_placeholder_parent=False) for task in tasks
    }
    positions_by_id = {
        task["id"]: int(task["position"]) for task in tasks if "position" in task
    }

    # Placeholder virtual root as parent for top-level tasks
    root_task = StructuredTask(
        {"id": "root", "title": "Root"}, is_placeholder_parent=False
    )

    for task in tasks:
        structured_task = tasks_by_id[task["id"]]
        parent_id = task.get("parent")
        parent = None

        if not parent_id:
            # Task without parent: parent to the virtual root
            parent = root_task
        elif parent_id in tasks_by_id:
            # Subtask: parent to its actual parent
            parent = tasks_by_id[parent_id]
        else:
            # Orphaned subtask: create placeholder parent
            # Due to paging or filtering, a subtask may have a parent that is not present in the list of tasks.
            # We will create placeholder StructuredTask objects for these missing parents to maintain the hierarchy.
            parent = StructuredTask({"id": parent_id}, is_placeholder_parent=True)
            tasks_by_id[parent_id] = parent
            root_task.add_subtask(parent)

        parent.add_subtask(structured_task)

    sort_structured_tasks(root_task, positions_by_id)
    return root_task.subtasks


def sort_structured_tasks(
    root_task: StructuredTask, positions_by_id: Dict[str, int]
) -> None:
    """
    Recursively sort--in place--StructuredTask objects and their subtasks based on position.

    Args:
        root_task: The root StructuredTask object.
        positions_by_id: Dictionary mapping task IDs to their positions.
    """

    def get_position(task: StructuredTask) -> int | float:
        # Tasks without position go to the end (infinity)
        result = positions_by_id.get(task.id, float("inf"))
        return result

    root_task.subtasks.sort(key=get_position)
    for subtask in root_task.subtasks:
        sort_structured_tasks(subtask, positions_by_id)


def serialize_tasks(structured_tasks: List[StructuredTask], subtask_level: int) -> str:
    """
    Serialize a list of StructuredTask objects into a formatted string with indentation for subtasks.
    Args:
        structured_tasks (list): List of StructuredTask objects.
        subtask_level (int): Current level of indentation for subtasks.

    Returns:
        str: Formatted string representation of the tasks.
    """
    response = ""
    placeholder_parent_count = 0
    placeholder_parent_title = "Unknown parent"
    for task in structured_tasks:
        indent = "  " * subtask_level
        bullet = "-" if subtask_level == 0 else "*"
        if task.title is not None:
            title = task.title
        elif task.is_placeholder_parent:
            title = placeholder_parent_title
            placeholder_parent_count += 1
        else:
            title = "Untitled"
        response += f"{indent}{bullet} {title} (ID: {task.id})\n"
        response += f"{indent}  Status: {task.status or 'N/A'}\n"
        response += f"{indent}  Due: {task.due}\n" if task.due else ""
        if task.notes:
            response += f"{indent}  Notes: {task.notes[:100]}{'...' if len(task.notes) > 100 else ''}\n"
        response += f"{indent}  Completed: {task.completed}\n" if task.completed else ""
        response += f"{indent}  Updated: {task.updated or 'N/A'}\n"
        response += "\n"

        response += serialize_tasks(task.subtasks, subtask_level + 1)

    if placeholder_parent_count > 0:
        # Placeholder parents should only appear at the top level
        assert subtask_level == 0
        response += f"""
{placeholder_parent_count} tasks with title {placeholder_parent_title} are included as placeholders.
These placeholders contain subtasks whose parents were not present in the task list.
This can occur due to pagination. Callers can often avoid this problem if max_results is large enough to contain all tasks (subtasks and their parents) without paging.
This can also occur due to filtering that excludes parent tasks while including their subtasks or due to deleted or hidden parent tasks.
"""

    return response


@planning_server.tool()
async def get_task(task_list_id: str, task_id: str) -> str:
    """
    Get details of a specific task.

    Args:
        task_list_id (str): The ID of the task list containing the task.
        task_id (str): The ID of the task to retrieve.

    Returns:
        str: Task details including title, notes, status, due date, etc.
    """
    # Validate input parameters
    try:
        request = GetTaskRequest(task_list_id=task_list_id, task_id=task_id)
    except Exception as e:
        error_msg = f"Invalid parameters: {str(e)}"
        logger.error(f"[get_task] {error_msg}")
        return GetTaskResponse(
            status="error", message="", error=error_msg
        ).model_dump_json(indent=2)

    logger.info(
        f"[get_task] Invoked. Task List ID: {request.task_list_id}, Task ID: {request.task_id}"
    )
    service = get_service()
    try:
        task = await asyncio.to_thread(
            service.tasks()
            .get(tasklist=request.task_list_id, task=request.task_id)
            .execute
        )

        message = f"""Task Details:
- Title: {task.get("title", "Untitled")}
- ID: {task["id"]}
- Status: {task.get("status", "N/A")}
- Updated: {task.get("updated", "N/A")}"""

        if task.get("due"):
            message += f"\n- Due: {task.get('due')}"
        if task.get("completed"):
            message += f"\n- Completed: {task.get('completed')}"
        if task.get("notes"):
            message += f"\n- Notes: {task.get('notes')}"
        if task.get("parent"):
            message += f"\n- Parent Task ID: {task.get('parent')}"
        if task.get("position"):
            message += f"\n- Position: {task.get('position')}"
        if task.get("selfLink"):
            message += f"\n- Self Link: {task.get('selfLink')}"
        if task.get("webViewLink"):
            message += f"\n- Web View Link: {task.get('webViewLink')}"

        logger.info(f"Retrieved task '{task.get('title', 'Untitled')}'")
        return GetTaskResponse(status="success", message=message).model_dump_json(
            indent=2
        )

    except HttpError as error:
        error_msg = f"API error: {error}"
        logger.error(f"[get_task] {error_msg}")
        return GetTaskResponse(
            status="error", message="", error=error_msg
        ).model_dump_json(indent=2)
    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        logger.error(f"[get_task] {error_msg}")
        return GetTaskResponse(
            status="error", message="", error=error_msg
        ).model_dump_json(indent=2)


@planning_server.tool()
async def create_task(
    task_list_id: str,
    title: str,
    notes: Optional[str] = None,
    due: Optional[str] = None,
    parent: Optional[str] = None,
    previous: Optional[str] = None,
) -> str:
    """
    Create a new task in a task list.

    Args:
        task_list_id (str): The ID of the task list to create the task in.
        title (str): The title of the task.
        notes (Optional[str]): Notes/description for the task.
        due (Optional[str]): Due date in RFC 3339 format (e.g., "2024-12-31T23:59:59Z").
        parent (Optional[str]): Parent task ID (for subtasks).
        previous (Optional[str]): Previous sibling task ID (for positioning).

    Returns:
        str: Confirmation message with the new task ID and details.
    """
    # Validate input parameters
    try:
        request = CreateTaskRequest(
            task_list_id=task_list_id,
            title=title,
            notes=notes,
            due=due,
            parent=parent,
            previous=previous,
        )
    except Exception as e:
        error_msg = f"Invalid parameters: {str(e)}"
        logger.error(f"[create_task] {error_msg}")
        return CreateTaskResponse(
            status="error", message="", task_id=None, error=error_msg
        ).model_dump_json(indent=2)

    service = get_service()
    logger.info(
        f"[create_task] Invoked. Task List ID: {request.task_list_id}, Title: '{request.title}'"
    )

    try:
        body = {"title": request.title}
        if request.notes:
            body["notes"] = request.notes
        if request.due:
            body["due"] = request.due

        params = {"tasklist": request.task_list_id, "body": body}
        if request.parent:
            params["parent"] = request.parent
        if request.previous:
            params["previous"] = request.previous

        result = await asyncio.to_thread(service.tasks().insert(**params).execute)

        message = f"""Task Created:
- Title: {result["title"]}
- ID: {result["id"]}
- Status: {result.get("status", "N/A")}
- Updated: {result.get("updated", "N/A")}"""

        if result.get("due"):
            message += f"\n- Due: {result.get('due')}"
        if result.get("notes"):
            message += f"\n- Notes: {result.get('notes')}"
        if result.get("webViewLink"):
            message += f"\n- Web View Link: {result.get('webViewLink')}"

        logger.info(f"Created task '{request.title}' with ID {result['id']}")
        return CreateTaskResponse(
            status="success", message=message, task_id=result["id"]
        ).model_dump_json(indent=2)

    except HttpError as error:
        error_msg = f"API error: {error}"
        logger.error(f"[create_task] {error_msg}")
        return CreateTaskResponse(
            status="error", message="", task_id=None, error=error_msg
        ).model_dump_json(indent=2)
    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        logger.error(f"[create_task] {error_msg}")
        return CreateTaskResponse(
            status="error", message="", task_id=None, error=error_msg
        ).model_dump_json(indent=2)


@planning_server.tool()
async def update_task(
    task_list_id: str,
    task_id: str,
    title: Optional[str] = None,
    notes: Optional[str] = None,
    status: Optional[str] = None,
    due: Optional[str] = None,
) -> str:
    """
    Update an existing task.

    Args:
        task_list_id (str): The ID of the task list containing the task.
        task_id (str): The ID of the task to update.
        title (Optional[str]): New title for the task.
        notes (Optional[str]): New notes/description for the task.
        status (Optional[str]): New status ("needsAction" or "completed").
        due (Optional[str]): New due date in RFC 3339 format.

    Returns:
        str: Confirmation message with updated task details.
    """
    # Validate input parameters
    try:
        request = UpdateTaskRequest(
            task_list_id=task_list_id,
            task_id=task_id,
            title=title,
            notes=notes,
            status=status,
            due=due,
        )
    except Exception as e:
        error_msg = f"Invalid parameters: {str(e)}"
        logger.error(f"[update_task] {error_msg}")
        return UpdateTaskResponse(
            status="error", message="", error=error_msg
        ).model_dump_json(indent=2)

    service = get_service()
    logger.info(
        f"[update_task] Invoked. Task List ID: {request.task_list_id}, Task ID: {request.task_id}"
    )

    try:
        # First get the current task to build the update body
        current_task = await asyncio.to_thread(
            service.tasks()
            .get(tasklist=request.task_list_id, task=request.task_id)
            .execute
        )

        body = {
            "id": request.task_id,
            "title": request.title
            if request.title is not None
            else current_task.get("title", ""),
            "status": request.status
            if request.status is not None
            else current_task.get("status", "needsAction"),
        }

        if request.notes is not None:
            body["notes"] = request.notes
        elif current_task.get("notes"):
            body["notes"] = current_task.get("notes")

        if request.due is not None:
            body["due"] = request.due
        elif current_task.get("due"):
            body["due"] = current_task.get("due")

        result = await asyncio.to_thread(
            service.tasks()
            .update(tasklist=request.task_list_id, task=request.task_id, body=body)
            .execute
        )

        message = f"""Task Updated:
- Title: {result["title"]}
- ID: {result["id"]}
- Status: {result.get("status", "N/A")}
- Updated: {result.get("updated", "N/A")}"""

        if result.get("due"):
            message += f"\n- Due: {result.get('due')}"
        if result.get("notes"):
            message += f"\n- Notes: {result.get('notes')}"
        if result.get("completed"):
            message += f"\n- Completed: {result.get('completed')}"

        logger.info(f"Updated task {request.task_id}")
        return UpdateTaskResponse(
            status="success", message=message, task_id=request.task_id
        ).model_dump_json(indent=2)

    except HttpError as error:
        error_msg = f"API error: {error}"
        logger.error(f"[update_task] {error_msg}")
        return UpdateTaskResponse(
            status="error", message="", error=error_msg
        ).model_dump_json(indent=2)
    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        logger.error(f"[update_task] {error_msg}")
        return UpdateTaskResponse(
            status="error", message="", error=error_msg
        ).model_dump_json(indent=2)


@planning_server.tool()
async def delete_task(task_list_id: str, task_id: str) -> str:
    """
    Delete a task from a task list.

    Args:
        task_list_id (str): The ID of the task list containing the task.
        task_id (str): The ID of the task to delete.

    Returns:
        str: Confirmation message.
    """
    # Validate input parameters
    try:
        request = DeleteTaskRequest(task_list_id=task_list_id, task_id=task_id)
    except Exception as e:
        error_msg = f"Invalid parameters: {str(e)}"
        logger.error(f"[delete_task] {error_msg}")
        return DeleteTaskResponse(
            status="error", message="", error=error_msg
        ).model_dump_json(indent=2)

    service = get_service()
    logger.info(
        f"[delete_task] Invoked. Task List ID: {request.task_list_id}, Task ID: {request.task_id}"
    )

    try:
        await asyncio.to_thread(
            service.tasks()
            .delete(tasklist=request.task_list_id, task=request.task_id)
            .execute
        )

        message = f"Task {request.task_id} has been deleted from task list {request.task_list_id}"

        logger.info(f"Deleted task {request.task_id}")
        return DeleteTaskResponse(status="success", message=message).model_dump_json(
            indent=2
        )

    except HttpError as error:
        error_msg = f"API error: {error}"
        logger.error(f"[delete_task] {error_msg}")
        return DeleteTaskResponse(
            status="error", message="", error=error_msg
        ).model_dump_json(indent=2)
    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        logger.error(f"[delete_task] {error_msg}")
        return DeleteTaskResponse(
            status="error", message="", error=error_msg
        ).model_dump_json(indent=2)


@planning_server.tool()
async def move_task(
    task_list_id: str,
    task_id: str,
    parent: Optional[str] = None,
    previous: Optional[str] = None,
    destination_task_list: Optional[str] = None,
) -> str:
    """
    Move a task to a different position or parent within the same list, or to a different list.

    Args:
        task_list_id (str): The ID of the current task list containing the task.
        task_id (str): The ID of the task to move.
        parent (Optional[str]): New parent task ID (for making it a subtask).
        previous (Optional[str]): Previous sibling task ID (for positioning).
        destination_task_list (Optional[str]): Destination task list ID (for moving between lists).

    Returns:
        str: Confirmation message with updated task details.
    """
    # Validate input parameters
    try:
        request = MoveTaskRequest(
            task_list_id=task_list_id,
            task_id=task_id,
            parent=parent,
            previous=previous,
            destination_task_list=destination_task_list,
        )
    except Exception as e:
        error_msg = f"Invalid parameters: {str(e)}"
        logger.error(f"[move_task] {error_msg}")
        return MoveTaskResponse(
            status="error", message="", error=error_msg
        ).model_dump_json(indent=2)

    service = get_service()
    logger.info(
        f"[move_task] Invoked. Task List ID: {request.task_list_id}, Task ID: {request.task_id}"
    )

    try:
        params = {"tasklist": request.task_list_id, "task": request.task_id}
        if request.parent:
            params["parent"] = request.parent
        if request.previous:
            params["previous"] = request.previous
        if request.destination_task_list:
            params["destinationTasklist"] = request.destination_task_list

        result = await asyncio.to_thread(service.tasks().move(**params).execute)

        message = f"""Task Moved:
- Title: {result["title"]}
- ID: {result["id"]}
- Status: {result.get("status", "N/A")}
- Updated: {result.get("updated", "N/A")}"""

        if result.get("parent"):
            message += f"\n- Parent Task ID: {result['parent']}"
        if result.get("position"):
            message += f"\n- Position: {result['position']}"

        move_details = []
        if request.destination_task_list:
            move_details.append(f"moved to task list {request.destination_task_list}")
        if request.parent:
            move_details.append(f"made a subtask of {request.parent}")
        if request.previous:
            move_details.append(f"positioned after {request.previous}")

        if move_details:
            message += f"\n- Move Details: {', '.join(move_details)}"

        logger.info(f"Moved task {request.task_id}")
        return MoveTaskResponse(status="success", message=message).model_dump_json(
            indent=2
        )

    except HttpError as error:
        error_msg = f"API error: {error}"
        logger.error(f"[move_task] {error_msg}")
        return MoveTaskResponse(
            status="error", message="", error=error_msg
        ).model_dump_json(indent=2)
    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        logger.error(f"[move_task] {error_msg}")
        return MoveTaskResponse(
            status="error", message="", error=error_msg
        ).model_dump_json(indent=2)


@planning_server.tool()
async def clear_completed_tasks(task_list_id: str) -> str:
    """
    Clear all completed tasks from a task list. The tasks will be marked as hidden.

    Args:
        task_list_id (str): The ID of the task list to clear completed tasks from.

    Returns:
        str: Confirmation message.
    """
    # Validate input parameters
    try:
        request = ClearCompletedTasksRequest(task_list_id=task_list_id)
    except Exception as e:
        error_msg = f"Invalid parameters: {str(e)}"
        logger.error(f"[clear_completed_tasks] {error_msg}")
        return ClearCompletedTasksResponse(
            status="error", message="", error=error_msg
        ).model_dump_json(indent=2)

    logger.info(
        f"[clear_completed_tasks] Invoked. Task List ID: {request.task_list_id}"
    )
    service = get_service()

    try:
        await asyncio.to_thread(
            service.tasks().clear(tasklist=request.task_list_id).execute
        )

        message = f"All completed tasks have been cleared from task list {request.task_list_id}. The tasks are now hidden and won't appear in default task list views."

        logger.info(f"Cleared completed tasks from list {request.task_list_id}")
        return ClearCompletedTasksResponse(
            status="success", message=message
        ).model_dump_json(indent=2)

    except HttpError as error:
        error_msg = f"API error: {error}"
        logger.error(f"[clear_completed_tasks] {error_msg}")
        return ClearCompletedTasksResponse(
            status="error", message="", error=error_msg
        ).model_dump_json(indent=2)
    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        logger.error(f"[clear_completed_tasks] {error_msg}")
        return ClearCompletedTasksResponse(
            status="error", message="", error=error_msg
        ).model_dump_json(indent=2)
