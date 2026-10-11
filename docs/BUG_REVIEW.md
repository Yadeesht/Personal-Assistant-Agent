# Bug review — full repository

Date: 2026-10-11 · Branch: `ccr-a5978e08-c4gf8i`

Every Python file was read: `main.py`, `config/`, `core/`, `utils/`, `rag/`,
`app_tools/` (all Google tools, helpers and Docs managers) and the tests. Each fix
below has a regression test in `tests/test_review_fixes.py`. All 23 of those tests
were checked to **fail on the original code** and pass after the fix.

**Test results:** 95 passed. Before the review: 56 passed, 16 failed. The 16
failures had one root cause, fixed in item 3.

---

## 1. Critical: features that never worked

| # | Where | Bug | Fix |
|---|---|---|---|
| 1 | `gdrive_tools.create_drive_file` | Calls with `content` (the normal case) crashed every time. `parsed_url` was only defined when `fileUrl` was given. Calls with both created two files. | Uses `content` if given, otherwise `fileUrl`. Exactly one file is created. |
| 2 | `gsearch_tools.search_custom_siterestrict` | Passed `service=` to `search_custom`, which has no such parameter, so every call raised `TypeError`. | Removed the argument. |
| 3 | `utils/helper.count_tokens` | `tiktoken` downloads its tokenizer on first use. When offline or behind a proxy, `count_tokens` raised, and it runs on every turn (summary check), so every message failed. | Loads the tokenizer once and falls back to an estimate (~4 chars/token) with a warning. |
| 4 | `calendar_tools.create_event` / `modify_event` | Custom reminders were validated into pydantic objects, then rejected as "not a dict". Defaults were turned off and no overrides were sent, so **events had no reminders at all**. | Reminders are converted back to dicts before they are sent. |
| 5 | `gform_tools.create_form` | Put `description` (and the wrong key `document_title`) into `forms.create`, which the API rejects. Confirmed against Google's discovery document. | Creates the form with `title`/`documentTitle`, then sets the description with `batchUpdate`. |
| 6 | `gform_tools.set_publish_settings` | Sent `publishAsTemplate`/`requireAuthentication`, which are not fields of this API, so every call was rejected. | Now sends `publishSettings.publishState` (`is_published`, `is_accepting_responses`). The tool's parameters changed to match. |
| 7 | `gform_tools.get_form` | Required an unused `user_google_email` argument, so the model had to invent an email. | Parameter removed. |

## 2. Agent / graph logic (`core/`, `main.py`)

| # | Bug | Fix |
|---|---|---|
| 8 | **Bare command words hijacked normal messages.** "new meeting with Bob at 5" started a fresh chat. "help me write…" showed help. "clear my calendar…" cleared the screen. "thread summary" switched threads. | `parse_command()`: a bare word (`new`, `help`, `exit`…) is a command only when it is the whole message. Slash commands work as before. |
| 9 | **Code-agent results never reached the user.** After running approved code, control passed to the supervisor without the result, so it replied from stale context. | The result is handed to the supervisor as a handoff message, the same way `work_completion` does for workers. |
| 10 | **No way to decline generated code.** "no" or "cancel" made the code agent generate code again, trapping the user. | Decline words cancel the run and hand back to the supervisor. Other replies are still treated as feedback, and approvals use `core.confirm.is_approval` plus "run it" and similar. Only the latest pending code block can be approved. |
| 11 | Rate-limit and network errors in supervisor/worker nodes were written only to `messages`. The screen shows the agent's own list, so the user saw nothing, or an old reply. | `_error_reply()` writes the error to both lists. |
| 12 | `route_to_agent` with a misspelled agent name reported "Successfully routed" while no agent took the task. | Unknown names return an error listing the valid agents. |
| 13 | Code blocks were found only with an exact `` ```python\n `` fence. `` ```py `` or a bare fence sent prose to the sandbox. | One regex handles `python`, `py` and bare fences. Also replaced the deprecated pydantic `.schema()` call. |
| 14 | A failure while drawing the knowledge-graph picture made a successful memory save report failure. | The picture step is guarded separately. |

## 3. Memory / RAG

| # | Bug | Fix |
|---|---|---|
| 15 | `delete_entity`, `delete_relationship` and `preprocess_graph` built Cypher queries with f-strings, so names with quotes (e.g. "O'Brien") broke them, and query injection was possible. | Parameterised queries. |
| 16 | `retrieve_from_knowledge_graph` returned a pandas DataFrame. As text, its columns are truncated (`...`). It also redrew a 300-dpi PNG on every lookup. | Returns JSON records, and no longer draws on reads. `visualize()` writes to an absolute path and logs instead of using `print`. |
| 17 | `retrieve_relevant_chunks` ignored a date filter unless both `start_time` and `end_time` were set, and returned `None` on failure. | Either bound works alone. Returns clear messages when nothing is stored or nothing matches. |

## 4. Google tools: smaller bugs

| # | Where | Bug | Fix |
|---|---|---|---|
| 18 | `gmail.read_email` | HTML-only emails returned an empty body. The charset was ignored. A `text/plain` attachment could be read as the body. | `extract_email_body()`: plain part first, then HTML as text. Uses the declared charset and skips attachments. |
| 19 | `gmail.get_unread_emails` | The invalid-parameter path crashed while building its own error response. `date=0` (used in the code-agent prompt example) was invalid. | Response model fixed. `date=0` now means today. |
| 20 | `gmail.send_email` | Auth and network errors were not caught (only `HttpError`). `get_service()` was called twice. | Catches all errors, single call. |
| 21 | `gmail.decode_mime_header` | An unknown charset name raised `LookupError`. | Falls back to UTF-8. |
| 22 | `calendar.get_events` | "No events in this range" came back as `status: error`, so agents reported a failure instead of "you're free". Errors were logged as `[delete_event]`. | Empty range returns success. Logging fixed. |
| 23 | `calendar` | `2026-03-01T09:30` (no seconds) was sent without an offset. `modify_event` made a blocking API call inside the event loop and re-sent existing Meet data. | Adds `:00+05:30`, uses `to_thread`, and relies on `patch` to keep existing fields. |
| 24 | `tasks.list_tasks` | Asked for more than 100 tasks per page (the API maximum). | Page size capped at 100. Larger requests are fetched page by page. |
| 25 | `gdocs.inspect_doc_structure`, `insert_doc_elements`, `create_table_with_data` | The invalid-parameter path crashed (unbound `request`, missing required fields). Found with an AST check of every response model call. | Proper error responses. |
| 26 | `gdocs.modify_doc_text` | Inserting text and formatting it in one call was rejected for missing `end_index`. | `end_index` is only required when formatting existing text. |
| 27 | `TableOperationManager` | Filled "the last table" in the document. Inserting a table above an existing one wrote the data into the **wrong table**. | Works out the new table's position before inserting it. |
| 28 | `HeaderFooterManager` | Guessed the header from its ID and otherwise took the first one, so a first-page header request could overwrite the default header. | Uses the IDs in `documentStyle` (`defaultHeaderId`, `firstPageHeaderId`, …). |
| 29 | `batch_update_doc` | `find_replace` was supported by the manager but blocked by the request model. | Model accepts `find_replace`, `find_text`, `replace_text` and `match_case`. |
| 30 | `gdrive` (URL upload) | `Content-Type` with parameters ("text/html; charset=utf-8") was used as the MIME type. | Parameters stripped. |
| 31 | `gslide.get_presentation` | API errors weren't handled, unlike every other tool. | Returns an error response. |
| 32 | Office `.pptx` text extraction | Slides came out in name order (slide10 before slide2). | Sorted by slide number. |

## 5. Security decisions you made

| # | Issue | Decision → done |
|---|---|---|
| 33 | Code generated by the code agent could call `send_email` through the tool bridge, skipping the per-email confirmation. | **Excluded.** Tools in `CONFIRM_BEFORE_TOOLS` are neither offered to nor callable from generated code. |
| 34 | `create_drive_file` with `file://` could upload any local file, including OAuth tokens and `.env`. A prompt-injected email could try this. | You had no preference, so the recommended default was applied: `file://` still works, but `app_tools/cred/`, `.env`, `*_token.json` and `data/` are refused. |

## 6. Test fixes

- `tests/test_llm.py` assumed the client's default temperature is 0.7. With the pinned
  `langchain-openai==1.0.2` it is `None`, so the test failed on the original code too. The test now
  checks the library default, and the comment in `config/settings.py` was corrected.

---

## Known issues not changed (worth a decision later)

- **The code "sandbox" is not isolated.** `core/sandbox_server.py` runs generated code as
  a normal Python process with your user's permissions (files, network). Approving
  the code is the only safeguard. Real isolation (a container, or nsjail/firejail)
  would be a separate project.
- **Docs indexes vs. Unicode:** text lengths use Python characters, but the Docs API
  counts UTF-16 units. Text with emoji can shift formatting ranges by a few characters.
- **Sheets ranges:** a bare sheet name with no `!` (e.g. `"Sheet2"`) is read as a
  cell reference by the formatting tools.
- **`read_sheet_values`** shows only the first 50 rows (it says how many more there are).
- Pydantic v1-style `@validator` warnings (still work; removed in pydantic v3).

## How it was verified

- `pytest`: 95 passed. The 23 new tests failed on the original code.
- `pyflakes` on changed files: no errors.
- Smoke test: every tool module imports, the graph builds (17 nodes), and all 53
  content tools go to exactly one of the docs, data or slides agents.
- Environment note: no live Google/Azure calls were made. API behaviour was checked
  against Google's discovery documents shipped with `google-api-python-client`.
