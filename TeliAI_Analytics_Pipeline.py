"""
TeliAI Campaign Analytics API (FastAPI).

Aggregates campaign metrics from Supabase, writes visualization JSON for the React dashboard,
and exposes optional Claude-based sentiment. See README.md to run locally.
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import json
from datetime import datetime, timezone
import os
import random
import uuid
from typing import Any, Dict, List, Optional, Tuple, Iterable, Set
import re

from pydantic import BaseModel, Field

# Env load order (later files override earlier):
#   1. backend/.env.defaults — committed Supabase + default ANTHROPIC_API_KEY for zero-config runs
#   2. backend/claude.env — optional overrides (gitignored)
#   3. backend/.env — optional local overrides (gitignored)
#   4. .env — optional repo-root overrides
def _parse_simple_env_file(path: Path) -> None:
    if not path.is_file():
        return
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError:
        return
    for line in raw.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if "=" not in s:
            continue
        key, _, val = s.partition("=")
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        # Fill missing keys, or replace empty placeholders from shell/IDE (override=False skips those).
        if key and (key not in os.environ or not (os.environ.get(key) or "").strip()):
            os.environ[key] = val


def _load_env_files() -> None:
    backend_dir = Path(__file__).resolve().parent
    teliai_root = backend_dir.parent
    # (path, override): defaults first without clobbering shell; user files override.
    layered: List[Tuple[Path, bool]] = [
        (backend_dir / ".env.defaults", False),
        (backend_dir / "claude.env", True),
        (backend_dir / ".env", True),
        (teliai_root / ".env", True),
    ]
    try:
        from dotenv import load_dotenv

        for path, override in layered:
            if path.is_file():
                load_dotenv(path, override=override)
    except ImportError:
        for path, _override in layered:
            _parse_simple_env_file(path)


_load_env_files()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def _repo_paths() -> Tuple[Path, Path, Path]:
    backend_dir = Path(__file__).resolve().parent
    teliai_root = backend_dir.parent
    repo_root = teliai_root.parent
    return backend_dir, teliai_root, repo_root


def _coerce_datetime(val: Any) -> Optional[datetime]:
    """
    Parse timestamps from CSV, Supabase/Postgres, or numeric epochs.
    Naive datetimes are treated as UTC for heatmap bucketing.
    """
    if val is None or val == "":
        return None
    if isinstance(val, datetime):
        return val
    if isinstance(val, (int, float)):
        ts = float(val)
        if ts > 1e12:
            ts /= 1000.0
        try:
            return datetime.fromtimestamp(ts, tz=timezone.utc)
        except (ValueError, OSError):
            return None
    s = str(val).strip()
    if not s:
        return None
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    if "T" not in s and " " in s:
        s = s.replace(" ", "T", 1)
    if re.search(r"[+-]\d{2}$", s) and not re.search(r"[+-]\d{2}:\d{2}$", s):
        s = f"{s}:00"
    try:
        return datetime.fromisoformat(s)
    except ValueError:
        return None


def _parse_date_query(val: Optional[str]) -> Optional[datetime]:
    """
    Accepts ISO strings like:
    - 2026-04-15
    - 2026-04-15T13:45:00Z
    - 2026-04-15T13:45:00+00:00
    Returns a datetime (tz-aware if input is tz-aware).
    """
    if val is None:
        return None
    s = str(val).strip()
    if not s:
        return None
    if len(s) == 10 and re.match(r"^\d{4}-\d{2}-\d{2}$", s):
        try:
            return datetime.fromisoformat(s + "T00:00:00")
        except ValueError:
            return None
    return _coerce_datetime(s)


def _norm_area_code(value: Any) -> str:
    digits = "".join(c for c in str(value or "") if c.isdigit())[:3]
    return digits if len(digits) == 3 else ""


def _norm_state(value: Any) -> str:
    s = str(value or "").strip().upper()
    return s if len(s) == 2 else ""


class AnalyticsFilters(BaseModel):
    campaign_ids: List[str] = Field(default_factory=list)
    area_codes: List[str] = Field(default_factory=list)
    states: List[str] = Field(default_factory=list)
    start: Optional[datetime] = None
    end: Optional[datetime] = None

    def is_empty(self) -> bool:
        return (
            not self.campaign_ids
            and not self.area_codes
            and not self.states
            and self.start is None
            and self.end is None
        )


# Primary U.S. state for common NPAs (NANPA). Used when dev_clients.state is missing.
_NPA_PRIMARY_STATE: Dict[str, str] = {
    "201": "NJ", "202": "DC", "203": "CT", "205": "AL", "206": "WA", "207": "ME",
    "208": "ID", "209": "CA", "210": "TX", "212": "NY", "213": "CA", "214": "TX",
    "215": "PA", "216": "OH", "217": "IL", "218": "MN", "219": "IN", "224": "IL",
    "225": "LA", "228": "MS", "229": "GA", "231": "MI", "234": "OH", "239": "FL",
    "240": "MD", "248": "MI", "251": "AL", "252": "NC", "253": "WA", "254": "TX",
    "256": "AL", "260": "IN", "262": "WI", "267": "PA", "269": "MI", "270": "KY",
    "276": "VA", "281": "TX", "301": "MD", "302": "DE", "303": "CO", "304": "WV",
    "305": "FL", "307": "WY", "308": "NE", "309": "IL", "310": "CA", "312": "IL",
    "313": "MI", "314": "MO", "315": "NY", "316": "KS", "317": "IN", "318": "LA",
    "319": "IA", "320": "MN", "321": "FL", "323": "CA", "325": "TX", "330": "OH",
    "331": "IL", "334": "AL", "336": "NC", "337": "LA", "339": "MA", "346": "TX",
    "347": "NY", "352": "FL", "360": "WA", "361": "TX", "385": "UT", "386": "FL",
    "401": "RI", "402": "NE", "404": "GA", "405": "OK", "406": "MT", "407": "FL",
    "408": "CA", "409": "TX", "410": "MD", "412": "PA", "413": "MA", "414": "WI",
    "415": "CA", "417": "MO", "419": "OH", "423": "TN", "424": "CA", "425": "WA",
    "430": "TX", "432": "TX", "434": "VA", "435": "UT", "440": "OH", "442": "CA",
    "443": "MD", "458": "OR", "469": "TX", "470": "GA", "475": "CT", "478": "GA",
    "479": "AR", "480": "AZ", "484": "PA", "501": "AR", "502": "KY", "503": "OR",
    "504": "LA", "505": "NM", "507": "MN", "508": "MA", "509": "WA", "510": "CA",
    "512": "TX", "513": "OH", "515": "IA", "516": "NY", "517": "MI", "518": "NY",
    "520": "AZ", "530": "CA", "531": "NE", "541": "OR", "551": "NJ", "559": "CA",
    "561": "FL", "562": "CA", "563": "IA", "564": "WA", "567": "OH", "570": "PA",
    "571": "VA", "573": "MO", "574": "IN", "575": "NM", "580": "OK", "585": "NY",
    "586": "MI", "601": "MS", "602": "AZ", "603": "NH", "605": "SD", "606": "KY",
    "607": "NY", "608": "WI", "609": "NJ", "610": "PA", "612": "MN", "614": "OH",
    "615": "TN", "616": "MI", "617": "MA", "618": "IL", "619": "CA", "620": "KS",
    "623": "AZ", "626": "CA", "628": "CA", "629": "TN", "630": "IL", "631": "NY",
    "636": "MO", "641": "IA", "646": "NY", "650": "CA", "651": "MN", "657": "CA",
    "660": "MO", "661": "CA", "662": "MS", "667": "MD", "669": "CA", "678": "GA",
    "681": "WV", "682": "TX", "701": "ND", "702": "NV", "703": "VA", "704": "NC",
    "706": "GA", "707": "CA", "708": "IL", "712": "IA", "713": "TX", "714": "CA",
    "715": "WI", "716": "NY", "717": "PA", "718": "NY", "719": "CO", "720": "CO",
    "724": "PA", "725": "NV", "727": "FL", "730": "IL", "731": "TN", "732": "NJ",
    "734": "MI", "737": "TX", "740": "OH", "743": "NC", "747": "CA", "754": "FL",
    "757": "VA", "760": "CA", "762": "GA", "763": "MN", "765": "IN", "769": "MS",
    "770": "GA", "772": "FL", "773": "IL", "774": "MA", "775": "NV", "779": "IL",
    "781": "MA", "785": "KS", "786": "FL", "801": "UT", "802": "VT", "803": "SC",
    "804": "VA", "805": "CA", "806": "TX", "808": "HI", "810": "MI", "812": "IN",
    "813": "FL", "814": "PA", "815": "IL", "816": "MO", "817": "TX", "818": "CA",
    "828": "NC", "830": "TX", "831": "CA", "832": "TX", "843": "SC", "845": "NY",
    "847": "IL", "848": "NJ", "850": "FL", "856": "NJ", "857": "MA", "858": "CA",
    "859": "KY", "860": "CT", "862": "NJ", "863": "FL", "864": "SC", "865": "TN",
    "870": "AR", "872": "IL", "878": "PA", "901": "TN", "903": "TX", "904": "FL",
    "906": "MI", "907": "AK", "908": "NJ", "909": "CA", "910": "NC", "912": "GA",
    "913": "KS", "914": "NY", "915": "TX", "916": "CA", "917": "NY", "918": "OK",
    "919": "NC", "920": "WI", "925": "CA", "928": "AZ", "929": "NY", "930": "IN",
    "931": "TN", "936": "TX", "937": "OH", "938": "AL", "940": "TX", "941": "FL",
    "947": "MI", "949": "CA", "951": "CA", "952": "MN", "954": "FL", "956": "TX",
    "959": "CT", "970": "CO", "971": "OR", "972": "TX", "973": "NJ", "975": "MO",
    "978": "MA", "979": "TX", "980": "NC", "984": "NC", "985": "LA", "989": "MI",
}


def _state_from_area_code(raw: Any) -> str:
    digits = "".join(c for c in str(raw or "") if c.isdigit())[:3]
    if len(digits) < 3:
        return ""
    return _NPA_PRIMARY_STATE.get(digits, "")


def _supabase_client():
    try:
        from supabase import create_client
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "supabase-py is required for SUPABASE mode. Install it with `pip install supabase`."
        ) from e

    url = (os.getenv("SUPABASE_URL") or os.getenv("VITE_PUBLIC_SUPABASE_URL") or "").strip()
    key = (
        os.getenv("SUPABASE_ANON_KEY")
        or os.getenv("SUPABASE_PUBLISHABLE_KEY")
        or os.getenv("VITE_PUBLIC_SUPABASE_ANON_KEY")
        or ""
    ).strip()
    if not url or not key:
        raise RuntimeError(
            "Missing Supabase credentials. Set SUPABASE_URL and SUPABASE_ANON_KEY "
            "(or SUPABASE_PUBLISHABLE_KEY, or VITE_PUBLIC_*)."
        )

    return create_client(url, key)


def _format_supabase_user_error(exc: BaseException) -> str:
    """Turn low-level client errors into something actionable in the UI."""
    msg = str(exc)
    low = msg.lower()
    if "401" in msg or "invalid api key" in low:
        return (
            f"{msg}\n\n"
            "Most often this means the Python client needs the **legacy anon JWT**, not only the "
            "publishable key: in Supabase → Project Settings → API, copy **anon public** "
            "(a long string starting with `eyJ`) into backend/.env as SUPABASE_ANON_KEY=...\n"
            "Also: upgrade the client (`pip install -U 'supabase>=2.15'`), ensure no spaces around "
            "`=` in .env, and that URL/key are not quoted or truncated."
        )
    return msg

def _supabase_iter_rows(client, table: str, select: str, page_size: int) -> Iterable[Dict[str, Any]]:
    offset = 0
    while True:
        resp = client.table(table).select(select).range(offset, offset + page_size - 1).execute()
        batch = getattr(resp, "data", None) or []
        if not batch:
            return
        for r in batch:
            yield r
        offset += page_size


# Matches Supabase UUID strings (with or without braces); other ids pass through unchanged.
_UUID_RE = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)


def _normalize_campaign_id(value: Any) -> str:
    """
    Canonical campaign id string so the `campaigns.campaign_id` row matches
    `messages.campaign_id` (UUID case, int vs str, etc.).
    """
    if value is None:
        return ""
    if isinstance(value, uuid.UUID):
        return str(value).lower()
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        try:
            iv = int(value)
            if iv == value:
                return str(iv)
        except (ValueError, OverflowError):
            pass
        return str(value).strip()
    s = str(value).strip()
    if not s:
        return ""
    if _UUID_RE.match(s):
        return str(uuid.UUID(s)).lower()
    return s


def _campaign_id_from_row(r: Dict[str, Any]) -> str:
    """Prefer `campaign_id`; use `id` when `campaign_id` is null (common in Bubble / external IDs)."""
    v = r.get("campaign_id")
    if v is not None and str(v).strip() != "":
        return _normalize_campaign_id(v)
    return _normalize_campaign_id(r.get("id"))


def _campaign_row_keys(r: Dict[str, Any]) -> set[str]:
    """All normalized ids that might appear in `messages.campaign_id` for this campaigns row."""
    keys: set[str] = set()
    for v in (r.get("campaign_id"), r.get("id")):
        k = _normalize_campaign_id(v)
        if k:
            keys.add(k)
    return keys


def _campaign_starting_message_from_row(r: Dict[str, Any]) -> str:
    """
    Text used for leaderboard grouping / aggregation keys.

    Prefer real outbound copy fields. Do **not** use `title` or `name` (often values like
    \"Refinance\") — those are labels, not the SMS template.
    `message` is last: in some schemas it is a short category, while `starting_message` holds the template.
    """
    for k in (
        "starting_message",
        "start_message",
        "initial_message",
        "sms_template",
        "outbound_message",
        "prompt",
        "body",
        "content",
        "template",
        "message",
    ):
        v = r.get(k)
        if v is not None and str(v).strip():
            return str(v).strip()
    return ""


def _dev_message_body_preview(r: Dict[str, Any]) -> str:
    """Outbound / template text on a messages table row (column names vary by schema)."""
    for k in (
        "message",
        "body",
        "content",
        "text",
        "sms_body",
        "sms_text",
        "message_body",
        "msg",
        "full_message",
        "starting_message",
    ):
        v = r.get(k)
        if v is not None and str(v).strip():
            return str(v).strip()
    return ""


def _is_likely_inbound_message(r: Dict[str, Any]) -> bool:
    """Skip client/inbound rows when inferring prompt text from dev_messages (avoid picking replies)."""
    s = str(r.get("sender") or "").strip().lower()
    if s in ("client", "user", "customer", "lead", "recipient"):
        return True
    d = str(r.get("direction") or "").strip().lower()
    if d in ("inbound", "in", "incoming"):
        return True
    if r.get("is_inbound") is True:
        return True
    if r.get("inbound") is True:
        return True
    return False


def _compute_campaign_dashboard_from_supabase(filters: Optional[AnalyticsFilters] = None) -> Dict[str, Any]:
    """
    Robust + low-memory Supabase pipeline.

    We avoid pulling the full dataset into Python by streaming `dev_messages` pages and updating
    aggregate counters. This matches the notebook’s *data contract* while staying efficient.

    Non-obvious definitions:
    - responded (thread-level): a thread is "responded" if `dev_conversation_history.sender == 'client'`
    - touches per lead (thread-level): number of `sender == 'agent'` rows in conversation history
    """
    client = _supabase_client()
    filters = filters or AnalyticsFilters()
    campaign_allow: Optional[Set[str]] = (
        { _normalize_campaign_id(c) for c in (filters.campaign_ids or []) if str(c).strip() }
        or None
    )
    area_allow: Optional[Set[str]] = (
        { _norm_area_code(a) for a in (filters.area_codes or []) if _norm_area_code(a) }
        or None
    )
    state_allow: Optional[Set[str]] = (
        { _norm_state(s) for s in (filters.states or []) if _norm_state(s) }
        or None
    )

    messages_table = os.getenv("SUPABASE_MESSAGES_TABLE", "dev_messages")
    campaigns_table = os.getenv("SUPABASE_CAMPAIGNS_TABLE", "dev_campaigns")
    convos_table = os.getenv("SUPABASE_CONVOS_TABLE", "dev_conversation_history")
    clients_table = os.getenv("SUPABASE_CLIENTS_TABLE", "dev_clients")

    page_size = int(os.getenv("SUPABASE_PAGE_SIZE", "1000"))
    # Safety cap. Default is "all time / no cap".
    # Set SUPABASE_MAX_ROWS to a positive integer to limit processing time during development.
    max_rows = int(os.getenv("SUPABASE_MAX_ROWS", "0"))

    MIN_MESSAGES = int(os.getenv("DASH_MIN_MESSAGES", "1"))
    TOP_N = int(os.getenv("DASH_TOP_N", "10"))
    # Defaults favor small demo CSVs; raise for production noise filtering (e.g. 30 / 20 / 100).
    MIN_CELL_MESSAGES = int(os.getenv("DASH_MIN_CELL_MESSAGES", "1"))
    MIN_LEADS = int(os.getenv("DASH_MIN_LEADS", "1"))
    MIN_STATE_MESSAGES = int(os.getenv("DASH_MIN_STATE_MESSAGES", "1"))
    PREVIEW_MAX = max(64, int(os.getenv("DASH_PROMPT_PREVIEW_MAX_CHARS", "4000")))
    # Ignore very short dev_messages fallbacks ("No", "OK") when starting_message is empty.
    PREVIEW_MIN_FROM_MESSAGES = max(8, int(os.getenv("DASH_PROMPT_PREVIEW_MIN_FROM_MESSAGES", "24")))

    # Campaign Overview prompt preview: only `starting_message` on `dev_campaigns` (override table/column via env).
    starting_message_col = os.getenv("SUPABASE_CAMPAIGNS_STARTING_MESSAGE_COLUMN", "starting_message")

    campaign_starting: Dict[str, str] = {}
    campaign_starting_message_column: Dict[str, str] = {}
    for r in _supabase_iter_rows(client, campaigns_table, "*", page_size):
        keys = _campaign_row_keys(r)
        if not keys:
            continue
        if campaign_allow is not None and not (keys & campaign_allow):
            continue
        sm_row = _campaign_starting_message_from_row(r)
        raw_preview = r.get(starting_message_col)
        preview_from_campaigns = str(raw_preview).strip() if raw_preview is not None else ""
        for k in keys:
            if k not in campaign_starting_message_column:
                campaign_starting_message_column[k] = preview_from_campaigns
            if k not in campaign_starting:
                campaign_starting[k] = sm_row

    # Earliest message body per campaign (fallback when campaigns table has no prompt text).
    prompt_ts: Dict[str, datetime] = {}
    prompt_from_messages: Dict[str, str] = {}
    messages_select = (os.getenv("SUPABASE_MESSAGES_SELECT") or "*").strip() or "*"

    # Thread-level lookups (sender values are normalized; any non-client row = outbound touch)
    responded_threads: set[str] = set()
    touches_by_thread: Dict[str, int] = {}
    convos_select = "thread_id_anonymized,sender"
    if campaign_allow is not None:
        convos_select += ",campaign_id"
    for r in _supabase_iter_rows(client, convos_table, convos_select, page_size):
        tid = str(r.get("thread_id_anonymized") or "")
        if not tid:
            continue
        if campaign_allow is not None:
            cid = _normalize_campaign_id(r.get("campaign_id"))
            if not cid or cid not in campaign_allow:
                continue
        sender_l = str(r.get("sender") or "").strip().lower()
        if sender_l == "client":
            responded_threads.add(tid)
        else:
            touches_by_thread[tid] = touches_by_thread.get(tid, 0) + 1

    state_by_thread: Dict[str, str] = {}
    for r in _supabase_iter_rows(client, clients_table, "thread_id_anonymized,state", page_size):
        tid = str(r.get("thread_id_anonymized") or "")
        st = str(r.get("state") or "").strip().upper()
        if tid and st and tid not in state_by_thread:
            state_by_thread[tid] = st

    # Aggregators (streamed).
    # Important: Supabase tables are message-level. To avoid double-counting the same lead/thread
    # multiple times, we aggregate on *thread_id_anonymized* (lead-level) for the notebook figures.
    leaderboard: Dict[Tuple[str, str], Dict[str, int]] = {}
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    heat_counts: Dict[Tuple[str, int], Dict[str, int]] = {}
    state_agg: Dict[str, Dict[str, int]] = {}
    # drip derived from thread-level tables only (no need to stream messages)

    # We use the earliest timestamp per thread to place that thread into the heatmap (day/hour).
    first_ts_by_thread: Dict[str, datetime] = {}
    area_by_thread: Dict[str, str] = {}
    seen = 0
    included_threads: set[str] = set()
    for msg in _supabase_iter_rows(
        client,
        messages_table,
        messages_select,
        page_size,
    ):
        if max_rows > 0 and seen >= max_rows:
            break
        seen += 1

        tid = str(msg.get("thread_id_anonymized") or "")
        cid = _normalize_campaign_id(msg.get("campaign_id"))
        if campaign_allow is not None and (not cid or cid not in campaign_allow):
            continue
        starting_message = campaign_starting.get(cid, "")
        responded = 1 if tid and tid in responded_threads else 0

        # Capture first timestamp per thread (for heatmap) and aggregate each thread only once.
        ts_raw = msg.get("timestamp")
        ts_parsed_row = _coerce_datetime(ts_raw) if ts_raw else None
        if ts_parsed_row:
            if filters.start and ts_parsed_row < filters.start:
                continue
            if filters.end and ts_parsed_row > filters.end:
                continue
        if tid and ts_parsed_row and tid not in first_ts_by_thread:
            first_ts_by_thread[tid] = ts_parsed_row

        body_preview = _dev_message_body_preview(msg)
        if (
            cid
            and body_preview
            and ts_parsed_row
            and not _is_likely_inbound_message(msg)
        ):
            prev_ts = prompt_ts.get(cid)
            if prev_ts is None or ts_parsed_row < prev_ts:
                prompt_ts[cid] = ts_parsed_row
                prompt_from_messages[cid] = body_preview[:PREVIEW_MAX]
        ac_norm = ""
        if tid:
            if tid in area_by_thread:
                ac_norm = area_by_thread[tid]
            else:
                ac_norm = _norm_area_code(msg.get("from_area_code"))
                if ac_norm:
                    area_by_thread[tid] = ac_norm
        if area_allow is not None:
            if not ac_norm or ac_norm not in area_allow:
                continue

        # If filtering by state, evaluate the thread's state now so *all* figures use the same eligibility.
        st = (state_by_thread.get(tid, "") or _state_from_area_code(area_by_thread.get(tid, ""))).strip().upper()
        if state_allow is not None and (not st or st not in state_allow):
            continue

        # Figure 1: leaderboard (thread-level)
        if tid:
            key = (cid, starting_message)
            a = leaderboard.setdefault(key, {"messages_sent": 0, "replies_received": 0})
            # Count each thread once per campaign.
            # (A thread can have many message rows; counting rows would inflate totals.)
            thread_key = (key, tid)
            # Store the marker in the dict itself to stay concise without extra structures.
            if "_threads" not in a:
                a["_threads"] = set()  # type: ignore[assignment]
            threads = a["_threads"]  # type: ignore[assignment]
            if tid not in threads:
                threads.add(tid)
                a["messages_sent"] += 1
                a["replies_received"] += responded
                included_threads.add(tid)

        # Figure 4: state choropleth (thread-level); fall back to NPA → state from from_area_code
        if tid and st:
            s = state_agg.setdefault(st, {"messages_sent": 0, "replies_received": 0, "_threads": set()})
            threads = s["_threads"]  # type: ignore[index]
            if tid not in threads:
                threads.add(tid)
                s["messages_sent"] += 1
                s["replies_received"] += responded

    # Figure 2: heatmap (thread-level, using earliest timestamp per thread; UTC if tz-aware)
    if filters.area_codes or filters.states:
        first_ts_by_thread = {tid: ts for tid, ts in first_ts_by_thread.items() if tid in included_threads}

    for tid, ts in first_ts_by_thread.items():
        if ts.tzinfo is not None:
            ts = ts.astimezone(timezone.utc).replace(tzinfo=None)
        day = ts.strftime("%A")
        hour = ts.hour
        if day not in day_order or hour not in range(24):
            continue
        responded = 1 if tid in responded_threads else 0
        k = (day, hour)
        c = heat_counts.setdefault(k, {"sent": 0, "replied": 0})
        c["sent"] += 1
        c["replied"] += responded


    # Build Figure 1 payload
    leaderboard_rows = []
    for (campaign_id, _starting_message), a in leaderboard.items():
        # Drop internal tracking set, if present.
        if "_threads" in a:
            a.pop("_threads", None)
        ms = int(a["messages_sent"])
        if ms < MIN_MESSAGES:
            continue
        rr = (int(a["replies_received"]) / ms) if ms else 0.0
        # Prompt preview: only dev_campaigns.starting_message (campaign_starting_message_column), then
        # dev_messages (long enough). Do not use campaign_starting / tuple starting_message — those
        # merge other columns (e.g. short `message` = "No") and are wrong for this UI.
        preview_raw = (campaign_starting_message_column.get(campaign_id, "") or "").strip()
        if not preview_raw:
            pm = (prompt_from_messages.get(campaign_id, "") or "").strip()
            if len(pm) >= PREVIEW_MIN_FROM_MESSAGES:
                preview_raw = pm
        preview = (preview_raw or "").replace("\n", " ")[:PREVIEW_MAX].strip()
        if not preview:
            preview = (
                f"No prompt text found for campaign {campaign_id}. "
                f"Set {starting_message_col!r} on {campaigns_table!r}, or ensure dev_messages has body text "
                f"(optional: SUPABASE_MESSAGES_SELECT={messages_select!r})."
            )
        leaderboard_rows.append(
            {
                "campaign_id": campaign_id,
                "messages_sent": ms,
                "replies_received": int(a["replies_received"]),
                "response_rate": float(rr),
                "prompt_preview": preview,
            }
        )
    # Best response rate first; tie-break by more sends (stronger signal on large campaigns).
    leaderboard_rows.sort(key=lambda x: (x["response_rate"], x["messages_sent"]), reverse=True)
    top = leaderboard_rows[:TOP_N]

    # Build Figure 2 payload
    hours = list(range(24))
    z: List[List[Optional[float]]] = []
    sent_matrix: List[List[int]] = []
    for day in day_order:
        row_rates: List[Optional[float]] = []
        row_sent: List[int] = []
        for hour in hours:
            c = heat_counts.get((day, hour))
            if not c or c["sent"] < MIN_CELL_MESSAGES:
                row_rates.append(None)
                row_sent.append(0)
            else:
                row_rates.append(c["replied"] / c["sent"])
                row_sent.append(int(c["sent"]))
        z.append(row_rates)
        sent_matrix.append(row_sent)

    # Build Figure 3 payload (drip funnel): every thread with a message gets ≥1 touch
    drip_agg: Dict[int, Dict[str, int]] = {}
    for tid in first_ts_by_thread.keys():
        raw_touches = int(touches_by_thread.get(tid, 0))
        t = max(raw_touches, 1)
        d = drip_agg.setdefault(t, {"leads": 0, "responded": 0})
        d["leads"] += 1
        d["responded"] += 1 if tid in responded_threads else 0
    drip_rows = []
    for touches, d in drip_agg.items():
        leads = int(d["leads"])
        if leads < MIN_LEADS:
            continue
        responded = int(d["responded"])
        drip_rows.append(
            {
                "touches": touches,
                "leads": leads,
                "responded": responded,
                "response_rate": responded / leads if leads else 0.0,
            }
        )
    drip_rows.sort(key=lambda x: x["touches"])

    # Build Figure 4 payload
    state_rows = []
    for st, a in state_agg.items():
        if "_threads" in a:
            a.pop("_threads", None)
        ms = int(a["messages_sent"])
        if ms < MIN_STATE_MESSAGES:
            continue
        rr = (int(a["replies_received"]) / ms) if ms else 0.0
        state_rows.append(
            {
                "state": st,
                "messages_sent": ms,
                "replies_received": int(a["replies_received"]),
                "response_rate": float(rr),
            }
        )
    state_rows.sort(key=lambda x: x["state"])


    return {
        "leaderboard": {
            "campaign_id": [r["campaign_id"] for r in top],
            "response_rate": [float(r["response_rate"]) for r in top],
            "messages_sent": [int(r["messages_sent"]) for r in top],
            "replies_received": [int(r["replies_received"]) for r in top],
            "prompt_preview": [r["prompt_preview"] for r in top],
        },
        "heatmap": {
            "hours": hours,
            "days": day_order,
            "z": z,
            "messages_sent": sent_matrix,
        },
        "drip": {
            "touches": [int(r["touches"]) for r in drip_rows],
            "response_rate": [float(r["response_rate"]) for r in drip_rows],
            "leads": [int(r["leads"]) for r in drip_rows],
            "responded": [int(r["responded"]) for r in drip_rows],
        },
        "states": {
            "state_code": [r["state"] for r in state_rows],
            "response_rate": [r["response_rate"] for r in state_rows],
            "messages_sent": [r["messages_sent"] for r in state_rows],
            "replies_received": [r["replies_received"] for r in state_rows],
        },
        "meta": {
            "min_messages": MIN_MESSAGES,
            "min_cell_messages": MIN_CELL_MESSAGES,
            "min_leads": MIN_LEADS,
            "min_state_messages": MIN_STATE_MESSAGES,
            "supabase_max_rows": max_rows,
            "filters": {
                "campaign_ids": list(campaign_allow or []),
                "area_codes": list(area_allow or []),
                "states": list(state_allow or []),
                "start": filters.start.isoformat() if filters.start else None,
                "end": filters.end.isoformat() if filters.end else None,
            },
        },
    }


CLAUDE_OPT_OUT_KEYWORDS = [
    "stop",
    "unsubscribe",
    "remove me",
    "opt out",
    "don't contact",
    "do not contact",
    "leave me alone",
]


def _classify_thread_sentiment_claude(conversation_text: str, client_ai: Any) -> Tuple[str, str]:
    """Port of `classify_thread_sentiment` from the Streamlit app (Claude Haiku JSON output)."""
    client_lines = [
        line.replace("[CLIENT]", "").strip().lower()
        for line in conversation_text.split("\n")
        if line.strip().upper().startswith("[CLIENT]")
    ]
    if any(kw in msg for msg in client_lines for kw in CLAUDE_OPT_OUT_KEYWORDS):
        return "NEGATIVE", "Client opted out"

    prompt = f"""Classify the OVERALL sentiment of the CLIENT in this SMS thread.
- POSITIVE: interested or engaged
- NEGATIVE: annoyed, dismissive, or opted out
- NEUTRAL: factual replies, no clear emotion

Respond ONLY with JSON: {{"sentiment": "POSITIVE", "reason": "one sentence"}}

Thread:
{conversation_text[:2000]}"""

    try:
        response = client_ai.messages.create(
            model=os.getenv("CLAUDE_SENTIMENT_MODEL", "claude-haiku-4-5-20251001"),
            max_tokens=150,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.content[0].text.strip()
        if raw.startswith("```"):
            parts = raw.split("```")
            raw = parts[1] if len(parts) > 1 else raw
            raw = raw.strip()
            if raw.lower().startswith("json"):
                raw = raw[4:].lstrip()
        result = json.loads(raw.strip())
        sentiment = str(result.get("sentiment", "NEUTRAL")).upper()
        if sentiment not in ("POSITIVE", "NEGATIVE", "NEUTRAL"):
            sentiment = "NEUTRAL"
        return sentiment, str(result.get("reason", ""))
    except Exception as e:  # pragma: no cover
        return "NEUTRAL", f"Error: {str(e)}"


def _collect_threads_claude_from_supabase() -> List[Dict[str, str]]:
    from collections import defaultdict

    client = _supabase_client()
    convos_table = os.getenv("SUPABASE_CONVOS_TABLE", "dev_conversation_history")
    page_size = int(os.getenv("SUPABASE_PAGE_SIZE", "1000"))

    # thread_id -> list of (timestamp_sort_key, sender, message)
    msgs_by_thread: Dict[str, List[Tuple[str, str, str]]] = defaultdict(list)
    campaign_by_thread: Dict[str, str] = {}
    client_threads: set[str] = set()

    for r in _supabase_iter_rows(
        client,
        convos_table,
        "thread_id_anonymized,timestamp,sender,message,campaign_id",
        page_size,
    ):
        tid = str(r.get("thread_id_anonymized") or "")
        if not tid:
            continue
        ts_raw = r.get("timestamp")
        ts_key = str(ts_raw) if ts_raw is not None else ""
        sender = str(r.get("sender") or "").strip().lower() or "unknown"
        msg = str(r.get("message") or "")
        cid = _normalize_campaign_id(r.get("campaign_id"))
        if cid and tid not in campaign_by_thread:
            campaign_by_thread[tid] = cid
        msgs_by_thread[tid].append((ts_key, sender, msg))
        if sender == "client":
            client_threads.add(tid)

    out: List[Dict[str, str]] = []
    for tid in client_threads:
        rows = msgs_by_thread.get(tid) or []
        rows_sorted = sorted(rows, key=lambda x: x[0])
        lines = []
        for _, sender, message in rows_sorted:
            label = sender.upper() if sender else "UNKNOWN"
            lines.append(f"[{label}] {message}")
        conv = "\n".join(lines)
        if conv.strip():
            out.append(
                {
                    "thread_id": tid,
                    "campaign_id": campaign_by_thread.get(tid, ""),
                    "conversation": conv,
                }
            )
    return out


def _collect_threads_for_claude_sentiment() -> List[Dict[str, str]]:
    return _collect_threads_claude_from_supabase()


class ClaudeSentimentRequest(BaseModel):
    sample_size: int = Field(100, ge=1, le=1000)
    seed: int = Field(42, ge=0)


def _keyword_sentiment(text: str) -> str:
    t = (text or "").strip().lower()
    if not t:
        return "NEGATIVE"
    negative_markers = [
        "stop",
        "opt out",
        "unsubscribe",
        "wrong number",
        "not interested",
        "no",
        "do not",
        "don't",
    ]
    if any(m in t for m in negative_markers):
        return "NEGATIVE"
    return "POSITIVE"


def _compute_campaign_sentiment_from_supabase() -> Dict[str, Any]:
    """
    Backwards-compatible sentiment summary for pages that still use it.

    We derive response text from `dev_conversation_history` where `sender == 'client'`
    (this is the closest analog to "client reply bodies" in the demo notebook).
    """
    from collections import Counter

    client = _supabase_client()
    convos_table = os.getenv("SUPABASE_CONVOS_TABLE", "dev_conversation_history")
    page_size = int(os.getenv("SUPABASE_PAGE_SIZE", "1000"))

    neg = 0
    pos = 0
    bodies: List[str] = []

    for r in _supabase_iter_rows(
        client, convos_table, "sender,message", page_size
    ):
        if str(r.get("sender") or "") != "client":
            continue
        msg = str(r.get("message") or "").strip()
        if not msg:
            continue
        bodies.append(msg.lower())
        s = _keyword_sentiment(msg)
        if s == "NEGATIVE":
            neg += 1
        else:
            pos += 1

    top = Counter(bodies).most_common(10)
    return {
        "sentiment_summary": {
            "labels": ["NEGATIVE", "POSITIVE"],
            "counts": [neg, pos],
        },
        "top_responses": {
            "responses": [t[0] for t in top],
            "counts": [int(t[1]) for t in top],
        },
        "meta": {
            "total_responses": int(neg + pos),
            "positive_count": int(pos),
            "negative_count": int(neg),
            "model": "keyword-baseline",
            "source": "supabase.dev_conversation_history(sender=client)",
        },
    }


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _latest_visualization_paths() -> Dict[str, Path]:
    backend_dir, teliai_root, _ = _repo_paths()
    backend_output = backend_dir / "output"
    frontend_public_data = teliai_root / "campaign_ui_dashboard" / "public" / "data"

    return {
        "backend_campaign_dashboard": backend_output / "campaign_dashboard.json",
        "backend_campaign_sentiment": backend_output / "campaign_sentiment.json",
        "frontend_campaign_dashboard": frontend_public_data / "campaign_dashboard.json",
        "frontend_campaign_sentiment": frontend_public_data / "campaign_sentiment.json",
    }


def _read_json_file(path: Path) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def _try_load_cached_visualizations() -> Optional[Tuple[Dict[str, Any], Dict[str, Any], str]]:
    """
    Read last refresh output from disk (backend/output preferred, then public/data).
    Returns (dashboard, sentiment, generated_at_iso) or None if cache is unusable.
    """
    paths = _latest_visualization_paths()
    dashboard = _read_json_file(paths["backend_campaign_dashboard"]) or _read_json_file(
        paths["frontend_campaign_dashboard"]
    )
    sentiment = _read_json_file(paths["backend_campaign_sentiment"]) or _read_json_file(
        paths["frontend_campaign_sentiment"]
    )
    if not dashboard or not sentiment:
        return None
    lb = (dashboard.get("leaderboard") or {}).get("campaign_id") or []
    if not isinstance(lb, list) or len(lb) == 0:
        return None
    if not sentiment.get("sentiment_summary"):
        return None

    mtimes: List[float] = []
    for k in ("backend_campaign_dashboard", "backend_campaign_sentiment"):
        p = paths[k]
        if p.is_file():
            mtimes.append(p.stat().st_mtime)
    if not mtimes:
        for k in ("frontend_campaign_dashboard", "frontend_campaign_sentiment"):
            p = paths[k]
            if p.is_file():
                mtimes.append(p.stat().st_mtime)
    gen_ts = max(mtimes) if mtimes else datetime.now(timezone.utc).timestamp()
    gen_at = datetime.fromtimestamp(gen_ts, tz=timezone.utc).isoformat()
    return dashboard, sentiment, gen_at


@app.get("/api/analytics-status")
def analytics_status():
    """Env-only snapshot so the UI can show whether Supabase analytics are configured."""
    url = (os.getenv("SUPABASE_URL") or os.getenv("VITE_PUBLIC_SUPABASE_URL") or "").strip()
    key = (
        os.getenv("SUPABASE_ANON_KEY")
        or os.getenv("SUPABASE_PUBLISHABLE_KEY")
        or os.getenv("VITE_PUBLIC_SUPABASE_ANON_KEY")
        or ""
    ).strip()
    return {
        "success": True,
        "analytics_backend": "supabase",
        "configured_data_source": "supabase",
        "supabase_credentials_configured": bool(url and key),
    }


@app.get("/api/visualizations")
def get_visualizations(
    live: bool = Query(
        False,
        description="If true, recompute from Supabase (slow). Default: serve cached JSON from disk.",
    ),
    campaign_id: Optional[List[str]] = Query(
        None,
        description="Optional campaign id filter (repeatable): ?campaign_id=...&campaign_id=...",
    ),
    area_code: Optional[List[str]] = Query(
        None,
        description="Optional area code filter (repeatable): ?area_code=248&area_code=313",
    ),
    state: Optional[List[str]] = Query(
        None,
        description="Optional 2-letter state filter (repeatable): ?state=MI&state=CA",
    ),
    start: Optional[str] = Query(
        None,
        description="Optional start timestamp/date (ISO). Example: 2026-04-15 or 2026-04-15T00:00:00Z",
    ),
    end: Optional[str] = Query(
        None,
        description="Optional end timestamp/date (ISO). Example: 2026-04-30 or 2026-04-30T23:59:59Z",
    ),
):
    """
    Default: return **cached JSON** written by `GET /api/refresh-statistics` (fast, no DB scan).

    Use **Refresh Statistics** in the UI to rebuild cache from Supabase after new data.

    `?live=1` forces a live Supabase aggregation (for debugging).
    """
    filters = AnalyticsFilters(
        campaign_ids=[_normalize_campaign_id(x) for x in (campaign_id or []) if str(x).strip()],
        area_codes=[_norm_area_code(x) for x in (area_code or []) if _norm_area_code(x)],
        states=[_norm_state(x) for x in (state or []) if _norm_state(x)],
        start=_parse_date_query(start),
        end=_parse_date_query(end),
    )
    require_live = not filters.is_empty()

    if not live and not require_live:
        cached = _try_load_cached_visualizations()
        if cached:
            dashboard, sentiment, gen_at = cached
            return {
                "success": True,
                "generated_at": gen_at,
                "campaign_dashboard": dashboard,
                "campaign_sentiment": sentiment,
                "analytics_source": "cached_json",
                "fallback_note": None,
            }
        raise HTTPException(
            status_code=503,
            detail=(
                "No analytics cache on disk yet. Call GET /api/refresh-statistics once "
                "(or click **Refresh Statistics** in Message Analytics) to build JSON from Supabase, "
                "then reload this page."
            ),
        )

    try:
        dashboard = _compute_campaign_dashboard_from_supabase(filters=filters if require_live else None)
        sentiment = _compute_campaign_sentiment_from_supabase()
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=_format_supabase_user_error(e),
        ) from e

    return {
        "success": True,
        "generated_at": datetime.utcnow().isoformat(),
        "campaign_dashboard": dashboard,
        "campaign_sentiment": sentiment,
        "analytics_source": "supabase_live_filtered" if require_live else "supabase_live",
        "fallback_note": (
            "Filters require live Supabase aggregation (cache is unfiltered)." if (require_live and not live) else None
        ),
    }


@app.get("/api/refresh-statistics")
def refresh_statistics():
    try:
        campaign_dashboard = _compute_campaign_dashboard_from_supabase()
        campaign_sentiment = _compute_campaign_sentiment_from_supabase()
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=_format_supabase_user_error(e),
        ) from e

    source_label = "supabase (streaming aggregates)"
    generated_at = datetime.utcnow().isoformat()
    paths = _latest_visualization_paths()
    for k, p in paths.items():
        if k.endswith("campaign_dashboard"):
            _write_json(p, campaign_dashboard)
        elif k.endswith("campaign_sentiment"):
            _write_json(p, campaign_sentiment)

    return {
        "success": True,
        "generated_at": generated_at,
        "message": f"Refreshed visualization JSONs from {source_label}.",
        "files_written": {k: str(v) for k, v in paths.items()},
    }


@app.post("/api/insights/claude-sentiment")
def run_claude_thread_sentiment(body: ClaudeSentimentRequest):
    """
    Thread-level sentiment using Anthropic Claude (Haiku), matching the Streamlit app logic.
    Requires ANTHROPIC_API_KEY in the environment (or backend/.env).
    """
    api_key = (os.getenv("ANTHROPIC_API_KEY") or "").strip()
    if not api_key:
        raise HTTPException(
            status_code=400,
            detail="Missing ANTHROPIC_API_KEY. It should be in backend/.env.defaults; optional override: backend/claude.env",
        )

    cap = int(os.getenv("CLAUDE_SENTIMENT_MAX_SAMPLE", "500"))
    sample_size = min(body.sample_size, cap)

    try:
        import anthropic
    except ImportError as e:
        raise HTTPException(
            status_code=501,
            detail="Install the Anthropic SDK: pip install anthropic",
        ) from e

    threads = _collect_threads_for_claude_sentiment()
    if not threads:
        raise HTTPException(
            status_code=400,
            detail="No eligible threads found (need client replies in Supabase conversation history).",
        )

    rng = random.Random(body.seed)
    sample_n = min(sample_size, len(threads))
    sample = rng.sample(threads, sample_n)

    client_ai = anthropic.Anthropic(api_key=api_key)
    results: List[Dict[str, Any]] = []
    breakdown: Dict[str, int] = {"POSITIVE": 0, "NEGATIVE": 0, "NEUTRAL": 0}
    by_campaign: Dict[str, Dict[str, int]] = {}

    for row in sample:
        sentiment, reason = _classify_thread_sentiment_claude(row["conversation"], client_ai)
        if sentiment not in breakdown:
            sentiment = "NEUTRAL"
        breakdown[sentiment] += 1

        cid = _normalize_campaign_id(row.get("campaign_id"))
        cm = by_campaign.setdefault(cid, {"POSITIVE": 0, "NEGATIVE": 0, "NEUTRAL": 0})
        cm[sentiment] += 1

        results.append(
            {
                "thread_id": row["thread_id"],
                "campaign_id": cid,
                "sentiment": sentiment,
                "reason": (reason or "")[:500],
            }
        )

    by_campaign_rows = [
        {
            "campaign_id": cid,
            "POSITIVE": counts["POSITIVE"],
            "NEGATIVE": counts["NEGATIVE"],
            "NEUTRAL": counts["NEUTRAL"],
        }
        for cid, counts in sorted(by_campaign.items(), key=lambda x: x[0])
    ]

    return {
        "success": True,
        "generated_at": datetime.utcnow().isoformat(),
        "model": os.getenv("CLAUDE_SENTIMENT_MODEL", "claude-haiku-4-5-20251001"),
        "eligible_threads": len(threads),
        "analyzed": len(sample),
        "sample_size_requested": body.sample_size,
        "sample_size_effective": sample_n,
        "breakdown": breakdown,
        "by_campaign": by_campaign_rows,
        "threads": results,
    }