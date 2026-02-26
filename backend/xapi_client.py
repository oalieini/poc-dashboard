"""
xAPI LRS REST API Client
─────────────────────────────────────────────────────────────────────────────
Supports two API modes, auto-detected:

  1. TRAX LRS (Maskott) — custom envelope format:
       GET /xapi/ext/statements?skip=N&limit=N
       Response: { "data": [...], "paging": { "limit", "skip", "count" } }
       Statements inside data[i]["data"] (nested)

  2. Standard xAPI LRS — ADL spec format:
       GET /statements?limit=N
       Response: { "statements": [...], "more": "/statements?..." }

Auth: Basic Auth or Bearer token

"""

from __future__ import annotations

import json
import os
import re
import logging
import time
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, List, Callable, Any
from urllib.parse import urlparse

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)

DEFAULT_PAGE_SIZE = 500
DEFAULT_TIMEOUT = 120
MAX_STATEMENTS = 50_000
RATE_LIMIT_SLEEP = 0.1


def _iso(dt: Optional[datetime]) -> Optional[str]:
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _unwrap_trax(raw: Dict) -> Dict:
    """
    TRAX wraps the xAPI statement inside raw["data"].
    Returns the inner statement dict, or raw if already a plain statement.
    """
    inner = raw.get("data")
    if isinstance(inner, dict) and "verb" in inner:
        return inner
    return raw


class XAPIClient:
    """
    Unified xAPI client for TRAX (Maskott) and standard LRS endpoints.

    """

    def __init__(
        self,
        endpoint: str,
        username: str = "",
        password: str = "",
        token: str = "",
        mode: str = "auto",
        timeout: int = DEFAULT_TIMEOUT,
        page_size: int = DEFAULT_PAGE_SIZE,
        max_retries: int = 3,
        rate_limit_sleep: float = RATE_LIMIT_SLEEP,
    ):
        self.endpoint = endpoint.rstrip("/")
        self.username = username or os.getenv("LRS_USERNAME", "") or os.getenv("XAPI_USERNAME", "")
        self.password = password or os.getenv("LRS_PASSWORD", "") or os.getenv("XAPI_PASSWORD", "")
        self.token = token or os.getenv("LRS_TOKEN", "") or os.getenv("XAPI_TOKEN", "")
        self.timeout = timeout
        self.page_size = page_size
        self.rate_limit_sleep = rate_limit_sleep

        self.session = requests.Session()
        retry = Retry(
            total=max_retries,
            backoff_factor=1.0,
            status_forcelist=[429, 500, 502, 503, 504],
            respect_retry_after_header=True,
        )
        adapter = HTTPAdapter(max_retries=retry)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)
        self.session.headers.update({
            "X-Experience-API-Version": "1.0.3",
            "Accept": "application/json",
        })
        if self.token:
            self.session.headers["Authorization"] = f"Bearer {self.token}"
        elif self.username:
            self.session.auth = (self.username, self.password)

        # Detect mode
        if mode == "auto":
            if "ext/statements" in self.endpoint or "trax" in self.endpoint.lower():
                self._mode = "trax"
            else:
                self._mode = "standard"
        else:
            self._mode = mode

    @property
    def detected_mode(self) -> str:
        return self._mode

    def _trax_url(self) -> str:
        """Build the TRAX statements URL."""
        if self.endpoint.endswith("statements"):
            return self.endpoint
        if "ext/statements" in self.endpoint:
            return self.endpoint
        return self.endpoint + "/ext/statements"

    def _standard_url(self) -> str:
        if self.endpoint.endswith("statements"):
            return self.endpoint
        return self.endpoint + "/statements"


    def _fetch_trax(
            self,
            since=None, until=None, verb=None,
            actor_id=None, activity_id=None,
            max_statements=MAX_STATEMENTS,
            progress_cb=None,
    ) -> List[Dict]:
        """
        Fetch statements from TRAX LRS using proper filter syntax and ID-based pagination.
        """
        url = self._trax_url()

        logger.info(f"Fetching from TRAX: {url}")
        logger.info(f"Input params: since={since}, until={until}, verb={verb}, max_statements={max_statements}")

        filters = {}

        filter_str = None

        if since and until:
            lte_part = f'"data->timestamp":{{"$lte":"{until.strftime("%Y-%m-%d")}"}}'
            gte_part = f'"data->timestamp":{{"$gte":"{since.strftime("%Y-%m-%d")}"}}'
            filter_str = "{" + lte_part + "," + gte_part + "}"
            logger.info(f"Date filter (duplicate keys): {filter_str}")
        elif until:
            filters["data->timestamp"] = {"$lte": until.strftime("%Y-%m-%d")}
            logger.info(f"Date filter: $lte only")
        elif since:
            filters["data->timestamp"] = {"$gte": since.strftime("%Y-%m-%d")}
            logger.info(f"Date filter: $gte only")

        if verb:
            filters["data->verb->id"] = {"$in": [verb]}

        if actor_id:
            filters["data->actor->account->name"] = actor_id

        if activity_id:
            filters["data->object->id"] = activity_id

        all_statements: List[Dict] = []
        last_id = 0
        total_count: Optional[int] = None
        page_num = 0

        while True:
            page_num += 1

            page_filters = filters.copy() if filters else {}
            if last_id > 0:
                page_filters["id"] = {"$gt": last_id}

            params = {
                "limit": self.page_size
            }


            if filter_str and last_id == 0:
                params["filters"] = filter_str
                logger.info(f"Page {page_num}: Filters (raw) = {filter_str}")
            elif filter_str and last_id > 0:

                import json
                filter_with_id = filter_str[:-1] + f',"id":{{"$gt":{last_id}}}' + "}"
                params["filters"] = filter_with_id
                logger.info(f"Page {page_num}: Filters (raw+ID) = {filter_with_id}")
            elif page_filters:
                params["filters"] = json.dumps(page_filters)
                logger.info(f"Page {page_num}: Filters (JSON) = {params['filters']}")
            else:
                logger.info(f"Page {page_num}: No filters (fetching all)")

            max_retries = 3
            for attempt in range(max_retries):
                try:
                    resp = self.session.get(url, params=params, timeout=self.timeout)
                    logger.info(f"Request URL: {resp.url}")
                    logger.info(f"Response status: {resp.status_code}")
                    resp.raise_for_status()
                    break
                except requests.exceptions.HTTPError as e:
                    if e.response.status_code == 400:
                        logger.error(f"[ERROR] TRAX 400 error")
                        logger.error(f"[ERROR] URL: {resp.url}")
                        logger.error(f"[ERROR] Params: {params}")
                        if all_statements:
                            logger.error(f"[ERROR] Returning {len(all_statements)} statements fetched so far")
                            return all_statements
                        else:
                            logger.error(f"[ERROR] No statements fetched, returning empty list")
                            return []
                    else:
                        raise
                except requests.exceptions.ReadTimeout:
                    if attempt < max_retries - 1:
                        wait_time = 2 ** attempt
                        logger.warning(f"Timeout on attempt {attempt + 1}/{max_retries}, retrying in {wait_time}s...")
                        time.sleep(wait_time)
                    else:
                        logger.error(f"Failed after {max_retries} attempts")
                        raise
                except requests.exceptions.RequestException as e:
                    logger.error(f"Request failed: {e}")
                    raise

            body = resp.json()
            logger.debug(f"Response keys: {list(body.keys())}")


            raw_items = body.get("data", [])

            if page_num == 1:
                paging = body.get("paging", {})
                total_count = paging.get("count")
                if total_count:
                    logger.info(f"TRAX paging count: {total_count:,} (from skip-based query)")
                else:
                    logger.info(f"No paging info (ID-based pagination) - will fetch until data array is empty")

            if not raw_items:
                if page_num == 1:
                    logger.warning("[WARN] No statements in response")
                logger.info(f"No more statements (page {page_num} was empty)")
                break

            page_stmts = [_unwrap_trax(r) for r in raw_items]

            if raw_items:
                last_id = raw_items[-1].get("id", 0)

            all_statements.extend(page_stmts)
            fetched = len(all_statements)

            if progress_cb:
                if total_count:
                    progress_cb(fetched, total_count)
                else:
                    progress_cb(fetched, fetched)  # Progress bar will show as 100% but with count

            logger.debug(f"Page {page_num}: got {len(page_stmts)}, total={fetched}, last_id={last_id}")

            if len(page_stmts) < self.page_size:
                logger.info(f"Fetched all available (last page: {len(page_stmts)} statements)")
                break
            if max_statements and fetched >= max_statements:
                logger.info(f"Cap reached: {max_statements}")
                break
            if total_count and fetched >= total_count:
                logger.info(f"Fetched all {total_count} statements")
                break

            time.sleep(self.rate_limit_sleep)

        logger.info(f"Total fetched: {len(all_statements):,} statements")
        return all_statements
    def _fetch_standard(
        self,
        since=None, until=None, verb=None,
        actor_mbox=None, activity_id=None,
        max_statements=MAX_STATEMENTS,
        progress_cb=None,
    ) -> List[Dict]:
        params: Dict[str, Any] = {"limit": self.page_size, "ascending": "true"}
        if since:
            params["since"] = _iso(since)
        if until:
            params["until"] = _iso(until)
        if verb:
            params["verb"] = verb
        if actor_mbox:
            params["agent"] = f'{{"mbox":"{actor_mbox}"}}'
        if activity_id:
            params["activity"] = activity_id

        all_statements: List[Dict] = []
        url = self._standard_url()
        current_params: Optional[Dict] = params

        while url:
            resp = self.session.get(url, params=current_params, timeout=self.timeout)
            resp.raise_for_status()
            body = resp.json()

            page = body.get("statements", [])
            all_statements.extend(page)
            fetched = len(all_statements)

            if progress_cb:
                progress_cb(fetched, fetched)

            more = body.get("more")
            if more:
                parsed = urlparse(self.endpoint)
                base = f"{parsed.scheme}://{parsed.netloc}"
                url = base + more if more.startswith("/") else more
                current_params = None
            else:
                url = None

            if not page:
                break
            if max_statements and fetched >= max_statements:
                break

            time.sleep(self.rate_limit_sleep)

        return all_statements

    def get_statements(
        self,
        since=None, until=None, verb=None, actor=None,
        activity_id=None, max_statements=MAX_STATEMENTS,
        progress_cb=None,
    ) -> List[Dict]:
        if self._mode == "trax":
            return self._fetch_trax(
                since=since, until=until, verb=verb,
                actor_id=actor, activity_id=activity_id,
                max_statements=max_statements, progress_cb=progress_cb,
            )
        return self._fetch_standard(
            since=since, until=until, verb=verb,
            actor_mbox=actor, activity_id=activity_id,
            max_statements=max_statements, progress_cb=progress_cb,
        )

    def get_total_count(self, since=None, until=None) -> Optional[int]:
        """
        Fast count via TRAX paging info.
        """
        if self._mode == "trax":
            params: Dict[str, Any] = {
                "limit": 1,
                "skip": 0
            }
            filters = {}
            filter_str = None

            if since and until:
                # Use duplicate key format for date range
                lte_part = f'"data->timestamp":{{"$lte":"{until.strftime("%Y-%m-%d")}"}}'
                gte_part = f'"data->timestamp":{{"$gte":"{since.strftime("%Y-%m-%d")}"}}'
                filter_str = "{" + lte_part + "," + gte_part + "}"
            elif until:
                filters["data->timestamp"] = {"$lte": until.strftime("%Y-%m-%d")}
            elif since:
                filters["data->timestamp"] = {"$gte": since.strftime("%Y-%m-%d")}

            if filter_str:
                params["filters"] = filter_str
                logger.debug(f"Count query filters (raw): {filter_str}")
            elif filters:
                params["filters"] = json.dumps(filters)
                logger.debug(f"Count query filters: {params['filters']}")

            try:
                resp = self.session.get(self._trax_url(), params=params, timeout=self.timeout)
                resp.raise_for_status()
                body = resp.json()
                paging = body.get("paging", {})
                count = paging.get("count", 0)

                if count > 0:
                    logger.info(f"TRAX count query: {count:,} statements")
                else:
                    data_len = len(body.get("data", []))
                    if data_len > 0:
                        logger.info(f"TRAX count query: paging.count={count}, but got {data_len} statements (count may be unreliable)")
                        return None  # Unknown total
                    else:
                        logger.info(f"TRAX count query: {count} statements")

                return count
            except Exception as e:
                logger.warning(f"Count query failed: {e}")
        return None

    def get_all_statements_batched(
        self, since=None, until=None,
        batch_size=DEFAULT_PAGE_SIZE,
        max_statements=MAX_STATEMENTS,
        progress_cb=None,
    ) -> List[Dict]:
        old = self.page_size
        self.page_size = batch_size
        try:
            return self.get_statements(
                since=since, until=until,
                max_statements=max_statements,
                progress_cb=progress_cb,
            )
        finally:
            self.page_size = old

    def get_statements_chunked(
        self,
        since: Optional[datetime],
        until: Optional[datetime],
        chunk_days: int = 7,
        max_per_chunk: int = MAX_STATEMENTS,
        progress_cb=None,
    ) -> List[Dict]:
        """
        Split a large date range into weekly chunks.
        """
        if since is None:
            since = datetime.now(timezone.utc) - timedelta(days=30)
        if until is None:
            until = datetime.now(timezone.utc)
        if since.tzinfo is None:
            since = since.replace(tzinfo=timezone.utc)
        if until.tzinfo is None:
            until = until.replace(tzinfo=timezone.utc)

        all_statements: List[Dict] = []
        chunk_start = since
        while chunk_start < until:
            chunk_end = min(chunk_start + timedelta(days=chunk_days), until)
            logger.info(f"Chunk {chunk_start.date()} → {chunk_end.date()}")
            chunk = self.get_statements(
                since=chunk_start, until=chunk_end,
                max_statements=max_per_chunk,
                progress_cb=progress_cb,
            )
            all_statements.extend(chunk)
            chunk_start = chunk_end + timedelta(seconds=1)
            time.sleep(self.rate_limit_sleep)

        return all_statements

    def get_completed(self, **kw): return self.get_statements(verb="http://adlnet.gov/expapi/verbs/completed", **kw)
    def get_passed(self, **kw): return self.get_statements(verb="http://adlnet.gov/expapi/verbs/passed", **kw)
    def get_failed(self, **kw): return self.get_statements(verb="http://adlnet.gov/expapi/verbs/failed", **kw)
    def get_answered(self, **kw): return self.get_statements(verb="http://adlnet.gov/expapi/verbs/answered", **kw)
    def get_logged_in(self, **kw): return self.get_statements(verb="https://w3id.org/xapi/adl/verbs/logged-in", **kw)
    def get_initialized(self, **kw): return self.get_statements(verb="http://adlnet.gov/expapi/verbs/initialized", **kw)
    def get_suspended(self, **kw): return self.get_statements(verb="http://adlnet.gov/expapi/verbs/suspended", **kw)
    def get_terminated(self, **kw): return self.get_statements(verb="http://adlnet.gov/expapi/verbs/terminated", **kw)
    def get_created(self, **kw): return self.get_statements(verb="http://activitystrea.ms/schema/1.0/create", **kw)
    def get_launched(self, **kw): return self.get_statements(verb="http://adlnet.gov/expapi/verbs/launched", **kw)
    def get_satisfied(self, **kw): return self.get_statements(verb="http://adlnet.gov/expapi/verbs/satisfied", **kw)
    def get_resumed(self, **kw): return self.get_statements(verb="http://adlnet.gov/expapi/verbs/resumed", **kw)
    def get_experienced(self, **kw): return self.get_statements(verb="http://adlnet.gov/expapi/verbs/experienced", **kw)
    def get_evaluated(self, **kw): return self.get_statements(verb="http://www.tincanapi.co.uk/verbs/evaluated", **kw)


    def ping(self) -> bool:
        try:
            if self._mode == "trax":
                resp = self.session.get(self._trax_url(), params={"limit": 1, "skip": 0}, timeout=10)
            else:
                resp = self.session.get(self._standard_url(), params={"limit": 1}, timeout=10)
            return resp.status_code in (200, 400)
        except Exception:
            return False

    def get_lrs_info(self) -> Dict:
        return {
            "endpoint": self.endpoint,
            "mode": self._mode,
            "connected": self.ping(),
            "total_statements": self.get_total_count(),
            "auth": "bearer" if self.token else ("basic" if self.username else "none"),
        }


class StatementParser:

    @staticmethod
    def actor_id(stmt: Dict) -> str:
        actor = stmt.get("actor", {})
        mbox = actor.get("mbox", "")
        if mbox:
            return mbox
        account = stmt.get("account", {})
        return account.get("name", "") or actor.get("name", "unknown")

    @staticmethod
    def actor_name(stmt: Dict) -> str:
        actor = stmt.get("actor", {})
        return actor.get("name", "") or StatementParser.actor_id(stmt)

    @staticmethod
    def verb_id(stmt: Dict) -> str:
        return stmt.get("verb", {}).get("id", "")

    @staticmethod
    def verb_display(stmt: Dict) -> str:
        display = stmt.get("verb", {}).get("display", {})
        return (
            display.get("en-US")
            or display.get("en")
            or display.get("fr-FR")
            or StatementParser.verb_id(stmt).split("/")[-1]
        )

    @staticmethod
    def activity_id(stmt: Dict) -> str:
        return stmt.get("object", {}).get("id", "")

    @staticmethod
    def activity_name(stmt: Dict) -> str:
        definition = stmt.get("object", {}).get("definition", {})
        names = definition.get("name", {})
        return (
            names.get("fr-FR")
            or names.get("en-US")
            or names.get("en")
            or StatementParser.activity_id(stmt).split("/")[-1]
        )

    @staticmethod
    def timestamp(stmt: Dict) -> Optional[datetime]:
        ts_str = stmt.get("timestamp") or stmt.get("stored")
        if not ts_str:
            return None
        try:
            ts_str = ts_str.replace("Z", "+00:00")
            ts_str = re.sub(r"(\+\d{2}:\d{2}):\d{2}$", r"\1", ts_str)
            return datetime.fromisoformat(ts_str)
        except Exception:
            return None

    @staticmethod
    def score_scaled(stmt: Dict) -> Optional[float]:
        result = stmt.get("result", {})
        score = result.get("score", {})
        scaled = score.get("scaled")
        if scaled is not None:
            return float(scaled) * 100
        raw = score.get("raw")
        max_s = score.get("max")
        min_s = score.get("min", 0)
        if raw is not None and max_s is not None:
            max_f, min_f = float(max_s), float(min_s)
            if max_f > min_f:
                return (float(raw) - min_f) / (max_f - min_f) * 100
        return None

    @staticmethod
    def duration_minutes(stmt: Dict) -> Optional[float]:
        duration_str = stmt.get("result", {}).get("duration", "")
        if not duration_str:
            return None
        try:
            m = re.match(
                r"P(?:(\d+)Y)?(?:(\d+)M)?(?:(\d+)D)?T?(?:(\d+)H)?(?:(\d+)M)?(?:([\d.]+)S)?",
                duration_str,
            )
            if not m:
                return None
            yr, mo, d, h, mn, s = (float(x) if x else 0 for x in m.groups())
            total = yr * 525960 + mo * 43800 + d * 1440 + h * 60 + mn + s / 60
            return total if total > 0 else None
        except Exception:
            return None

    @staticmethod
    def progress(stmt: Dict) -> Optional[float]:
        for key, val in stmt.get("result", {}).get("extensions", {}).items():
            if "progress" in key.lower():
                try:
                    return float(val)
                except Exception:
                    pass
        return None

    @staticmethod
    def success(stmt: Dict) -> Optional[bool]:
        return stmt.get("result", {}).get("success")

    @staticmethod
    def platform(stmt: Dict) -> Optional[str]:
        return stmt.get("context", {}).get("platform")

    @staticmethod
    def session_id(stmt: Dict) -> Optional[str]:
        for key, val in stmt.get("context", {}).get("extensions", {}).items():
            if "sessionid" in key.lower():
                return str(val).split("/")[-1]
        return None

    @staticmethod
    def role(stmt: Dict) -> Optional[str]:
        """Extract Tactileo/Maskott invitee role from context extensions."""
        for key, val in stmt.get("context", {}).get("extensions", {}).items():
            if "invitee" in key.lower():
                return str(val)
        return None

    @staticmethod
    def group_session_id(stmt: Dict) -> Optional[str]:
        """
        Extract session ID from context.contextActivities.parent.

        """
        parents = stmt.get("context", {}).get("contextActivities", {}).get("parent", [])
        for parent in parents:
            # Look for tutor-session or session-related URLs
            parent_type = parent.get("definition", {}).get("type", "")
            parent_id = parent.get("id", "")
            if (
                "tutor-session" in parent_type or
                "groupsessions" in parent_id or
                "coursesessions" in parent_id or
                ("course" in parent_id and "session" in parent_id.lower())
            ):
                return parent_id
        return None

    @staticmethod
    def attempt_id(stmt: Dict) -> Optional[str]:
        """
        Extract attempt ID from context.contextActivities.grouping.
        """
        groupings = stmt.get("context", {}).get("contextActivities", {}).get("grouping", [])
        for grouping in groupings:
            grouping_type = grouping.get("definition", {}).get("type", "")
            grouping_id = grouping.get("id", "")
            if "attempt" in grouping_type or "attempts" in grouping_id:
                return grouping_id
        return None

    @staticmethod
    def module_id(stmt: Dict) -> Optional[str]:
        """
        Extract module ID from context.contextActivities.parent.
        """
        parents = stmt.get("context", {}).get("contextActivities", {}).get("parent", [])
        for parent in parents:
            parent_type = parent.get("definition", {}).get("type", "")
            parent_id = parent.get("id", "")
            if "module" in parent_type or "modules" in parent_id:
                return parent_id
        return None

    @staticmethod
    def registration(stmt: Dict) -> Optional[str]:
        """Extract registration UUID (launch session identifier)."""
        return stmt.get("context", {}).get("registration")


class MockXAPIClient(XAPIClient):

    def __init__(self):
        self.endpoint = "mock://local"
        self._mode = "trax"
        self.page_size = DEFAULT_PAGE_SIZE
        self.rate_limit_sleep = 0
        self.timeout = 5
        self.session = None
        self.username = ""
        self.password = ""
        self.token = ""

    def ping(self) -> bool:
        return True

    def get_lrs_info(self) -> Dict:
        return {
            "endpoint": "mock://local (Maskott TRAX demo)",
            "mode": "trax",
            "connected": True,
            "total_statements": 746415,
            "auth": "none",
        }

    def get_total_count(self, **kw) -> int:
        return 746415

    def get_statements(
        self,
        since=None, until=None, verb=None, actor=None,
        activity_id=None, max_statements=MAX_STATEMENTS,
        progress_cb=None, **kw,
    ) -> List[Dict]:
        import random

        VERBS = {
            "initialized": "http://adlnet.gov/expapi/verbs/initialized",
            "completed": "http://adlnet.gov/expapi/verbs/completed",
            "passed": "http://adlnet.gov/expapi/verbs/passed",
            "failed": "http://adlnet.gov/expapi/verbs/failed",
            "launched": "http://adlnet.gov/expapi/verbs/launched",
            "answered": "http://adlnet.gov/expapi/verbs/answered",
            "suspended": "http://adlnet.gov/expapi/verbs/suspended",
            "resumed": "http://adlnet.gov/expapi/verbs/resumed",
            "terminated": "http://adlnet.gov/expapi/verbs/terminated",
            "experienced": "http://adlnet.gov/expapi/verbs/experienced",
            "satisfied": "http://adlnet.gov/expapi/verbs/satisfied",
            "logged-in": "https://w3id.org/xapi/adl/verbs/logged-in",
            "created": "http://activitystrea.ms/schema/1.0/create",
            "evaluated": "http://www.tincanapi.co.uk/verbs/evaluated",
        }
        VERB_WEIGHTS = [
            ("initialized", 0.10), ("completed", 0.09), ("passed", 0.10),
            ("failed", 0.04), ("launched", 0.12), ("answered", 0.10),
            ("suspended", 0.02), ("resumed", 0.02), ("terminated", 0.02),
            ("experienced", 0.15), ("satisfied", 0.05), ("logged-in", 0.05),
            ("created", 0.02), ("evaluated", 0.12),
        ]
        ACTIVITIES = [
            ("75c288fa-79dd-be0c-f90e-5bfca1dcf082", "Cahier De Test Mathscope"),
            ("a1b2c3d4-1234-5678-abcd-aabbccddeeff", "Exercice Fractions"),
            ("b2c3d4e5-2345-6789-bcde-bbccddeeff00", "Quiz Géométrie"),
            ("c3d4e5f6-3456-789a-cdef-ccddeeff0011", "Vidéo Introduction Algèbre"),
            ("d4e5f6g7-4567-89ab-defa-ddeeff001122", "TP Probabilités"),
            ("e5f6g7h8-5678-9abc-efab-eeff00112233", "Devoir Trigonométrie"),
            ("f6g7h8i9-6789-abcd-fabc-ff0011223344", "Exercice Statistiques"),
            ("g7h8i9j0-789a-bcde-abcd-001122334455", "Module Calcul Intégral"),
            ("h8i9j0k1-89ab-cdef-bcde-112233445566", "Quiz Fonctions"),
            ("i9j0k1l2-9abc-defa-cdef-223344556677", "Cahier Géométrie Analytique"),
        ]
        HOMEPAGES = [
            "https://edu.tactileo.fr/logon/0050012L",
            "https://edu.tactileo.fr/logon/EduGo_recette",
            "https://edu.tactileo.fr/logon/EduGo_conception",
        ]
        PLATFORMS = ["learningconnect-SSO", "Tactileo-simple", "Tactileo-mobile"]
        ROLES = [
            "Teacher", "Teacher, LtiAuthor",
            "ResourceAuthor, Teacher, EntityAdmin",
            "LTI_Cabri_primaire", "LTI_Cabri_secondaire",
        ]

        now = until or datetime.now(timezone.utc)
        start = since or (now - timedelta(days=30))
        if isinstance(now, datetime) and now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
        if isinstance(start, datetime) and start.tzinfo is None:
            start = start.replace(tzinfo=timezone.utc)

        total_seconds = max(int((now - start).total_seconds()), 1)
        n = min(max_statements or 500, random.randint(300, 600))
        learner_ids = [f"fb{i:032x}"[:32] for i in range(1, 61)]
        verb_keys = [k for k, _ in VERB_WEIGHTS]
        weights = [w for _, w in VERB_WEIGHTS]

        statements = []
        for i in range(n):
            actor_name_val = random.choice(learner_ids)
            homepage = random.choice(HOMEPAGES)
            act_uuid, act_name = random.choice(ACTIVITIES)
            ts_offset = random.randint(0, total_seconds)
            ts = start + timedelta(seconds=ts_offset)

            verb_key = random.choices(verb_keys, weights=weights)[0]
            if verb:
                verb_key = next((k for k, v in VERBS.items() if v == verb), verb_key)

            score_raw = max(0, min(100, random.gauss(68, 20)))
            result: Dict = {}
            ctx: Dict = {
                "platform": random.choice(PLATFORMS),
                "registration": "",
                "extensions": {
                    "https://w3id.org/xapi/cmi5/context/extensions/sessionid":
                        f"https://edu.tactileo.fr/activitysessions/run/sess-{i:06d}",
                    "http://id.tincanapi.com/extension/invitee": random.choice(ROLES),
                },
                "contextActivities": {},
            }

            if verb_key in ("passed", "failed", "evaluated", "satisfied", "completed"):
                result["score"] = {"raw": round(score_raw, 1), "max": 100, "min": 0}
                result["success"] = score_raw >= 60
                result["duration"] = f"PT{random.randint(30, 3600)}S"

            if verb_key == "experienced":
                result["extensions"] = {
                    "https://w3id.org/xapi/cmi5/result/extensions/progress": random.randint(10, 100)
                }
                result["duration"] = f"PT{random.randint(5, 600)}S"

            if verb_key == "answered":
                interaction_types = ["choice", "sequencing", "long-fill-in", "true-false", "matching"]
                interaction_type = random.choice(interaction_types)

                is_success = random.random() > 0.3  # 70% success rate
                result["score"] = {"raw": 1 if is_success else 0, "max": 1, "min": 0}
                result["success"] = is_success
                result["duration"] = f"PT{random.randint(5, 120)}S"
                result["response"] = f"answer-{random.randint(1, 5)}"

                ctx["_temp_interaction_type"] = interaction_type

            if verb_key == "suspended":
                progress = random.choice([8, 25, 45, 67, 78, 93])
                ending_points = [
                    "https://xapi.maskott.com/identifiers/activity-item/item-001",
                    "https://xapi.maskott.com/identifiers/activity-item/item-002",
                    "https://xapi.maskott.com/identifiers/activity-item/item-003",
                ]
                result["extensions"] = {
                    "https://w3id.org/xapi/cmi5/result/extensions/progress": progress,
                    "http://id.tincanapi.com/extension/ending-point": random.choice(ending_points)
                }
                result["score"] = {"raw": progress}
                result["duration"] = f"PT{random.randint(60, 7200)}S"

            if verb_key == "resumed":
                ending_points = [
                    "https://xapi.maskott.com/identifiers/activity-item/item-001",
                    "https://xapi.maskott.com/identifiers/activity-item/item-002",
                ]
                result["extensions"] = {
                    "http://id.tincanapi.com/extension/ending-point": random.choice(ending_points)
                }

            if verb_key == "terminated":
                progress = random.choice([60, 75, 100])
                result["extensions"] = {
                    "https://w3id.org/xapi/cmi5/result/extensions/progress": progress
                }
                result["duration"] = f"PT{random.randint(60, 1800)}S"

            if verb_key == "launched":
                # Add launch metadata
                ctx["extensions"] = {
                    "https://w3id.org/xapi/cmi5/context/extensions/moveon": "CompletedOrPassed",
                    "https://w3id.org/xapi/cmi5/context/extensions/launchmode": "Normal",
                    "https://w3id.org/xapi/cmi5/context/extensions/masteryscore": "0.6"
                }

            if verb_key == "satisfied":
                result["score"] = {"raw": round(score_raw, 1), "max": 100, "min": 0, "scaled": round(score_raw / 100, 2)}
                result["completion"] = True
                result["duration"] = f"PT{random.randint(120, 3600)}S"
                result["extensions"] = {
                    "https://w3id.org/xapi/cmi5/result/extensions/progress": 100
                }

            if verb_key == "create":
                # Add team and planning metadata
                ctx["team"] = {
                    "member": [],
                    "account": {"name": group_session_id, "homePage": f"https://edu.tactileo.fr/activities/stats/{module_id}/?groupSessionId={group_session_id}"},
                    "objectType": "Group"
                }
                result["extensions"] = {
                    "http://id.tincanapi.com/extension/private-area": random.choice([True, False]),
                    "http://id.tincanapi.com/extension/planned-duration": "P30DT23H55M",
                    "http://id.tincanapi.com/extension/planned-start-time": (ts - timedelta(hours=1)).isoformat()
                }

            if verb_key == "logged-in":
                ctx["platform"] = "learningconnect-SSO"
                ctx["extensions"] = {
                    "http://id.tincanapi.com/extension/invitee": random.choice(["Teacher", "Student"])
                }

            session_num = i // random.randint(10, 20) + 1
            group_session_id = f"https://xapi.maskott.com/identifiers/groupsessions/{random.choice(['feb71653-c869-4a58-a259-b49394175f75', 'abc12345-1234-5678-abcd-aabbccddeeff', 'def67890-9876-5432-fedc-bbccddee0011'][:session_num % 3 + 1])}"

            attempt_num = i // random.randint(3, 8) + 1
            attempt_id = f"https://xapi.maskott.com/identifiers/attempts/{random.choice(['489636e0-b9c0-c8ff-c9ad-ecca2501b500', '123456e0-aaaa-bbbb-cccc-ddddeeee1111', '789012e0-ffff-eeee-dddd-ccccbbbb2222'][:attempt_num % 3 + 1])}"

            module_id = f"https://xapi.maskott.com/identifiers/modules/{act_uuid}"
            registration_id = f"{random.choice(['e5fb43d8-561d-474b-ab19-d840746c5d15', 'f1234567-89ab-cdef-0123-456789abcdef', 'a9876543-fedc-ba98-7654-321098765432'][:session_num % 3 + 1])}"

            act_type = "http://adlnet.gov/expapi/activities/module"
            act_definition = {
                "name": {"en-US": act_name, "fr-FR": act_name},
                "type": act_type,
            }

            if verb_key == "experienced":
                content_types = [
                    "https://w3id.org/xapi/acrossx/activities/webpage",
                    "http://adlnet.gov/expapi/activities/media",
                    "https://w3id.org/xapi/video/activity-type/video",
                ]
                act_definition["type"] = random.choice(content_types)
                if "video" in act_definition["type"]:
                    act_definition["moreInfo"] = "https://player.vimeo.com/video/192287033"
                elif "webpage" in act_definition["type"]:
                    act_definition["moreInfo"] = "https://taceduresourceassets.blob.core.windows.net/resources/intro-html5/train.html"
                act_definition["description"] = {"fr-FR": ""}
            elif verb_key == "answered":
                act_definition["type"] = "http://adlnet.gov/expapi/activities/cmi.interaction"
                interaction_type = ctx.get("_temp_interaction_type", "choice")
                act_definition["interactionType"] = interaction_type
                if interaction_type == "choice":
                    act_definition["correctResponsesPattern"] = [f"choice-{random.randint(1, 4)}"]
            elif verb_key == "completed":
                act_definition["type"] = "http://adlnet.gov/expapi/activities/attempt"

            # Clean up temp fields
            if "_temp_interaction_type" in ctx:
                del ctx["_temp_interaction_type"]

            # Set registration
            ctx["registration"] = registration_id

            # Build contextActivities
            ctx["contextActivities"] = {
                "parent": [
                    {
                        "id": module_id,
                        "objectType": "Activity",
                        "definition": {
                            "name": {"en-US": act_name, "fr-FR": act_name},
                            "type": "http://adlnet.gov/expapi/activities/module",
                            "description": {"en-US": f"Module: {act_name}", "fr-FR": f"Module: {act_name}"}
                        }
                    },
                    {
                        "id": group_session_id,
                        "objectType": "Activity",
                        "definition": {
                            "type": "http://adlnet.gov/expapi/activitytype/tutor-session"
                        }
                    }
                ],
                "grouping": [
                    {
                        "id": attempt_id,
                        "objectType": "Activity",
                        "definition": {
                            "type": "http://id.tincanapi.com/extension/attempt-id"
                        }
                    }
                ]
            }

            statements.append({
                "id": f"stmt-mock-{i:06d}",
                "timestamp": ts.isoformat(),
                "stored": ts.isoformat(),
                "version": "1.0.0",
                "actor": {
                    "objectType": "Agent",
                    "account": {"name": actor_name_val, "homePage": homepage},
                },
                "verb": {
                    "id": VERBS[verb_key],
                    "display": {"en-US": verb_key, "fr-FR": verb_key},
                },
                "object": {
                    "objectType": "Activity",
                    "id": f"https://xapi.maskott.com/identifiers/elements/{act_uuid}",
                    "definition": act_definition,
                },
                "result": result,
                "context": ctx,
            })

        return statements

    def get_all_statements_batched(self, since=None, until=None, batch_size=DEFAULT_PAGE_SIZE, **kw):
        return self.get_statements(since=since, until=until)