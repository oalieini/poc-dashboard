"""
xAPI Learning Analytics - Indicator Calculator
Computes all indicators defined in config/indicators.yaml from raw statement lists.
"""

import re
import logging
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Optional, Tuple, Any
from collections import defaultdict
from statistics import mean, median, stdev

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


SESSION_GAP_MINUTES = 240



class StatementParser:
    """Utility helpers to extract fields from xAPI statement dicts."""

    @staticmethod
    def actor_id(stmt: Dict) -> str:
        actor = stmt.get("actor", {})
        mbox = actor.get("mbox", "")
        account = actor.get("account", {})
        return mbox or account.get("homePage", "") + "|" + account.get("name", "")

    @staticmethod
    def actor_name(stmt: Dict) -> str:
        return stmt.get("actor", {}).get("name", StatementParser.actor_id(stmt))

    @staticmethod
    def verb_id(stmt: Dict) -> str:
        return stmt.get("verb", {}).get("id", "")

    @staticmethod
    def verb_display(stmt: Dict) -> str:
        display = stmt.get("verb", {}).get("display", {})
        return display.get("en-US", display.get("en", StatementParser.verb_id(stmt)))

    @staticmethod
    def activity_id(stmt: Dict) -> str:
        return stmt.get("object", {}).get("id", "")

    @staticmethod
    def activity_name(stmt: Dict) -> str:
        definition = stmt.get("object", {}).get("definition", {})
        names = definition.get("name", {})
        return names.get("en-US", names.get("en", StatementParser.activity_id(stmt)))

    @staticmethod
    def timestamp(stmt: Dict) -> Optional[datetime]:
        ts_str = stmt.get("timestamp")
        if not ts_str:
            return None
        try:
            # Handle various ISO formats
            ts_str = ts_str.replace("Z", "+00:00")
            return datetime.fromisoformat(ts_str)
        except Exception:
            return None

    @staticmethod
    def score_scaled(stmt: Dict) -> Optional[float]:
        result = stmt.get("result", {})
        score = result.get("score", {})
        scaled = score.get("scaled")
        if scaled is not None:
            return float(scaled) * 100  # Convert to 0–100
        raw = score.get("raw")
        max_s = score.get("max", 100)
        min_s = score.get("min", 0)
        if raw is not None and max_s > min_s:
            return (float(raw) - float(min_s)) / (float(max_s) - float(min_s)) * 100
        return None

    @staticmethod
    def duration_minutes(stmt: Dict) -> Optional[float]:
        """Parse ISO 8601 duration from result.duration → minutes."""
        duration_str = stmt.get("result", {}).get("duration", "")
        if not duration_str:
            return None
        try:
            pattern = r"P(?:(\d+)Y)?(?:(\d+)M)?(?:(\d+)D)?T?(?:(\d+)H)?(?:(\d+)M)?(?:([\d.]+)S)?"
            m = re.match(pattern, duration_str)
            if not m:
                return None
            years, months, days, hours, minutes, seconds = (
                float(x) if x else 0 for x in m.groups()
            )
            return years * 525960 + months * 43800 + days * 1440 + hours * 60 + minutes + seconds / 60
        except Exception:
            return None

    @staticmethod
    def progress(stmt: Dict) -> Optional[float]:
        extensions = stmt.get("result", {}).get("extensions", {})
        for key, val in extensions.items():
            if "progress" in key.lower():
                return float(val)
        return None

    @staticmethod
    def success(stmt: Dict) -> Optional[bool]:
        result = stmt.get("result", {})
        return result.get("success")

    @staticmethod
    def group_session_id(stmt: Dict) -> Optional[str]:
        """
        Extract session ID from context.contextActivities.parent (Maskott TRAX format).

        """
        parents = stmt.get("context", {}).get("contextActivities", {}).get("parent", [])
        for parent in parents:
            parent_type = parent.get("definition", {}).get("type", "")
            parent_id = parent.get("id", "")
            # Match tutor-session type OR URLs containing groupsessions/sessions/courses
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
        """Extract attempt ID from context.contextActivities.grouping (Maskott TRAX format)."""
        groupings = stmt.get("context", {}).get("contextActivities", {}).get("grouping", [])
        for grouping in groupings:
            grouping_type = grouping.get("definition", {}).get("type", "")
            grouping_id = grouping.get("id", "")
            if "attempt" in grouping_type or "attempts" in grouping_id:
                return grouping_id
        return None

    @staticmethod
    def module_id(stmt: Dict) -> Optional[str]:
        """Extract module ID from context.contextActivities.parent (Maskott TRAX format)."""
        parents = stmt.get("context", {}).get("contextActivities", {}).get("parent", [])
        for parent in parents:
            parent_type = parent.get("definition", {}).get("type", "")
            parent_id = parent.get("id", "")
            if "module" in parent_type or "modules" in parent_id:
                return parent_id
        return None

    @staticmethod
    def format_duration(minutes: float) -> str:
        """
        Format duration intelligently based on length.
        """
        if minutes == 0:
            return "0 min"
        elif minutes < 60:
            return f"{minutes:.1f} min"
        elif minutes < 1440:  # Less than 24 hours
            hours = minutes / 60
            return f"{hours:.1f} hr"
        else:  # 24 hours or more
            days = minutes / 1440
            if days < 7:
                return f"{days:.1f} days"
            else:
                weeks = days / 7
                return f"{weeks:.1f} weeks"

    @staticmethod
    def registration(stmt: Dict) -> Optional[str]:
        """Extract registration UUID (launch session identifier)."""
        return stmt.get("context", {}).get("registration")

    @staticmethod
    def platform(stmt: Dict) -> Optional[str]:
        """Extract platform from context."""
        return stmt.get("context", {}).get("platform")

    @staticmethod
    def progress_percent(stmt: Dict) -> Optional[float]:
        """Extract progress percentage from extensions (0-100)."""
        exts = stmt.get("result", {}).get("extensions", {})
        for key, val in exts.items():
            if "progress" in key.lower():
                try:
                    return float(val)
                except (ValueError, TypeError):
                    return None
        return None

    @staticmethod
    def ending_point(stmt: Dict) -> Optional[str]:
        """Extract ending point (where student stopped) from extensions."""
        exts = stmt.get("result", {}).get("extensions", {})
        for key, val in exts.items():
            if "ending-point" in key.lower() or "ending_point" in key.lower():
                return str(val) if val else None
        return None

    @staticmethod
    def interaction_type(stmt: Dict) -> Optional[str]:
        """Extract interaction type for answered statements."""
        return stmt.get("object", {}).get("definition", {}).get("interactionType")

    @staticmethod
    def content_type(stmt: Dict) -> Optional[str]:
        """Extract content type from object.definition.type."""
        obj_type = stmt.get("object", {}).get("definition", {}).get("type", "")
        if "video" in obj_type:
            return "video"
        elif "webpage" in obj_type:
            return "webpage"
        elif "media" in obj_type:
            return "media"
        elif "interaction" in obj_type:
            return "interaction"
        elif "module" in obj_type:
            return "module"
        elif "attempt" in obj_type:
            return "attempt"
        return "other"

    @staticmethod
    def response(stmt: Dict) -> Optional[str]:
        """Extract student response from result."""
        resp = stmt.get("result", {}).get("response")
        if resp and isinstance(resp, str) and len(resp) > 200:
            return resp[:200] + "..."  # Truncate long responses
        return resp

    @staticmethod
    def correct_pattern(stmt: Dict) -> Optional[str]:
        """Extract correct response pattern."""
        patterns = stmt.get("object", {}).get("definition", {}).get("correctResponsesPattern", [])
        return patterns[0] if patterns else None

    @staticmethod
    def moveon_criteria(stmt: Dict) -> Optional[str]:
        """Extract move-on criteria from launch extensions."""
        exts = stmt.get("context", {}).get("extensions", {})
        for key, val in exts.items():
            if "moveon" in key.lower():
                return str(val)
        return None

    @staticmethod
    def mastery_score_threshold(stmt: Dict) -> Optional[float]:
        """Extract mastery score threshold from launch extensions."""
        exts = stmt.get("context", {}).get("extensions", {})
        for key, val in exts.items():
            if "masteryscore" in key.lower():
                try:
                    return float(val)
                except (ValueError, TypeError):
                    return None
        return None

    @staticmethod
    def planned_duration_minutes(stmt: Dict) -> Optional[float]:
        """Extract planned duration from create statements (ISO 8601)."""
        exts = stmt.get("result", {}).get("extensions", {})
        for key, val in exts.items():
            if "planned-duration" in key.lower() or "planned_duration" in key.lower():
                if val:
                    # Reuse existing duration parser
                    return StatementParser.duration_minutes({"result": {"duration": val}})
        return None

    @staticmethod
    def team_size(stmt: Dict) -> int:
        """Extract team member count from create statements."""
        members = stmt.get("context", {}).get("team", {}).get("member", [])
        return len(members) if isinstance(members, list) else 0

    @staticmethod
    def progress_percent(stmt: Dict) -> Optional[float]:
        """Extract progress percentage from result extensions (0-100)."""
        exts = stmt.get("result", {}).get("extensions", {})
        for key, val in exts.items():
            if "progress" in key.lower():
                try:
                    return float(val)
                except (ValueError, TypeError):
                    pass
        return None

    @staticmethod
    def ending_point(stmt: Dict) -> Optional[str]:
        """Extract ending point (where activity was suspended/resumed)."""
        exts = stmt.get("result", {}).get("extensions", {})
        for key, val in exts.items():
            if "ending-point" in key.lower() or "ending_point" in key.lower():
                return str(val) if val else None
        return None

    @staticmethod
    def interaction_type(stmt: Dict) -> Optional[str]:
        """Extract interaction type for answered/question statements."""
        return stmt.get("object", {}).get("definition", {}).get("interactionType")

    @staticmethod
    def content_type(stmt: Dict) -> Optional[str]:
        """Extract and categorize content type from object.definition.type."""
        obj_type = stmt.get("object", {}).get("definition", {}).get("type", "")
        if not obj_type:
            return None

        obj_type_lower = obj_type.lower()
        if "video" in obj_type_lower:
            return "video"
        elif "webpage" in obj_type_lower:
            return "webpage"
        elif "media" in obj_type_lower:
            return "media"
        elif "interaction" in obj_type_lower or "cmi.interaction" in obj_type_lower:
            return "interaction"
        elif "module" in obj_type_lower:
            return "module"
        elif "attempt" in obj_type_lower:
            return "attempt"
        elif "tutor-session" in obj_type_lower:
            return "session"
        return "other"

    @staticmethod
    def response(stmt: Dict) -> Optional[str]:
        """Extract student response from result."""
        resp = stmt.get("result", {}).get("response")
        return str(resp) if resp else None

    @staticmethod
    def correct_pattern(stmt: Dict) -> Optional[str]:
        """Extract correct response pattern from definition."""
        patterns = stmt.get("object", {}).get("definition", {}).get("correctResponsesPattern", [])
        return patterns[0] if patterns else None

    @staticmethod
    def moveon_criteria(stmt: Dict) -> Optional[str]:
        """Extract moveon completion criteria from launched statements."""
        exts = stmt.get("context", {}).get("extensions", {})
        for key, val in exts.items():
            if "moveon" in key.lower():
                return str(val) if val else None
        return None

    @staticmethod
    def mastery_score_threshold(stmt: Dict) -> Optional[float]:
        """Extract mastery score threshold from launched statements."""
        exts = stmt.get("context", {}).get("extensions", {})
        for key, val in exts.items():
            if "masteryscore" in key.lower():
                try:
                    return float(val)
                except (ValueError, TypeError):
                    pass
        return None

    @staticmethod
    def planned_duration_minutes(stmt: Dict) -> Optional[float]:
        """Extract planned duration from create statements."""
        exts = stmt.get("result", {}).get("extensions", {})
        for key, val in exts.items():
            if "planned-duration" in key.lower() or "planned_duration" in key.lower():
                if val:
                    # Parse ISO 8601 duration
                    return StatementParser.duration_minutes({"result": {"duration": val}})
        return None

    @staticmethod
    def team_size(stmt: Dict) -> int:
        """Extract team member count from create statements."""
        members = stmt.get("context", {}).get("team", {}).get("member", [])
        return len(members) if isinstance(members, list) else 0

    @staticmethod
    def role(stmt: Dict) -> Optional[str]:
        """Extract role (Teacher/Student) from context extensions."""
        exts = stmt.get("context", {}).get("extensions", {})
        for key, val in exts.items():
            if "invitee" in key.lower():
                return str(val) if val else None
        return None


class IndicatorCalculator:
    """
    Computes all learning analytics indicators from a list of xAPI statements.
    All public methods return a dict with keys: value, trend, details, chart_data.
    """

    # Standard ADL verbs
    VERB_COMPLETED = "http://adlnet.gov/expapi/verbs/completed"
    VERB_PASSED = "http://adlnet.gov/expapi/verbs/passed"
    VERB_FAILED = "http://adlnet.gov/expapi/verbs/failed"
    VERB_INITIALIZED = "http://adlnet.gov/expapi/verbs/initialized"
    VERB_LAUNCHED = "http://adlnet.gov/expapi/verbs/launched"
    VERB_ANSWERED = "http://adlnet.gov/expapi/verbs/answered"
    VERB_SUSPENDED = "http://adlnet.gov/expapi/verbs/suspended"
    VERB_RESUMED = "http://adlnet.gov/expapi/verbs/resumed"
    VERB_TERMINATED = "http://adlnet.gov/expapi/verbs/terminated"
    VERB_EXPERIENCED = "http://adlnet.gov/expapi/verbs/experienced"
    VERB_SATISFIED = "http://adlnet.gov/expapi/verbs/satisfied"

    # Additional verbs specific to your TRAX LRS
    VERB_LOGGED_IN = "https://w3id.org/xapi/adl/verbs/logged-in"
    VERB_CREATED = "http://activitystrea.ms/schema/1.0/create"
    VERB_EVALUATED = "http://www.tincanapi.co.uk/verbs/evaluated"

    # Legacy aliases (kept for backward compatibility)
    VERB_SCORED = VERB_PASSED  # TRAX uses 'passed' instead of 'scored'
    VERB_PROGRESSED = VERB_EXPERIENCED  # Similar concept
    VERB_INTERACTED = VERB_ANSWERED  # User interaction
    VERB_ATTEMPTED = VERB_LAUNCHED  # Activity attempt
    VERB_SHARED = VERB_CREATED  # Content creation/sharing

    def __init__(self, statements: List[Dict], prev_statements: Optional[List[Dict]] = None):
        """
        Args:
            statements: Current period statements
            prev_statements: Previous period statements (for trend %)
        """
        self.stmts = statements
        self.prev_stmts = prev_statements or []
        self._df: Optional[pd.DataFrame] = None
        self._prev_df: Optional[pd.DataFrame] = None

    def _build_df(self, stmts: List[Dict]) -> pd.DataFrame:
        if not stmts:
            return pd.DataFrame()
        rows = []
        for s in stmts:
            rows.append({
                "stmt_id": s.get("id", ""),
                "actor_id": StatementParser.actor_id(s),
                "actor_name": StatementParser.actor_name(s),
                "verb_id": StatementParser.verb_id(s),
                "verb_display": StatementParser.verb_display(s),
                "activity_id": StatementParser.activity_id(s),
                "activity_name": StatementParser.activity_name(s),
                "timestamp": StatementParser.timestamp(s),
                "score": StatementParser.score_scaled(s),
                "duration_min": StatementParser.duration_minutes(s),
                "progress": StatementParser.progress(s),
                "success": StatementParser.success(s),
                "group_session_id": StatementParser.group_session_id(s),
                "attempt_id": StatementParser.attempt_id(s),
                "module_id": StatementParser.module_id(s),
                "registration": StatementParser.registration(s),
                "platform": StatementParser.platform(s),
                # New fields from TRAX analysis
                "progress_percent": StatementParser.progress_percent(s),
                "ending_point": StatementParser.ending_point(s),
                "interaction_type": StatementParser.interaction_type(s),
                "content_type": StatementParser.content_type(s),
                "response": StatementParser.response(s),
                "correct_pattern": StatementParser.correct_pattern(s),
                "moveon": StatementParser.moveon_criteria(s),
                "mastery_score_threshold": StatementParser.mastery_score_threshold(s),
                "role": StatementParser.role(s),
                "planned_duration_min": StatementParser.planned_duration_minutes(s),
                "team_size": StatementParser.team_size(s),
            })
        df = pd.DataFrame(rows)
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
            df = df.sort_values("timestamp")
            df["hour"] = df["timestamp"].dt.hour
            df["date"] = df["timestamp"].dt.date
        return df

    @property
    def df(self) -> pd.DataFrame:
        if self._df is None:
            self._df = self._build_df(self.stmts)
        return self._df

    @property
    def prev_df(self) -> pd.DataFrame:
        if self._prev_df is None:
            self._prev_df = self._build_df(self.prev_stmts)
        return self._prev_df

    def _trend(self, current: float, previous: float) -> Optional[float]:
        """Compute % change vs previous period."""
        if previous == 0:
            return None
        return round((current - previous) / abs(previous) * 100, 1)

    def _verb_df(self, df: pd.DataFrame, verb_id: str) -> pd.DataFrame:
        return df[df["verb_id"] == verb_id] if not df.empty else df

    def _safe_dropna(self, df: pd.DataFrame, col: str) -> pd.DataFrame:
        """dropna on a column only if the column actually exists."""
        if df.empty or col not in df.columns:
            return pd.DataFrame(columns=df.columns.tolist() + ([col] if col not in df.columns else []))
        return df.dropna(subset=[col])

    def _sessions(self, df: pd.DataFrame) -> List[Dict]:
        """
        Extract sessions per actor based on parent ID (group_session_id).
        Sessions are differentiated by the parent context activity, which can be:
        - Group sessions: Can contain multiple modules
        - Course sessions: Contain only one module
        """
        if df.empty:
            return []

        # Check if group_session_id column exists and has data
        if "group_session_id" not in df.columns or df["group_session_id"].isna().all():
            logger.warning("No group_session_id found in statements - cannot identify sessions")
            return []

        sessions = []

        # Group by actor and session ID
        for (actor, session_id), group in df.groupby(["actor_id", "group_session_id"], dropna=True):
            if not session_id or session_id == "" or pd.isna(session_id):
                continue

            group = group.sort_values("timestamp")
            start_ts = group["timestamp"].min()
            end_ts = group["timestamp"].max()
            duration = (end_ts - start_ts).total_seconds() / 60 if pd.notna(start_ts) and pd.notna(end_ts) else 0

            # Identify session type (group vs course)
            session_type = "session"
            if "groupsessions" in session_id.lower():
                session_type = "group"
            elif "course" in session_id.lower() and "session" in session_id.lower():
                session_type = "course"

            # Count unique modules in this session
            modules = []
            if "module_id" in group.columns:
                modules = [m for m in group["module_id"].dropna().unique() if m]

            sessions.append({
                "actor_id": actor,
                "session_id": session_id,
                "session_type": session_type,
                "start": start_ts,
                "end": end_ts,
                "duration_min": duration,
                "statement_count": len(group),
                "module_count": len(modules),
            })

        return sessions

    # ─────────────────────────────────────────────
    # ENGAGEMENT INDICATORS
    # ─────────────────────────────────────────────

    def active_learners(self) -> Dict:
        curr = self.df["actor_id"].nunique() if not self.df.empty else 0
        prev = self.prev_df["actor_id"].nunique() if not self.prev_df.empty else 0
        daily = {}
        if not self.df.empty and "date" in self.df.columns:
            daily = self.df.groupby("date")["actor_id"].nunique().to_dict()
        return {
            "value": curr,
            "trend": self._trend(curr, prev),
            "chart_data": {"x": [str(k) for k in daily.keys()], "y": list(daily.values())},
            "details": f"{curr} unique learners active",
        }

    def total_statements(self) -> Dict:
        curr = len(self.stmts)
        prev = len(self.prev_stmts)
        daily = {}
        if not self.df.empty and "date" in self.df.columns:
            daily = self.df.groupby("date").size().to_dict()
        return {
            "value": curr,
            "trend": self._trend(curr, prev),
            "chart_data": {"x": [str(k) for k in daily.keys()], "y": list(daily.values())},
            "details": f"{curr} total statements",
        }

    def sessions(self) -> Dict:
        curr_sessions = self._sessions(self.df)
        prev_sessions = self._sessions(self.prev_df)
        curr = len(curr_sessions)
        prev = len(prev_sessions)

        # Count session types
        group_count = sum(1 for s in curr_sessions if s.get("session_type") == "group")
        course_count = sum(1 for s in curr_sessions if s.get("session_type") == "course")

        return {
            "value": curr,
            "trend": self._trend(curr, prev),
            "chart_data": {},
            "details": f"{curr} sessions ({group_count} group, {course_count} course)",
        }

    def avg_session_duration(self) -> Dict:
        curr_sessions = self._sessions(self.df)
        prev_sessions = self._sessions(self.prev_df)
        durations = [s["duration_min"] for s in curr_sessions if s["duration_min"] > 0]
        prev_durations = [s["duration_min"] for s in prev_sessions if s["duration_min"] > 0]
        curr = round(mean(durations), 1) if durations else 0
        prev = round(mean(prev_durations), 1) if prev_durations else 0
        return {
            "value": curr,
            "trend": self._trend(curr, prev),
            "chart_data": {"values": durations[:200]},
            "details": f"Average {curr:.1f} min per session",
        }

    def statements_per_learner(self) -> Dict:
        n_learners = self.df["actor_id"].nunique() if not self.df.empty else 1
        curr = round(len(self.stmts) / max(n_learners, 1), 1)
        p_learners = self.prev_df["actor_id"].nunique() if not self.prev_df.empty else 1
        prev = round(len(self.prev_stmts) / max(p_learners, 1), 1)
        return {
            "value": curr,
            "trend": self._trend(curr, prev),
            "chart_data": {},
            "details": f"{curr} statements per active learner",
        }

    # ─────────────────────────────────────────────
    # COMPLETION INDICATORS
    # ─────────────────────────────────────────────

    def completion_rate(self) -> Dict:
        if self.df.empty:
            return {"value": 0, "trend": None, "chart_data": {}, "details": "No data"}
        completed = len(self._verb_df(self.df, self.VERB_COMPLETED))
        # Count attempts as INITIALIZED or LAUNCHED (your TRAX doesn't have 'attempted' verb)
        attempted = len(self.df[self.df["verb_id"].isin([
            self.VERB_INITIALIZED, self.VERB_LAUNCHED, self.VERB_COMPLETED
        ])])
        curr = round(completed / max(attempted, 1) * 100, 1)

        p_completed = len(self._verb_df(self.prev_df, self.VERB_COMPLETED))
        p_attempted = len(self.prev_df[self.prev_df["verb_id"].isin([
            self.VERB_INITIALIZED, self.VERB_LAUNCHED, self.VERB_COMPLETED
        ])]) if not self.prev_df.empty else 0
        prev = round(p_completed / max(p_attempted, 1) * 100, 1)

        return {
            "value": curr,
            "trend": self._trend(curr, prev),
            "chart_data": {"completed": completed, "attempted": attempted},
            "details": f"{completed} completions out of {attempted} attempts",
        }

    def completions_total(self) -> Dict:
        curr = len(self._verb_df(self.df, self.VERB_COMPLETED))
        prev = len(self._verb_df(self.prev_df, self.VERB_COMPLETED))
        daily = {}
        if not self.df.empty:
            daily = self._verb_df(self.df, self.VERB_COMPLETED).groupby("date").size().to_dict()
        return {
            "value": curr,
            "trend": self._trend(curr, prev),
            "chart_data": {"x": [str(k) for k in daily.keys()], "y": list(daily.values())},
            "details": f"{curr} total completions",
        }

    def dropout_rate(self) -> Dict:
        if self.df.empty:
            return {"value": 0, "trend": None, "chart_data": {}, "details": "No data"}
        started = self.df["actor_id"].nunique()
        completed_actors = self._verb_df(self.df, self.VERB_COMPLETED)["actor_id"].nunique()
        curr = round((started - completed_actors) / max(started, 1) * 100, 1)
        p_started = self.prev_df["actor_id"].nunique() if not self.prev_df.empty else 0
        p_completed = self._verb_df(self.prev_df, self.VERB_COMPLETED)["actor_id"].nunique() if not self.prev_df.empty else 0
        prev = round((p_started - p_completed) / max(p_started, 1) * 100, 1)
        return {
            "value": curr,
            "trend": self._trend(curr, prev),
            "chart_data": {},
            "details": f"{started - completed_actors} learners dropped out",
        }

    def progress_avg(self) -> Dict:
        if self.df.empty:
            return {"value": 0, "trend": None, "chart_data": {}, "details": "No data"}
        prog_df = self._safe_dropna(self._verb_df(self.df, self.VERB_PROGRESSED), "progress")
        curr = round(prog_df["progress"].mean(), 1) if not prog_df.empty else 0
        p_prog = self._safe_dropna(self._verb_df(self.prev_df, self.VERB_PROGRESSED), "progress")
        prev = round(p_prog["progress"].mean(), 1) if not p_prog.empty else 0
        hist_vals = prog_df["progress"].tolist()
        return {
            "value": curr,
            "trend": self._trend(curr, prev),
            "chart_data": {"values": hist_vals},
            "details": f"Average progress: {curr:.1f}%",
        }

    # ─────────────────────────────────────────────
    # PERFORMANCE INDICATORS
    # ─────────────────────────────────────────────

    def avg_score(self) -> Dict:
        # TRAX uses 'passed' and 'failed' with scores, not 'scored' verb
        scored = self.df[self.df["verb_id"].isin([self.VERB_PASSED, self.VERB_FAILED])]
        scored = self._safe_dropna(scored, "score")
        curr = round(scored["score"].mean(), 1) if not scored.empty else 0

        p_scored = self.prev_df[self.prev_df["verb_id"].isin([self.VERB_PASSED, self.VERB_FAILED])] if not self.prev_df.empty else pd.DataFrame()
        p_scored = self._safe_dropna(p_scored, "score")
        prev = round(p_scored["score"].mean(), 1) if not p_scored.empty else 0

        daily = {}
        if not scored.empty and "date" in scored.columns:
            daily = scored.groupby("date")["score"].mean().round(1).to_dict()
        return {
            "value": curr,
            "trend": self._trend(curr, prev),
            "chart_data": {"x": [str(k) for k in daily.keys()], "y": list(daily.values())},
            "details": f"Mean score: {curr:.1f}/100",
        }

    def pass_rate(self) -> Dict:
        passed = len(self._verb_df(self.df, self.VERB_PASSED))
        failed = len(self._verb_df(self.df, self.VERB_FAILED))
        total = passed + failed
        curr = round(passed / max(total, 1) * 100, 1)
        p_passed = len(self._verb_df(self.prev_df, self.VERB_PASSED))
        p_failed = len(self._verb_df(self.prev_df, self.VERB_FAILED))
        prev = round(p_passed / max(p_passed + p_failed, 1) * 100, 1)
        return {
            "value": curr,
            "trend": self._trend(curr, prev),
            "chart_data": {"passed": passed, "failed": failed},
            "details": f"{passed} passed / {failed} failed",
        }

    def fail_rate(self) -> Dict:
        p = self.pass_rate()
        curr = round(100 - p["value"], 1)
        prev_pass = (p["trend"] or 0)
        return {
            "value": curr,
            "trend": -prev_pass if prev_pass else None,
            "chart_data": p["chart_data"],
            "details": f"Fail rate: {curr:.1f}%",
        }

    def score_distribution(self) -> dict:
        scored = self._safe_dropna(self.df[self.df["verb_id"].isin([self.VERB_PASSED, self.VERB_FAILED])], "score")
        values = list(scored["score"]) if not scored.empty else []
        bins = list(range(0, 101, 10))
        labels = [f"{bins[i]:.0f}-{bins[i+1]:.0f}" for i in range(len(bins) - 1)]
        if values:
            import numpy as np
            hist, _ = np.histogram(values, bins=bins)
            hist_list = [int(x) for x in hist]
        else:
            hist_list = [0] * (len(bins) - 1)
        return {
            "value": round(sum(values) / len(values), 1) if values else 0,
            "trend": None,
            "chart_data": {"x": labels, "y": hist_list},
            "details": f"{len(values)} scored statements",
        }

    def mastery_rate(self, threshold: float = 80.0) -> Dict:
        scored = self._safe_dropna(self.df[self.df["verb_id"].isin([self.VERB_PASSED, self.VERB_FAILED])], "score")
        curr = round(len(scored[scored["score"] >= threshold]) / max(len(scored), 1) * 100, 1) if not scored.empty else 0
        p_scored = self._safe_dropna(self.prev_df[self.prev_df["verb_id"].isin([self.VERB_PASSED, self.VERB_FAILED])] if not self.prev_df.empty else pd.DataFrame(), "score")
        prev = round(len(p_scored[p_scored["score"] >= threshold]) / max(len(p_scored), 1) * 100, 1) if not p_scored.empty else 0
        return {
            "value": curr,
            "trend": self._trend(curr, prev),
            "chart_data": {},
            "details": f"{curr:.1f}% scored ≥ {threshold:.0f}",
        }

    def median_score(self) -> Dict:
        scored = self._safe_dropna(self.df[self.df["verb_id"].isin([self.VERB_PASSED, self.VERB_FAILED])], "score")
        curr = round(scored["score"].median(), 1) if not scored.empty else 0
        return {
            "value": curr,
            "trend": None,
            "chart_data": {},
            "details": f"Median score: {curr:.1f}/100",
        }

    def attempts_per_pass(self) -> Dict:
        if self.df.empty:
            return {"value": 0, "trend": None, "chart_data": {}, "details": "No data"}
        attempt_counts = []
        passed_actors = self._verb_df(self.df, self.VERB_PASSED)["actor_id"].unique()
        for actor in passed_actors:
            actor_df = self.df[self.df["actor_id"] == actor]
            n_attempts = len(actor_df[actor_df["verb_id"].isin([
                self.VERB_LAUNCHED, self.VERB_INITIALIZED
            ])])
            attempt_counts.append(max(n_attempts, 1))
        curr = round(mean(attempt_counts), 1) if attempt_counts else 1
        return {
            "value": curr,
            "trend": None,
            "chart_data": {"values": list(attempt_counts)},
            "details": f"Avg {curr:.1f} attempts before passing",
        }

    # ─────────────────────────────────────────────
    # INTERACTION INDICATORS
    # ─────────────────────────────────────────────

    def interactions_total(self) -> Dict:
        curr = len(self._verb_df(self.df, self.VERB_INTERACTED))
        prev = len(self._verb_df(self.prev_df, self.VERB_INTERACTED))
        return {
            "value": curr,
            "trend": self._trend(curr, prev),
            "chart_data": {},
            "details": f"{curr} interaction events",
        }

    def response_accuracy(self) -> Dict:
        answered = self._safe_dropna(self._verb_df(self.df, self.VERB_ANSWERED), "success")
        total = len(answered)
        correct = len(answered[answered["success"] == True])
        curr = round(correct / max(total, 1) * 100, 1)
        return {
            "value": curr,
            "trend": None,
            "chart_data": {"correct": correct, "incorrect": total - correct},
            "details": f"{correct}/{total} correct responses",
        }

    def verb_distribution(self) -> Dict:
        if self.df.empty:
            return {"value": 0, "trend": None, "chart_data": {}, "details": "No data"}
        dist = self.df.groupby("verb_display").size().sort_values(ascending=False)
        return {
            "value": len(dist),
            "trend": None,
            "chart_data": {"labels": dist.index.tolist(), "values": dist.values.tolist()},
            "details": f"{len(dist)} distinct verbs used",
        }

    def most_active_hours(self) -> Dict:
        if self.df.empty or "hour" not in self.df.columns:
            return {"value": 0, "trend": None, "chart_data": {}, "details": "No data"}
        hourly = self.df.groupby("hour").size().reindex(range(24), fill_value=0)
        peak_hour = int(hourly.idxmax())
        return {
            "value": peak_hour,
            "trend": None,
            "chart_data": {"x": list(range(24)), "y": hourly.values.tolist()},
            "details": f"Peak activity at {peak_hour:02d}:00",
        }

    def resumed_rate(self) -> Dict:
        suspended = len(self._verb_df(self.df, self.VERB_SUSPENDED))
        resumed = len(self._verb_df(self.df, self.VERB_RESUMED))
        curr = round(resumed / max(suspended, 1) * 100, 1)
        return {
            "value": curr,
            "trend": None,
            "chart_data": {"suspended": suspended, "resumed": resumed},
            "details": f"{resumed} resumed out of {suspended} suspended",
        }

    # ─────────────────────────────────────────────
    # SOCIAL INDICATORS
    # ─────────────────────────────────────────────

    def shared_count(self) -> Dict:
        curr = len(self._verb_df(self.df, self.VERB_SHARED))
        prev = len(self._verb_df(self.prev_df, self.VERB_SHARED))
        return {"value": curr, "trend": self._trend(curr, prev), "chart_data": {}, "details": f"{curr} shares"}

    # ─────────────────────────────────────────────
    # RETENTION INDICATORS
    # ─────────────────────────────────────────────

    def retention(self, days: int = 7) -> Dict:
        if self.df.empty or "timestamp" not in self.df.columns:
            return {"value": 0, "trend": None, "chart_data": {}, "details": "No data"}
        first_seen = self.df.groupby("actor_id")["timestamp"].min()
        retained = 0
        total = len(first_seen)
        for actor_id, first_ts in first_seen.items():
            target_day = first_ts + pd.Timedelta(days=days)
            actor_df = self.df[self.df["actor_id"] == actor_id]
            # Check if active within ±1 day of target
            window_start = target_day - pd.Timedelta(days=1)
            window_end = target_day + pd.Timedelta(days=1)
            active = actor_df[
                (actor_df["timestamp"] >= window_start) &
                (actor_df["timestamp"] <= window_end)
            ]
            if len(active) > 0:
                retained += 1
        curr = round(retained / max(total, 1) * 100, 1)
        return {
            "value": curr,
            "trend": None,
            "chart_data": {"retained": retained, "total": total},
            "details": f"{retained}/{total} learners retained at day {days}",
        }

    def churn_rate(self) -> Dict:
        if self.df.empty or self.prev_df.empty:
            return {"value": 0, "trend": None, "chart_data": {}, "details": "Insufficient data"}
        prev_actors = set(self.prev_df["actor_id"].unique())
        curr_actors = set(self.df["actor_id"].unique())
        churned = len(prev_actors - curr_actors)
        curr = round(churned / max(len(prev_actors), 1) * 100, 1)
        return {
            "value": curr,
            "trend": None,
            "chart_data": {},
            "details": f"{churned} learners churned from previous period",
        }

    # ─────────────────────────────────────────────
    # CONTENT ANALYTICS
    # ─────────────────────────────────────────────

    def top_activities(self, n: int = 10) -> Dict:
        if self.df.empty:
            return {"value": 0, "trend": None, "chart_data": {}, "details": "No data"}
        top = (
            self.df.groupby(["activity_id", "activity_name"])
            .size()
            .reset_index(name="count")
            .sort_values("count", ascending=False)
            .head(n)
        )
        return {
            "value": len(top),
            "trend": None,
            "chart_data": {
                "labels": top["activity_name"].tolist(),
                "values": top["count"].tolist(),
            },
            "details": f"Top {n} most accessed activities",
        }

    def hardest_activities(self, n: int = 10) -> Dict:
        if self.df.empty:
            return {"value": 0, "trend": None, "chart_data": {}, "details": "No data"}
        scored = self._safe_dropna(self.df[self.df["verb_id"].isin([self.VERB_PASSED, self.VERB_FAILED])], "score")
        if scored.empty:
            return {"value": 0, "trend": None, "chart_data": {}, "details": "No scored data"}
        avg_by_activity = (
            scored.groupby(["activity_id", "activity_name"])["score"]
            .mean()
            .reset_index()
            .sort_values("score")
            .head(n)
        )
        return {
            "value": len(avg_by_activity),
            "trend": None,
            "chart_data": {
                "labels": avg_by_activity["activity_name"].tolist(),
                "values": avg_by_activity["score"].round(1).tolist(),
            },
            "details": f"Bottom {n} activities by average score",
        }

    def avg_time_per_activity(self) -> Dict:
        if self.df.empty:
            return {"value": 0, "trend": None, "chart_data": {}, "details": "No data"}
        duration_df = self._safe_dropna(self.df, "duration_min")
        if duration_df.empty:
            return {"value": 0, "trend": None, "chart_data": {}, "details": "No duration data"}
        avg_by_act = (
            duration_df.groupby("activity_name")["duration_min"]
            .mean()
            .sort_values(ascending=False)
            .head(10)
        )
        return {
            "value": round(duration_df["duration_min"].mean(), 1),
            "trend": None,
            "chart_data": {
                "labels": avg_by_act.index.tolist(),
                "values": avg_by_act.round(1).tolist(),
            },
            "details": "Average minutes spent per activity",
        }

    def content_dropout_map(self) -> Dict:
        if self.df.empty:
            return {"value": 0, "trend": None, "chart_data": {}, "details": "No data"}
        started = self.df[self.df["verb_id"].isin([
            self.VERB_INITIALIZED, self.VERB_ATTEMPTED
        ])].groupby("activity_name")["actor_id"].nunique()
        completed = self._verb_df(self.df, self.VERB_COMPLETED).groupby("activity_name")["actor_id"].nunique()
        dropout = (started - completed).dropna().sort_values(ascending=False).head(10)
        dropout = dropout[dropout > 0]
        return {
            "value": int(dropout.sum()),
            "trend": None,
            "chart_data": {
                "labels": dropout.index.tolist(),
                "values": dropout.astype(int).tolist(),
            },
            "details": "Learners who started but didn't complete",
        }


    # ─────────────────────────────────────────────
    # ADVANCED TRAX-SPECIFIC INDICATORS
    # ─────────────────────────────────────────────

    def resume_rate(self) -> Dict:
        """Calculate resume rate: % of suspended activities that were resumed."""
        suspended = len(self._verb_df(self.df, self.VERB_SUSPENDED))
        resumed = len(self._verb_df(self.df, self.VERB_RESUMED))

        curr = round(resumed / max(suspended, 1) * 100, 1) if suspended > 0 else 0

        p_suspended = len(self._verb_df(self.prev_df, self.VERB_SUSPENDED))
        p_resumed = len(self._verb_df(self.prev_df, self.VERB_RESUMED))
        prev = round(p_resumed / max(p_suspended, 1) * 100, 1) if p_suspended > 0 else 0

        return {
            "value": curr,
            "trend": self._trend(curr, prev),
            "chart_data": {"suspended": suspended, "resumed": resumed},
            "details": f"{resumed} resumed out of {suspended} suspended",
        }

    def near_completion_abandonment(self) -> Dict:
        """% of suspensions that occurred at >90% progress."""
        suspended = self._verb_df(self.df, self.VERB_SUSPENDED)
        if suspended.empty or "progress_percent" not in suspended.columns:
            return {"value": 0, "trend": None, "chart_data": {}, "details": "No progress data"}

        suspended_with_progress = suspended.dropna(subset=["progress_percent"])
        if suspended_with_progress.empty:
            return {"value": 0, "trend": None, "chart_data": {}, "details": "No progress data"}

        high_progress = suspended_with_progress[suspended_with_progress["progress_percent"] > 90]
        curr = round(len(high_progress) / max(len(suspended_with_progress), 1) * 100, 1)

        return {
            "value": curr,
            "trend": None,
            "chart_data": {
                "high_progress": len(high_progress),
                "total_suspended": len(suspended_with_progress)
            },
            "details": f"{len(high_progress)} / {len(suspended_with_progress)} suspensions at >90% progress",
        }

    def avg_progress_at_suspension(self) -> Dict:
        """Average progress percentage when students suspend activities."""
        suspended = self._verb_df(self.df, self.VERB_SUSPENDED)
        if suspended.empty or "progress_percent" not in suspended.columns:
            return {"value": 0, "trend": None, "chart_data": {}, "details": "No progress data"}

        suspended_with_progress = suspended.dropna(subset=["progress_percent"])
        if suspended_with_progress.empty:
            return {"value": 0, "trend": None, "chart_data": {}, "details": "No progress data"}

        curr = round(suspended_with_progress["progress_percent"].mean(), 1)

        p_suspended = self._verb_df(self.prev_df, self.VERB_SUSPENDED)
        prev = 0
        if not p_suspended.empty and "progress_percent" in p_suspended.columns:
            p_with_prog = p_suspended.dropna(subset=["progress_percent"])
            if not p_with_prog.empty:
                prev = round(p_with_prog["progress_percent"].mean(), 1)

        return {
            "value": curr,
            "trend": self._trend(curr, prev),
            "chart_data": {"values": suspended_with_progress["progress_percent"].tolist()[:100]},
            "details": f"Avg {curr}% progress at suspension",
        }

    def content_engagement_by_type(self) -> Dict:
        """Average viewing time by content type (video, webpage, media)."""
        experienced = self._verb_df(self.df, self.VERB_EXPERIENCED)
        if experienced.empty or "content_type" not in experienced.columns:
            return {"value": 0, "trend": None, "chart_data": {}, "details": "No content type data"}

        by_type = experienced.groupby("content_type")["duration_min"].agg(['mean', 'count'])
        by_type = by_type[by_type['count'] >= 3]  # At least 3 views

        if by_type.empty:
            return {"value": 0, "trend": None, "chart_data": {}, "details": "Insufficient data"}

        by_type = by_type.sort_values('mean', ascending=False)

        return {
            "value": len(by_type),
            "trend": None,
            "chart_data": {
                "labels": by_type.index.tolist(),
                "values": by_type['mean'].round(1).tolist(),
                "counts": by_type['count'].tolist()
            },
            "details": f"{len(by_type)} content types with ≥3 views each",
        }

    def question_difficulty_ranking(self) -> Dict:
        """Identify hardest questions by success rate."""
        answered = self._verb_df(self.df, self.VERB_ANSWERED)
        if answered.empty or "success" not in answered.columns:
            return {"value": 0, "trend": None, "chart_data": {}, "details": "No question data"}

        difficulty = answered.groupby("activity_name").agg({
            'success': 'mean',
            'stmt_id': 'count'
        }).rename(columns={'stmt_id': 'count'})

        difficulty = difficulty[difficulty['count'] >= 3]  # Min 3 attempts

        if difficulty.empty:
            return {"value": 0, "trend": None, "chart_data": {}, "details": "Insufficient attempts per question"}

        difficulty = difficulty.sort_values('success')  # Hardest first
        avg_difficulty = round((1 - difficulty['success'].mean()) * 100, 1)

        top_10 = difficulty.head(10)

        return {
            "value": avg_difficulty,
            "trend": None,
            "chart_data": {
                "labels": top_10.index.tolist(),
                "values": (top_10['success'] * 100).round(1).tolist(),
                "counts": top_10['count'].tolist()
            },
            "details": f"{avg_difficulty}% avg difficulty across {len(difficulty)} questions",
        }

    def interaction_type_performance(self) -> Dict:
        """Success rate by interaction type (choice, sequencing, long-fill-in, etc)."""
        answered = self._verb_df(self.df, self.VERB_ANSWERED)
        if answered.empty or "interaction_type" not in answered.columns or "success" not in answered.columns:
            return {"value": 0, "trend": None, "chart_data": {}, "details": "No interaction type data"}

        by_type = answered.groupby("interaction_type").agg({
            'success': 'mean',
            'stmt_id': 'count'
        }).rename(columns={'stmt_id': 'count'})

        by_type = by_type[by_type['count'] >= 3]  # Min 3 attempts

        if by_type.empty:
            return {"value": 0, "trend": None, "chart_data": {}, "details": "Insufficient data"}

        by_type = by_type.sort_values('success', ascending=False)

        return {
            "value": len(by_type),
            "trend": None,
            "chart_data": {
                "labels": by_type.index.tolist(),
                "values": (by_type['success'] * 100).round(1).tolist(),
                "counts": by_type['count'].tolist()
            },
            "details": f"{len(by_type)} interaction types tracked",
        }

    def dropout_hotspots(self) -> Dict:
        """Most common ending points where students suspend activities."""
        suspended = self._verb_df(self.df, self.VERB_SUSPENDED)
        if suspended.empty or "ending_point" not in suspended.columns:
            return {"value": 0, "trend": None, "chart_data": {}, "details": "No ending point data"}

        hotspots = suspended["ending_point"].value_counts().head(10)

        if hotspots.empty:
            return {"value": 0, "trend": None, "chart_data": {}, "details": "No ending points recorded"}

        # Shorten activity IDs for display
        labels = [str(ep).split('/')[-1][:30] if ep else "Unknown" for ep in hotspots.index]

        return {
            "value": len(hotspots),
            "trend": None,
            "chart_data": {
                "labels": labels,
                "values": hotspots.values.tolist()
            },
            "details": f"Top {len(hotspots)} dropout points identified",
        }

    def launch_success_rate(self) -> Dict:
        """% of launched modules that were satisfied."""
        launched = len(self._verb_df(self.df, self.VERB_LAUNCHED))
        satisfied = len(self._verb_df(self.df, self.VERB_SATISFIED))

        curr = round(satisfied / max(launched, 1) * 100, 1) if launched > 0 else 0

        p_launched = len(self._verb_df(self.prev_df, self.VERB_LAUNCHED))
        p_satisfied = len(self._verb_df(self.prev_df, self.VERB_SATISFIED))
        prev = round(p_satisfied / max(p_launched, 1) * 100, 1) if p_launched > 0 else 0

        return {
            "value": curr,
            "trend": self._trend(curr, prev),
            "chart_data": {"launched": launched, "satisfied": satisfied},
            "details": f"{satisfied} / {launched} launched modules satisfied",
        }

    # ─────────────────────────────────────────────
    # STUDENT-LEVEL ANALYTICS
    # ─────────────────────────────────────────────

    def get_student_list(self) -> List[Dict]:
        """Return list of all students with basic stats."""
        if self.df.empty:
            return []

        students = []
        for actor_id in self.df["actor_id"].unique():
            actor_df = self.df[self.df["actor_id"] == actor_id]
            actor_name = actor_df["actor_name"].iloc[0] if "actor_name" in actor_df.columns else actor_id

            students.append({
                "actor_id": actor_id,
                "actor_name": actor_name,
                "total_statements": len(actor_df),
                "first_activity": actor_df["timestamp"].min() if "timestamp" in actor_df.columns else None,
                "last_activity": actor_df["timestamp"].max() if "timestamp" in actor_df.columns else None,
                "unique_activities": actor_df["activity_id"].nunique() if not actor_df.empty else 0,
            })

        return sorted(students, key=lambda x: x["total_statements"], reverse=True)

    def get_student_sessions(self, actor_id: str) -> List[Dict]:
        """
        Get all sessions for a specific student based on parent ID.
        Sessions are identified by context.contextActivities.parent (group_session_id).
        """
        actor_df = self.df[self.df["actor_id"] == actor_id] if not self.df.empty else pd.DataFrame()
        if actor_df.empty:
            return []

        actor_df = actor_df.sort_values("timestamp")

        # Use group_session_id from context.contextActivities.parent
        if "group_session_id" not in actor_df.columns or actor_df["group_session_id"].isna().all():
            logger.warning(f"No parent session IDs found for actor {actor_id}")
            return []

        sessions = []
        session_num = 1

        for group_session_id, group_df in actor_df.groupby("group_session_id", dropna=True):
            if not group_session_id or group_session_id == "" or pd.isna(group_session_id):
                continue

            group_df = group_df.sort_values("timestamp")
            start_ts = group_df["timestamp"].min()
            end_ts = group_df["timestamp"].max()
            duration = (end_ts - start_ts).total_seconds() / 60 if start_ts and end_ts else 0

            # Detect session type from ID
            session_type = "Session"
            if "groupsessions" in group_session_id.lower():
                session_type = "Group Session"
            elif "course" in group_session_id.lower() and "session" in group_session_id.lower():
                session_type = "Course Session"

            # Get unique attempts within this session
            attempts = []
            if "attempt_id" in group_df.columns:
                attempts = [a for a in group_df["attempt_id"].dropna().unique() if a]

            # Get unique modules
            modules = []
            if "module_id" in group_df.columns:
                modules = [m for m in group_df["module_id"].dropna().unique() if m]

            # Get session name from first statement's parent module
            session_name = None
            if modules:
                # Try to get module name from activity_name where activity_id matches module
                for module_id in modules:
                    match = group_df[group_df["module_id"] == module_id]
                    if not match.empty and "activity_name" in match.columns:
                        session_name = match["activity_name"].iloc[0]
                        break

            if not session_name:
                # Extract a human-readable part from the session ID
                session_id_short = group_session_id.split('/')[-1][:12]
                session_name = f"{session_type} {session_id_short}"

            sessions.append({
                "session_id": group_session_id,
                "session_number": session_num,
                "session_name": session_name,
                "session_type": session_type,
                "start": start_ts,
                "end": end_ts,
                "duration_min": duration,
                "statement_count": len(group_df),
                "activities": list(group_df["activity_id"].dropna().unique()),
                "verbs": list(group_df["verb_display"].dropna().unique()),
                "attempts": attempts,
                "modules": modules,
                "registration": group_df["registration"].iloc[0] if "registration" in group_df.columns and not group_df["registration"].isna().all() else None,
            })
            session_num += 1

        # Sort by start time
        sessions = sorted(sessions, key=lambda s: s["start"] if s["start"] else datetime.min.replace(tzinfo=timezone.utc))
        # Renumber after sorting
        for i, s in enumerate(sessions, 1):
            s["session_number"] = i

        return sessions

    def compute_student_indicators(self, actor_id: str, session_id: Optional[str] = None) -> Dict[str, Dict]:
        """
        Compute indicators for a specific student.
        If session_id is provided, compute only for that session.
        Otherwise, compute across all student's data.

        session_id can be:
        - Real group_session_id URL (e.g., https://xapi.maskott.com/identifiers/groupsessions/...)
        - Time-based session_id (e.g., "time-gap-session-1")
        """
        # Filter to student
        student_df = self.df[self.df["actor_id"] == actor_id] if not self.df.empty else pd.DataFrame()

        if student_df.empty:
            return {}

        # Further filter to session if specified
        if session_id:
            # Check if using real group_session_id
            if "group_session_id" in student_df.columns and not student_df["group_session_id"].isna().all():
                # Filter by group_session_id
                student_df = student_df[student_df["group_session_id"] == session_id]

            # If no matches or time-based session, fall back to time window lookup
            if student_df.empty or session_id.startswith("time-gap-session"):
                # Need to use timestamp range from the session
                sessions = self.get_student_sessions(actor_id)
                target_session = next((s for s in sessions if s["session_id"] == session_id), None)
                if target_session:
                    session_start = target_session["start"]
                    session_end = target_session["end"]
                    # Re-filter from original student_df
                    student_df = self.df[self.df["actor_id"] == actor_id]
                    student_df = student_df[
                        (student_df["timestamp"] >= session_start) &
                        (student_df["timestamp"] <= session_end)
                    ]

        # Create a temporary calculator with just this student's data
        student_stmts = [self.stmts[i] for i in student_df.index if i < len(self.stmts)]
        temp_calc = IndicatorCalculator(student_stmts, [])

        # Compute key indicators
        results = {
            "total_statements": {"value": len(student_df), "trend": None, "chart_data": {}, "details": "Total statements"},
            "unique_activities": {"value": student_df["activity_id"].nunique(), "trend": None, "chart_data": {}, "details": "Unique activities accessed"},
        }

        # Time spent - keep as numeric for calculations, formatted in display
        total_minutes = round(student_df["duration_min"].sum(), 1) if "duration_min" in student_df.columns and not student_df["duration_min"].isna().all() else 0
        results["time_spent"] = {
            "value": total_minutes,
            "trend": None,
            "chart_data": {},
            "details": "Total time spent"
        }

        # Score-based indicators (if applicable)
        scored = self._safe_dropna(student_df, "score")
        if not scored.empty:
            results["avg_score"] = {
                "value": round(scored["score"].mean(), 1),
                "trend": None,
                "chart_data": {"values": scored["score"].tolist()},
                "details": f"Average score across {len(scored)} scored activities"
            }
            results["min_score"] = {"value": round(scored["score"].min(), 1), "trend": None, "chart_data": {}, "details": "Lowest score"}
            results["max_score"] = {"value": round(scored["score"].max(), 1), "trend": None, "chart_data": {}, "details": "Highest score"}

        # Completion indicators
        completed = student_df[student_df["verb_id"] == self.VERB_COMPLETED]
        attempted = student_df[student_df["verb_id"].isin([self.VERB_INITIALIZED, self.VERB_COMPLETED])]
        results["completions"] = {"value": len(completed), "trend": None, "chart_data": {}, "details": "Completed activities"}
        results["completion_rate"] = {
            "value": round(len(completed) / max(len(attempted), 1) * 100, 1),
            "trend": None,
            "chart_data": {},
            "details": f"{len(completed)} / {len(attempted)} attempted"
        }

        # Pass/Fail
        passed = student_df[student_df["verb_id"] == self.VERB_PASSED]
        failed = student_df[student_df["verb_id"] == self.VERB_FAILED]
        total_pf = len(passed) + len(failed)
        results["passed"] = {"value": len(passed), "trend": None, "chart_data": {}, "details": "Passed assessments"}
        results["failed"] = {"value": len(failed), "trend": None, "chart_data": {}, "details": "Failed assessments"}
        results["pass_rate"] = {
            "value": round(len(passed) / max(total_pf, 1) * 100, 1) if total_pf > 0 else 0,
            "trend": None,
            "chart_data": {},
            "details": f"{len(passed)} / {total_pf} passed"
        }

        # Verb distribution
        verb_counts = student_df.groupby("verb_display").size().sort_values(ascending=False)
        results["verb_distribution"] = {
            "value": len(verb_counts),
            "trend": None,
            "chart_data": {"labels": verb_counts.index.tolist(), "values": verb_counts.values.tolist()},
            "details": f"{len(verb_counts)} distinct action types"
        }

        # Activity breakdown
        activity_counts = student_df.groupby("activity_name").size().sort_values(ascending=False).head(10)
        results["top_activities"] = {
            "value": len(activity_counts),
            "trend": None,
            "chart_data": {"labels": activity_counts.index.tolist(), "values": activity_counts.values.tolist()},
            "details": "Most accessed activities"
        }

        return results

    # ==================== NEW ADVANCED TRAX INDICATORS ====================

    def resume_rate(self) -> Dict:
        """Calculate resume rate (resumed / suspended)."""
        suspended = len(self._verb_df(self.df, self.VERB_SUSPENDED))
        resumed = len(self._verb_df(self.df, self.VERB_RESUMED))
        if suspended == 0:
            return {"value": 0, "trend": None, "chart_data": {}, "details": "No suspensions"}

        rate = round(resumed / suspended * 100, 1)
        p_suspended = len(self._verb_df(self.prev_df, self.VERB_SUSPENDED)) if not self.prev_df.empty else 0
        p_resumed = len(self._verb_df(self.prev_df, self.VERB_RESUMED)) if not self.prev_df.empty else 0
        prev_rate = round(p_resumed / max(p_suspended, 1) * 100, 1) if p_suspended > 0 else 0

        return {
            "value": rate,
            "trend": self._trend(rate, prev_rate),
            "chart_data": {"suspended": suspended, "resumed": resumed},
            "details": f"{resumed} resumed out of {suspended} suspended"
        }

    def near_completion_abandonment(self) -> Dict:
        """% of suspensions with progress > 90%."""
        suspended = self._verb_df(self.df, self.VERB_SUSPENDED)
        if suspended.empty or "progress_percent" not in suspended.columns:
            return {"value": 0, "trend": None, "chart_data": {}, "details": "No suspension data"}

        suspended_with_progress = suspended[suspended["progress_percent"].notna()]
        if suspended_with_progress.empty:
            return {"value": 0, "trend": None, "chart_data": {}, "details": "No progress tracking"}

        # Ensure progress_percent is numeric
        suspended_with_progress['progress_percent'] = pd.to_numeric(suspended_with_progress['progress_percent'], errors='coerce')
        suspended_with_progress = suspended_with_progress[suspended_with_progress['progress_percent'].notna()]

        if suspended_with_progress.empty:
            return {"value": 0, "trend": None, "chart_data": {}, "details": "No valid progress data"}

        high_progress = suspended_with_progress[suspended_with_progress["progress_percent"] > 90]
        rate = round(len(high_progress) / len(suspended_with_progress) * 100, 1)

        return {
            "value": rate,
            "trend": None,
            "chart_data": {},
            "details": f"{len(high_progress)} / {len(suspended_with_progress)} suspensions at >90% progress"
        }

    def avg_progress_at_suspension(self) -> Dict:
        """Average progress percentage when students suspend."""
        suspended = self._verb_df(self.df, self.VERB_SUSPENDED)
        if suspended.empty or "progress_percent" not in suspended.columns:
            return {"value": 0, "trend": None, "chart_data": {}, "details": "No data"}

        suspended_with_progress = suspended[suspended["progress_percent"].notna()]
        if suspended_with_progress.empty:
            return {"value": 0, "trend": None, "chart_data": {}, "details": "No progress data"}

        # Ensure progress_percent is numeric
        suspended_with_progress['progress_percent'] = pd.to_numeric(suspended_with_progress['progress_percent'], errors='coerce')
        suspended_with_progress = suspended_with_progress[suspended_with_progress['progress_percent'].notna()]

        if suspended_with_progress.empty:
            return {"value": 0, "trend": None, "chart_data": {}, "details": "No valid progress data"}

        avg_progress = round(float(suspended_with_progress["progress_percent"].mean()), 1)

        # Create progress distribution bins
        try:
            bins = [0, 25, 50, 75, 90, 100]
            labels = ["0-25%", "25-50%", "50-75%", "75-90%", "90-100%"]
            counts = pd.cut(suspended_with_progress["progress_percent"], bins=bins, labels=labels, include_lowest=True).value_counts()

            chart_data = {
                "labels": [str(x) for x in counts.index.tolist()],
                "values": [int(x) for x in counts.values.tolist()]
            }
        except Exception:
            # If binning fails, just return empty chart
            chart_data = {"labels": [], "values": []}

        return {
            "value": avg_progress,
            "trend": None,
            "chart_data": chart_data,
            "details": f"Average: {avg_progress}% across {len(suspended_with_progress)} suspensions"
        }

    def content_engagement_by_type(self) -> Dict:
        """Average viewing time by content type (video, webpage, media)."""
        experienced = self._verb_df(self.df, self.VERB_EXPERIENCED)
        if experienced.empty or "content_type" not in experienced.columns:
            return {"value": 0, "trend": None, "chart_data": {}, "details": "No content data"}

        experienced = experienced[experienced["content_type"].notna()].copy()
        if experienced.empty:
            return {"value": 0, "trend": None, "chart_data": {}, "details": "No content types"}

        # Ensure duration is numeric
        experienced['duration_min'] = pd.to_numeric(experienced['duration_min'], errors='coerce')
        experienced = experienced[experienced['duration_min'].notna()]

        if experienced.empty:
            return {"value": 0, "trend": None, "chart_data": {}, "details": "No valid duration data"}

        by_type = experienced.groupby("content_type")["duration_min"].mean().sort_values(ascending=False)
        total_types = len(by_type)

        return {
            "value": total_types,
            "trend": None,
            "chart_data": {
                "labels": by_type.index.tolist(),
                "values": [round(float(x), 1) for x in by_type.values]
            },
            "details": f"{total_types} content types tracked"
        }

    def question_difficulty_ranking(self) -> Dict:
        """Identify hardest questions by success rate."""
        answered = self._verb_df(self.df, self.VERB_ANSWERED)
        if answered.empty or "success" not in answered.columns:
            return {"value": 0, "trend": None, "chart_data": {}, "details": "No question data"}

        # Ensure success is boolean/numeric
        answered = answered.copy()
        answered['success'] = answered['success'].astype(float)

        difficulty = answered.groupby("activity_name").agg({'success': ['mean', 'count']})
        difficulty.columns = ['success_rate', 'attempts']
        difficulty = difficulty[difficulty['attempts'] >= 3]
        if difficulty.empty:
            return {"value": 0, "trend": None, "chart_data": {}, "details": "Not enough attempts per question"}

        difficulty = difficulty.sort_values('success_rate')
        difficulty['difficulty'] = (1 - difficulty['success_rate'].astype(float)) * 100
        avg_difficulty = round(float(difficulty['difficulty'].mean()), 1)
        top_hardest = difficulty.head(10)

        return {
            "value": avg_difficulty,
            "trend": None,
            "chart_data": {
                "labels": top_hardest.index.tolist(),
                "values": [round(float(x), 1) for x in top_hardest['difficulty'].values]
            },
            "details": f"Top 10 hardest questions (min 3 attempts each)"
        }

    def interaction_type_performance(self) -> Dict:
        """Success rate by interaction type."""
        answered = self._verb_df(self.df, self.VERB_ANSWERED)
        if answered.empty or "interaction_type" not in answered.columns:
            return {"value": 0, "trend": None, "chart_data": {}, "details": "No interaction data"}

        answered = answered[answered["interaction_type"].notna()].copy()
        if answered.empty:
            return {"value": 0, "trend": None, "chart_data": {}, "details": "No interaction types"}

        # Ensure success is numeric
        answered['success'] = answered['success'].astype(float)

        by_type = answered.groupby("interaction_type")["success"].mean() * 100
        by_type = by_type.sort_values(ascending=False)
        overall_avg = round(float(answered["success"].mean() * 100), 1)

        return {
            "value": overall_avg,
            "trend": None,
            "chart_data": {
                "labels": by_type.index.tolist(),
                "values": [round(float(x), 1) for x in by_type.values]
            },
            "details": f"{len(by_type)} interaction types"
        }

    def dropout_hotspots(self) -> Dict:
        """Identify activity items where students most commonly suspend."""
        suspended = self._verb_df(self.df, self.VERB_SUSPENDED)
        if suspended.empty or "ending_point" not in suspended.columns:
            return {"value": 0, "trend": None, "chart_data": {}, "details": "No dropout data"}

        suspended = suspended[suspended["ending_point"].notna()]
        if suspended.empty:
            return {"value": 0, "trend": None, "chart_data": {}, "details": "No ending points tracked"}

        suspended["endpoint_short"] = suspended["ending_point"].apply(
            lambda x: x.split('/')[-1][:20] if isinstance(x, str) else "unknown"
        )
        hotspots = suspended["endpoint_short"].value_counts().head(10)

        return {
            "value": len(hotspots),
            "trend": None,
            "chart_data": {"labels": hotspots.index.tolist(), "values": hotspots.tolist()},
            "details": f"Top 10 dropout points"
        }

    def launch_success_rate(self) -> Dict:
        """Success rate from launched to satisfied."""
        launched = len(self._verb_df(self.df, self.VERB_LAUNCHED))
        satisfied = len(self._verb_df(self.df, self.VERB_SATISFIED))

        if launched == 0:
            return {"value": 0, "trend": None, "chart_data": {}, "details": "No launches"}

        rate = round(satisfied / launched * 100, 1)
        p_launched = len(self._verb_df(self.prev_df, self.VERB_LAUNCHED)) if not self.prev_df.empty else 0
        p_satisfied = len(self._verb_df(self.prev_df, self.VERB_SATISFIED)) if not self.prev_df.empty else 0
        prev_rate = round(p_satisfied / max(p_launched, 1) * 100, 1) if p_launched > 0 else 0

        return {
            "value": rate,
            "trend": self._trend(rate, prev_rate),
            "chart_data": {"launched": launched, "satisfied": satisfied},
            "details": f"{satisfied} satisfied out of {launched} launched"
        }

    def compute_all(self) -> Dict[str, Dict]:
        """Run all indicators and return a flat dict keyed by indicator name."""
        return {
            # Engagement
            "active_learners": self.active_learners(),
            "total_statements": self.total_statements(),
            "sessions": self.sessions(),
            "avg_session_duration": self.avg_session_duration(),
            "statements_per_learner": self.statements_per_learner(),
            # Completion
            "completion_rate": self.completion_rate(),
            "completions_total": self.completions_total(),
            "dropout_rate": self.dropout_rate(),
            "progress_avg": self.progress_avg(),
            # Performance
            "avg_score": self.avg_score(),
            "pass_rate": self.pass_rate(),
            "fail_rate": self.fail_rate(),
            "score_distribution": self.score_distribution(),
            "mastery_rate": self.mastery_rate(),
            "median_score": self.median_score(),
            "attempts_per_pass": self.attempts_per_pass(),
            # Interaction
            "interactions_total": self.interactions_total(),
            "response_accuracy": self.response_accuracy(),
            "verb_distribution": self.verb_distribution(),
            "most_active_hours": self.most_active_hours(),
            "resumed_rate": self.resumed_rate(),
            # Social
            "shared_count": self.shared_count(),
            # Retention
            "retention_d1": self.retention(1),
            "retention_d7": self.retention(7),
            "retention_d30": self.retention(30),
            "churn_rate": self.churn_rate(),
            # Content
            "top_activities": self.top_activities(),
            "hardest_activities": self.hardest_activities(),
            "avg_time_per_activity": self.avg_time_per_activity(),
            "content_dropout_map": self.content_dropout_map(),
            # Advanced TRAX-specific indicators
            "resume_rate": self.resume_rate(),
            "near_completion_abandonment": self.near_completion_abandonment(),
            "avg_progress_at_suspension": self.avg_progress_at_suspension(),
            "content_engagement_by_type": self.content_engagement_by_type(),
            "question_difficulty_ranking": self.question_difficulty_ranking(),
            "interaction_type_performance": self.interaction_type_performance(),
            "dropout_hotspots": self.dropout_hotspots(),
            "launch_success_rate": self.launch_success_rate(),
        }

    # ─────────────────────────────────────────────
    # SESSION-LEVEL ANALYTICS & COMPARISONS
    # ─────────────────────────────────────────────

    def get_session_averages(self, session_id: str) -> Dict[str, float]:
        """
        Calculate average indicators for all students in a session.
        Returns a dict of indicator_name -> average_value.
        """
        if self.df.empty or "group_session_id" not in self.df.columns:
            return {}

        session_df = self.df[self.df["group_session_id"] == session_id]
        if session_df.empty:
            return {}

        averages = {}

        # Group by actor to get per-student metrics, then average across students
        actors = session_df["actor_id"].unique()

        # Engagement metrics
        avg_statements = session_df.groupby("actor_id").size().mean()
        averages["statements_per_learner"] = round(avg_statements, 1)

        # # Time metrics
        # if "duration_min" in session_df.columns:
        #     avg_duration = session_df.groupby("actor_id")["duration_min"].sum().mean()
        #     averages["avg_time_spent"] = round(avg_duration, 1)
        # Time metrics - session duration (first to last statement per student)
        if "timestamp" in session_df.columns:
            student_durations = []
            for actor in actors:
                actor_session_df = session_df[session_df["actor_id"] == actor]
                if len(actor_session_df) > 0:
                    actor_session_df = actor_session_df.sort_values("timestamp")
                    first_ts = actor_session_df["timestamp"].min()
                    last_ts = actor_session_df["timestamp"].max()
                    if pd.notna(first_ts) and pd.notna(last_ts):
                        duration_min = (last_ts - first_ts).total_seconds() / 60
                        student_durations.append(duration_min)

            if student_durations:
                averages["avg_session_duration"] = round(mean(student_durations), 1)

        # Performance metrics
        scored = session_df[session_df["score"].notna()]
        if not scored.empty:
            averages["avg_score"] = round(scored["score"].mean(), 1)
            averages["median_score"] = round(scored["score"].median(), 1)

        # Completion metrics
        completed = session_df[session_df["verb_id"] == self.VERB_COMPLETED]
        attempted = session_df[session_df["verb_id"].isin([self.VERB_INITIALIZED, self.VERB_COMPLETED])]
        if len(attempted) > 0:
            completion_rates = []
            for actor in actors:
                actor_completed = len(completed[completed["actor_id"] == actor])
                actor_attempted = len(attempted[attempted["actor_id"] == actor])
                if actor_attempted > 0:
                    completion_rates.append(actor_completed / actor_attempted * 100)
            if completion_rates:
                averages["avg_completion_rate"] = round(mean(completion_rates), 1)

        # Pass rate
        passed = session_df[session_df["verb_id"] == self.VERB_PASSED]
        failed = session_df[session_df["verb_id"] == self.VERB_FAILED]
        if len(passed) + len(failed) > 0:
            pass_rates = []
            for actor in actors:
                actor_passed = len(passed[passed["actor_id"] == actor])
                actor_failed = len(failed[failed["actor_id"] == actor])
                total = actor_passed + actor_failed
                if total > 0:
                    pass_rates.append(actor_passed / total * 100)
            if pass_rates:
                averages["avg_pass_rate"] = round(mean(pass_rates), 1)

        # Activity diversity
        avg_activities = session_df.groupby("actor_id")["activity_id"].nunique().mean()
        averages["avg_activities_accessed"] = round(avg_activities, 1)

        return averages

    def compare_student_to_session(self, actor_id: str, session_id: str) -> Dict[str, Dict]:
        """
        Compare a student's performance to session averages.
        Returns dict with student value, session average, and delta.
        """
        if self.df.empty or "group_session_id" not in self.df.columns:
            return {}

        # Get student data for this session
        student_session_df = self.df[
            (self.df["actor_id"] == actor_id) &
            (self.df["group_session_id"] == session_id)
        ]

        if student_session_df.empty:
            return {}

        # Get session averages
        session_avg = self.get_session_averages(session_id)

        comparison = {}

        # Statements
        student_stmts = len(student_session_df)
        if "statements_per_learner" in session_avg:
            comparison["statements"] = {
                "student": student_stmts,
                "session_avg": session_avg["statements_per_learner"],
                "delta": round(student_stmts - session_avg["statements_per_learner"], 1),
                "delta_pct": round((student_stmts / session_avg["statements_per_learner"] - 1) * 100, 1) if session_avg["statements_per_learner"] > 0 else 0
            }

        # if "duration_min" in student_session_df.columns and "avg_time_spent" in session_avg:
        if "timestamp" in student_session_df.columns and "avg_session_duration" in session_avg:
            student_session_df_sorted = student_session_df.sort_values("timestamp")
            first_ts = student_session_df_sorted["timestamp"].min()
            last_ts = student_session_df_sorted["timestamp"].max()

            if pd.notna(first_ts) and pd.notna(last_ts):
                student_duration = (last_ts - first_ts).total_seconds() / 60
                comparison["session_duration"] = {
                    "student": round(student_duration, 1),
                    "session_avg": session_avg["avg_session_duration"],
                    "delta": round(student_duration - session_avg["avg_session_duration"], 1),
                    "delta_pct": round((student_duration / session_avg["avg_session_duration"] - 1) * 100, 1) if session_avg["avg_session_duration"] > 0 else 0
                }

        # Score
        scored = student_session_df[student_session_df["score"].notna()]
        if not scored.empty and "avg_score" in session_avg:
            student_score = scored["score"].mean()
            comparison["score"] = {
                "student": round(student_score, 1),
                "session_avg": session_avg["avg_score"],
                "delta": round(student_score - session_avg["avg_score"], 1),
                "delta_pct": round((student_score / session_avg["avg_score"] - 1) * 100, 1) if session_avg["avg_score"] > 0 else 0
            }

        # Completion rate
        # # Pass rate
        return comparison

    # ─────────────────────────────────────────────
    # BEHAVIOR PATTERN ANALYSIS
    # ─────────────────────────────────────────────

    def analyze_action_sequences(self, actor_id: Optional[str] = None, min_sequence_length: int = 3) -> List[Dict]:
        """
        Analyze sequences of verbs (behavior patterns) and correlate with outcomes.
        Returns common action paths and their success rates.
        """
        df = self.df if actor_id is None else self.df[self.df["actor_id"] == actor_id]
        if df.empty or "verb_display" not in df.columns:
            return []

        sequences = []

        # Group by session and actor to get sequences
        if "group_session_id" in df.columns:
            groups = df.groupby(["actor_id", "group_session_id"])
        else:
            groups = df.groupby("actor_id")

        for name, group in groups:
            if len(group) < min_sequence_length:
                continue

            group = group.sort_values("timestamp")
            verbs = group["verb_display"].tolist()

            # Extract sequences of min_sequence_length
            for i in range(len(verbs) - min_sequence_length + 1):
                sequence = tuple(verbs[i:i + min_sequence_length])

                # Check if this sequence led to success
                # Look ahead to see if there's a "passed" or "completed" within next 5 actions
                success = False
                for j in range(i + min_sequence_length, min(i + min_sequence_length + 5, len(verbs))):
                    if verbs[j] in ["passed", "completed", "Passed", "Completed"]:
                        success = True
                        break

                sequences.append({
                    "sequence": " → ".join(sequence),
                    "length": min_sequence_length,
                    "success": success
                })

        # Aggregate sequences
        from collections import Counter
        sequence_counter = Counter(s["sequence"] for s in sequences)
        sequence_success = {}

        for seq in sequences:
            seq_key = seq["sequence"]
            if seq_key not in sequence_success:
                sequence_success[seq_key] = {"total": 0, "successes": 0}
            sequence_success[seq_key]["total"] += 1
            if seq["success"]:
                sequence_success[seq_key]["successes"] += 1

        # Build results
        results = []
        for seq_key, count in sequence_counter.most_common(20):
            stats = sequence_success[seq_key]
            success_rate = (stats["successes"] / stats["total"] * 100) if stats["total"] > 0 else 0
            results.append({
                "sequence": seq_key,
                "occurrences": count,
                "success_rate": round(success_rate, 1),
                "successes": stats["successes"]
            })

        return results

    def analyze_time_per_activity(self, actor_id: Optional[str] = None) -> List[Dict]:
        """
        Analyze time spent per activity/question.
        Returns activities sorted by average time spent.
        """
        df = self.df if actor_id is None else self.df[self.df["actor_id"] == actor_id]
        if df.empty or "duration_min" not in df.columns or "activity_name" not in df.columns:
            return []

        # Filter to activities with duration data
        timed = df[df["duration_min"].notna() & (df["duration_min"] > 0)]
        if timed.empty:
            return []

        # Group by activity
        activity_stats = timed.groupby("activity_name").agg({
            "duration_min": ["mean", "median", "std", "count"],
            "score": "mean"
        }).reset_index()

        activity_stats.columns = ["activity", "avg_time", "median_time", "std_time", "attempts", "avg_score"]

        # Sort by average time
        activity_stats = activity_stats.sort_values("avg_time", ascending=False)

        results = []
        for _, row in activity_stats.head(20).iterrows():
            results.append({
                "activity": row["activity"],
                "avg_time_min": round(row["avg_time"], 1),
                "median_time_min": round(row["median_time"], 1),
                "std_time_min": round(row["std_time"], 1) if pd.notna(row["std_time"]) else 0,
                "attempts": int(row["attempts"]),
                "avg_score": round(row["avg_score"], 1) if pd.notna(row["avg_score"]) else None
            })

        return results

    # ─────────────────────────────────────────────
    # CORRELATION ANALYSIS
    # ─────────────────────────────────────────────

    def engagement_performance_correlation(self) -> Dict:
        """
        Calculate correlation between engagement metrics and performance.
        Returns correlation coefficients and scatter plot data.
        """
        if self.df.empty:
            return {"correlation": 0, "data": []}

        # Build per-student metrics
        student_metrics = []

        for actor in self.df["actor_id"].unique():
            actor_df = self.df[self.df["actor_id"] == actor]

            # Engagement: statement count
            engagement = len(actor_df)

            # Performance: average score
            scored = actor_df[actor_df["score"].notna()]
            if scored.empty:
                continue

            performance = scored["score"].mean()

            student_metrics.append({
                "engagement": engagement,
                "performance": performance,
                "actor_id": actor
            })

        if len(student_metrics) < 2:
            return {"correlation": 0, "data": []}

        # Calculate correlation
        engagements = [m["engagement"] for m in student_metrics]
        performances = [m["performance"] for m in student_metrics]

        try:
            correlation = np.corrcoef(engagements, performances)[0, 1]
        except:
            correlation = 0

        return {
            "correlation": round(correlation, 3),
            "significance": "strong" if abs(correlation) > 0.7 else "moderate" if abs(correlation) > 0.4 else "weak",
            "data": student_metrics[:50],  # Limit for chart
            "interpretation": self._interpret_correlation(correlation)
        }

    def _interpret_correlation(self, r: float) -> str:
        """Interpret correlation coefficient."""
        if r > 0.7:
            return "Strong positive correlation: More engagement strongly predicts better performance"
        elif r > 0.4:
            return "Moderate positive correlation: More engagement tends to improve performance"
        elif r > 0.1:
            return "Weak positive correlation: Slight relationship between engagement and performance"
        elif r > -0.1:
            return "No correlation: Engagement and performance are independent"
        elif r > -0.4:
            return "Weak negative correlation: Unexpected inverse relationship"
        else:
            return "Negative correlation: May indicate struggling students attempting more"

    # ─────────────────────────────────────────────
    # MINIMUM BASELINE ANALYSIS
    # ─────────────────────────────────────────────

    def find_minimum_baseline_students(self, success_threshold: float = 70.0) -> Dict:
        """
        Find students who succeeded with minimum effort (minimum baseline).
        Success = pass_rate > threshold OR avg_score > threshold.
        Returns the minimum successful profile and comparison stats.
        """
        if self.df.empty:
            return {}

        successful_students = []

        for actor in self.df["actor_id"].unique():
            actor_df = self.df[self.df["actor_id"] == actor]

            # Check success criteria
            passed = actor_df[actor_df["verb_id"] == self.VERB_PASSED]
            failed = actor_df[actor_df["verb_id"] == self.VERB_FAILED]
            total_pf = len(passed) + len(failed)

            pass_rate = (len(passed) / total_pf * 100) if total_pf > 0 else 0

            scored = actor_df[actor_df["score"].notna()]
            avg_score = scored["score"].mean() if not scored.empty else 0

            # Is this student successful?
            is_successful = pass_rate >= success_threshold or avg_score >= success_threshold

            if is_successful:
                successful_students.append({
                    "actor_id": actor,
                    "statements": len(actor_df),
                    "time_spent": actor_df["duration_min"].sum() if "duration_min" in actor_df.columns else 0,
                    "pass_rate": pass_rate,
                    "avg_score": avg_score,
                    "activities": actor_df["activity_id"].nunique()
                })

        if not successful_students:
            return {"baseline": None, "students": []}

        # Find minimum effort student (fewest statements)
        baseline_student = min(successful_students, key=lambda x: x["statements"])

        # Calculate average of all successful students
        avg_statements = mean([s["statements"] for s in successful_students])
        avg_time = mean([s["time_spent"] for s in successful_students])
        avg_activities = mean([s["activities"] for s in successful_students])

        return {
            "baseline": {
                "actor_id": baseline_student["actor_id"],
                "statements": baseline_student["statements"],
                "time_spent": round(baseline_student["time_spent"], 1),
                "pass_rate": round(baseline_student["pass_rate"], 1),
                "avg_score": round(baseline_student["avg_score"], 1),
                "activities": baseline_student["activities"],
                "efficiency_score": round(baseline_student["avg_score"] / max(baseline_student["statements"], 1), 2)
            },
            "successful_avg": {
                "statements": round(avg_statements, 1),
                "time_spent": round(avg_time, 1),
                "activities": round(avg_activities, 1)
            },
            "total_successful": len(successful_students),
            "baseline_percentile": round((1 / len(successful_students)) * 100, 1)
        }

    def compare_to_minimum_baseline(self, actor_id: str) -> Dict:
        """
        Compare a student to the minimum baseline (most efficient successful student).
        Shows if student is over-working or under-performing.
        """
        baseline_data = self.find_minimum_baseline_students()
        if not baseline_data or "baseline" not in baseline_data:
            return {}

        baseline = baseline_data["baseline"]

        # Get student data
        student_df = self.df[self.df["actor_id"] == actor_id]
        if student_df.empty:
            return {}

        # Calculate student metrics
        passed = student_df[student_df["verb_id"] == self.VERB_PASSED]
        failed = student_df[student_df["verb_id"] == self.VERB_FAILED]
        total_pf = len(passed) + len(failed)
        pass_rate = (len(passed) / total_pf * 100) if total_pf > 0 else 0

        scored = student_df[student_df["score"].notna()]
        avg_score = scored["score"].mean() if not scored.empty else 0

        student_statements = len(student_df)
        student_time = student_df["duration_min"].sum() if "duration_min" in student_df.columns else 0
        student_activities = student_df["activity_id"].nunique()

        # Compare
        comparison = {
            "student": {
                "statements": student_statements,
                "time_spent": round(student_time, 1),
                "pass_rate": round(pass_rate, 1),
                "avg_score": round(avg_score, 1),
                "activities": student_activities
            },
            "baseline": baseline,
            "delta": {
                "statements": student_statements - baseline["statements"],
                "statements_pct": round((student_statements / baseline["statements"] - 1) * 100, 1) if baseline["statements"] > 0 else 0,
                "time_spent": round(student_time - baseline["time_spent"], 1),
                "performance_gap": round(avg_score - baseline["avg_score"], 1)
            },
            "assessment": self._assess_efficiency(
                student_statements, baseline["statements"],
                avg_score, baseline["avg_score"]
            )
        }

        return comparison

    def _assess_efficiency(self, student_stmts: int, baseline_stmts: int,
                          student_score: float, baseline_score: float) -> str:
        """Assess student efficiency vs baseline."""
        effort_ratio = student_stmts / baseline_stmts if baseline_stmts > 0 else 1
        score_diff = student_score - baseline_score

        if effort_ratio < 1.2 and score_diff >= 0:
            return "Efficient: Matching or exceeding baseline with similar effort"
        elif effort_ratio < 1.2 and score_diff < -10:
            return "Struggling: Similar effort but underperforming"
        elif effort_ratio > 1.5 and score_diff >= 0:
            return "Over-working: Good results but excessive effort"
        elif effort_ratio > 1.5 and score_diff < -10:
            return "Critical: High effort but poor results - needs intervention"
        else:
            return "Normal: Within expected range"

    def compute_all_indicators(self) -> Dict[str, Dict]:
        """Compute all available indicators and return as a dict."""
        return {
            # Engagement
            "active_learners": self.active_learners(),
            "total_statements": self.total_statements(),
            "sessions": self.sessions(),
            "avg_session_duration": self.avg_session_duration(),
            "statements_per_learner": self.statements_per_learner(),
            # Completion
            "completion_rate": self.completion_rate(),
            "completions_total": self.completions_total(),
            "dropout_rate": self.dropout_rate(),
            "progress_avg": self.progress_avg(),
            # Performance
            "avg_score": self.avg_score(),
            "pass_rate": self.pass_rate(),
            "fail_rate": self.fail_rate(),
            "score_distribution": self.score_distribution(),
            "mastery_rate": self.mastery_rate(),
            "median_score": self.median_score(),
            "attempts_per_pass": self.attempts_per_pass(),
            # Interaction
            "interactions_total": self.interactions_total(),
            "response_accuracy": self.response_accuracy(),
            "verb_distribution": self.verb_distribution(),
            "most_active_hours": self.most_active_hours(),
            "resumed_rate": self.resumed_rate(),
            # Social
            "shared_count": self.shared_count(),
            # Retention
            "retention_d1": self.retention(1),
            "retention_d7": self.retention(7),
            "retention_d30": self.retention(30),
            "churn_rate": self.churn_rate(),
            # Content
            "top_activities": self.top_activities(),
            "hardest_activities": self.hardest_activities(),
            "avg_time_per_activity": self.avg_time_per_activity(),
            "content_dropout_map": self.content_dropout_map(),
            # Advanced TRAX-specific indicators
            "resume_rate": self.resume_rate(),
            "near_completion_abandonment": self.near_completion_abandonment(),
            "avg_progress_at_suspension": self.avg_progress_at_suspension(),
            "content_engagement_by_type": self.content_engagement_by_type(),
            "question_difficulty_ranking": self.question_difficulty_ranking(),
            "interaction_type_performance": self.interaction_type_performance(),
            "dropout_hotspots": self.dropout_hotspots(),
            "launch_success_rate": self.launch_success_rate(),
        }