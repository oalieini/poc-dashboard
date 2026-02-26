"""
xAPI Learning Analytics Dashboard
Run with: streamlit run frontend/app.py
"""

import sys
import os
import logging
import requests
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional

import streamlit as st
import yaml

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from backend.xapi_client import XAPIClient, MockXAPIClient, _unwrap_trax
from backend.calculator import IndicatorCalculator
from backend.cache import cache
from frontend.components import (
    inject_css,
    metric_card,
    section_header,
    render_indicator_chart,
    connection_status_badge,
    kpi_summary_bar,
    CATEGORY_COLORS,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="xAPI Analytics Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_css()


@st.cache_resource
def load_catalog() -> Dict:
    catalog_path = ROOT / "config" / "indicators.yaml"
    with open(catalog_path, "r") as f:
        return yaml.safe_load(f)


catalog = load_catalog()


def get_client(endpoint: str, username: str, password: str, token: str, use_mock: bool):
    if use_mock:
        return MockXAPIClient()
    return XAPIClient(
        endpoint=endpoint,
        username=username,
        password=password,
        token=token,
    )


def load_statements(client, since: datetime, until: datetime, period_label: str,
                    max_stmts: int = 10_000, use_chunked: bool = False, chunk_days: int = 7) -> List[Dict]:
    cache_key = f"stmts:{period_label}:{since.date()}:{until.date()}:{max_stmts}:{use_chunked}"
    cached = cache.get(cache_key)
    if cached is not None:
        return cached

    if use_chunked:
        progress_bar = st.progress(0, text=f"Loading {period_label} statements in chunks…")

        def progress_cb(fetched, total):
            pct = min(fetched / max(total, 1), 1.0)
            progress_bar.progress(pct, text=f"Loading: {fetched:,} / {total:,} statements")

        try:
            stmts = client.get_statements_chunked(
                since=since, until=until,
                chunk_days=chunk_days,
                max_per_chunk=max_stmts,
                progress_cb=progress_cb,
            )
            progress_bar.empty()
        except requests.exceptions.ReadTimeout as e:
            progress_bar.empty()
            st.error("Connection timeout - The query took too long to complete.")

            st.caption(f"Technical details: {e}")
            return []
        except Exception as e:
            progress_bar.empty()
            st.error(f"LRS error: {e}")

            return []
    else:
        # Show a count estimate first (TRAX supports fast count)
        total_count = client.get_total_count(since=since, until=until)
        if total_count and total_count > max_stmts:
            st.info(
                f"Note: {total_count:,} statements in this period — "
                f"loading first {max_stmts:,}. Increase the cap or use chunked mode for more."
            )

        progress_bar = st.progress(0, text=f"Loading {period_label} statements from LRS…")

        def progress_cb(fetched, total):
            pct = min(fetched / max(total or max_stmts, 1), 1.0)
            progress_bar.progress(pct, text=f"Fetching: {fetched:,} statements…")

        try:
            stmts = client.get_all_statements_batched(
                since=since, until=until,
                batch_size=500,
                max_statements=max_stmts,
                progress_cb=progress_cb,
            )
            progress_bar.empty()
        except requests.exceptions.ReadTimeout as e:
            progress_bar.empty()
            st.error("Connection timeout - The query took too long to complete.")

            st.caption(f"Technical details: {e}")
            return []
        except Exception as e:
            progress_bar.empty()
            st.error(f"LRS error: {e}")
            if "timeout" in str(e).lower():
                st.info("Tip: Try reducing the date range or enabling chunked mode")
            return []

    cache.set(cache_key, stmts)
    return stmts


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────

max_stmts = 10_000
use_chunked = False
chunk_days = 7

with st.sidebar:
    st.markdown("""
    <div style="display:flex;align-items:center;gap:10px;margin-bottom:20px;padding-bottom:16px;border-bottom:1px solid #1E2235;">
        <div>
            <div style="font-size:16px;font-weight:700;color:#E8EAF6;">xAPI Dashboard</div>
            <div style="font-size:11px;color:#4A5068;">Learning Analytics</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("#### LRS Connection")

    use_mock = st.toggle("Use Mock / Demo Data", value=True, help="Toggle off to connect to a real LRS")

    if not use_mock:
        lrs_endpoint = st.text_input(
            "LRS Endpoint",
            value=os.getenv("LRS_ENDPOINT",
                            "https://lrs.maskott.com/trax/api/3914708f-d72c-4a96-8003-f1f2f8d232af/xapi/ext/statements"),
            help="Base xAPI endpoint URL",
        )
        auth_method = st.radio("Auth Method", ["Basic Auth", "Bearer Token"], horizontal=True)
        if auth_method == "Basic Auth":
            lrs_username = st.text_input("Username", value=os.getenv("LRS_USERNAME"))
            lrs_password = st.text_input("Password", type="password", value=os.getenv("LRS_PASSWORD"))
            lrs_token = ""
        else:
            lrs_token = st.text_input("Bearer Token", type="password", value=os.getenv("LRS_TOKEN", ""))
            lrs_username = lrs_password = ""
    else:
        lrs_endpoint = lrs_username = lrs_password = lrs_token = ""

    st.divider()

    # ── Date range ─────────────────────────────────────────────────────────
    st.markdown("####  Period")

    preset = st.selectbox(
        "Quick range",
        ["Last 7 days", "Last 30 days", "Last 90 days", "Custom"],
        index=1,
    )

    now = datetime.now(timezone.utc)
    if preset == "Last 7 days":
        default_since = now - timedelta(days=7)
        default_until = now
    elif preset == "Last 30 days":
        default_since = now - timedelta(days=30)
        default_until = now
    elif preset == "Last 90 days":
        default_since = now - timedelta(days=90)
        default_until = now
    else:
        default_since = now - timedelta(days=30)
        default_until = now

    if preset == "Custom":
        col1, col2 = st.columns(2)
        with col1:
            since_date = st.date_input("From", value=default_since.date())
        with col2:
            until_date = st.date_input("To", value=default_until.date())
        since = datetime(since_date.year, since_date.month, since_date.day, tzinfo=timezone.utc)
        until = datetime(until_date.year, until_date.month, until_date.day, 23, 59, 59, tzinfo=timezone.utc)

        # Validate date range
        if since > until:
            st.error("[ERROR] 'From' date must be before 'To' date!")
            st.info("Automatically swapping dates...")
            since, until = until, since  # Swap them
            st.success(f"[OK] Corrected to: {since.date()} to {until.date()}")
    else:
        since = default_since
        until = default_until

    prev_since = since - (until - since)
    prev_until = since

    st.divider()

    # ── Indicator category filter ───────────────────────────────────────────
    st.markdown("#### Categories")

    all_categories = list(catalog["indicators"].keys())
    selected_categories = []

    HIDDEN_BY_DEFAULT = []  #  ["social", "retention"]

    for cat_key, cat_data in catalog["indicators"].items():
        default_visible = cat_key not in HIDDEN_BY_DEFAULT

        checked = st.checkbox(
            f'{cat_data["label"]}',
            value=default_visible,
            key=f"cat_{cat_key}",
        )
        if checked:
            selected_categories.append(cat_key)

    st.divider()

    # ── Display options ─────────────────────────────────────────────────────
    st.markdown("#### Display")

    show_charts = st.toggle("Show charts in cards", value=True)
    show_details = st.toggle("Show indicator details", value=True)
    mastery_threshold = st.slider("Mastery threshold (%)", 50, 100, 80, step=5)
    cards_per_row = st.select_slider("Cards per row", options=[2, 3, 4], value=3)

    st.divider()

    # ── Large dataset options ───────────────────────────────────────────────
    st.divider()
    st.markdown("#### Performance")

    max_stmts = st.select_slider(
        "Max statements to load",
        options=[1_000, 5_000, 10_000, 25_000, 50_000, 100_000],
        value=25_000,
        help="TRAX now supports ID-based pagination - can access all 760k+ statements!",
        format_func=lambda x: f"{x:,}",
    )

    use_chunked = st.toggle(
        "Date-chunked loading",
        value=False,
        help="Split date range into weekly chunks (slower but more complete for large ranges).",
    )

    chunk_days = 7
    if use_chunked:
        chunk_days = st.slider("Chunk size (days)", 3, 30, 7)

    st.divider()

    st.markdown("#### Cache")

    cache_stats = cache.stats()
    st.caption(f"{cache_stats['alive_keys']} keys cached · TTL {cache_stats['ttl_seconds']}s")

    if st.button("Clear cache & refresh", use_container_width=True):
        cache.clear_all()
        st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# MAIN — Load data
# ─────────────────────────────────────────────────────────────────────────────

client = get_client(lrs_endpoint, lrs_username if not use_mock else "", lrs_password if not use_mock else "",
                    lrs_token if not use_mock else "", use_mock)

# Test connection
connected = client.ping()


stmts_curr = load_statements(client, since, until, "current",
                             max_stmts=max_stmts, use_chunked=use_chunked, chunk_days=chunk_days)
stmts_prev = load_statements(client, prev_since, prev_until, "previous",
                             max_stmts=max_stmts, use_chunked=use_chunked, chunk_days=chunk_days)



calc = IndicatorCalculator(stmts_curr, stmts_prev)

# Compute all indicators
with st.spinner("Computing indicators…"):
    results_cache_key = f"results:{len(stmts_curr)}:{len(stmts_prev)}:{mastery_threshold}"
    results = cache.get(results_cache_key)
    if results is None:
        results = calc.compute_all()
        # Override mastery_rate with custom threshold
        results["mastery_rate"] = calc.mastery_rate(threshold=mastery_threshold)
        cache.set(results_cache_key, results)

# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────

col_title, col_status = st.columns([3, 1])
with col_title:
    st.markdown("""
    <div style="margin-bottom: 4px;">
        <span style="font-size: 28px; font-weight: 800; color: #E8EAF6;">Learning Analytics</span>
        <span style="font-size: 14px; color: #4A5068; margin-left: 12px;">xAPI Statement Dashboard</span>
    </div>
    """, unsafe_allow_html=True)
    period_str = f"{since.strftime('%b %d, %Y')} → {until.strftime('%b %d, %Y')}"
    st.caption(
        f"Period: {period_str} · {len(stmts_curr):,} statements · {results.get('active_learners', {}).get('value', 0)} learners")

with col_status:
    st.markdown("<div style='margin-top:14px;text-align:right;'>", unsafe_allow_html=True)
    connection_status_badge(connected or use_mock, lrs_endpoint if not use_mock else "Mock LRS")
    st.markdown("</div>", unsafe_allow_html=True)

st.divider()

# ── KPI Summary bar ─────────────────────────────────────────────────────────
kpi_summary_bar(results, catalog)

st.divider()


tab_dash, tab_students, tab_raw = st.tabs([
    "Dashboard",
    "Students",
    "Raw Statements",
])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────

with tab_dash:
    if not selected_categories:
        st.warning("Select at least one category in the sidebar.")
    else:
        for cat_key in selected_categories:
            if cat_key not in catalog["indicators"]:
                continue

            cat_data = catalog["indicators"][cat_key]
            cat_color = CATEGORY_COLORS.get(cat_key, "#4F8EF7")
            indicators_in_cat = cat_data["indicators"]
            visible_indicators = {k: v for k, v in indicators_in_cat.items() if k in results}

            section_header(
                title=cat_data["label"],
                # icon=cat_data["icon"],
                count=len(visible_indicators),
                color=cat_color,
            )

            # Render cards in rows
            ind_keys = list(visible_indicators.keys())
            for row_start in range(0, len(ind_keys), cards_per_row):
                row_keys = ind_keys[row_start: row_start + cards_per_row]
                cols = st.columns(cards_per_row)
                for col, ind_key in zip(cols, row_keys):
                    ind_meta = visible_indicators[ind_key]
                    result = results.get(ind_key, {})
                    with col:
                        with st.container():
                            metric_card(
                                label=ind_meta["label"],
                                value=result.get("value", 0),
                                unit=ind_meta.get("unit", ""),
                                trend=result.get("trend") if ind_meta.get("trend") else None,
                                description=ind_meta.get("description", "") if show_details else "",
                                category=cat_key,
                                # icon=cat_data["icon"],
                            )
                            if show_charts:
                                render_indicator_chart(
                                    indicator_key=ind_key,
                                    result=result,
                                    category_color=cat_color,
                                    chart_type=ind_meta.get("chart", "bar"),
                                )
                # Fill empty cols
                for empty_col in cols[len(row_keys):]:
                    with empty_col:
                        st.empty()



# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — STUDENT VIEW
# ─────────────────────────────────────────────────────────────────────────────

with tab_students:
    st.markdown("###  Student Analytics")
    st.caption("Select a student to view detailed individual performance and session breakdown.")

    if not stmts_curr:
        st.info("No statements loaded. Adjust the date range or check your LRS connection.")
    else:
        # Get student list
        students = calc.get_student_list()

        if not students:
            st.warning("No students found in the current period.")
        else:
            st.markdown(f"**{len(students)} students** found in this period")

            # Student selection
            col_select, col_search = st.columns([3, 1])

            with col_search:
                search_student = st.text_input(" Search", placeholder="Student name or ID...", label_visibility="collapsed")

            # Filter students
            filtered_students = students
            if search_student:
                filtered_students = [
                    s for s in students
                    if search_student.lower() in s["actor_name"].lower()
                       or search_student.lower() in s["actor_id"].lower()
                ]

            with col_select:
                if not filtered_students:
                    st.warning(f"No students match '{search_student}'")
                    selected_student_name = None
                else:
                    student_options = {
                        f"{s['actor_name']} ({s['total_statements']} stmts)": s["actor_id"]
                        for s in filtered_students[:50]  # Limit to first 50 for performance
                    }
                    selected_student_name = st.selectbox(
                        "Select Student",
                        options=list(student_options.keys()),
                        index=0,
                    )

            if selected_student_name:
                selected_actor_id = student_options[selected_student_name]
                selected_student = next(s for s in students if s["actor_id"] == selected_actor_id)

                st.divider()

                # Student summary card
                from frontend.components import student_card, session_card, student_metric_grid

                student_card(selected_student)

                # Session tabs
                st.markdown("####  View by")
                view_mode = st.radio(
                    "View mode",
                    ["All Time", "By Session"],
                    horizontal=True,
                    label_visibility="collapsed",
                    key=f"view_mode_{selected_actor_id}"
                )

                st.divider()

                if view_mode == "All Time":
                    # Show indicators across all student's data
                    st.markdown("###  Overall Performance")
                    student_results = calc.compute_student_indicators(selected_actor_id, session_id=None)

                    if student_results:
                        student_metric_grid(student_results, cols=4)

                        # Charts
                        st.markdown("####  Detailed Charts")

                        chart_cols = st.columns(2)

                        # Verb distribution
                        with chart_cols[0]:
                            if "verb_distribution" in student_results:
                                vd = student_results["verb_distribution"]
                                if vd["chart_data"].get("labels"):
                                    from frontend.components import pie_chart

                                    pie_chart(
                                        vd["chart_data"]["labels"],
                                        vd["chart_data"]["values"],
                                        title="Actions Distribution",
                                        height=300,
                                    )

                        # Top activities
                        with chart_cols[1]:
                            if "top_activities" in student_results:
                                ta = student_results["top_activities"]
                                if ta["chart_data"].get("labels"):
                                    from frontend.components import bar_chart

                                    bar_chart(
                                        ta["chart_data"]["labels"],
                                        ta["chart_data"]["values"],
                                        title="Most Accessed Activities",
                                        color="#4F8EF7",
                                        height=300,
                                        horizontal=True,
                                    )

                        # Score timeline (if available)
                        if "avg_score" in student_results and student_results["avg_score"]["chart_data"].get("values"):
                            st.markdown("#### Score Progression")
                            score_vals = student_results["avg_score"]["chart_data"]["values"]
                            from frontend.components import line_chart

                            line_chart(
                                x=list(range(1, len(score_vals) + 1)),
                                y=score_vals,
                                title="Scores in Chronological Order",
                                color="#A855F7",
                                height=200,
                            )
                            st.caption(f"X-axis: Scored activities (1st to {len(score_vals)}th) • Y-axis: Score (0-100)")

                        # Engagement timeline - Statement count over time
                        st.markdown("#### Engagement Over Time")
                        student_df = calc.df[calc.df["actor_id"] == selected_actor_id].copy()
                        if not student_df.empty and "timestamp" in student_df.columns:
                            student_df = student_df.sort_values("timestamp")

                            # Group by date to show daily statement count
                            student_df["date"] = student_df["timestamp"].dt.date
                            daily_counts = student_df.groupby("date").size().reset_index(name="count")

                            if len(daily_counts) > 0:
                                from frontend.components import line_chart

                                # Format dates for display
                                date_labels = [d.strftime("%b %d") for d in daily_counts["date"]]

                                line_chart(
                                    x=date_labels,
                                    y=daily_counts["count"].tolist(),
                                    title="Daily Statement Count (by date)",
                                    color="#4F8EF7",
                                    height=200,
                                )

                                # Add caption with date range
                                start_date = daily_counts["date"].iloc[0].strftime("%b %d, %Y")
                                end_date = daily_counts["date"].iloc[-1].strftime("%b %d, %Y")
                                st.caption(f"X-axis: Dates ({start_date} to {end_date}) • Y-axis: Number of statements per day")
                            else:
                                st.info("No engagement data available")
                        else:
                            st.info("No engagement data available")
                    else:
                        st.info("No indicator data available for this student.")


                else:  # By Session
                    sessions = calc.get_student_sessions(selected_actor_id)

                    if not sessions:
                        st.info("No sessions detected for this student.")
                    else:
                        st.markdown(f"**{len(sessions)} sessions** detected")

                        # Session selector
                        col_sess_left, col_sess_right = st.columns([1, 3])

                        with col_sess_left:
                            st.markdown("##### Sessions")
                            selected_session_id = st.radio(
                                "Select session",
                                options=[s["session_id"] for s in sessions],
                                format_func=lambda
                                    x: f"Session {next(s['session_number'] for s in sessions if s['session_id'] == x)}",
                                label_visibility="collapsed",
                            )

                        with col_sess_right:
                            selected_session = next(s for s in sessions if s["session_id"] == selected_session_id)
                            session_card(selected_session, selected=True)

                            st.markdown("#####  Session Metrics")

                            # Compute session-specific indicators
                            session_results = calc.compute_student_indicators(
                                selected_actor_id,
                                session_id=selected_session_id
                            )

                            if session_results:
                                student_metric_grid(session_results, cols=4)

                                # Session charts
                                st.markdown("####  Session Details")

                                sess_chart_cols = st.columns(2)

                                with sess_chart_cols[0]:
                                    if "verb_distribution" in session_results:
                                        vd = session_results["verb_distribution"]
                                        if vd["chart_data"].get("labels"):
                                            from frontend.components import pie_chart

                                            pie_chart(
                                                vd["chart_data"]["labels"],
                                                vd["chart_data"]["values"],
                                                title="Actions in This Session",
                                                height=280,
                                            )

                                with sess_chart_cols[1]:
                                    if "top_activities" in session_results:
                                        ta = session_results["top_activities"]
                                        if ta["chart_data"].get("labels"):
                                            from frontend.components import bar_chart

                                            bar_chart(
                                                ta["chart_data"]["labels"],
                                                ta["chart_data"]["values"],
                                                title="Activities in This Session",
                                                color="#22C55E",
                                                height=280,
                                                horizontal=True,
                                            )
                            else:
                                st.info("No metrics available for this session.")

                            # Session Comparison
                            st.divider()
                            st.markdown("#### Session Analysis")

                            st.caption("How this student compares to others in the same session")

                            session_comparison = calc.compare_student_to_session(
                                selected_actor_id,
                                selected_session_id
                            )

                            if session_comparison:
                                from frontend.components import session_comparison_card

                                session_comparison_card(session_comparison)
                            else:
                                st.info("Session comparison requires multiple students in the same session")



# ─────────────────────────────────────────────────────────────────────────────
# TAB — RAW STATEMENTS
# ─────────────────────────────────────────────────────────────────────────────

with tab_raw:
    st.markdown("### Raw xAPI Statements")

    if not stmts_curr:
        st.info("No statements loaded. Adjust the date range or check your LRS connection.")
    else:
        import pandas as pd
        from backend.calculator import StatementParser

        col_f1, col_f2, col_f3 = st.columns(3)
        with col_f1:
            verb_filter = st.multiselect(
                "Filter by verb",
                options=sorted(set(StatementParser.verb_display(s) for s in stmts_curr)),
            )
        with col_f2:
            actor_filter = st.text_input("Filter by actor")
        with col_f3:
            activity_filter = st.text_input("Filter by activity name")

        rows = []
        for s in stmts_curr:
            rows.append({
                "Timestamp": StatementParser.timestamp(s),
                "Actor": StatementParser.actor_name(s),
                "Verb": StatementParser.verb_display(s),
                "Activity": StatementParser.activity_name(s),
                "Score": StatementParser.score_scaled(s),
                "Duration (min)": StatementParser.duration_minutes(s),
                "Success": StatementParser.success(s),
                "Progress": StatementParser.progress(s),
            })

        df_raw = pd.DataFrame(rows)
        if verb_filter:
            df_raw = df_raw[df_raw["Verb"].isin(verb_filter)]
        if actor_filter:
            df_raw = df_raw[df_raw["Actor"].str.contains(actor_filter, case=False, na=False)]
        if activity_filter:
            df_raw = df_raw[df_raw["Activity"].str.contains(activity_filter, case=False, na=False)]

        st.caption(f"Showing {len(df_raw):,} of {len(stmts_curr):,} statements")
        st.dataframe(df_raw, use_container_width=True, hide_index=True)
