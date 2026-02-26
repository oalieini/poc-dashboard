"""
Reusable Streamlit UI components for the xAPI dashboard.
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from typing import Dict, List, Optional, Any

# ─────────────────────────────────────────────
# COLOR PALETTE
# ─────────────────────────────────────────────

CATEGORY_COLORS = {
    "engagement": "#4F8EF7",
    "completion": "#22C55E",
    "performance": "#F59E0B",
    "interaction": "#A855F7",
    "social": "#EC4899",
    "retention": "#14B8A6",
    "content": "#F97316",
}

BG_COLOR = "#0F1117"
SURFACE_COLOR = "#1A1D2E"
BORDER_COLOR = "#2D3057"
TEXT_SECONDARY = "#8B92B0"


# ─────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────

def format_duration(minutes: float) -> str:
    """
    Format duration intelligently based on length.
    Returns a human-readable string.
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

def inject_css():
    """Inject global CSS overrides for dark professional styling."""
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

        html, body, [class*="css"] {
            font-family: 'Space Grotesk', sans-serif;
        }

        .stApp {
            background: #0A0C14;
        }

        /* Sidebar */
        section[data-testid="stSidebar"] {
            background: #0F1117;
            border-right: 1px solid #1E2235;
        }

        /* Metric cards */
        .xapi-card {
            background: linear-gradient(135deg, #12172A 0%, #1A1D30 100%);
            border: 1px solid #1E2640;
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 12px;
            transition: transform 0.2s ease, border-color 0.2s ease;
            position: relative;
            overflow: hidden;
        }

        .xapi-card:hover {
            transform: translateY(-2px);
            border-color: #3A4080;
        }

        .xapi-card::before {
            content: '';
            position: absolute;
            top: 0; left: 0;
            right: 0; height: 3px;
            border-radius: 12px 12px 0 0;
        }

        .card-engagement::before { background: #4F8EF7; }
        .card-completion::before { background: #22C55E; }
        .card-performance::before { background: #F59E0B; }
        .card-interaction::before { background: #A855F7; }
        .card-social::before { background: #EC4899; }
        .card-retention::before { background: #14B8A6; }
        .card-content::before { background: #F97316; }

        .card-label {
            font-size: 11px;
            font-weight: 600;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: #6B7494;
            margin-bottom: 6px;
        }

        .card-value {
            font-size: 32px;
            font-weight: 700;
            color: #E8EAF6;
            line-height: 1;
            font-family: 'JetBrains Mono', monospace;
        }

        .card-unit {
            font-size: 14px;
            color: #6B7494;
            margin-left: 4px;
        }

        .card-trend-up {
            font-size: 12px;
            color: #22C55E;
            font-weight: 600;
        }

        .card-trend-down {
            font-size: 12px;
            color: #EF4444;
            font-weight: 600;
        }

        .card-trend-neutral {
            font-size: 12px;
            color: #6B7494;
        }

        .card-desc {
            font-size: 11px;
            color: #4A5068;
            margin-top: 4px;
        }

        /* Section headers */
        .section-header {
            display: flex;
            align-items: center;
            gap: 10px;
            margin: 28px 0 16px 0;
            padding-bottom: 10px;
            border-bottom: 1px solid #1E2235;
        }

        .section-icon {
            font-size: 20px;
        }

        .section-title {
            font-size: 15px;
            font-weight: 700;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            color: #C0C8E8;
        }

        .section-count {
            font-size: 11px;
            background: #1E2640;
            color: #6B7494;
            padding: 2px 8px;
            border-radius: 10px;
        }

        /* KPI bar at top */
        .kpi-bar {
            background: linear-gradient(90deg, #12172A 0%, #1A1F35 100%);
            border: 1px solid #1E2640;
            border-radius: 10px;
            padding: 14px 20px;
            margin-bottom: 24px;
        }

        /* Tags */
        .verb-tag {
            display: inline-block;
            padding: 3px 10px;
            border-radius: 20px;
            font-size: 11px;
            font-weight: 600;
            margin: 2px;
            background: #1E2640;
            color: #8B99C0;
            border: 1px solid #2D3A60;
        }

        /* Dividers */
        hr { border-color: #1E2235 !important; }

        /* Streamlit overrides */
        div[data-testid="metric-container"] {
            background: #12172A;
            border: 1px solid #1E2640;
            border-radius: 10px;
            padding: 12px;
        }

        .stSelectbox > div > div {
            background: #12172A;
            border-color: #2D3A60;
        }

        .stMultiSelect > div > div {
            background: #12172A;
        }

        .stDateInput > div > div {
            background: #12172A;
        }

        /* Scrollbar */
        ::-webkit-scrollbar { width: 6px; }
        ::-webkit-scrollbar-track { background: #0A0C14; }
        ::-webkit-scrollbar-thumb { background: #2D3A60; border-radius: 3px; }

        .stMarkdown h3 {
            color: #C0C8E8;
        }
    </style>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# METRIC CARD
# ─────────────────────────────────────────────

def metric_card(
    label: str,
    value: Any,
    unit: str = "",
    trend: Optional[float] = None,
    description: str = "",
    category: str = "engagement",
    icon: str = "",
):
    """Render a styled metric card."""
    if trend is not None:
        if trend > 0:
            trend_html = f'<div class="card-trend-up">▲ {abs(trend):.1f}% vs prev</div>'
        elif trend < 0:
            trend_html = f'<div class="card-trend-down">▼ {abs(trend):.1f}% vs prev</div>'
        else:
            trend_html = '<div class="card-trend-neutral">→ unchanged</div>'
    else:
        trend_html = '<div class="card-trend-neutral">— no comparison</div>'

    icon_html = f'<span style="font-size:18px; margin-right: 6px;">{icon}</span>' if icon else ""

    # Smart formatting for duration values
    if unit == "minutes" and isinstance(value, (int, float)) and value > 0:
        # Format duration intelligently
        value_str = format_duration(value)
        unit = ""  # Unit already included in formatted string
    elif isinstance(value, float):
        value_str = f"{value:.1f}"
    else:
        value_str = f"{int(value):,}" if isinstance(value, (int, float)) else str(value)

    st.markdown(f"""
    <div class="xapi-card card-{category}">
        <div class="card-label">{icon_html}{label}</div>
        <div>
            <span class="card-value">{value_str}</span>
            <span class="card-unit">{unit}</span>
        </div>
        <div class="card-desc">{description}</div>
    </div>
    """, unsafe_allow_html=True)


def section_header(title: str, count: int = 0, color: str = "#4F8EF7"):
    """Render a section header."""
    count_badge = f'<span class="section-count">{count} indicators</span>' if count else ""
    st.markdown(f"""
    <div class="section-header">
        <span class="section-title" style="color: {color};">{title}</span>
        {count_badge}
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# ─────────────────────────────────────────────

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Space Grotesk, sans-serif", color="#8B92B0", size=11),
    margin=dict(l=10, r=10, t=30, b=30),
    xaxis=dict(
        showgrid=True,
        gridcolor="#1E2235",
        showline=False,
        tickfont=dict(color="#6B7494", size=10),
        zeroline=False,
    ),
    yaxis=dict(
        showgrid=True,
        gridcolor="#1E2235",
        showline=False,
        tickfont=dict(color="#6B7494", size=10),
        zeroline=False,
    ),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#8B92B0")),
)


def line_chart(x: List, y: List, title: str = "", color: str = "#4F8EF7", height: int = 200):
    if not x or not y:
        st.caption("No time series data")
        return
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=y,
        mode="lines+markers",
        line=dict(color=color, width=2),
        marker=dict(size=4, color=color),
        fill="tozeroy",
        fillcolor=f"rgba({int(color[1:3], 16)},{int(color[3:5], 16)},{int(color[5:7], 16)},0.1)",
    ))
    layout = {**PLOTLY_LAYOUT, "height": height, "title": dict(text=title, font=dict(size=12, color="#C0C8E8"))}
    fig.update_layout(**layout)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def bar_chart(labels: List, values: List, title: str = "", color: str = "#4F8EF7", height: int = 200, horizontal: bool = False):
    if not labels or not values:
        st.caption("No data")
        return
    if horizontal:
        fig = go.Figure(go.Bar(y=labels, x=values, orientation="h",
                               marker_color=color, marker_line_width=0))
    else:
        fig = go.Figure(go.Bar(x=labels, y=values, marker_color=color, marker_line_width=0))
    layout = {**PLOTLY_LAYOUT, "height": height, "title": dict(text=title, font=dict(size=12, color="#C0C8E8"))}
    fig.update_layout(**layout)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def pie_chart(labels: List, values: List, title: str = "", height: int = 250):
    if not labels or not values:
        st.caption("No data")
        return
    colors = ["#4F8EF7", "#22C55E", "#F59E0B", "#A855F7", "#EC4899", "#14B8A6", "#F97316",
              "#64748B", "#EF4444", "#06B6D4", "#84CC16", "#F472B6"]
    fig = go.Figure(go.Pie(
        labels=labels, values=values,
        marker=dict(colors=colors[:len(labels)], line=dict(color="#0A0C14", width=2)),
        textfont=dict(color="#C0C8E8", size=11),
        hole=0.4,
    ))
    layout = {**PLOTLY_LAYOUT, "height": height, "title": dict(text=title, font=dict(size=12, color="#C0C8E8")),
              "showlegend": True}
    del layout["xaxis"], layout["yaxis"]
    fig.update_layout(**layout)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def gauge_chart(value: float, title: str = "", color: str = "#4F8EF7",
                min_val: float = 0, max_val: float = 100, height: int = 200):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number=dict(suffix="%", font=dict(color="#E8EAF6", size=28, family="JetBrains Mono")),
        gauge=dict(
            axis=dict(range=[min_val, max_val], tickfont=dict(color="#6B7494")),
            bar=dict(color=color),
            bgcolor="#1A1D30",
            borderwidth=0,
            steps=[
                dict(range=[min_val, max_val * 0.4], color="#12172A"),
                dict(range=[max_val * 0.4, max_val * 0.7], color="#1A1D30"),
                dict(range=[max_val * 0.7, max_val], color="#1E2235"),
            ],
            threshold=dict(
                line=dict(color="#6B7494", width=2),
                thickness=0.75,
                value=max_val * 0.7,
            ),
        ),
        title=dict(text=title, font=dict(color="#C0C8E8", size=12)),
    ))
    layout = {**PLOTLY_LAYOUT, "height": height}
    del layout["xaxis"], layout["yaxis"]
    fig.update_layout(**layout)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def histogram_chart(values: List, title: str = "", color: str = "#4F8EF7", height: int = 200, nbins: int = 15):
    if not values:
        st.caption("No data")
        return
    fig = go.Figure(go.Histogram(
        x=values, nbinsx=nbins,
        marker_color=color, marker_line_color="#0A0C14", marker_line_width=1,
        opacity=0.85,
    ))
    layout = {**PLOTLY_LAYOUT, "height": height, "title": dict(text=title, font=dict(size=12, color="#C0C8E8")),
              "bargap": 0.05}
    fig.update_layout(**layout)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def heatmap_hour(x: List[int], y: List[int], title: str = "", height: int = 220):
    """Activity heatmap by hour of day."""
    if not x or not y:
        st.caption("No data")
        return
    fig = go.Figure(go.Bar(
        x=[f"{h:02d}:00" for h in x],
        y=y,
        marker=dict(
            color=y,
            colorscale=[[0, "#12172A"], [0.3, "#1E3A5F"], [0.7, "#2563EB"], [1, "#60A5FA"]],
            line=dict(width=0),
        ),
    ))
    layout = {**PLOTLY_LAYOUT, "height": height, "title": dict(text=title, font=dict(size=12, color="#C0C8E8"))}
    fig.update_layout(**layout)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def render_indicator_chart(indicator_key: str, result: Dict, category_color: str, chart_type: str):
    """Route indicator result to the appropriate chart type."""
    cd = result.get("chart_data", {})

    if chart_type == "line" and cd.get("x") and cd.get("y"):
        line_chart(cd["x"], cd["y"], color=category_color, height=180)

    elif chart_type == "bar" and cd.get("labels") and cd.get("values"):
        bar_chart(cd["labels"], cd["values"], color=category_color, height=220, horizontal=True)

    elif chart_type == "bar" and cd.get("x") and cd.get("y"):
        bar_chart(cd["x"], cd["y"], color=category_color, height=180)

    elif chart_type == "gauge":
        gauge_chart(result["value"], color=category_color, height=180)

    elif chart_type == "histogram" and cd.get("values"):
        histogram_chart(cd["values"], color=category_color, height=180)

    elif chart_type == "histogram" and cd.get("x") and cd.get("y"):
        bar_chart(cd["x"], cd["y"], color=category_color, height=180)

    elif chart_type == "pie" and cd.get("labels") and cd.get("values"):
        pie_chart(cd["labels"], cd["values"], height=260)

    elif chart_type == "heatmap" and cd.get("x") and cd.get("y"):
        heatmap_hour(cd["x"], cd["y"], height=180)


def connection_status_badge(connected: bool, endpoint: str = ""):
    if connected:
        st.markdown(
            f'<div style="display:inline-flex;align-items:center;gap:6px;background:#052E16;border:1px solid #166534;'
            f'border-radius:20px;padding:4px 12px;font-size:12px;color:#4ADE80;">'
            f'<span style="width:6px;height:6px;background:#4ADE80;border-radius:50%;"></span>'
            f'Connected · {endpoint[:40] or "LRS"}</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<div style="display:inline-flex;align-items:center;gap:6px;background:#2D0A0A;border:1px solid #7F1D1D;'
            'border-radius:20px;padding:4px 12px;font-size:12px;color:#FCA5A5;">'
            '<span style="width:6px;height:6px;background:#EF4444;border-radius:50%;"></span>'
            'Disconnected · using mock data</div>',
            unsafe_allow_html=True
        )


def student_card(
    student: Dict,
    on_click_label: str = "View Details",
):
    """Render a student summary card with click action."""
    actor_name = student.get("actor_name", student.get("actor_id", "Unknown"))[:30]
    total_stmts = student.get("total_statements", 0)
    first = student.get("first_activity")
    last = student.get("last_activity")
    activities = student.get("unique_activities", 0)

    first_str = first.strftime("%Y-%m-%d %H:%M") if first else "—"
    last_str = last.strftime("%Y-%m-%d %H:%M") if last else "—"

    st.markdown(f"""
    <div class="xapi-card card-engagement" style="margin-bottom: 12px;">
        <div style="display: flex; justify-content: space-between; align-items: start;">
            <div style="flex: 1;">
                <div class="card-label">STUDENT</div>
                <div style="font-size: 16px; font-weight: 600; color: #E8EAF6; margin: 4px 0;">
                    {actor_name}
                </div>
                <div style="font-size: 11px; color: #6B7494; margin-top: 8px;">
                    {total_stmts:,} statements · {activities} activities
                </div>
                <div style="font-size: 10px; color: #4A5068; margin-top: 4px;">
                    First: {first_str} · Last: {last_str}
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def session_card(session: Dict, selected: bool = False):
    """Render a session summary card with Maskott-specific metadata."""
    session_num = session.get("session_number", 0)
    session_name = session.get("session_name", f"Session {session_num}")
    session_type = session.get("session_type", "Session")
    start = session.get("start")
    end = session.get("end")
    duration = session.get("duration_min", 0)
    stmt_count = session.get("statement_count", 0)
    activities = session.get("activities", [])
    verbs = session.get("verbs", [])
    attempts = session.get("attempts", [])
    modules = session.get("modules", [])
    registration = session.get("registration")

    start_str = start.strftime("%Y-%m-%d %H:%M") if start else "—"
    end_str = end.strftime("%H:%M") if end else "—"

    border_color = "#4F8EF7" if selected else "#1E2640"
    bg_color = "#1A2540" if selected else "#12172A"

    # Session type icon
    display_name = session_name if len(session_name) <= 50 else session_name[:47] + "..."

    # Format duration
    duration_display = format_duration(duration)

    # Build metadata lines
    meta_lines = [
        f"{start_str} → {end_str} ({duration_display})",
        f"{stmt_count} statements · {len(activities)} activities · {len(verbs)} verbs"
    ]

    if attempts:
        attempt_display = f"{len(attempts)} attempt(s)" if len(attempts) > 1 else "1 attempt"
        meta_lines.append(f"Attempts: {attempt_display}")

    if modules:
        module_display = f"{len(modules)} module(s)" if len(modules) > 1 else "1 module"
        meta_lines.append(f"Modules: {module_display}")
    meta_html = "<br>".join(f'<div style="font-size: 10px; color: #6B7494; margin-top: 4px;">{line}</div>' for line in meta_lines)

    st.markdown(f"""
    <div style="background: {bg_color}; border: 2px solid {border_color}; border-radius: 8px;
                padding: 12px; margin-bottom: 8px; transition: all 0.2s;">
        <div style="display: flex; justify-content: space-between; align-items: start;">
            <div style="flex: 1;">
                <div style="font-size: 12px; font-weight: 700; color: #4F8EF7; letter-spacing: 0.05em;">
                    {session_type.upper()} {session_num}
                </div>
                <div style="font-size: 11px; color: #C0C8E8; margin-top: 2px; font-weight: 500;">
                    {display_name}
                </div>
                {meta_html}
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def student_metric_grid(results: Dict, cols: int = 2):
    """Render student indicators in a compact grid."""
    metric_keys = [
        ("total_statements", "#4F8EF7"),
        ("unique_activities", "#22C55E"),
        ("time_spent", "#F59E0B"),
        ("avg_score", "#A855F7"),
        ("completion_rate", "#22C55E"),
        ("pass_rate", "#14B8A6"),
        ("completions", "#22C55E"),
        ("passed", "#14B8A6"),
        ("failed", "#EF4444"),
        ("min_score", "#EF4444"),
        ("max_score", "#22C55E"),
    ]

    # Filter to available metrics that exist in results and have non-zero values
    available_metrics = []
    for k, c in metric_keys:
        if k in results:
            value = results[k]["value"]
            if k in ("avg_score", "min_score", "max_score"):
                available_metrics.append((k, c))
            elif isinstance(value, str):
                if value and value.strip():
                    available_metrics.append((k,  c))
            elif value != 0:
                available_metrics.append((k, c))

    if not available_metrics:
        st.info("No metrics available for this student")
        return

    for row_start in range(0, len(available_metrics), cols):
        row_metrics = available_metrics[row_start:row_start + cols]
        columns = st.columns(cols)

        for col, (key, color) in zip(columns, row_metrics):
            with col:
                r = results[key]
                value = r["value"]
                unit = ""

                # Format time_spent intelligently
                if key == "time_spent":
                    # Format based on duration length
                    formatted_time = format_duration(value)
                    value_str = formatted_time
                    unit = ""  # Already included in formatted string
                elif key in ("avg_score", "min_score", "max_score"):
                    value_str = f"{value:.1f}"
                    unit = "/100"
                elif key in ("completion_rate", "pass_rate"):
                    value_str = f"{value:.1f}"
                    unit = "%"
                elif isinstance(value, float):
                    value_str = f"{value:.1f}"
                else:
                    value_str = f"{int(value):,}"

                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #12172A 0%, #1A1D30 100%);
                            border-left: 3px solid {color}; border-radius: 6px; padding: 12px; margin-bottom: 8px;">
                    <div style="font-size: 9px; font-weight: 600; color: #6B7494; letter-spacing: 0.08em;
                                text-transform: uppercase; margin-bottom: 4px;">
                        {key.replace('_', ' ')}
                    </div>
                    <div style="font-size: 24px; font-weight: 700; color: #E8EAF6; font-family: 'JetBrains Mono', monospace;">
                        {value_str}<span style="font-size: 12px; color: #6B7494; margin-left: 2px;">{unit}</span>
                    </div>
                    <div style="font-size: 9px; color: #4A5068; margin-top: 2px;">
                        {r.get('details', '')}
                    </div>
                </div>
                """, unsafe_allow_html=True)


def kpi_summary_bar(results: Dict, catalog: Dict):
    """Top-level KPI strip."""
    keys = ["active_learners", "avg_score" ]
    labels = ["Active Learners", "Avg Score"]
    units = ["", "%", "/100", "%"]
    cols = st.columns(len(keys))
    for col, key, label, unit in zip(cols, keys, labels, units):
        with col:
            r = results.get(key, {})
            v = r.get("value", 0)
            col.metric(
                label=label,
                value=f"{v:.1f}{unit}" if isinstance(v, float) else f"{v}{unit}",
            )


# ─────────────────────────────────────────────
# SESSION COMPARISON COMPONENTS
# ─────────────────────────────────────────────

def session_comparison_card(comparison: Dict):
    """Display student vs session average comparison."""
    if not comparison:
        st.info("No session comparison data available")
        return

    st.markdown("#### Session Comparison")
    st.caption("How this student compares to session average")

    for metric, data in comparison.items():
        student_val = data.get("student", 0)
        session_avg = data.get("session_avg", 0)
        delta = data.get("delta", 0)
        delta_pct = data.get("delta_pct", 0)

        # Format duration values
        if metric == "session_duration":
            student_display = format_duration(student_val)
            session_avg_display = format_duration(session_avg)
            delta_display = f"{delta:+.1f} min"  # Show delta in minutes
        else:
            student_display = f"{student_val:.1f}" if isinstance(student_val, float) else student_val
            session_avg_display = f"{session_avg:.1f}" if isinstance(session_avg, float) else session_avg
            delta_display = f"{delta:+.1f}"

        # Determine color based on metric type
        if metric in ["score", "completion_rate", "pass_rate"]:
            # Higher is better
            color = "#22C55E" if delta > 0 else "#EF4444" if delta < 0 else "#6B7494"
        else:
            color = "#4F8EF7"

        delta_icon = "▲" if delta > 0 else "▼" if delta < 0 else "="

        st.markdown(f"""
        <div style="background: #12172A; border-left: 3px solid {color};
                    border-radius: 6px; padding: 12px; margin-bottom: 8px;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <div style="font-size: 10px; color: #6B7494; text-transform: uppercase;
                                letter-spacing: 0.05em; margin-bottom: 4px;">
                        {metric.replace('_', ' ')}
                    </div>
                    <div style="font-size: 20px; font-weight: 700; color: #E8EAF6;">
                        {student_display} <span style="font-size: 11px; color: #6B7494;">student</span>
                    </div>
                    <div style="font-size: 14px; color: #8B92B0; margin-top: 2px;">
                        {session_avg_display} <span style="font-size: 9px;">session avg</span>
                    </div>
                </div>
                <div style="text-align: right;">
                    <div style="font-size: 24px; font-weight: 700; color: {color};">
                        {delta_icon}
                    </div>
                    <div style="font-size: 16px; font-weight: 600; color: {color};">
                        {delta_display}
                    </div>
                    <div style="font-size: 11px; color: #6B7494;">
                        ({delta_pct:+.1f}%)
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)


def behavior_patterns_table(patterns: List[Dict]):
    """Display behavior pattern analysis."""
    if not patterns:
        st.info("No behavior patterns detected")
        return

    st.markdown("#### Behavior Patterns")
    st.caption("Common action sequences and their success rates")

    for pattern in patterns[:10]:
        sequence = pattern.get("sequence", "")
        occurrences = pattern.get("occurrences", 0)
        success_rate = pattern.get("success_rate", 0)

        # Color based on success rate
        if success_rate >= 70:
            color = "#22C55E"
            label = "High Success"
        elif success_rate >= 40:
            color = "#F59E0B"
            label = "Moderate"
        else:
            color = "#EF4444"
            label = "Low Success"

        st.markdown(f"""
        <div style="background: #12172A; border-left: 3px solid {color};
                    border-radius: 6px; padding: 10px; margin-bottom: 6px;">
            <div style="font-size: 14px; font-weight: 600; color: #E8EAF6; margin-bottom: 4px;">
                {sequence}
            </div>
            <div style="display: flex; justify-content: space-between; font-size: 11px; color: #6B7494;">
                <span>{occurrences} occurrences</span>
                <span style="color: {color}; font-weight: 600;">{success_rate:.1f}% success · {label}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)


def time_per_activity_table(activities: List[Dict]):
    """Display time spent analysis per activity."""
    if not activities:
        st.info("No time tracking data available")
        return

    st.markdown("#### Time Spent per Activity")
    st.caption("Activities ranked by average time invested")

    for activity in activities[:15]:
        name = activity.get("activity", "Unknown")
        avg_time = activity.get("avg_time_min", 0)
        median_time = activity.get("median_time_min", 0)
        attempts = activity.get("attempts", 0)
        avg_score = activity.get("avg_score")

        if avg_time > 30:
            color = "#A855F7"
        elif avg_time > 15:
            color = "#F59E0B"
        else:
            color = "#4F8EF7"

        score_display = f"{avg_score:.1f}% avg score" if avg_score else "No scores"

        st.markdown(f"""
        <div style="background: #12172A; border-left: 3px solid {color};
                    border-radius: 6px; padding: 10px; margin-bottom: 6px;">
            <div style="font-size: 14px; font-weight: 600; color: #E8EAF6; margin-bottom: 4px;">
                {name[:60]}{'...' if len(name) > 60 else ''}
            </div>
            <div style="display: flex; justify-content: space-between; font-size: 11px; color: #6B7494;">
                <span>Avg: {avg_time:.1f} min · Median: {median_time:.1f} min</span>
                <span>{attempts} attempts · {score_display}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)


def correlation_display(correlation_data: Dict):
    """Display engagement-performance correlation."""
    if not correlation_data:
        st.info("Insufficient data for correlation analysis")
        return

    corr = correlation_data.get("correlation", 0)
    sig = correlation_data.get("significance", "")
    interp = correlation_data.get("interpretation", "")

    # Color based on correlation strength
    if abs(corr) > 0.7:
        color = "#22C55E"
    elif abs(corr) > 0.4:
        color = "#F59E0B"
    else:
        color = "#6B7494"

    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #12172A 0%, #1A1D30 100%);
                border: 2px solid {color}; border-radius: 12px; padding: 20px; margin-bottom: 16px;">
        <div style="font-size: 12px; font-weight: 700; color: {color}; letter-spacing: 0.08em;
                    text-transform: uppercase; margin-bottom: 8px;">
            Engagement ↔ Performance Correlation
        </div>
        <div style="font-size: 36px; font-weight: 700; color: #E8EAF6; margin-bottom: 8px;">
            {corr:.3f}
        </div>
        <div style="font-size: 11px; color: #8B92B0; margin-bottom: 12px;">
            <span style="color: {color}; font-weight: 600;">{sig.upper()}</span> correlation
        </div>
        <div style="font-size: 10px; color: #6B7494; line-height: 1.4;">
            {interp}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Scatter plot
    data = correlation_data.get("data", [])
    if data and len(data) > 1:
        import plotly.express as px
        import pandas as pd

        df = pd.DataFrame(data)
        fig = px.scatter(
            df,
            x="engagement",
            y="performance",
            title="Engagement vs Performance",
            labels={"engagement": "Statement Count", "performance": "Average Score"},
            trendline="ols"
        )
        fig.update_layout(
            plot_bgcolor="#0F1117",
            paper_bgcolor="#0F1117",
            font_color="#E8EAF6",
            height=300
        )
        st.plotly_chart(fig, use_container_width=True)


def baseline_comparison_card(baseline_comparison: Dict):
    """Display comparison to minimum baseline (most efficient successful student)."""
    if not baseline_comparison:
        st.info("No baseline data available")
        return

    student = baseline_comparison.get("student", {})
    baseline = baseline_comparison.get("baseline", {})
    delta = baseline_comparison.get("delta", {})
    assessment = baseline_comparison.get("assessment", "")

    st.subheader("Minimum Baseline Comparison")
    st.caption("Comparing to most efficient successful student (minimum effort, maximum results)")

    # Create 3 columns for metrics
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            label="Statements",
            value=student.get("statements", 0),
            delta=f"{delta.get('statements_pct', 0):+.1f}%",
            help=f"Baseline: {baseline.get('statements', 0)} statements"
        )

    with col2:
        # Format time durations
        student_time = student.get('time_spent', 0)
        baseline_time = baseline.get('time_spent', 0)
        delta_time = delta.get('time_spent', 0)

        st.metric(
            label="Time Spent",
            value=format_duration(student_time),
            delta=f"{delta_time:+.1f} min",
            help=f"Baseline: {format_duration(baseline_time)}"
        )

    with col3:
        st.metric(
            label="Performance",
            value=f"{student.get('avg_score', 0):.1f}%",
            delta=f"{delta.get('performance_gap', 0):+.1f}",
            help=f"Baseline: {baseline.get('avg_score', 0):.1f}%"
        )

    if "Efficient" in assessment:
        st.success(f"**Assessment:** {assessment}")
    elif "Critical" in assessment:
        st.error(f"**Assessment:** {assessment}")
    elif "Struggling" in assessment or "Over-working" in assessment:
        st.warning(f"**Assessment:** {assessment}")
    else:
        st.info(f"**Assessment:** {assessment}")