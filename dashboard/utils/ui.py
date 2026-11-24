"""Reusable visual and interaction helpers for all dashboard pages."""

from html import escape
from pathlib import Path
import re
from typing import Any

import pandas as pd
import streamlit as st

from .data_loader import load_sentiment_by_app


APP_LABELS = {
    "BRImo": "BRImo",
    "Livin_by_Mandiri": "Livin_by_Mandiri",
    "myBCA": "myBCA",
    "wondr_by_BNI": "wondr_by_BNI",
}
MONTHS_ID = {1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "Mei", 6: "Jun", 7: "Jul", 8: "Agu", 9: "Sep", 10: "Okt", 11: "Nov", 12: "Des"}


def inject_css() -> None:
    """Read and inject the dashboard stylesheet from its own directory."""
    css_path = Path(__file__).resolve().parents[1] / "assets" / "style.css"
    try:
        css = css_path.read_text(encoding="utf-8")
    except OSError:
        st.warning("File styling dashboard tidak ditemukan; tampilan standar digunakan.")
        return
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def page_header(title: str, description: str) -> None:
    st.markdown(f'<div class="page-heading"><h1>{escape(title)}</h1><p>{escape(description)}</p></div>', unsafe_allow_html=True)


def app_filter() -> list[str]:
    """Shared app filter that survives Streamlit page widget cleanup."""
    apps = load_sentiment_by_app().get("app_name", pd.Series(dtype=str)).dropna().tolist()
    if "selected_apps" not in st.session_state:
        st.session_state["selected_apps"] = apps.copy()
    st.session_state["selected_apps"] = [a for a in st.session_state["selected_apps"] if a in apps]
    widget_key = "selected_apps_widget"
    if widget_key in st.session_state and not st.session_state[widget_key] and st.session_state["selected_apps"]:
        # Streamlit may clear a page-scoped widget key to [] on page change.
        # Restore from the persistent key, while a user-cleared selection stays empty
        # because its on_change callback also stores [] in selected_apps.
        del st.session_state[widget_key]

    def sync_apps() -> None:
        st.session_state["selected_apps"] = list(st.session_state[widget_key])

    st.multiselect(
        "Filter Aplikasi",
        options=apps,
        format_func=lambda app: APP_LABELS.get(app, app),
        default=list(st.session_state["selected_apps"]),
        key=widget_key,
        on_change=sync_apps,
        help="Filter aplikasi tersinkronisasi pada seluruh halaman.",
    )
    selected = list(st.session_state.get(widget_key, st.session_state["selected_apps"]))
    st.session_state["selected_apps"] = selected
    if not selected:
        st.warning("Pilih setidaknya satu aplikasi untuk menampilkan analisis.")
        st.stop()
    return selected


def fmt_int(value: Any) -> str:
    try:
        return f"{int(round(float(value))):,}".replace(",", ".")
    except (TypeError, ValueError, OverflowError):
        return "—"


def fmt_num(value: Any, decimals: int = 2) -> str:
    try:
        return f"{float(value):,.{decimals}f}".replace(",", "_").replace(".", ",").replace("_", ".")
    except (TypeError, ValueError, OverflowError):
        return "—"


def fmt_pct(value: Any, decimals: int = 1) -> str:
    try:
        number = float(value)
        if not pd.notna(number):
            return "—"
        return f"{fmt_num(number * 100, decimals)}%"
    except (TypeError, ValueError, OverflowError):
        return "—"


def fmt_month(value: Any) -> str:
    text = str(value)
    try:
        year, month = text.split("-")[:2]
        return f"{MONTHS_ID[int(month)]} {year}"
    except (ValueError, KeyError, IndexError):
        return text


def _display_text(value: Any) -> str:
    text = str(value)
    for raw, label in APP_LABELS.items():
        text = text.replace(raw, label)
    text = re.sub(
        r"\b(20\d{2})-(0[1-9]|1[0-2])\b",
        lambda match: f"{MONTHS_ID[int(match.group(2))]} {match.group(1)}",
        text,
    )
    text = re.sub(r"(?<=\d),(?=\d{3}(?:\D|$))", ".", text)
    return re.sub(r"(?<=\d)\.(?=\d{1,2}(?:\D|$))", ",", text)


def kpi_card(label: str, value: Any, sub: str, variant: str = "default") -> None:
    st.markdown(
        f'<div class="kpi-card kpi-{escape(variant)}"><div class="kpi-label">{escape(label)}</div>'
        f'<div class="kpi-value">{escape(_display_text(value))}</div><div class="kpi-detail">{escape(_display_text(sub))}</div></div>',
        unsafe_allow_html=True,
    )


def kpi_row(items: list[tuple[str, Any, str, str]]) -> None:
    for column, (label, value, sub, variant) in zip(st.columns(len(items)), items):
        with column:
            kpi_card(label, value, sub, variant)


def chart_card(title: str, figure: Any, caption: str | None = None) -> None:
    with st.container(border=True):
        st.markdown(f'<div class="chart-title">{escape(title)}</div>', unsafe_allow_html=True)
        st.plotly_chart(figure, use_container_width=True)
        if caption:
            st.caption(caption)


def insight_row(items: list[tuple[str, str]]) -> None:
    for index, (column, (title, text)) in enumerate(zip(st.columns(len(items)), items), start=1):
        with column:
            insight_card(index, title, text)


def insight_card(index: int, title: str, text: str) -> None:
    st.markdown(
        f'<div class="insight-card"><div class="insight-title">{int(index)}. {escape(title.upper())}</div>'
        f'<div class="insight-text">{escape(_display_text(text))}</div></div>',
        unsafe_allow_html=True,
    )


def badge(text: Any, kind: str = "neutral") -> str:
    kind_map = {
        "reliable": "badge-good", "stable": "badge-good", "significant": "badge-positive",
        "highly_significant": "badge-good", "unreliable": "badge-bad", "thin": "badge-warning",
        "sparse": "badge-warning", "not_significant": "badge-neutral", "neutral": "badge-neutral",
    }
    return f'<span class="status-badge {kind_map.get(kind, "badge-neutral")}">{escape(str(text))}</span>'


def html_table(df: pd.DataFrame, spec: dict[str, Any]) -> None:
    """Render an HTML table with optional progress, badges, formatters and scrolling.

    `spec` accepts `columns` (names or {key,label}), `progress` and `badges`
    maps keyed by column name, `formatters`, and `max_height` in pixels.
    """
    if df.empty:
        st.warning(spec.get("empty_message", "Tidak ada data untuk ditampilkan pada filter ini."))
        return
    columns = spec.get("columns", list(df.columns))
    progress = spec.get("progress", {})
    badges = spec.get("badges", {})
    formatters = spec.get("formatters", {})
    heading = []
    for item in columns:
        key = item.get("key") if isinstance(item, dict) else item
        label = item.get("label", key) if isinstance(item, dict) else spec.get("labels", {}).get(key, key)
        heading.append(f"<th>{escape(str(label))}</th>")
    rows = []
    for _, row in df.iterrows():
        cells = []
        for item in columns:
            key = item.get("key") if isinstance(item, dict) else item
            value = row.get(key, "")
            raw = value
            if key in formatters:
                value = formatters[key](raw)
            if key in progress and pd.notna(raw):
                config = progress[key] if isinstance(progress[key], dict) else {"color": progress[key]}
                try:
                    fraction = float(raw)
                    width = min(100.0, max(0.0, fraction * (100 if abs(fraction) <= 1 else 1)))
                    value = f'<div class="progress-cell"><span class="progress-label">{escape(str(value))}</span><span class="progress-track"><span class="progress-fill" style="width:{width:.1f}%;background:{escape(config.get("color", "#176b87"))}"></span></span></div>'
                except (TypeError, ValueError):
                    value = escape(str(value))
            elif key in badges:
                config = badges[key] if isinstance(badges[key], dict) else {"kind": "neutral"}
                mapping = config.get("mapping", {})
                mapped = mapping.get(str(raw), raw)
                if isinstance(mapped, dict):
                    value = badge(mapped.get("text", raw), mapped.get("kind", config.get("kind", "neutral")))
                else:
                    value = badge(mapped, config.get("kind", "neutral"))
            else:
                value = escape(str(value)) if value is not None and not pd.isna(value) else "—"
            cells.append(f"<td>{value}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    max_height = int(spec.get("max_height", 520))
    table = f'<div class="html-table-scroll" style="max-height:{max_height}px"><table><thead><tr>{"".join(heading)}</tr></thead><tbody>{"".join(rows)}</tbody></table></div>'
    st.markdown(table, unsafe_allow_html=True)


# Kept as an alias for pages from earlier versions of this dashboard.
inject_styles = inject_css
