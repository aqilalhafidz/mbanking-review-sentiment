"""Plotly chart builders for the dashboard pages."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .ui import APP_LABELS


APP_COLORS = {"BRImo": "#315a87", "Livin_by_Mandiri": "#db6675", "myBCA": "#318969", "wondr_by_BNI": "#d69a39"}
SENTIMENT_COLORS = {"negative": "#cf4d55", "neutral": "#929baa", "positive": "#3b9672"}
SENTIMENT_LABELS = {"negative": "Negatif", "neutral": "Netral", "positive": "Positif"}
METHOD_COLORS = {"predicted": "#315f96", "soft": "#d889a8", "corrected": "#369271"}
SUBSET_COLORS = {"all versioned reviews": "#315f96", "reliable versions only": "#cf4d55", "without blocking/boycott topic": "#369271"}
SUBSET_LABELS = {"all versioned reviews": "Semua Data", "reliable versions only": "Hanya Versi Reliabel", "without blocking/boycott topic": "Tanpa Topik Boikot"}


def _layout(fig: go.Figure, height: int = 390) -> go.Figure:
    fig.update_layout(
        height=height, margin=dict(l=28, r=24, t=26, b=36),
        font=dict(family="Inter, Segoe UI, sans-serif", color="#21344b", size=12),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#edf1f5", zerolinecolor="#d8e0e8")
    fig.update_yaxes(showgrid=True, gridcolor="#edf1f5", zerolinecolor="#d8e0e8")
    return fig


def sentiment_composition_figure(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    app_names = [APP_LABELS.get(a, a) for a in df["app_name"]]
    for sentiment, label in [("negative", "Negatif"), ("neutral", "Netral"), ("positive", "Positif")]:
        values = df[f"{sentiment}_corrected"].fillna(0)
        fig.add_trace(go.Bar(
            name=label, y=app_names, x=values, orientation="h", marker_color=SENTIMENT_COLORS[sentiment],
            text=[f"{v:.1%}" if v >= .055 else "" for v in values], textposition="inside",
            hovertemplate=f"{label}<br>%{{y}}<br>Proporsi: %{{x:.1%}}<extra></extra>",
        ))
    fig.update_layout(barmode="stack", xaxis=dict(range=[0, 1], tickformat=".0%", title="Proporsi sentimen terkoreksi"), yaxis=dict(autorange="reversed"), height=max(340, len(df) * 90))
    return _layout(fig, max(340, len(df) * 90))


def estimation_audit_figure(df: pd.DataFrame, sentiment: str) -> go.Figure:
    fig = go.Figure()
    for method in ["predicted", "soft", "corrected"]:
        values = df[f"{sentiment}_{method}"].fillna(0)
        options = {}
        if method == "corrected" and {f"{sentiment}_ci_low", f"{sentiment}_ci_high"}.issubset(df.columns):
            options["error_y"] = dict(
                type="data", symmetric=False,
                array=(df[f"{sentiment}_ci_high"] - values).clip(lower=0),
                arrayminus=(values - df[f"{sentiment}_ci_low"]).clip(lower=0), visible=True, thickness=1.4, width=4,
            )
        fig.add_bar(name=method.capitalize(), x=[APP_LABELS.get(a, a) for a in df["app_name"]], y=values, marker_color=METHOD_COLORS[method], **options)
    fig.update_layout(barmode="group", yaxis=dict(tickformat=".0%", title="Proporsi sentimen"), height=370)
    return _layout(fig, 370)


def monthly_trend_figure(df: pd.DataFrame, sentiment: str = "negative") -> go.Figure:
    apps = [app for app in df["app_name"].drop_duplicates().tolist()]
    rows, cols = 2, 2
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=[APP_LABELS.get(a, a) for a in apps[:4]], vertical_spacing=.17, horizontal_spacing=.1)
    for i, app in enumerate(apps[:4]):
        r, c = divmod(i, cols)
        r += 1
        c += 1
        group = df[df["app_name"] == app].sort_values("review_month")
        color = APP_COLORS.get(app, "#315f96")
        x = group["review_month"].astype(str)
        lower = group[f"{sentiment}_ci_low"].fillna(group[f"{sentiment}_corrected"])
        upper = group[f"{sentiment}_ci_high"].fillna(group[f"{sentiment}_corrected"])
        fig.add_trace(go.Scatter(x=x, y=upper, mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip", legendgroup=app), row=r, col=c)
        fig.add_trace(go.Scatter(x=x, y=lower, mode="lines", line=dict(width=0), fill="tonexty", fillcolor=f"rgba(49,95,150,.12)", name="CI 95%", showlegend=(i == 0), hoverinfo="skip", legendgroup="ci"), row=r, col=c)
        fig.add_trace(go.Scatter(x=x, y=group[f"{sentiment}_corrected"], mode="lines+markers", line=dict(color=color, width=2), marker=dict(color=color, size=7), name=APP_LABELS.get(app, app), showlegend=False, customdata=np.c_[group["n"], group["mom_neg_delta"], group["is_spike"], group["thin_cell"]], hovertemplate=f"%{{x}}<br>{sentiment.title()}: %{{y:.1%}}<br>Ulasan: %{{customdata[0]:,.0f}}<br>Delta negatif MoM: %{{customdata[1]:+.1%}}<br>Lonjakan negatif: %{{customdata[2]}}<br>Thin cell: %{{customdata[3]}}<extra></extra>"), row=r, col=c)
        for _, point in group.iterrows():
            symbol = "star-open" if bool(point["is_spike"]) and bool(point["thin_cell"]) else "star" if bool(point["is_spike"]) else "circle-open" if bool(point["thin_cell"]) else None
            if symbol:
                fig.add_trace(go.Scatter(x=[str(point["review_month"])], y=[point[f"{sentiment}_corrected"]], mode="markers", marker=dict(symbol=symbol, size=12, color=color, line=dict(width=1.5)), showlegend=False, hoverinfo="skip"), row=r, col=c)
    fig.update_yaxes(tickformat=".0%", title_text=f"Sentimen {SENTIMENT_LABELS.get(sentiment, sentiment)} terkoreksi")
    fig.update_layout(height=540, margin=dict(l=45, r=15, t=58, b=34), legend=dict(orientation="h", y=1.08, x=0))
    return _layout(fig, 540)


def boycott_sensitivity_figure(summary: pd.DataFrame) -> go.Figure:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=.22, row_heights=[.66, .34], subplot_titles=["Proporsi negatif tertimbang", "Selisih dampak topik"])
    apps = [APP_LABELS.get(a, a) for a in summary["app_name"]]
    fig.add_bar(name="Dengan Topik Boikot", x=apps, y=summary["neg_with_boycott"], marker_color="#203a5f", row=1, col=1)
    fig.add_bar(name="Tanpa Topik Boikot", x=apps, y=summary["neg_without_boycott"], marker_color="#e29ab2", row=1, col=1)
    fig.add_bar(name="Dampak", x=apps, y=summary["weighted_diff"], marker_color="#dc8b37", showlegend=False, row=2, col=1)
    fig.update_layout(barmode="group", height=490, margin=dict(l=35, r=20, t=50, b=35))
    fig.update_yaxes(tickformat=".0%", row=1, col=1)
    fig.update_yaxes(tickformat="+.1%", row=2, col=1)
    return _layout(fig, 490)


def version_timeline_figure(df: pd.DataFrame) -> go.Figure:
    apps = df["app_name"].drop_duplicates().tolist()[:4]
    fig = make_subplots(rows=2, cols=2, subplot_titles=[APP_LABELS.get(a, a) for a in apps], vertical_spacing=.17, horizontal_spacing=.1)
    for i, app in enumerate(apps):
        r, c = divmod(i, 2)
        r += 1
        c += 1
        group = df[df["app_name"] == app].copy().sort_values("first_date")
        fig.add_trace(go.Scatter(
            x=group["first_date"], y=group["negative_share"], mode="markers", name=APP_LABELS.get(app, app),
            marker=dict(size=np.clip(np.sqrt(group["n"]), 5, 22), color=APP_COLORS.get(app, "#315f96"), opacity=.75, line=dict(color="white", width=1)),
            error_y=dict(type="data", symmetric=False, array=(group["ci_high"] - group["negative_share"]).clip(lower=0), arrayminus=(group["negative_share"] - group["ci_low"]).clip(lower=0), thickness=1, width=2),
            customdata=list(zip(group["app_version"].astype(str), group["n"].astype(float))), hovertemplate="Versi %{customdata[0]}<br>%{x|%d %b %Y}<br>Keluhan: %{y:.1%}<br>Ulasan: %{customdata[1]:,.0f}<extra></extra>",
        ), row=r, col=c)
        if len(group) >= 2:
            ordinal = group["first_date"].map(pd.Timestamp.toordinal).to_numpy(dtype=float)
            coefficient = np.polyfit(ordinal, group["negative_share"].to_numpy(dtype=float), 1)
            fig.add_trace(go.Scatter(x=group["first_date"], y=np.polyval(coefficient, ordinal), mode="lines", line=dict(color="#59697a", dash="dash", width=1.5), showlegend=False, hoverinfo="skip"), row=r, col=c)
    fig.update_yaxes(tickformat=".0%", title_text="Negatif share")
    fig.update_layout(height=550, margin=dict(l=40, r=18, t=58, b=35), showlegend=False)
    return _layout(fig, 550)


def version_leaderboard_figure(worst: pd.DataFrame, best: pd.DataFrame, order="Keluhan") -> go.Figure:
    worst_part = worst.copy()
    worst_part["group"] = "Terburuk"
    best_part = best.copy()
    best_part["group"] = "Terbaik"
    if order == "Kronologis":
        worst_part = worst_part.sort_values(["first_date", "app_name"])
        best_part = best_part.sort_values(["first_date", "app_name"])
    else:
        worst_part = worst_part.sort_values(["negative_share", "n"], ascending=[False, False])
        best_part = best_part.sort_values(["negative_share", "n"], ascending=[True, False])
    both = pd.concat([worst_part, best_part], ignore_index=True)
    both["label"] = both.apply(lambda r: f"{APP_LABELS.get(r['app_name'], r['app_name'])} · v{r['app_version']}", axis=1)
    colors = ["#cf4d55" if x == "Terburuk" else "#369271" for x in both["group"]]
    fig = go.Figure(go.Bar(
        x=both["negative_share"], y=both["label"], orientation="h", marker_color=colors,
        error_x=dict(type="data", symmetric=False, array=(both["ci_high"] - both["negative_share"]).clip(lower=0), arrayminus=(both["negative_share"] - both["ci_low"]).clip(lower=0), thickness=1.2, width=3),
        customdata=np.c_[both["n"], both["active_days"]], hovertemplate="%{y}<br>Keluhan: %{x:.1%}<br>CI 95%<br>Ulasan: %{customdata[0]:,.0f}<br>Hari aktif: %{customdata[1]}<extra></extra>",
    ))
    fig.update_layout(xaxis=dict(tickformat=".0%", title="Tingkat keluhan"), yaxis=dict(autorange="reversed"), height=max(390, min(680, 28 * len(both))))
    return _layout(fig, max(390, min(680, 28 * len(both))))


def version_duration_figure(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for app, group in df.groupby("app_name"):
        fig.add_trace(go.Scatter(x=group["active_days"], y=group["negative_share"], mode="markers", name=APP_LABELS.get(app, app), marker=dict(size=np.clip(np.sqrt(group["n"]), 5, 20), color=APP_COLORS.get(app, "#315f96"), opacity=.75), customdata=list(zip(group["app_version"].astype(str), group["n"].astype(float))), hovertemplate="Versi %{customdata[0]}<br>Hari aktif: %{x}<br>Keluhan: %{y:.1%}<br>Ulasan: %{customdata[1]:,.0f}<extra></extra>"))
    fig.update_layout(xaxis_title="Durasi aktif (hari)", yaxis=dict(tickformat=".0%", title="Negative share"), height=440)
    return _layout(fig, 440)


def odds_ratio_forest_figure(df: pd.DataFrame, order: str = "Odds Ratio Terendah") -> go.Figure:
    frame = df.copy()
    frame["scenario_label"] = frame["subset"].map(SUBSET_LABELS).fillna(frame["subset"])
    frame["app_label"] = frame["app_name"].map(APP_LABELS).fillna(frame["app_name"])
    frame["label"] = frame["app_label"] + " · " + frame["scenario_label"]
    frame = frame.sort_values("odds_ratio_per_version", ascending=order == "Odds Ratio Terendah")
    fig = go.Figure()
    for subset, group in frame.groupby("subset", sort=False):
        color = SUBSET_COLORS.get(subset, "#64748b")
        fig.add_trace(go.Scatter(
            x=group["odds_ratio_per_version"], y=group["label"], mode="markers", name=SUBSET_LABELS.get(subset, subset),
            marker=dict(size=10, color=color),
            error_x=dict(type="data", symmetric=False, array=(group["or_ci_high"] - group["odds_ratio_per_version"]).clip(lower=0), arrayminus=(group["odds_ratio_per_version"] - group["or_ci_low"]).clip(lower=0), thickness=1.4, width=4),
            customdata=np.c_[group["corr_version_month"], group["reviews"], group["versions"]],
            hovertemplate="OR %{x:.3f}<br>%{y}<br>Korelasi r=%{customdata[0]:.3f}<br>Ulasan %{customdata[1]:,.0f}<br>Versi %{customdata[2]}<extra></extra>",
        ))
    fig.add_vline(x=1, line_dash="dash", line_color="#283b50", annotation_text="OR = 1,0", annotation_position="top")
    fig.update_layout(xaxis_title="Odds ratio per versi (95% CI)", yaxis_title="", height=max(390, 62 * len(frame)))
    return _layout(fig, max(390, 62 * len(frame)))


def chronology_correlation_figure(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for subset, group in df.groupby("subset", sort=False):
        fig.add_bar(name=SUBSET_LABELS.get(subset, subset), x=[APP_LABELS.get(a, a) for a in group["app_name"]], y=group["corr_version_month"], marker_color=SUBSET_COLORS.get(subset, "#64748b"), text=[f"{x:.2f}" if pd.notna(x) else "—" for x in group["corr_version_month"]], textposition="outside", cliponaxis=False)
    fig.update_layout(barmode="group", yaxis=dict(range=[-1, 1], title="Korelasi versi-bulan (r)"), height=420)
    return _layout(fig, 420)
