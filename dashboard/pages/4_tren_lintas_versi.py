import pandas as pd
import streamlit as st

from utils.charts import chronology_correlation_figure, odds_ratio_forest_figure
from utils.data_loader import (
    compute_page4_features, load_blocking_topic_comparison, load_sentiment_by_app,
    load_sentiment_by_app_month, load_sentiment_monthly, load_version_table, load_version_trend,
)
from utils.ui import (
    APP_LABELS, app_filter, chart_card, fmt_int, fmt_num, fmt_pct, html_table,
    inject_css, insight_row, kpi_row, page_header,
)


SUBSETS = ["all versioned reviews", "reliable versions only", "without blocking/boycott topic"]
SUBSET_LABELS = {"all versioned reviews": "Semua Data", "reliable versions only": "Hanya Versi Reliabel", "without blocking/boycott topic": "Tanpa Topik Boikot"}

inject_css()
header, filters = st.columns([1.15, 2.2], vertical_alignment="bottom")
with header:
    page_header("Pemodelan Tren Lintas Versi & Ekspor Data", "Mengukur perubahan odds keluhan per urutan rilis dan membandingkan skenario ketahanan.")
with filters:
    app_col, scenario_col, order_col = st.columns([1.05, 1.35, .85])
    with app_col:
        apps = app_filter()
    with scenario_col:
        selected_subsets = st.multiselect("Filter Skenario", SUBSETS, default=SUBSETS, format_func=lambda s: SUBSET_LABELS[s], key="p4_subsets")
    with order_col:
        order = st.radio("Urutan", ["Odds Ratio Terendah", "Odds Ratio Tertinggi"], key="p4_order")
if not selected_subsets:
    st.warning("Pilih setidaknya satu skenario untuk melihat hasil model.")
    st.stop()

df_trend = load_version_trend()
df_app = load_sentiment_by_app()
df_monthly = load_sentiment_monthly()
df_version = load_version_table()
df, robustness, benchmark, kpis, insights = compute_page4_features(df_trend, df_app, df_monthly, df_version, apps, selected_subsets)
if df.empty:
    st.warning("Tidak ada hasil regresi untuk aplikasi dan skenario yang dipilih.")
    st.stop()
df = df.sort_values("odds_ratio_per_version", ascending=order == "Odds Ratio Terendah").reset_index(drop=True)
best = df.loc[df["odds_ratio_per_version"].idxmin()]
reliable_count = int(df_version[df_version["app_name"].isin(apps)].groupby("app_name")["reliable"].sum().sum())

kpi_row([
    ("Perbaikan Versi Terkuat", f"OR {best['odds_ratio_per_version']:.3f}", f"{APP_LABELS.get(best['app_name'], best['app_name'])} · {best['pct_odds_change_per_version']:+.2f}% per versi", "positive" if best["odds_ratio_per_version"] < 1 else "negative"),
    ("Rata-rata Korelasi Kronologis", fmt_num(kpis["rata2_korelasi_kronologis"], 3), "Rata-rata corr_version_month pada subset rujukan", "blue"),
    ("Konsistensi Robustness Model", kpis["konsistensi_robustness"], "Aplikasi dengan arah signifikansi sama pada ketiga skenario", "default"),
    ("Total Sampel & Versi Teruji", kpis["total_sampel_teruji"], f"{fmt_int(reliable_count)} versi reliabel terdeteksi · {kpis['sub_sampel_teruji']}", "blue"),
])

left, right = st.columns([1.25, 1])
with left:
    chart_card("Forest Plot Odds Ratio Multi-Skenario (95% CI)", odds_ratio_forest_figure(df, order), "OR < 1 mengarah pada odds label negatif yang lebih rendah di urutan versi lebih baru; CI yang melintasi 1,0 tidak signifikan pada taraf 5%.")
with right:
    chart_card("Audit Korelasi Kronologis (corr_version_month)", chronology_correlation_figure(df), "Korelasi tinggi berarti urutan versi dan waktu sulit dipisahkan.")

insight_row([
    ("Arah Tren Utama Lintas Bank", insights["1_arah_tren_utama"]),
    ("Ketahanan Terhadap Isu Boikot", insights["2_ketahanan_boikot"]),
    ("Validitas Urutan Rilis", insights["3_validitas_urutan_rilis"]),
])

st.markdown("### Detail Hasil Regresi Lintas Versi · `version_trend.csv`")
detail = df.copy()
detail["Aplikasi"] = detail["app_name"].map(lambda a: APP_LABELS.get(a, a))
detail["Skenario"] = detail["subset"].map(lambda s: SUBSET_LABELS.get(s, s))
detail["Odds Ratio (OR)"] = detail["odds_ratio_per_version"].map(lambda x: fmt_num(x, 3))
detail["95% Confidence Interval (CI)"] = detail.apply(lambda r: f"{fmt_num(r['or_ci_low'], 3)} – {fmt_num(r['or_ci_high'], 3)}", axis=1)
detail["Korelasi (r)"] = detail["corr_version_month"].map(lambda x: fmt_num(x, 3))
detail["Status Signifikansi"] = detail["significance_status"]
robustness_map = robustness.set_index("app_name")["is_direction_consistent"].to_dict() if not robustness.empty else {}
detail["Konsistensi Arah"] = detail["app_name"].map(lambda app: "Konsisten" if robustness_map.get(app, False) else "Berbeda antar skenario")
detail_columns = ["Aplikasi", "Skenario", "Odds Ratio (OR)", "95% Confidence Interval (CI)", "Korelasi (r)"]
stat_column = next((col for col in ["t_statistic", "t_stat"] if col in detail.columns), None)
if stat_column:
    detail["Statistik-t"] = detail[stat_column].map(lambda x: fmt_num(x, 3))
    detail_columns.append("Statistik-t")
detail_columns.extend(["Status Signifikansi", "Konsistensi Arah"])
html_table(detail, {
    "columns": detail_columns,
    "badges": {
        "Status Signifikansi": {"mapping": {
            "Signifikan Membaik (OR < 1)": {"text": "Signifikan Membaik (CI < 1)", "kind": "significant"},
            "Signifikan Memburuk (OR > 1)": {"text": "Signifikan Memburuk (CI > 1)", "kind": "unreliable"},
            "Tidak Signifikan": {"text": "Tidak Signifikan", "kind": "not_significant"},
        }},
        "Konsistensi Arah": {"mapping": {
            "Konsisten": {"text": "Konsisten", "kind": "stable"},
            "Berbeda antar skenario": {"text": "Berbeda antar skenario", "kind": "thin"},
        }},
    },
    "max_height": 440,
})

st.markdown("### Pusat Ekspor Data")
export_cols = st.columns(4)
export_items = [
    ("Unduh Hasil Regresi (version_trend.csv)", df_trend[df_trend["app_name"].isin(apps) & df_trend["subset"].isin(selected_subsets)], "version_trend.csv", "export_trend"),
    ("Unduh Statistik Bulanan (sentiment_by_app_month.csv)", load_sentiment_by_app_month().query("app_name in @apps"), "sentiment_by_app_month.csv", "export_month"),
    ("Unduh Siklus Versi (version_table.csv)", df_version[df_version["app_name"].isin(apps)], "version_table.csv", "export_version"),
    ("Unduh Blokir Perbandingan (blocking_topic_comparison.csv)", load_blocking_topic_comparison().query("app_name in @apps"), "blocking_topic_comparison.csv", "export_blocking"),
]
for col, (label, frame, filename, key) in zip(export_cols, export_items):
    with col:
        st.download_button(label, data=frame.to_csv(index=False).encode("utf-8-sig"), file_name=filename, mime="text/csv", key=key, use_container_width=True)

st.markdown('<div class="method-note">Model tren adalah regresi logistik label negatif terprediksi terhadap urutan versi dengan efek tetap bulan. Versi reliabel memiliki ≥100 ulasan. Odds ratio dan CI berasal dari model tren di pipeline; korelasi urutan versi-bulan menunjukkan potensi confounding temporal. Versi menggambarkan versi terpasang pada reviewer, bukan versi rilis saat ulasan dibuat; hasil merupakan asosiasi dan tidak mengidentifikasi dampak update. Estimasi topik boikot memiliki keterbatasan cakupan sebagaimana dijelaskan di RESULTS.md.</div>', unsafe_allow_html=True)
