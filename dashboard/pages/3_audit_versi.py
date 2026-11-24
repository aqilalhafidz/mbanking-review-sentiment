import pandas as pd
import streamlit as st

from utils.charts import version_duration_figure, version_leaderboard_figure, version_timeline_figure
from utils.data_loader import compute_page3_features, load_version_table
from utils.ui import APP_LABELS, app_filter, chart_card, fmt_int, fmt_num, fmt_pct, html_table, inject_css, insight_row, kpi_row, page_header


inject_css()
header, filters = st.columns([1.15, 2.3], vertical_alignment="bottom")
with header:
    page_header("Audit Kualitas Versi Rilis", "Membaca sebaran keluhan, coverage versi, dan ketidakpastian pada riwayat rilis.")
with filters:
    app_col, reliable_col, volume_col, order_col = st.columns([1.05, .95, 1.05, .9])
    with app_col:
        apps = app_filter()
    raw = load_version_table()
    scoped = raw[raw["app_name"].isin(apps)]
    max_reviews = max(1, int(scoped["n"].max())) if not scoped.empty else 1
    with reliable_col:
        only_reliable = st.toggle("Hanya Versi Reliabel", value=True, help="Versi reliabel ditentukan oleh n ≥ 100.", key="p3_only_reliable")
    with volume_col:
        min_default = min(5, max_reviews)
        min_reviews = st.slider("Min Volume Ulasan (n)", min_value=1, max_value=max_reviews, value=min_default, key="p3_min_reviews")
    with order_col:
        order = st.radio("Urutan", ["Kronologis", "Keluhan"], horizontal=True, key="p3_order")

df, pareto, worst, best, kpis, insights = compute_page3_features(raw, apps, only_reliable, min_reviews, search_version="", top_n=5)
if df.empty:
    st.warning("Tidak ada versi yang memenuhi pilihan aplikasi, reliabilitas, volume minimum, dan pencarian saat ini.")
    st.text_input("Cari Versi...", placeholder="Contoh: 3.7.0", key="p3_search")
    st.stop()
if order == "Kronologis":
    df = df.sort_values(["first_date", "app_name", "app_version"]).reset_index(drop=True)
    worst = worst.sort_values(["first_date", "app_name"]).reset_index(drop=True)
    best = best.sort_values(["first_date", "app_name"]).reset_index(drop=True)
else:
    df = df.sort_values(["negative_share", "n"], ascending=[False, False]).reset_index(drop=True)

per_app = scoped.groupby("app_name").size().to_dict()
coverage_detail = kpis["sub_cakupan"]
longest = kpis["sub_masa_aktif"]
kpi_row([
    ("Total Versi Terdeteksi", f"{len(scoped):,}", " · ".join(f"{APP_LABELS.get(a,a)}: {per_app.get(a,0)}" for a in apps), "blue"),
    ("Cakupan Ulasan Reliabel", kpis["cakupan_ulasan_reliabel"], coverage_detail, "positive"),
    ("Median Masa Aktif Versi", kpis["median_masa_aktif"], longest, "default"),
    ("Versi Keluhan Tertinggi", kpis["versi_keluhan_tertinggi"], kpis["sub_versi_tertinggi"], "negative"),
])

left, right = st.columns([1.25, 1])
with left:
    chart_card("Peta Riwayat Rilis 4 Aplikasi (Timeline vs Negative Share)", version_timeline_figure(df), "Bubble = volume ulasan; whisker = CI 95% Wilson. Garis putus-putus merangkum pola linear deskriptif terhadap tanggal ulasan pertama.")
with right:
    st.markdown('<div class="chart-title">Leaderboard Versi &amp; Durasi Aktif</div>', unsafe_allow_html=True)
    tab_board, tab_duration = st.tabs(["Top Versi Terburuk & Terbaik", "Durasi Aktif vs Keluhan"])
    with tab_board:
        chart_card("Top Versi Terburuk & Terbaik", version_leaderboard_figure(worst, best, order), "Leaderboard dibatasi 5 versi terburuk dan 5 terbaik berdasarkan hasil perhitungan notebook.")
    with tab_duration:
        chart_card("Durasi Aktif vs Keluhan", version_duration_figure(df), "Durasi dihitung dari tanggal ulasan pertama hingga terakhir, inklusif.")

insight_row([
    ("Konsentrasi Rilis Utama", insights["1_konsentrasi_rilis"]),
    ("Anomali Rilis Kritis", insights["2_anomali_rilis"]),
    ("Audit Kalibrasi Versi", insights["3_audit_kalibrasi_versi"]),
])

st.markdown("### Inspektor Siklus Hidup Versi · `version_table.csv`")
search_col, download_col = st.columns([1.15, 1], vertical_alignment="bottom")
with search_col:
    st.text_input("Cari Versi...", placeholder="Contoh: 3.7.0", key="p3_search")
search_version = st.session_state.get("p3_search", "").strip()
table = df.copy()
if search_version:
    table = table[table["app_version"].astype(str).str.contains(search_version, case=False, na=False)].copy()
with download_col:
    st.download_button("Unduh CSV", data=table.to_csv(index=False).encode("utf-8-sig"), file_name="version_table_terfilter.csv", mime="text/csv", key="p3_download")
table["Aplikasi"] = table["app_name"].map(lambda x: APP_LABELS.get(x, x))
table["Jumlah Ulasan (n)"] = table["n"].map(fmt_int)
table["Durasi (Hari)"] = table["active_days"].map(fmt_int)
table["95% CI"] = table.apply(lambda r: f"{fmt_pct(r['ci_low'])} – {fmt_pct(r['ci_high'])}", axis=1)
table["Status Reliabel"] = table["reliable"].map(lambda x: "Reliabel (Tervalidasi)" if bool(x) else "Tidak Reliabel (Data Sedikit)")
table["soft_hard_diff"] = table["soft_hard_diff"].map(lambda x: f"{fmt_num(x*100, 2)} pp") if "soft_hard_diff" in table else "—"
table = table.rename(columns={"app_version": "Versi", "negative_share": "Tingkat Keluhan"})
columns = ["Aplikasi", "Versi", "Jumlah Ulasan (n)", "Durasi (Hari)", "Tingkat Keluhan", "95% CI", "Status Reliabel"]
if "soft_hard_diff" in table:
    columns.append("soft_hard_diff")
html_table(table, {
    "columns": columns,
    "labels": {"soft_hard_diff": "Selisih Hard–Soft"},
    "progress": {"Tingkat Keluhan": {"color": "#cf4d55"}},
    "formatters": {"Tingkat Keluhan": fmt_pct},
    "badges": {"Status Reliabel": {"mapping": {
        "Reliabel (Tervalidasi)": {"text": "Reliabel (Tervalidasi)", "kind": "reliable"},
        "Tidak Reliabel (Data Sedikit)": {"text": "Tidak Reliabel (Data Sedikit)", "kind": "unreliable"},
    }}},
    "max_height": 500,
})
st.markdown('<div class="method-note">Versi ditandai reliabel jika memiliki sedikitnya 100 ulasan (`MIN_REVIEWS_PER_VERSION=100`). Wilson CI 95% dipakai untuk negative share per versi. Versi yang digunakan reviewer adalah versi terpasang, bukan versi terbaru saat ulasan ditulis; urutan versi dan waktu berkorelasi sehingga perbandingan tidak mengukur dampak pembaruan.</div>', unsafe_allow_html=True)
