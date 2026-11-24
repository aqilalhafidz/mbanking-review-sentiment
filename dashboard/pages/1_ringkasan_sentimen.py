import pandas as pd
import streamlit as st

from utils.charts import estimation_audit_figure, sentiment_composition_figure
from utils.data_loader import compute_page1_features, load_sentiment_by_app
from utils.ui import (
    APP_LABELS, app_filter, chart_card, fmt_int, fmt_num, fmt_pct, html_table,
    inject_css, insight_row, kpi_row, page_header,
)


inject_css()
header, filters = st.columns([1.2, 1.8], vertical_alignment="bottom")
with header:
    page_header("Ringkasan Eksekutif & Kalibrasi Sentimen", "Ringkasan kondisi sentimen makro empat aplikasi m-banking dan dampak koreksi bias (kalibrasi) terhadap klasifikasi mentah model.")
with filters:
    app_col, sort_col, focus_col = st.columns([1.2, 1.05, .8])
    with app_col:
        apps = app_filter()
    with sort_col:
        sort_by = st.selectbox("Urutkan Grafik", ["negative_corrected", "positive_corrected", "net_sentiment_score", "n"], format_func={"negative_corrected": "Sentimen Negatif Tertinggi", "positive_corrected": "Sentimen Positif Tertinggi", "net_sentiment_score": "Net Sentiment Score", "n": "Total Ulasan"}.get)
    with focus_col:
        raw = load_sentiment_by_app()
        available = [s for s in ["negative", "neutral", "positive"] if all(f"{s}_{m}" in raw for m in ["predicted", "soft", "corrected", "ci_low", "ci_high"])]
        if len(available) < 3:
            missing = [s.title() for s in ["negative", "neutral", "positive"] if s not in available]
            st.info(f"Fokus audit dinonaktifkan untuk segmen yang kolomnya tidak tersedia: {', '.join(missing)}.")
        sentiment = st.segmented_control("Fokus Audit", options=available, default="negative" if "negative" in available else available[0], format_func={"negative": "Negatif", "neutral": "Netral", "positive": "Positif"}.get, key="p1_focus") if available else None
        if sentiment is None:
            st.warning("Kolom distribusi untuk fokus audit tidak tersedia pada CSV.")
            st.stop()

df, kpis, insights = compute_page1_features(raw, apps, sort_by)
if df.empty:
    st.warning("Tidak ada ringkasan aplikasi untuk pilihan filter ini.")
    st.stop()

top_polarization = df.loc[df["polarization_ratio"].idxmax()]
kpi_row([
    ("Total Sampel Ulasan", fmt_int(kpis["total_ulasan"]), kpis["sub_total_ulasan"], "blue"),
    ("Tingkat Negatif Terkalibrasi", fmt_num(kpis["tingkat_negatif_terkalibrasi"], 1) + "%", kpis["sub_tingkat_negatif"], "negative"),
    ("Net Sentiment Score (NSS)", f"{fmt_num(kpis['net_sentiment_score'], 1)}", f"Skala −100 s.d. +100 · Polarisasi tertinggi: {APP_LABELS.get(top_polarization['app_name'], top_polarization['app_name'])} ({fmt_num(top_polarization['polarization_ratio'], 2)})", "positive" if kpis["net_sentiment_score"] > 0 else "negative"),
    ("Pergeseran Bias Kalibrasi", f"{fmt_num(kpis['pergeseran_bias_abs_rata2'], 2)} pp", "Rata-rata |corrected − predicted| untuk sentimen negatif", "default"),
])

left, right = st.columns([1.15, 1])
with left:
    chart_card("Perbandingan Proporsi Sentimen Terkalibrasi (100% Horizontal Stacked Bar)", sentiment_composition_figure(df), "Proporsi corrected per aplikasi; bar horizontal berjumlah 100%.")
with right:
    sentiment_label = {"negative": "Negatif", "neutral": "Netral", "positive": "Positif"}[sentiment]
    chart_card(f"Audit Metode Estimasi (Tingkat Sentimen {sentiment_label}) · Predicted vs Soft vs Corrected", estimation_audit_figure(df, sentiment), "Error bar 95% CI ditampilkan untuk seri corrected.")

insight_row([
    ("Keluhan Tertinggi", insights["1_keluhan_tertinggi"]),
    ("Dampak Kalibrasi", insights["2_dampak_kalibrasi"]),
    ("Presisi Estimasi", insights["3_presisi_estimasi"]),
])

st.markdown("### Eksplorasi Data Detail · `sentiment_by_app.csv`")
detail = df.copy()
detail["Aplikasi"] = detail["app_name"].map(lambda x: APP_LABELS.get(x, x))
detail["95% CI Negatif"] = detail.apply(lambda r: f"{fmt_pct(r['negative_ci_low'])} – {fmt_pct(r['negative_ci_high'])}", axis=1)
detail["NSS"] = detail["net_sentiment_score"].map(lambda x: fmt_num(x, 1))
detail = detail.rename(columns={"n": "Total Ulasan (n)", "negative_corrected": "Sentimen Negatif (Corrected)", "neutral_corrected": "Sentimen Netral (Corrected)", "positive_corrected": "Sentimen Positif (Corrected)", "polarization_ratio": "Polarization Ratio"})
detail["Total Ulasan (n)"] = detail["Total Ulasan (n)"].map(fmt_int)
detail["Sentimen Netral (Corrected)"] = detail["Sentimen Netral (Corrected)"].map(fmt_pct)
detail["Polarization Ratio"] = detail["Polarization Ratio"].map(lambda x: fmt_num(x, 2))
columns = ["Aplikasi", "Total Ulasan (n)", "Sentimen Negatif (Corrected)", "95% CI Negatif", "Sentimen Netral (Corrected)", "Sentimen Positif (Corrected)", "NSS"]
if "polarization_ratio" in df:
    columns.append("Polarization Ratio")
spec = {
    "columns": columns,
    "progress": {"Sentimen Negatif (Corrected)": {"color": "#cf4d55"}, "Sentimen Positif (Corrected)": {"color": "#369271"}},
    "formatters": {"Sentimen Negatif (Corrected)": fmt_pct, "Sentimen Positif (Corrected)": fmt_pct},
    "max_height": 390,
}
download_col, _ = st.columns([1, 3])
with download_col:
    st.download_button("Unduh CSV", data=df.to_csv(index=False).encode("utf-8-sig"), file_name="sentiment_by_app_terfilter.csv", mime="text/csv", key="p1_download")
html_table(detail, spec)
st.markdown('<div class="method-note">Koreksi prevalensi memakai inversi matriks kebingungan dengan kendala non-negativitas. CI 95% berasal dari bootstrap test set dengan resampling kelompok teks; karena periode ulasan merupakan sensus data terkumpul, CI ini menggambarkan ketidakpastian classifier saja. Share netral memiliki interval lebar dan perlu dibandingkan secara hati-hati. NSS = (positif terkoreksi − negatif terkoreksi) × 100.</div>', unsafe_allow_html=True)
