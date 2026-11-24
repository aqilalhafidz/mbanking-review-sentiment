import pandas as pd
import streamlit as st

from utils.charts import boycott_sensitivity_figure, monthly_trend_figure
from utils.data_loader import compute_page2_features, load_sentiment_monthly
from utils.ui import (
    APP_LABELS, app_filter, chart_card, fmt_int, fmt_month, fmt_num, fmt_pct,
    html_table, inject_css, insight_row, kpi_row, page_header,
)


inject_css()
header, filters = st.columns([1.15, 2.1], vertical_alignment="bottom")
with header:
    page_header("Dinamika Bulanan & Sensitivitas Topik Boikot", "Analisis tren bulanan, pola volatilitas, dan simulasi dampak penghilangan topik keluhan tertentu berbasis model.")
monthly = load_sentiment_monthly()
months = sorted(monthly.get("review_month", pd.Series(dtype=str)).dropna().astype(str).unique())
if not months:
    st.warning("Dataset bulanan tidak berisi bulan yang dapat dianalisis.")
    st.stop()
with filters:
    app_col, month_col, focus_col, thin_col = st.columns([1.05, 1.05, .82, .78])
    with app_col:
        apps = app_filter()
    with month_col:
        if len(months) > 1:
            month_range = st.select_slider("Rentang Bulan", options=months, value=(months[0], months[-1]), format_func=fmt_month, key="p2_month_range")
        else:
            st.selectbox("Rentang Bulan", months, format_func=fmt_month, disabled=True)
            month_range = (months[0], months[0])
    with focus_col:
        available = [s for s in ["negative", "neutral", "positive"] if all(f"{s}_{m}" in monthly for m in ["corrected", "ci_low", "ci_high"])]
        if len(available) < 3:
            missing = [s.title() for s in ["negative", "neutral", "positive"] if s not in available]
            st.info(f"Fokus analisis terbatas; kolom tidak tersedia untuk: {', '.join(missing)}.")
        focus = st.segmented_control("Fokus Analisis", available, default="negative" if "negative" in available else (available[0] if available else None), format_func={"negative": "Negatif", "neutral": "Netral", "positive": "Positif"}.get, key="p2_focus") if available else None
    with thin_col:
        hide_thin = st.checkbox("Sembunyikan thin_cell", value=False, key="p2_hide_thin")

if focus is None:
    st.warning("Kolom sentimen untuk grafik bulanan tidak tersedia.")
    st.stop()
df, app_volatility, boycott_summary, kpis, insights = compute_page2_features(monthly, apps, month_range, hide_thin_cell=hide_thin)
table_source = compute_page2_features(monthly, apps, month_range, hide_thin_cell=False)[0]
if df.empty:
    st.warning("Tidak ada observasi pada aplikasi, rentang bulan, dan pilihan thin_cell ini.")
    st.stop()

peak_row = df.loc[df["negative_corrected"].idxmax()]
peak_detail = f"{fmt_month(peak_row['review_month'])} · {APP_LABELS.get(peak_row['app_name'], peak_row['app_name'])}"
valid_mom = df.dropna(subset=["mom_neg_delta"])
if valid_mom.empty:
    delta_detail = "Belum ada bulan pembanding"
else:
    max_mom_row = valid_mom.loc[valid_mom["mom_neg_delta"].idxmax()]
    delta_detail = f"{APP_LABELS.get(max_mom_row['app_name'], max_mom_row['app_name'])} · {fmt_month(max_mom_row['review_month'])}"
kpi_row([
    ("Puncak Keluhan Bulanan", fmt_num(kpis["puncak_keluhan_bulanan"], 1) + "%", peak_detail, "negative"),
    ("Rata-rata & Volatilitas", kpis["rata2_dan_volatilitas"], "Rata-rata bulanan ± simpangan baku", "blue"),
    ("Lonjakan Bulan-ke-Bulan", kpis["lonjakan_mom_terbesar"], delta_detail or "Delta tertinggi dibanding bulan sebelumnya", "default"),
    ("Dampak Maks Topik Boikot", kpis["dampak_maks_boikot"], kpis["sub_boikot"], "blue"),
])

left, right = st.columns([1.25, 1])
with left:
    chart_card("Tren Bulanan per Aplikasi (95% CI & thin_cell)", monthly_trend_figure(df, focus), "Panel terisi mengikuti aplikasi terpilih. Bintang = lonjakan negatif MoM (> 5 poin); lingkaran kosong = thin_cell.")
with right:
    chart_card("Simulasi Sensitivitas Topik Boikot/Blokir", boycott_sensitivity_figure(boycott_summary), "Estimasi tertimbang volume: negatif dengan topik dibanding negatif tanpa topik. Perubahan populasi topik bukan eksperimen kausal.")

insight_row([
    ("Lonjakan Terbesar (MoM)", insights["1_lonjakan_terbesar"]),
    ("Puncak Keluhan Bulanan", insights["2_puncak_keluhan"]),
    ("Sensitivitas Topik Boikot", insights["3_sensitivitas_boikot"]),
])

st.markdown("### Eksplorasi Data Bulanan · `sentiment_monthly_merged.csv`")
table_mode = st.segmented_control("Tampilan Tabel", ["Semua Data", "Hanya Lonjakan (is_spike)", "Sparsity Filter"], default="Semua Data", key="p2_table_mode")
table = table_source.copy()
if table_mode == "Hanya Lonjakan (is_spike)":
    table = table[table["is_spike"]]
elif table_mode == "Sparsity Filter":
    table = table[table["thin_cell"].fillna(False)]
elif hide_thin:
    table = table[~table["thin_cell"].fillna(False)]
table["Bulan"] = table["review_month"].map(fmt_month)
table["Aplikasi"] = table["app_name"].map(lambda x: APP_LABELS.get(x, x))
table["Volume Ulasan"] = table["n"].map(fmt_int)
table["95% Confidence Interval"] = table.apply(lambda r: f"{fmt_pct(r['negative_ci_low'])} – {fmt_pct(r['negative_ci_high'])}", axis=1)
table["Status thin_cell"] = table["thin_cell"].map(lambda x: "Min (Sparse)" if bool(x) else "Stabil (Normal)")
table["Dampak"] = table["difference"].map(lambda x: fmt_pct(x, 2))
table = table.rename(columns={"negative_corrected": "Tingkat Negatif", "negative_without": "% Estimasi Tanpa Boikot"})
table["Status thin_cell"] = table["Status thin_cell"].astype(str)
spec = {
    "columns": ["Bulan", "Aplikasi", "Volume Ulasan", "Tingkat Negatif", "% Estimasi Tanpa Boikot", "Dampak", "95% Confidence Interval", "Status thin_cell"],
    "progress": {"Tingkat Negatif": {"color": "#cf4d55"}, "% Estimasi Tanpa Boikot": {"color": "#369271"}},
    "formatters": {"Tingkat Negatif": fmt_pct, "% Estimasi Tanpa Boikot": fmt_pct},
    "badges": {"Status thin_cell": {"kind": "neutral", "mapping": {"Stabil (Normal)": {"text": "Stabil (Normal)", "kind": "stable"}, "Min (Sparse)": {"text": "Min (Sparse)", "kind": "thin"}}}},
    "max_height": 480,
}
download_col, _ = st.columns([1, 3])
with download_col:
    st.download_button("Unduh CSV", data=table.to_csv(index=False).encode("utf-8-sig"), file_name="sentiment_monthly_merged_terfilter.csv", mime="text/csv", key="p2_download")
html_table(table, spec)
st.markdown('<div class="method-note">Perubahan bulanan dihitung sebagai MoM negatif terkoreksi dan dihitung sebelum filter rentang bulan. Penanda lonjakan memakai ambang notebook 0,05 pada skala proporsi; CI classifier bukan interval untuk perubahan jumlah ulasan. Topik boikot/blokir berasal dari pola/label topik dengan cakupan terbatas, sehingga selisihnya adalah sensitivitas deskriptif.</div>', unsafe_allow_html=True)
