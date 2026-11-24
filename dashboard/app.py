"""Streamlit entry point and multipage router."""

from pathlib import Path

import streamlit as st

from utils.data_loader import dataset_status
from utils.ui import inject_css


st.set_page_config(
    page_title="Sentimen M-Banking Indonesia",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_css()

dashboard_dir = Path(__file__).resolve().parent
with st.sidebar:
    st.markdown("<div class='brand-title'>SENTIMEN M-BANKING</div><div class='brand-subtitle'>Play Store m-Banking Sentiment</div>", unsafe_allow_html=True)

pages = [
    st.Page("pages/1_ringkasan_sentimen.py", title="Ringkasan Eksekutif & Kalibrasi", default=True),
    st.Page("pages/2_dinamika_boikot.py", title="Dinamika Bulanan & Boikot"),
    st.Page("pages/3_audit_versi.py", title="Audit Kualitas Versi"),
    st.Page("pages/4_tren_lintas_versi.py", title="Pemodelan Tren (Odds Ratio) & Ekspor"),
]
page = st.navigation(pages, position="sidebar")

with st.sidebar:
    with st.expander("Metodologi & Status Dataset", expanded=False):
        st.markdown(
            "Estimasi *corrected* mengoreksi prevalensi label terhadap error classifier pada test set. "
            "CI 95% merepresentasikan ketidakpastian klasifikasi berdasarkan bootstrap teks berkelompok, "
            "bukan variasi sampling ulasan populasi. Ulasan mencakup periode Jan–Agu 2026; pembahasan topik "
            "Livin' terbatas pada ulasan negatif yang diperiksa. Perbandingan versi bersifat asosiasi karena "
            "versi terpasang berkorelasi dengan waktu dan tidak mengidentifikasi dampak pembaruan."
        )
        st.markdown("**Sumber CSV**")
        for item in dataset_status():
            icon = "✅" if item["ok"] else "⚠️"
            st.caption(f"{icon} `{item['file']}` · {item['message']}")

page.run()
