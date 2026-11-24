"""Data loading and feature calculations for the M-Banking sentiment dashboard."""

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st


# data_loader.py lives in dashboard/utils/, while results/ is a sibling of
# dashboard/ at the repository root. Resolve from this file, never cwd.
RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"


@st.cache_data(show_spinner=False)
def load_sentiment_by_app() -> pd.DataFrame:
    return pd.read_csv(RESULTS_DIR / "sentiment_by_app.csv")


@st.cache_data(show_spinner=False)
def load_version_table() -> pd.DataFrame:
    return pd.read_csv(RESULTS_DIR / "version_table.csv")


@st.cache_data(show_spinner=False)
def load_version_trend() -> pd.DataFrame:
    return pd.read_csv(RESULTS_DIR / "version_trend.csv")


@st.cache_data(show_spinner=False)
def load_sentiment_by_app_month() -> pd.DataFrame:
    return pd.read_csv(RESULTS_DIR / "sentiment_by_app_month.csv")


@st.cache_data(show_spinner=False)
def load_blocking_topic_comparison() -> pd.DataFrame:
    return pd.read_csv(RESULTS_DIR / "blocking_topic_comparison.csv")


@st.cache_data(show_spinner=False)
def load_sentiment_monthly() -> pd.DataFrame:
    merged_path = RESULTS_DIR / "sentiment_monthly_merged.csv"
    if merged_path.exists():
        return pd.read_csv(merged_path)

    df_month = load_sentiment_by_app_month()
    df_blocking = load_blocking_topic_comparison()
    return (
        pd.merge(
            df_month,
            df_blocking[["app_name", "review_month", "n_without", "negative_without", "difference"]],
            on=["app_name", "review_month"],
            how="left",
        )
        .sort_values(["app_name", "review_month"])
        .reset_index(drop=True)
    )


def load_all_data() -> dict[str, pd.DataFrame]:
    """Load the four dashboard datasets using their cached loaders."""
    return {
        "app": load_sentiment_by_app(),
        "version": load_version_table(),
        "trend": load_version_trend(),
        "monthly": load_sentiment_monthly(),
    }


def dataset_status() -> list[dict[str, object]]:
    """Report expected source files and row counts without hiding load errors."""
    sources = [
        ("sentiment_by_app.csv", load_sentiment_by_app),
        ("sentiment_by_app_month.csv", load_sentiment_by_app_month),
        ("blocking_topic_comparison.csv", load_blocking_topic_comparison),
        ("sentiment_monthly_merged.csv", load_sentiment_monthly),
        ("version_table.csv", load_version_table),
        ("version_trend.csv", load_version_trend),
    ]
    status = []
    for filename, loader in sources:
        path = RESULTS_DIR / filename
        if not path.exists():
            if filename == "sentiment_monthly_merged.csv":
                try:
                    frame = load_sentiment_monthly()
                    status.append({"file": filename, "ok": True, "rows": len(frame), "message": f"Fallback merge · {len(frame):,} baris"})
                    continue
                except Exception as exc:
                    status.append({"file": filename, "ok": False, "rows": 0, "message": f"File tidak ditemukan dan fallback gagal: {exc}"})
                    continue
            status.append({"file": filename, "ok": False, "rows": 0, "message": "File tidak ditemukan"})
            continue
        try:
            frame = loader()
            status.append({"file": filename, "ok": True, "rows": len(frame), "message": f"{len(frame):,} baris"})
        except Exception as exc:  # report the actual CSV/data error to the sidebar operator
            status.append({"file": filename, "ok": False, "rows": 0, "message": str(exc)})
    return status


def compute_page1_features(df_app: pd.DataFrame, selected_apps=None, sort_by="negative_corrected"):
    df = df_app.copy()
    if selected_apps:
        df = df[df["app_name"].isin(selected_apps)].copy()

    df["net_sentiment_score"] = (df["positive_corrected"] - df["negative_corrected"]) * 100
    df["polarization_ratio"] = (df["positive_corrected"] + df["negative_corrected"]) / df["neutral_corrected"].replace(0, np.nan)

    for cls in ["negative", "neutral", "positive"]:
        df[f"{cls}_shift_pred"] = df[f"{cls}_corrected"] - df[f"{cls}_predicted"]
        df[f"{cls}_shift_soft"] = df[f"{cls}_corrected"] - df[f"{cls}_soft"]
        df[f"{cls}_ci_width"] = df[f"{cls}_ci_high"] - df[f"{cls}_ci_low"]
        df[f"{cls}_est_count"] = (df[f"{cls}_corrected"] * df["n"]).round().fillna(0).astype(int)

    sort_map = {
        "negative_corrected": ("negative_corrected", False),
        "positive_corrected": ("positive_corrected", False),
        "net_sentiment_score": ("net_sentiment_score", False),
        "n": ("n", False),
    }
    col_sort, asc = sort_map.get(sort_by, ("negative_corrected", False))
    df = df.sort_values(col_sort, ascending=asc).reset_index(drop=True)
    if df.empty:
        return df, {"total_ulasan": 0, "tingkat_negatif_terkalibrasi": 0, "net_sentiment_score": 0, "pergeseran_bias_abs_rata2": 0}, {
            "1_keluhan_tertinggi": "Tidak ada aplikasi yang cocok dengan filter.",
            "2_dampak_kalibrasi": "Tidak ada data untuk menghitung kalibrasi.",
            "3_presisi_estimasi": "Tidak ada data untuk menghitung interval kepercayaan.",
        }

    total_n = int(df["n"].sum())
    weights = df["n"].clip(lower=0)
    weighted_neg = float(np.average(df["negative_corrected"], weights=weights)) if weights.sum() else float(df["negative_corrected"].mean())
    weighted_pos = float(np.average(df["positive_corrected"], weights=weights)) if weights.sum() else float(df["positive_corrected"].mean())
    weighted_nss = (weighted_pos - weighted_neg) * 100
    mean_abs_shift = df["negative_shift_pred"].abs().mean()

    top_vol_row = df.loc[df["n"].idxmax()]
    top_neg_row = df.loc[df["negative_corrected"].idxmax()]
    max_shift_row = df.loc[df["negative_shift_pred"].abs().idxmax()]
    narrowest_ci_row = df.loc[df["negative_ci_width"].idxmin()]
    widest_ci_row = df.loc[df["negative_ci_width"].idxmax()]
    kpis = {
        "total_ulasan": total_n,
        "sub_total_ulasan": f"{top_vol_row['app_name']} terbanyak: {top_vol_row['n']:,.0f} ulasan",
        "tingkat_negatif_terkalibrasi": round(weighted_neg * 100, 2),
        "sub_tingkat_negatif": f"Tertinggi: {top_neg_row['app_name']} ({top_neg_row['negative_corrected']*100:.1f}%)",
        "net_sentiment_score": round(weighted_nss, 2),
        "pergeseran_bias_abs_rata2": round(float(mean_abs_shift) * 100, 2),
    }
    insights = {
        "1_keluhan_tertinggi": f"{top_neg_row['app_name']} mencatat keluhan tertinggi sebesar {top_neg_row['negative_corrected']*100:.1f}% (95% CI: {top_neg_row['negative_ci_low']*100:.1f}%–{top_neg_row['negative_ci_high']*100:.1f}%).",
        "2_dampak_kalibrasi": f"Koreksi bias terbesar terjadi pada {max_shift_row['app_name']} ({max_shift_row['negative_shift_pred']*100:+.2f} poin persentase dibanding prediksi mentah).",
        "3_presisi_estimasi": f"Rentang 95% CI tersempit pada {narrowest_ci_row['app_name']} (lebar {narrowest_ci_row['negative_ci_width']*100:.2f}%, n={narrowest_ci_row['n']:,.0f}) dan terlebar pada {widest_ci_row['app_name']} (lebar {widest_ci_row['negative_ci_width']*100:.2f}%, n={widest_ci_row['n']:,.0f}).",
    }
    return df, kpis, insights


def compute_page2_features(df_monthly: pd.DataFrame, selected_apps=None, month_range=None, hide_thin_cell=False, spike_threshold=0.05):
    df = df_monthly.sort_values(["app_name", "review_month"]).copy()
    df["mom_neg_delta"] = df.groupby("app_name")["negative_corrected"].diff()
    df["mom_n_pct_change"] = df.groupby("app_name")["n"].pct_change() * 100
    df["is_spike"] = df["mom_neg_delta"] > spike_threshold
    if selected_apps:
        df = df[df["app_name"].isin(selected_apps)].copy()
    if month_range:
        start_m, end_m = month_range
        df = df[(df["review_month"] >= start_m) & (df["review_month"] <= end_m)].copy()
    if hide_thin_cell:
        df = df[~df["thin_cell"].fillna(False)].copy()
    df = df.reset_index(drop=True)

    empty_vol = pd.DataFrame(columns=["app_name", "months", "thin_cells", "mean_neg", "std_neg", "median_neg", "iqr_neg", "min_neg", "max_neg", "mean_boycott_diff", "max_boycott_diff", "corr_n_vs_neg"])
    empty_boycott = pd.DataFrame(columns=["app_name", "neg_with_boycott", "neg_without_boycott", "mean_diff", "max_diff", "weighted_diff"])
    if df.empty:
        kpis = {"puncak_keluhan_bulanan": 0, "sub_puncak": "Tidak ada data", "rata2_dan_volatilitas": "—", "lonjakan_mom_terbesar": "—", "sub_lonjakan": "Tidak ada data", "dampak_maks_boikot": "—", "sub_boikot": "Tidak ada data"}
        insights = {"1_lonjakan_terbesar": "Tidak ada data untuk menghitung lonjakan.", "2_puncak_keluhan": "Tidak ada data pada kombinasi filter ini.", "3_sensitivitas_boikot": "Tidak ada data untuk menghitung sensitivitas topik."}
        return df, empty_vol, empty_boycott, kpis, insights

    def volatility(g):
        return pd.Series({
            "months": g["review_month"].count(), "thin_cells": g["thin_cell"].fillna(False).sum(),
            "mean_neg": g["negative_corrected"].mean(), "std_neg": g["negative_corrected"].std(),
            "median_neg": g["negative_corrected"].median(),
            "iqr_neg": g["negative_corrected"].quantile(.75) - g["negative_corrected"].quantile(.25),
            "min_neg": g["negative_corrected"].min(), "max_neg": g["negative_corrected"].max(),
            "mean_boycott_diff": g["difference"].mean(), "max_boycott_diff": g["difference"].max(),
            "corr_n_vs_neg": g["n"].corr(g["negative_corrected"]),
        })
    app_volatility = pd.DataFrame(
        [{"app_name": app_name, **volatility(group).to_dict()} for app_name, group in df.groupby("app_name")]
    )
    summaries = []
    for app_name, group in df.groupby("app_name"):
        n = group["n"].fillna(0).clip(lower=0)
        n_without = group["n_without"].fillna(group["n"]).fillna(0).clip(lower=0)
        neg_without = group["negative_without"].fillna(group["negative_corrected"])
        summaries.append({
            "app_name": app_name,
            "neg_with_boycott": np.average(group["negative_corrected"], weights=n) if n.sum() else group["negative_corrected"].mean(),
            "neg_without_boycott": np.average(neg_without, weights=n_without) if n_without.sum() else neg_without.mean(),
            "mean_diff": group["difference"].mean(), "max_diff": group["difference"].max(),
        })
    boycott_summary = pd.DataFrame(summaries)
    boycott_summary["weighted_diff"] = boycott_summary["neg_with_boycott"] - boycott_summary["neg_without_boycott"]
    peak_row = df.loc[df["negative_corrected"].idxmax()]
    mean_neg_all, std_neg_all = df["negative_corrected"].mean(), df["negative_corrected"].std()
    valid_mom = df.dropna(subset=["mom_neg_delta"])
    max_spike_row = valid_mom.loc[valid_mom["mom_neg_delta"].idxmax()] if not valid_mom.empty else peak_row
    max_diff_row = df.loc[df["difference"].idxmax()]
    delta = max_spike_row["mom_neg_delta"]
    kpis = {
        "puncak_keluhan_bulanan": round(peak_row["negative_corrected"] * 100, 2),
        "sub_puncak": f"{peak_row['review_month']} ({peak_row['app_name']}, n={peak_row['n']:,.0f})",
        "rata2_dan_volatilitas": f"{mean_neg_all*100:.1f}% ± {std_neg_all*100:.1f}%" if pd.notna(std_neg_all) else f"{mean_neg_all*100:.1f}% ± —",
        "lonjakan_mom_terbesar": f"{delta*100:+.1f}%" if pd.notna(delta) else "—",
        "sub_lonjakan": f"{max_spike_row['app_name']} pada {max_spike_row['review_month']}",
        "dampak_maks_boikot": f"{max_diff_row['difference']*100:+.2f}%",
        "sub_boikot": f"{max_diff_row['app_name']} ({max_diff_row['review_month']})",
    }
    insights = {
        "1_lonjakan_terbesar": f"{max_spike_row['app_name']} mencatat lonjakan bulanan tertinggi ({delta*100:+.1f} poin) pada {max_spike_row['review_month']}." if pd.notna(delta) else "Belum ada bulan pembanding untuk menghitung lonjakan.",
        "2_puncak_keluhan": f"Rekor keluhan bulanan mencapai {peak_row['negative_corrected']*100:.1f}% pada {peak_row['app_name']} ({peak_row['review_month']}) dengan {peak_row['n']:,.0f} ulasan.",
        "3_sensitivitas_boikot": f"Rata-rata kontribusi topik boikot/blokir hanya {df['difference'].mean()*100:.2f}% (maksimum {max_diff_row['difference']*100:.2f}% pada {max_diff_row['app_name']} bulan {max_diff_row['review_month']}).",
    }
    return df, app_volatility, boycott_summary, kpis, insights


def classify_or_significance(row):
    if row["or_ci_high"] < 1.0:
        return "Signifikan Membaik (OR < 1)"
    if row["or_ci_low"] > 1.0:
        return "Signifikan Memburuk (OR > 1)"
    return "Tidak Signifikan"


def compute_page3_features(df_version: pd.DataFrame, selected_apps=None, only_reliable=True, min_reviews=1, search_version="", top_n=5):
    df_raw = df_version.copy()
    if selected_apps:
        df_raw = df_raw[df_raw["app_name"].isin(selected_apps)].copy()
    df_raw["first_date"] = pd.to_datetime(df_raw["first_date"])
    df_raw["last_date"] = pd.to_datetime(df_raw["last_date"])
    df_raw["active_days"] = (df_raw["last_date"] - df_raw["first_date"]).dt.days + 1
    df_raw["soft_hard_diff"] = df_raw["negative_share"] - df_raw["soft_negative"]
    df_raw["ci_width"] = df_raw["ci_high"] - df_raw["ci_low"]
    total_versions_all = len(df_raw)
    reliable_versions_all = int(df_raw["reliable"].sum())
    total_reviews_all = int(df_raw["n"].sum())
    reliable_reviews_all = int(df_raw.loc[df_raw["reliable"], "n"].sum())
    reliable_coverage_pct = reliable_reviews_all / total_reviews_all * 100 if total_reviews_all > 0 else 0.0
    df_filtered = df_raw.copy()
    if only_reliable:
        df_filtered = df_filtered[df_filtered["reliable"]].copy()
    if min_reviews > 1:
        df_filtered = df_filtered[df_filtered["n"] >= min_reviews].copy()
    if search_version.strip():
        df_filtered = df_filtered[df_filtered["app_version"].astype(str).str.contains(search_version.strip(), case=False, na=False)].copy()
    df_filtered = df_filtered.sort_values(["app_name", "first_date", "app_version"]).reset_index(drop=True)
    df_pareto = df_raw.sort_values(["app_name", "n"], ascending=[True, False]).copy()
    if not df_pareto.empty:
        df_pareto["cum_reviews"] = df_pareto.groupby("app_name")["n"].cumsum()
        df_pareto["total_app_reviews"] = df_pareto.groupby("app_name")["n"].transform("sum")
        df_pareto["cum_review_share"] = df_pareto["cum_reviews"] / df_pareto["total_app_reviews"].replace(0, np.nan) * 100
        df_pareto["version_rank"] = df_pareto.groupby("app_name").cumcount() + 1
    worst_versions = df_filtered.sort_values(["negative_share", "n"], ascending=[False, False]).head(top_n)
    best_versions = df_filtered.sort_values(["negative_share", "n"], ascending=[True, False]).head(top_n)
    median_days = df_filtered["active_days"].median() if not df_filtered.empty else 0
    longest_app = df_filtered.groupby("app_name")["active_days"].median().idxmax() if not df_filtered.empty else "-"
    worst_row = df_filtered.loc[df_filtered["negative_share"].idxmax()] if not df_filtered.empty else None
    max_soft_diff_row = df_filtered.loc[df_filtered["soft_hard_diff"].abs().idxmax()] if not df_filtered.empty else None
    kpis = {
        "total_versi_terdeteksi": f"{total_versions_all} versi ({reliable_versions_all} reliabel)",
        "cakupan_ulasan_reliabel": f"{reliable_coverage_pct:.1f}%",
        "sub_cakupan": f"{reliable_reviews_all:,} dari {total_reviews_all:,} ulasan berversi",
        "median_masa_aktif": f"{median_days:.0f} Hari",
        "sub_masa_aktif": f"Median terlama: {longest_app}",
        "versi_keluhan_tertinggi": f"{worst_row['negative_share']*100:.1f}%" if worst_row is not None else "—",
        "sub_versi_tertinggi": f"{worst_row['app_name']} v{worst_row['app_version']} (n={worst_row['n']:,.0f})" if worst_row is not None else "Tidak ada versi yang memenuhi filter",
    }
    top5_share = df_pareto[df_pareto["version_rank"] <= 5].groupby("app_name")["cum_review_share"].max().mean() if not df_pareto.empty else 0
    insights = {
        "1_konsentrasi_rilis": f"Sebanyak {reliable_coverage_pct:.1f}% ulasan terkumpul pada {reliable_versions_all} versi reliabel; rata-rata 5 versi teratas mencakup {top5_share:.1f}% total ulasan tiap bank." if pd.notna(top5_share) else "Belum ada data versi untuk menghitung konsentrasi rilis.",
        "2_anomali_rilis": f"Keluhan tertinggi pada sampel terpilih tercatat pada {worst_row['app_name']} versi {worst_row['app_version']} sebesar {worst_row['negative_share']*100:.1f}% (n={worst_row['n']:,.0f})." if worst_row is not None else "Tidak ada versi yang memenuhi filter.",
        "3_audit_kalibrasi_versi": f"Selisih terbesar antara prediksi keras dan probabilitas softmax muncul pada {max_soft_diff_row['app_name']} v{max_soft_diff_row['app_version']} ({max_soft_diff_row['soft_hard_diff']*100:+.2f}%)." if max_soft_diff_row is not None else "Tidak ada data untuk audit kalibrasi versi.",
    }
    return df_filtered, df_pareto, worst_versions, best_versions, kpis, insights


def compute_page4_features(df_trend: pd.DataFrame, df_app: pd.DataFrame, df_monthly: pd.DataFrame, df_version: pd.DataFrame, selected_apps=None, selected_subsets=None):
    df_all = df_trend.copy()
    df_for_robustness = df_all[df_all["app_name"].isin(selected_apps)].copy() if selected_apps else df_all.copy()
    if df_all.empty:
        empty_robust = pd.DataFrame(columns=["app_name", "or_all", "or_reliable", "or_without_boycott", "boycott_or_shift", "is_direction_consistent"])
        matrix = pd.DataFrame()
        kpis = {"perbaikan_versi_terkuat": "—", "sub_perbaikan": "Tidak ada data", "rata2_korelasi_kronologis": 0, "konsistensi_robustness": "0 dari 0 Aplikasi", "total_sampel_teruji": "0 ulasan", "sub_sampel_teruji": "Tidak ada data"}
        insights = {"1_arah_tren_utama": "Tidak ada hasil model untuk filter ini.", "2_ketahanan_boikot": "Tidak ada data robustness.", "3_validitas_urutan_rilis": "Tidak ada data korelasi kronologis."}
        return df_all, empty_robust, matrix, kpis, insights

    df_for_robustness["_sig"] = df_for_robustness.apply(classify_or_significance, axis=1)
    piv_or = df_for_robustness.pivot(index="app_name", columns="subset", values="odds_ratio_per_version")
    piv_sig = df_for_robustness.pivot(index="app_name", columns="subset", values="_sig")
    robustness_df = pd.DataFrame({
        "app_name": piv_or.index,
        "or_all": piv_or["all versioned reviews"] if "all versioned reviews" in piv_or else np.nan,
        "or_reliable": piv_or["reliable versions only"] if "reliable versions only" in piv_or else np.nan,
        "or_without_boycott": piv_or["without blocking/boycott topic"] if "without blocking/boycott topic" in piv_or else np.nan,
    }).reset_index(drop=True)
    robustness_df["boycott_or_shift"] = robustness_df["or_all"] - robustness_df["or_without_boycott"]
    statuses = []
    for app_name in robustness_df["app_name"]:
        app_sigs = piv_sig.loc[app_name]
        values = [app_sigs.get(s) for s in ["all versioned reviews", "without blocking/boycott topic", "reliable versions only"]]
        statuses.append(pd.notna(values).all() and values[0] == values[1] == values[2])
    robustness_df["is_direction_consistent"] = statuses
    df = df_all
    if selected_apps:
        df = df[df["app_name"].isin(selected_apps)].copy()
    if selected_subsets:
        df = df[df["subset"].isin(selected_subsets)].copy()
    df["pct_odds_change_per_version"] = (df["odds_ratio_per_version"] - 1.0) * 100
    df["or_ci_width"] = df["or_ci_high"] - df["or_ci_low"]
    df["significance_status"] = df.apply(classify_or_significance, axis=1)
    ref_df = df[df["subset"] == "reliable versions only"]
    ref_df = ref_df if not ref_df.empty else df
    if ref_df.empty:
        kpis = {"perbaikan_versi_terkuat": "—", "sub_perbaikan": "Tidak ada data", "rata2_korelasi_kronologis": 0, "konsistensi_robustness": f"{int(robustness_df['is_direction_consistent'].sum())} dari {len(robustness_df)} Aplikasi", "total_sampel_teruji": "0 ulasan", "sub_sampel_teruji": "Tidak ada data"}
        insights = {"1_arah_tren_utama": "Tidak ada hasil untuk subset terpilih.", "2_ketahanan_boikot": "Tidak ada data robustness.", "3_validitas_urutan_rilis": "Tidak ada data korelasi kronologis."}
        return df, robustness_df, pd.DataFrame(), kpis, insights
    best_or_row = ref_df.loc[ref_df["odds_ratio_per_version"].idxmin()]
    mean_corr_chrono = ref_df["corr_version_month"].mean()
    consistent_count, total_apps_count = int(robustness_df["is_direction_consistent"].sum()), len(robustness_df)
    all_sub = df[df["subset"] == "all versioned reviews"]
    sample_ref = all_sub if not all_sub.empty else ref_df
    kpis = {
        "perbaikan_versi_terkuat": f"OR {best_or_row['odds_ratio_per_version']:.3f} ({best_or_row['pct_odds_change_per_version']:+.2f}%/versi)",
        "sub_perbaikan": f"{best_or_row['app_name']} ({best_or_row['subset']})",
        "rata2_korelasi_kronologis": round(float(mean_corr_chrono), 3) if pd.notna(mean_corr_chrono) else 0,
        "konsistensi_robustness": f"{consistent_count} dari {total_apps_count} Aplikasi",
        "total_sampel_teruji": f"{int(sample_ref['reviews'].sum()):,} ulasan",
        "sub_sampel_teruji": f"Mencakup {int(sample_ref['versions'].sum())} versi ({sample_ref['subset'].iloc[0]})",
    }
    improving_apps = ref_df.loc[ref_df["significance_status"] == "Signifikan Membaik (OR < 1)", "app_name"].tolist()
    valid_shift = robustness_df.dropna(subset=["boycott_or_shift"])
    max_boycott_shift = valid_shift.loc[valid_shift["boycott_or_shift"].abs().idxmax()] if not valid_shift.empty else None
    top_corr_row = ref_df.loc[ref_df["corr_version_month"].idxmax()]
    insights = {
        "1_arah_tren_utama": f"Aplikasi dengan tren perbaikan signifikan per versi: {', '.join(improving_apps) if improving_apps else 'Tidak ada'}. Terkuat pada {best_or_row['app_name']} (OR = {best_or_row['odds_ratio_per_version']:.3f}).",
        "2_ketahanan_boikot": f"Pergeseran OR terbesar saat topik boikot dihapus terjadi pada {max_boycott_shift['app_name']} ({max_boycott_shift['boycott_or_shift']:+.5f}), menunjukkan hasil pemodelan sangat stabil." if max_boycott_shift is not None else "Tidak tersedia pasangan skenario untuk menghitung pergeseran OR.",
        "3_validitas_urutan_rilis": f"Korelasi kronologis tertinggi antara versi dan bulan ulasan dicapai {top_corr_row['app_name']} (r = {top_corr_row['corr_version_month']:.3f}).",
    }
    m_agg = df_monthly.groupby("app_name", as_index=False).agg(monthly_neg_mean=("negative_corrected", "mean"), monthly_neg_peak=("negative_corrected", "max"), boycott_diff_max=("difference", "max"))
    v_agg = df_version.groupby("app_name", as_index=False).agg(total_versions=("app_version", "count"), reliable_versions=("reliable", "sum"))
    t_agg = df_trend[df_trend["subset"] == "reliable versions only"][ ["app_name", "corr_version_month", "odds_ratio_per_version", "or_ci_low", "or_ci_high"] ]
    benchmark_matrix = (df_app[["app_name", "n", "negative_corrected", "neutral_corrected", "positive_corrected"]]
        .assign(net_sentiment_score=lambda d: (d["positive_corrected"] - d["negative_corrected"]) * 100)
        .merge(m_agg, on="app_name", how="left").merge(v_agg, on="app_name", how="left").merge(t_agg, on="app_name", how="left"))
    if selected_apps:
        benchmark_matrix = benchmark_matrix[benchmark_matrix["app_name"].isin(selected_apps)].reset_index(drop=True)
    return df, robustness_df, benchmark_matrix, kpis, insights
