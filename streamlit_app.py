import io
import os
import math
from typing import Optional, List

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="Tech YouTubers — GTM Benchmark & Budget Planner", layout="wide")

# ==============================
# Constants / Required Columns
# ==============================
REQUIRED_BASE_COLS = [
    "ChannelTitle","Subscribers","TotalViews","VideoCount",
    "Views_30d","Views_90d","Uploads_90d","AvgViewsPerRecent",
    "SpendUSD","CPM_lifetime","CPM_90d","EstRevenue_lifetime","EstRevenue_90d",
    "ROI_lifetime","ROI_90d"
]

# Replace with your latest export name if different
DEFAULT_CSV = "yt_cpm_gtm_2025-09-25.csv"

# ==============================
# Helpers
# ==============================
def ensure_column(df: pd.DataFrame, col: str, default=0):
    if col not in df.columns:
        df[col] = default
    return df

def load_dataset(uploaded, default_path: Optional[str] = None) -> pd.DataFrame:
    if uploaded is not None:
        df = pd.read_csv(uploaded)
    elif default_path and os.path.exists(default_path):
        df = pd.read_csv(default_path)
    else:
        st.warning("Upload a CSV exported by your GTM script (Baseline/Plans).")
        st.stop()

    # Normalize expected columns / types
    for c in REQUIRED_BASE_COLS:
        ensure_column(df, c, 0)

    df["EngagementRate"] = df.apply(
        lambda r: (r["TotalViews"]/r["Subscribers"]) if r["Subscribers"] else 0.0, axis=1
    )

    # Optional tags
    if "Category" not in df.columns:
        df["Category"] = "General"
    if "Region" not in df.columns:
        df["Region"] = "Global"

    for c in ["CPM_lifetime","CPM_90d","ROI_lifetime","ROI_90d","Views_30d","Views_90d",
              "Uploads_90d","AvgViewsPerRecent","SpendUSD","Subscribers","TotalViews"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df

def kpi_cards(df: pd.DataFrame):
    total_subs = int(df["Subscribers"].sum())
    total_views_90d = int(df["Views_90d"].sum())
    # Weighted CPM (fallback to lifetime)
    weights = df["SpendUSD"].replace(0, np.nan)
    if weights.isna().all():
        weights = pd.Series(np.ones(len(df)), index=df.index)
    cpm_col = "CPM_90d" if df["CPM_90d"].notna().any() else "CPM_lifetime"
    weighted_cpm = np.average(df[cpm_col].fillna(df["CPM_lifetime"]), weights=weights)
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Subscribers", f"{total_subs:,}")
    col2.metric("90-Day Views (sum)", f"{total_views_90d:,}")
    col3.metric("Weighted CPM", f"${weighted_cpm:,.2f}")

def herfindahl_hhi(shares: np.ndarray) -> float:
    """Herfindahl–Hirschman concentration index (0–1)."""
    s = shares[shares > 0]
    return float(np.sum((s)**2))

def plan_allocate(
    df: pd.DataFrame,
    total_budget: float,
    method: str,
    cap_share: float,
    categories: Optional[List[str]] = None,
    cpm_multiplier: float = 1.0,
    diminishing_returns: bool = False,
    dim_beta: float = 0.25,
):
    """Compute a budget allocation plan.
    - method: by_inverse_cpm | by_recent_views | by_subscribers
    - cap_share: max share per creator (0–1)
    - categories: filter to allowed categories (if None → all)
    - cpm_multiplier: scenario scalar for CPM (e.g., 1.2 pessimistic, 0.8 optimistic)
    - diminishing_returns: if True, penalize very concentrated spend
    - dim_beta: penalty strength (0..1). Applies when share exceeds cap_share/2.
    """
    data = df.copy()

    if categories:
        data = data[data["Category"].isin(categories)].copy()
        if data.empty:
            st.warning("No channels in the selected categories.")
            return pd.DataFrame(), np.nan, 0, 0, 0

    # Choose CPM column and apply multiplier
    use_cpm = data["CPM_90d"].where(
        data["CPM_90d"].notna() & (data["CPM_90d"] > 0), data["CPM_lifetime"]
    )
    use_cpm = use_cpm * max(cpm_multiplier, 1e-9)  # avoid zero

    # Choose weights
    if method == "by_recent_views" and data["Views_90d"].fillna(0).sum() > 0:
        weights = data["Views_90d"].clip(lower=0).fillna(0.0)
    elif method == "by_subscribers":
        weights = data["Subscribers"].clip(lower=0).fillna(0.0)
    else:
        inv = 1.0 / use_cpm.replace(0, np.nan)
        weights = inv.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    if weights.sum() == 0:
        weights = pd.Series([1.0]*len(data), index=data.index)

    alloc = (weights / weights.sum()) * total_budget
    cap = total_budget * cap_share
    alloc = alloc.clip(upper=cap)

    # Rebalance leftover proportionally to available headroom
    leftover = total_budget - alloc.sum()
    if abs(leftover) > 1e-6:
        headroom = cap - alloc
        headroom = headroom.clip(lower=0)
        if headroom.sum() > 0:
            alloc += headroom / headroom.sum() * leftover
        else:
            alloc += leftover / len(alloc)

    # Raw impressions (linear)
    raw_impr = (alloc / use_cpm.replace(0, np.nan)) * 1000
    raw_impr = raw_impr.replace([np.inf, -np.inf], 0).fillna(0)

    # Optional diminishing returns penalty (reduce impressions for very concentrated spend)
    if diminishing_returns and total_budget > 0:
        share = alloc / total_budget
        # Penalty kicks in above half of cap: factor decreases linearly with slope dim_beta
        thresh = cap_share / 2.0
        penalty = np.where(
            share > thresh,
            np.clip(1.0 - dim_beta * ((share - thresh) / max(1e-9, 1 - thresh)), 0.5, 1.0),
            1.0,
        )
        eff_impr = raw_impr * penalty
    else:
        eff_impr = raw_impr

    out = data[["ChannelTitle","Category","Subscribers","Views_90d","CPM_90d","CPM_lifetime"]].copy()
    out["RecommendedSpendUSD"] = alloc.round(2)
    out["ProjectedImpressions"] = eff_impr.round(0).astype(int)
    out["ShareOfBudget"] = (alloc / total_budget).round(3)
    out = out.sort_values("RecommendedSpendUSD", ascending=False).reset_index(drop=True)

    total_impr = int(out["ProjectedImpressions"].sum())
    weighted_cpm = (out["RecommendedSpendUSD"].sum() / (total_impr/1000.0)) if total_impr > 0 else np.nan

    hhi = herfindahl_hhi(out["ShareOfBudget"].values)
    top_share = float(out["ShareOfBudget"].max()) if not out.empty else 0.0

    return out, weighted_cpm, total_impr, hhi, top_share

def df_download_button(df: pd.DataFrame, filename: str, label: str):
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(label, csv, file_name=filename, mime="text/csv")

def xlsx_download_button(dfs: dict, filename: str, label: str):
    import xlsxwriter
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as w:
        for sheet, data in dfs.items():
            data.to_excel(w, index=False, sheet_name=sheet)
    st.download_button(label, data=buf.getvalue(),
                       file_name=filename,
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

def one_pager(df: pd.DataFrame, plan: pd.DataFrame, total_budget: float,
              plan_cpm: float, total_impr: int, hhi: float, top_share: float,
              method: str, cap_share: float) -> str:
    top_reach = df.loc[df["Views_90d"].idxmax(), "ChannelTitle"] if df["Views_90d"].sum() > 0 else df.loc[df["Subscribers"].idxmax(), "ChannelTitle"]
    cpm_col = "CPM_90d" if df["CPM_90d"].notna().any() else "CPM_lifetime"
    best_cpm_chan = df.loc[df[cpm_col].idxmin(), "ChannelTitle"]

    lines = []
    lines.append("# GTM One-Pager — Tech YouTubers\n")
    lines.append("**Goal:** Maximize paid reach and efficiency for a startup campaign using tech creators.\n")
    lines.append("## Snapshot")
    lines.append(f"- Channels analyzed: {len(df)}")
    lines.append(f"- Combined subscribers: {int(df['Subscribers'].sum()):,}")
    lines.append(f"- 90-day total views (approx): {int(df['Views_90d'].sum()):,}\n")
    lines.append("## Leaders")
    lines.append(f"- Top recent reach: **{top_reach}**")
    lines.append(f"- Best CPM source: **{best_cpm_chan}**\n")
    lines.append("## Plan")
    lines.append(f"- Method: **{method}**, Budget: **${total_budget:,.0f}**, Cap/creator: **{int(cap_share*100)}%**")
    lines.append(f"- Plan weighted CPM: **${plan_cpm:,.2f}**")
    lines.append(f"- Projected impressions (90d CPM basis): **{total_impr:,}**")
    lines.append(f"- Concentration — HHI: **{hhi:.3f}**, Top creator share: **{top_share:.0%}**\n")
    lines.append("## Allocation")
    for _, r in plan.iterrows():
        lines.append(f"- {r['ChannelTitle']}: ${r['RecommendedSpendUSD']:,.0f} → {int(r['ProjectedImpressions']):,} est. impressions ({r['ShareOfBudget']:.0%} of budget)")
    lines.append("\n## Recommendation")
    lines.append("- Start with efficiency-led allocation (inverse CPM), then blend with recent-views weighting for coverage.")
    lines.append("- Run a 2-week pilot with UTMs/unique codes; track CTR, sign-ups, CAC proxy; rebalance weekly.")
    return "\n".join(lines)

# ==============================
# UI — Sidebar
# ==============================
st.title("Tech YouTubers — GTM Benchmark & Budget Planner")

with st.sidebar:
    st.header("Data")
    uploaded = st.file_uploader("Upload GTM CSV (yt_cpm_gtm_*.csv)", type=["csv"])
    default_path = st.text_input("Default CSV path (checked if no upload)", value=DEFAULT_CSV)

df = load_dataset(uploaded, default_path=default_path)

# ==============================
# Tagging / Filters
# ==============================
with st.expander("✏️ Tagging (Category, Region)", expanded=False):
    st.write("Edit categories/regions here. Use the download buttons to save your tagged dataset for next runs.")
    tagged = st.data_editor(df[["ChannelTitle","Category","Region"]], use_container_width=True, num_rows="dynamic")
    # Merge back
    df = df.drop(columns=["Category","Region"], errors="ignore").merge(tagged, on="ChannelTitle", how="left")
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        df_download_button(df, f"gtm_dataset_tagged_{pd.Timestamp.today().date().isoformat()}.csv",
                           "⬇️ Download tagged dataset")
    with col_t2:
        unique_cats = sorted(df["Category"].dropna().unique().tolist())
        st.caption("Unique categories: " + (", ".join(unique_cats) if unique_cats else "None"))

with st.expander("🔍 Filters", expanded=False):
    cats = sorted(df["Category"].dropna().unique().tolist())
    selected_categories = st.multiselect("Include categories", options=cats, default=cats)

# ==============================
# Tabs
# ==============================
tab1, tab2 = st.tabs(["📊 Benchmark", "🧮 Planner"])

with tab1:
    st.subheader("Baseline overview")
    kpi_cards(df)

    sort_by = st.selectbox("Sort channels by",
                           ["Views_90d","Subscribers","CPM_90d","ROI_90d","CPM_lifetime","TotalViews"],
                           index=0)
    ascending = sort_by in ["CPM_90d","CPM_lifetime"]  # low CPM is good
    filt = df[df["Category"].isin(selected_categories)] if selected_categories else df
    show = filt.sort_values(sort_by, ascending=ascending).reset_index(drop=True)

    st.dataframe(
        show[["ChannelTitle","Category","Subscribers","TotalViews","VideoCount","Views_30d","Views_90d","Uploads_90d","CPM_90d","ROI_90d","CPM_lifetime"]],
        use_container_width=True
    )

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**90-Day Views by Channel**")
        fig1, ax1 = plt.subplots()
        ax1.bar(show["ChannelTitle"], show["Views_90d"])
        ax1.set_ylabel("Views (90d)")
        ax1.set_xticklabels(show["ChannelTitle"], rotation=45, ha="right")
        st.pyplot(fig1, clear_figure=True)
    with c2:
        st.markdown("**CPM (90-Day) by Channel**")
        fig2, ax2 = plt.subplots()
        if show["CPM_90d"].notna().any():
            ax2.bar(show["ChannelTitle"], show["CPM_90d"])
            ax2.set_ylabel("CPM (USD per 1,000 views)")
        else:
            ax2.bar(show["ChannelTitle"], show["CPM_lifetime"])
            ax2.set_ylabel("CPM (USD per 1,000 views) — Lifetime (fallback)")
        ax2.set_xticklabels(show["ChannelTitle"], rotation=45, ha="right")
        st.pyplot(fig2, clear_figure=True)

with tab2:
    st.subheader("Budget allocation simulator")

    # Quick scenario buttons via session state
    if "budget" not in st.session_state:
        st.session_state.budget = 50000.0

    col_top = st.columns(4)
    with col_top[0]:
        total_budget = st.number_input("Total budget (USD)", min_value=1000.0,
                                       value=st.session_state.budget, step=1000.0, format="%.2f", key="budget_input")
    with col_top[1]:
        if st.button("$25k"):
            st.session_state.budget = 25000.0
    with col_top[2]:
        if st.button("$50k"):
            st.session_state.budget = 50000.0
    with col_top[3]:
        if st.button("$100k"):
            st.session_state.budget = 100000.0
    total_budget = st.session_state.budget

    c1, c2, c3 = st.columns(3)
    with c1:
        method = st.selectbox("Allocation method", ["by_inverse_cpm","by_recent_views","by_subscribers"])
    with c2:
        cap_share = st.slider("Cap per creator (share of budget)", min_value=0.1, max_value=1.0, value=0.45, step=0.05)
    with c3:
        categories_used = selected_categories if selected_categories else None
        st.text("Using categories:")
        st.code(", ".join(categories_used) if categories_used else "All")

    # Advanced options
    with st.expander("⚙️ Advanced options", expanded=False):
        s1, s2, s3 = st.columns(3)
        with s1:
            use_sensitivity = st.checkbox("Enable CPM sensitivity bands", value=True)
        with s2:
            pessimistic = st.slider("Pessimistic CPM +%", min_value=0, max_value=100, value=25, step=5)
        with s3:
            optimistic  = st.slider("Optimistic CPM -%",  min_value=0, max_value=100, value=20, step=5)
        reopt = st.checkbox("Re-optimize allocation per scenario", value=False)

        d1, d2 = st.columns(2)
        with d1:
            diminishing = st.checkbox("Apply diminishing returns penalty", value=True)
        with d2:
            dim_beta = st.slider("Penalty strength (β)", min_value=0.0, max_value=1.0, value=0.25, step=0.05)

    # Base plan
    plan_base, cpm_base, impr_base, hhi_base, top_share_base = plan_allocate(
        df, total_budget, method, cap_share, categories=categories_used,
        cpm_multiplier=1.0, diminishing_returns=diminishing, dim_beta=dim_beta
    )

    st.markdown(f"**Base plan weighted CPM:** ${cpm_base:,.2f}" if not np.isnan(cpm_base) else "**Base plan weighted CPM:** N/A")
    st.markdown(f"**Projected impressions:** {impr_base:,}")
    st.markdown(f"**Concentration:** HHI={hhi_base:.3f}, Top share={top_share_base:.0%}")
    st.dataframe(plan_base, use_container_width=True)

    # Chart: Spend vs Projected Impressions
    if not plan_base.empty:
        fig, ax = plt.subplots()
        ax.scatter(plan_base["RecommendedSpendUSD"], plan_base["ProjectedImpressions"])
        for _, row in plan_base.iterrows():
            ax.annotate(row["ChannelTitle"],
                        (row["RecommendedSpendUSD"], row["ProjectedImpressions"]),
                        fontsize=8, xytext=(3,3), textcoords="offset points")
        ax.set_xlabel("Recommended Spend (USD)")
        ax.set_ylabel("Projected Impressions (est.)")
        st.pyplot(fig, clear_figure=True)

    # Sensitivity analysis
    results = {"Base": (plan_base, cpm_base, impr_base)}
    if use_sensitivity and not plan_base.empty:
        pess_mult = 1.0 + pessimistic/100.0
        opt_mult  = max(1.0 - optimistic/100.0, 0.01)

        if reopt:
            plan_pess, cpm_pess, impr_pess, _, _ = plan_allocate(
                df, total_budget, method, cap_share, categories=categories_used,
                cpm_multiplier=pess_mult, diminishing_returns=diminishing, dim_beta=dim_beta
            )
            plan_opt,  cpm_opt,  impr_opt,  _, _ = plan_allocate(
                df, total_budget, method, cap_share, categories=categories_used,
                cpm_multiplier=opt_mult, diminishing_returns=diminishing, dim_beta=dim_beta
            )
        else:
            # Hold allocation, only recompute outcomes by scaling CPM
            plan_pess = plan_base.copy()
            plan_opt  = plan_base.copy()
            # Recompute impressions by scaling CPM:
            cpm_col = df["CPM_90d"].where(df["CPM_90d"].notna() & (df["CPM_90d"]>0), df["CPM_lifetime"])
            cpm_map = dict(zip(df["ChannelTitle"], cpm_col))
            def recompute_impr(plan_df, mult):
                cpm_vec = plan_df["ChannelTitle"].map(cpm_map) * mult
                impr = (plan_df["RecommendedSpendUSD"] / cpm_vec.replace(0, np.nan)) * 1000
                return impr.replace([np.inf, -np.inf], 0).fillna(0).round(0).astype(int)
            plan_pess["ProjectedImpressions"] = recompute_impr(plan_pess, pess_mult)
            plan_opt["ProjectedImpressions"]  = recompute_impr(plan_opt,  opt_mult)

            def weighted_cpm(plan_df):
                tot_impr = plan_df["ProjectedImpressions"].sum()
                return (plan_df["RecommendedSpendUSD"].sum() / (tot_impr/1000.0)) if tot_impr>0 else np.nan
            cpm_pess = weighted_cpm(plan_pess)
            cpm_opt  = weighted_cpm(plan_opt)
            impr_pess = int(plan_pess["ProjectedImpressions"].sum())
            impr_opt  = int(plan_opt["ProjectedImpressions"].sum())

        results["Pessimistic"] = (plan_pess, cpm_pess, impr_pess)
        results["Optimistic"]  = (plan_opt,  cpm_opt,  impr_opt)

        st.subheader("Sensitivity summary")
        sdf = pd.DataFrame({
            "Scenario": ["Optimistic","Base","Pessimistic"],
            "Weighted CPM": [results["Optimistic"][1], results["Base"][1], results["Pessimistic"][1]],
            "Projected Impressions": [results["Optimistic"][2], results["Base"][2], results["Pessimistic"][2]],
        })
        st.dataframe(sdf, use_container_width=True)

    # Downloads
    cdl1, cdl2, cdl3 = st.columns(3)
    if not plan_base.empty:
        with cdl1:
            df_download_button(plan_base,
                               f"gtm_plan_{method}_{cap_share:.2f}_{pd.Timestamp.today().date().isoformat()}.csv",
                               "⬇️ Download base plan CSV")
        with cdl2:
            payload = {"Plan_Base": plan_base,
                       "Baseline": df[["ChannelTitle","Category","Subscribers","Views_90d","CPM_90d","CPM_lifetime","ROI_90d"]]}
            if "Optimistic" in results:
                payload["Plan_Optimistic"] = results["Optimistic"][0]
            if "Pessimistic" in results:
                payload["Plan_Pessimistic"] = results["Pessimistic"][0]
            xlsx_download_button(payload,
                                 f"gtm_plans_{method}_{pd.Timestamp.today().date().isoformat()}.xlsx",
                                 "⬇️ Download plans Excel")
        with cdl3:
            md = one_pager(df, plan_base, total_budget, results["Base"][1], results["Base"][2],
                           herfindahl_hhi(plan_base["ShareOfBudget"].values), plan_base["ShareOfBudget"].max(),
                           method, cap_share)
            st.download_button("⬇️ Download one-pager (Markdown)", data=md.encode("utf-8"),
                               file_name="gtm_one_pager.md", mime="text/markdown")

    st.caption("Tip: Use category filters to tailor the plan to your product ICP. Enable sensitivity to see outcome ranges.")