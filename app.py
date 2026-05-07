"""
Market Froth Dashboard — Streamlit replacement
Uses S&P 500 total return history as the single base market series for:
- forward return calculations
- downturn detection
- indicator alignment
- technicals base chart

Run:
    streamlit run froth_app.py
"""

import warnings
warnings.filterwarnings("ignore")

import os
from io import StringIO
from itertools import combinations
from bisect import bisect_right, insort

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Market Froth Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

DATA_DIR = "."


# ══════════════════════════════════════════════════════════════════════════════
# 1. THEME
# ══════════════════════════════════════════════════════════════════════════════
BLUE   = "#2563eb"
ORANGE = "#ea580c"
GREEN  = "#16a34a"
RED    = "#dc2626"
PURPLE = "#7c3aed"
SLATE  = "#94a3b8"
TEAL   = "#0f766e"

BORDER = "#e2e8f0"
BG2    = "#f8fafc"
BG     = "#ffffff"
TEXT   = "#0f172a"
MUTED  = "#64748b"

INVERSE_INDICATORS = [
    "Earnings Yield",
    "Dividend Yield",
    "Equity Risk Premium",
    "Yield Curve (10Y–2Y)",
    "Insider Buy/Sell",
    "AAII Bullish",
    "BB Lower",
    "SPX SMA200",
    "Froth Yield Curve",
    "Froth Real Policy Rate",
    "Froth Current Account (3Y Avg)",
    "Froth Fiscal Deficit",
    "Froth IPO Median Age",
]

GMD_START_YEAR = 1880
GMD_END_YEAR = 2026


# ══════════════════════════════════════════════════════════════════════════════
# 2. STYLES
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(f"""
<style>
.block-container {{
    padding-top: 1.2rem;
    padding-bottom: 2rem;
}}
.metric-pill {{
    display:inline-block;
    padding:5px 10px;
    margin:4px 6px 0 0;
    border-radius:999px;
    background:#eef2ff;
    border:1px solid #c7d2fe;
    color:{TEXT};
    font-size:11px;
    font-weight:600;
}}
.info-card {{
    padding:12px 14px;
    border-radius:12px;
    border:1px solid {BORDER};
    border-left:4px solid {BLUE};
    background:#eff6ff;
    margin:6px 0 10px 0;
}}
.small-muted {{
    color:{MUTED};
    font-size:12px;
}}
.section-divider {{
    color:{MUTED};
    font-size:10px;
    font-weight:800;
    letter-spacing:.08em;
    text-transform:uppercase;
    margin:8px 0 6px;
    border-bottom:1px solid {BORDER};
    padding-bottom:4px;
}}
table.pretty-table {{
    width:100%;
    border-collapse:collapse;
    font-size:12px;
    color:{TEXT};
    background:{BG};
    border:1px solid {BORDER};
    border-radius:10px;
    overflow:hidden;
}}
table.pretty-table thead {{
    background:#f1f5f9;
}}
table.pretty-table th, table.pretty-table td {{
    padding:8px 10px;
    border-bottom:1px solid {BORDER};
}}
div[data-testid="stTabs"] button[role="tab"] {{
    min-height:48px;
    padding:0 16px;
    border-radius:14px 14px 0 0;
    font-weight:800;
}}
div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {{
    background:#eff6ff;
    color:{TEXT};
}}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# 3. DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════
def p(name: str) -> str:
    return os.path.join(DATA_DIR, name)


def parse_date_series(values):
    parsed = pd.to_datetime(values, errors="coerce", utc=True)
    if isinstance(parsed, pd.Series):
        return parsed.dt.tz_localize(None)
    if isinstance(parsed, pd.DatetimeIndex):
        return parsed.tz_localize(None)
    if isinstance(parsed, pd.Timestamp):
        return parsed.tz_localize(None) if parsed.tzinfo is not None else parsed
    return parsed


def safe_divide(num, den):
    num_s = pd.Series(num, copy=False, dtype=float)
    den_s = pd.Series(den, copy=False, dtype=float)
    out = num_s / den_s.replace(0, np.nan)
    return out.replace([np.inf, -np.inf], np.nan)


def standardize_full_sample(series):
    s = pd.Series(series, dtype=float)
    mean = s.mean(skipna=True)
    std = s.std(skipna=True, ddof=0)
    if pd.isna(std) or std == 0:
        return pd.Series(np.nan, index=s.index, dtype=float)
    return (s - mean) / std


def hp_filter_cycle(series, lamb=6.25):
    s = pd.Series(series, dtype=float)
    valid = s.dropna()
    if len(valid) < 5:
        return pd.Series(np.nan, index=s.index, dtype=float)

    y = np.log(valid.to_numpy(dtype=float))
    n = len(y)
    d = np.zeros((n - 2, n))
    idx = np.arange(n - 2)
    d[idx, idx] = 1.0
    d[idx, idx + 1] = -2.0
    d[idx, idx + 2] = 1.0

    trend = np.linalg.solve(np.eye(n) + lamb * (d.T @ d), y)
    cycle = pd.Series(np.nan, index=s.index, dtype=float)
    cycle.loc[valid.index] = 100.0 * (y - trend)
    return cycle


def build_us_gmd_frame(gmd_raw, country="United States", start_year=GMD_START_YEAR, end_year=GMD_END_YEAR):
    us = gmd_raw.loc[
        (gmd_raw["countryname"] == country) &
        (gmd_raw["year"] >= start_year) &
        (gmd_raw["year"] <= end_year)
    ].copy()
    us["year"] = pd.to_numeric(us["year"], errors="coerce")
    us = us.dropna(subset=["year"]).sort_values("year").reset_index(drop=True)
    us["year"] = us["year"].astype(int)
    us["Date"] = pd.to_datetime(us["year"].astype(str) + "-12-31", errors="coerce")

    numeric_cols = [c for c in us.columns if c not in {"countryname", "ISO3", "id", "income_group", "Date"}]
    for col in numeric_cols:
        us[col] = pd.to_numeric(us[col], errors="coerce")

    return us


def load_loans_to_nonfin():
    for filename in ["loans_to_nonfin.csv", "loands_to_nonfin.csv"]:
        file_path = p(filename)
        if os.path.exists(file_path):
            raw = pd.read_csv(file_path)
            raw.columns = [str(c).strip() for c in raw.columns]
            date_col = next((c for c in raw.columns if c.lower() in {"date", "observation_date"}), raw.columns[0])
            value_candidates = [c for c in raw.columns if c != date_col]
            value_col = value_candidates[0] if value_candidates else None
            if value_col is None:
                return pd.DataFrame(columns=["Date", "Value"])
            out = raw[[date_col, value_col]].rename(columns={date_col: "Date", value_col: "Value"}).copy()
            out["Date"] = parse_date_series(out["Date"])
            out["Value"] = pd.to_numeric(out["Value"], errors="coerce")
            return out.dropna(subset=["Date", "Value"]).sort_values("Date").reset_index(drop=True)
    return pd.DataFrame(columns=["Date", "Value"])


def build_nonfinancial_loan_indicators(us_gmd, loans_to_nonfin):
    if loans_to_nonfin is None or loans_to_nonfin.empty:
        return {}

    base = us_gmd[["Date", "nGDP", "ltrate"]].copy().sort_values("Date")
    loans = loans_to_nonfin[["Date", "Value"]].copy().sort_values("Date")
    aligned = pd.merge_asof(
        loans,
        base,
        on="Date",
        direction="backward",
        tolerance=pd.Timedelta(days=460)
    )
    debt_service = safe_divide(aligned["Value"] * 1000.0 * (aligned["ltrate"] / 100.0), aligned["nGDP"])
    loan_qoq_growth = aligned["Value"].pct_change(fill_method=None) * 100.0

    return {
        "Froth Nonfinancial Debt Service": pd.DataFrame({"Date": aligned["Date"], "Value": debt_service}),
        "Froth Nonfinancial Loan Acceleration": pd.DataFrame({"Date": aligned["Date"], "Value": loan_qoq_growth.diff()}),
    }


def build_global_gmd_indicators(gmd_raw, excluded_country="United States", start_year=GMD_START_YEAR,
                                end_year=GMD_END_YEAR, gdp_coverage=0.90, min_countries=5):
    panel = gmd_raw.copy()
    panel["year"] = pd.to_numeric(panel["year"], errors="coerce")
    panel = panel.loc[
        (panel["countryname"] != excluded_country) &
        (panel["year"] >= start_year) &
        (panel["year"] <= end_year)
    ].copy()

    for col in ["M3", "nGDP", "nGDP_USD", "ltrate", "strate"]:
        panel[col] = pd.to_numeric(panel[col], errors="coerce")

    panel = panel.sort_values(["countryname", "year"]).reset_index(drop=True)
    panel["credit_to_gdp"] = safe_divide(panel["M3"], panel["nGDP"])
    panel["credit_to_gdp_growth"] = panel.groupby("countryname")["credit_to_gdp"].pct_change(fill_method=None) * 100.0
    panel["yield_curve_slope"] = panel["ltrate"] - panel["strate"]

    def gdp_weighted_top_coverage(frame, value_col):
        valid = frame.dropna(subset=[value_col, "nGDP_USD"]).copy()
        valid = valid.loc[valid["nGDP_USD"] > 0].sort_values("nGDP_USD", ascending=False)
        if len(valid) < min_countries:
            return pd.Series({"Value": np.nan, "n_countries": int(len(valid)), "gdp_coverage": np.nan})

        total_gdp = valid["nGDP_USD"].sum()
        valid["cum_gdp_share"] = valid["nGDP_USD"].cumsum() / total_gdp
        cutoff = valid["cum_gdp_share"] <= float(gdp_coverage)
        cutoff.iloc[0] = True
        if cutoff.sum() < min(min_countries, len(valid)):
            cutoff.iloc[:min(min_countries, len(valid))] = True
        selected = valid.loc[cutoff].copy()
        weights = selected["nGDP_USD"] / selected["nGDP_USD"].sum()
        return pd.Series({
            "Value": float((selected[value_col] * weights).sum()),
            "n_countries": int(len(selected)),
            "gdp_coverage": float(selected["nGDP_USD"].sum() / total_gdp),
        })

    global_credit = (
        panel.dropna(subset=["year", "credit_to_gdp_growth"])
        .groupby("year")
        .apply(lambda x: gdp_weighted_top_coverage(x, "credit_to_gdp_growth"))
        .reset_index()
    )
    global_credit["Date"] = pd.to_datetime(global_credit["year"].astype(int).astype(str) + "-12-31", errors="coerce")

    global_slope = (
        panel.dropna(subset=["year", "yield_curve_slope"])
        .groupby("year")
        .apply(lambda x: gdp_weighted_top_coverage(x, "yield_curve_slope"))
        .reset_index()
    )
    global_slope["Date"] = pd.to_datetime(global_slope["year"].astype(int).astype(str) + "-12-31", errors="coerce")

    return {
        "Froth Global Credit Growth ex-US": global_credit[["Date", "Value"]].sort_values("Date").reset_index(drop=True),
        "Froth Global Yield Curve ex-US": global_slope[["Date", "Value"]].sort_values("Date").reset_index(drop=True),
        "global_credit_growth_ex_us_details": global_credit[["Date", "n_countries", "gdp_coverage"]].sort_values("Date").reset_index(drop=True),
        "global_yield_curve_ex_us_details": global_slope[["Date", "n_countries", "gdp_coverage"]].sort_values("Date").reset_index(drop=True),
    }


def build_ipo_froth_indicators():
    rows = [
        (1980, 71, 6, 23, 32, 1, 1, 22, 64), (1981, 192, 8, 53, 28, 1, 1, 72, 40),
        (1982, 77, 5, 21, 27, 2, 3, 42, 36), (1983, 451, 7, 116, 26, 17, 4, 173, 39),
        (1984, 171, 8, 44, 26, 5, 3, 50, 52), (1985, 186, 9, 39, 21, 18, 10, 37, 43),
        (1986, 393, 8, 79, 20, 42, 11, 77, 40), (1987, 285, 8, 66, 23, 41, 14, 59, 66),
        (1988, 105, 8, 32, 30, 9, 9, 28, 61), (1989, 116, 8, 40, 34, 10, 9, 35, 66),
        (1990, 110, 9, 42, 38, 13, 12, 32, 75), (1991, 286, 10, 115, 40, 73, 26, 71, 63),
        (1992, 412, 10, 138, 33, 98, 24, 115, 58), (1993, 510, 9, 172, 34, 79, 15, 127, 69),
        (1994, 402, 9, 129, 32, 22, 5, 115, 56), (1995, 462, 8, 190, 41, 30, 6, 205, 56),
        (1996, 677, 8, 266, 39, 34, 5, 276, 56), (1997, 474, 10, 134, 28, 38, 8, 174, 42),
        (1998, 283, 9, 80, 28, 30, 11, 113, 49), (1999, 476, 5, 280, 59, 30, 6, 370, 68),
        (2000, 380, 6, 245, 64, 32, 8, 261, 70), (2001, 80, 12, 32, 40, 21, 26, 24, 70),
        (2002, 66, 15, 23, 35, 20, 30, 20, 65), (2003, 63, 11, 25, 40, 21, 33, 18, 67),
        (2004, 173, 8, 79, 46, 43, 25, 61, 66), (2005, 159, 13, 45, 28, 68, 43, 45, 49),
        (2006, 157, 13, 56, 36, 66, 42, 48, 56), (2007, 159, 9, 79, 50, 30, 19, 76, 76),
        (2008, 21, 14, 9, 43, 3, 14, 6, 67), (2009, 41, 15, 12, 29, 19, 46, 14, 43),
        (2010, 91, 11, 40, 44, 28, 31, 33, 73), (2011, 81, 11, 46, 57, 18, 22, 36, 83),
        (2012, 93, 12, 49, 53, 28, 30, 40, 87), (2013, 158, 12, 81, 52, 37, 23, 45, 78),
        (2014, 206, 11, 132, 64, 38, 18, 53, 75), (2015, 118, 10, 78, 65, 20, 17, 38, 76),
        (2016, 75, 10, 49, 65, 13, 17, 21, 71), (2017, 106, 12, 64, 60, 19, 18, 30, 80),
        (2018, 134, 10, 91, 68, 15, 11, 39, 77), (2019, 113, 10, 77, 69, 11, 10, 37, 70),
        (2020, 165, 9, 113, 68, 22, 13, 46, 73), (2021, 311, 11, 175, 56, 67, 22, 121, 64),
        (2022, 38, 8, 14, 37, 0, 0, 6, 17), (2023, 54, 10, 23, 43, 5, 9, 9, 44),
        (2024, 72, 14, 37, 51, 13, 18, 14, 57), (2025, 90, 12, 49, 54, 14, 16, 31, 74),
    ]
    ipo = pd.DataFrame(rows, columns=[
        "year", "ipo_count", "median_age", "vc_backed_count", "vc_backed_pct",
        "buyout_backed_count", "buyout_backed_pct", "technology_count", "tech_vc_backed_pct"
    ])
    ipo["Date"] = pd.to_datetime(ipo["year"].astype(str) + "-12-31", errors="coerce")
    ipo["technology_pct"] = safe_divide(ipo["technology_count"], ipo["ipo_count"]) * 100.0
    ipo["ipo_count_yoy_growth"] = ipo["ipo_count"].pct_change(fill_method=None) * 100.0
    ipo["vc_backed_count_yoy_growth"] = ipo["vc_backed_count"].pct_change(fill_method=None) * 100.0
    return {
        "Froth IPO Count": pd.DataFrame({"Date": ipo["Date"], "Value": ipo["ipo_count"]}),
        "Froth IPO Count YoY Growth": pd.DataFrame({"Date": ipo["Date"], "Value": ipo["ipo_count_yoy_growth"]}),
        "Froth IPO Median Age": pd.DataFrame({"Date": ipo["Date"], "Value": ipo["median_age"]}),
        "Froth VC-Backed IPO Share": pd.DataFrame({"Date": ipo["Date"], "Value": ipo["vc_backed_pct"]}),
        "Froth VC-Backed IPO Count Growth": pd.DataFrame({"Date": ipo["Date"], "Value": ipo["vc_backed_count_yoy_growth"]}),
        "Froth Buyout-Backed IPO Share": pd.DataFrame({"Date": ipo["Date"], "Value": ipo["buyout_backed_pct"]}),
        "Froth Technology IPO Count": pd.DataFrame({"Date": ipo["Date"], "Value": ipo["technology_count"]}),
        "Froth Technology IPO Share": pd.DataFrame({"Date": ipo["Date"], "Value": ipo["technology_pct"]}),
        "Froth VC-Backed Technology IPO Share": pd.DataFrame({"Date": ipo["Date"], "Value": ipo["tech_vc_backed_pct"]}),
        "ipo_issuance_details": ipo,
    }


def monthly_returns_from_price(price_df, value_col="Value"):
    monthly = price_df[["Date", value_col]].copy()
    monthly["Date"] = parse_date_series(monthly["Date"])
    monthly[value_col] = pd.to_numeric(monthly[value_col], errors="coerce")
    monthly = monthly.dropna(subset=["Date", value_col]).sort_values("Date")
    monthly = monthly.set_index("Date")[value_col].resample("ME").last().dropna()
    return monthly.pct_change(fill_method=None).dropna()


def rolling_volatility_ratio(returns, short_window=3, long_window=24):
    short_vol = returns.rolling(short_window, min_periods=short_window).std()
    long_vol = returns.rolling(long_window, min_periods=long_window).std()
    return safe_divide(short_vol, long_vol)


def rolling_capm_residual_volatility(asset_returns, market_returns, window=36, annualize=True):
    aligned = pd.concat([asset_returns.rename("asset"), market_returns.rename("market")], axis=1).dropna()
    out = pd.Series(np.nan, index=aligned.index, dtype=float)
    scale = np.sqrt(12.0) if annualize else 1.0
    for end in range(window - 1, len(aligned)):
        sample = aligned.iloc[end - window + 1:end + 1]
        market_var = sample["market"].var(ddof=1)
        if pd.isna(market_var) or market_var == 0:
            continue
        beta = sample["asset"].cov(sample["market"]) / market_var
        alpha = sample["asset"].mean() - beta * sample["market"].mean()
        residuals = sample["asset"] - alpha - beta * sample["market"]
        out.iloc[end] = residuals.std(ddof=1) * scale * 100.0
    return out


def build_fama_volatility_indicators(market_price_df, industry_price_df=None, industry_label="Industry"):
    market_returns = monthly_returns_from_price(market_price_df)
    market_vol = market_returns.rolling(36, min_periods=36).std() * np.sqrt(12.0) * 100.0
    market_vol_ratio = rolling_volatility_ratio(market_returns, short_window=3, long_window=24)
    out = {
        "Froth SPX 36M Volatility": pd.DataFrame({"Date": market_vol.index, "Value": market_vol.to_numpy()}),
        "Froth SPX Volatility Ratio (3M/24M)": pd.DataFrame({"Date": market_vol_ratio.index, "Value": market_vol_ratio.to_numpy()}),
    }
    if industry_price_df is not None:
        industry_returns = monthly_returns_from_price(industry_price_df)
        residual_vol = rolling_capm_residual_volatility(industry_returns, market_returns, window=36, annualize=True)
        out[f"Froth {industry_label} Residual Volatility (36M)"] = pd.DataFrame({
            "Date": residual_vol.index,
            "Value": residual_vol.to_numpy(),
        })
    return out


def build_us_froth_indicators(us_gmd, loans_to_nonfin=None):
    base = us_gmd.copy().sort_values("Date").reset_index(drop=True)
    combined_consumption = base["hcons"] + base["gcons"]
    hpi_yoy = base["HPI"].pct_change(fill_method=None)
    hpi_3y_annualized = safe_divide(base["HPI"], base["HPI"].shift(3)).pow(1.0 / 3.0) - 1.0
    hpi_2y_return = safe_divide(base["HPI"], base["HPI"].shift(2)) - 1.0
    hpi_first_year_return = safe_divide(base["HPI"].shift(1), base["HPI"].shift(2)) - 1.0
    out = {"us_gmd": base.copy()}

    series_map = {
        "Froth Real M2 Growth": (base["M2"].pct_change(fill_method=None) * 100) - base["infl"],
        "Froth Real M3 Growth": (base["M3"].pct_change(fill_method=None) * 100) - base["infl"],
        "Froth Credit-to-GDP Gap": safe_divide(base["M3"], base["nGDP"]) - safe_divide(base["M3"], base["nGDP"]).rolling(10, min_periods=10).mean(),
        "Froth Housing Momentum (3Y)": safe_divide(base["HPI"], base["HPI"].shift(3)),
        "Froth Housing Momentum Gap": safe_divide(hpi_yoy, hpi_3y_annualized),
        "Froth HPI Acceleration (2Y)": hpi_2y_return - hpi_first_year_return,
        "Froth Yield Curve": base["ltrate"] - base["strate"],
        "Froth Real Policy Rate": base["cbrate"] - base["infl"],
        "Froth REER": base["REER"],
        "Froth REER Z-Score": standardize_full_sample(base["REER"]),
        "Froth Current Account (3Y Avg)": base["CA_GDP"].rolling(3, min_periods=3).mean(),
        "Froth Debt Velocity": base["govdebt_GDP"].diff(),
        "Froth Fiscal Deficit": base["govdef_GDP"],
        "Froth Fixed Investment / GDP": safe_divide(base["finv"], base["nGDP"]),
        "Froth Consumption Gap": (base["hcons"].pct_change(fill_method=None) * 100) - (base["rGDP"].pct_change(fill_method=None) * 100),
        "Froth Combined Consumption Gap": (combined_consumption.pct_change(fill_method=None) * 100) - (base["rGDP"].pct_change(fill_method=None) * 100),
        "Froth Unemployment Acceleration": base["unemp"].diff().diff(),
        "Froth Output Gap (HP)": hp_filter_cycle(base["rGDP"]),
    }

    for name, values in series_map.items():
        out[name] = pd.DataFrame({"Date": base["Date"], "Value": values})

    out.update(build_nonfinancial_loan_indicators(base, loans_to_nonfin))
    return out


@st.cache_data(show_spinner="Loading data files…")
def load_all_data():
    def load_xlsx(filename):
        df = pd.read_excel(p(filename), skiprows=4, header=0, usecols=[0, 1])
        df.columns = ["Date", "Value"]
        df["Date"] = parse_date_series(df["Date"])
        return df.dropna(subset=["Date", "Value"]).reset_index(drop=True)

    def load_shiller_cape():
        try:
            shiller_raw = pd.read_excel(p("shiller_data.xls"), sheet_name="Data", skiprows=6).iloc[1:].copy()
            shiller_raw = shiller_raw.rename(columns={"Unnamed: 0": "Date", "P/E10 or": "Value"})
            shiller_ratio = shiller_raw[["Date", "Value"]].dropna().copy()
            shiller_ratio["Date"] = shiller_ratio["Date"].astype(float)
            shiller_ratio["Year"] = shiller_ratio["Date"].astype(int)
            shiller_ratio["Month"] = ((shiller_ratio["Date"] % 1) * 12 + 1).round().astype(int).clip(1, 12)
            shiller_ratio["Date"] = parse_date_series(shiller_ratio[["Year", "Month"]].assign(day=1))
            return shiller_ratio[["Date", "Value"]].dropna().reset_index(drop=True)
        except Exception:
            fallback_file = "S&P 500 Shiller CAPE Ratio 2026-02-13 19_58_32.xlsx"
            return load_xlsx(fallback_file)

    def load_spx_technicals(filename):
        expected_cols = [
            "Date", "Price", "SMA50", "SMA200", "Golden_Cross", "Death_Cross",
            "RSI_14", "MACD", "MACD_Signal", "BB_Upper", "BB_Lower"
        ]

        raw = pd.read_csv(p(filename), header=None)
        raw = raw.dropna(how="all").reset_index(drop=True)

        # Handle Yahoo-style exports that include header/ticker/date metadata rows.
        first_col = raw.iloc[:, 0].astype(str).str.strip() if not raw.empty else pd.Series(dtype=str)
        if len(raw) >= 3 and first_col.iloc[0] == "Price" and first_col.iloc[1] == "Ticker" and first_col.iloc[2] == "Date":
            technicals = raw.iloc[3:, :len(expected_cols)].copy()
            technicals.columns = expected_cols
        else:
            technicals = pd.read_csv(p(filename))
            technicals = technicals.iloc[:, :len(expected_cols)].copy()
            technicals.columns = expected_cols

        technicals["Date"] = parse_date_series(technicals["Date"])
        technicals = technicals.dropna(subset=["Date"]).reset_index(drop=True)
        for col in ["Price", "SMA50", "SMA200", "RSI_14", "MACD", "MACD_Signal", "BB_Upper", "BB_Lower"]:
            technicals[col] = pd.to_numeric(technicals[col], errors="coerce")
        technicals["Golden_Cross"] = technicals["Golden_Cross"].astype(str).str.strip().str.lower() == "true"
        technicals["Death_Cross"] = technicals["Death_Cross"].astype(str).str.strip().str.lower() == "true"
        return technicals

    # CSV files
    yield_curve = pd.read_csv(p("10Y-2Y.csv")).rename(
        columns={"observation_date": "Date", "T10Y2Y": "Value"}
    )
    bbb_spread = pd.read_csv(p("BBB-treasury_credit_spread.csv")).rename(
        columns={"observation_date": "Date", "BAMLC0A4CBBB": "Value"}
    )
    buffett_csv = pd.read_csv(p("Buffet_indicator.csv")).rename(
        columns={"observation_date": "Date", "DDDM01USA156NWDB": "Value"}
    )
    erp = pd.read_csv(p("ERP.csv")).rename(
        columns={"observation_date": "Date", "TENEXPCHAREARISPRE": "Value"}
    )
    fear_greed = pd.read_csv(p("fear_greed_indicator.csv")).rename(columns={"Fear Greed": "Value"})
    financial_stress = pd.read_csv(p("financial_stress_index.csv")).rename(
        columns={"observation_date": "Date", "STLFSI4": "Value"}
    )
    misery = pd.read_csv(p("Misery_index_urban_USA.csv")).rename(
        columns={"observation_date": "Date", "UNRATE_CPIAUCSL_PC1": "Value"}
    )
    sahm = pd.read_csv(p("Sahm_indicator.csv")).rename(
        columns={"observation_date": "Date", "SAHMREALTIME": "Value"}
    )
    vix = pd.read_csv(p("VIX_data.csv")).rename(
        columns={"observation_date": "Date", "VIXCLS": "Value"}
    )
    sp500_total_return = pd.read_csv(p("sp500_total_return_daily.csv")).rename(
        columns={"Total_Return": "Value", "Total Return": "Value"}
    )

    # OFR / margin debt
    with open(p("margin_debt_GDP.csv"), "r") as f:
        content = f.read().replace('"', "")
    ofr_raw = pd.read_csv(StringIO(content))
    ofr_raw["Date"] = parse_date_series(ofr_raw["Date"])
    ofr_raw = ofr_raw.dropna(subset=["Date"])
    ofr_fsi = ofr_raw[["Date", "OFR FSI"]].rename(columns={"OFR FSI": "Value"})
    equity_valuation = ofr_raw[["Date", "Equity valuation"]].rename(columns={"Equity valuation": "Value"})

    # SKEW
    skew = pd.read_csv(p("CBOE_SKEW_historical.csv"), usecols=["Date", "Close"]).rename(columns={"Close": "Value"})
    skew["Date"] = parse_date_series(skew["Date"])
    skew = skew.dropna()

    # Shiller
    shiller_ratio = load_shiller_cape()

    # Sentiment
    sentiment = pd.read_excel(p("sentiment.xls"), skiprows=3, header=0, usecols=[0, 1, 2, 3])
    sentiment.columns = ["Date", "Bullish", "Neutral", "Bearish"]
    sentiment["Date"] = parse_date_series(sentiment["Date"])
    sentiment = sentiment.dropna(subset=["Date"])

    # Brent & Gold
    brent = pd.read_excel(p("brent_price.xlsx"), skiprows=1, header=0, usecols=[0, 1])
    brent.columns = ["Date", "Value"]
    brent["Date"] = parse_date_series(brent["Date"])
    brent = brent.dropna()

    gold = pd.read_excel(p("gold_prices.xlsx"), skiprows=1, header=0, usecols=[0, 1])
    gold.columns = ["Date", "Value"]
    gold["Date"] = parse_date_series(gold["Date"])
    gold = gold.dropna()

    # Standard xlsx files
    tobin_q = load_xlsx("Tobin Q 2026-02-13 19_56_32.xlsx")
    pe_ratio = load_xlsx("S&P 500 PE Ratio 2026-02-13 20_01_50.xlsx")
    pb_ratio = load_xlsx("S&P 500 Price to Book Value 2026-02-13 20_01_32.xlsx")
    ps_ratio = load_xlsx("S&P 500 Price to Sales 2026-02-13 20_01_40.xlsx")
    earnings_yield = load_xlsx("S&P 500 Earnings Yield 2026-02-13 20_02_48.xlsx")
    dividend_yield = load_xlsx("S&P 500 Dividend Yield 2026-02-13 20_17_09.xlsx")
    corp_margin = load_xlsx("Corporate Profit Margin (After Tax)  2026-02-13 19_59_45.xlsx")
    mktcap_gdp = load_xlsx("USA Ratio of Total Market Cap over GDP 2026-02-13 19_58_12.xlsx")
    insider_ratio = load_xlsx("Insider Buy_Sell Ratio - USA Overall Market 2026-02-13 20_10_59.xlsx")
    unemployment = load_xlsx("Civilian Unemployment Rate 2026-02-13 20_08_59.xlsx")
    gdp_recession = load_xlsx("GDP-Based Recession Indicator Index 2026-02-13 20_04_13.xlsx")

    # Technicals
    technicals = load_spx_technicals("spx_technicals_corrected.csv")

    # GMD macro panel and derived froth indicators
    gmd_raw = pd.read_csv(p("GMD.csv"))
    us_gmd = build_us_gmd_frame(gmd_raw)
    loans_to_nonfin = load_loans_to_nonfin()
    froth_macro = build_us_froth_indicators(us_gmd, loans_to_nonfin)
    froth_macro.update(build_global_gmd_indicators(gmd_raw))
    froth_macro.update(build_ipo_froth_indicators())
    froth_macro.update(build_fama_volatility_indicators(sp500_total_return))

    # Gold / Brent
    gold_brent = pd.merge(gold, brent, on="Date", how="inner").iloc[1:].copy()
    gold_brent["gold_brent_ratio"] = gold_brent["Value_x"] / gold_brent["Value_y"]

    return {
        "yield_curve": yield_curve,
        "bbb_spread": bbb_spread,
        "buffett_csv": buffett_csv,
        "erp": erp,
        "fear_greed": fear_greed,
        "financial_stress": financial_stress,
        "misery": misery,
        "sahm": sahm,
        "vix": vix,
        "sp500_total_return": sp500_total_return,
        "ofr_fsi": ofr_fsi,
        "equity_valuation": equity_valuation,
        "skew": skew,
        "shiller_ratio": shiller_ratio,
        "sentiment": sentiment,
        "brent": brent,
        "gold": gold,
        "tobin_q": tobin_q,
        "pe_ratio": pe_ratio,
        "pb_ratio": pb_ratio,
        "ps_ratio": ps_ratio,
        "earnings_yield": earnings_yield,
        "dividend_yield": dividend_yield,
        "corp_margin": corp_margin,
        "mktcap_gdp": mktcap_gdp,
        "insider_ratio": insider_ratio,
        "unemployment": unemployment,
        "gdp_recession": gdp_recession,
        "technicals": technicals,
        "gold_brent": gold_brent,
        "us_gmd": us_gmd,
        "loans_to_nonfin": loans_to_nonfin,
        "froth_macro": froth_macro,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 4. OBJECT CONSTRUCTION
# ══════════════════════════════════════════════════════════════════════════════
def make_indicator(df, value_col, indicator_name):
    out = df.copy()
    if "Date" not in out.columns:
        raise ValueError(f"'Date' column missing for indicator {indicator_name}")

    out["Date"] = parse_date_series(out["Date"])
    out[value_col] = pd.to_numeric(out[value_col], errors="coerce")
    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.dropna(subset=["Date", value_col]).sort_values("Date")
    out = out.drop_duplicates(subset=["Date"], keep="last")
    out = out.rename(columns={value_col: "value"})
    out["inverted"] = indicator_name in INVERSE_INDICATORS
    return out[["Date", "value", "inverted"]]


def load_total_return_series():
    tr = pd.read_csv(p("sp500_total_return_daily.csv"))
    tr.columns = [str(c).strip() for c in tr.columns]

    possible_date_cols = ["Date", "date", "DATE"]
    date_col = next((c for c in possible_date_cols if c in tr.columns), None)
    if date_col is None:
        raise ValueError(
            f"Could not find a date column in sp500_total_return_daily.csv. "
            f"Columns found: {list(tr.columns)}"
        )

    possible_price_cols = [
        "Close", "close", "Adj Close", "adj_close", "AdjClose",
        "Total Return", "Total_Return", "total_return", "TR", "Index", "index",
        "PX_LAST", "Price", "price", "Value", "value"
    ]
    price_col = next((c for c in possible_price_cols if c in tr.columns), None)

    if price_col is None:
        candidate_cols = [c for c in tr.columns if c != date_col]
        found = None
        for c in candidate_cols:
            test = pd.to_numeric(tr[c], errors="coerce")
            if test.notna().sum() > 0:
                found = c
                break
        price_col = found

    if price_col is None:
        raise ValueError(
            f"Could not find a usable market series column in sp500_total_return_daily.csv. "
            f"Columns found: {list(tr.columns)}"
        )

    tr[date_col] = parse_date_series(tr[date_col])
    tr[price_col] = pd.to_numeric(tr[price_col], errors="coerce")
    tr = tr.dropna(subset=[date_col, price_col]).sort_values(date_col)
    tr = tr.rename(columns={date_col: "Date", price_col: "spx_close"})
    tr = tr.drop_duplicates(subset=["Date"], keep="last")

    return tr[["Date", "spx_close"]].copy()


def prep_spx_from_technicals(technicals):
    spx = technicals.copy()
    spx["Date"] = parse_date_series(spx["Date"])
    spx["Price"] = pd.to_numeric(spx["Price"], errors="coerce")
    spx = spx.dropna(subset=["Date", "Price"]).sort_values("Date")
    spx = spx.rename(columns={"Price": "spx_close"})
    spx = spx.drop_duplicates(subset=["Date"], keep="last")
    return spx[["Date", "spx_close"]].copy()


def build_notebook_objects(d):
    technicals = d["technicals"]
    sentiment = d["sentiment"]
    gold_brent = d["gold_brent"]
    froth_macro = d.get("froth_macro", {})

    indicator_configs = {
        "Shiller CAPE":              (d["shiller_ratio"],    "Value"),
        "PE Ratio":                  (d["pe_ratio"],         "Value"),
        "PB Ratio":                  (d["pb_ratio"],         "Value"),
        "PS Ratio":                  (d["ps_ratio"],         "Value"),
        "Earnings Yield":            (d["earnings_yield"],   "Value"),
        "Dividend Yield":            (d["dividend_yield"],   "Value"),
        "Tobin Q":                   (d["tobin_q"],          "Value"),
        "Buffet Indicator":          (d["mktcap_gdp"],       "Value"),
        "Corp Profit Margin":        (d["corp_margin"],      "Value"),
        "Equity Risk Premium":       (d["erp"],              "Value"),
        "Equity Valuation (OFR)":    (d["equity_valuation"], "Value"),

        "Yield Curve (10Y–2Y)":      (d["yield_curve"],      "Value"),
        "BBB–Treasury Spread":       (d["bbb_spread"],       "Value"),
        "Financial Stress (StL)":    (d["financial_stress"], "Value"),
        "OFR Financial Stress":      (d["ofr_fsi"],          "Value"),
        "VIX":                       (d["vix"],              "Value"),
        "CBOE SKEW":                 (d["skew"],             "Value"),

        "Fear & Greed":              (d["fear_greed"],        "Value"),
        "Insider Buy/Sell":          (d["insider_ratio"],     "Value"),
        "AAII Bullish":              (sentiment.rename(columns={"Bullish": "Value"}), "Value"),
        "AAII Bearish":              (sentiment.rename(columns={"Bearish": "Value"}), "Value"),
        "AAII Bull-Bear Spread":     (sentiment.assign(Value=sentiment["Bullish"] - sentiment["Bearish"]), "Value"),

        "Sahm Rule":                 (d["sahm"],            "Value"),
        "Misery Index":              (d["misery"],          "Value"),
        "GDP Recession Indicator":   (d["gdp_recession"],   "Value"),

        "Gold / Brent Ratio":        (gold_brent,           "gold_brent_ratio"),
        "Gold":                      (d["gold"],            "Value"),
        "Brent Oil":                 (d["brent"],           "Value"),

        "SPX SMA50":                 (technicals,           "SMA50"),
        "SPX SMA200":                (technicals,           "SMA200"),
        "RSI (14)":                  (technicals,           "RSI_14"),
        "MACD":                      (technicals,           "MACD"),
        "BB Upper":                  (technicals,           "BB_Upper"),
        "BB Lower":                  (technicals,           "BB_Lower"),
        "Froth SPX 36M Volatility":  (froth_macro["Froth SPX 36M Volatility"], "Value"),
        "Froth SPX Volatility Ratio (3M/24M)": (froth_macro["Froth SPX Volatility Ratio (3M/24M)"], "Value"),

        "Froth Real M2 Growth":      (froth_macro["Froth Real M2 Growth"], "Value"),
        "Froth Real M3 Growth":      (froth_macro["Froth Real M3 Growth"], "Value"),
        "Froth Credit-to-GDP Gap":   (froth_macro["Froth Credit-to-GDP Gap"], "Value"),
        "Froth Global Credit Growth ex-US": (froth_macro["Froth Global Credit Growth ex-US"], "Value"),
        "Froth Housing Momentum (3Y)": (froth_macro["Froth Housing Momentum (3Y)"], "Value"),
        "Froth Housing Momentum Gap": (froth_macro["Froth Housing Momentum Gap"], "Value"),
        "Froth HPI Acceleration (2Y)": (froth_macro["Froth HPI Acceleration (2Y)"], "Value"),
        "Froth Yield Curve":         (froth_macro["Froth Yield Curve"], "Value"),
        "Froth Global Yield Curve ex-US": (froth_macro["Froth Global Yield Curve ex-US"], "Value"),
        "Froth Real Policy Rate":    (froth_macro["Froth Real Policy Rate"], "Value"),
        "Froth REER Z-Score":        (froth_macro["Froth REER Z-Score"], "Value"),
        "Froth Current Account (3Y Avg)": (froth_macro["Froth Current Account (3Y Avg)"], "Value"),
        "Froth Debt Velocity":       (froth_macro["Froth Debt Velocity"], "Value"),
        "Froth Fiscal Deficit":      (froth_macro["Froth Fiscal Deficit"], "Value"),
        "Froth Fixed Investment / GDP": (froth_macro["Froth Fixed Investment / GDP"], "Value"),
        "Froth Consumption Gap":     (froth_macro["Froth Consumption Gap"], "Value"),
        "Froth Combined Consumption Gap": (froth_macro["Froth Combined Consumption Gap"], "Value"),
        "Froth Unemployment Acceleration": (froth_macro["Froth Unemployment Acceleration"], "Value"),
        "Froth Output Gap (HP)":     (froth_macro["Froth Output Gap (HP)"], "Value"),

        "Froth IPO Count": (froth_macro["Froth IPO Count"], "Value"),
        "Froth IPO Count YoY Growth": (froth_macro["Froth IPO Count YoY Growth"], "Value"),
        "Froth IPO Median Age": (froth_macro["Froth IPO Median Age"], "Value"),
        "Froth VC-Backed IPO Share": (froth_macro["Froth VC-Backed IPO Share"], "Value"),
        "Froth VC-Backed IPO Count Growth": (froth_macro["Froth VC-Backed IPO Count Growth"], "Value"),
        "Froth Buyout-Backed IPO Share": (froth_macro["Froth Buyout-Backed IPO Share"], "Value"),
        "Froth Technology IPO Count": (froth_macro["Froth Technology IPO Count"], "Value"),
        "Froth Technology IPO Share": (froth_macro["Froth Technology IPO Share"], "Value"),
        "Froth VC-Backed Technology IPO Share": (froth_macro["Froth VC-Backed Technology IPO Share"], "Value"),
    }

    for optional_name in [
        "Froth Nonfinancial Debt Service",
        "Froth Nonfinancial Loan Acceleration",
    ]:
        if optional_name in froth_macro:
            indicator_configs[optional_name] = (froth_macro[optional_name], "Value")

    indicators = {name: make_indicator(df, col, name) for name, (df, col) in indicator_configs.items()}
    spx_c = prep_spx_from_technicals(technicals)

    return indicators, spx_c, technicals


# ══════════════════════════════════════════════════════════════════════════════
# 5. DATA UTILS
# ══════════════════════════════════════════════════════════════════════════════
def ensure_date_col(df):
    temp = df.copy()
    if "Date" not in temp.columns:
        if isinstance(temp.index, pd.DatetimeIndex):
            temp = temp.reset_index().rename(columns={temp.index.name or "index": "Date"})
        else:
            temp = temp.reset_index().rename(columns={temp.columns[0]: "Date"})
    temp["Date"] = pd.to_datetime(temp["Date"], errors="coerce", utc=True).dt.tz_localize(None)
    temp = temp.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    return temp


def find_value_column(df):
    cols = [c for c in df.columns if c != "Date"]
    return cols[0] if cols else None


def fmt_pct(x):
    return f"{x*100:.2f}%" if pd.notna(x) else "—"


def calc_var_cvar(x, alpha=0.05):
    s = pd.Series(x).dropna()
    if len(s) < 8:
        return np.nan, np.nan
    s = np.sort(s.values)
    idx = max(1, int(alpha * len(s)))
    idx = min(idx, len(s) - 1)
    return s[idx], s[:idx].mean()


def get_clean_indicator(indicators, indicator_name):
    raw = ensure_date_col(indicators[indicator_name].copy())
    value_col = find_value_column(raw)
    if value_col is None:
        return pd.DataFrame(columns=["Date", indicator_name])

    raw[value_col] = pd.to_numeric(raw[value_col], errors="coerce")
    raw = raw.dropna(subset=[value_col])[["Date", value_col]].rename(columns={value_col: indicator_name})
    raw["Date"] = pd.to_datetime(raw["Date"], errors="coerce", utc=True).dt.tz_localize(None)
    raw = raw.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    return raw.copy()


def build_spx_forward(spx_c, horizon_years):
    H = int(round(float(horizon_years) * 252))
    spx = ensure_date_col(spx_c).copy()
    px_col = find_value_column(spx)
    spx = spx[["Date", px_col]].rename(columns={px_col: "close"})
    spx["fwd"] = spx["close"].shift(-H) / spx["close"] - 1
    spx = spx.dropna(subset=["fwd"]).copy()
    return spx.copy()


def get_spx_price_series(spx_c):
    spx = ensure_date_col(spx_c).copy()
    px_col = find_value_column(spx)
    spx = spx[["Date", px_col]].rename(columns={px_col: "close"}).dropna()
    spx["Date"] = pd.to_datetime(spx["Date"], errors="coerce", utc=True).dt.tz_localize(None)
    spx = spx.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    return spx.copy()


def get_regime_mask(technicals, merged, regime_mode="all"):
    if regime_mode == "all":
        return pd.Series(True, index=merged.index)

    needed = {"Date", "Price", "SMA200"}
    if technicals is None or not needed.issubset(set(technicals.columns)):
        return pd.Series(True, index=merged.index)

    tech = ensure_date_col(technicals)[["Date", "Price", "SMA200"]].copy()
    tech["Price"] = pd.to_numeric(tech["Price"], errors="coerce")
    tech["SMA200"] = pd.to_numeric(tech["SMA200"], errors="coerce")
    tech = tech.dropna(subset=["Date", "Price", "SMA200"]).sort_values("Date")

    if tech.empty or merged.empty or "Date" not in merged.columns:
        return pd.Series(True, index=merged.index)

    frame = merged[["Date"]].copy()
    frame["_orig_index"] = merged.index
    frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce", utc=True).dt.tz_localize(None)
    frame = frame.dropna(subset=["Date"]).sort_values("Date")

    aligned = pd.merge_asof(
        frame,
        tech,
        on="Date",
        direction="backward",
        tolerance=pd.Timedelta(days=7),
    ).set_index("_orig_index")

    if regime_mode == "bull":
        mask = aligned["Price"] > aligned["SMA200"]
    elif regime_mode == "bear":
        mask = aligned["Price"] < aligned["SMA200"]
    else:
        mask = pd.Series(True, index=aligned.index)

    return mask.reindex(merged.index).fillna(False).astype(bool)


def percentile_as_of(raw_df, value_col, anchor_date, value):
    hist = raw_df.loc[raw_df["Date"] <= anchor_date, value_col].dropna()
    if len(hist) < 10 or pd.isna(value):
        return np.nan
    return (hist <= value).mean()


def lookup_indicator_value_at(indicators, indicator_name, anchor_date):
    raw = get_clean_indicator(indicators, indicator_name)
    if raw.empty:
        return np.nan, np.nan, raw, indicator_name

    value_col = indicator_name
    match = pd.merge_asof(
        pd.DataFrame({"Date": [pd.to_datetime(anchor_date, utc=True).tz_localize(None)]}),
        raw[["Date", value_col]],
        on="Date",
        direction="backward"
    )
    val = match[value_col].iloc[0] if value_col in match.columns else np.nan
    actual_date = match["Date"].iloc[0] if "Date" in match.columns else pd.NaT
    return val, actual_date, raw, value_col


def expanding_percentile_series(s, min_periods=20, invert=False):
    s = pd.Series(s).astype(float)
    result = np.full(len(s), np.nan, dtype=float)
    sorted_vals = []
    valid_n = 0
    for i, x in enumerate(s.values):
        if pd.notna(x):
            insort(sorted_vals, float(x))
            valid_n += 1
            if valid_n >= min_periods:
                pct = bisect_right(sorted_vals, float(x)) / valid_n
                result[i] = 1.0 - pct if invert else pct
    return pd.Series(result, index=s.index)


def identify_spx_downturns(spx_c, drawdown_threshold=0.20, recovery_threshold=None, min_spacing_days=None):
    spx = get_spx_price_series(spx_c).copy()
    spx["peak"] = spx["close"].cummax()
    spx["drawdown"] = spx["close"] / spx["peak"] - 1

    events = []
    i = 1
    last_start = None

    while i < len(spx):
        breached = (
            spx["drawdown"].iloc[i] <= -drawdown_threshold and
            spx["drawdown"].iloc[i - 1] > -drawdown_threshold
        )
        spacing_ok = (
            min_spacing_days is None or
            last_start is None or
            (spx["Date"].iloc[i] - last_start).days >= min_spacing_days
        )

        if breached and spacing_ok:
            start_idx = i
            start_date = spx["Date"].iloc[start_idx]
            j = i
            # Simpler rule: the event stays active while drawdown remains deeper
            # than the chosen threshold and ends once it climbs back above it.
            while j < len(spx) - 1 and spx["drawdown"].iloc[j] <= -drawdown_threshold:
                j += 1
            end_idx = j
            event_slice = spx.iloc[start_idx:end_idx + 1].copy()
            trough_local_idx = event_slice["drawdown"].idxmin()
            trough_row = spx.loc[trough_local_idx]

            events.append({
                "start_date": start_date,
                "end_date": spx["Date"].iloc[end_idx],
                "trough_date": trough_row["Date"],
                "start_price": spx["close"].iloc[start_idx],
                "trough_price": trough_row["close"],
                "end_price": spx["close"].iloc[end_idx],
                "max_drawdown": trough_row["drawdown"],
                "days_to_trough": int((trough_row["Date"] - start_date).days),
                "event_days": int((spx["Date"].iloc[end_idx] - start_date).days),
            })
            last_start = start_date
            i = end_idx + 1
        else:
            i += 1

    return spx, pd.DataFrame(events)


@st.cache_data(show_spinner=False)
def compute_downturn_heatmap_matrix(indicators, spx_c, selected, drawdown_level):
    selected = list(selected)
    _, events = identify_spx_downturns(spx_c, float(drawdown_level), None, None)
    if events.empty:
        return pd.DataFrame()

    lookbacks = [
        ("3M", int(round(0.25 * 252))),
        ("6M", int(round(0.50 * 252))),
        ("1Y", int(round(1.00 * 252))),
        ("2Y", int(round(2.00 * 252))),
        ("3Y", int(round(3.00 * 252))),
        ("4Y", int(round(4.00 * 252))),
        ("5Y", int(round(5.00 * 252))),
    ]

    rows = []
    for _, ev in events.iterrows():
        start_date = pd.to_datetime(ev["start_date"])
        for lb_name, lb_days in lookbacks:
            anchor = start_date - pd.Timedelta(days=int(lb_days * 365 / 252))
            rec = {"event_start": start_date, "lookback": lb_name}
            for lbl in selected:
                val, _, raw, value_col = lookup_indicator_value_at(indicators, lbl, anchor)
                rec[f"{lbl}__pct"] = percentile_as_of(raw, value_col, anchor, val)
            rows.append(rec)

    lookback_df = pd.DataFrame(rows)
    if lookback_df.empty:
        return pd.DataFrame()

    avg_pct = pd.DataFrame(index=selected, columns=[x[0] for x in lookbacks], dtype=float)
    for lbl in selected:
        for lb_name, _ in lookbacks:
            mask = lookback_df["lookback"] == lb_name
            avg_pct.loc[lbl, lb_name] = lookback_df.loc[mask, f"{lbl}__pct"].mean()

    return avg_pct.astype(float)


def render_downturn_heatmap_section(indicators, spx_c, selected, key_prefix="downturn_heatmap", default_drawdown=0.20):
    if not selected:
        return

    st.markdown("<div class='section-divider'>Indicator Percentiles Before Downturns</div>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class="info-card">
            <div style="font-size:12px;font-weight:800;color:#0f172a;margin-bottom:4px;">What this shows</div>
            <div style="font-size:12px;color:#64748b;line-height:1.45;">
                This heatmap averages where the selected indicators ranked before historical SPX downturns of the chosen size. Higher values mean the indicator was usually nearer its expensive or stretched end before those downturns began.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    drawdown_level = st.slider(
        "Downturn size for heatmap",
        0.10,
        0.40,
        float(default_drawdown),
        0.01,
        key=f"{key_prefix}_drawdown"
    )

    avg_pct_float = compute_downturn_heatmap_matrix(
        indicators=indicators,
        spx_c=spx_c,
        selected=tuple(selected),
        drawdown_level=float(drawdown_level)
    )
    if avg_pct_float.empty:
        st.info("No downturn episodes were found for the chosen drawdown size.")
        return

    heat_text = [[f"{v:.0%}" if pd.notna(v) else "—" for v in row] for row in avg_pct_float.values]
    heat = go.Figure(data=go.Heatmap(
        z=avg_pct_float.values.tolist(),
        x=avg_pct_float.columns.tolist(),
        y=avg_pct_float.index.tolist(),
        colorscale=[
            [0.00, "#2f7d32"],
            [0.35, "#b8d98a"],
            [0.50, "#fff7bc"],
            [0.70, "#f7b267"],
            [1.00, "#c0392b"],
        ],
        zmin=0,
        zmax=1,
        text=heat_text,
        texttemplate="%{text}",
        hovertemplate="Indicator: %{y}<br>Lookback: %{x}<br>Avg percentile: %{z:.1%}<extra></extra>",
        xgap=1,
        ygap=1,
        colorbar=dict(title="Percentile", tickformat=".0%")
    ))
    apply_layout(heat, "Average historical percentile before downturn start", 220 + 36 * max(2, len(selected)))
    heat.update_xaxes(title="Lookback from downturn start")
    heat.update_yaxes(title="")
    st.plotly_chart(heat, use_container_width=True)




# ══════════════════════════════════════════════════════════════════════════════
# 7. SHARED UI STATE
# ══════════════════════════════════════════════════════════════════════════════
def init_state(all_indicator_names):
    names = list(all_indicator_names)
    defaults = [
        "Shiller CAPE",
        "Tobin Q",
        "Buffet Indicator",
        "Froth Credit-to-GDP Gap",
        "Froth Housing Momentum Gap",
    ]
    defaults = [x for x in defaults if x in names]
    if not defaults:
        defaults = names[:min(4, len(names))]

    current = st.session_state.get("combo_selected")
    if current is None:
        st.session_state["combo_selected"] = defaults
    else:
        cleaned = [x for x in current if x in names]
        st.session_state["combo_selected"] = cleaned if cleaned else defaults


def render_shared_indicator_picker(all_indicator_names):
    names = list(all_indicator_names)
    current = [x for x in st.session_state.get("combo_selected", []) if x in names]
    selected = st.multiselect(
        "Selected indicators",
        names,
        default=current,
        key="combo_selected_picker"
    )
    st.session_state["combo_selected"] = selected
    return selected


def get_combo_selected():
    return list(st.session_state.get("combo_selected", []))


def render_tab_intro(title, body):
    st.markdown(
        f"""
        <div class="info-card">
            <div style="font-size:12px;font-weight:800;color:{TEXT};margin-bottom:4px;">{title}</div>
            <div style="font-size:12px;color:{MUTED};line-height:1.45;">{body}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def apply_layout(fig, title, height=420):
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor=BG,
        plot_bgcolor=BG,
        height=int(height),
        margin=dict(l=50, r=30, t=58, b=42),
        title=dict(text=title, x=0, font=dict(size=15, color=TEXT)),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        font=dict(color=TEXT),
        hovermode="x unified",
    )
    fig.update_xaxes(showgrid=True, gridcolor=BORDER, zeroline=False, linecolor=BORDER)
    fig.update_yaxes(showgrid=True, gridcolor=BORDER, zeroline=False, linecolor=BORDER)
    return fig


def _apply_rule(flags, selected, rule, n_val, composite_score, cutoff):
    flags = pd.DataFrame(flags).fillna(False).astype(bool)
    selected = [x for x in selected if x in flags.columns]
    if not selected:
        return pd.Series(False, index=flags.index), "No selected indicators"

    if rule == "all":
        return flags[selected].all(axis=1), "All selected indicators"
    if rule == "at_least_n":
        n_req = int(max(1, min(n_val, len(selected))))
        return flags[selected].sum(axis=1) >= n_req, f"At least {n_req} selected indicators"
    if rule == "composite":
        return pd.Series(composite_score, index=flags.index).astype(float) >= float(cutoff), f"Composite >= {cutoff:.0%}"
    return flags[selected].any(axis=1), "Any selected indicator"


def mask_to_periods(dates, mask):
    dates = pd.Series(pd.to_datetime(dates)).reset_index(drop=True)
    mask = pd.Series(mask).fillna(False).astype(bool).reset_index(drop=True)
    periods = []
    start = None

    for dt, is_on in zip(dates, mask):
        if is_on and start is None:
            start = dt
        elif not is_on and start is not None:
            periods.append((start, prev_dt))
            start = None
        prev_dt = dt

    if start is not None and len(dates):
        periods.append((start, dates.iloc[-1]))

    return periods


def mask_to_starts(dates, mask):
    dates = pd.Series(pd.to_datetime(dates)).reset_index(drop=True)
    mask = pd.Series(mask).fillna(False).astype(bool).reset_index(drop=True)
    starts = []
    prev_on = False

    for dt, is_on in zip(dates, mask):
        if is_on and not prev_on:
            starts.append(dt)
        prev_on = bool(is_on)

    return starts

def render_indicator_page(indicators, spx_c):
    st.subheader("Indicator Analysis")
    render_tab_intro(
        "How this works",
        "Indicator chooses the series to analyze. Horizon sets how far ahead SPX forward returns are measured. "
        "Rolling window controls the lookback used for the rolling correlation chart. Time series toggles the historical indicator chart on or off."
    )
    with st.expander("ℹ️ How to use this tab", expanded=False):
        st.markdown("""
        - **Indicator** – The individual series you want to compare against future SPX returns.
        - **Horizon (Y)** – Forward return window used when calculating SPX performance after each historical date.
        - **Rolling window (Y)** – Lookback length used in the rolling correlation chart.
        - **Time series** – Shows or hides the historical indicator path chart above the scatter plot.
        """)
    all_indicator_names = sorted(indicators.keys())

    label = st.selectbox(
        "Indicator",
        all_indicator_names,
        index=all_indicator_names.index("Shiller CAPE") if "Shiller CAPE" in all_indicator_names else 0,
        key="ind_label"
    )
    c1, c2 = st.columns(2)
    with c1:
        h_val = st.slider("Horizon (Y)", 0.25, 10.0, 1.0, 0.25, key="ind_horizon")
    with c2:
        rolling_window = st.slider("Rolling window (Y)", 3, 30, 10, 1, key="ind_rolling")

    show_ts = st.checkbox("Time series", value=True, key="ind_show_ts")

    df_raw = get_clean_indicator(indicators, label)
    spx = build_spx_forward(spx_c, h_val)
    ind = df_raw[["Date", label]].rename(columns={label: "val"})
    ds = pd.merge_asof(spx, ind, on="Date", direction="backward").dropna(subset=["val", "fwd"])

    if len(ds) < 10:
        st.error("Insufficient overlap.")
        with st.expander("Diagnostics"):
            st.write("Indicator rows:", len(df_raw))
            st.write("SPX forward rows:", len(spx))
            st.write("Merged rows:", len(ds))
            if not df_raw.empty:
                st.write("Indicator range:", df_raw["Date"].min(), "→", df_raw["Date"].max())
            if not spx.empty:
                st.write("SPX range:", spx["Date"].min(), "→", spx["Date"].max())
            st.dataframe(df_raw.head(10), use_container_width=True)
        return

    corr = ds["val"].corr(ds["fwd"])
    r_sq = corr ** 2
    b1, b0 = np.polyfit(ds["val"], ds["fwd"], 1)
    n = len(ds)

    latest_val = df_raw[label].dropna().iloc[-1]
    latest_date = df_raw["Date"].dropna().iloc[-1]
    current_pct = (df_raw[label].dropna() <= latest_val).mean()
    pct_color = RED if current_pct >= 0.85 else (TEAL if current_pct <= 0.15 else ORANGE)

    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Indicator", label)
    m2.metric("Observations", f"{n:,}")
    m3.metric("Horizon", f"{h_val:g}Y")
    m4.metric("Correlation", f"{corr:.3f}")
    m5.metric("R²", f"{r_sq:.3f}")
    m6.metric("Slope", f"{b1*100:.3f}%")

    st.markdown(f"""
    <div style="padding:14px 18px;border-radius:14px;border:1px solid {BORDER};
        border-left:5px solid {pct_color};background:{BG2};margin-bottom:12px;
        display:flex;gap:24px;flex-wrap:wrap;align-items:center;">
        <div>
            <div style="font-size:10px;font-weight:800;color:{MUTED};text-transform:uppercase;letter-spacing:.06em;">Current value</div>
            <div style="font-size:22px;font-weight:900;color:{TEXT};margin-top:2px;">{latest_val:.2f}</div>
            <div style="font-size:11px;color:{MUTED};margin-top:2px;">as of {latest_date.strftime('%Y-%m-%d')}</div>
        </div>
        <div>
            <div style="font-size:10px;font-weight:800;color:{MUTED};text-transform:uppercase;letter-spacing:.06em;">Historical percentile</div>
            <div style="font-size:22px;font-weight:900;color:{pct_color};margin-top:2px;">{current_pct:.0%}</div>
            <div style="font-size:11px;color:{MUTED};margin-top:2px;">of all observations since {df_raw['Date'].min().strftime('%Y')}</div>
        </div>
        <div style="flex:1;min-width:220px;">
            <div style="font-size:10px;font-weight:800;color:{MUTED};text-transform:uppercase;letter-spacing:.06em;margin-bottom:6px;">Positioning gauge</div>
            <div style="background:{BORDER};border-radius:999px;height:10px;width:100%;">
                <div style="background:{pct_color};border-radius:999px;height:10px;width:{current_pct*100:.1f}%;"></div>
            </div>
            <div style="display:flex;justify-content:space-between;font-size:10px;color:{MUTED};margin-top:3px;">
                <span>Cheap (0%)</span><span>Expensive (100%)</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if show_ts:
        fig_ts = go.Figure()
        fig_ts.add_trace(go.Scatter(
            x=df_raw["Date"].tolist(),
            y=df_raw[label].tolist(),
            mode="lines",
            name=label,
            line=dict(color=BLUE, width=2)
        ))
        fig_ts.add_hline(
            y=latest_val, line_dash="dot", line_color=pct_color, line_width=1.5,
            annotation_text=f"Current: {latest_val:.2f} ({current_pct:.0%}ile)",
            annotation_position="top right", annotation_font=dict(color=pct_color, size=11)
        )
        apply_layout(fig_ts, f"Historical path — {label}", 280)
        fig_ts.update_yaxes(title=label)
        st.plotly_chart(fig_ts, use_container_width=True)

    x_range = np.linspace(ds["val"].min(), ds["val"].max(), 200)
    y_fit = b1 * x_range + b0
    x_vals = ds["val"].to_numpy(dtype=float)
    y_vals = ds["fwd"].to_numpy(dtype=float)
    date_text = ds["Date"].dt.strftime("%Y-%m-%d").tolist()

    fig_reg = go.Figure()
    fig_reg.add_trace(go.Scatter(
        x=x_vals.tolist(),
        y=y_vals.tolist(),
        mode="markers",
        name="Observations",
        text=date_text,
        hovertemplate=f"<b>{label}</b>: %{{x:.2f}}<br><b>Fwd return</b>: %{{y:.1%}}<br><b>Date</b>: %{{text}}<extra></extra>",
        marker=dict(
            color="rgba(37,99,235,0.45)",
            size=5
        )
    ))
    fig_reg.add_trace(go.Scatter(
        x=x_range.tolist(),
        y=y_fit.tolist(),
        mode="lines",
        name="OLS fit",
        line=dict(color=TEXT, width=2.5), hoverinfo="skip"
    ))
    fig_reg.add_vline(
        x=latest_val, line_dash="dot", line_color=pct_color, line_width=1.5,
        annotation_text=f"Today: {latest_val:.2f}",
        annotation_font=dict(color=pct_color, size=11)
    )
    apply_layout(fig_reg, f"{label} vs {h_val:g}Y SPX forward return", 480)
    fig_reg.update_xaxes(title=label)
    fig_reg.update_yaxes(title="SPX forward return", tickformat=".0%")
    st.plotly_chart(fig_reg, use_container_width=True)

    rw = int(rolling_window * 252)
    ds_sorted = ds.sort_values("Date").copy()
    ds_sorted["rolling_corr"] = ds_sorted["val"].rolling(rw, min_periods=max(30, rw // 4)).corr(ds_sorted["fwd"])
    ds_sorted = ds_sorted.dropna(subset=["rolling_corr"])
    if len(ds_sorted) > 10:
        fig_rc = go.Figure()
        fig_rc.add_trace(go.Scatter(
            x=ds_sorted["Date"].tolist(),
            y=ds_sorted["rolling_corr"].tolist(),
            mode="lines",
            name=f"{rolling_window}Y rolling corr",
            line=dict(color=PURPLE, width=2)
        ))
        fig_rc.add_hline(
            y=corr, line_dash="dot", line_color=SLATE, line_width=1,
            annotation_text=f"Full-sample: {corr:.3f}",
            annotation_position="top right", annotation_font=dict(color=MUTED, size=11)
        )
        fig_rc.add_hline(y=0, line_color=BORDER, line_width=1)
        apply_layout(fig_rc, f"Rolling {rolling_window}Y correlation — {label} vs {h_val:g}Y fwd return", 300)
        fig_rc.update_yaxes(title="Correlation", range=[-1.05, 1.05])
        fig_rc.update_xaxes(title="Date")
        st.plotly_chart(fig_rc, use_container_width=True)


def render_tech_page(spx_c, technicals):
    st.subheader("SPX Technicals")
    render_tab_intro(
        "How this works",
        "Date range controls the window shown on the chart. The black line is SPX, while SMA50 and SMA200 show the short- and long-term trend. "
        "Golden Cross and Death Cross markers show where those two moving averages crossed."
    )
    with st.expander("ℹ️ How to use this tab", expanded=False):
        st.markdown("""
        - **Date range** – Controls the start and end dates shown on the technical chart.
        - **S&P 500 Total Return / Price** – Main SPX line used as the base market series for the selected window.
        - **SMA50 / SMA200** – Short- and long-term moving averages used to judge trend direction.
        - **Golden Cross / Death Cross** – Markers showing where SMA50 crossed above or below SMA200.
        """)

    tech_dates = sorted(pd.to_datetime(technicals["Date"].dropna().unique()))
    min_d, max_d = pd.Timestamp(tech_dates[0]), pd.Timestamp(tech_dates[-1])
    default_start = max(min_d, max_d - pd.Timedelta(days=1260))

    d_s, d_e = st.slider(
        "Date range",
        min_value=min_d.date(),
        max_value=max_d.date(),
        value=(default_start.date(), max_d.date()),
        format="YYYY-MM-DD",
        key="tech_date_range"
    )
    d_s = pd.Timestamp(d_s)
    d_e = pd.Timestamp(d_e)

    t = technicals[(technicals["Date"] >= d_s) & (technicals["Date"] <= d_e)].copy()
    base_series = get_spx_price_series(spx_c)
    base_t = base_series[(base_series["Date"] >= d_s) & (base_series["Date"] <= d_e)].copy()

    if t.empty:
        st.error("No data in range.")
        return

    gc, dc = t[t["Golden_Cross"]], t[t["Death_Cross"]]
    last_p = t["Price"].iloc[-1]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Current SPX", f"{last_p:,.0f}")
    c2.metric("Golden Crosses", len(gc))
    c3.metric("Death Crosses", len(dc))
    c4.metric("Period", f"{d_s.year} – {d_e.year}")

    fig = go.Figure()
    if not base_t.empty:
        fig.add_trace(go.Scatter(
            x=base_t["Date"].tolist(),
            y=base_t["close"].tolist(),
            mode="lines",
            name="S&P 500 Total Return",
            line=dict(color=TEXT, width=2.0)
        ))
    else:
        fig.add_trace(go.Scatter(
            x=t["Date"].tolist(),
            y=t["Price"].tolist(),
            mode="lines",
            name="Price",
            line=dict(color=TEXT, width=1.8)
        ))

    fig.add_trace(go.Scatter(
        x=t["Date"].tolist(),
        y=t["SMA50"].tolist(),
        mode="lines",
        name="SMA50",
        line=dict(color=BLUE, dash="dot", width=1.4)
    ))
    fig.add_trace(go.Scatter(
        x=t["Date"].tolist(),
        y=t["SMA200"].tolist(),
        mode="lines",
        name="SMA200",
        line=dict(color=RED, dash="dot", width=1.4)
    ))
    if not gc.empty:
        fig.add_trace(go.Scatter(
            x=gc["Date"].tolist(),
            y=gc["Price"].tolist(),
            mode="markers",
            name="Golden Cross",
            marker=dict(symbol="triangle-up", size=12, color=GREEN)
        ))
    if not dc.empty:
        fig.add_trace(go.Scatter(
            x=dc["Date"].tolist(),
            y=dc["Price"].tolist(),
            mode="markers",
            name="Death Cross",
            marker=dict(symbol="triangle-down", size=12, color=RED)
        ))
    apply_layout(fig, "SPX Levels & Moving Average Signals", 550)
    st.plotly_chart(fig, use_container_width=True)


def render_combo_page(indicators, spx_c, technicals):
    st.subheader("Tail Combo Analysis")
    render_tab_intro(
        "How this works",
        "Selected indicators are the signals included in the test. Right tail cutoff and Left tail cutoff define what counts as expensive or cheap. "
        "Rule decides whether one, all, at least N, or a composite score must hit the tail. Min N only matters for At Least N. "
        "Horizon sets the forward SPX return window, and Regime filters the sample to all, bull, or bear conditions."
    )
    render_shared_indicator_picker(sorted(indicators.keys()))
    selected = get_combo_selected()

    with st.expander("ℹ️ How to use this tab", expanded=False):
        st.markdown("""
        - **Selected indicators** – The indicators included in the tail-condition test.
        - **Right tail cutoff** – Percentile threshold used to define an overextended or frothy reading.
        - **Left tail cutoff** – Percentile threshold used to define a cheap or washed-out reading.
        - **Rule** – Whether the signal triggers when any indicator, all indicators, at least N indicators, or a composite score hits the tail.
        - **Min N** – Number of indicators required when the rule is `At Least N`.
        - **Horizon (Y)** – Forward SPX return window measured after each historical date.
        - **Regime** – Restricts the sample to all dates, bull dates, or bear dates using SPX vs SMA200.
        """)

    if not selected:
        st.warning("Select at least one indicator.")
        return

    q_right = st.slider("Right tail cutoff", 0.70, 0.99, 0.85, 0.01, key="combo_right_cutoff")
    q_left = st.slider("Left tail cutoff", 0.01, 0.30, 0.15, 0.01, key="combo_left_cutoff")

    c3, c4 = st.columns(2)
    with c3:
        rule_label = st.selectbox("Rule", ["Any in Tail", "At Least N", "All in Tail", "Composite Score"], key="combo_rule")
        rule = {
            "Any in Tail": "any",
            "At Least N": "at_least_n",
            "All in Tail": "all",
            "Composite Score": "composite"
        }[rule_label]
    with c4:
        regime_label = st.selectbox(
            "Regime",
            ["All regimes", "Bull only (price > SMA200)", "Bear only (price < SMA200)"],
            key="combo_regime"
        )
        regime_mode = {
            "All regimes": "all",
            "Bull only (price > SMA200)": "bull",
            "Bear only (price < SMA200)": "bear"
        }[regime_label]

    c5, c6 = st.columns(2)
    with c5:
        n_val = st.slider("Min N", 1, max(1, len(selected)), min(2, max(1, len(selected))), 1, key="combo_min_n")
    with c6:
        horizon = st.slider("Horizon (Y)", 0.25, 10.0, 1.0, 0.25, key="combo_horizon")

    spx = build_spx_forward(spx_c, horizon)
    merged = spx[["Date", "fwd"]].copy()
    for lbl in selected:
        ind = get_clean_indicator(indicators, lbl)
        merged = pd.merge_asof(merged, ind, on="Date", direction="backward")
    merged = merged.dropna(subset=selected + ["fwd"]).copy()

    if len(merged) < 20:
        st.error(f"Not enough aligned observations at {horizon:.2f}Y.")
        with st.expander("Diagnostics"):
            st.write("Selected:", selected)
            st.write("Merged rows:", len(merged))
            st.dataframe(merged.head(10), use_container_width=True)
        return

    regime_mask = get_regime_mask(technicals, merged, regime_mode)
    merged = merged[regime_mask].copy()

    if len(merged) < 20:
        st.error("Not enough observations after applying regime filter.")
        return

    positioning_data = []
    st.markdown("<div class='section-divider'>Current positioning — where are we today?</div>", unsafe_allow_html=True)
    cols = st.columns(len(selected))

    for i, lbl in enumerate(selected):
        raw = get_clean_indicator(indicators, lbl)
        latest_val = raw[lbl].iloc[-1]
        latest_date = raw["Date"].iloc[-1]
        current_pct = (raw[lbl] <= latest_val).mean()
        pct_color = RED if current_pct >= q_right else (TEAL if current_pct <= q_left else ORANGE)
        tag = "⚠ Right tail" if current_pct >= q_right else ("✓ Left tail" if current_pct <= q_left else "● Neutral")
        positioning_data.append({
            "label": lbl,
            "latest_val": latest_val,
            "latest_date": latest_date,
            "current_pct": current_pct,
            "pct_color": pct_color,
            "tag": tag
        })
        with cols[i]:
            st.metric(lbl, f"{latest_val:.2f}", f"{current_pct:.0%}ile — {tag}")

    n_ind = len(positioning_data)
    height = max(160, 80 + n_ind * 52)
    fig_pos = go.Figure()

    for i, d in enumerate(positioning_data):
        y_pos = i
        for x0, x1, fc in [
            (0, q_left, "rgba(15,118,110,0.15)"),
            (q_left, q_right, "rgba(148,163,184,0.10)"),
            (q_right, 1.0, "rgba(220,38,38,0.15)")
        ]:
            fig_pos.add_shape(type="rect", x0=x0, x1=x1, y0=y_pos-0.32, y1=y_pos+0.32, fillcolor=fc, line_width=0, layer="below")
        fig_pos.add_shape(type="rect", x0=0, x1=1.0, y0=y_pos-0.32, y1=y_pos+0.32, fillcolor="rgba(0,0,0,0)", line_color=BORDER, line_width=1)
        fig_pos.add_shape(type="rect", x0=0, x1=d["current_pct"], y0=y_pos-0.18, y1=y_pos+0.18, fillcolor=d["pct_color"], opacity=0.25, line_width=0, layer="below")
        fig_pos.add_shape(type="line", x0=d["current_pct"], x1=d["current_pct"], y0=y_pos-0.38, y1=y_pos+0.38, line=dict(color=d["pct_color"], width=3))
        fig_pos.add_trace(go.Scatter(
            x=[d["current_pct"]],
            y=[y_pos],
            mode="markers+text",
            marker=dict(color=d["pct_color"], size=10, line=dict(color=BG, width=2)),
            text=[f"  {d['current_pct']:.0%}  {d['tag']}"],
            textposition="middle right",
            textfont=dict(size=11, color=d["pct_color"]),
            hovertemplate=(f"<b>{d['label']}</b><br>Value: {d['latest_val']:.2f}<br>"
                           f"Percentile: {d['current_pct']:.1%}<br>As of: {d['latest_date'].strftime('%Y-%m-%d')}<extra></extra>"),
            showlegend=False, name=d["label"]
        ))

    fig_pos.add_vline(x=q_left, line_dash="dot", line_color=TEAL, line_width=1.5,
                      annotation_text=f"Left ({q_left:.0%})", annotation_position="top left",
                      annotation_font=dict(color=TEAL, size=10))
    fig_pos.add_vline(x=q_right, line_dash="dot", line_color=RED, line_width=1.5,
                      annotation_text=f"Right ({q_right:.0%})", annotation_position="top right",
                      annotation_font=dict(color=RED, size=10))

    fig_pos.update_layout(
        template="plotly_white",
        paper_bgcolor=BG,
        plot_bgcolor=BG,
        height=height,
        margin=dict(l=160, r=60, t=40, b=36),
        title=dict(text="Today's percentile vs full history — tail zones shaded", x=0, font=dict(size=14, color=TEXT)),
        xaxis=dict(range=[-0.01, 1.18], tickformat=".0%", showgrid=True, gridcolor=BORDER,
                   zeroline=False, linecolor=BORDER, title="Historical percentile"),
        yaxis=dict(
            tickmode="array",
            tickvals=list(range(n_ind)),
            ticktext=[d["label"] for d in positioning_data],
            showgrid=False,
            zeroline=False,
            range=[-0.6, n_ind - 0.4],
            tickfont=dict(size=12, color=TEXT)
        ),
        hovermode="closest"
    )
    st.plotly_chart(fig_pos, use_container_width=True)

    right_thresh = {lbl: merged[lbl].quantile(q_right) for lbl in selected}
    left_thresh = {lbl: merged[lbl].quantile(q_left) for lbl in selected}
    right_flags = pd.DataFrame({lbl: merged[lbl] >= right_thresh[lbl] for lbl in selected})
    left_flags = pd.DataFrame({lbl: merged[lbl] <= left_thresh[lbl] for lbl in selected})
    pct_scores = pd.DataFrame({lbl: merged[lbl].rank(pct=True) for lbl in selected})
    composite_right = pct_scores.mean(axis=1)
    composite_left = 1 - composite_right

    right_mask, _ = _apply_rule(right_flags, selected, rule, n_val, composite_right, q_right)
    left_mask, _ = _apply_rule(left_flags, selected, rule, n_val, composite_left, q_left)

    top10_thresh = {lbl: merged[lbl].quantile(0.90) for lbl in selected}
    top10_flags = pd.DataFrame({lbl: merged[lbl] >= top10_thresh[lbl] for lbl in selected})
    top10_composite = pct_scores.mean(axis=1)
    top10_mask, top10_rule_label = _apply_rule(top10_flags, selected, rule, n_val, top10_composite, 0.90)
    top10_count = top10_flags.sum(axis=1)

    overlay_df = merged[["Date"]].copy()
    overlay_df["top10_count"] = top10_count.astype(int).tolist()
    overlay_df["top10_hit"] = top10_mask.astype(bool).tolist()
    overlay_df["active_labels"] = [
        ", ".join([lbl for lbl in selected if bool(top10_flags.iloc[i][lbl])]) or "None"
        for i in range(len(top10_flags))
    ]

    spx_price = get_spx_price_series(spx_c)
    overlay_df = pd.merge_asof(
        overlay_df.sort_values("Date"),
        spx_price[["Date", "close"]].sort_values("Date"),
        on="Date",
        direction="backward"
    ).dropna(subset=["close"])

    st.markdown("<div class='section-divider'>History of top-10% tail conditions</div>", unsafe_allow_html=True)
    st.caption(
        f"SPX is overlaid with the first date each fixed top-10% episode triggered under the current rule. "
        f"Rule used: {top10_rule_label}."
    )

    overlay_min_d = pd.Timestamp(overlay_df["Date"].min())
    overlay_max_d = pd.Timestamp(overlay_df["Date"].max())
    overlay_default_start = max(overlay_min_d, overlay_max_d - pd.Timedelta(days=3650))
    combo_hist_start, combo_hist_end = st.slider(
        "History date range",
        min_value=overlay_min_d.date(),
        max_value=overlay_max_d.date(),
        value=(overlay_default_start.date(), overlay_max_d.date()),
        format="YYYY-MM-DD",
        key="combo_history_date_range"
    )
    combo_hist_start = pd.Timestamp(combo_hist_start)
    combo_hist_end = pd.Timestamp(combo_hist_end)
    overlay_df = overlay_df[
        (overlay_df["Date"] >= combo_hist_start) & (overlay_df["Date"] <= combo_hist_end)
    ].copy()

    fig_hist = go.Figure()
    fig_hist.add_trace(go.Scatter(
        x=overlay_df["Date"].tolist(),
        y=overlay_df["close"].tolist(),
        mode="lines",
        name="S&P 500 Total Return",
        line=dict(color=TEXT, width=2.0)
    ))

    for start_date, end_date in mask_to_periods(overlay_df["Date"], overlay_df["top10_hit"]):
        fig_hist.add_vrect(
            x0=start_date,
            x1=end_date,
            fillcolor="rgba(220,38,38,0.10)",
            line_width=0,
            layer="below"
        )

    start_dates = set(mask_to_starts(overlay_df["Date"], overlay_df["top10_hit"]))
    hit_df = overlay_df[overlay_df["Date"].isin(start_dates)].copy()
    if not hit_df.empty:
        fig_hist.add_trace(go.Scatter(
            x=hit_df["Date"].tolist(),
            y=hit_df["close"].tolist(),
            mode="markers",
            name="Episode start",
            text=[
                f"Indicators in top 10%: {labels}<br>Count: {count}"
                for labels, count in zip(hit_df["active_labels"], hit_df["top10_count"])
            ],
            marker=dict(
                color="rgba(220,38,38,0.75)",
                size=(10 + hit_df["top10_count"].astype(float) * 2).tolist(),
                line=dict(color=BG, width=1)
            ),
            hovertemplate="<b>%{text}</b><br>Date: %{x|%Y-%m-%d}<br>SPX: %{y:,.0f}<extra></extra>"
        ))

    apply_layout(fig_hist, "SPX with first top-10% tail-condition triggers", 420)
    fig_hist.update_yaxes(title="Index level")
    fig_hist.update_xaxes(title="Date")
    st.plotly_chart(fig_hist, use_container_width=True)

    render_downturn_heatmap_section(
        indicators=indicators,
        spx_c=spx_c,
        selected=selected,
        key_prefix="combo_downturn_heatmap",
        default_drawdown=0.10
    )

    merged["is_right_tail"] = right_mask
    merged["is_left_tail"] = left_mask

    right_ev = merged[merged["is_right_tail"]].copy()
    left_ev = merged[merged["is_left_tail"]].copy()

    base_avg = merged["fwd"].mean()
    right_avg = right_ev["fwd"].mean() if len(right_ev) else np.nan
    left_avg = left_ev["fwd"].mean() if len(left_ev) else np.nan
    base_var, base_cvar = calc_var_cvar(merged["fwd"])
    right_var, right_cvar = calc_var_cvar(right_ev["fwd"]) if len(right_ev) else (np.nan, np.nan)
    left_var, left_cvar = calc_var_cvar(left_ev["fwd"]) if len(left_ev) else (np.nan, np.nan)

    regime_label_short = {"all": "All regimes", "bull": "Bull regime only", "bear": "Bear regime only"}[regime_mode]

    st.markdown(f"<div class='section-divider'>Right tail — overvaluation / froth · {regime_label_short}</div>", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Right Tail Events", f"{len(right_ev):,}", f"{fmt_pct(len(right_ev)/len(merged))} of dates")
    c2.metric("Avg Fwd Return", fmt_pct(right_avg), f"Base: {fmt_pct(base_avg)}")
    c3.metric("95% VaR", fmt_pct(right_var), f"Base: {fmt_pct(base_var)}")
    c4.metric("95% CVaR", fmt_pct(right_cvar), f"Base: {fmt_pct(base_cvar)}")

    st.markdown(f"<div class='section-divider'>Left tail — undervaluation / cheap · {regime_label_short}</div>", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Left Tail Events", f"{len(left_ev):,}", f"{fmt_pct(len(left_ev)/len(merged))} of dates")
    c2.metric("Avg Fwd Return", fmt_pct(left_avg), f"Base: {fmt_pct(base_avg)}")
    c3.metric("95% VaR", fmt_pct(left_var), f"Base: {fmt_pct(base_var)}")
    c4.metric("95% CVaR", fmt_pct(left_cvar), f"Base: {fmt_pct(base_cvar)}")


def render_recession_page(indicators, spx_c):
    st.subheader("SPX Downturn Lookback")
    render_tab_intro(
        "How this works",
        "Selected indicators are measured before each downturn event. Drawdown level sets how deep the selloff must get before an event starts. "
        "A downturn then remains active only while drawdown stays beyond that same level, which keeps the definition tied to one intuitive threshold."
    )
    render_shared_indicator_picker(sorted(indicators.keys()))
    selected = get_combo_selected()

    with st.expander("ℹ️ How to use this tab", expanded=False):
        st.markdown("""
        - **Selected indicators** – The indicators looked up before each downturn start date.
        - **Drawdown level** – The percent drop from the prior peak that starts a new SPX downturn event.
        - **Event rule** – Once drawdown climbs back above that level, the downturn episode ends.
        """)

    if not selected:
        st.warning("Select at least one indicator.")
        return

    dd_thresh = st.slider("Drawdown level", 0.10, 0.40, 0.20, 0.01, key="recess_trigger")

    spx, events = identify_spx_downturns(spx_c, dd_thresh, None, None)
    if events.empty:
        st.error("No downturn events found with the current settings.")
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Detected Downturns", f"{len(events):,}")
    c2.metric("Avg Max Drawdown", fmt_pct(events["max_drawdown"].mean()))
    c3.metric("Avg Days to Trough", f"{events['days_to_trough'].mean():.0f}")
    c4.metric("Avg Event Length", f"{events['event_days'].mean():.0f}")

    fig_dd = go.Figure()
    fig_dd.add_trace(go.Scatter(
        x=spx["Date"].tolist(),
        y=spx["close"].tolist(),
        mode="lines",
        name="S&P 500 Total Return",
        line=dict(color=TEXT, width=1.8)
    ))
    for _, row in events.iterrows():
        fig_dd.add_vrect(x0=row["start_date"], x1=row["end_date"], fillcolor="rgba(220,38,38,0.10)", line_width=0, layer="below")
        fig_dd.add_trace(go.Scatter(
            x=[row["start_date"]],
            y=[row["start_price"]],
            mode="markers",
            marker=dict(size=9, color=ORANGE, symbol="diamond"), showlegend=False,
            hovertemplate=f"<b>Start</b><br>Date: {row['start_date'].strftime('%Y-%m-%d')}<br>Price: {row['start_price']:.1f}<extra></extra>"
        ))
        fig_dd.add_trace(go.Scatter(
            x=[row["trough_date"]],
            y=[row["trough_price"]],
            mode="markers",
            marker=dict(size=10, color=RED, symbol="triangle-down"), showlegend=False,
            hovertemplate=f"<b>Trough</b><br>Date: {row['trough_date'].strftime('%Y-%m-%d')}<br>Price: {row['trough_price']:.1f}<br>Max DD: {row['max_drawdown']:.1%}<extra></extra>"
        ))
    apply_layout(fig_dd, "S&P 500 Total Return with detected downturn episodes", 420)
    fig_dd.update_yaxes(title="Index level")
    st.plotly_chart(fig_dd, use_container_width=True)

    lookbacks = [
        ("3M", int(round(0.25*252))),
        ("6M", int(round(0.50*252))),
        ("1Y", int(round(1.00*252))),
        ("2Y", int(round(2.00*252))),
        ("3Y", int(round(3.00*252))),
        ("4Y", int(round(4.00*252))),
        ("5Y", int(round(5.00*252)))
    ]

    rows = []
    for _, ev in events.iterrows():
        start_date = pd.to_datetime(ev["start_date"])
        for lb_name, lb_days in lookbacks:
            anchor = start_date - pd.Timedelta(days=int(lb_days * 365 / 252))
            rec = {
                "event_start": start_date,
                "trough_date": ev["trough_date"],
                "lookback": lb_name,
                "anchor_date": anchor,
                "max_drawdown": ev["max_drawdown"]
            }
            for lbl in selected:
                val, actual_date, raw, value_col = lookup_indicator_value_at(indicators, lbl, anchor)
                pct = percentile_as_of(raw, value_col, anchor, val)
                rec[f"{lbl}__value"] = val
                rec[f"{lbl}__pct"] = pct
                rec[f"{lbl}__obs_date"] = actual_date
            rows.append(rec)

    lookback_df = pd.DataFrame(rows)

    avg_pct = pd.DataFrame(index=selected, columns=[x[0] for x in lookbacks], dtype=float)
    for lbl in selected:
        for lb_name, _ in lookbacks:
            mask = lookback_df["lookback"] == lb_name
            avg_pct.loc[lbl, lb_name] = lookback_df.loc[mask, f"{lbl}__pct"].mean()

    st.markdown("<div class='section-divider'>Average indicator percentile before downturns</div>", unsafe_allow_html=True)
    heat_text = [[f"{v:.0%}" if pd.notna(v) else "—" for v in row] for row in avg_pct.values]
    heat = go.Figure(data=go.Heatmap(
        z=avg_pct.fillna(np.nan).values.tolist(),
        x=avg_pct.columns.tolist(),
        y=avg_pct.index.tolist(),
        colorscale=[
            [0.00, "#2f7d32"],
            [0.35, "#b8d98a"],
            [0.50, "#fff7bc"],
            [0.70, "#f7b267"],
            [1.00, "#c0392b"],
        ],
        zmin=0,
        zmax=1,
        text=heat_text,
        texttemplate="%{text}",
        hovertemplate="Indicator: %{y}<br>Lookback: %{x}<br>Avg percentile: %{z:.1%}<extra></extra>",
        xgap=1,
        ygap=1,
        colorbar=dict(title="Percentile", tickformat=".0%")
    ))
    apply_layout(heat, "Average historical percentile before downturn start", 120 + 42 * max(4, len(selected)))
    heat.update_xaxes(title="Lookback from downturn start")
    heat.update_yaxes(title="")
    st.plotly_chart(heat, use_container_width=True)

    lookback_order = {"3M": 0, "6M": 1, "1Y": 2, "2Y": 3, "3Y": 4, "4Y": 5, "5Y": 6}
    st.markdown("<div class='section-divider'>Detailed lookback matrix</div>", unsafe_allow_html=True)

    blocks = []
    for _, ev in events.iterrows():
        start_date = pd.to_datetime(ev["start_date"])
        sub = lookback_df[lookback_df["event_start"] == start_date].copy().sort_values(
            "lookback", key=lambda s: s.map(lookback_order)
        )
        header_cols = "".join([f"<th style='padding:8px 10px;text-align:right;'>{lb}</th>" for lb, _ in lookbacks])
        body_rows = []
        for lbl in selected:
            vals = []
            for lb, _ in lookbacks:
                r = sub[sub["lookback"] == lb]
                if len(r):
                    pct = r[f"{lbl}__pct"].iloc[0]
                    vals.append(
                        f"<td style='padding:8px 10px;text-align:right;'>{pct:.0%}</td>"
                        if pd.notna(pct) else
                        f"<td style='padding:8px 10px;text-align:right;'>—</td>"
                    )
                else:
                    vals.append(f"<td style='padding:8px 10px;text-align:right;'>—</td>")
            body_rows.append(
                f"<tr><td style='padding:8px 10px;border-top:1px solid {BORDER};'>{lbl}</td>{''.join(vals)}</tr>"
            )

        blocks.append(f"""
        <div style="padding:12px 14px;border-radius:12px;border:1px solid {BORDER};background:{BG2};margin-bottom:12px;">
            <div style="font-size:12px;font-weight:800;color:{TEXT};margin-bottom:6px;">
                Downturn starting {start_date.strftime('%Y-%m-%d')}
            </div>
            <div style="font-size:11px;color:{MUTED};margin-bottom:10px;">
                Trough: {ev['trough_date'].strftime('%Y-%m-%d')} · Max drawdown: {ev['max_drawdown']:.1%}
            </div>
            <div style="overflow-x:auto;">
                <table class="pretty-table">
                    <thead>
                        <tr><th style="padding:8px 10px;text-align:left;">Indicator</th>{header_cols}</tr>
                    </thead>
                    <tbody>{''.join(body_rows)}</tbody>
                </table>
            </div>
        </div>""")

    st.markdown("".join(blocks), unsafe_allow_html=True)





# ══════════════════════════════════════════════════════════════════════════════
# 9. MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    st.markdown(f"""
    <div style="
        padding:18px 20px 16px;
        border-radius:18px;
        border:1px solid {BORDER};
        background:linear-gradient(135deg,#f8fbff 0%,#ffffff 100%);
        margin-bottom:14px;
        text-align:center;">
        <div style="font-size:12px;color:{BLUE};font-weight:800;letter-spacing:.08em;text-transform:uppercase;">
            Market Dashboard
        </div>
        <div style="font-size:28px;font-weight:900;color:{TEXT};margin-top:6px;line-height:1.05;">
            Market Froth Dashboard
        </div>
        <div style="font-size:13px;color:{MUTED};margin:8px auto 0;max-width:720px;line-height:1.5;">
            Track valuation extremes, compare indicator baskets, inspect SPX downturn setups, and search for early-warning combinations.
        </div>
    </div>
    """, unsafe_allow_html=True)

    try:
        data = load_all_data()
        indicators, spx_c, technicals = build_notebook_objects(data)
    except Exception as e:
        st.error(f"Failed to load/build dashboard data: {e}")
        st.info(f"Make sure all data files are in: `{DATA_DIR}`")
        return

    all_indicator_names = sorted(indicators.keys())
    init_state(all_indicator_names)

    page_options = ["Indicator Analysis", "SPX Technicals", "Tail Combo"]
    nav_left, nav_center, nav_right = st.columns([1.2, 6, 1.2])
    with nav_center:
        page = st.radio(
            "View",
            page_options,
            horizontal=True,
            key="page_selector",
            label_visibility="collapsed"
        )

    if page == "Indicator Analysis":
        render_indicator_page(indicators, spx_c)
    elif page == "SPX Technicals":
        render_tech_page(spx_c, technicals)
    else:
        render_combo_page(indicators, spx_c, technicals)


if __name__ == "__main__":
    main()
