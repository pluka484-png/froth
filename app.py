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
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.tree import DecisionTreeClassifier, export_text


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
]

WF_WARMUP_DAYS = 252
WF_ALERT_THRESHOLD = 0.10
WF_RESET_DAYS = 21
WF_SMOOTH_DAYS = 20
WF_SAMPLE_START_DATE = "1993-01-01"
WF_MIN_TEST_ROWS = 63
WF_MIN_TRAIN_ROWS = 252 * 3
WF_N_FOLDS = 4
WF_MAX_COMBO_SIZE = 4
ML_INNER_OOF_SPLITS = 3
ML_MAX_MISSING_FEATURE_FRAC = 0.35
ML_RANDOM_STATE = 42
ML_RF_N_ESTIMATORS = 300
ML_RF_MAX_DEPTH = 5
ML_RF_MIN_SAMPLES_LEAF = 10
ML_TREE_MAX_DEPTH = 4
ML_TREE_MIN_SAMPLES_LEAF = 10
ML_MIN_FOLDS_REQUIRED = 2
ML_N_FINALISTS = 12
ML_TOP_COMBOS = 12
REGIME_AUGMENT_CANDIDATES = [
    "Shiller CAPE",
    "Tobin Q",
    "PB Ratio",
    "PE Ratio",
    "PS Ratio",
    "Buffet Indicator",
    "Earnings Yield",
    "Dividend Yield",
    "Equity Valuation (OFR)",
]


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
div[data-testid="stRadio"] > label {{
    display:none;
}}
div[data-testid="stRadio"] {{
    display:flex;
    justify-content:center;
}}
div[role="radiogroup"] {{
    display:flex;
    flex-wrap:wrap;
    gap:0.45rem;
    justify-content:center;
    width:fit-content;
    margin:0 auto;
}}
div[role="radiogroup"] label[data-baseweb="radio"] {{
    background:#ffffff;
    border:1px solid {BORDER};
    border-radius:999px;
    padding:0.4rem 0.85rem;
}}
div[role="radiogroup"] label[data-baseweb="radio"]:has(input:checked) {{
    background:#eff6ff;
    border-color:#bfdbfe;
}}
div[role="radiogroup"] label[data-baseweb="radio"] > div {{
    gap:0.35rem;
}}
div[role="radiogroup"] label[data-baseweb="radio"] p {{
    color:{TEXT};
    font-size:0.93rem;
    font-weight:700;
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
    }

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


def build_indicator_panel(indicators, spx_c, selected):
    spx = get_spx_price_series(spx_c)[["Date"]].copy()
    result_panel = spx.copy()
    for lbl in selected:
        ind = get_clean_indicator(indicators, lbl)
        if ind.empty:
            continue
        result_panel = pd.merge_asof(result_panel, ind, on="Date", direction="backward")

    usable = [c for c in result_panel.columns if c != "Date"]
    if not usable:
        return pd.DataFrame(columns=["Date"])

    result_panel = result_panel.dropna(subset=usable, how="all").copy()
    return result_panel.copy()


def make_downturn_prediction_dataset(indicators, spx_c, selected, horizon_years=1.0,
                                     drawdown_threshold=0.20, recovery_threshold=0.05,
                                     min_spacing_days=126):
    result_panel = build_indicator_panel(indicators, spx_c, selected)
    _, events = identify_spx_downturns(
        spx_c,
        drawdown_threshold=drawdown_threshold,
        recovery_threshold=recovery_threshold,
        min_spacing_days=min_spacing_days
    )

    if result_panel.empty or events.empty:
        return result_panel, events

    ds = result_panel.copy()
    horizon_days = int(round(horizon_years * 365))
    event_starts = np.sort(pd.to_datetime(events["start_date"]).values.astype("datetime64[ns]"))
    dates = pd.to_datetime(ds["Date"]).values.astype("datetime64[ns]")
    future_limits = dates + np.timedelta64(horizon_days, "D")
    left_idx = np.searchsorted(event_starts, dates, side="left")
    right_idx = np.searchsorted(event_starts, future_limits, side="right")
    ds["target_downturn"] = (right_idx > left_idx).astype(int)

    for lbl in selected:
        if lbl in ds.columns:
            is_inverted = False
            if lbl in indicators and "inverted" in indicators[lbl].columns:
                is_inverted = bool(indicators[lbl]["inverted"].iloc[0])
            ds[f"{lbl}__pct"] = expanding_percentile_series(ds[lbl], min_periods=20, invert=is_inverted)

    return ds, events


def auc_from_scores(y_true, scores):
    y = pd.Series(y_true).astype(int)
    s = pd.Series(scores).astype(float)
    valid = y.notna() & s.notna()
    y = y[valid]
    s = s[valid]
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return np.nan
    ranks = s.rank(method="average")
    rank_sum_pos = ranks[y == 1].sum()
    return float((rank_sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def compute_top_bucket_metrics(y_true, scores, top_frac=0.10):
    y = pd.Series(y_true).astype(int)
    s = pd.Series(scores).astype(float)
    valid = y.notna() & s.notna()
    y = y[valid]
    s = s[valid]
    if len(y) < 20:
        return {"precision": np.nan, "recall": np.nan, "f1": np.nan, "lift": np.nan, "flag_rate": np.nan}

    cutoff = s.quantile(1 - top_frac)
    pred = (s >= cutoff).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    precision = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    recall = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    f1 = (2 * precision * recall / (precision + recall)
          if pd.notna(precision) and pd.notna(recall) and (precision + recall) > 0 else np.nan)
    base_rate = y.mean()
    lift = precision / base_rate if pd.notna(precision) and pd.notna(base_rate) and base_rate > 0 else np.nan
    return {"precision": precision, "recall": recall, "f1": f1, "lift": lift, "flag_rate": pred.mean()}


def evaluate_indicator_combos(indicators, spx_c, selected, horizon_years=1.0,
                              drawdown_threshold=0.20, recovery_threshold=0.05,
                              min_spacing_days=126, max_combo_size=4):
    if not selected:
        return pd.DataFrame(), pd.DataFrame()

    ds, _ = make_downturn_prediction_dataset(
        indicators=indicators,
        spx_c=spx_c,
        selected=selected,
        horizon_years=horizon_years,
        drawdown_threshold=drawdown_threshold,
        recovery_threshold=recovery_threshold,
        min_spacing_days=min_spacing_days
    )

    if ds.empty or "target_downturn" not in ds.columns:
        return pd.DataFrame(), ds

    X_cols = [f"{lbl}__pct" for lbl in selected if f"{lbl}__pct" in ds.columns]
    if not X_cols:
        return pd.DataFrame(), ds

    X = ds[X_cols].to_numpy(dtype=float)
    y = ds["target_downturn"].to_numpy(dtype=int)

    label_to_xcol = {}
    xcol = 0
    for lbl in selected:
        if f"{lbl}__pct" in ds.columns:
            label_to_xcol[lbl] = xcol
            xcol += 1

    results = []
    for k in range(1, min(max_combo_size, len(selected)) + 1):
        for combo in combinations(selected, k):
            idxs = [label_to_xcol[lbl] for lbl in combo if lbl in label_to_xcol]
            if len(idxs) != len(combo):
                continue

            subX = X[:, idxs]
            valid_mask = ~np.isnan(subX).any(axis=1)
            if valid_mask.sum() < 50:
                continue

            signal = subX[valid_mask].mean(axis=1)
            yv = y[valid_mask]
            auc = auc_from_scores(yv, signal)

            top_decile_cut = np.nanquantile(signal, 0.90)
            top_decile_mask = signal >= top_decile_cut
            hit_rate_top_decile = yv[top_decile_mask].mean() if top_decile_mask.sum() else np.nan

            pos_mask = yv == 1
            neg_mask = yv == 0
            pos_mean = signal[pos_mask].mean() if pos_mask.any() else np.nan
            neg_mean = signal[neg_mask].mean() if neg_mask.any() else np.nan
            separation = pos_mean - neg_mean if pd.notna(pos_mean) and pd.notna(neg_mean) else np.nan

            m10 = compute_top_bucket_metrics(yv, signal, top_frac=0.10)
            m05 = compute_top_bucket_metrics(yv, signal, top_frac=0.05)

            results.append({
                "combo": " + ".join(combo),
                "size": k,
                "auc": auc,
                "hit_rate_top_decile": hit_rate_top_decile,
                "pos_mean_signal": pos_mean,
                "neg_mean_signal": neg_mean,
                "signal_separation": separation,
                "precision_top10": m10["precision"],
                "recall_top10": m10["recall"],
                "f1_top10": m10["f1"],
                "lift_top10": m10["lift"],
                "precision_top5": m05["precision"],
                "recall_top5": m05["recall"],
                "f1_top5": m05["f1"],
                "lift_top5": m05["lift"],
                "n_obs": int(valid_mask.sum()),
                "n_positive": int(yv.sum())
            })

    res = pd.DataFrame(results)
    if not res.empty:
        res = res.sort_values(
            ["f1_top10", "precision_top10", "auc", "signal_separation", "n_obs"],
            ascending=[False, False, False, False, False]
        ).reset_index(drop=True)

    return res, ds


@st.cache_data(show_spinner=False)
def build_selected_master_dataset(
    indicators,
    spx_c,
    selected,
    drawdown_threshold=0.20,
    recovery_threshold=0.05,
    min_spacing_days=126,
    lead_days=252
):
    spx = get_spx_price_series(spx_c).rename(columns={"close": "close"}).copy()
    spx = spx.sort_values("Date").reset_index(drop=True)
    spx["target_downturn"] = 0.0

    _, events = identify_spx_downturns(
        spx_c,
        drawdown_threshold=drawdown_threshold,
        recovery_threshold=recovery_threshold,
        min_spacing_days=min_spacing_days
    )
    event_dates = set(pd.to_datetime(events["start_date"])) if not events.empty else set()
    for ev_date in sorted(event_dates):
        ev_idx_list = spx.index[spx["Date"] == ev_date].tolist()
        if not ev_idx_list:
            continue
        ev_idx = ev_idx_list[0]
        start_idx = max(0, ev_idx - int(lead_days))
        spx.loc[start_idx:ev_idx - 1, "target_downturn"] = 1.0
        spx.loc[ev_idx, "target_downturn"] = np.nan

    master = spx[["Date", "close", "target_downturn"]].copy()
    for name in selected:
        ind = get_clean_indicator(indicators, name)
        tol_days = 7 if len(ind) > 10 and ind["Date"].diff().dt.days.median() <= 5 else 90
        master = pd.merge_asof(
            master.sort_values("Date"),
            ind.sort_values("Date"),
            on="Date",
            tolerance=pd.Timedelta(days=tol_days),
            direction="backward"
        )
        if name in master.columns:
            master[name] = master[name].ffill()

    master = master.loc[master["Date"] >= pd.Timestamp(WF_SAMPLE_START_DATE)].reset_index(drop=True)
    return master


@st.cache_data(show_spinner=False)
def build_percentile_feature_master(master_ds, selected):
    master = pd.DataFrame(master_ds).copy()
    selected = list(selected)
    for name in selected:
        if name in master.columns:
            master[f"{name}__pct"] = expanding_percentile_series(master[name], min_periods=WF_WARMUP_DAYS)
    return master


def fit_alert_threshold(raw_signal, alert_threshold=WF_ALERT_THRESHOLD, smooth_days=WF_SMOOTH_DAYS):
    s = pd.Series(raw_signal, dtype=float)
    smoothed = s.rolling(smooth_days, min_periods=smooth_days).mean().dropna()
    if len(smoothed) == 0:
        return np.nan
    return np.nanpercentile(smoothed, 100 * (1 - alert_threshold))


def extract_discrete_events(is_on, reset_days=WF_RESET_DAYS):
    is_on = pd.Series(is_on).fillna(False).astype(bool)
    events = np.zeros(len(is_on), dtype=int)
    last_event_idx = -(10 ** 9)
    prev_on = False

    for i in range(len(is_on)):
        if is_on.iloc[i]:
            if (not prev_on) or ((i - last_event_idx) >= reset_days):
                events[i] = 1
                last_event_idx = i
            prev_on = True
        else:
            prev_on = False

    return events


def build_trigger_engine(raw_signal, threshold, smooth_days=WF_SMOOTH_DAYS, reset_days=WF_RESET_DAYS):
    s = pd.Series(raw_signal, dtype=float)
    smoothed = s.rolling(smooth_days, min_periods=smooth_days).mean()
    is_on = smoothed >= threshold
    events = extract_discrete_events(is_on, reset_days=reset_days)
    return pd.DataFrame({
        "raw_signal": s,
        "smoothed_signal": smoothed,
        "is_on": is_on,
        "event": events
    })


def compute_distinct_crisis_metrics(trigger_dates, event_dates, horizon_days=252):
    trigger_dates = pd.to_datetime(pd.Series(trigger_dates)).dropna().sort_values().reset_index(drop=True)
    event_dates = pd.to_datetime(pd.Series(event_dates)).dropna().sort_values().reset_index(drop=True)
    used_trigger_idx = set()
    tp = 0
    lead_sum = 0.0

    for ev in event_dates:
        window_start = ev - pd.Timedelta(days=int(horizon_days * 365.25 / 252))
        eligible = [
            (idx, dt) for idx, dt in enumerate(trigger_dates)
            if idx not in used_trigger_idx and window_start <= dt < ev
        ]
        if not eligible:
            continue
        idx, first = eligible[0]
        used_trigger_idx.add(idx)
        tp += 1
        lead_sum += (ev - first).days

    precision = tp / len(trigger_dates) if len(trigger_dates) else 0.0
    recall = tp / len(event_dates) if len(event_dates) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    avg_lead = lead_sum / tp if tp else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "avg_lead": avg_lead,
        "n_triggers": len(trigger_dates),
        "n_crises": len(event_dates),
        "n_caught": tp
    }


@st.cache_data(show_spinner=False)
def run_walk_forward_search_cached(master_ds, selected, horizon_years, max_combo_size):
    master_ds = pd.DataFrame(master_ds).copy()
    master_ds = master_ds.dropna(subset=["target_downturn"]).reset_index(drop=True)
    if len(master_ds) <= WF_WARMUP_DAYS + WF_MIN_TEST_ROWS:
        return pd.DataFrame(), pd.DataFrame()

    fold_size = max(1, (len(master_ds) - WF_WARMUP_DAYS) // WF_N_FOLDS)
    fold_rows = []
    purge_days = int(round(float(horizon_years) * 252))

    for i in range(WF_N_FOLDS):
        test_start = WF_WARMUP_DAYS + i * fold_size
        test_end = WF_WARMUP_DAYS + (i + 1) * fold_size if i < WF_N_FOLDS - 1 else len(master_ds)
        train_end = max(WF_WARMUP_DAYS, test_start - purge_days)

        prefix = master_ds.iloc[:test_end].copy()
        for name in selected:
            if name in prefix.columns:
                prefix[f"{name}__pct"] = expanding_percentile_series(prefix[name], min_periods=WF_WARMUP_DAYS)

        train_df = prefix.iloc[WF_WARMUP_DAYS:train_end].copy()
        test_df = prefix.iloc[test_start:test_end].copy()
        if len(train_df) < WF_MIN_TRAIN_ROWS or len(test_df) < WF_MIN_TEST_ROWS:
            continue

        event_starts_test = test_df.loc[
            (test_df["target_downturn"].fillna(0).astype(int) == 1) &
            (test_df["target_downturn"].shift(1).fillna(0).astype(int) == 0),
            "Date"
        ]
        if len(event_starts_test) < 1:
            continue

        for k in range(1, min(max_combo_size, len(selected)) + 1):
            for combo in combinations(selected, k):
                cols = [f"{c}__pct" for c in combo if f"{c}__pct" in train_df.columns]
                if len(cols) != len(combo):
                    continue
                sub_train = train_df[["Date"] + cols].dropna().copy()
                sub_test = test_df[["Date"] + cols].dropna().copy()
                if len(sub_train) < WF_WARMUP_DAYS or len(sub_test) < WF_MIN_TEST_ROWS:
                    continue

                sig_train = sub_train[cols].mean(axis=1)
                sig_test = sub_test[cols].mean(axis=1)
                threshold = fit_alert_threshold(sig_train)
                if pd.isna(threshold):
                    continue
                trig_test = build_trigger_engine(sig_test, threshold)
                trigger_dates = sub_test.loc[trig_test["event"] == 1, "Date"]
                m = compute_distinct_crisis_metrics(trigger_dates, event_starts_test, horizon_days=int(round(horizon_years * 252)))
                score = m["recall"] * min(2.0, 1 + (m["avg_lead"] / 252)) * m["precision"]
                fold_rows.append({
                    "combo": " + ".join(combo),
                    "fold": i + 1,
                    "size": k,
                    "f1": m["f1"],
                    "recall": m["recall"],
                    "prec": m["precision"],
                    "avg_lead": m["avg_lead"],
                    "score": score
                })

    if not fold_rows:
        return pd.DataFrame(), pd.DataFrame()

    fold_results = pd.DataFrame(fold_rows)
    agg = (
        fold_results.groupby("combo")
        .agg(
            mean_f1=("f1", "mean"),
            mean_rec=("recall", "mean"),
            mean_prec=("prec", "mean"),
            mean_lead=("avg_lead", "mean"),
            mean_score=("score", "mean"),
            n_folds=("fold", "nunique"),
            size=("size", "first")
        )
        .reset_index()
        .query("n_folds >= 2")
        .sort_values(["n_folds", "mean_score", "mean_f1"], ascending=[False, False, False])
        .reset_index(drop=True)
    )
    return agg, fold_results


@st.cache_data(show_spinner=False)
def prepare_ml_fold_inputs_cached(master_ds, selected, horizon_years):
    later_df = pd.DataFrame(master_ds).copy()
    selected = list(selected)
    feature_cols = [f"{name}__pct" for name in selected if f"{name}__pct" in later_df.columns]
    empty_result = {
        "keep_features": [],
        "valid_folds": [],
        "eval_start": pd.NaT,
        "eval_end": pd.NaT,
        "n_realized_events": 0
    }

    if not feature_cols:
        return empty_result

    later_df = later_df.loc[later_df["Date"] >= pd.Timestamp(WF_SAMPLE_START_DATE)].dropna(subset=["target_downturn"]).reset_index(drop=True)
    coverage = later_df[feature_cols].notna().mean().sort_values(ascending=False)
    keep_features = coverage[coverage >= (1 - ML_MAX_MISSING_FEATURE_FRAC)].index.tolist()
    if not keep_features:
        return empty_result

    later_df[keep_features] = later_df[keep_features].ffill()
    first_test_start = WF_WARMUP_DAYS + WF_MIN_TRAIN_ROWS + int(round(float(horizon_years) * 252))
    remaining = len(later_df) - first_test_start
    if remaining <= 0:
        return empty_result

    fold_size = remaining // WF_N_FOLDS
    if fold_size < WF_MIN_TEST_ROWS:
        return empty_result

    valid_folds = []
    purge_days = int(round(float(horizon_years) * 252))
    for i in range(WF_N_FOLDS):
        test_start = first_test_start + i * fold_size
        test_end = first_test_start + (i + 1) * fold_size if i < WF_N_FOLDS - 1 else len(later_df)
        train_end = test_start - purge_days
        train_df = later_df.iloc[WF_WARMUP_DAYS:train_end].copy().reset_index(drop=True)
        test_df = later_df.iloc[test_start:test_end].copy().reset_index(drop=True)
        if len(train_df) < WF_MIN_TRAIN_ROWS or len(test_df) < WF_MIN_TEST_ROWS:
            continue
        medians = train_df[keep_features].median()
        train_df[keep_features] = train_df[keep_features].fillna(medians)
        test_df[keep_features] = test_df[keep_features].fillna(medians)
        y_train = train_df["target_downturn"].astype(int)
        if y_train.nunique() < 2:
            continue
        valid_folds.append((i + 1, train_df.to_dict("list"), test_df.to_dict("list")))

    if len(valid_folds) < 2:
        return empty_result

    eval_start = min(pd.to_datetime(test_df["Date"]).min() for _, _, test_df in [(fid, pd.DataFrame(tr), pd.DataFrame(te)) for fid, tr, te in valid_folds])
    eval_end = max(pd.to_datetime(test_df["Date"]).max() for _, _, test_df in [(fid, pd.DataFrame(tr), pd.DataFrame(te)) for fid, tr, te in valid_folds])
    realized_events_eval = (
        pd.concat([get_event_starts_from_target(pd.DataFrame(test_df)) for _, _, test_df in valid_folds], ignore_index=True)
        .dropna()
        .drop_duplicates()
        .sort_values()
        .reset_index(drop=True)
    )

    return {
        "keep_features": keep_features,
        "valid_folds": valid_folds,
        "eval_start": eval_start,
        "eval_end": eval_end,
        "n_realized_events": int(len(realized_events_eval))
    }


def choose_ml_model(model_type="random_forest"):
    if model_type == "decision_tree":
        return DecisionTreeClassifier(
            max_depth=ML_TREE_MAX_DEPTH,
            min_samples_leaf=ML_TREE_MIN_SAMPLES_LEAF,
            random_state=ML_RANDOM_STATE,
            class_weight="balanced"
        )

    return RandomForestClassifier(
        n_estimators=ML_RF_N_ESTIMATORS,
        max_depth=ML_RF_MAX_DEPTH,
        min_samples_leaf=ML_RF_MIN_SAMPLES_LEAF,
        random_state=ML_RANDOM_STATE,
        class_weight="balanced",
        n_jobs=-1
    )


def get_oof_train_probs(X_train, y_train, n_splits=ML_INNER_OOF_SPLITS, model_type="random_forest"):
    X_train = X_train.reset_index(drop=True)
    y_train = pd.Series(y_train).reset_index(drop=True)
    oof_prob = pd.Series(np.nan, index=X_train.index, dtype=float)
    if len(X_train) < (n_splits + 1) * 20 or y_train.nunique() < 2:
        return oof_prob.values
    tscv = TimeSeriesSplit(n_splits=n_splits)
    for inner_train_idx, inner_val_idx in tscv.split(X_train):
        X_tr = X_train.iloc[inner_train_idx]
        y_tr = y_train.iloc[inner_train_idx]
        X_val = X_train.iloc[inner_val_idx]
        if y_tr.nunique() < 2:
            continue
        model = choose_ml_model(model_type=model_type)
        model.fit(X_tr, y_tr)
        oof_prob.iloc[inner_val_idx] = model.predict_proba(X_val)[:, 1]
    return oof_prob.values


@st.cache_data(show_spinner=False)
def run_ml_search_cached(master_ds, selected, horizon_years, max_combo_size, model_type="random_forest"):
    prep = prepare_ml_fold_inputs_cached(master_ds, selected, horizon_years)
    keep_features = prep["keep_features"]
    valid_folds = [
        (fold_id, pd.DataFrame(train_dict), pd.DataFrame(test_dict))
        for fold_id, train_dict, test_dict in prep["valid_folds"]
    ]
    if not keep_features:
        return {
            "single_summary": pd.DataFrame(),
            "combo_summary": pd.DataFrame(),
            "combo_fold_results": pd.DataFrame(),
            "individual_combo_table": pd.DataFrame(),
            "augmented_combo_table": pd.DataFrame(),
            "eval_start": pd.NaT,
            "eval_end": pd.NaT,
            "n_realized_events": 0,
            "tree_details": None
        }
    if len(valid_folds) < 2:
        return {
            "single_summary": pd.DataFrame(),
            "combo_summary": pd.DataFrame(),
            "combo_fold_results": pd.DataFrame(),
            "individual_combo_table": pd.DataFrame(),
            "augmented_combo_table": pd.DataFrame(),
            "eval_start": pd.NaT,
            "eval_end": pd.NaT,
            "n_realized_events": 0,
            "tree_details": None
        }

    single_rows = []
    for feat in keep_features:
        for fold_id, train_df, test_df in valid_folds:
            X_train = train_df[[feat]].astype(float)
            X_test = test_df[[feat]].astype(float)
            y_train = train_df["target_downturn"].astype(int)
            if y_train.nunique() < 2:
                continue
            oof_train_prob = get_oof_train_probs(X_train, y_train, model_type=model_type)
            threshold = fit_alert_threshold(oof_train_prob)
            if pd.isna(threshold):
                continue
            model = choose_ml_model(model_type=model_type)
            model.fit(X_train, y_train)
            test_prob = model.predict_proba(X_test)[:, 1]
            metrics, _ = compute_ml_fold_alarm_metrics(test_df, test_prob, threshold, horizon_years)
            single_rows.append({
                "indicator": feat.replace("__pct", ""),
                "fold": fold_id,
                "test_precision": metrics["precision"],
                "test_recall": metrics["recall"],
                "test_f1": metrics["f1"],
                "test_avg_lead": metrics["avg_lead"],
                "n_caught": metrics["n_caught"]
            })
    single_results = pd.DataFrame(single_rows)
    if single_results.empty:
        return {
            "single_summary": pd.DataFrame(),
            "combo_summary": pd.DataFrame(),
            "combo_fold_results": pd.DataFrame(),
            "individual_combo_table": pd.DataFrame(),
            "augmented_combo_table": pd.DataFrame(),
            "eval_start": pd.NaT,
            "eval_end": pd.NaT,
            "n_realized_events": 0,
            "tree_details": None
        }

    single_summary = (
        single_results.groupby("indicator", as_index=False)
        .agg(
            mean_test_precision=("test_precision", "mean"),
            mean_test_recall=("test_recall", "mean"),
            mean_test_f1=("test_f1", "mean"),
            mean_test_avg_lead=("test_avg_lead", "mean"),
            mean_n_caught=("n_caught", "mean"),
            n_folds=("fold", "nunique")
        )
        .sort_values(["mean_test_f1", "mean_test_recall", "mean_test_precision"], ascending=False)
        .reset_index(drop=True)
    )

    finalist_features = [f"{name}__pct" for name in single_summary.head(min(ML_N_FINALISTS, len(single_summary)))["indicator"].tolist()]
    combo_rows = []
    for k in range(1, min(max_combo_size, len(finalist_features)) + 1):
        for combo in combinations(finalist_features, k):
            for fold_id, train_df, test_df in valid_folds:
                X_train = train_df[list(combo)].astype(float)
                X_test = test_df[list(combo)].astype(float)
                y_train = train_df["target_downturn"].astype(int)
                if y_train.nunique() < 2:
                    continue
                oof_train_prob = get_oof_train_probs(X_train, y_train, model_type=model_type)
                threshold = fit_alert_threshold(oof_train_prob)
                if pd.isna(threshold):
                    continue
                model = choose_ml_model(model_type=model_type)
                model.fit(X_train, y_train)
                test_prob = model.predict_proba(X_test)[:, 1]
                metrics, _ = compute_ml_fold_alarm_metrics(test_df, test_prob, threshold, horizon_years)
                combo_rows.append({
                    "combo": " + ".join([c.replace("__pct", "") for c in combo]),
                    "size": len(combo),
                    "fold": fold_id,
                    "test_precision": metrics["precision"],
                    "test_recall": metrics["recall"],
                    "test_f1": metrics["f1"],
                    "test_avg_lead": metrics["avg_lead"],
                    "n_caught": metrics["n_caught"]
                })
    combo_fold_results = pd.DataFrame(combo_rows)
    if combo_fold_results.empty:
        return {
            "single_summary": single_summary,
            "combo_summary": pd.DataFrame(),
            "combo_fold_results": pd.DataFrame(),
            "individual_combo_table": pd.DataFrame(),
            "augmented_combo_table": pd.DataFrame(),
            "eval_start": pd.NaT,
            "eval_end": pd.NaT,
            "n_realized_events": 0,
            "tree_details": None
        }

    combo_summary = (
        combo_fold_results.groupby("combo", as_index=False)
        .agg(
            size=("size", "first"),
            mean_test_precision=("test_precision", "mean"),
            mean_test_recall=("test_recall", "mean"),
            mean_test_f1=("test_f1", "mean"),
            mean_test_avg_lead=("test_avg_lead", "mean"),
            mean_n_caught=("n_caught", "mean"),
            n_folds=("fold", "nunique")
        )
        .query("n_folds >= @ML_MIN_FOLDS_REQUIRED")
        .sort_values(["mean_test_f1", "mean_test_recall", "mean_test_precision"], ascending=False)
        .reset_index(drop=True)
    )
    realized_events_eval = (
        pd.concat([get_event_starts_from_target(test_df) for _, _, test_df in valid_folds], ignore_index=True)
        .dropna()
        .drop_duplicates()
        .sort_values()
        .reset_index(drop=True)
    )
    eval_start = min(test_df["Date"].min() for _, _, test_df in valid_folds)
    eval_end = max(test_df["Date"].max() for _, _, test_df in valid_folds)
    individual_combo_table = build_individual_combo_table(
        combo_summary_df=combo_summary,
        evaluation_folds=valid_folds,
        realized_events_eval=realized_events_eval,
        horizon_years=horizon_years,
        top_n=ML_TOP_COMBOS,
        model_type=model_type
    )
    augmented_combo_table = build_augmented_combo_table(
        combo_summary_df=combo_summary,
        evaluation_folds=valid_folds,
        realized_events_eval=realized_events_eval,
        keep_features=keep_features,
        horizon_years=horizon_years,
        top_n=ML_TOP_COMBOS,
        model_type=model_type
    )
    tree_details = None
    if model_type == "decision_tree" and not combo_summary.empty:
        tree_details = build_decision_tree_details(
            combo_name=combo_summary.iloc[0]["combo"],
            fold_data=valid_folds,
            horizon_years=horizon_years
        )
    return {
        "single_summary": single_summary,
        "combo_summary": combo_summary,
        "combo_fold_results": combo_fold_results,
        "individual_combo_table": individual_combo_table,
        "augmented_combo_table": augmented_combo_table,
        "eval_start": prep["eval_start"],
        "eval_end": prep["eval_end"],
        "n_realized_events": prep["n_realized_events"],
        "tree_details": tree_details
    }


def compute_ml_fold_alarm_metrics(test_df, test_prob, threshold, horizon_years):
    test_engine = build_trigger_engine(test_prob, threshold)
    trigger_dates = test_df.loc[test_engine["event"].values == 1, "Date"].reset_index(drop=True)
    event_starts = get_event_starts_from_target(test_df)
    metrics = compute_distinct_crisis_metrics(trigger_dates, event_starts, horizon_days=int(round(horizon_years * 252)))
    return metrics, trigger_dates


def get_event_starts_from_target(df):
    y = pd.Series(df["target_downturn"]).fillna(0).astype(int).reset_index(drop=True)
    dates = pd.to_datetime(df["Date"]).reset_index(drop=True)
    starts = []
    prev = 0
    for i in range(len(y)):
        if y.iloc[i] == 1 and prev == 0:
            starts.append(dates.iloc[i])
        prev = y.iloc[i]
    return pd.Series(starts, dtype="datetime64[ns]")


def combo_name_to_feature_list(combo_name):
    return [f"{x.strip()}__pct" for x in combo_name.split(" + ")]


def get_trigger_dates_for_combo(combo_name, fold_data, model_type="random_forest"):
    feats = combo_name_to_feature_list(combo_name)
    all_trigger_dates = []

    for _, train_df, test_df in fold_data:
        if any(f not in train_df.columns for f in feats):
            continue

        X_train = train_df[feats].astype(float)
        X_test = test_df[feats].astype(float)
        y_train = train_df["target_downturn"].astype(int)

        if y_train.nunique() < 2:
            continue

        oof_train_prob = get_oof_train_probs(X_train, y_train, model_type=model_type)
        threshold = fit_alert_threshold(oof_train_prob)
        if pd.isna(threshold):
            continue

        model = choose_ml_model(model_type=model_type)
        model.fit(X_train, y_train)
        test_prob = model.predict_proba(X_test)[:, 1]
        test_engine = build_trigger_engine(test_prob, threshold)
        fold_trigger_dates = test_df.loc[test_engine["event"].values == 1, "Date"].reset_index(drop=True)
        all_trigger_dates.append(pd.to_datetime(fold_trigger_dates))

    if not all_trigger_dates:
        return pd.Series([], dtype="datetime64[ns]")

    return (
        pd.concat(all_trigger_dates, ignore_index=True)
        .dropna()
        .drop_duplicates()
        .sort_values()
        .reset_index(drop=True)
    )


def build_individual_combo_table(combo_summary_df, evaluation_folds, realized_events_eval, horizon_years, top_n=ML_TOP_COMBOS, model_type="random_forest"):
    if combo_summary_df.empty:
        return pd.DataFrame()

    rows = []
    for combo_name in combo_summary_df.head(top_n)["combo"].tolist():
        trigger_dates = get_trigger_dates_for_combo(combo_name, evaluation_folds, model_type=model_type)
        m = compute_distinct_crisis_metrics(
            trigger_dates=trigger_dates,
            event_dates=realized_events_eval,
            horizon_days=int(round(horizon_years * 252))
        )
        rows.append({
            "combo": combo_name,
            "precision": m["precision"],
            "recall": m["recall"],
            "f1": m["f1"],
            "avg_lead": m["avg_lead"],
            "n_triggers": m["n_triggers"],
            "n_crises": m["n_crises"],
            "n_caught": m["n_caught"]
        })

    if not rows:
        return pd.DataFrame()

    return (
        pd.DataFrame(rows)
        .sort_values(["recall", "f1", "precision", "avg_lead"], ascending=False)
        .reset_index(drop=True)
    )


def build_augmented_combo_table(combo_summary_df, evaluation_folds, realized_events_eval, keep_features, horizon_years, top_n=ML_TOP_COMBOS, model_type="random_forest"):
    if combo_summary_df.empty:
        return pd.DataFrame()

    top_combos = combo_summary_df.head(top_n)["combo"].tolist()
    keep_feature_names = {c.replace("__pct", "") for c in keep_features}
    available_augmenters = [x for x in REGIME_AUGMENT_CANDIDATES if x in keep_feature_names]
    rows = []

    for base_combo in top_combos:
        base_triggers = get_trigger_dates_for_combo(base_combo, evaluation_folds, model_type=model_type)
        base_parts = {x.strip() for x in base_combo.split(" + ")}

        for aug_indicator in available_augmenters:
            if aug_indicator in base_parts:
                continue

            aug_triggers = get_trigger_dates_for_combo(aug_indicator, evaluation_folds, model_type=model_type)
            union_triggers = (
                pd.concat([pd.Series(base_triggers), pd.Series(aug_triggers)], ignore_index=True)
                .dropna()
                .drop_duplicates()
                .sort_values()
                .reset_index(drop=True)
            )
            m = compute_distinct_crisis_metrics(
                trigger_dates=union_triggers,
                event_dates=realized_events_eval,
                horizon_days=int(round(horizon_years * 252))
            )
            rows.append({
                "base_combo": base_combo,
                "augmenter": aug_indicator,
                "augmented_combo": f"{base_combo} || {aug_indicator}",
                "precision": m["precision"],
                "recall": m["recall"],
                "f1": m["f1"],
                "avg_lead": m["avg_lead"],
                "n_triggers": m["n_triggers"],
                "n_crises": m["n_crises"],
                "n_caught": m["n_caught"]
            })

    if not rows:
        return pd.DataFrame()

    return (
        pd.DataFrame(rows)
        .sort_values(["recall", "f1", "precision", "avg_lead"], ascending=False)
        .reset_index(drop=True)
    )


def build_decision_tree_details(combo_name, fold_data, horizon_years):
    feats = combo_name_to_feature_list(combo_name)
    if not fold_data:
        return None

    fold_id, train_df, test_df = fold_data[-1]
    if any(f not in train_df.columns for f in feats):
        return None

    X_train = train_df[feats].astype(float)
    X_test = test_df[feats].astype(float)
    y_train = train_df["target_downturn"].astype(int)
    if y_train.nunique() < 2:
        return None

    oof_train_prob = get_oof_train_probs(X_train, y_train, model_type="decision_tree")
    threshold = fit_alert_threshold(oof_train_prob)
    if pd.isna(threshold):
        return None

    model = choose_ml_model(model_type="decision_tree")
    model.fit(X_train, y_train)
    test_prob = model.predict_proba(X_test)[:, 1]
    feature_names = [f.replace("__pct", "") for f in feats]
    rules_text = export_text(model, feature_names=feature_names)
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False).reset_index(drop=True)
    test_view = pd.DataFrame({
        "Date": pd.to_datetime(test_df["Date"]),
        "probability": test_prob,
        "target_downturn": pd.Series(test_df["target_downturn"]).fillna(0).astype(int)
    })
    test_view["event_start"] = (
        (test_view["target_downturn"] == 1) &
        (test_view["target_downturn"].shift(1).fillna(0) == 0)
    )

    return {
        "combo": combo_name,
        "fold": int(fold_id),
        "threshold": float(threshold),
        "rules_text": rules_text,
        "importance": importance_df.to_dict("records"),
        "test_view": test_view.to_dict("list"),
        "horizon_years": float(horizon_years)
    }


@st.cache_data(show_spinner=False)
def build_decision_tree_details_cached(master_ds, selected, horizon_years, combo_name):
    prep = prepare_ml_fold_inputs_cached(master_ds, selected, horizon_years)
    valid_folds = [
        (fold_id, pd.DataFrame(train_dict), pd.DataFrame(test_dict))
        for fold_id, train_dict, test_dict in prep["valid_folds"]
    ]
    if not valid_folds:
        return None
    return build_decision_tree_details(combo_name, valid_folds, horizon_years)


def apply_layout(fig, title="", height=450):
    fig.update_layout(
        template="plotly_white",
        title=dict(text=title, x=0, font=dict(size=15, color=TEXT)),
        height=height,
        margin=dict(l=52, r=20, t=56, b=42),
        paper_bgcolor=BG,
        plot_bgcolor=BG2,
        hovermode="closest",
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02,
            xanchor="right", x=1, bgcolor="rgba(255,255,255,0.8)"
        )
    )
    fig.update_xaxes(showgrid=True, gridcolor=BORDER, zeroline=False, linecolor=BORDER)
    fig.update_yaxes(showgrid=True, gridcolor=BORDER, zeroline=False, linecolor=BORDER)


# ══════════════════════════════════════════════════════════════════════════════
# 6. APP STATE
# ══════════════════════════════════════════════════════════════════════════════
def init_state(all_indicator_names):
    if "shared_selected" not in st.session_state:
        st.session_state.shared_selected = [x for x in ["Shiller CAPE", "Tobin Q"] if x in all_indicator_names]
    if "shared_search_input" not in st.session_state:
        st.session_state.shared_search_input = ""


def get_combo_selected():
    return list(st.session_state.shared_selected)


def set_combo_selection(names, all_indicator_names):
    st.session_state.shared_selected = [x for x in names if x in all_indicator_names]


def render_selection_pills(selected):
    if not selected:
        st.markdown(f"<div class='small-muted'>No indicators selected.</div>", unsafe_allow_html=True)
        return
    pills = "".join([f"<span class='metric-pill'>{x}</span>" for x in selected])
    st.markdown(
        f"<div class='small-muted' style='font-weight:700;margin-bottom:4px;'>Selected ({len(selected)})</div>{pills}",
        unsafe_allow_html=True
    )


def render_shared_indicator_picker(all_indicator_names):
    current = get_combo_selected()
    current = get_combo_selected()
    pills = "".join([f"<span class='metric-pill'>{x}</span>" for x in current]) if current else ""
    summary_html = f"""
        <div style="
            padding:12px 14px;
            border-radius:16px;
            border:1px solid {BORDER};
            background:linear-gradient(135deg,#f8fbff 0%,#ffffff 100%);
            margin:4px 0 10px;">
            <div style="display:flex;justify-content:space-between;gap:12px;align-items:flex-start;flex-wrap:wrap;">
                <div>
                    <div style="font-size:12px;font-weight:800;color:{TEXT};">Selected indicators</div>
                    <div style="font-size:11px;color:{MUTED};margin-top:3px;">{len(current)} selected</div>
                </div>
                <div style="font-size:11px;color:{MUTED};font-weight:700;">Edit below</div>
            </div>
            <div style="margin-top:8px;">{pills if pills else f"<span class='small-muted'>No indicators selected yet.</span>"}</div>
        </div>
    """
    st.markdown(summary_html, unsafe_allow_html=True)

    with st.expander("Edit indicators", expanded=(len(current) == 0)):
        c1, c2 = st.columns([1.3, 1])
        with c1:
            st.markdown(
                f"<div style='font-size:12px;font-weight:800;color:{TEXT};margin:0 0 6px;'>Choose your basket</div>",
                unsafe_allow_html=True
            )
        with c2:
            st.text_input("Search", key="shared_search_input", placeholder="Search indicators...", label_visibility="collapsed")

        query = st.session_state.shared_search_input.strip().lower()
        visible_options = [x for x in all_indicator_names if not query or query in x.lower()]
        if not visible_options:
            visible_options = all_indicator_names

        c1, c2, c3, c4 = st.columns([1.3, 0.8, 0.8, 0.7])
        with c1:
            if st.button("Froth Defaults", key="picker_defaults", use_container_width=True):
                set_combo_selection(["Shiller CAPE", "Tobin Q", "Buffet Indicator", "Equity Risk Premium"], all_indicator_names)
                st.experimental_rerun()
        with c2:
            if st.button("Top 2", key="picker_top2", use_container_width=True):
                cur = get_combo_selected()
                set_combo_selection(cur[:2] if cur else ["Shiller CAPE", "Tobin Q"], all_indicator_names)
                st.experimental_rerun()
        with c3:
            if st.button("Top 4", key="picker_top4", use_container_width=True):
                cur = get_combo_selected()
                set_combo_selection(cur[:4] if cur else ["Shiller CAPE", "Tobin Q", "Buffet Indicator", "Equity Risk Premium"], all_indicator_names)
                st.experimental_rerun()
        with c4:
            if st.button("Clear", key="picker_clear", use_container_width=True):
                set_combo_selection([], all_indicator_names)
                st.experimental_rerun()

        current = get_combo_selected()
        st.markdown(
            f"<div class='small-muted' style='font-weight:700;margin:8px 0 6px;'>Browse indicators</div>",
        unsafe_allow_html=True
        )
        checkbox_cols = st.columns(3)
        updated_selection = list(current)
        for idx, name in enumerate(visible_options):
            key = f"indicator_checkbox_{name}"
            checked_now = name in current
            col = checkbox_cols[idx % 3]
            with col:
                checked = st.checkbox(name, value=checked_now, key=key)
            if checked and name not in updated_selection:
                updated_selection.append(name)
            if not checked and name in updated_selection:
                updated_selection.remove(name)

        if updated_selection != current:
            set_combo_selection(updated_selection, all_indicator_names)


# ══════════════════════════════════════════════════════════════════════════════
# 7. REGIME HELPER
# ══════════════════════════════════════════════════════════════════════════════
def render_tab_intro(title, body):
    st.markdown(
        f"""
        <div class="info-card" style="margin-top:2px;margin-bottom:14px;">
            <div style="font-size:12px;font-weight:800;color:{TEXT};margin-bottom:4px;">{title}</div>
            <div style="font-size:12px;color:{MUTED};line-height:1.45;">{body}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def get_regime_mask(technicals, merged_df, mode):
    if mode == "all" or "Price" not in technicals.columns:
        return pd.Series(True, index=merged_df.index)

    tech = technicals[["Date", "Price", "SMA200"]].dropna().copy()
    tech["Date"] = pd.to_datetime(tech["Date"], errors="coerce", utc=True).dt.tz_localize(None)
    tech = tech.sort_values("Date")
    tech["bull"] = tech["Price"] > tech["SMA200"]

    aligned = pd.merge_asof(
        merged_df[["Date"]].reset_index(),
        tech[["Date", "bull"]],
        on="Date",
        direction="backward"
    ).set_index("index")["bull"].fillna(True)

    return aligned if mode == "bull" else ~aligned


def _apply_rule(flag_df, selected, rule, n, composite_scores=None, q=None):
    tail_count = flag_df[selected].sum(axis=1)
    if rule == "all":
        return tail_count == len(selected), "All selected indicators in tail"
    elif rule == "any":
        return tail_count >= 1, "Any selected indicator in tail"
    elif rule == "at_least_n":
        return tail_count >= n, f"At least {n} indicators in tail"
    else:
        return composite_scores >= q, "Composite percentile score"


def mask_to_periods(dates, mask):
    dates = pd.to_datetime(pd.Series(dates)).reset_index(drop=True)
    mask = pd.Series(mask).fillna(False).astype(bool).reset_index(drop=True)
    periods = []
    start = None
    prev_date = None

    for date_val, is_on in zip(dates, mask):
        if is_on and start is None:
            start = date_val
        if not is_on and start is not None:
            periods.append((start, prev_date))
            start = None
        prev_date = date_val

    if start is not None and prev_date is not None:
        periods.append((start, prev_date))

    return periods


def mask_to_starts(dates, mask):
    dates = pd.to_datetime(pd.Series(dates)).reset_index(drop=True)
    mask = pd.Series(mask).fillna(False).astype(bool).reset_index(drop=True)
    starts = []
    prev_on = False

    for date_val, is_on in zip(dates, mask):
        if is_on and not prev_on:
            starts.append(date_val)
        prev_on = is_on

    return starts


# ══════════════════════════════════════════════════════════════════════════════
# 8. PAGES
# ══════════════════════════════════════════════════════════════════════════════
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


def render_predictor_page(indicators, spx_c):
    st.subheader("Downturn Predictor Search")
    render_tab_intro(
        "How this works",
        "Selected indicators are first aligned to the SPX history and converted into expanding percentile signals, so each date is ranked only against information available up to that date. "
        "Downturn trigger defines what size future selloff counts as the event to predict, and Predict window defines how far ahead the model looks for that event. "
        "The tab then compares three approaches: Walk Forward tests simple average baskets, Random Forest builds an ensemble classifier, and Decision Tree builds a smaller rule-based classifier that is easier to inspect."
    )
    render_shared_indicator_picker(sorted(indicators.keys()))
    selected = get_combo_selected()

    with st.expander("ℹ️ How to use this tab", expanded=False):
        st.markdown("""
        - **Selected indicators** – These are the raw signals the tab turns into expanding percentile features. The same basket is used by all three models.
        - **Downturn trigger** – This is the future selloff size the models try to warn about. A higher value means fewer but more severe target events.
        - **Predict window (Y)** – A date is labeled positive if a downturn of that size begins within this many years after the date. A longer window usually increases recall and lowers precision.
        - **Walk Forward model** – A simple basket model. It averages selected percentile signals and checks whether the basket fired before later downturns across rolling folds.
        - **Random Forest model** – An ensemble of many small trees. It is usually the most stable nonlinear model here, but hardest to interpret rule by rule.
        - **Decision Tree model** – A single small tree. It is less stable than the forest, but much easier to understand because you can inspect the exact split rules.
        - **Mean Precision / Mean Recall / Mean F1** – These are averaged over walk-forward folds, so they reflect out-of-sample behavior rather than one static backtest window.
        - **Mean Lead (days)** – Average number of days between the first warning and the later downturn event for warnings that were counted as successful.
        """)

    if not selected:
        st.warning("Select at least one indicator.")
        return

    dd_thresh = st.slider("Downturn trigger", 0.10, 0.40, 0.20, 0.01, key="pred_trigger")
    horizon = st.slider("Predict window (Y)", 0.25, 3.0, 1.0, 0.25, key="pred_horizon")
    master_ds = build_selected_master_dataset(
        indicators=indicators,
        spx_c=spx_c,
        selected=selected,
        drawdown_threshold=float(dd_thresh),
        lead_days=int(round(float(horizon) * 252))
    )
    wf_agg, wf_folds = run_walk_forward_search_cached(
        master_ds=master_ds.to_dict("list"),
        selected=tuple(selected),
        horizon_years=float(horizon),
        max_combo_size=min(4, len(selected))
    )
    ml_master = build_percentile_feature_master(master_ds.to_dict("list"), tuple(selected))
    rf_results = run_ml_search_cached(
        master_ds=ml_master.to_dict("list"),
        selected=tuple(selected),
        horizon_years=float(horizon),
        max_combo_size=min(3, len(selected)),
        model_type="random_forest"
    )
    dt_results = run_ml_search_cached(
        master_ds=ml_master.to_dict("list"),
        selected=tuple(selected),
        horizon_years=float(horizon),
        max_combo_size=min(3, len(selected)),
        model_type="decision_tree"
    )
    predictor_tabs = st.tabs(["Walk Forward Model", "Random Forest Model", "Decision Tree Model"])

    with predictor_tabs[0]:
        st.markdown(
            """
            <div class="info-card">
                <div style="font-size:12px;font-weight:800;color:#0f172a;margin-bottom:4px;">Walk Forward model</div>
                <div style="font-size:12px;color:#64748b;line-height:1.45;">
                    This is the simplest and most transparent approach on the page. For each fold, the app converts the selected indicators into percentile signals, averages them into a basket score, fits an alert threshold using only training data, and then tests whether that basket warned early enough before later downturns in the next holdout fold.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        if wf_agg.empty:
            st.info("Walk-forward search did not produce enough valid folds for the current settings.")
        else:
            best_wf = wf_agg.iloc[0]
            c1, c2, c3 = st.columns(3)
            c1.metric("Best walk-forward combo", best_wf["combo"], f"Mean F1: {best_wf['mean_f1']:.3f}")
            c2.metric("Mean recall", f"{best_wf['mean_rec']:.1%}", f"Mean precision: {best_wf['mean_prec']:.1%}")
            c3.metric("Mean lead", f"{best_wf['mean_lead']:.0f} days", f"Across {int(best_wf['n_folds'])} folds")
            st.dataframe(
                wf_agg.head(20).rename(columns={
                    "combo": "Combo",
                    "mean_f1": "Mean F1",
                    "mean_rec": "Mean Recall",
                    "mean_prec": "Mean Precision",
                    "mean_lead": "Mean Lead (days)",
                    "mean_score": "Mean Score",
                    "n_folds": "Folds",
                    "size": "Size"
                }),
                use_container_width=True
            )
            with st.expander("Fold-level walk-forward results", expanded=False):
                st.dataframe(
                    wf_folds.rename(columns={
                        "combo": "Combo",
                        "fold": "Fold",
                        "size": "Size",
                        "f1": "F1",
                        "recall": "Recall",
                        "prec": "Precision",
                        "avg_lead": "Avg Lead (days)",
                        "score": "Score"
                    }),
                    use_container_width=True
                )

    def render_model_results(results, model_label, model_body, show_tree_view=False, tree_master_ds=None, tree_selected=None, tree_horizon=None):
        st.markdown(
            f"""
            <div class="info-card">
                <div style="font-size:12px;font-weight:800;color:#0f172a;margin-bottom:4px;">{model_label}</div>
                <div style="font-size:12px;color:#64748b;line-height:1.45;">{model_body}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

        if results["single_summary"].empty and results["combo_summary"].empty:
            st.info(f"{model_label} did not produce enough valid folds for the current settings.")
            return

        if pd.notna(results["eval_start"]) and pd.notna(results["eval_end"]):
            st.markdown(
                f"""
                <div class="info-card">
                    <div style="font-size:12px;font-weight:800;color:{TEXT};margin-bottom:4px;">Walk-forward evaluation window</div>
                    <div style="font-size:12px;color:{MUTED};line-height:1.45;">
                        {pd.Timestamp(results["eval_start"]).strftime("%Y-%m-%d")} to {pd.Timestamp(results["eval_end"]).strftime("%Y-%m-%d")}
                        · realized downturns across folds: {results["n_realized_events"]}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        tab_names = ["Singles", "Individual Combos", "Augmented Combos", "Fold Detail"]
        if show_tree_view:
            tab_names.append("Tree View")
        ml_tabs = st.tabs(tab_names)

        with ml_tabs[0]:
            if results["single_summary"].empty:
                st.info("No single-indicator results available.")
            else:
                st.dataframe(
                    results["single_summary"].head(20).rename(columns={
                        "indicator": "Indicator",
                        "mean_test_precision": "Mean Precision",
                        "mean_test_recall": "Mean Recall",
                        "mean_test_f1": "Mean F1",
                        "mean_test_avg_lead": "Mean Lead (days)",
                        "mean_n_caught": "Mean Crises Caught",
                        "n_folds": "Folds"
                    }),
                    use_container_width=True
                )

        with ml_tabs[1]:
            if results["individual_combo_table"].empty:
                st.info("No individual combo table was produced for the current settings.")
            else:
                st.dataframe(
                    results["individual_combo_table"].rename(columns={
                        "combo": "Combo",
                        "precision": "Precision",
                        "recall": "Recall",
                        "f1": "F1",
                        "avg_lead": "Avg Lead (days)",
                        "n_triggers": "Triggers",
                        "n_crises": "Crises",
                        "n_caught": "Crises Caught"
                    }),
                    use_container_width=True
                )

        with ml_tabs[2]:
            if results["augmented_combo_table"].empty:
                st.info("No augmented combo table was produced from the selected indicators.")
            else:
                st.dataframe(
                    results["augmented_combo_table"].rename(columns={
                        "base_combo": "Base Combo",
                        "augmenter": "Augmenter",
                        "augmented_combo": "Augmented Combo",
                        "precision": "Precision",
                        "recall": "Recall",
                        "f1": "F1",
                        "avg_lead": "Avg Lead (days)",
                        "n_triggers": "Triggers",
                        "n_crises": "Crises",
                        "n_caught": "Crises Caught"
                    }),
                    use_container_width=True
                )

        with ml_tabs[3]:
            if results["combo_summary"].empty:
                st.info("No combo-level results available.")
            else:
                st.dataframe(
                    results["combo_summary"].head(20).rename(columns={
                        "combo": "Combo",
                        "size": "Size",
                        "mean_test_precision": "Mean Precision",
                        "mean_test_recall": "Mean Recall",
                        "mean_test_f1": "Mean F1",
                        "mean_test_avg_lead": "Mean Lead (days)",
                        "mean_n_caught": "Mean Crises Caught",
                        "n_folds": "Folds"
                    }),
                    use_container_width=True
                )
                with st.expander("Fold-level combo results", expanded=False):
                    st.dataframe(
                        results["combo_fold_results"].rename(columns={
                            "combo": "Combo",
                            "size": "Size",
                            "fold": "Fold",
                            "test_precision": "Precision",
                            "test_recall": "Recall",
                            "test_f1": "F1",
                            "test_avg_lead": "Avg Lead (days)",
                            "n_caught": "Crises Caught"
                        }),
                        use_container_width=True
                    )

        if show_tree_view:
            with ml_tabs[4]:
                combo_options = results["combo_summary"]["combo"].head(12).tolist() if not results["combo_summary"].empty else []
                if combo_options:
                    default_idx = 0
                    for idx, combo_name in enumerate(combo_options):
                        if " + " in combo_name:
                            default_idx = idx
                            break
                    chosen_combo = st.selectbox(
                        "Combo to inspect",
                        combo_options,
                        index=default_idx,
                        key="decision_tree_combo_inspect"
                    )
                else:
                    chosen_combo = None

                tree_details = (
                    build_decision_tree_details_cached(tree_master_ds, tree_selected, tree_horizon, chosen_combo)
                    if chosen_combo and tree_master_ds is not None and tree_selected is not None and tree_horizon is not None
                    else None
                )
                if not tree_details:
                    st.info("No decision-tree visualization was produced for the current settings.")
                else:
                    st.markdown(
                        """
                        <div class="info-card">
                            <div style="font-size:12px;font-weight:800;color:#0f172a;margin-bottom:4px;">How the tree separates downturn risk</div>
                            <div style="font-size:12px;color:#64748b;line-height:1.45;">
                                The chart below shows the chosen decision-tree combo on the latest walk-forward fold. Feature importance is normalized, so the values sum to 1 across the features actually used by that tree. The purple line is the predicted downturn probability, the dashed red line is the alert threshold, and red markers show dates inside actual downturn-warning windows.
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    imp_df = pd.DataFrame(tree_details["importance"])
                    if not imp_df.empty:
                        fig_imp = go.Figure()
                        fig_imp.add_trace(go.Bar(
                            x=imp_df["importance"].astype(float).tolist(),
                            y=imp_df["feature"].tolist(),
                            orientation="h",
                            marker=dict(color=ORANGE),
                            hovertemplate="<b>%{y}</b><br>Importance: %{x:.3f}<extra></extra>"
                        ))
                        apply_layout(fig_imp, f"Decision tree feature importance — {tree_details['combo']}", 300)
                        fig_imp.update_xaxes(title="Importance")
                        fig_imp.update_yaxes(title="", autorange="reversed")
                        st.plotly_chart(fig_imp, use_container_width=True)

                    test_view = pd.DataFrame(tree_details["test_view"])
                    if not test_view.empty:
                        fig_prob = go.Figure()
                        fig_prob.add_trace(go.Scatter(
                            x=pd.to_datetime(test_view["Date"]).tolist(),
                            y=pd.to_numeric(test_view["probability"]).astype(float).tolist(),
                            mode="lines",
                            name="Predicted probability",
                            line=dict(color=PURPLE, width=2)
                        ))
                        pos = test_view[test_view["target_downturn"] == 1]
                        if not pos.empty:
                            fig_prob.add_trace(go.Scatter(
                                x=pd.to_datetime(pos["Date"]).tolist(),
                                y=pd.to_numeric(pos["probability"]).astype(float).tolist(),
                                mode="markers",
                                name="Actual downturn window",
                                marker=dict(color=RED, size=8)
                            ))
                        fig_prob.add_hline(
                            y=float(tree_details["threshold"]),
                            line_dash="dot",
                            line_color=RED,
                            annotation_text=f"Alert threshold: {tree_details['threshold']:.2f}",
                            annotation_position="top right"
                        )
                        apply_layout(fig_prob, f"Decision tree separation on latest fold — fold {tree_details['fold']}", 360)
                        fig_prob.update_yaxes(title="Predicted probability", range=[0, 1])
                        st.plotly_chart(fig_prob, use_container_width=True)

                    with st.expander("Decision tree rules", expanded=False):
                        st.code(tree_details["rules_text"], language="text")

    with predictor_tabs[1]:
        render_model_results(
            rf_results,
            "Random Forest model",
            "Random Forest combines many shallow trees and averages them. In this tab it is still evaluated in walk-forward fashion across the valid folds, so it is not just training on one late period like a static backtest would. It is usually more stable than a single tree, captures nonlinear interactions between indicators, and tends to smooth out noisy one-off splits, but the final signal is harder to interpret rule by rule.",
            show_tree_view=False
        )

    with predictor_tabs[2]:
        render_model_results(
            dt_results,
            "Decision Tree model",
            "Decision Tree uses one compact set of split rules. In this tab it is also evaluated across the walk-forward folds, but unlike the forest you can inspect the actual branching logic that turned indicator percentiles into higher or lower downturn risk. That makes it useful when you want a model that is easier to explain, even if it is usually less robust than the forest.",
            show_tree_view=True,
            tree_master_ds=ml_master.to_dict("list"),
            tree_selected=tuple(selected),
            tree_horizon=float(horizon)
        )



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

    page = st.radio(
        "View",
        ["Indicator Analysis", "SPX Technicals", "Tail Combo", "Downturn Predictor Search"],
        horizontal=True,
        key="page_selector",
        label_visibility="collapsed"
    )

    if page == "Indicator Analysis":
        render_indicator_page(indicators, spx_c)
    elif page == "SPX Technicals":
        render_tech_page(spx_c, technicals)
    elif page == "Tail Combo":
        render_combo_page(indicators, spx_c, technicals)
    else:
        render_predictor_page(indicators, spx_c)


if __name__ == "__main__":
    main()
