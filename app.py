"""
Market Froth Dashboard — Streamlit version
Converted from Colab ipywidgets notebook.

Run with:
    streamlit run froth_app.py

Put all your data files in the same directory (or adjust DATA_DIR below).
"""

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from io import StringIO
from itertools import combinations
from bisect import bisect_right, insort

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Market Froth Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Data directory ────────────────────────────────────────────────────────────
# Change this if your files live somewhere else
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
    "Earnings Yield", "Dividend Yield", "Equity Risk Premium",
    "Yield Curve (10Y–2Y)", "Insider Buy/Sell", "AAII Bullish",
    "BB Lower", "SPX SMA200",
]

# ══════════════════════════════════════════════════════════════════════════════
# 2. DATA LOADING  (cached so it only runs once per session)
# ══════════════════════════════════════════════════════════════════════════════
def p(name):
    """Build a path relative to DATA_DIR."""
    import os
    return os.path.join(DATA_DIR, name)


@st.cache_data(show_spinner="Loading data files…")
def load_all_data():
    def load_xlsx(filename):
        df = pd.read_excel(p(filename), skiprows=4, header=0, usecols=[0, 1])
        df.columns = ["Date", "Value"]
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        return df.dropna(subset=["Date", "Value"]).reset_index(drop=True)

    # CSVs
    yield_curve      = pd.read_csv(p("10Y-2Y.csv")).rename(columns={"observation_date": "Date", "T10Y2Y": "Value"})
    bbb_spread       = pd.read_csv(p("BBB-treasury_credit_spread.csv")).rename(columns={"observation_date": "Date", "BAMLC0A4CBBB": "Value"})
    buffett_csv      = pd.read_csv(p("Buffet_indicator.csv")).rename(columns={"observation_date": "Date", "DDDM01USA156NWDB": "Value"})
    erp              = pd.read_csv(p("ERP.csv")).rename(columns={"observation_date": "Date", "TENEXPCHAREARISPRE": "Value"})
    fear_greed       = pd.read_csv(p("fear_greed_indicator.csv")).rename(columns={"Fear Greed": "Value"})
    financial_stress = pd.read_csv(p("financial_stress_index.csv")).rename(columns={"observation_date": "Date", "STLFSI4": "Value"})
    misery           = pd.read_csv(p("Misery_index_urban_USA.csv")).rename(columns={"observation_date": "Date", "UNRATE_CPIAUCSL_PC1": "Value"})
    sahm             = pd.read_csv(p("Sahm_indicator.csv")).rename(columns={"observation_date": "Date", "SAHMREALTIME": "Value"})
    vix              = pd.read_csv(p("VIX_data.csv")).rename(columns={"observation_date": "Date", "VIXCLS": "Value"})

    # OFR / margin-debt
    with open(p("margin_debt GDP.csv"), "r") as f:
        content = f.read().replace('"', '')
    ofr_raw = pd.read_csv(StringIO(content))
    ofr_raw["Date"] = pd.to_datetime(ofr_raw["Date"], errors="coerce")
    ofr_raw = ofr_raw.dropna(subset=["Date"])
    ofr_fsi          = ofr_raw[["Date", "OFR FSI"]].rename(columns={"OFR FSI": "Value"})
    equity_valuation = ofr_raw[["Date", "Equity valuation"]].rename(columns={"Equity valuation": "Value"})

    # SKEW
    skew = pd.read_csv(p("CBOE_SKEW_historical.csv"), usecols=["Date", "Close"]).rename(columns={"Close": "Value"})
    skew["Date"] = pd.to_datetime(skew["Date"], errors="coerce")
    skew = skew.dropna()

    # Shiller
    shiller_raw = pd.read_excel(p("shiller_data.xls"), sheet_name="Data", skiprows=6).iloc[1:]
    shiller_raw = shiller_raw.rename(columns={"Unnamed: 0": "Date", "P/E10 or": "Value"})
    shiller_ratio = shiller_raw[["Date", "Value"]].dropna().copy()
    shiller_ratio["Date"] = shiller_ratio["Date"].astype(float)
    shiller_ratio["Year"] = shiller_ratio["Date"].astype(int)
    shiller_ratio["Month"] = ((shiller_ratio["Date"] % 1) * 12 + 1).round().astype(int).clip(1, 12)
    shiller_ratio["Date"] = pd.to_datetime(shiller_ratio[["Year", "Month"]].assign(day=1))
    shiller_ratio = shiller_ratio[["Date", "Value"]].dropna()

    # Sentiment
    sentiment = pd.read_excel(p("sentiment.xls"), skiprows=3, header=0, usecols=[0, 1, 2, 3])
    sentiment.columns = ["Date", "Bullish", "Neutral", "Bearish"]
    sentiment["Date"] = pd.to_datetime(sentiment["Date"], errors="coerce")
    sentiment = sentiment.dropna(subset=["Date"])

    # Brent & Gold
    brent = pd.read_excel(p("brent_price.xlsx"), skiprows=1, header=0, usecols=[0, 1])
    brent.columns = ["Date", "Value"]
    brent["Date"] = pd.to_datetime(brent["Date"], errors="coerce")
    brent = brent.dropna()

    gold = pd.read_excel(p("gold_prices.xlsx"), skiprows=1, header=0, usecols=[0, 1])
    gold.columns = ["Date", "Value"]
    gold["Date"] = pd.to_datetime(gold["Date"], errors="coerce")
    gold = gold.dropna()

    # Standard xlsx files
    tobin_q        = load_xlsx("Tobin Q 2026-02-13 19_56_32.xlsx")
    pe_ratio       = load_xlsx("S&P 500 PE Ratio 2026-02-13 20_01_50.xlsx")
    pb_ratio       = load_xlsx("S&P 500 Price to Book Value 2026-02-13 20_01_32.xlsx")
    ps_ratio       = load_xlsx("S&P 500 Price to Sales 2026-02-13 20_01_40.xlsx")
    earnings_yield = load_xlsx("S&P 500 Earnings Yield 2026-02-13 20_02_48.xlsx")
    dividend_yield = load_xlsx("S&P 500 Dividend Yield 2026-02-13 20_17_09.xlsx")
    corp_margin    = load_xlsx("Corporate Profit Margin (After Tax)  2026-02-13 19_59_45.xlsx")
    mktcap_gdp     = load_xlsx("USA Ratio of Total Market Cap over GDP 2026-02-13 19_58_12.xlsx")
    insider_ratio  = load_xlsx("Insider Buy_Sell Ratio - USA Overall Market 2026-02-13 20_10_59.xlsx")
    unemployment   = load_xlsx("Civilian Unemployment Rate 2026-02-13 20_08_59.xlsx")
    gdp_recession  = load_xlsx("GDP-Based Recession Indicator Index 2026-02-13 20_04_13.xlsx")

    # Technicals
    technicals = pd.read_csv(p("spx_technicals_corrected.csv"), skiprows=2)
    technicals.columns = ["Date", "Price", "SMA50", "SMA200", "Golden_Cross", "Death_Cross",
                          "RSI_14", "MACD", "MACD_Signal", "BB_Upper", "BB_Lower"]
    technicals["Date"] = pd.to_datetime(technicals["Date"], errors="coerce")
    technicals = technicals.dropna(subset=["Date"]).reset_index(drop=True)
    for col in ["Price", "SMA50", "SMA200", "RSI_14", "MACD", "MACD_Signal", "BB_Upper", "BB_Lower"]:
        technicals[col] = pd.to_numeric(technicals[col], errors="coerce")
    technicals["Golden_Cross"] = technicals["Golden_Cross"].astype(str).str.strip().str.lower() == "true"
    technicals["Death_Cross"]  = technicals["Death_Cross"].astype(str).str.strip().str.lower() == "true"

    # Gold-Brent ratio
    gold_brent = pd.merge(gold, brent, on="Date", how="inner").iloc[1:]
    gold_brent["gold_brent_ratio"] = gold_brent.Value_x / gold_brent.Value_y

    return dict(
        yield_curve=yield_curve, bbb_spread=bbb_spread, buffett_csv=buffett_csv,
        erp=erp, fear_greed=fear_greed, financial_stress=financial_stress,
        misery=misery, sahm=sahm, vix=vix,
        ofr_fsi=ofr_fsi, equity_valuation=equity_valuation,
        skew=skew, shiller_ratio=shiller_ratio, sentiment=sentiment,
        brent=brent, gold=gold, tobin_q=tobin_q, pe_ratio=pe_ratio,
        pb_ratio=pb_ratio, ps_ratio=ps_ratio, earnings_yield=earnings_yield,
        dividend_yield=dividend_yield, corp_margin=corp_margin,
        mktcap_gdp=mktcap_gdp, insider_ratio=insider_ratio,
        unemployment=unemployment, gdp_recession=gdp_recession,
        technicals=technicals, gold_brent=gold_brent,
    )


def build_indicators(d):
    """Build the indicators dict from loaded data."""
    def make_indicator(df, value_col, indicator_name):
        out = df.copy()
        out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
        out[value_col] = pd.to_numeric(out[value_col], errors="coerce")
        out = out.replace([np.inf, -np.inf], np.nan)
        out = out.dropna(subset=["Date", value_col]).sort_values("Date")
        out = out.drop_duplicates(subset=["Date"], keep="last")
        out = out.rename(columns={value_col: "value"})
        out["inverted"] = indicator_name in INVERSE_INDICATORS
        return out[["Date", "value", "inverted"]]

    sentiment = d["sentiment"]
    gold_brent = d["gold_brent"]
    technicals = d["technicals"]

    configs = {
        "Shiller CAPE":           (d["shiller_ratio"],         "Value"),
        "PE Ratio":               (d["pe_ratio"],              "Value"),
        "PB Ratio":               (d["pb_ratio"],              "Value"),
        "PS Ratio":               (d["ps_ratio"],              "Value"),
        "Earnings Yield":         (d["earnings_yield"],        "Value"),
        "Dividend Yield":         (d["dividend_yield"],        "Value"),
        "Tobin Q":                (d["tobin_q"],               "Value"),
        "Buffet Indicator":       (d["mktcap_gdp"],            "Value"),
        "Corp Profit Margin":     (d["corp_margin"],           "Value"),
        "Equity Risk Premium":    (d["erp"],                   "Value"),
        "Equity Valuation (OFR)":(d["equity_valuation"],      "Value"),
        "Yield Curve (10Y–2Y)":   (d["yield_curve"],           "Value"),
        "BBB–Treasury Spread":    (d["bbb_spread"],            "Value"),
        "Financial Stress (StL)": (d["financial_stress"],      "Value"),
        "OFR Financial Stress":   (d["ofr_fsi"],               "Value"),
        "VIX":                    (d["vix"],                   "Value"),
        "CBOE SKEW":              (d["skew"],                  "Value"),
        "Fear & Greed":           (d["fear_greed"],            "Value"),
        "Insider Buy/Sell":       (d["insider_ratio"],         "Value"),
        "AAII Bullish":           (sentiment.rename(columns={"Bullish": "Value"}), "Value"),
        "AAII Bearish":           (sentiment.rename(columns={"Bearish": "Value"}), "Value"),
        "AAII Bull-Bear Spread":  (sentiment.assign(Value=sentiment["Bullish"] - sentiment["Bearish"]), "Value"),
        "Sahm Rule":              (d["sahm"],                  "Value"),
        "Misery Index":           (d["misery"],                "Value"),
        "GDP Recession Indicator":(d["gdp_recession"],         "Value"),
        "Gold / Brent Ratio":     (gold_brent,                 "gold_brent_ratio"),
        "Gold":                   (d["gold"],                  "Value"),
        "Brent Oil":              (d["brent"],                 "Value"),
        "SPX SMA50":              (technicals,                 "SMA50"),
        "SPX SMA200":             (technicals,                 "SMA200"),
        "RSI (14)":               (technicals,                 "RSI_14"),
        "MACD":                   (technicals,                 "MACD"),
        "BB Upper":               (technicals,                 "BB_Upper"),
        "BB Lower":               (technicals,                 "BB_Lower"),
    }
    return {name: make_indicator(df, col, name) for name, (df, col) in configs.items()}


# ══════════════════════════════════════════════════════════════════════════════
# 3. DATA UTILS
# ══════════════════════════════════════════════════════════════════════════════
def ensure_date_col(df):
    temp = df.copy()
    if "Date" not in temp.columns:
        if isinstance(temp.index, pd.DatetimeIndex):
            temp = temp.reset_index().rename(columns={temp.index.name or "index": "Date"})
        else:
            temp = temp.reset_index().rename(columns={temp.columns[0]: "Date"})
    temp["Date"] = pd.to_datetime(temp["Date"], errors="coerce", utc=True).dt.tz_localize(None)
    return temp.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)


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


def get_clean_indicator(indicators, name):
    raw = ensure_date_col(indicators[name].copy())
    value_col = find_value_column(raw)
    if value_col is None:
        return pd.DataFrame(columns=["Date", name])
    raw[value_col] = pd.to_numeric(raw[value_col], errors="coerce")
    raw = (raw.dropna(subset=[value_col])[["Date", value_col]]
              .rename(columns={value_col: name}))
    raw["Date"] = pd.to_datetime(raw["Date"], errors="coerce", utc=True).dt.tz_localize(None)
    return raw.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)


def build_spx_forward(technicals, horizon_years):
    H = int(round(float(horizon_years) * 252))
    spx = ensure_date_col(technicals[["Date", "Price"]].dropna().copy())
    spx = spx.rename(columns={"Price": "close"})
    spx["fwd"] = spx["close"].shift(-H) / spx["close"] - 1
    return spx.dropna(subset=["fwd"]).copy()


def get_spx_price_series(technicals):
    spx = ensure_date_col(technicals[["Date", "Price"]].dropna().copy())
    spx = spx.rename(columns={"Price": "close"})
    spx["Date"] = pd.to_datetime(spx["Date"], errors="coerce", utc=True).dt.tz_localize(None)
    return spx.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)


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


def identify_spx_downturns(technicals, drawdown_threshold=0.20,
                            recovery_threshold=0.05, min_spacing_days=126):
    spx = get_spx_price_series(technicals).copy()
    spx["peak"] = spx["close"].cummax()
    spx["drawdown"] = spx["close"] / spx["peak"] - 1
    events = []
    i, last_start = 1, None
    while i < len(spx):
        breached = (spx["drawdown"].iloc[i] <= -drawdown_threshold and
                    spx["drawdown"].iloc[i - 1] > -drawdown_threshold)
        spacing_ok = (last_start is None or
                      (spx["Date"].iloc[i] - last_start).days >= min_spacing_days)
        if breached and spacing_ok:
            start_date = spx["Date"].iloc[i]
            j = i
            while j < len(spx) - 1 and spx["drawdown"].iloc[j] <= -recovery_threshold:
                j += 1
            event_slice = spx.iloc[i:j + 1].copy()
            trough_row = spx.loc[event_slice["drawdown"].idxmin()]
            events.append({
                "start_date": start_date,
                "end_date": spx["Date"].iloc[j],
                "trough_date": trough_row["Date"],
                "start_price": spx["close"].iloc[i],
                "trough_price": trough_row["close"],
                "end_price": spx["close"].iloc[j],
                "max_drawdown": trough_row["drawdown"],
                "days_to_trough": int((trough_row["Date"] - start_date).days),
                "event_days": int((spx["Date"].iloc[j] - start_date).days),
            })
            last_start = start_date
            i = j + 1
        else:
            i += 1
    return spx, pd.DataFrame(events)


def percentile_as_of(raw_df, value_col, anchor_date, value):
    hist = raw_df.loc[raw_df["Date"] <= anchor_date, value_col].dropna()
    if len(hist) < 10 or pd.isna(value):
        return np.nan
    return (hist <= value).mean()


def lookup_indicator_value_at(indicators, name, anchor_date):
    raw = get_clean_indicator(indicators, name)
    if raw.empty:
        return np.nan, pd.NaT, raw, name
    match = pd.merge_asof(
        pd.DataFrame({"Date": [pd.to_datetime(anchor_date, utc=True).tz_localize(None)]}),
        raw[["Date", name]], on="Date", direction="backward"
    )
    val = match[name].iloc[0] if name in match.columns else np.nan
    actual_date = match["Date"].iloc[0]
    return val, actual_date, raw, name


def auc_from_scores(y_true, scores):
    y = pd.Series(y_true).astype(int)
    s = pd.Series(scores).astype(float)
    valid = y.notna() & s.notna()
    y, s = y[valid], s[valid]
    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return np.nan
    ranks = s.rank(method="average")
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def compute_top_bucket_metrics(y_true, scores, top_frac=0.10):
    y = pd.Series(y_true).astype(int)
    s = pd.Series(scores).astype(float)
    valid = y.notna() & s.notna()
    y, s = y[valid], s[valid]
    if len(y) < 20:
        return {k: np.nan for k in ["precision", "recall", "f1", "lift", "flag_rate"]}
    cutoff = s.quantile(1 - top_frac)
    pred = (s >= cutoff).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    precision = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    recall    = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    f1 = (2 * precision * recall / (precision + recall)
          if pd.notna(precision) and pd.notna(recall) and (precision + recall) > 0 else np.nan)
    base_rate = y.mean()
    lift = precision / base_rate if pd.notna(precision) and base_rate > 0 else np.nan
    return {"precision": precision, "recall": recall, "f1": f1, "lift": lift, "flag_rate": pred.mean()}


def apply_layout(fig, title="", height=450):
    fig.update_layout(
        template="plotly_white",
        title=dict(text=title, x=0, font=dict(size=15, color=TEXT)),
        height=height,
        margin=dict(l=52, r=20, t=56, b=42),
        paper_bgcolor=BG, plot_bgcolor=BG2,
        hovermode="closest",
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1, bgcolor="rgba(255,255,255,0.8)"),
    )
    fig.update_xaxes(showgrid=True, gridcolor=BORDER, zeroline=False, linecolor=BORDER)
    fig.update_yaxes(showgrid=True, gridcolor=BORDER, zeroline=False, linecolor=BORDER)


# ══════════════════════════════════════════════════════════════════════════════
# 4. TABS
# ══════════════════════════════════════════════════════════════════════════════

def tab_indicator(indicators, technicals):
    st.subheader("Indicator Analysis")
    all_names = sorted(indicators.keys())

    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        label = st.selectbox("Indicator", all_names,
                             index=all_names.index("Shiller CAPE") if "Shiller CAPE" in all_names else 0)
    with col2:
        h_val = st.slider("Horizon (years)", 0.25, 10.0, 1.0, 0.25)
    with col3:
        rolling_window = st.slider("Rolling window (years)", 3, 30, 10, 1)

    show_ts = st.checkbox("Show time series", value=True)

    df_raw = get_clean_indicator(indicators, label)
    spx = build_spx_forward(technicals, h_val)
    ind = df_raw[["Date", label]].rename(columns={label: "val"})
    ds = pd.merge_asof(spx, ind, on="Date", direction="backward").dropna(subset=["val", "fwd"])

    if len(ds) < 10:
        st.error("Insufficient overlap between indicator and SPX data.")
        return

    corr = ds["val"].corr(ds["fwd"])
    r_sq = corr ** 2
    b1, b0 = np.polyfit(ds["val"], ds["fwd"], 1)
    latest_val  = df_raw[label].dropna().iloc[-1]
    latest_date = df_raw["Date"].dropna().iloc[-1]
    current_pct = (df_raw[label].dropna() <= latest_val).mean()
    pct_color   = RED if current_pct >= 0.85 else (TEAL if current_pct <= 0.15 else ORANGE)

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Observations",    f"{len(ds):,}")
    c2.metric("Horizon",         f"{h_val:g}Y")
    c3.metric("Correlation",     f"{corr:.3f}")
    c4.metric("R²",              f"{r_sq:.3f}")
    c5.metric("Slope",           f"{b1*100:.3f}%")
    c6.metric("Current value",   f"{latest_val:.2f}")

    st.markdown(f"""
    **Historical percentile:** `{current_pct:.0%}`  
    As of `{latest_date.strftime('%Y-%m-%d')}` — data since `{df_raw['Date'].min().strftime('%Y')}`
    """)
    st.progress(float(current_pct), text=f"{current_pct:.0%}ile — {'⚠ Expensive' if current_pct >= 0.85 else ('✓ Cheap' if current_pct <= 0.15 else '● Neutral')}")

    if show_ts:
        fig_ts = go.Figure()
        fig_ts.add_trace(go.Scatter(
            x=df_raw["Date"], y=df_raw[label], mode="lines", name=label,
            line=dict(color=BLUE, width=2), fill="tozeroy", fillcolor="rgba(37,99,235,0.07)"
        ))
        fig_ts.add_hline(y=latest_val, line_dash="dot", line_color=pct_color, line_width=1.5,
                         annotation_text=f"Current: {latest_val:.2f} ({current_pct:.0%}ile)",
                         annotation_position="top right", annotation_font=dict(color=pct_color, size=11))
        apply_layout(fig_ts, f"Historical path — {label}", 280)
        st.plotly_chart(fig_ts, use_container_width=True)

    x_range = np.linspace(ds["val"].min(), ds["val"].max(), 200)
    y_fit   = b1 * x_range + b0
    residuals = ds["fwd"] - (b1 * ds["val"] + b0)
    std_err = residuals.std()

    fig_reg = go.Figure()
    fig_reg.add_trace(go.Scatter(
        x=ds["val"], y=ds["fwd"], mode="markers", name="Observations",
        text=ds["Date"].dt.strftime("%Y-%m-%d"),
        hovertemplate=f"<b>{label}</b>: %{{x:.2f}}<br><b>Fwd return</b>: %{{y:.1%}}<br><b>Date</b>: %{{text}}<extra></extra>",
        marker=dict(color=ds["fwd"], colorscale="RdYlGn",
                    cmin=ds["fwd"].quantile(0.05), cmax=ds["fwd"].quantile(0.95),
                    size=5, opacity=0.65,
                    colorbar=dict(title=dict(text="Fwd return", side="right"), tickformat=".0%", thickness=12, len=0.6))
    ))
    fig_reg.add_trace(go.Scatter(x=x_range, y=y_fit, mode="lines", name="OLS fit",
                                  line=dict(color=TEXT, width=2.5), hoverinfo="skip"))
    fig_reg.add_trace(go.Scatter(
        x=np.concatenate([x_range, x_range[::-1]]),
        y=np.concatenate([y_fit + std_err, (y_fit - std_err)[::-1]]),
        fill="toself", fillcolor="rgba(15,23,42,0.06)", line=dict(width=0),
        hoverinfo="skip", showlegend=False
    ))
    fig_reg.add_vline(x=latest_val, line_dash="dot", line_color=pct_color, line_width=1.5,
                       annotation_text=f"Today: {latest_val:.2f}",
                       annotation_font=dict(color=pct_color, size=11))
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
        fig_rc.add_trace(go.Scatter(x=ds_sorted["Date"], y=ds_sorted["rolling_corr"],
                                     mode="lines", name=f"{rolling_window}Y rolling corr",
                                     line=dict(color=PURPLE, width=2), fill="tozeroy",
                                     fillcolor="rgba(124,58,237,0.07)"))
        fig_rc.add_hline(y=corr, line_dash="dot", line_color=SLATE, line_width=1,
                          annotation_text=f"Full-sample: {corr:.3f}", annotation_position="top right",
                          annotation_font=dict(color=MUTED, size=11))
        fig_rc.add_hline(y=0, line_color=BORDER, line_width=1)
        apply_layout(fig_rc, f"Rolling {rolling_window}Y correlation — {label} vs {h_val:g}Y fwd return", 300)
        fig_rc.update_yaxes(title="Correlation", range=[-1.05, 1.05])
        st.plotly_chart(fig_rc, use_container_width=True)


def tab_technicals(technicals):
    st.subheader("SPX Technicals")
    tech_dates = sorted(technicals["Date"].dropna().unique())
    min_d, max_d = pd.Timestamp(tech_dates[0]), pd.Timestamp(tech_dates[-1])
    default_start = max(min_d, max_d - pd.Timedelta(days=1260))
    d_s, d_e = st.slider("Date range", min_value=min_d.to_pydatetime(),
                          max_value=max_d.to_pydatetime(),
                          value=(default_start.to_pydatetime(), max_d.to_pydatetime()),
                          format="YYYY-MM-DD")
    t = technicals[(technicals["Date"] >= d_s) & (technicals["Date"] <= d_e)].copy()
    if t.empty:
        st.error("No data in selected range.")
        return
    gc, dc = t[t["Golden_Cross"]], t[t["Death_Cross"]]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Current SPX",    f"{t['Price'].iloc[-1]:,.0f}")
    c2.metric("Golden Crosses", len(gc))
    c3.metric("Death Crosses",  len(dc))
    c4.metric("Period",         f"{d_s.year} – {d_e.year}")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t["Date"], y=t["Price"], name="Price", line=dict(color=TEXT, width=1.8)))
    fig.add_trace(go.Scatter(x=t["Date"], y=t["SMA50"], name="SMA50", line=dict(color=BLUE, dash="dot", width=1.4)))
    fig.add_trace(go.Scatter(x=t["Date"], y=t["SMA200"], name="SMA200", line=dict(color=RED, dash="dot", width=1.4)))
    if not gc.empty:
        fig.add_trace(go.Scatter(x=gc["Date"], y=gc["Price"], mode="markers", name="Golden Cross",
                                  marker=dict(symbol="triangle-up", size=12, color=GREEN)))
    if not dc.empty:
        fig.add_trace(go.Scatter(x=dc["Date"], y=dc["Price"], mode="markers", name="Death Cross",
                                  marker=dict(symbol="triangle-down", size=12, color=RED)))
    apply_layout(fig, "SPX Levels & Moving Average Signals", 550)
    st.plotly_chart(fig, use_container_width=True)


def tab_combo(indicators, technicals):
    st.subheader("Tail Combo Analysis")
    all_names = sorted(indicators.keys())
    with st.expander("ℹ️ How to use this tab", expanded=False):
        st.markdown("""
        - **Right tail cutoff** – Top percentile signalling overvaluation / froth (e.g. 0.85 = top 15%)
        - **Left tail cutoff** – Bottom percentile signalling undervaluation (e.g. 0.15 = bottom 15%)
        - **Rule** – How indicators are combined: Any, At Least N, All, or Composite Score
        - **Horizon (Y)** – How far ahead SPX returns are measured
        - **Regime filter** – Restrict to bull or bear market conditions (SPX vs SMA200)
        """)

    col1, col2 = st.columns([3, 2])
    with col1:
        selected = st.multiselect("Select indicators", all_names,
                                   default=["Shiller CAPE", "Tobin Q"] if "Shiller CAPE" in all_names else [all_names[0]])
    with col2:
        q_right = st.slider("Right tail cutoff", 0.70, 0.99, 0.85, 0.01)
        q_left  = st.slider("Left tail cutoff",  0.01, 0.30, 0.15, 0.01)

    col3, col4, col5 = st.columns(3)
    with col3:
        rule = st.selectbox("Rule", ["All in Tail", "Any in Tail", "At Least N", "Composite Score"])
        rule_key = {"All in Tail": "all", "Any in Tail": "any",
                    "At Least N": "at_least_n", "Composite Score": "composite"}[rule]
    with col4:
        n_val   = st.slider("Min N (for At Least N)", 1, max(1, len(selected)), min(2, max(1, len(selected))))
        horizon = st.slider("Horizon (years)", 0.25, 10.0, 1.0, 0.25, key="combo_horizon")
    with col5:
        regime  = st.selectbox("Regime filter", ["All regimes", "Bull only (price > SMA200)", "Bear only (price < SMA200)"])
        regime_key = {"All regimes": "all", "Bull only (price > SMA200)": "bull", "Bear only (price < SMA200)": "bear"}[regime]

    if not selected:
        st.warning("Select at least one indicator.")
        return

    spx = build_spx_forward(technicals, horizon)
    merged = spx[["Date", "fwd"]].copy()
    for lbl in selected:
        ind = get_clean_indicator(indicators, lbl)
        merged = pd.merge_asof(merged, ind, on="Date", direction="backward")
    merged = merged.dropna(subset=selected + ["fwd"]).copy()

    if len(merged) < 20:
        st.error(f"Not enough aligned observations at {horizon:.2f}Y.")
        return

    # Regime filter
    if regime_key != "all":
        tech = technicals[["Date", "Price", "SMA200"]].dropna().copy()
        tech["Date"] = pd.to_datetime(tech["Date"], errors="coerce", utc=True).dt.tz_localize(None)
        tech["bull"] = tech["Price"] > tech["SMA200"]
        aligned = pd.merge_asof(merged[["Date"]].reset_index(),
                                tech[["Date", "bull"]], on="Date", direction="backward")
        bull_mask = aligned.set_index("index")["bull"].fillna(True)
        merged = merged[bull_mask if regime_key == "bull" else ~bull_mask].copy()

    if len(merged) < 20:
        st.error("Not enough observations after applying regime filter.")
        return

    # Current positioning
    st.markdown("#### Current positioning — where are we today?")
    pos_cols = st.columns(len(selected))
    pos_data = []
    for i, lbl in enumerate(selected):
        raw = get_clean_indicator(indicators, lbl)
        latest_val = raw[lbl].iloc[-1]
        latest_date = raw["Date"].iloc[-1]
        current_pct = (raw[lbl] <= latest_val).mean()
        tag = "⚠ Right tail" if current_pct >= q_right else ("✓ Left tail" if current_pct <= q_left else "● Neutral")
        pos_data.append({"label": lbl, "val": latest_val, "pct": current_pct, "tag": tag, "date": latest_date})
        with pos_cols[i]:
            delta_str = f"{current_pct:.0%}ile — {tag}"
            st.metric(lbl, f"{latest_val:.2f}", delta_str)
    st.markdown("---")

    # Tail logic
    right_thresh = {lbl: merged[lbl].quantile(q_right) for lbl in selected}
    left_thresh  = {lbl: merged[lbl].quantile(q_left)  for lbl in selected}
    right_flags  = pd.DataFrame({lbl: merged[lbl] >= right_thresh[lbl] for lbl in selected})
    left_flags   = pd.DataFrame({lbl: merged[lbl] <= left_thresh[lbl]  for lbl in selected})
    pct_scores   = pd.DataFrame({lbl: merged[lbl].rank(pct=True) for lbl in selected})
    composite_right = pct_scores.mean(axis=1)
    composite_left  = 1 - composite_right

    def apply_rule(flags, composite, q):
        cnt = flags.sum(axis=1)
        if rule_key == "all":    return cnt == len(selected)
        elif rule_key == "any":  return cnt >= 1
        elif rule_key == "at_least_n": return cnt >= n_val
        else:                    return composite >= q

    right_mask = apply_rule(right_flags, composite_right, q_right)
    left_mask  = apply_rule(left_flags,  composite_left,  q_right)
    merged["is_right_tail"] = right_mask
    merged["is_left_tail"]  = left_mask
    right_ev = merged[merged["is_right_tail"]]
    left_ev  = merged[merged["is_left_tail"]]

    base_avg  = merged["fwd"].mean()
    right_avg = right_ev["fwd"].mean() if len(right_ev) else np.nan
    left_avg  = left_ev["fwd"].mean()  if len(left_ev)  else np.nan
    base_var, base_cvar   = calc_var_cvar(merged["fwd"])
    right_var, right_cvar = calc_var_cvar(right_ev["fwd"]) if len(right_ev) else (np.nan, np.nan)
    left_var,  left_cvar  = calc_var_cvar(left_ev["fwd"])  if len(left_ev)  else (np.nan, np.nan)

    st.markdown(f"#### Right tail — overvaluation / froth · {regime}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Right tail events",    f"{len(right_ev):,}", f"{fmt_pct(len(right_ev)/len(merged))} of dates")
    c2.metric("Avg fwd return",       fmt_pct(right_avg),  f"Base: {fmt_pct(base_avg)}")
    c3.metric("95% VaR",              fmt_pct(right_var),  f"Base: {fmt_pct(base_var)}")
    c4.metric("95% CVaR",             fmt_pct(right_cvar), f"Base: {fmt_pct(base_cvar)}")

    st.markdown(f"#### Left tail — undervaluation / cheap · {regime}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Left tail events",  f"{len(left_ev):,}",  f"{fmt_pct(len(left_ev)/len(merged))} of dates")
    c2.metric("Avg fwd return",    fmt_pct(left_avg),   f"Base: {fmt_pct(base_avg)}")
    c3.metric("95% VaR",           fmt_pct(left_var),   f"Base: {fmt_pct(base_var)}")
    c4.metric("95% CVaR",          fmt_pct(left_cvar),  f"Base: {fmt_pct(base_cvar)}")


def tab_downturns(indicators, technicals):
    st.subheader("SPX Downturn Lookback")
    all_names = sorted(indicators.keys())
    with st.expander("ℹ️ How to use this tab", expanded=False):
        st.markdown("""
        - **Downturn trigger** – drawdown level that starts an event
        - **Recovery level** – drawdown level that ends an event
        - **Min spacing** – prevents counting the same bear market multiple times
        """)

    selected = st.multiselect("Select indicators", all_names,
                               default=["Shiller CAPE", "Tobin Q"] if "Shiller CAPE" in all_names else [],
                               key="recess_sel")
    col1, col2, col3 = st.columns(3)
    dd_thresh = col1.slider("Downturn trigger", 0.10, 0.40, 0.20, 0.01)
    rec_thresh = col2.slider("Recovery level", 0.00, 0.20, 0.05, 0.01)
    min_spacing = col3.slider("Min spacing (days)", 21, 504, 126, 21)

    if not selected:
        st.warning("Select at least one indicator.")
        return

    spx, events = identify_spx_downturns(technicals, dd_thresh, rec_thresh, min_spacing)
    if events.empty:
        st.error("No downturn events found with the current settings.")
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Detected downturns",  f"{len(events):,}")
    c2.metric("Avg max drawdown",    fmt_pct(events["max_drawdown"].mean()))
    c3.metric("Avg days to trough",  f"{events['days_to_trough'].mean():.0f}")
    c4.metric("Avg event length",    f"{events['event_days'].mean():.0f} days")

    fig_dd = go.Figure()
    fig_dd.add_trace(go.Scatter(x=spx["Date"], y=spx["close"], mode="lines", name="SPX",
                                 line=dict(color=TEXT, width=1.8)))
    for _, row in events.iterrows():
        fig_dd.add_vrect(x0=row["start_date"], x1=row["end_date"],
                          fillcolor="rgba(220,38,38,0.10)", line_width=0, layer="below")
        fig_dd.add_trace(go.Scatter(x=[row["trough_date"]], y=[row["trough_price"]],
                                     mode="markers", marker=dict(size=10, color=RED, symbol="triangle-down"),
                                     showlegend=False,
                                     hovertemplate=f"Trough: {row['trough_date'].strftime('%Y-%m-%d')}<br>Max DD: {row['max_drawdown']:.1%}<extra></extra>"))
    apply_layout(fig_dd, "SPX with detected downturn episodes", 420)
    st.plotly_chart(fig_dd, use_container_width=True)

    # Heatmap: avg indicator percentile before each downturn
    lookbacks = [("3M", int(0.25*252)), ("6M", int(0.50*252)), ("1Y", int(1*252)),
                 ("2Y", int(2*252)), ("3Y", int(3*252)), ("4Y", int(4*252)), ("5Y", int(5*252))]
    rows = []
    for _, ev in events.iterrows():
        start_date = pd.to_datetime(ev["start_date"])
        for lb_name, lb_days in lookbacks:
            anchor = start_date - pd.Timedelta(days=int(lb_days * 365 / 252))
            rec = {"event_start": start_date, "lookback": lb_name, "max_drawdown": ev["max_drawdown"]}
            for lbl in selected:
                val, _, raw, value_col = lookup_indicator_value_at(indicators, lbl, anchor)
                rec[f"{lbl}__pct"] = percentile_as_of(raw, value_col, anchor, val)
            rows.append(rec)
    lookback_df = pd.DataFrame(rows)

    avg_pct = pd.DataFrame(index=selected, columns=[x[0] for x in lookbacks], dtype=float)
    for lbl in selected:
        for lb_name, _ in lookbacks:
            mask = lookback_df["lookback"] == lb_name
            avg_pct.loc[lbl, lb_name] = lookback_df.loc[mask, f"{lbl}__pct"].mean()

    heat = go.Figure(data=go.Heatmap(
        z=avg_pct.values, x=avg_pct.columns.tolist(), y=avg_pct.index.tolist(),
        colorscale="RdYlGn_r", zmin=0, zmax=1,
        text=[[f"{v:.0%}" if pd.notna(v) else "—" for v in row] for row in avg_pct.values],
        texttemplate="%{text}",
        colorbar=dict(title="Percentile", tickformat=".0%")
    ))
    apply_layout(heat, "Average indicator percentile before downturn start",
                 120 + 42 * max(4, len(selected)))
    st.plotly_chart(heat, use_container_width=True)


def tab_predictor(indicators, technicals):
    st.subheader("Downturn Predictor Search")
    all_names = sorted(indicators.keys())
    with st.expander("ℹ️ How to use this tab", expanded=False):
        st.markdown("""
        - **Predict window** – a date is labelled positive if a downturn begins within this future window
        - **F1 / Precision / Recall** – evaluated at the top 10% and top 5% signal dates
        - **AUC** – measures how well the signal separates pre-downturn from normal dates
        """)

    selected = st.multiselect("Select indicators", all_names,
                               default=["Shiller CAPE", "Tobin Q"] if "Shiller CAPE" in all_names else [],
                               key="pred_sel")
    col1, col2, col3, col4 = st.columns(4)
    dd_thresh   = col1.slider("Downturn trigger",    0.10, 0.40, 0.20, 0.01, key="pred_dd")
    rec_thresh  = col2.slider("Recovery level",      0.00, 0.20, 0.05, 0.01, key="pred_rec")
    min_spacing = col3.slider("Min spacing (days)",  21, 504, 126, 21, key="pred_sp")
    horizon     = col4.slider("Predict window (Y)",  0.25, 3.0, 1.0, 0.25,  key="pred_hor")

    if not selected:
        st.warning("Select at least one indicator.")
        return

    _, events = identify_spx_downturns(technicals, dd_thresh, rec_thresh, min_spacing)
    if events.empty:
        st.error("No downturn events found.")
        return

    # Build dataset
    spx = get_spx_price_series(technicals)[["Date"]].copy()
    result_panel = spx.copy()
    for lbl in selected:
        ind = get_clean_indicator(indicators, lbl)
        result_panel = pd.merge_asof(result_panel, ind, on="Date", direction="backward")
    usable = [c for c in result_panel.columns if c != "Date"]
    result_panel = result_panel.dropna(subset=usable, how="all").copy()

    if result_panel.empty:
        st.error("No overlapping data.")
        return

    horizon_days = int(round(horizon * 365))
    event_starts = np.sort(pd.to_datetime(events["start_date"]).values.astype("datetime64[ns]"))
    dates = pd.to_datetime(result_panel["Date"]).values.astype("datetime64[ns]")
    future_limits = dates + np.timedelta64(horizon_days, "D")
    left_idx  = np.searchsorted(event_starts, dates, side="left")
    right_idx = np.searchsorted(event_starts, future_limits, side="right")
    result_panel["target_downturn"] = (right_idx > left_idx).astype(int)

    for lbl in selected:
        if lbl in result_panel.columns:
            is_inv = bool(indicators[lbl]["inverted"].iloc[0]) if "inverted" in indicators[lbl].columns else False
            result_panel[f"{lbl}__pct"] = expanding_percentile_series(result_panel[lbl], invert=is_inv)

    X_cols = [f"{lbl}__pct" for lbl in selected if f"{lbl}__pct" in result_panel.columns]
    if not X_cols:
        st.error("Could not compute percentile scores.")
        return

    X = result_panel[X_cols].to_numpy(dtype=float)
    y = result_panel["target_downturn"].to_numpy(dtype=int)
    label_to_idx = {lbl: i for i, lbl in enumerate(selected) if f"{lbl}__pct" in result_panel.columns}

    results = []
    max_size = min(4, len(selected))
    for k in range(1, max_size + 1):
        for combo in combinations(selected, k):
            idxs = [label_to_idx[lbl] for lbl in combo if lbl in label_to_idx]
            if len(idxs) != len(combo):
                continue
            subX = X[:, idxs]
            valid = ~np.isnan(subX).any(axis=1)
            if valid.sum() < 50:
                continue
            signal = subX[valid].mean(axis=1)
            yv = y[valid]
            auc = auc_from_scores(yv, signal)
            m10 = compute_top_bucket_metrics(yv, signal, 0.10)
            m05 = compute_top_bucket_metrics(yv, signal, 0.05)
            results.append({
                "Combo": " + ".join(combo),
                "Size": k,
                "AUC": round(auc, 3) if pd.notna(auc) else None,
                "Prec 10%": round(m10["precision"], 3) if pd.notna(m10["precision"]) else None,
                "Recall 10%": round(m10["recall"], 3) if pd.notna(m10["recall"]) else None,
                "F1 10%": round(m10["f1"], 3) if pd.notna(m10["f1"]) else None,
                "Lift 10%": round(m10["lift"], 2) if pd.notna(m10["lift"]) else None,
                "Prec 5%": round(m05["precision"], 3) if pd.notna(m05["precision"]) else None,
                "F1 5%": round(m05["f1"], 3) if pd.notna(m05["f1"]) else None,
                "N obs": int(valid.sum()),
            })

    if not results:
        st.error("Not enough data to score combinations.")
        return

    combo_res = pd.DataFrame(results).sort_values("F1 10%", ascending=False, na_position="last")
    st.markdown("#### Top combinations by F1 at top 10% signal threshold")
    st.dataframe(combo_res.head(20), use_container_width=True)

    top10 = combo_res.dropna(subset=["F1 10%"]).head(10)
    fig_b = go.Figure()
    fig_b.add_trace(go.Bar(x=top10["F1 10%"], y=top10["Combo"], orientation="h",
                            text=[str(x) for x in top10["F1 10%"]], textposition="outside",
                            marker=dict(color=PURPLE)))
    apply_layout(fig_b, "Top combos by F1 at top 10% signals", 120 + 42 * len(top10))
    fig_b.update_xaxes(title="F1 score")
    st.plotly_chart(fig_b, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# 5. MAIN APP
# ══════════════════════════════════════════════════════════════════════════════
def main():
    # Header
    st.markdown("""
    <div style="padding:16px 20px;border-radius:16px;border:1px solid #e2e8f0;
        background:linear-gradient(135deg,#eef4ff 0%,#ffffff 100%);margin-bottom:12px;">
        <h2 style="margin:0;color:#0f172a;">📊 Market Froth Dashboard</h2>
        <p style="margin:4px 0 0;color:#64748b;font-size:14px;">
            Tail combinations · Indicator relationships · SPX downturn lookbacks · Predictor search · Technical context
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Load data
    try:
        data = load_all_data()
    except Exception as e:
        st.error(f"Failed to load data files: {e}")
        st.info(f"Make sure all data files are in: `{DATA_DIR}`")
        return

    indicators = build_indicators(data)
    technicals = data["technicals"]

    # Tabs
    tabs = st.tabs(["📈 Indicator Analysis", "🔀 Tail Combo", "📉 SPX Downturns", "🔍 Predictor Search", "⚙️ SPX Technicals"])

    with tabs[0]:
        tab_indicator(indicators, technicals)
    with tabs[1]:
        tab_combo(indicators, technicals)
    with tabs[2]:
        tab_downturns(indicators, technicals)
    with tabs[3]:
        tab_predictor(indicators, technicals)
    with tabs[4]:
        tab_technicals(technicals)


if __name__ == "__main__":
    main()
