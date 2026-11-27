import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

st.set_page_config(page_title="NSE Swing Screener", layout="wide")
st.title("📈 NSE Volume Breakout & Momentum Screener")
st.caption("Screens liquid NSE stocks for uptrend + volume surges + Nifty outperformance.")

# Liquid Nifty 50 / Next 50 sample universe (expandable to full Nifty 500)
DEFAULT_TICKERS = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "ICICIBANK.NS", "INFY.NS",
    "BHARTIARTL.NS", "SBIN.NS", "LT.NS", "ITC.NS", "HINDUNILVR.NS",
    "AXISBANK.NS", "KOTAKBANK.NS", "BAJFINANCE.NS", "MARUTI.NS", "SUNPHARMA.NS",
    "TITAN.NS", "ULTRACEMCO.NS", "NTPC.NS", "TATAMOTORS.NS", "POWERGRID.NS",
    "M&M.NS", "ADANIPORTS.NS", "COALINDIA.NS", "BAJAJFINSV.NS", "TRENT.NS",
    "BEL.NS", "HAL.NS", "SIEMENS.NS", "ABB.NS", "INDIGO.NS",
    "ZOMATO.NS", "DLF.NS", "BANKBARODA.NS", "PNB.NS", "TATASTEEL.NS",
    "JSWSTEEL.NS", "HINDALCO.NS", "VEDL.NS", "BPCL.NS", "ONGC.NS"
]

# Sidebar / Mobile-friendly controls
with st.expander("⚙️ Screener Filters (Tap to Adjust)", expanded=False):
    min_vol_ratio = st.slider("Min Daily Volume Spike (x 20-day Avg)", 1.0, 4.0, 1.3, 0.1)
    min_weekly_vol_ratio = st.slider("Min Weekly Volume Surge (x 4-wk Avg)", 0.8, 3.0, 1.1, 0.1)
    rr_ratio = st.slider("Target Risk-to-Reward Ratio", 1.5, 4.0, 2.5, 0.5)
    only_uptrend = st.checkbox("Require Price > 50 SMA & 200 SMA", value=True)
    custom_tickers = st.text_area(
        "Custom NSE Symbols (comma-separated, without .NS)",
        value=""
    )

if custom_tickers.strip():
    tickers = [f"{t.strip().upper()}.NS" for t in custom_tickers.split(",") if t.strip()]
else:
    tickers = DEFAULT_TICKERS

@st.cache_data(ttl=900)
def run_screener(symbol_list, vol_thresh, wk_vol_thresh, rr, req_uptrend):
    # Fetch Nifty 50 benchmark for Relative Strength
    nifty = yf.download("^NSEI", period="6mo", interval="1d", progress=False, auto_adjust=True)
    if isinstance(nifty.columns, pd.MultiIndex):
        nifty.columns = nifty.columns.get_level_values(0)
    nifty_1m_ret = (nifty["Close"].iloc[-1] / nifty["Close"].iloc[-21] - 1) * 100

    # Batch download stock data
    raw = yf.download(symbol_list, period="1y", interval="1d", group_by="ticker", progress=False, auto_adjust=True)
    results = []

    for sym in symbol_list:
        try:
            df = raw[sym].dropna() if len(symbol_list) > 1 else raw.dropna()
            if len(df) < 200:
                continue

            close = df["Close"]
            high = df["High"]
            low = df["Low"]
            vol = df["Volume"]

            curr_price = float(close.iloc[-1])
            sma_20 = float(close.rolling(20).mean().iloc[-1])
            sma_50 = float(close.rolling(50).mean().iloc[-1])
            sma_200 = float(close.rolling(200).mean().iloc[-1])

            if req_uptrend and not (curr_price > sma_50 > sma_200):
                continue

            # Daily volume spike vs 20-day average
            avg_vol_20 = float(vol.iloc[-21:-1].mean())
            curr_vol = float(vol.iloc[-1])
            daily_vol_ratio = curr_vol / avg_vol_20 if avg_vol_20 > 0 else 0

            # Weekly volume surge (last 5 trading days vs prior 20 trading days avg)
            curr_wk_vol = float(vol.iloc[-5:].sum())
            prev_wk_avg = float(vol.iloc[-25:-5].sum()) / 4.0
            weekly_vol_ratio = curr_wk_vol / prev_wk_avg if prev_wk_avg > 0 else 0

            if daily_vol_ratio < vol_thresh or weekly_vol_ratio < wk_vol_thresh:
                continue

            # Relative Strength vs Nifty 50 (1-Month)
            stock_1m_ret = (curr_price / float(close.iloc[-21]) - 1) * 100
            rs_vs_nifty = stock_1m_ret - nifty_1m_ret

            # Breakout check (within 3% of 20-day high)
            high_20 = float(high.iloc[-20:].max())
            dist_from_high = ((high_20 - curr_price) / high_20) * 100

            # ATR-based Stop-Loss & Target
            tr = pd.concat([
                high - low,
                (high - close.shift()).abs(),
                (low - close.shift()).abs()
            ], axis=1).max(axis=1)
            atr_14 = float(tr.rolling(14).mean().iloc[-1])

            stop_loss = round(curr_price - (1.5 * atr_14), 2)
            risk_per_share = curr_price - stop_loss
            target_price = round(curr_price + (risk_per_share * rr), 2)

            # Composite Score to rank top candidates
            score = round((daily_vol_ratio * 35) + (weekly_vol_ratio * 25) + (rs_vs_nifty * 2), 1)

            setup = "Darvas / High Breakout" if dist_from_high <= 2.5 else "Trend Momentum"

            results.append({
                "Symbol": sym.replace(".NS", ""),
                "Setup": setup,
                "Price (₹)": round(curr_price, 2),
                "Daily Vol (x)": round(daily_vol_ratio, 2),
                "Weekly Vol (x)": round(weekly_vol_ratio, 2),
                "1M vs Nifty (%)": round(rs_vs_nifty, 2),
                "Stop Loss (₹)": stop_loss,
                "Target (₹)": target_price,
                "Score": score
            })
        except Exception:
            continue

    res_df = pd.DataFrame(results)
    if not res_df.empty:
        res_df = res_df.sort_values(by="Score", ascending=False).reset_index(drop=True)
    return res_df

if st.button("🚀 Run Screener Now", use_container_width=True, type="primary"):
    with st.spinner("Scanning NSE stocks & calculating volume surges..."):
        df_hits = run_screener(
            tickers, min_vol_ratio, min_weekly_vol_ratio, rr_ratio, only_uptrend
        )

    if df_hits.empty:
        st.warning("No stocks matched your strict filters today. Try lowering the volume spike sliders.")
    else:
        st.success(f"Found {len(df_hits)} qualifying stocks! Top setups ranked by Score:")
        st.dataframe(df_hits, use_container_width=True, hide_index=True)

        csv = df_hits.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Download Trade List (CSV)",
            data=csv,
            file_name="nse_trade_candidates.csv",
            mime="text/csv",
            use_container_width=True
        )
