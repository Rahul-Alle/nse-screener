import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import re
from datetime import datetime, timedelta, timezone

st.set_page_config(page_title="Screener.in + X Swing Agent", layout="wide")
st.title("🤖 Screener.in & X (Twitter) Swing Agent")
st.caption("Tracks Screener.in screens and daily tweets from specific X profiles, then filters for high-move NSE setups.")

# Common non-stock uppercase words to ignore when parsing tweets
STOPWORDS = {
    "THE", "AND", "FOR", "WITH", "THIS", "THAT", "FROM", "HAVE", "WILL", "WHAT",
    "BUY", "SELL", "HOLD", "LONG", "SHORT", "TARGET", "STOP", "LOSS", "HIGH",
    "LOW", "OPEN", "CLOSE", "TODAY", "WEEK", "MONTH", "YEAR", "STOCK", "STOCKS",
    "SHARE", "SHARES", "MARKET", "NIFTY", "SENSEX", "BANKNIFTY", "FINNIFTY",
    "NSE", "BSE", "IPO", "CMP", "AVG", "VOL", "VOLUME", "BREAKOUT", "SWING",
    "TRADE", "TRADING", "PROFIT", "BOOK", "EXIT", "ENTRY", "ALERT", "UPDATE",
    "NEWS", "Q1", "Q2", "Q3", "Q4", "FY25", "FY26", "FY27", "YOY", "QOQ",
    "EBITDA", "PAT", "ROE", "ROCE", "PE", "PB", "EPS", "RSI", "MACD", "EMA",
    "SMA", "ATR", "ADX", "VWAP", "FII", "DII", "SEBI", "RBI", "INR", "USD",
    "GOOD", "BEST", "TOP", "NEW", "ALL", "BIG", "NEXT", "FREE", "LIVE", "CHART"
}

PRESET_SCREENS = {
    "Companies Creating New Highs": "https://www.screener.in/screens/248440/companies-creating-new-highs/",
    "Strong Quarterly Growth + High ROCE": "https://www.screener.in/screens/71064/bullish-stocks/",
    "High Growth Midcaps (Low Debt)": "https://www.screener.in/screens/357649/high-growth-midcaps/",
    "Custom Screener.in URL": ""
}

# --- Shared Trade Filter Controls ---
with st.expander("⚙️ Price, Volume & Move Filters (Applies to All Sources)", expanded=False):
    col1, col2 = st.columns(2)
    with col1:
        min_price = st.number_input("Min Price (₹)", 10, 50000, 50, 10)
    with col2:
        max_price = st.number_input("Max Price (₹)", 20, 100000, 10000, 50)

    min_atr_pct = st.slider("Min Avg Daily Move / ATR (%)", 0.5, 6.0, 1.5, 0.2)
    min_vol_ratio = st.slider("Min Daily Volume Spike (x 20-day Avg)", 0.5, 4.0, 1.0, 0.1)
    rr_ratio = st.slider("Target Risk-to-Reward Ratio", 1.5, 4.0, 2.5, 0.5)
    only_uptrend = st.checkbox("Require Uptrend (Price > 50 SMA)", value=True)


def extract_tickers_from_text(text: str) -> list:
    """Extracts candidate NSE ticker symbols ($SYMBOL, #SYMBOL, or ALL-CAPS words)."""
    # Prioritize cashtags ($TRENT) and hashtags (#HAL), plus standalone uppercase words (2-15 chars)
    tagged = re.findall(r"[\$#]([A-Za-z]{2,15})\b", text)
    caps = re.findall(r"\b([A-Z]{2,15})\b", text)
    combined = [t.upper() for t in tagged] + caps
    cleaned = []
    for sym in combined:
        if sym not in STOPWORDS and sym not in cleaned:
            cleaned.append(sym)
    return cleaned


def fetch_x_profile_mentions(handles: list, api_key: str, hours_back: int = 24) -> dict:
    """
    Fetches tweets from the last `hours_back` hours for each handle via TwitterAPI.io
    and returns { 'SYMBOL.NS': {'count': int, 'by': set_of_handles} }.
    """
    url = "https://api.twitterapi.io/twitter/tweet/advanced_search"
    headers = {"X-API-Key": api_key.strip()}
    since_ts = int((datetime.now(timezone.utc) - timedelta(hours=hours_back)).timestamp())

    mention_map = {}
    for handle in handles:
        clean_handle = handle.strip().lstrip("@")
        if not clean_handle:
            continue
        query = f"from:{clean_handle} since_time:{since_ts}"
        try:
            resp = requests.get(
                url,
                headers=headers,
                params={"query": query, "queryType": "Latest"},
                timeout=12
            )
            if resp.status_code != 200:
                continue
            data = resp.json()
            tweets = data.get("tweets", [])
            for tw in tweets:
                text = tw.get("text", "")
                candidates = extract_tickers_from_text(text)
                for sym in candidates:
                    ns_sym = f"{sym}.NS"
                    if ns_sym not in mention_map:
                        mention_map[ns_sym] = {"count": 0, "by": set()}
                    mention_map[ns_sym]["count"] += 1
                    mention_map[ns_sym]["by"].add(f"@{clean_handle}")
        except Exception:
            continue
    return mention_map


@st.cache_data(ttl=1800)
def fetch_symbols_from_screener(base_url: str, pages: int) -> list:
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    }
    symbols = []
    clean_base = base_url.split("?")[0].rstrip("/") + "/"

    for page in range(1, pages + 1):
        try:
            resp = requests.get(f"{clean_base}?page={page}", headers=headers, timeout=12)
            if resp.status_code != 200:
                break
            soup = BeautifulSoup(resp.text, "html.parser")
            table = soup.find("table", class_="data-table")
            if not table:
                break
            found = 0
            for link in table.find_all("a", href=True):
                match = re.match(r"^/company/([^/]+)/", link["href"])
                if match:
                    code = match.group(1).strip().upper()
                    formatted = f"{code}.BO" if code.isdigit() else f"{code}.NS"
                    if formatted not in symbols:
                        symbols.append(formatted)
                        found += 1
            if found == 0:
                break
        except Exception:
            break
    return symbols


@st.cache_data(ttl=900)
def analyze_candidates(symbol_list, p_min, p_max, atr_pct_thresh, vol_thresh, rr, req_uptrend, mention_info=None):
    if not symbol_list:
        return pd.DataFrame()

    nifty = yf.download("^NSEI", period="6mo", interval="1d", progress=False, auto_adjust=True)
    if isinstance(nifty.columns, pd.MultiIndex):
        nifty.columns = nifty.columns.get_level_values(0)
    nifty_1m_ret = (nifty["Close"].iloc[-1] / nifty["Close"].iloc[-21] - 1) * 100

    raw = yf.download(symbol_list, period="1y", interval="1d", group_by="ticker", progress=False, auto_adjust=True)
    results = []

    for sym in symbol_list:
        try:
            df = raw[sym].dropna() if len(symbol_list) > 1 else raw.dropna()
            if len(df) < 60:
                continue

            close = df["Close"]
            high = df["High"]
            low = df["Low"]
            vol = df["Volume"]

            curr_price = float(close.iloc[-1])
            if not (p_min <= curr_price <= p_max):
                continue

            sma_50 = float(close.rolling(50).mean().iloc[-1])
            if req_uptrend and curr_price < sma_50:
                continue

            tr = pd.concat([
                high - low,
                (high - close.shift()).abs(),
                (low - close.shift()).abs()
            ], axis=1).max(axis=1)
            atr_14 = float(tr.rolling(14).mean().iloc[-1])
            atr_pct = (atr_14 / curr_price) * 100

            if atr_pct < atr_pct_thresh:
                continue

            avg_vol_20 = float(vol.iloc[-21:-1].mean())
            curr_vol = float(vol.iloc[-1])
            daily_vol_ratio = curr_vol / avg_vol_20 if avg_vol_20 > 0 else 0

            if daily_vol_ratio < vol_thresh:
                continue

            stock_1m_ret = (curr_price / float(close.iloc[-21]) - 1) * 100
            rs_vs_nifty = stock_1m_ret - nifty_1m_ret

            stop_loss = round(curr_price - (1.5 * atr_14), 2)
            risk_per_share = curr_price - stop_loss
            target_price = round(curr_price + (risk_per_share * rr), 2)
            expected_gain_pct = round(((target_price - curr_price) / curr_price) * 100, 1)

            x_count = 0
            x_profiles = "-"
            if mention_info and sym in mention_info:
                x_count = mention_info[sym]["count"]
                x_profiles = ", ".join(sorted(mention_info[sym]["by"]))

            # Bonus points when multiple tracked X profiles tweeted the stock today
            score = round((atr_pct * 20) + (daily_vol_ratio * 25) + (rs_vs_nifty * 1.5) + (x_count * 10), 1)

            row = {
                "Symbol": sym.replace(".NS", "").replace(".BO", ""),
                "Price (₹)": round(curr_price, 2),
                "Avg Move/Day (%)": round(atr_pct, 2),
                "Daily Vol (x)": round(daily_vol_ratio, 2),
                "1M vs Nifty (%)": round(rs_vs_nifty, 2),
                "Stop Loss (₹)": stop_loss,
                "Target (₹)": target_price,
                "Target Gain (%)": expected_gain_pct,
                "Score": score
            }
            if mention_info is not None:
                row["X Mentions (24h)"] = x_count
                row["Tweeted By"] = x_profiles

            results.append(row)
        except Exception:
            continue

    res_df = pd.DataFrame(results)
    if not res_df.empty:
        res_df = res_df.sort_values(by="Score", ascending=False).reset_index(drop=True)
    return res_df


# --- Two Tabs: X Profile Watcher & Screener.in Auto-Scanner ---
tab_x, tab_scr = st.tabs(["🐦 X (Twitter) Profile Scanner", "📊 Screener.in Auto-Scanner"])

with tab_x:
    st.subheader("Check Stocks Tweeted by Selected Profiles Today")
    x_handles_input = st.text_area(
        "X Profile Usernames (comma or newline separated, without @)",
        placeholder="Example:\nChartink\nNSEIndia\nYourFavoriteSwingTrader",
        height=100
    )
    hours_window = st.selectbox("Lookback Window", [24, 48, 72], index=0, format_func=lambda h: f"Last {h} Hours")

    api_key_input = st.text_input(
        "TwitterAPI.io API Key (leave blank to use Manual Tweet Paste below)",
        type="password"
    )

    manual_tweets_paste = st.text_area(
        "Or Paste Tweets Directly from Your Phone (No API Key Needed)",
        placeholder="Paste copied tweets here — the app will auto-extract all stock symbols and run the ATR/volume check...",
        height=100
    )

    if st.button("🐦 Scan Tweeted Stocks Now", use_container_width=True, type="primary"):
        mention_data = {}

        if api_key_input.strip() and x_handles_input.strip():
            handles = [h for line in x_handles_input.splitlines() for h in line.split(",") if h.strip()]
            with st.spinner(f"Checking tweets from {len(handles)} profiles over the last {hours_window}h..."):
                mention_data = fetch_x_profile_mentions(handles, api_key_input, hours_window)
        elif manual_tweets_paste.strip():
            extracted = extract_tickers_from_text(manual_tweets_paste)
            for sym in extracted:
                mention_data[f"{sym}.NS"] = {"count": 1, "by": {"Pasted Feed"}}
        else:
            st.error("Enter X handles + an API key, OR paste tweets into the manual box.")

        if mention_data:
            candidate_syms = list(mention_data.keys())
            st.info(f"Extracted {len(candidate_syms)} candidate ticker words from tweets. Validating against live NSE data & volume filters...")
            with st.spinner("Running technical & volume checks on tweeted stocks..."):
                df_x = analyze_candidates(
                    candidate_syms, min_price, max_price, min_atr_pct, min_vol_ratio, rr_ratio, only_uptrend, mention_data
                )
            if df_x.empty:
                st.warning("None of the tweeted stocks passed your price, uptrend, or volume spike filters today.")
            else:
                st.success(f"Found {len(df_x)} valid NSE setups from today's tweets!")
                st.dataframe(df_x, use_container_width=True, hide_index=True)
        elif api_key_input.strip() or manual_tweets_paste.strip():
            st.warning("No stock symbols were detected in the tweets from that time window.")

with tab_scr:
    st.subheader("Auto-Fetch from Screener.in")
    selected_preset = st.selectbox("Choose a Screener.in Screen", list(PRESET_SCREENS.keys()))
    screener_url = (
        st.text_input("Paste Public Screener.in URL", placeholder="https://www.screener.in/screens/...")
        if selected_preset == "Custom Screener.in URL"
        else st.text_input("Screener.in URL", value=PRESET_SCREENS[selected_preset])
    )
    max_pages = st.slider("Pages to Fetch (25 stocks/page)", 1, 5, 3)

    if st.button("🚀 Fetch from Screener.in & Rank", use_container_width=True, type="primary"):
        if not screener_url.strip():
            st.error("Please enter a valid Screener.in screen URL.")
        else:
            with st.spinner("Fetching stocks from Screener.in..."):
                fetched = fetch_symbols_from_screener(screener_url, max_pages)
            if not fetched:
                st.error("Could not extract symbols. Verify the URL is a public Screener.in screen.")
            else:
                st.info(f"Fetched {len(fetched)} stocks. Analyzing live price action & volume...")
                with st.spinner("Calculating ATR moves, volume surges & targets..."):
                    df_scr = analyze_candidates(
                        fetched, min_price, max_price, min_atr_pct, min_vol_ratio, rr_ratio, only_uptrend
                    )
                if df_scr.empty:
                    st.warning("No Screener.in stocks passed your volume/ATR filters today.")
                else:
                    st.success(f"Top {len(df_scr)} Swing Trade Picks:")
                    st.dataframe(df_scr, use_container_width=True, hide_index=True)
