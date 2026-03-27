import yfinance as yf
import pandas as pd
import streamlit as st

@st.cache_data(ttl=3600)
def check_smart_money(tickers):
    """
    Scans a list of tickers for Institutional Ownership.
    Insider Activity has been temporarily disabled due to yfinance API mapping errors for Indian markets.
    """
    results = []

    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            info = stock.info

            # 1. Check Institutional Holdings
            inst_holding = "Data Unavailable"
            if 'institutionsPercentHeld' in info and info['institutionsPercentHeld'] is not None:
                inst_holding = f"{info['institutionsPercentHeld'] * 100:.2f}%"
            elif 'heldPercentInstitutions' in info and info['heldPercentInstitutions'] is not None:
                inst_holding = f"{info['heldPercentInstitutions'] * 100:.2f}%"

            # ==========================================
            # 2. INSIDER ACTIVITY (DISABLED)
            # yfinance returns incorrect aggregate promoter holdings
            # instead of recent transactions for NSE/BSE stocks.
            # ==========================================
            # insider_action = "Neutral / No Recent Data"
            # try:
            #     insider_trades = stock.insider_purchases
            #     if isinstance(insider_trades, pd.DataFrame) and not insider_trades.empty:
            #         clean_df = insider_trades.dropna(how='all')
            #         if not clean_df.empty:
            #             if 'Shares' in clean_df.columns:
            #                 max_shares = pd.to_numeric(clean_df['Shares'], errors='coerce').max()
            #                 if max_shares > 0:
            #                     insider_action = f"🔥 Insider Bought {int(max_shares):,} Shares"
            #             elif 'Purchases' in clean_df.columns:
            #                 max_purchases = pd.to_numeric(clean_df['Purchases'], errors='coerce').max()
            #                 if max_purchases > 0:
            #                     insider_action = f"🔥 {int(max_purchases)} Insider Purchases Detected"
            # except Exception:
            #     pass

            # Append only reliable data to the results
            results.append({
                "Ticker": ticker,
                "Institutional Ownership": inst_holding
                # "Insider Footprint": insider_action  <-- Commented out to hide from UI
            })

        except Exception as e:
            results.append({
                "Ticker": ticker,
                "Institutional Ownership": "API Error"
            })

    return pd.DataFrame(results)