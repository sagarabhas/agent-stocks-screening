import yfinance as yf
import pandas as pd
import streamlit as st

@st.cache_data(ttl=3600)
def get_sector_performance():
    """
    Downloads historical data for major Nifty Sectoral Indices.
    Calculates percentage returns to identify where institutional money is flowing.
    """
    # Yahoo Finance tickers for Indian Sectors
    sectors = {
        "Nifty Bank": "^NSEBANK",
        "Nifty IT": "^CNXIT",
        "Nifty Auto": "^CNXAUTO",
        "Nifty Pharma": "^CNXPHARMA",
        "Nifty Metal": "^CNXMETAL",
        "Nifty FMCG": "^CNXFMCG",
        "Nifty Energy": "^CNXENERGY",
        "Nifty Realty": "^CNXREALTY",
        "Nifty 50 (Baseline)": "^NSEI"
    }

    try:
        # Download 3 months of data for all sectors at once
        tickers = list(sectors.values())
        data = yf.download(tickers, period="3mo", progress=False)['Close']

        results = []
        for name, ticker in sectors.items():
            if ticker in data.columns:
                series = data[ticker].dropna()

                # Ensure we have enough trading days (approx 21 days in a month)
                if len(series) >= 21:
                    # Calculate relative momentum (%)
                    ret_1w = ((series.iloc[-1] - series.iloc[-5]) / series.iloc[-5]) * 100
                    ret_1m = ((series.iloc[-1] - series.iloc[-21]) / series.iloc[-21]) * 100
                    ret_3m = ((series.iloc[-1] - series.iloc[0]) / series.iloc[0]) * 100

                    results.append({
                        "Sector": name,
                        "1-Week (%)": float(ret_1w),
                        "1-Month (%)": float(ret_1m),
                        "3-Month (%)": float(ret_3m)
                    })

        # Convert to DataFrame and sort by the 1-Month trend (the sweet spot for swing trading)
        df = pd.DataFrame(results)
        if not df.empty:
            df = df.set_index("Sector")
            df = df.sort_values(by="1-Month (%)", ascending=False)

        return df
    except Exception as e:
        return pd.DataFrame()