import yfinance as yf
import pandas as pd
import streamlit as st

@st.cache_data(ttl=3600) # Cache for 1 hour to prevent API spam
def get_market_regime():
    """
    Pulls Nifty 50 and India VIX data to classify the current market environment.
    Returns a dictionary of metrics and the recommended systematic action.
    """
    try:
        # Download Nifty 50 and India VIX data
        nifty = yf.download("^NSEI", period="1y", progress=False)
        vix = yf.download("^INDIAVIX", period="1y", progress=False)

        # Calculate Nifty Moving Averages
        nifty['SMA_50'] = nifty['Close'].rolling(window=50).mean()
        nifty['SMA_200'] = nifty['Close'].rolling(window=200).mean()

        # Extract the latest values cleanly
        latest_nifty = float(nifty['Close'].iloc[-1].iloc[0]) if isinstance(nifty['Close'], pd.DataFrame) else float(nifty['Close'].iloc[-1])
        sma_50 = float(nifty['SMA_50'].iloc[-1].iloc[0]) if isinstance(nifty['SMA_50'], pd.DataFrame) else float(nifty['SMA_50'].iloc[-1])
        sma_200 = float(nifty['SMA_200'].iloc[-1].iloc[0]) if isinstance(nifty['SMA_200'], pd.DataFrame) else float(nifty['SMA_200'].iloc[-1])
        latest_vix = float(vix['Close'].iloc[-1].iloc[0]) if isinstance(vix['Close'], pd.DataFrame) else float(vix['Close'].iloc[-1])

        # Determine the Regime
        if latest_nifty > sma_50 and sma_50 > sma_200 and latest_vix < 20:
            regime = "🟢 BULL MARKET (Risk On)"
            action = "Run Breakout & Momentum Strategies (e.g., Minervini)."
        elif latest_nifty < sma_200 or latest_vix > 25:
            regime = "🔴 BEAR MARKET (Risk Off)"
            action = "Hold Cash. Do not buy breakouts."
        else:
            regime = "🟡 CHOPPY / TRANSITION (Caution)"
            action = "Run Mean Reversion (Buy the dip on Blue-Chips)."

        return {
            "regime": regime,
            "action": action,
            "nifty": round(latest_nifty, 2),
            "vix": round(latest_vix, 2),
            "sma_50": round(sma_50, 2),
            "sma_200": round(sma_200, 2)
        }
    except Exception as e:
        return {"error": str(e)}