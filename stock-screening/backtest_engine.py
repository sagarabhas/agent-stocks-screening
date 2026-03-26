import yfinance as yf
import pandas as pd
import numpy as np
import pandas_ta as ta
import datetime
import os
import streamlit as st
from groq import Groq

@st.cache_data(ttl=3600)
def run_vectorized_backtest(tickers, years, strategy_type, custom_query=None, stop_loss_pct=0.0, take_profit_pct=0.0, oos_split=0.0):
    """
    Downloads historical data, calculates AI-required features, and applies dynamic strategies.
    """
    all_equity_curves = pd.DataFrame()
    metrics_list = []

    end_date = datetime.date.today()
    start_date = end_date - datetime.timedelta(days=years*365)

    for ticker in tickers:
        try:
            # 1. Download Data
            df_backtest = yf.download(ticker, start=start_date, end=end_date, progress=False)

            if isinstance(df_backtest.columns, pd.MultiIndex):
                df_backtest.columns = df_backtest.columns.droplevel(1)

            if df_backtest.empty:
                continue

            # 2. PRE-CALCULATE ALL ALLOWED AI FEATURES
            # We must compute these before the custom query tries to read them!
            df_backtest['SMA_20'] = ta.sma(df_backtest['Close'], length=20)
            df_backtest['SMA_50'] = ta.sma(df_backtest['Close'], length=50)
            df_backtest['SMA_150'] = ta.sma(df_backtest['Close'], length=150)
            df_backtest['SMA_200'] = ta.sma(df_backtest['Close'], length=200)
            df_backtest['RSI_14'] = ta.rsi(df_backtest['Close'], length=14)

            # MACD returns multiple columns, we grab the main ones safely
            macd = ta.macd(df_backtest['Close'])
            if macd is not None and not macd.empty:
                df_backtest['MACD'] = macd.iloc[:, 0]
                df_backtest['MACD_Histogram'] = macd.iloc[:, 1]
            else:
                df_backtest['MACD'] = 0
                df_backtest['MACD_Histogram'] = 0

            df_backtest['High_52week'] = df_backtest['High'].rolling(window=252).max()
            df_backtest['Low_52week'] = df_backtest['Low'].rolling(window=252).min()
            df_backtest['Volume_SMA_20'] = ta.sma(df_backtest['Volume'], length=20)

            # Drop NaNs created by rolling windows so the backtest starts clean
            df_backtest.dropna(inplace=True)
            if df_backtest.empty:
                continue

            df_backtest['Return'] = df_backtest['Close'].pct_change()

            # 3. APPLY THE DYNAMIC AI STRATEGY
            if strategy_type == "Custom AI Strategy" and custom_query:
                try:
                    # df.eval() takes the string "Close > SMA_50" and evaluates it mathematically!
                    # It returns a boolean (True/False). np.where converts that to 1/0.
                    df_backtest['Signal'] = np.where(df_backtest.eval(custom_query), 1, 0)
                except Exception as e:
                    print(f"Error evaluating custom query for {ticker}: {e}")
                    df_backtest['Signal'] = 0 # Fail safely to cash

            elif strategy_type == "SMA Crossover (50 vs 200)":
                df_backtest['Signal'] = np.where(df_backtest['SMA_50'] > df_backtest['SMA_200'], 1, 0)

            elif strategy_type == "RSI Mean Reversion (<30 Buy)":
                df_backtest['Signal'] = 0
                df_backtest.loc[df_backtest['RSI_14'] < 30, 'Signal'] = 1
                df_backtest.loc[df_backtest['RSI_14'] > 70, 'Signal'] = 0
                df_backtest['Signal'] = df_backtest['Signal'].ffill().fillna(0)
            else:
                df_backtest['Signal'] = 0

            # ---------------------------------------------------------
            # 3.5: HARD RISK MANAGEMENT (Stop-Loss & Take-Profit)
            # ---------------------------------------------------------
            if stop_loss_pct > 0 or take_profit_pct > 0:
                # 1. Identify the exact day a new trade starts (Safe Shift)
                df_backtest['Trade_Start'] = (df_backtest['Signal'] == 1) & (df_backtest['Signal'].shift(1).fillna(0) == 0)

                # 2. Record the entry price and forward-fill it
                df_backtest['Entry_Price'] = np.where(df_backtest['Trade_Start'], df_backtest['Close'], np.nan)
                df_backtest['Entry_Price'] = df_backtest['Entry_Price'].ffill()

                # 3. Calculate the unrealized return
                df_backtest['Unrealized_Pct'] = (df_backtest['Close'] - df_backtest['Entry_Price']) / df_backtest['Entry_Price']

                # 4. Bulletproof Series generation (Prevents TypeErrors if SL or TP is 0)
                sl_series = (df_backtest['Unrealized_Pct'] <= -stop_loss_pct) if stop_loss_pct > 0 else pd.Series(False, index=df_backtest.index)
                tp_series = (df_backtest['Unrealized_Pct'] >= take_profit_pct) if take_profit_pct > 0 else pd.Series(False, index=df_backtest.index)

                # 5. Force the signal to 0 if a limit is hit
                df_backtest.loc[sl_series | tp_series, 'Signal'] = 0


            # ---------------------------------------------------------

            # 4. Prevent Lookahead Bias & Apply The Reality Tax
            df_backtest['Position'] = df_backtest['Signal'].shift(1).fillna(0)

            # Detect exactly when a trade occurs (Position changes from 0 to 1, or 1 to 0)
            # .diff() subtracts yesterday's position from today's position.
            # .abs() turns -1 (sells) and 1 (buys) into an absolute 1 so we can tax both legs.
            df_backtest['Trades'] = df_backtest['Position'].diff().fillna(0).abs()

            # Define the friction penalty: 0.15% per transaction
            friction_penalty = 0.0015

            # Calculate Return: Daily market return MINUS the friction penalty on trading days
            df_backtest['Strategy_Return'] = (df_backtest['Position'] * df_backtest['Return']) - (df_backtest['Trades'] * friction_penalty)

            # 5. Calculate Equity Curves
            df_backtest['Cumulative_Hold'] = (1 + df_backtest['Return']).cumprod()
            df_backtest['Cumulative_Strategy'] = (1 + df_backtest['Strategy_Return']).cumprod()

            all_equity_curves[f"{ticker} (Hold)"] = df_backtest['Cumulative_Hold']
            all_equity_curves[f"{ticker} (Strategy)"] = df_backtest['Cumulative_Strategy']

            # 6. Calculate Metrics, Walk-Forward, and Kelly Criterion
            total_strategy_return = (df_backtest['Cumulative_Strategy'].iloc[-1] - 1) * 100
            total_hold_return = (df_backtest['Cumulative_Hold'].iloc[-1] - 1) * 100

            rolling_max = df_backtest['Cumulative_Strategy'].cummax()
            max_drawdown = (df_backtest['Cumulative_Strategy'] / rolling_max - 1.0).min() * 100

            daily_returns = df_backtest['Strategy_Return']
            days_in_market = (df_backtest['Position'] == 1).sum()

            if days_in_market > 0:
                ann_return = daily_returns.mean() * 252
                ann_volatility = daily_returns.std() * np.sqrt(252)
                sharpe_ratio = ann_return / ann_volatility if ann_volatility != 0 else 0

                winning_days = (daily_returns > 0).sum()
                win_rate = winning_days / days_in_market

                # --- NEW: KELLY CRITERION (POSITION SIZING) ---
                avg_win = daily_returns[daily_returns > 0].mean() if winning_days > 0 else 0
                avg_loss = abs(daily_returns[daily_returns < 0].mean()) if (days_in_market - winning_days) > 0 else 0

                if avg_loss > 0:
                    reward_risk = avg_win / avg_loss
                    # Calculate the raw Kelly fraction
                    kelly_pct = win_rate - ((1 - win_rate) / reward_risk)

                    # FULL KELLY: No division by 2. We just floor it at 0% so it doesn't tell you to short.
                    target_kelly = max(0, kelly_pct * 100)
                else:
                    target_kelly = 100 if win_rate > 0 else 0 # 100% win rate edge case
            else:
                sharpe_ratio = 0
                win_rate = 0
                target_kelly = 0

            # --- NEW: WALK-FORWARD VALIDATION (OOS TEST) ---
            oos_return_str = "N/A"
            if oos_split > 0:
                split_idx = int(len(df_backtest) * (1 - oos_split))
                df_oos = df_backtest.iloc[split_idx:].copy()

                if not df_oos.empty:
                    df_oos['OOS_Cumulative'] = (1 + df_oos['Strategy_Return']).cumprod()
                    oos_return = (df_oos['OOS_Cumulative'].iloc[-1] - 1) * 100
                    oos_return_str = f"{oos_return:.2f}%"

            # Append everything to the final table
            metrics_list.append({
                "Ticker": ticker,
                "Total Return": f"{total_strategy_return:.2f}%",
                "Buy & Hold": f"{total_hold_return:.2f}%",
                "Max DD": f"{max_drawdown:.2f}%",
                "Sharpe": f"{sharpe_ratio:.2f}",
                "Win Rate": f"{win_rate*100:.1f}%",
                "Rec. Allocation (Full Kelly)": f"{target_kelly:.1f}%", # Updated Label!
                "Out-of-Sample Return": oos_return_str
            })

        except Exception as e:
            print(f"Error backtesting {ticker}: {e}")

    metrics_df = pd.DataFrame(metrics_list) if metrics_list else pd.DataFrame()
    return all_equity_curves, metrics_df

def analyze_backtest_with_ai(metrics_df, tickers, strategy_name):
    """
    Sends the backtest results to an LLM for a professional quantitative review,
    now including Walk-Forward and Kelly Criterion analysis.
    """
    from groq import Groq
    import os

    try:
        client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

        # Convert the dataframe to a string format the LLM can easily read
        metrics_str = metrics_df.to_string(index=False)

        system_prompt = """
        You are the Chief Quantitative Officer at an elite algorithmic trading firm. Your job is to bluntly evaluate backtest metrics for a proposed trading strategy.
        
        You must evaluate the strategy based on the following data points provided:
        1. **Total Return vs Buy & Hold:** Did the active strategy actually beat the market, or did we take on extra risk for no reason?
        2. **Max Drawdown & Sharpe Ratio:** Is the volatility acceptable? (Sharpe > 1.0 is good, > 2.0 is excellent).
        3. **Out-of-Sample Return (Walk-Forward Validation):** This is the most critical metric. If the Total Return is high but the Out-of-Sample return is negative or "N/A", the strategy is over-fitted (curve-fit) to the past and will fail in live trading. Call this out aggressively.
        4. **Rec. Allocation (Full Kelly):** This is the Kelly Criterion percentage. If it recommends 0%, the strategy is mathematically guaranteed to lose money over time. If it recommends > 25%, warn the user that Full Kelly is highly aggressive and they should expect massive volatility.

        Output Format:
        Use bolding and bullet points for readability. Be concise, mathematically rigorous, and do not sugarcoat bad strategies. End with a definitive "Verdict: [Pass / Fail / Needs Optimization]".
        """

        user_prompt = f"Strategy Name: {strategy_name}\nTarget Tickers: {tickers}\n\nHere are the backtest metrics:\n{metrics_str}\n\nProvide your brutal quantitative assessment."

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3 # Keep it analytical and grounded
        )

        return response.choices[0].message.content

    except Exception as e:
        return f"Error connecting to AI Analyst: {e}"

def run_grid_search_optimization(ticker, years, fast_range, slow_range, stop_loss_pct=0.0, take_profit_pct=0.0):
    """
    Runs a brute-force grid search to find the optimal SMA crossover parameters.
    fast_range: tuple (start, end, step) e.g., (10, 50, 5)
    slow_range: tuple (start, end, step) e.g., (50, 200, 10)
    """
    import itertools

    end_date = datetime.date.today()
    start_date = end_date - datetime.timedelta(days=years*365)

    try:
        df_raw = yf.download(ticker, start=start_date, end=end_date, progress=False)
        if isinstance(df_raw.columns, pd.MultiIndex):
            df_raw.columns = df_raw.columns.droplevel(1)
        if df_raw.empty:
            return pd.DataFrame()

        df_raw['Return'] = df_raw['Close'].pct_change()
    except Exception as e:
        print(f"Failed to download data for optimization: {e}")
        return pd.DataFrame()

    fast_smas = range(fast_range[0], fast_range[1] + 1, fast_range[2])
    slow_smas = range(slow_range[0], slow_range[1] + 1, slow_range[2])

    results = []

    # Pre-calculate all possible SMAs to save massive amounts of time
    all_smas = set(fast_smas).union(set(slow_smas))
    for ma in all_smas:
        df_raw[f'SMA_{ma}'] = ta.sma(df_raw['Close'], length=ma)

    df_clean = df_raw.dropna().copy()
    if df_clean.empty: return pd.DataFrame()

    # Iterate through every combination
    for fast, slow in itertools.product(fast_smas, slow_smas):
        if fast >= slow: continue # Fast MA must be faster than Slow MA!

        df = df_clean[['Close', 'Return', f'SMA_{fast}', f'SMA_{slow}']].copy()

        # 1. Base Signal
        df['Signal'] = np.where(df[f'SMA_{fast}'] > df[f'SMA_{slow}'], 1, 0)

        # 2. Hard Risk Management
        if stop_loss_pct > 0 or take_profit_pct > 0:
            df['Trade_Start'] = (df['Signal'] == 1) & (df['Signal'].shift(1).fillna(0) == 0)
            df['Entry_Price'] = np.where(df['Trade_Start'], df['Close'], np.nan)
            df['Entry_Price'] = df['Entry_Price'].ffill()
            df['Unrealized_Pct'] = (df['Close'] - df['Entry_Price']) / df['Entry_Price']

            sl_series = (df['Unrealized_Pct'] <= -stop_loss_pct) if stop_loss_pct > 0 else pd.Series(False, index=df.index)
            tp_series = (df['Unrealized_Pct'] >= take_profit_pct) if take_profit_pct > 0 else pd.Series(False, index=df.index)

            df.loc[sl_series | tp_series, 'Signal'] = 0

        # 3. Apply Reality Tax & Shift
        df['Position'] = df['Signal'].shift(1).fillna(0)
        df['Trades'] = df['Position'].diff().fillna(0).abs()
        df['Strategy_Return'] = (df['Position'] * df['Return']) - (df['Trades'] * 0.0015)

        # 4. Metrics
        df['Cumulative_Strategy'] = (1 + df['Strategy_Return']).cumprod()
        total_ret = (df['Cumulative_Strategy'].iloc[-1] - 1) * 100

        days_in_market = (df['Position'] == 1).sum()
        if days_in_market > 0:
            ann_ret = df['Strategy_Return'].mean() * 252
            ann_vol = df['Strategy_Return'].std() * np.sqrt(252)
            sharpe = ann_ret / ann_vol if ann_vol != 0 else 0
        else:
            sharpe = 0

        results.append({
            "Fast SMA": fast,
            "Slow SMA": slow,
            "Total Return": round(total_ret, 2),
            "Sharpe Ratio": round(sharpe, 2)
        })

    # Convert to DataFrame and sort by Sharpe Ratio (Highest first)
    res_df = pd.DataFrame(results)
    if not res_df.empty:
        res_df = res_df.sort_values(by="Sharpe Ratio", ascending=False).reset_index(drop=True)
    return res_df

def optimize_strategy_with_ai(original_query, metrics_df, tickers):
    """
    Acts as a quantitative risk manager. Ingests backtest metrics and rewrites
    the Pandas query to improve the Sharpe Ratio and reduce Drawdown.
    """
    from groq import Groq
    import os

    try:
        client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

        metrics_str = metrics_df.to_dict()

        system_prompt = """
        You are a quantitative risk manager and algorithmic trading expert. Your job is to mathematically OPTIMIZE an existing pandas query based on its backtest results.
        Your goal is to increase the Sharpe Ratio and Win Rate, while significantly reducing the Max Drawdown.
        
        Available Columns: Close, Open, High, Low, Volume, SMA_20, SMA_50, SMA_150, SMA_200, RSI_14, MACD, MACD_Histogram, High_52week, Low_52week, Volume_SMA_20.
        
        Rules for Optimization:
        1. DO NOT just blindly append 'and' conditions. This leads to over-fitting and zero trades.
        2. TUNE EXISTING PARAMETERS: If the query uses `RSI_14 < 30`, maybe adjust it to `< 40` or `< 20`. If it uses `SMA_50`, maybe `SMA_20` or `SMA_150` works better for risk control.
        3. REPLACE BAD LOGIC: If the original query is bleeding money, you have permission to delete the bad conditions and rewrite the logic to be smarter (e.g., swapping a pure mean-reversion strategy to a trend-following one).
        4. Keep the final query elegant and concise. 
        5. Output ONLY the raw pandas query string. NO markdown formatting, NO explanations, NO backticks.
        """

        user_prompt = f"Original Query: {original_query}\nBacktest Metrics for {tickers}: {metrics_str}\n\nProvide the optimized pandas query string to improve these metrics."

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1 # Very low temperature for strict mathematical output
        )

        # Clean the output to ensure it's a valid Pandas string
        raw_query = response.choices[0].message.content.strip()
        clean_query = raw_query.replace("```python", "").replace("```", "").replace("`", "").strip()

        return clean_query

    except Exception as e:
        return f"Error: {e}"