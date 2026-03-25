import yfinance as yf
import pandas as pd
import numpy as np
import pandas_ta as ta
import datetime
import os
from groq import Groq

def run_vectorized_backtest(tickers, years, strategy_type, custom_query=None, stop_loss_pct=0.0, take_profit_pct=0.0):
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

            # 6. Calculate Metrics & Institutional Risk
            total_strategy_return = (df_backtest['Cumulative_Strategy'].iloc[-1] - 1) * 100
            total_hold_return = (df_backtest['Cumulative_Hold'].iloc[-1] - 1) * 100

            # Max Drawdown
            rolling_max = df_backtest['Cumulative_Strategy'].cummax()
            drawdown = df_backtest['Cumulative_Strategy'] / rolling_max - 1.0
            max_drawdown = drawdown.min() * 100

            # --- NEW: INSTITUTIONAL RISK METRICS ---
            daily_returns = df_backtest['Strategy_Return']
            days_in_market = (df_backtest['Position'] == 1).sum()

            if days_in_market > 0:
                # Annualized Return and Volatility (Assuming 252 trading days in a year)
                ann_return = daily_returns.mean() * 252
                ann_volatility = daily_returns.std() * np.sqrt(252)

                # Sharpe Ratio (Assuming 0% risk-free rate for pure strategy evaluation)
                sharpe_ratio = ann_return / ann_volatility if ann_volatility != 0 else 0

                # Sortino Ratio (Only penalizes negative returns)
                downside_returns = daily_returns[daily_returns < 0]
                downside_volatility = downside_returns.std() * np.sqrt(252)
                sortino_ratio = ann_return / downside_volatility if downside_volatility != 0 and not pd.isna(downside_volatility) else 0

                # Win Rate (Percentage of invested days that were positive)
                winning_days = (daily_returns > 0).sum()
                win_rate = (winning_days / days_in_market) * 100
            else:
                sharpe_ratio = 0
                sortino_ratio = 0
                win_rate = 0

            # Append everything to the final table
            metrics_list.append({
                "Ticker": ticker,
                "Strategy Return": f"{total_strategy_return:.2f}%",
                "Buy & Hold Return": f"{total_hold_return:.2f}%",
                "Max Drawdown": f"{max_drawdown:.2f}%",
                "Sharpe": f"{sharpe_ratio:.2f}",
                "Sortino": f"{sortino_ratio:.2f}",
                "Win Rate": f"{win_rate:.1f}%"
            })

        except Exception as e:
            print(f"Error backtesting {ticker}: {e}")

    metrics_df = pd.DataFrame(metrics_list) if metrics_list else pd.DataFrame()
    return all_equity_curves, metrics_df

def analyze_backtest_with_ai(metrics_df, tickers, strategy_type):
    """
    Passes the backtest results to the native Groq API for quantitative analysis.
    """
    try:
        client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        prompt = f"Act as a quantitative analyst. I backtested {tickers} using {strategy_type}. Here are the results: {metrics_df.to_dict()}. Briefly analyze if this is a safe strategy to deploy live, focusing on the relationship between Total Return and Max Drawdown."

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"AI Analysis failed: {e}"

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