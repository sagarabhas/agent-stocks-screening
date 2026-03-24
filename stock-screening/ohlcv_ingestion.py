import pandas as pd
import yfinance as yf
import os
import time

def fetch_ohlcv_data(input_csv="screener_results_with_tickers.csv", output_dir="stock_data"):
    # 1. Create a directory to store our historical data (our mini Data Lake)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}/")

    # 2. Load the tickers from Phase 1
    if not os.path.exists(input_csv):
        print(f"Error: '{input_csv}' not found. Please run Phase 1 first.")
        return

    df = pd.read_csv(input_csv)

    # Filter out any rows where the Ticker is missing (NaN)
    valid_stocks = df.dropna(subset=['Ticker']).copy()
    tickers = valid_stocks['Ticker'].tolist()

    print(f"Starting OHLCV ingestion for {len(tickers)} valid tickers...\n")

    successful_downloads = 0

    # 3. Ingestion Loop
    for ticker in tickers:
        print(f"Fetching 1-year data for {ticker}...")
        try:
            # Initialize the yfinance Ticker object
            stock = yf.Ticker(ticker)

            # Fetch 1 year of daily data
            hist = stock.history(period="1y")

            if not hist.empty:
                # Data Cleaning: yfinance sometimes adds timezone info to the Date index.
                # We strip it out because it causes compatibility issues with technical analysis libraries later.
                hist.index = hist.index.tz_localize(None)

                # Save the individual stock's history to our data folder
                file_path = os.path.join(output_dir, f"{ticker}.csv")
                hist.to_csv(file_path)

                successful_downloads += 1
            else:
                print(f"  [!] No historical data returned. (Stock might be recently listed or suspended)")

        except Exception as e:
            print(f"  [!] Failed to download: {e}")

        # 4. Rate-Limit Protection
        # Yahoo Finance will block your IP if you hit them with 100 requests in 2 seconds.
        time.sleep(0.5)

    print("\n--- Time-Series Ingestion Complete ---")
    print(f"Successfully downloaded data for {successful_downloads}/{len(tickers)} stocks.")
    print(f"All OHLCV data is saved in the '{output_dir}/' folder.")

# --- Main Execution ---
if __name__ == "__main__":
    fetch_ohlcv_data()