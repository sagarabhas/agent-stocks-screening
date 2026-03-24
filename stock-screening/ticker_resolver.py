import pandas as pd
import requests
import time
import os

def get_yahoo_ticker(company_name):
    # Yahoo Finance internal search API
    url = f"https://query2.finance.yahoo.com/v1/finance/search?q={company_name}"

    # Yahoo blocks requests without a valid User-Agent header
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }

    try:
        response = requests.get(url, headers=headers, timeout=5)
        response.raise_for_status()
        data = response.json()

        bse_fallback = None

        # Parse the JSON response looking for Indian exchanges
        for quote in data.get('quotes', []):
            symbol = quote.get('symbol', '')

            # We prefer NSE (.NS) for better volume/liquidity
            if symbol.endswith('.NS'):
                return symbol
            # Keep BSE (.BO) as a backup just in case
            elif symbol.endswith('.BO') and not bse_fallback:
                bse_fallback = symbol

        return bse_fallback

    except Exception as e:
        print(f"  [!] Error connecting to API for {company_name}")
        return None

# --- Main Execution ---
if __name__ == "__main__":
    input_file = "screener_results.csv"
    output_file = "screener_results_with_tickers.csv"

    if not os.path.exists(input_file):
        print(f"Error: Could not find '{input_file}'. Please run the Screener scraper first.")
    else:
        print(f"Loading {input_file}...")
        df = pd.read_csv(input_file)

        tickers = []
        unresolved = []

        print(f"Resolving tickers for {len(df)} companies...\n")

        for index, row in df.iterrows():
            name = row['Name']
            print(f"Processing ({index + 1}/{len(df)}): {name}")

            ticker = get_yahoo_ticker(name)
            tickers.append(ticker)

            if ticker:
                print(f"  -> Found: {ticker}")
            else:
                print(f"  -> [!] Could not resolve ticker.")
                unresolved.append(name)

            # Sleep for a fraction of a second so Yahoo doesn't block our IP
            time.sleep(0.5)

            # Add the new column to our DataFrame
        df['Ticker'] = tickers

        # Save the updated CSV
        df.to_csv(output_file, index=False)

        print("\n--- Resolution Complete ---")
        print(f"Successfully resolved: {len(df) - len(unresolved)}/{len(df)}")
        print(f"Data saved to: {output_file}")

        if unresolved:
            print("\nThe following companies require manual ticker mapping:")
            for u in unresolved:
                print(f"- {u}")