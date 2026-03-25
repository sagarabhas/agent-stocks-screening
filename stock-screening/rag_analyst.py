import os
import pandas as pd
import urllib.request
import urllib.parse
import urllib.parse
import urllib.request
import yfinance as yf
import xml.etree.ElementTree as ET
from groq import Groq
from dotenv import load_dotenv


load_dotenv()

def fetch_google_news(ticker, num_articles=5):
    """
    Retrieves the latest news headlines for a given stock using Google News RSS.
    """
    # 1. Base fallback ticker
    clean_ticker = ticker.replace('.NS', '').replace('.BO', '')
    search_term = clean_ticker

    # 2. GET THE REAL COMPANY NAME
    try:
        # Fetch official company name from Yahoo Finance
        info = yf.Ticker(ticker).info
        if 'longName' in info and info['longName']:
            # Strip out generic corporate suffixes to make the news search cleaner
            long_name = info['longName']
            for suffix in [" Limited", " Ltd.", " Ltd", " L.t.d", " Corporation", " Inc."]:
                long_name = long_name.replace(suffix, "")

            search_term = long_name.strip()
    except Exception as e:
        print(f"  [!] Could not fetch long name for {ticker}, falling back to ticker: {e}")

    # 3. STRICT SEARCH QUERY
    # We wrap the search term in quotes so Google does an exact match!
    # e.g., '"Valiant Communications" stock news India'
    query = f'"{search_term}" stock news India'

    # URL encode the query and format for Indian localized news
    encoded_query = urllib.parse.quote(query)
    rss_url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-IN&gl=IN&ceid=IN:en"

    print(f"  -> Fetching news for: {search_term} ({ticker})...")

    try:
        # We use a standard User-Agent so Google doesn't block the request
        req = urllib.request.Request(rss_url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response:
            xml_data = response.read()

        # Parse the XML
        root = ET.fromstring(xml_data)
        articles = []

        # Google News RSS wraps items in a <channel>
        for item in root.findall('./channel/item')[:num_articles]:
            title = item.find('title').text
            pub_date = item.find('pubDate').text
            articles.append(f"- {pub_date}: {title}")

        if not articles:
            return f"No recent news found for exactly '{search_term}'."

        return "\n".join(articles)

    except Exception as e:
        print(f"  [!] Failed to fetch news for {ticker}: {e}")
        return "No recent news found."

def analyze_stock_with_rag(ticker, news_context, strategy_name):
    """
    Acts as the AI CIO. Analyases the news context and makes a final verdict.
    """
    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

    system_prompt = """
    You are an elite Chief Investment Officer (CIO) for an Indian quantitative fund. 
    Your quantitative systems have already verified that this stock has excellent fundamentals and is in a perfect technical setup.
    
    Your ONLY job now is to review the recent news headlines for this company to act as a risk manager.
    
    Output a short, punchy analysis with three sections:
    1. 📰 NEWS SUMMARY: One sentence summarizing the current narrative.
    2. 🚩 RED FLAGS / 🚀 CATALYSTS: Bullet points of any major risks (e.g., SEBI probes, bad earnings) or positive catalysts.
    3. ⚖️ FINAL VERDICT: Given the user's trading style, state "STRONG BUY", "HOLD", or "REJECT" with a one-sentence justification.
    """

    user_prompt = f"""
    Stock Ticker: {ticker}
    User's Trading Style: {strategy_name}
    
    Recent News:
    {news_context}
    
    Analyze the above and give me your verdict.
    """

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3 # Slight creativity allowed for analysis, but still focused
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        return f"Analysis failed: {e}"

# --- Main Execution ---
if __name__ == "__main__":
    print("--- 🧠 AI Chief Investment Officer (RAG Pipeline) ---")

    input_file = "step3_ai_shortlist.csv"

    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found. Please run the strategy engine first.")
    else:
        # Load the winning tickers
        df = pd.read_csv(input_file)
        winning_tickers = df['Ticker'].tolist()

        if not winning_tickers:
            print("No winning tickers to analyze.")
        else:
            # We need the user's strategy context to make the final decision
            user_strategy = input("\nRemind the CIO of your trading style (e.g., 'Momentum Breakout' or 'Value Dip'): ")

            print(f"\nAnalyzing {len(winning_tickers)} winning stocks...\n")
            print("="*60)

            for ticker in winning_tickers:
                print(f"\n🔍 Processing: {ticker}")

                # 1. Retrieval
                news_text = fetch_google_news(ticker, num_articles=7)

                # 2. Generation
                if "No recent news" not in news_text:
                    analysis = analyze_stock_with_rag(ticker, news_text, user_strategy)
                    print("\n" + analysis)
                else:
                    print("\n⚠️ Skipping analysis due to lack of news data.")

                print("\n" + "="*60)