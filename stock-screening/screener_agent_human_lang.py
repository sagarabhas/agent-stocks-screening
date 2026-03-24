import os
import time
import pandas as pd
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright
from groq import Groq
from dotenv import load_dotenv
from io import StringIO

# --- AI AGENT TRANSLATOR ---
def generate_screener_syntax(natural_language_strategy):
    """
    Acts as the AI Translator. Takes natural language and returns exact Screener.in syntax.
    """
    print(f"\n[AI Agent] Translating your strategy: '{natural_language_strategy}'")

    # Initialize Groq client
    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

    system_prompt = """
    You are an expert quantitative analyst for the Indian stock market. 
    Your ONLY job is to translate the user's natural language trading strategy into the exact query syntax used by Screener.in.
    
    RULES:
    1. Use ONLY the allowed variables listed below. DO NOT invent your own.
    2. Output ONLY the query string. No Markdown formatting, no code blocks, no explanations.
    3. Use standard operators: >, <, =, AND, OR.
    
    ALLOWED VARIABLES (Exact spelling required):
    - Market Capitalization
    - Price to Earning
    - Return on capital employed
    - Debt to equity
    - Promoter holding
    - Sales growth 3Years
    - Profit growth 3Years
    - Current price
    - Dividend yield
    - PEG Ratio
    
    EXAMPLES:
    User: "Find me mid-cap companies with good return on capital and low debt."
    AI: Market Capitalization > 5000 AND Market Capitalization < 20000 AND Return on capital employed > 20 AND Debt to equity < 0.2
    
    User: "Value stocks under 500 rupees with high promoter holding."
    AI: Price to Earning < 15 AND Current price < 500 AND Promoter holding > 50
    """

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": natural_language_strategy}
            ],
            temperature=0.0 # Strict zero temperature for deterministic output
        )

        raw_query = response.choices[0].message.content.strip()
        # Clean up any accidental markdown backticks the LLM might generate
        clean_query = raw_query.replace("```python", "").replace("```", "").replace("`", "").strip()

        print(f"[AI Agent] Generated Screener Syntax: {clean_query}")
        return clean_query

    except Exception as e:
        print(f"[!] AI Generation Failed: {e}")
        return None

# --- PLAYWRIGHT SCRAPER ---
def run_screener_query(query_string):
    print(f"\nExecuting Query on Screener.in: {query_string}")

    with sync_playwright() as p:
        # Launch browser (set headless=True if you don't want to see the browser pop up)
        browser = p.chromium.launch(
            headless=True,
            args=[
                "--no-sandbox",
                "--disable-setuid-sandbox",
                "--disable-dev-shm-usage",
                "--disable-gpu"
            ]
        )
        context = browser.new_context()
        page = context.new_page()

        try:
            # 1. Login to Screener.in using Environment Variables
            print("Logging in to Screener.in...")
            page.goto("https://www.screener.in/login/")

            email = os.getenv("SCREENER_EMAIL")
            password = os.getenv("SCREENER_PASSWORD")

            if not email or not password:
                print("Error: SCREENER_EMAIL or SCREENER_PASSWORD not found in environment variables.")
                return None

            page.fill("input[name='username']", email)
            page.fill("input[name='password']", password)
            page.click("button[type='submit']")
            page.wait_for_load_state("networkidle")

            # 2. Navigate to the Raw Query page & execute
            print("Running query...")
            page.goto("https://www.screener.in/screen/raw/")
            page.fill("textarea[name='query']", query_string)
            page.click("button:has-text('Run this query')")

            all_dfs = []
            page_number = 1

            # 3. Pagination Loop
            while True:
                print(f"Scraping Page {page_number}...")

                # Wait for the table to load
                page.wait_for_selector("table.data-table", timeout=10000)
                html = page.content()

                # Parse the HTML table
                soup = BeautifulSoup(html, "html.parser")
                table = soup.find("table", class_="data-table")
                df = pd.read_html(StringIO(str(table)))[0]

                # Clean up S.No. column immediately
                if 'S.No.' in df.columns:
                    df = df.drop(columns=['S.No.'])

                all_dfs.append(df)

                # Check for the "Next" pagination button
                next_button = page.locator("a", has_text="Next").first

                if next_button.is_visible():
                    next_button.click()
                    page.wait_for_load_state("networkidle")
                    page.wait_for_timeout(1000) # 1-second buffer to allow DOM to re-render
                    page_number += 1
                else:
                    print("Reached the last page.")
                    break

            # 4. Combine all pages into a single DataFrame
            final_df = pd.concat(all_dfs, ignore_index=True)

            # 5. Clean Data: Remove duplicate rows and Screener's 'Median'/'Average' summary rows
            final_df = final_df[~final_df['Name'].astype(str).str.contains("Median|Average", na=False)]
            final_df = final_df.drop_duplicates(subset=['Name']).reset_index(drop=True)

            return final_df

        except Exception as e:
            print(f"Error during execution: {e}")
            return None

        finally:
            browser.close()

# --- MAIN EXECUTION BLOCK ---
if __name__ == "__main__":
    # Load environment variables from the .env file
    load_dotenv()

    print("--- 🤖 Agentic Fundamental Screener ---")

    # Take natural language input from the user
    user_input = input("\nEnter your fundamental strategy in plain English (e.g., 'High growth midcaps with zero debt'): \n> ").strip()

    if not user_input:
        print("Query cannot be empty. Exiting.")
    else:
        # 1. AI translates English to Screener syntax
        ai_generated_syntax = generate_screener_syntax(user_input)

        if ai_generated_syntax:
            # 2. Pass the AI syntax to the Web Scraper
            results_df = run_screener_query(ai_generated_syntax)

            if results_df is not None and not results_df.empty:
                total_stocks = len(results_df)
                print(f"\nSuccess! Found {total_stocks} stocks.")

                # Save the result to a CSV file
                csv_filename = "screener_results.csv"
                results_df.to_csv(csv_filename, index=False)

                print(f"Data successfully saved to: {os.path.abspath(csv_filename)}")
                print("\nHead of data:")
                print(results_df.head())
            else:
                print("\nNo stocks found or an error occurred during scraping.")