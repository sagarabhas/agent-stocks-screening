from playwright.sync_api import sync_playwright
import pandas as pd
from bs4 import BeautifulSoup
import os

def run_screener_query(query_string):
    print(f"\nExecuting Query: {query_string}")

    with sync_playwright() as p:
        # Launch browser (set headless=True if you don't want to see the browser pop up)
        browser = p.chromium.launch(headless=False)
        context = browser.new_context()
        page = context.new_page()

        try:
            # 1. Login to Screener.in
            print("Logging in to Screener.in...")
            page.goto("https://www.screener.in/login/")
            page.fill("input[name='username']", "sagar.gndude@gmail.com") # UPDATE THIS
            page.fill("input[name='password']", "Qwerty@123")          # UPDATE THIS
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
                df = pd.read_html(str(table))[0]

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

# --- Main Execution Block ---
if __name__ == "__main__":
    # Take the query as user input from the terminal
    user_query = input("Enter your Screener.in query: ").strip()

    if not user_query:
        print("Query cannot be empty. Exiting.")
    else:
        # Run the scraping function
        results_df = run_screener_query(user_query)

        if results_df is not None and not results_df.empty:
            total_stocks = len(results_df)
            print(f"\nSuccess! Found {total_stocks} stocks.")

            # Save the result to a CSV file
            csv_filename = "screener_results.csv"
            results_df.to_csv(csv_filename, index=False)

            print(f"Data successfully saved to: {os.path.abspath(csv_filename)}")
        else:
            print("\nNo stocks found or an error occurred.")