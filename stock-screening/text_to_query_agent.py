import os
from groq import Groq
import pandas as pd
from dotenv import load_dotenv


load_dotenv()

from strategy_engine import run_strategy_engine


# ... (Keep your existing build_feature_store and run_strategy_engine functions here) ...

def generate_pandas_query(user_strategy):
    """
    Acts as the AI Agent using Groq. Takes natural language and returns a pandas query string.
    """
    print(f"\n[AI Agent] Translating your strategy: '{user_strategy}'")

    # Initialize the Groq client
    # Make sure you have set your environment variable: export GROQ_API_KEY="your_key"
    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

    system_prompt = """
    You are an expert quantitative developer. Your job is to translate a user's natural language trading strategy into a valid Python pandas `df.query()` string.

    Available Columns (DO NOT invent others):
    Close, Open, High, Low, Volume
    SMA_20, SMA_50, SMA_150, SMA_200
    RSI_14, MACD, MACD_Histogram
    High_52week, Low_52week, Volume_SMA_20

    Rules:
    1. Use standard pandas query syntax (e.g., 'and', 'or', '>', '<', '==').
    2. Output ONLY the raw query string.
    3. DO NOT wrap the output in markdown formatting.
    4. CRITICAL: Ensure the mathematical logic does not contradict itself.

    EXAMPLES OF GOOD LOGIC:
    - User: "Trend pullback" -> AI: `Close > SMA_200 and RSI_14 < 40 and Close < SMA_20`
    - User: "High volume breakout" -> AI: `Close >= (High_52week * 0.95) and Volume > (Volume_SMA_20 * 1.5)`
    - User: "Value dip" -> AI: `Close < SMA_50 and RSI_14 < 30 and Close > Low_52week`
    """

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile", # Groq's fast Llama 3 model (perfect for logic)
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Write a query for this strategy: {user_strategy}"}
            ],
            temperature=0.0 # Strict zero temperature for deterministic, code-like output
        )

        # Clean the output just in case the LLM tries to sneak in backticks
        raw_query = response.choices[0].message.content.strip()
        clean_query = raw_query.replace("```python", "").replace("```", "").replace("`", "").strip()

        print(f"[AI Agent] Generated Math Logic: {clean_query}")
        return clean_query

    except Exception as e:
        print(f"[!] AI Generation Failed: {e}")
        return None

# --- New Main Execution ---
if __name__ == "__main__":
    print("--- Welcome to the Agentic Strategy Engine (Powered by Groq) ---")

    # 1. Take natural language input from the user
    user_input = input("\nEnter your trading style or a Guru's name (e.g., 'Warren Buffett value dip' or 'High volume breakouts'): \n> ")

    # 2. Let the Agent translate it into math
    ai_generated_query = generate_pandas_query(user_input)

    if ai_generated_query:
        # 3. Pass the AI's math into your deterministic engine
        # (Assuming you still have the test_query setup from earlier)
        winners = run_strategy_engine(ai_generated_query)

        print(f"\n--- Scan Complete ---")
        print(f"Stocks matching your '{user_input}' strategy: {len(winners)}")

        if winners:
            print(f"Final Watchlist: {winners}")
            pd.DataFrame({"Ticker": winners}).to_csv("step3_ai_shortlist.csv", index=False)
            print("Saved to step3_ai_shortlist.csv")