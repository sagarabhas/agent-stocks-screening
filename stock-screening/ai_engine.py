import json
import os
from groq import Groq

def generate_master_strategy(user_idea):
    """
    Translates a plain English trading idea into both a Screener.in fundamental query
    and a Pandas technical query, returning them as a parsed dictionary.
    """
    try:
        client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

        system_prompt = """
        You are an elite Quantitative Architect. The user will give you a high-level trading idea. 
        You must translate this idea into TWO separate queries:
        
        1. FUNDAMENTAL QUERY (For Screener.in): 
        Use exact variables like 'Market Capitalization', 'Return on equity', 'Dividend yield', 'Quarterly profit growth', 'Average volume'. Use 'AND' to separate.
        
        2. TECHNICAL QUERY (For Pandas Backtester):
        Available variables: Close, Open, High, Low, Volume, SMA_20, SMA_50, SMA_150, SMA_200, RSI_14, MACD, MACD_Histogram, High_52week, Low_52week, Volume_SMA_20.
        Use standard Python math (e.g., `Close > SMA_200 and RSI_14 < 30`).
        
        OUTPUT FORMAT: 
        You MUST output ONLY a valid JSON object with exactly three keys: "fundamental", "technical", and "explanation". Do not include markdown blocks or any other text.
        
        Example Output:
        {
            "fundamental": "Market Capitalization > 10000 AND Return on equity > 15",
            "technical": "Close > SMA_200 and RSI_14 < 40",
            "explanation": "This targets high-quality large caps experiencing a short-term oversold pullback."
        }
        """

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_idea}
            ],
            temperature=0.2, # Low temperature for strict JSON formatting
            response_format={"type": "json_object"} # Forces Groq to return clean JSON
        )

        # Parse the JSON string into a Python dictionary
        return json.loads(response.choices[0].message.content)

    except Exception as e:
        return {"error": str(e)}