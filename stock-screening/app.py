import streamlit as st
import pandas as pd
import os, sys, subprocess
import asyncio
from dotenv import load_dotenv

# --- PLAYWRIGHT CLOUD FIX (RUNS ONLY ONCE) ---
@st.cache_resource(show_spinner="Booting Cloud Browser Environment...")
def install_playwright():
    """Ensures Playwright is installed exactly once per server boot."""
    try:
        # Install the Chromium binary.
        subprocess.run([f"{sys.executable}", "-m", "playwright", "install", "chromium"], check=True)
        # REMOVED the install-deps line!
    except Exception as e:
        print(f"Failed to install Playwright: {e}")

install_playwright()

# --- WINDOWS ASYNCIO BUG FIX ---
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

# --- IMPORTS FROM YOUR BACKEND SCRIPTS ---
from screener_agent_human_lang import generate_screener_syntax, run_screener_query
from ticker_resolver import get_yahoo_ticker
from ohlcv_ingestion import fetch_ohlcv_data
from text_to_query_agent import generate_pandas_query
from strategy_engine import run_strategy_engine
from rag_analyst import fetch_google_news, analyze_stock_with_rag

# --- Page Configuration ---
st.set_page_config(page_title="AI Quant Trading Desk", page_icon="📈", layout="wide")
load_dotenv()

st.title("🤖 AI Quant Trading Desk")
st.markdown("Your end-to-end automated quantitative pipeline.")

# --- Session State Management ---
if 'fundamental_df' not in st.session_state:
    st.session_state.fundamental_df = None
if 'technical_winners' not in st.session_state:
    st.session_state.technical_winners = []
if 'fund_strategy_text' not in st.session_state:
    st.session_state.fund_strategy_text = ""

# --- Create the Tabs ---
tab1, tab2, tab3 = st.tabs(["1️⃣ Fundamental Screener", "2️⃣ Technical Engine", "3️⃣ AI CIO (RAG)"])

# ==========================================
# TAB 1: FUNDAMENTAL SCREENING
# ==========================================
with tab1:
    st.header("Step 1: Fundamental Screening")
    user_strategy = st.text_area("Describe your fundamental strategy:", placeholder="e.g., High growth midcaps with zero debt...")

    if st.button("Run Fundamental Screen", type="primary"):
        st.session_state.fund_strategy_text = user_strategy # Save for the RAG prompt later

        with st.status("Running Fundamental Pipeline...", expanded=True) as status:
            st.write("Translating strategy...")
            syntax = generate_screener_syntax(user_strategy)

            if syntax:
                st.write(f"Scraping Screener.in for: {syntax}")
                df = run_screener_query(syntax)

                if df is not None and not df.empty:
                    st.session_state.fundamental_df = df
                    status.update(label=f"✅ Found {len(df)} stocks!", state="complete")
                else:
                    status.update(label="❌ No stocks found or scraper blocked.", state="error")

                    # --- NEW VISUAL DEBUGGER ---
                    if os.path.exists("debug_cloud_error.png"):
                        st.error("⚠️ The cloud bot hit a roadblock. Here is exactly what the bot saw:")
                        st.image("debug_cloud_error.png", width=800)
                        os.remove("debug_cloud_error.png") # Clean up

    if st.session_state.fundamental_df is not None:
        st.dataframe(st.session_state.fundamental_df, width="stretch")

# ==========================================
# TAB 2: TECHNICAL ENGINE
# ==========================================
with tab2:
    st.header("Step 2: Technical & Momentum Engine")
    st.markdown("Filter the fundamental winners based on chart patterns and momentum.")

    if st.session_state.fundamental_df is None:
        st.warning("⚠️ Please run the Fundamental Screener in Tab 1 first.")
    else:
        st.success(f"Ready to run technicals on {len(st.session_state.fundamental_df)} stocks.")
        tech_strategy = st.text_input("Enter your technical strategy:", placeholder="e.g., Price above 50 SMA and RSI > 50")

        if st.button("Run Technical Engine", type="primary"):
            if not tech_strategy:
                st.warning("Please enter a technical strategy.")
            else:
                # 1. Resolve Tickers
                st.write("🔍 **1. Resolving Yahoo Finance Tickers...**")
                progress_bar = st.progress(0)
                tickers = []
                names = st.session_state.fundamental_df['Name'].tolist()

                for i, name in enumerate(names):
                    ticker = get_yahoo_ticker(name)
                    tickers.append(ticker)
                    progress_bar.progress((i + 1) / len(names))

                # Save to CSV so ingestion script can use it
                st.session_state.fundamental_df['Ticker'] = tickers
                st.session_state.fundamental_df.to_csv("step1_with_tickers.csv", index=False)
                st.write(f"✅ Resolved {len([t for t in tickers if t])} valid tickers.")

                # 2. Download Data
                with st.spinner("📥 **2. Downloading 1-Year OHLCV Data...**"):
                    fetch_ohlcv_data("step1_with_tickers.csv", "stock_data")
                    st.write("✅ Data downloaded to local Data Lake.")

                # 3. AI Math Translation
                with st.spinner("🧠 **3. Translating Technical Strategy...**"):
                    pandas_query = generate_pandas_query(tech_strategy)
                    st.code(pandas_query, language="python")

                # 4. Strategy Execution
                with st.spinner("⚙️ **4. Applying Mathematical Filters...**"):
                    if pandas_query:
                        winners = run_strategy_engine(pandas_query, "stock_data")
                        st.session_state.technical_winners = winners
                        st.success(f"✅ Technical Scan Complete! {len(winners)} stocks passed the test.")
                        st.write(winners)
                    else:
                        st.error("Failed to generate pandas query.")

# ==========================================
# TAB 3: AI CIO / RAG ANALYST
# ==========================================
with tab3:
    st.header("Step 3: AI Chief Investment Officer")
    st.markdown("Read the latest news for the final winning stocks to check for red flags.")

    if not st.session_state.technical_winners:
        st.warning("⚠️ Please run the Technical Engine in Tab 2 first. We need winning tickers!")
    else:
        st.success(f"Ready to analyze news for: {', '.join(st.session_state.technical_winners)}")

        if st.button("Fetch News & Analyze", type="primary"):
            # Loop through the final winners
            for ticker in st.session_state.technical_winners:
                with st.expander(f"📊 Analysis for {ticker}", expanded=True):

                    # 1. Fetch News
                    with st.spinner("Scraping Google News RSS..."):
                        news_context = fetch_google_news(ticker, num_articles=5)

                    # 2. Generate RAG Analysis
                    with st.spinner("Groq AI is analyzing the data..."):
                        # Combine fundamental and technical strategies for the prompt context
                        full_context = f"Fundamental: {st.session_state.fund_strategy_text} | Technical: {tech_strategy}"
                        analysis = analyze_stock_with_rag(ticker, news_context, full_context)

                    # Display the final output
                    st.markdown(analysis)

                    with st.popover("View Raw News Sources"):
                        st.text(news_context)