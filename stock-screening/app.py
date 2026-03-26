import streamlit as st
import pandas as pd
import os, sys, subprocess
import asyncio
from dotenv import load_dotenv

from ai_engine import generate_master_strategy
from backtest_engine import run_vectorized_backtest, analyze_backtest_with_ai, run_grid_search_optimization, optimize_strategy_with_ai
from notifier import send_telegram_alert


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
tab1, tab2, tab3, tab4 = st.tabs(["📊 Fundamentals", "📈 Technicals", "🤖 AI Analyst", "🧪 Backtesting"])

# ==========================================
# 🤖 MASTER AI STRATEGIST (Place right above Tab 1)
# ==========================================
with st.expander("🤖 Agentic Idea Generator: Chat-to-Portfolio", expanded=False):
    st.markdown("Type a trading idea in plain English. The AI will instantly generate the Fundamental Screener and Technical Math for you.")

    user_idea = st.text_area("What is your trading strategy?", placeholder="e.g., 'Find me highly liquid dividend-paying mega-caps that are currently crashing and severely oversold.'")

    if st.button("✨ Generate Strategy", type="primary"):
        with st.spinner("Translating idea into Quantitative Math..."):

            # Call the LLM to generate the JSON
            strategy_dict = generate_master_strategy(user_idea)

            if "error" not in strategy_dict:
                # Save the exact math to memory
                st.session_state.auto_fundamental = strategy_dict.get("fundamental", "")
                st.session_state.auto_technical = strategy_dict.get("technical", "")

                st.success("✅ Strategy Generated! The inputs in Tab 1 and Tab 2 have been auto-filled.")
                st.info(f"**AI Rationale:** {strategy_dict.get('explanation', '')}")

                st.code(f"Fundamental: {st.session_state.auto_fundamental}")
                st.code(f"Technical: {st.session_state.auto_technical}")
            else:
                st.error(f"Failed to generate strategy: {strategy_dict['error']}")

# ==========================================
# TAB 1: FUNDAMENTAL SCREENING
# ==========================================
with tab1:
    st.header("Step 1: Fundamental Screening")

    # 1. Grab the AI-generated query if it exists
    default_fund = st.session_state.get("auto_fundamental", "")
    user_strategy = st.text_area("Describe your fundamental strategy (or use auto-generated logic):", value=default_fund, placeholder="e.g., High growth midcaps with zero debt...")

    if st.button("Run Fundamental Screen", type="primary"):
        st.session_state.fund_strategy_text = user_strategy # Save for the RAG prompt later

        with st.status("Running Fundamental Pipeline...", expanded=True) as status:

            # 2. SMART BYPASS: Don't re-translate if it's already raw Screener math!
            if "AND" in user_strategy or ">" in user_strategy or "<" in user_strategy:
                st.write("Raw syntax detected. Skipping AI translation...")
                syntax = user_strategy
            else:
                st.write("Translating English to Screener syntax...")
                syntax = generate_screener_syntax(user_strategy)

            if syntax:
                st.write(f"Scraping Screener.in for: {syntax}")
                df = run_screener_query(syntax)

                if df is not None and not df.empty:
                    st.session_state.fundamental_df = df
                    status.update(label=f"✅ Found {len(df)} stocks!", state="complete")
                else:
                    status.update(label="❌ No stocks found or scraper blocked.", state="error")

                    # --- TEXT DEBUGGER ---
                    if os.path.exists("cloud_error.txt"):
                        with open("cloud_error.txt", "r") as f:
                            st.error(f"🛑 THE BOT CRASHED BECAUSE: {f.read()}")

                    # --- VISUAL DEBUGGER ---
                    if os.path.exists("debug_cloud_error.png"):
                        import time
                        st.image("debug_cloud_error.png", width=800, caption=f"Screenshot at {time.time()}")

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

        # 1. Grab the AI-generated query if it exists
        default_tech = st.session_state.get("auto_technical", "")
        # Changed to text_area so you can easily see long generated queries
        tech_strategy = st.text_area("Enter your technical strategy (or use auto-generated Pandas logic):", value=default_tech, placeholder="e.g., Close > SMA_50 and RSI_14 > 50")

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

                # Save to CSV
                st.session_state.fundamental_df['Ticker'] = tickers
                st.session_state.fundamental_df.to_csv("step1_with_tickers.csv", index=False)
                st.write(f"✅ Resolved {len([t for t in tickers if t])} valid tickers.")

                # 2. Download Data
                with st.spinner("📥 **2. Downloading 1-Year OHLCV Data...**"):
                    fetch_ohlcv_data("step1_with_tickers.csv", "stock_data")
                    st.write("✅ Data downloaded to local Data Lake.")

                # 3. AI Math Translation & SMART BYPASS
                with st.spinner("🧠 **3. Processing Technical Strategy...**"):

                    # Don't re-translate if it's already Pandas math!
                    if ">" in tech_strategy or "<" in tech_strategy or "==" in tech_strategy:
                        st.write("Raw Pandas logic detected. Skipping AI translation...")
                        pandas_query = tech_strategy
                    else:
                        st.write("Translating English to Pandas logic...")
                        pandas_query = generate_pandas_query(tech_strategy)

                    st.session_state.tech_query = pandas_query
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

        # ==========================================
        # PHASE 1: DO THE WORK AND SAVE TO MEMORY
        # ==========================================
        if st.button("Fetch News & Analyze", type="primary"):
            # Loop through the final winners
            for ticker in st.session_state.technical_winners:
                with st.spinner(f"Scraping news & generating AI report for {ticker}..."):

                    # 1. Fetch News
                    news_context = fetch_google_news(ticker, num_articles=5)

                    # 2. Generate RAG Analysis
                    full_context = f"Fundamental: {st.session_state.fund_strategy_text} | Technical: {tech_strategy}"
                    analysis = analyze_stock_with_rag(ticker, news_context, full_context)

                    # 3. SAVE IT TO MEMORY (Crucial Step!)
                    st.session_state[f"rag_analysis_{ticker}"] = analysis
                    st.session_state[f"raw_news_{ticker}"] = news_context

        # ==========================================
        # PHASE 2: INDEPENDENT DISPLAY & TELEGRAM BUTTON
        # ==========================================
        # Because this is outside the "Fetch" button, it survives page refreshes!
        for ticker in st.session_state.technical_winners:

            # Check if memory exists for this ticker
            if st.session_state.get(f"rag_analysis_{ticker}"):

                with st.expander(f"📊 Analysis for {ticker}", expanded=True):
                    # Display the saved output
                    st.markdown(st.session_state[f"rag_analysis_{ticker}"])

                    # --- NEW INDEPENDENT TELEGRAM BUTTON ---
                    if st.button(f"📱 Send {ticker} Alert to Telegram", key=f"tg_{ticker}"):
                        with st.spinner("Pushing to your phone..."):

                            # Construct the message format
                            tg_message = f"🚨 **QUANT ALERT: {ticker}** 🚨\n\n"
                            tg_message += f"*AI Analyst Summary:*\n{st.session_state[f'rag_analysis_{ticker}']}\n\n"
                            tg_message += f"Strategy: {tech_strategy}"

                            result = send_telegram_alert(tg_message)

                            if result == "Success":
                                st.success("✅ Alert sent to your phone!")
                            else:
                                st.error(result)

                    with st.popover("View Raw News Sources"):
                        st.text(st.session_state[f"raw_news_{ticker}"])

# ==========================================
# TAB 4: VECTORIZED BACKTESTING ENGINE
# ==========================================
with tab4:
    st.header("Step 4: The Reality Check")

    # Mode Toggle
    mode = st.radio("Select Engine Mode:", ["Single Strategy Test", "Parameter Optimizer (Grid Search)"], horizontal=True)
    st.divider()

    test_tickers = st.session_state.get("technical_winners", [])
    custom_query = st.session_state.get("tech_query", None)

    st.warning("⚠️ Run Tab 2 to backtest your winners, or enter a manual ticker below:")
    manual_ticker = st.text_input("Enter Ticker (e.g., RELIANCE.NS):", "RELIANCE.NS")
    if manual_ticker:
        test_tickers = [manual_ticker]

    if test_tickers:
        st.success(f"Target Tickers: {', '.join(test_tickers)}")

        if mode == "Single Strategy Test":
            # --- EXISTING SINGLE STRATEGY CODE ---
            if custom_query:
                st.info(f"**Loaded Custom AI Strategy:** `{custom_query}`")
                strategy_options = ["Custom AI Strategy", "SMA Crossover (50 vs 200)", "RSI Mean Reversion (<30 Buy)"]
            else:
                strategy_options = ["SMA Crossover (50 vs 200)", "RSI Mean Reversion (<30 Buy)"]

            col1, col2, col3 = st.columns(3)
            with col1:
                years = st.slider("Backtest Period (Years)", 1, 10, 4, key="single_yrs")
                stop_loss = st.number_input("Hard Stop-Loss % (0 to disable)", min_value=0.0, max_value=100.0, value=8.0, step=1.0)
            with col2:
                strategy_type = st.selectbox("Strategy to Test", strategy_options)
                take_profit = st.number_input("Take-Profit % (0 to disable)", min_value=0.0, max_value=500.0, value=25.0, step=1.0)
            with col3:
                # NEW: Walk-Forward Toggle
                st.markdown("**Walk-Forward Validation**")
                oos_split = st.slider("Out-of-Sample Split %", min_value=0, max_value=50, value=25, step=5,
                                      help="Reserves the final X% of the timeline as a blind test to prevent curve-fitting.")

            # --- THE SINGLE, MERGED BACKTEST BUTTON ---
            if st.button("🚀 Run Vectorized Backtest", type="primary", key="run_single_backtest_btn"):
                with st.spinner(f"Crunching {years} years of historical data..."):

                    # Pass ALL parameters, including the new oos_split!
                    curves_df, metrics_df = run_vectorized_backtest(
                        test_tickers, years, strategy_type, custom_query, stop_loss / 100, take_profit / 100, oos_split / 100
                    )

                    # Display the charts and metrics
                    if not curves_df.empty:
                        st.subheader("Equity Curve: Strategy vs Buy & Hold")
                        st.line_chart(curves_df)
                        st.subheader("Performance Metrics")
                        st.dataframe(metrics_df, use_container_width=True)
                        st.session_state.backtest_metrics = metrics_df
                    else:
                        st.error("🛑 CRITICAL FAILURE: The backtest engine crashed or returned empty data.")

            if st.session_state.get("backtest_metrics") is not None and not st.session_state.backtest_metrics.empty:
                if st.button("🤖 Ask AI to Analyze these results"):
                    with st.spinner("Consulting Chief Quant Officer..."):
                        strat_name = custom_query if strategy_type == "Custom AI Strategy" else strategy_type
                        analysis = analyze_backtest_with_ai(st.session_state.backtest_metrics, test_tickers, strat_name)
                        st.info(analysis)

                # --- AGENTIC FEEDBACK LOOP ---
                if custom_query and st.session_state.get("backtest_metrics") is not None and not st.session_state.backtest_metrics.empty:
                    st.divider()
                    st.markdown("### 🛠️ Agentic Strategy Optimizer")
                    st.markdown("Is the drawdown too high? Let the AI rewrite your math to make the strategy safer.")

                    if st.button("🪄 Auto-Optimize Custom Query", type="secondary"):
                        with st.spinner("AI Risk Manager is tightening your risk parameters..."):
                            new_query = optimize_strategy_with_ai(
                                custom_query,
                                st.session_state.backtest_metrics,
                                test_tickers
                            )

                            if not new_query.startswith("Error"):
                                # Overwrite the old strategy with the new, safer one
                                st.session_state.tech_query = new_query
                                st.success(f"Strategy Optimized! New Logic: `{new_query}`")

                                # Force Streamlit to refresh the UI immediately
                                st.rerun()
                            else:
                                st.error(new_query)

        elif mode == "Parameter Optimizer (Grid Search)":
            # --- NEW GRID SEARCH CODE ---
            st.markdown("Find the mathematically optimal SMA combination for your first target ticker.")
            col1, col2, col3 = st.columns(3)
            with col1:
                opt_years = st.slider("Backtest Period (Years)", 1, 10, 3, key="opt_yrs")
            with col2:
                st.markdown("**Fast SMA Range**")
                fast_start = st.number_input("Start", value=10, step=5)
                fast_end = st.number_input("End", value=50, step=5)
                fast_step = st.number_input("Step", value=5, key="f_step")
            with col3:
                st.markdown("**Slow SMA Range**")
                slow_start = st.number_input("Start", value=50, step=10)
                slow_end = st.number_input("End", value=200, step=10)
                slow_step = st.number_input("Step", value=10, key="s_step")

            if st.button("🔍 Run Grid Search", type="primary"):
                target_ticker = test_tickers[0] # Optimize one stock at a time
                with st.spinner(f"Simulating hundreds of variations for {target_ticker}..."):
                    opt_df = run_grid_search_optimization(
                        target_ticker, opt_years,
                        (fast_start, fast_end, fast_step),
                        (slow_start, slow_end, slow_step),
                        8.0 / 100, # Hardcoded default SL for optimizer
                        25.0 / 100 # Hardcoded default TP for optimizer
                    )

                    if not opt_df.empty:
                        st.success(f"✅ Optimization complete! Displaying top 10 combinations.")
                        st.dataframe(opt_df.head(10), use_container_width=True)
                    else:
                        st.error("Optimizer failed. Check inputs.")