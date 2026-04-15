import streamlit as st
import pandas as pd
import os, sys, subprocess
import asyncio
from dotenv import load_dotenv

from ai_engine import generate_master_strategy
from backtest_engine import run_vectorized_backtest, analyze_backtest_with_ai, run_grid_search_optimization, optimize_strategy_with_ai
from notifier import send_telegram_alert
from macro_engine import get_market_regime
from alt_data_engine import check_smart_money
from sector_engine import get_sector_performance


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

# ==========================================
# ONBOARDING: QUICK START GUIDE
# ==========================================
with st.expander("🚀 Quick Start Guide: How to use this Quant Terminal", expanded=False):
    st.markdown("""
    Welcome to your Systematic Trading Terminal. This tool removes human emotion from investing by using strict quantitative math and AI. 
    
    ### 📅 Your Daily 3:00 PM Routine:
    * **Step 1: Check the Weather (Tab 0):** Look at the **Macro Traffic Light**. If the market is Red (Bear), do not trade. If it is Green (Bull), note the top-performing sectors in the Heatmap.
    * **Step 2: Ask the AI (Tab 0):** Type a plain-English idea into the **Agentic Idea Generator** (e.g., *"Find me high-growth midcaps in the Auto sector pulling back to their 50-day moving average"*). 
    * **Step 3: Fundamental Screen (Tab 1):** Paste the AI's fundamental code here and click Run. This filters out the garbage and leaves you with fundamentally elite companies.
    * **Step 4: Technical Filter (Tab 2):** Paste the AI's technical code here and click Run. This checks the live charts to ensure you are buying at the exact right mathematical moment.
    * **Step 5: Verify & Size Your Bet (Tabs 3, 4, & 5):** Check if Institutions are buying (Tab 3) and ensure no Earnings are due tomorrow (Tab 4). Finally, calculate your risk in **Tab 5**.
    
    ---
    
    ### 🧠 Demystifying Tab 5 (Risk & Backtest)
    Tab 5 is the most important tab in this terminal. It tells you if your strategy actually works, and exactly how many shares you should buy. Here is how to read it:

    * **1. Win Rate:** If this is 40%, it means out of 100 trades, you will lose money on 60 of them. *This is normal in trend following!* As long as your winning trades are much bigger than your losing trades, you make money.
    * **2. Sharpe Ratio:** This measures how "bumpy" the ride is. 
        * Below 1.0 = Too risky for the reward.
        * 1.0 to 1.5 = Good, solid strategy.
        * Above 1.5 = Excellent.
    * **3. Maximum Drawdown:** The terrifying "worst-case scenario." If this says -25%, it means at some point in the past, this strategy lost 25% of its value before recovering. Ask yourself: *Can my stomach handle that?*
    * **4. The Kelly Criterion (%):**  This is the magic number. It uses a mathematical formula to tell you exactly what percentage of your total account you should invest in this one stock to maximize growth without going broke. 
        * **The Golden Rule:** The math is aggressive. Always cut the Kelly % in half (called "Half-Kelly") to protect yourself from unpredictable market crashes. If Kelly says 10%, you only risk 5%.
    """)

# --- Create the Tabs ---
tab_macro, tab1, tab2, tab_smart_money, tab3, tab4 = st.tabs([
    "🚦 0. Command Center",
    "1. Fundamental Screen",
    "2. Technical Engine",
    "3. Smart Money",
    "4. AI CIO (News)",
    "5. Risk & Backtest"
])

# ==========================================
# TAB 0: MACRO COMMAND CENTER
# ==========================================
with tab_macro:
    st.header("Global Macro & AI Strategy")
    st.markdown("Assess the market regime, follow the sector money flow, and generate your strategy.")

    # 1. Agentic Idea Generator
    with st.expander("🤖 Agentic Idea Generator: Chat-to-Portfolio", expanded=True):
        user_idea = st.text_input("Describe your trading idea in plain English:")
        if st.button("Generate Strategy"):
            with st.spinner("AI is translating your idea into Quant logic..."):
                strategy_dict = generate_master_strategy(user_idea)

                if "error" not in strategy_dict:
                    st.session_state.auto_fundamental = strategy_dict.get("fundamental", "")
                    st.session_state.auto_technical = strategy_dict.get("technical", "")
                    st.success("Strategy Generated! Proceed to Tab 1.")
                    st.json(strategy_dict)
                else:
                    st.error(f"AI Error: {strategy_dict['error']}")

    # 2. Macro Traffic Light
    st.markdown("### 🚦 Macro Traffic Light: Market Regime")
    regime_data = get_market_regime()
    if "error" not in regime_data:
        st.info(f"**Current Regime:** {regime_data['regime']}  \n**Systematic Action:** {regime_data['action']}")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Nifty 50", regime_data['nifty'])
        col2.metric("India VIX", regime_data['vix'])
        col3.metric("Nifty 50-Day SMA", regime_data['sma_50'])
        col4.metric("Nifty 200-Day SMA", regime_data['sma_200'])
    else:
        st.warning(f"Could not load Market Regime data. Error: {regime_data['error']}")

    st.markdown("---")

    # 3. Sector Rotation Heatmap
    with st.expander("📊 Macro Dashboard: Sector Rotation Heatmap", expanded=True): # Set to True so it's open by default here
        with st.spinner("Calculating sector momentum..."):
            sector_df = get_sector_performance()
            if not sector_df.empty:
                styled_df = sector_df.style.background_gradient(cmap="RdYlGn", vmin=-10, vmax=10)
                st.dataframe(styled_df, use_container_width=True)
                top_sector = sector_df.index[0]
                st.success(f"🔥 **Hot Sector Alert:** {top_sector} is leading the market.")
            else:
                st.warning("Sector data is currently unavailable.")

    st.markdown("---")

# ==========================================
# TAB 1: FUNDAMENTAL SCREENING
# ==========================================
with tab1:
    st.header("Step 1: Fundamental Screening")
    #Adding this comment for AI to delete

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
# TAB 3: SMART MONEY TRACKER (ALT DATA)
# ==========================================
with tab_smart_money:
    st.header("Step 3: Smart Money & Insider Tracking")
    st.markdown("Verify if Institutions and Insiders are buying the stocks your math just selected.")

    # Check if we have winners from Tab 2
    if 'technical_winners' not in st.session_state or not st.session_state.technical_winners:
        st.warning("⚠️ Please run the Technical Engine (Tab 2) first to generate a watchlist.")
    else:
        st.success(f"Found {len(st.session_state.technical_winners)} stocks ready for Smart Money analysis.")

        if st.button("🔍 Scan for Insider & Institutional Buying", type="primary"):
            with st.spinner("Querying block deals and insider holdings..."):

                # Extract the list of winning tickers from session state
                # (Assuming st.session_state.technical_winners is a list of strings like ['RELIANCE.NS', 'TCS.NS'])
                winners_list = st.session_state.technical_winners

                # Run the Alt Data Engine
                smart_money_df = check_smart_money(winners_list)

                if not smart_money_df.empty:
                    st.dataframe(smart_money_df, use_container_width=True)

                    # Highlight strong institutional backing
                    st.info("💡 **Quant Rule:** If Institutional Ownership is high (>40%) and Insider Activity is detected, the mathematical breakout has a significantly higher probability of success.")
                else:
                    st.error("Failed to retrieve Alternative Data.")


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
    manual_ticker = st.text_input("Enter Ticker (e.g., RELIANCE.NS):")
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