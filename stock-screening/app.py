import streamlit as st
import pandas as pd
import os, sys, subprocess
import asyncio
from dotenv import load_dotenv
from backtest_engine import run_vectorized_backtest, analyze_backtest_with_ai, run_grid_search_optimization


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

                    # --- TEXT DEBUGGER ---
                    if os.path.exists("cloud_error.txt"):
                        with open("cloud_error.txt", "r") as f:
                            st.error(f"🛑 THE BOT CRASHED BECAUSE: {f.read()}")
                        # os.remove("cloud_error.txt") # Clean up

                    # --- VISUAL DEBUGGER ---
                    if os.path.exists("debug_cloud_error.png"):
                        # We use a timestamp to stop Streamlit from showing you a cached white image!
                        import time
                        st.image("debug_cloud_error.png", width=800, caption=f"Screenshot at {time.time()}")

    if st.session_state.fundamental_df is not None:
        st.dataframe(st.session_state.fundamental_df, width="stretch")

# ==========================================
# TAB 1: FUNDAMENTAL SCREENING (Hybrid Upload)
# ==========================================
# with tab1:
#     st.header("Step 1: Fundamental Screening")
#     st.markdown("""
#     **To bypass cloud IP blocks, we use a hybrid ingestion method.**
#     1. Go to [Screener.in](https://www.screener.in/screen/raw/) and run your fundamental query.
#     2. Click the **'Export to Excel/CSV'** button.
#     3. Upload that file here.
#     """)
#
#     # Drag and drop file uploader
#     uploaded_file = st.file_uploader("Upload your Screener.in CSV/Excel file", type=["csv", "xlsx"])
#
#     # Also save the strategy text so Tab 3 (RAG) knows what you are doing
#     user_strategy = st.text_input("For the AI's context, briefly describe the fundamental strategy you used:")
#
#     if uploaded_file is not None:
#         try:
#             # Read the uploaded file (handles both CSV and Excel)
#             if uploaded_file.name.endswith('.csv'):
#                 df = pd.read_csv(uploaded_file)
#             else:
#                 df = pd.read_excel(uploaded_file)
#
#             # Clean up the Screener.in specific formatting
#             if 'S.No.' in df.columns:
#                 df = df.drop(columns=['S.No.'])
#
#             # Save to session state so Tab 2 can grab it!
#             st.session_state.fundamental_df = df
#             st.session_state.fund_strategy_text = user_strategy
#
#             st.success(f"✅ Successfully ingested {len(df)} stocks! You can now move to Tab 2.")
#             st.dataframe(df, width="stretch")
#
#         except Exception as e:
#             st.error(f"Error reading file: {e}")

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

    if not test_tickers:
        st.warning("⚠️ Run Tab 2 to backtest your winners, or enter a manual ticker below:")
        manual_ticker = st.text_input("Enter Ticker (e.g., RELIANCE.NS):", "RELIANCE.NS")
        if manual_ticker:
            test_tickers = [manual_ticker]

    if test_tickers:
        st.success(f"Target Tickers: {', '.join(test_tickers)}")

        if mode == "Single Strategy Test":
            # --- EXISTING SINGLE STRATEGY CODE ---
            if custom_query:
                strategy_options = ["Custom AI Strategy", "SMA Crossover (50 vs 200)", "RSI Mean Reversion (<30 Buy)"]
            else:
                strategy_options = ["SMA Crossover (50 vs 200)", "RSI Mean Reversion (<30 Buy)"]

            col1, col2 = st.columns(2)
            with col1:
                years = st.slider("Backtest Period (Years)", 1, 10, 3, key="single_yrs")
                stop_loss = st.number_input("Hard Stop-Loss % (0 to disable)", min_value=0.0, max_value=100.0, value=8.0, step=1.0)
            with col2:
                strategy_type = st.selectbox("Strategy to Test", strategy_options)
                take_profit = st.number_input("Take-Profit % (0 to disable)", min_value=0.0, max_value=500.0, value=25.0, step=1.0)

            if st.button("🚀 Run Vectorized Backtest", type="primary"):
                with st.spinner(f"Crunching {years} years of historical data..."):
                    curves_df, metrics_df = run_vectorized_backtest(
                        test_tickers, years, strategy_type, custom_query, stop_loss / 100, take_profit / 100
                    )
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