import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import yfinance as yf

# API Configuration
API_URL = "http://127.0.0.1:5000"  # Flask API base URL
NIFTY50_COMPANIES = [
    'RELIANCE.NS', 'TCS.NS', 'INFY.NS', 'HDFCBANK.NS',
    'ICICIBANK.NS', 'HINDUNILVR.NS', 'BAJFINANCE.NS',
    'SBIN.NS', 'KOTAKBANK.NS', 'BHARTIARTL.NS'
]

# Navigation
PAGES = ["Stock Dashboard", "Prediction Page"]

# App Title and Configuration
st.set_page_config(page_title="NIFTY50 Stock Analysis", layout="wide")
st.sidebar.title("Navigation")
selection = st.sidebar.radio("Go to", PAGES)

# --- PAGE 1: Stock Dashboard ---
if selection == "Stock Dashboard":
    st.title("📊 Company Fundamentals Dashboard")

    # Input field for the stock ticker
    # Input field for the stock ticker
    user_input = st.text_input("Enter a stock ticker (e.g., AAPL, MSFT, TSLA):")

    # Automatically append '.NS' if not already present
    if user_input:
        if not user_input.upper().endswith(".NS"):
            ticker = f"{user_input.upper()}.NS"
        else:
            ticker = user_input.upper()
    else:
        ticker = None


    if ticker:
        try:
            # Fetching data from Yahoo Finance
            stock = yf.Ticker(ticker)
            info = stock.info

            # Display company name and sector
            st.subheader(f"{info.get('longName', 'N/A')} ({ticker.upper()})")
            st.write(f"Sector: {info.get('sector', 'N/A')}")
            st.write(f"Industry: {info.get('industry', 'N/A')}")

            # Show key fundamentals
            st.subheader("Key Fundamentals")
            col1, col2 = st.columns(2)

            with col1:
                st.write(f"Market Cap: {info.get('marketCap', 'N/A'):,}")
                st.write(f"P/E Ratio: {info.get('trailingPE', 'N/A')}")
                st.write(f"52-Week High: {info.get('fiftyTwoWeekHigh', 'N/A')}")
                st.write(f"52-Week Low: {info.get('fiftyTwoWeekLow', 'N/A')}")

            with col2:
                st.write(f"Revenue (TTM): {info.get('totalRevenue', 'N/A'):,}")
                st.write(f"Net Income (TTM): {info.get('netIncomeToCommon', 'N/A'):,}")
                st.write(f"Dividend Yield: {info.get('dividendYield', 'N/A')}")
                st.write(f"Beta: {info.get('beta', 'N/A')}")

            # Display recent price chart
            st.subheader("Stock Price Chart")
            historical_data = stock.history(period="1y")
            st.line_chart(historical_data['Close'])

        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.info("Enter a stock ticker to see the fundamentals.")

# --- PAGE 2: Prediction Page ---
elif selection == "Prediction Page":
    st.header("🔮 Predict Stock Prices")

    # Stock and Interval Selector
    stock_symbol = st.selectbox("Select Stock", NIFTY50_COMPANIES)
    interval = st.selectbox("Select Interval", ["1 Day", "1 Week", "1 Month", "6 Months", "1 Year"])

    # Predict Prices Button
    if st.button("Predict Prices"):
        try:
            st.write(f"Fetching predictions for **{stock_symbol}** ({interval})...")
            # Call Flask API for prediction
            response = requests.post(f"{API_URL}/predict", json={"symbol": stock_symbol, "interval": interval})

            if response.status_code == 200:
                result = response.json()
                predicted_open = result.get("predicted_open")
                predicted_close = result.get("predicted_close")

                # Display Results
                st.subheader("Predicted Prices")
                st.write(f"**Predicted High:** ₹{predicted_open}")
                st.write(f"**Predicted Low:** ₹{predicted_close}")
            else:
                st.error(response.json().get("error", "Failed to fetch predictions"))
        except Exception as e:
            st.error(f"Error fetching predictions: {e}")
