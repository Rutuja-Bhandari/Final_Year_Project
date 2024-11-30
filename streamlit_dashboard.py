import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import yfinance as yf
import numpy as np

# API Configuration
API_URL = "http://127.0.0.1:5000"  # Flask API base URL
NIFTY50_COMPANIES = [
    'RELIANCE.NS', 'TCS.NS', 'INFY.NS', 'HDFCBANK.NS',
    'ICICIBANK.NS', 'HINDUNILVR.NS', 'BAJFINANCE.NS',
    'SBIN.NS', 'KOTAKBANK.NS', 'BHARTIARTL.NS'
]

stock_dict = {
    'RELIANCE': 'RELIANCE.NS',
    'TCS': 'TCS.NS',
    'INFY': 'INFY.NS',
    'HDFCBANK': 'HDFCBANK.NS',
    'ICICIBANK': 'ICICIBANK.NS',
    'HINDUNILVR': 'HINDUNILVR.NS',
    'BAJFINANCE': 'BAJFINANCE.NS',
    'SBIN': 'SBIN.NS',
    'KOTAKBANK': 'KOTAKBANK.NS',
    'BHARTIARTL': 'BHARTIARTL.NS'
}

# Function to calculate RSI
def calculate_rsi(data, window=14):
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Function to calculate moving averages
def calculate_moving_averages(data, short_window=20, long_window=50):
    data['Short_MA'] = data['Close'].rolling(window=short_window).mean()
    data['Long_MA'] = data['Close'].rolling(window=long_window).mean()
    return data

# Recommendation based on RSI and Moving Averages
def generate_recommendations(data):
    recommendations = []
    for i in range(len(data)):
        if data['RSI'][i] < 30 and data['Short_MA'][i] > data['Long_MA'][i]:
            recommendations.append("Buy")
        elif data['RSI'][i] > 70 and data['Short_MA'][i] < data['Long_MA'][i]:
            recommendations.append("Sell")
        else:
            recommendations.append("Hold")
    data['Recommendation'] = recommendations
    return data

# Navigation
PAGES = ["Stock Dashboard", "Prediction Page", "Technical Analysis"]

# App Title and Configuration
st.set_page_config(page_title="NIFTY50 Stock Analysis", layout="wide")
st.sidebar.title("Navigation")
selection = st.sidebar.radio("Go to", PAGES)

# --- PAGE 1: Stock Dashboard ---
if selection == "Stock Dashboard":
    st.title("📊 Company Fundamentals Dashboard")
    user_input = st.selectbox("Select a stock:", list(stock_dict.keys()))
    ticker = stock_dict.get(user_input.upper(), None)

    if st.button("Analyze") and ticker:
        try:
            with st.spinner("Fetching data..."):
                stock = yf.Ticker(ticker)
                info = stock.info
                st.subheader(f"{info.get('longName', 'N/A')} ({ticker})")
                st.write(f"Sector: {info.get('sector', 'N/A')}")
                st.write(f"Industry: {info.get('industry', 'N/A')}")
                st.write(f"Market Cap: {info.get('marketCap', 'N/A'):,}")

                st.subheader("Stock Price Chart")
                historical_data = stock.history(period="1y")
                st.line_chart(historical_data['Close'])

                st.subheader("News and Sentiment")
                response = requests.get(f"{API_URL}/news/{user_input}")
                if response.status_code == 200:
                    news_data = pd.DataFrame(response.json().get("news", []))
                    st.write(news_data)
                else:
                    st.error("Failed to fetch news data.")
        except Exception as e:
            st.error(f"An error occurred: {e}")

# --- PAGE 2: Prediction Page ---
elif selection == "Prediction Page":
    st.header("🔮 Predict Stock Prices")

    # Stock and Interval Selector
    stock_symbol = st.selectbox("Select Stock", NIFTY50_COMPANIES)
    interval = st.selectbox("Select Interval", ["1 Day", "1 Week", "1 Month", "6 Months", "1 Year"])

    # Predict Prices Button
    if st.button("Predict Prices"):
        try:
            st.write(f"Fetching predictions for {stock_symbol} ({interval})...")
            # Call Flask API for prediction
            response = requests.post(f"{API_URL}/predict", json={"symbol": stock_symbol, "interval": interval})

            if response.status_code == 200:
                result = response.json()
                predicted_open = result.get("predicted_open")
                predicted_close = result.get("predicted_close")
                st.subheader("Predicted Prices")
                st.write(f"Predicted High: ₹{predicted_open}")
                st.write(f"Predicted Low: ₹{predicted_close}")
            else:
                st.error(response.json().get("error", "Failed to fetch predictions"))
        except Exception as e:
            st.error(f"Error fetching predictions: {e}")
            
# --- PAGE 3: Technical Analysis ---
# --- PAGE 3: Technical Analysis ---
elif selection == "Technical Analysis":
    st.title("📈 Technical Analysis")

    # Sidebar Inputs
    st.sidebar.header("Technical Analysis Inputs")
    selected_stock = st.sidebar.selectbox("Select a stock:", list(stock_dict.keys()))
    start_date = st.sidebar.date_input("Start Date", value=datetime.now() - timedelta(days=365))
    end_date = st.sidebar.date_input("End Date", value=datetime.now())
    short_window = st.sidebar.slider("Short Moving Average Window", 5, 50, 20)
    long_window = st.sidebar.slider("Long Moving Average Window", 20, 200, 50)
    rsi_window = st.sidebar.slider("RSI Calculation Window", 5, 50, 14)

    # Validate Date Inputs
    if start_date >= end_date:
        st.error("Start Date must be before End Date.")
    else:
        ticker = stock_dict.get(selected_stock)

        if st.sidebar.button("Fetch and Analyze"):
            try:
                # Attempt to fetch stock data
                with st.spinner("Fetching stock data..."):
                    stock_data = yf.download(ticker, start=start_date, end=end_date)

                # Check if the fetched data is valid
                if stock_data.empty:
                    st.error("No data fetched for the selected ticker and date range. Please check your inputs.")
                else:
                    # Normalize column names if necessary
                    if isinstance(stock_data.columns, pd.MultiIndex):
                        stock_data.columns = stock_data.columns.get_level_values(0)

                    # Check if the 'Close' column exists
                    if 'Close' not in stock_data.columns or stock_data['Close'].isnull().all():
                        st.error("No 'Close' data available for the selected ticker and date range.")
                    else:
                        # Fill missing data for calculations
                        stock_data['Close'] = stock_data['Close'].fillna(method='ffill').fillna(method='bfill')

                        # Calculate RSI and Moving Averages
                        stock_data['RSI'] = calculate_rsi(stock_data)
                        stock_data = calculate_moving_averages(stock_data)

                        # Generate Recommendations
                        stock_data = generate_recommendations(stock_data)

                        # Display data
                        st.subheader(f"Stock Data for {ticker}")
                        st.write(stock_data.tail(10))

                        # Display Recommendations
                        st.subheader("Recommendations")
                        st.write(stock_data[['RSI', 'Short_MA', 'Long_MA', 'Recommendation']].tail(10))

                        # Plotting
                        st.subheader("Stock Price with Moving Averages")
                        st.line_chart(stock_data[['Close', 'Short_MA', 'Long_MA']])

                        st.subheader("RSI Over Time")
                        st.line_chart(stock_data['RSI'])

            except Exception as e:
                st.error(f"An error occurred: {e}")



# Function to calculate RSI
def calculate_rsi(data, window=14):
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Function to calculate moving averages
def calculate_moving_averages(data, short_window=20, long_window=50):
    data['Short_MA'] = data['Close'].rolling(window=short_window).mean()
    data['Long_MA'] = data['Close'].rolling(window=long_window).mean()
    return data

# Recommendation based on RSI and Moving Averages
def generate_recommendations(data):
    recommendations = []
    for i in range(len(data)):
        if data['RSI'][i] < 30 and data['Short_MA'][i] > data['Long_MA'][i]:
            recommendations.append("Buy")
        elif data['RSI'][i] > 70 and data['Short_MA'][i] < data['Long_MA'][i]:
            recommendations.append("Sell")
        else:
            recommendations.append("Hold")
    data['Recommendation'] = recommendations
    return data



