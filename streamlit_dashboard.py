import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

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

# Streamlit app
st.title("AI Stock Recommendation System")
st.write("""
This app uses *Relative Strength Index (RSI)* and *Moving Averages (MA)* to provide stock trading recommendations.
""")

# Main page for user input
ticker = st.text_input("Enter Stock Ticker", value="AAPL")
start_date = st.date_input("Start Date", value=pd.to_datetime("2022-01-01"))
end_date = st.date_input("End Date", value=pd.to_datetime("today") + pd.Timedelta(days=1))  # Add a day ahead to current date

if st.button("Fetch and Analyze"):
    # Fetch stock data
    stock_data = yf.download(ticker, start=start_date, end=end_date)

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

        # Reverse the data to show the most recent data first
        stock_data_reversed = stock_data.iloc[::-1]

        # Display data
        st.subheader(f"Stock Data for {ticker}")
        st.write(stock_data_reversed)

        # Display Recommendations
        st.subheader("Recommendations")
        st.write(stock_data_reversed[['RSI', 'Short_MA', 'Long_MA', 'Recommendation']].tail(10))

        # Plotting
        st.subheader("Stock Price with Moving Averages")
        st.line_chart(stock_data_reversed[['Close', 'Short_MA', 'Long_MA']])

        st.subheader("RSI Over Time")
        st.line_chart(stock_data_reversed['RSI'])


# import streamlit as st
# import requests
# import pandas as pd
# import matplotlib.pyplot as plt
# from datetime import datetime, timedelta
# import yfinance as yf

# # API Configuration
# API_URL = "http://127.0.0.1:5000"  # Flask API base URL
# NIFTY50_COMPANIES = [
#     'RELIANCE.NS', 'TCS.NS', 'INFY.NS', 'HDFCBANK.NS',
#     'ICICIBANK.NS', 'HINDUNILVR.NS', 'BAJFINANCE.NS',
#     'SBIN.NS', 'KOTAKBANK.NS', 'BHARTIARTL.NS'
# ]

# st.title("Nifty50 Stock Dashboard")
# st.subheader("Interactive Stock Price Visualization")

# # Stock Selector
# stock_symbol = st.selectbox("Select a Stock Symbol", options=NIFTY50_COMPANIES)

# # Date Range Selector
# col1, col2 = st.columns(2)
# start_date = col1.date_input("Start Date", value=datetime.now() - timedelta(days=365))
# end_date = col2.date_input("End Date", value=datetime.now())

# # Fetch Stock Data
# if st.button("Generate Graph"):
#     try:
#         st.write(f"Fetching data for *{stock_symbol}* from {start_date} to {end_date}...")
#         stock_data = yf.download(stock_symbol, start=start_date, end=end_date)

#         if stock_data.empty:
#             st.warning("No data available for the selected date range.")
#         else:
#             st.write("### Closing Price Trend")
#             fig, ax = plt.subplots(figsize=(10, 6))
#             ax.plot(stock_data.index, stock_data['Close'], label='Close Price', color='blue', marker='o')
#             ax.set_title(f"{stock_symbol} Closing Prices")
#             ax.set_xlabel("Date")
#             ax.set_ylabel("Price (INR)")
#             ax.legend()
#             ax.grid(True)
#             plt.xticks(rotation=45)
#             st.pyplot(fig)

#             if st.checkbox("Show Raw Data"):
#                 st.write("### Raw Data")
#                 st.dataframe(stock_data)
#     except Exception as e:
#         st.error(f"Failed to fetch stock data: {e}")