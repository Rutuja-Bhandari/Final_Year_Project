import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

NIFTY50_COMPANIES = [
    'RELIANCE.NS', 'TCS.NS', 'INFY.NS', 'HDFCBANK.NS',
    'ICICIBANK.NS', 'HINDUNILVR.NS', 'BAJFINANCE.NS',
    'SBIN.NS', 'KOTAKBANK.NS', 'BHARTIARTL.NS'
]

st.title("Nifty50 Stock Dashboard")
st.subheader("Interactive Stock Price Visualization")

# Stock Selector
stock_symbol = st.selectbox("Select a Stock Symbol", options=NIFTY50_COMPANIES)

# Date Range Selector
col1, col2 = st.columns(2)
start_date = col1.date_input("Start Date", value=datetime.now() - timedelta(days=365))
end_date = col2.date_input("End Date", value=datetime.now())

# Fetch Stock Data
if st.button("Generate Graph"):
    try:
        st.write(f"Fetching data for *{stock_symbol}* from {start_date} to {end_date}...")
        stock_data = yf.download(stock_symbol, start=start_date, end=end_date)

        if stock_data.empty:
            st.warning("No data available for the selected date range.")
        else:
            st.write("### Closing Price Trend")
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(stock_data.index, stock_data['Close'], label='Close Price', color='blue', marker='o')
            ax.set_title(f"{stock_symbol} Closing Prices")
            ax.set_xlabel("Date")
            ax.set_ylabel("Price (INR)")
            ax.legend()
            ax.grid(True)
            plt.xticks(rotation=45)
            st.pyplot(fig)

            if st.checkbox("Show Raw Data"):
                st.write("### Raw Data")
                st.dataframe(stock_data)
    except Exception as e:
        st.error(f"Failed to fetch stock data: {e}")