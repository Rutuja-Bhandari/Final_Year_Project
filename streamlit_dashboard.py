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

# Navigation
PAGES = ["Stock Dashboard", "Prediction Page"]

# App Title and Configuration
st.set_page_config(page_title="NIFTY50 Stock Analysis", layout="wide")
st.sidebar.title("Navigation")
selection = st.sidebar.radio("Go to", PAGES)

# --- PAGE 1: Stock Dashboard ---
if selection == "Stock Dashboard":
    st.title("📊 Company Fundamentals Dashboard")

    # Dropdown menu for selecting a stock
    user_input = st.selectbox(
        "Select a stock:",
        list(stock_dict.keys())  # Dropdown options
    )

    # Automatically append '.NS' if not already present
    if user_input:
        user_input = stock_dict.get(user_input.upper(), "Ticker not found")
        if not user_input.upper().endswith(".NS"):
            ticker = f"{user_input.upper()}.NS"
        else:
            ticker = user_input.upper()
    else:
        ticker = None

    # Analyze button
    if st.button("Analyze"):
        if ticker:
            try:
                # Display a spinner while fetching data
                with st.spinner("Fetching data..."):
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

                    # Stock News and Sentiment
                    st.subheader("Recent Stock News and Sentiment")
                    user_input = user_input[:-3]
                    response = requests.get(f"{API_URL}/news/{user_input}")
                    if response.status_code == 200:
                        news_data = response.json().get("news", [])
                        if news_data:
                            # Create a DataFrame from news
                            news_df = pd.DataFrame(news_data)

                            # Convert date column to datetime format
                            news_df['Date'] = pd.to_datetime(news_df['Date'])

                            # Filter data for the last 5 months
                            five_months_ago = datetime.now() - timedelta(days=5*30)
                            news_df = news_df[news_df['Date'] > five_months_ago]

                            # Group by date and average sentiment score
                            sentiment_avg = news_df.groupby('Date').agg({
                                'Sentiment': 'first',  # Keep the first sentiment value for that day
                                'Score': 'mean'  # Average the sentiment score for each day
                            }).reset_index()

                            # Plot sentiment scores over the last 5 months
                            fig, ax = plt.subplots(figsize=(12, 6))  # Increased width for better spacing
                            colors = []

                            # Assign a numeric index to each date to maintain equal distance between bars
                            dates_index = range(len(sentiment_avg))

                            # Loop through the rows and determine the color and sentiment score
                            for index, row in sentiment_avg.iterrows():
                                sentiment = row['Sentiment']
                                score = row['Score']
                                date = row['Date']

                                bar_width = 0.4  # Increase the width of the bars for better visibility
                                if sentiment == "positive":
                                    color = 'green'
                                elif sentiment == "negative":
                                    color = 'red'
                                    score=-score
                                else:  # NEUTRAL
                                    color = 'yellow'
                                    score=score-0.5

                                ax.bar(dates_index[index], score, color=color, width=bar_width)

                            # Customize the title and axis labels
                            ax.set_title(f"Sentiment Scores for {user_input} Over Time", fontsize=14)
                            ax.set_xlabel('Date', fontsize=12)
                            ax.set_ylabel('Sentiment Score (-1 to 1)', fontsize=12)

                            # Set x-axis ticks to the numeric index and format them with the corresponding date
                            ax.set_xticks(dates_index)
                            ax.set_xticklabels(sentiment_avg['Date'].dt.strftime('%d-%m-%Y'), rotation=45, ha='right', fontsize=10)

                            # Disable grid lines for a cleaner look
                            ax.grid(False)

                            # Add some padding between x-axis ticks
                            plt.tight_layout()

                            # Display the plot
                            st.pyplot(fig)


                            # Apply CSS to style the table
                            st.markdown("""
                            <style>
                            .custom-table {
                                width: 80%; /* Set width */
                                margin: 20px auto; /* Center the table with spacing */
                                border-collapse: collapse; /* Merge borders */
                                font-family: Arial, sans-serif; /* Set font */
                            }
                            .custom-table th, .custom-table td {
                                border: 1px solid #ddd; /* Border for cells */
                                padding: 8px; /* Cell padding */
                                text-align: left; /* Align text */
                            }
                            .custom-table th {
                               /* Light gray for header */
                                font-weight: bold; /* Bold header text */
                            }
                            </style>
                            """, unsafe_allow_html=True)

                            # Convert DataFrame to an HTML table
                            news_table_html = news_df.to_html(classes="custom-table", index=False, escape=False)

                            # Render the styled HTML table
                            st.markdown(news_table_html, unsafe_allow_html=True)
                        else:
                            st.info("No news data available.")
                    else:
                        st.error(response.json().get("error", "Failed to fetch news data."))
                st.success("Data loaded successfully!")
            except Exception as e:
                st.error(f"An error occurred: {e}")
        else:
            st.info("Select a stock to analyze.")


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

                # Display Results
                st.subheader("Predicted Prices")
                st.write(f"Predicted High: ₹{predicted_open}")
                st.write(f"Predicted Low: ₹{predicted_close}")
            else:
                st.error(response.json().get("error", "Failed to fetch predictions"))
        except Exception as e:
            st.error(f"Error fetching predictions: {e}")