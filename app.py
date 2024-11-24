from flask import Flask, jsonify
import yfinance as yf

app = Flask(_name_)

# List of supported stocks
NIFTY50_COMPANIES = [
    'RELIANCE.NS', 'TCS.NS', 'INFY.NS', 'HDFCBANK.NS',
    'ICICIBANK.NS', 'HINDUNILVR.NS', 'BAJFINANCE.NS',
    'SBIN.NS', 'KOTAKBANK.NS', 'BHARTIARTL.NS'
]

@app.route("/")
def home():
    return jsonify({"message": "Stock Data API is running"})

@app.route("/api/stock/<symbol>", methods=["GET"])
def stock_data(symbol):
    if symbol not in NIFTY50_COMPANIES:
        return jsonify({"error": f"Stock '{symbol}' is not supported."}), 400
    try:
        # Fetch historical stock data for the past year
        stock_data = yf.download(symbol, period="1y", interval="1d")
        if stock_data.empty:
            return jsonify({"error": f"No data available for stock '{symbol}'."}), 404

        # Prepare the response with close prices
        data = stock_data['Close'].dropna().tail(30)  # Last 30 days
        stock_dict = {
            date.strftime('%Y-%m-%d'): float(price)  # Ensure proper string formatting
            for date, price in data.items()
        }

        return jsonify({"stock": symbol, "data": stock_dict})
    except Exception as e:
        return jsonify({"error": f"Failed to fetch data for {symbol}: {str(e)}"}), 500

if _name_ == "_main_":
    app.run(debug=True)