from flask import Flask, jsonify, request
import yfinance as yf
import numpy as np
import pandas as pd
import pickle
import h5py
from tensorflow.keras.models import model_from_json
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import http.client
import json
from datetime import datetime

app = Flask(__name__)

# Constants
NIFTY50_COMPANIES = [
    'RELIANCE.NS', 'TCS.NS', 'INFY.NS', 'HDFCBANK.NS',
    'ICICIBANK.NS', 'HINDUNILVR.NS', 'BAJFINANCE.NS',
    'SBIN.NS', 'KOTAKBANK.NS', 'BHARTIARTL.NS'
]

MODEL_PATH = "models/all_models.h5"
SCALER_PATH = "scalers/all_scalers.pkl"
WEIGHTS_DIR = "weights"

# Load scalers
def load_scalers(scaler_path):
    try:
        with open(scaler_path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        print(f"Error loading scalers: {e}")
        return {}

SCALERS = load_scalers(SCALER_PATH)

# Load models dynamically
def load_models(model_path, nifty50_companies):
    models = {}
    try:
        with h5py.File(model_path, "r") as h5_file:
            for company in nifty50_companies:
                try:
                    model_group = h5_file[company]
                    model_json = model_group.attrs["model_json"]
                    model = model_from_json(model_json)
                    weights_path = f"{WEIGHTS_DIR}/{company}_weights.weights.h5"
                    model.load_weights(weights_path)
                    models[company] = model
                except Exception as e:
                    print(f"Error loading model for {company}: {e}")
    except Exception as e:
        print(f"Error reading models file: {e}")
    return models

MODELS = load_models(MODEL_PATH, NIFTY50_COMPANIES)

# Set up the FinBERT model for sentiment analysis
tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
nlp = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer)

# Function to get news related to the stock
def search_stock(stock_name):
    formatted_stock_name = stock_name.replace(" ", "+") + "+stock"
    conn = http.client.HTTPSConnection("google-news13.p.rapidapi.com")

    headers = {
        # 'x-rapidapi-key': "f03bb60d60msh270dfa3b81fb47cp1db8c7jsnf50666b62fba",
        'x-rapidapi-key': "702b824395mshaad8abc03259fa4p1901d9jsn83805da8ead7",
        'x-rapidapi-host': "google-news13.p.rapidapi.com"
    }

    try:
        conn.request("GET", f"/search?keyword={formatted_stock_name}&lr=en-US", headers=headers)
        res = conn.getresponse()
        if res.status != 200:
            print(f"API call failed with status {res.status}")
            return {"items": []}  # Return an empty list if API call fails

        data = res.read()
        response_data = json.loads(data.decode("utf-8"))
        print(f"API Response: {response_data}")  # Debugging response content
        return response_data
    except Exception as e:
        print(f"Error fetching news for {stock_name}: {e}")
        return {"items": []}

# Function to convert timestamp to readable date and time
def convert_timestamp(timestamp):
    try:
        timestamp = int(timestamp) // 1000
        dt_object = datetime.fromtimestamp(timestamp)
        formatted_date = dt_object.strftime('%d-%m-%Y')
        formatted_time = dt_object.strftime('%H:%M:%S')
        return formatted_date, formatted_time
    except Exception as e:
        print(f"Error converting timestamp: {e}")
        return "Unknown Date", "Unknown Time"

# Routes
@app.route("/")
def home():
    return jsonify({"message": "Stock Data and News Sentiment API is running"})


@app.route("/api/stock/<symbol>", methods=["GET"])
def stock_data(symbol):
    if symbol not in NIFTY50_COMPANIES:
        return jsonify({"error": f"Stock '{symbol}' is not supported."}), 400

    interval = request.args.get("interval", "1d")
    valid_intervals = ["1d", "1wk", "1mo"]
    if interval not in valid_intervals:
        return jsonify({"error": f"Invalid interval '{interval}'."}), 400

    try:
        stock_data = yf.download(symbol, period="1y", interval=interval)
        if stock_data.empty:
            return jsonify({"error": f"No data available for stock '{symbol}'."}), 404

        stock_data.reset_index(inplace=True)
        stock_dict = {
            row.Date.strftime('%Y-%m-%d'): row.Close
            for _, row in stock_data.iterrows()
            if not pd.isna(row.Close)
        }

        return jsonify({"stock": symbol, "interval": interval, "data": stock_dict})
    except Exception as e:
        return jsonify({"error": f"Failed to fetch data for {symbol}: {str(e)}"}), 500

@app.route("/news/<symbol>", methods=["GET"])
def news_sentiment(symbol):
    try:
        # Fetch news data
        news_data = search_stock(symbol)

        # Check if there are news items
        if not news_data.get("items"):
            return jsonify({"error": f"No news data found for {symbol}"}), 404

        sentiment_data = []
        for item in news_data.get("items", []):
            date, time = convert_timestamp(item.get("timestamp", 0))
            title = item.get("title", "No Title")
            sentiment = nlp(title)[0]
            sentiment_data.append({
                'Date': date,
                'Title': title,
                'Sentiment': sentiment['label'],
                'Score': round(sentiment['score'], 2)
            })

        return jsonify({"symbol": symbol, "news": sentiment_data})
    except Exception as e:
        return jsonify({"error": f"Failed to fetch news for {symbol}: {str(e)}"}), 500

@app.route('/predict_high_low', methods=['POST'])
def predict_high_low():
    try:
        data = request.json
        symbol = data.get("symbol")
        interval = data.get("interval")

        # Map intervals to days
        days_map = {
            "1 Day": 1,
            "1 Week": 7,
            "1 Month": 30,
            "6 Months": 182,
            "1 Year": 365
        }
        if interval not in days_map:
            return jsonify({"error": "Invalid interval. Choose from '1 Day', '1 Week', '1 Month', '6 Months', '1 Year'."}), 400

        days_ahead = days_map[interval]

        # Fetch recent stock data
        stock_data = yf.download(symbol, period="1y")
        if stock_data.empty:
            return jsonify({"error": f"No data available for stock '{symbol}'."}), 404

        # Get last 60 days for prediction
        recent_data = stock_data.tail(60)[["High", "Low"]]  # Change to High and Low
        if recent_data.isnull().values.any():
            return jsonify({"error": "Recent data contains NaN values. Cannot predict."}), 400

        # Scale data
        scaler = SCALERS.get(symbol)
        if not scaler:
            return jsonify({"error": f"No scaler found for stock '{symbol}'."}), 500
        scaled_data = scaler.transform(recent_data)

        # Prepare model and weights
        model = MODELS.get(symbol)
        if not model:
            return jsonify({"error": f"No model found for stock '{symbol}'."}), 500

        # Predict
        x_test = np.array([scaled_data])
        x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 2))
        for _ in range(days_ahead):
            pred = model.predict(x_test)
            new_input = np.append(x_test[0][1:], pred, axis=0)
            x_test = np.array([new_input])

        # Inverse scale final prediction
        final_prediction = scaler.inverse_transform(pred)[0]
        predicted_high, predicted_low = float(final_prediction[0]), float(final_prediction[1])  # Convert to standard Python float

        return jsonify({
            "symbol": symbol,
            "interval": interval,
            "predicted_high": round(predicted_high, 2),
            "predicted_low": round(predicted_low, 2)
        })
    except Exception as e:
        return jsonify({"error": f"Failed to generate predictions: {str(e)}"}), 500

    

# @app.route("/predict", methods=["POST"])
# def predict_open_close():
#     try:
#         data = request.json
#         symbol = data.get("symbol")
#         interval = data.get("interval")

#         # Map intervals to days
#         days_map = {
#             "1 Day": 1,
#             "1 Week": 7,
#             "1 Month": 30,
#             "6 Months": 182,
#             "1 Year": 365
#         }
#         if interval not in days_map:
#             return jsonify({"error": "Invalid interval. Choose from '1 Day', '1 Week', '1 Month', '6 Months', '1 Year'."}), 400

#         days_ahead = days_map[interval]

#         # Fetch recent stock data
#         stock_data = yf.download(symbol, period="1y")
#         if stock_data.empty:
#             return jsonify({"error": f"No data available for stock '{symbol}'."}), 404

#         # Get last 60 days for prediction
#         recent_data = stock_data.tail(60)[["Open", "Close"]]
#         if recent_data.isnull().values.any():
#             return jsonify({"error": "Recent data contains NaN values. Cannot predict."}), 400

#         # Scale data
#         scaler = SCALERS.get(symbol)
#         if not scaler:
#             return jsonify({"error": f"No scaler found for stock '{symbol}'."}), 500
#         scaled_data = scaler.transform(recent_data)

#         # Prepare model and weights
#         model = MODELS.get(symbol)
#         if not model:
#             return jsonify({"error": f"No model found for stock '{symbol}'."}), 500

#         # Predict
#         x_test = np.array([scaled_data])
#         x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 2))
#         for _ in range(days_ahead):
#             pred = model.predict(x_test)
#             new_input = np.append(x_test[0][1:], pred, axis=0)
#             x_test = np.array([new_input])

#         # Inverse scale final prediction
#         final_prediction = scaler.inverse_transform(pred)[0]
#         predicted_open, predicted_close = float(final_prediction[0]), float(final_prediction[1])  # Convert to standard Python float

#         return jsonify({
#             "symbol": symbol,
#             "interval": interval,
#             "predicted_open": round(predicted_open, 2),
#             "predicted_close": round(predicted_close, 2)
#         })
#     except Exception as e:
#         return jsonify({"error": f"Failed to generate predictions: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=True)