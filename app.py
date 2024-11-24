from flask import Flask, jsonify, request
import yfinance as yf
import numpy as np
import pickle
import h5py
from tensorflow.keras.models import model_from_json
import pandas as pd

app = Flask(__name__)

# List of supported stocks
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

@app.route("/")
def home():
    return jsonify({"message": "Stock Data API is running"})

@app.route("/api/stock/<symbol>", methods=["GET"])
def stock_data(symbol):
    if symbol not in NIFTY50_COMPANIES:
        return jsonify({"error": f"Stock '{symbol}' is not supported."}), 400

    # Retrieve query parameter for interval (default to "1d")
    interval = request.args.get("interval", "1d")

    # Validate the interval
    valid_intervals = ["1d", "1wk", "1mo"]
    if interval not in valid_intervals:
        return jsonify({"error": f"Invalid interval '{interval}'. Choose from {valid_intervals}."}), 400

    try:
        # Fetch historical stock data based on interval
        stock_data = yf.download(symbol, period="1y", interval=interval)
        if stock_data.empty:
            return jsonify({"error": f"No data available for stock '{symbol}'."}), 404

        # Ensure the index is treated as datetime for proper formatting
        stock_data.reset_index(inplace=True)

        # Prepare the response with close prices
        stock_dict = {
            row.Date.strftime('%Y-%m-%d'): row.Close
            for _, row in stock_data.iterrows()
            if not pd.isna(row.Close)  # Skip rows with NaN values
        }

        return jsonify({"stock": symbol, "interval": interval, "data": stock_dict})
    except Exception as e:
        return jsonify({"error": f"Failed to fetch data for {symbol}: {str(e)}"}), 500

@app.route("/predict", methods=["POST"])
def predict_open_close():
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
        recent_data = stock_data.tail(60)[["Open", "Close"]]
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
        predicted_open, predicted_close = float(final_prediction[0]), float(final_prediction[1])  # Convert to standard Python float

        return jsonify({
            "symbol": symbol,
            "interval": interval,
            "predicted_open": round(predicted_open, 2),
            "predicted_close": round(predicted_close, 2)
        })
    except Exception as e:
        return jsonify({"error": f"Failed to generate predictions: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=True)
