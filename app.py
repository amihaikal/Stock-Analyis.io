import io
import base64
import re
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
import matplotlib.pyplot as plt

from flask import Flask, render_template, request, jsonify
import yfinance as yf
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('Stock Predictor.html')

@app.route('/process', methods=['POST'])
def process():
    try:
        data_json = request.get_json()
        symbol = data_json.get('symbol', 'AAPL').upper().strip()
        
        # Validate the ticker symbol:
        # Allow only letters, numbers, and periods (for tickers like BRK.A)
        if not re.match(r'^[A-Z0-9\.]+$', symbol):
            return jsonify({"output": "Error: The string did not match the expected pattern. Please enter a valid ticker symbol."})
        
        # 1. Download Data from yfinance
        data = yf.download(symbol, start="2005-01-01", end="2025-01-01")
        if data.empty:
            return jsonify({"output": f"Error: No data found for symbol '{symbol}'. Check your input."})
        
        # 2. Compute daily returns and create lag features
        data['Return'] = data['Close'].pct_change()
        data['Lag_1'] = data['Return'].shift(1)
        data['Lag_2'] = data['Return'].shift(2)
        data['Lag_3'] = data['Return'].shift(3)
        data.dropna(inplace=True)
        
        if len(data) < 10:
            return jsonify({"output": f"Not enough data after processing for symbol '{symbol}'."})
        
        # 3. Define features and target
        X = data[['Lag_1', 'Lag_2', 'Lag_3']]
        y = data['Return']
        
        # 4. Train/Test split (no shuffling to preserve time order)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        
        # 5. Train a Linear Regression model
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # 6. Evaluate the model
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        
        # 7. Implement a simple trading strategy (using the sign of predicted returns)
        test_data = data.loc[X_test.index].copy()
        test_data['Predicted_Return'] = y_pred
        test_data['Signal'] = np.where(test_data['Predicted_Return'] > 0, 1, -1)
        test_data['Strategy_Return'] = test_data['Signal'] * test_data['Return']
        
        # 8. Compute cumulative returns
        test_data['Cumulative_Market'] = (1 + test_data['Return']).cumprod()
        test_data['Cumulative_Strategy'] = (1 + test_data['Strategy_Return']).cumprod()
        final_strategy_return = test_data['Cumulative_Strategy'].iloc[-1]
        final_market_return = test_data['Cumulative_Market'].iloc[-1]
        
        # 9. Calculate daily and annualized Sharpe Ratio
        mean_return = test_data['Strategy_Return'].mean()
        std_return = test_data['Strategy_Return'].std()
        sharpe_ratio = (mean_return / std_return) if std_return != 0 else 0
        sharpe_ratio_annualized = sharpe_ratio * np.sqrt(252)
        
        # 10. Create a plot comparing cumulative returns
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(test_data.index, test_data['Cumulative_Market'], label="Market Returns", linestyle="--")
        ax.plot(test_data.index, test_data['Cumulative_Strategy'], label="Strategy Returns", color="red")
        ax.set_xlabel("Date")
        ax.set_ylabel("Cumulative Returns")
        ax.set_title(f"{symbol} Trading Strategy vs. Market Performance")
        ax.legend()
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close(fig)
        buf.seek(0)
        plot_data = base64.b64encode(buf.getvalue()).decode('utf-8')
        
        # 11. Build output text
        output_text = (
            f"=== Model Performance for {symbol} ===\n"
            f"MAE: {mae:.6f}\n"
            f"R²: {r2:.4f}\n"
            f"RMSE: {rmse:.4f}\n\n"
            f"=== Strategy vs. Market ===\n"
            f"Final Strategy Return: {final_strategy_return:.4f}\n"
            f"Final Market Return: {final_market_return:.4f}\n"
            f"Daily Sharpe Ratio: {sharpe_ratio:.4f}\n"
            f"Annualized Sharpe Ratio: {sharpe_ratio_annualized:.4f}\n"
            "Done!"
        )
        
        return jsonify({"output": output_text, "plot": plot_data})
    
    except Exception as e:
        return jsonify({"output": f"Error: {str(e)}"})

if __name__ == '__main__':
    # Run on port 5001 (or another port if you prefer)
    app.run(debug=True, port=5002)