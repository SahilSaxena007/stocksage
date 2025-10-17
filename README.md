# StockSage — 21-Day Stock Return Prediction

This project builds and serves an ML pipeline that predicts 21-day excess stock returns using technical indicators (SMA, RSI, MACD, ATR) and regression/classification models.

## Setup Instructions

1. Navigate to the main directory and open it in Visual Studio Code:

2. Create and activate a virtual environment (if not already done):
   python -m venv .venv
   .venv\Scripts\activate # Windows
   source .venv/bin/activate # macOS/Linux

3. Install dependencies:
   pip install -r requirements.txt

## Pipeline Overview

Run the following scripts in order:

### 1. Model Training (src/train_all.py)

This automatically:

- Downloads 2 years of daily price data via Yahoo Finance
- Engineers technical features and excess returns
- Trains logistic (direction) and linear (magnitude) models
- Saves results in artifacts/models/<TICKER>/
- python .\src\train.py

### 2. Run the Prediction Dashboard (app.py)

You’ll see:

- Dropdown to select company
- Predicted direction (📈 / 📉)
- Confidence level (%)
- Expected 21-day excess move (%)
- Interactive closing-price chart
  Open the app in your browser (usually http://localhost:8501).
- python app.py
