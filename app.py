import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from datetime import datetime

def fetch_stock_info(symbol, exchange="NSE"):
    ticker = f"{symbol}.{exchange}" if exchange != "NSE" else symbol  # For NSE, no need to prefix
    stock = yf.Ticker(ticker)
    stock_info = stock.info
    return stock_info

# Function to fetch historical stock data
def fetch_stock_data(exchange, symbol, start_date, end_date):
    ticker = f"{symbol}.{exchange}" if exchange != "NSE" else symbol  # For NSE, no need to prefix
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    return stock_data

# Function for stock price prediction (using simple Linear Regression for demo)
def predict_stock_prices(data):
    data['Date'] = pd.to_datetime(data.index)
    data['Date'] = data['Date'].map(pd.Timestamp.toordinal)

    X = data['Date'].values.reshape(-1, 1)
    y = data['Close'].values

    model = LinearRegression()
    model.fit(X, y)

    future_dates = pd.date_range(data.index[-1], periods=30).map(pd.Timestamp.toordinal).values.reshape(-1, 1)
    future_predictions = model.predict(future_dates)

    return future_dates, future_predictions

# Function to display buy/sell recommendation (simple approach for demonstration)
def recommendation(past_data, future_predictions):
    recent_avg = past_data['Close'].tail(5).mean()
    predicted_avg = np.mean(future_predictions)

    if predicted_avg > recent_avg:
        return "Buy"
    elif predicted_avg < recent_avg:
        return "Sell"
    else:
        return "Hold"

page = st.sidebar.radio(["Home", "Stock Information", "Stock Prediction"])

if page  == "Home":
    st.title()

elif page == "Stock Information":
    st.title("Stock Information")

    # User input for stock symbol and exchange
    exchange = st.selectbox("Select Stock Exchange", ["NSE", "BSE", "LSE", "NYSE", "NASDAQ"])
    symbol = st.text_input("Enter Stock Symbol", "AAPL")  # Default example: AAPL, TSLA, etc.

    if st.button("Get Stock Information"):
        try:
            stock_info = fetch_stock_info(symbol, exchange)
            st.write(f"### {symbol} - Stock Information")
            
            # Displaying various stock details
            st.write(f"**Company Name**: {stock_info.get('longName', 'N/A')}")
            st.write(f"**Sector**: {stock_info.get('sector', 'N/A')}")
            st.write(f"**Industry**: {stock_info.get('industry', 'N/A')}")
            st.write(f"**Current Price**: ₹{stock_info.get('currentPrice', 'N/A')}")
            st.write(f"**Market Cap**: ₹{stock_info.get('marketCap', 'N/A')}")
            st.write(f"**PE Ratio**: {stock_info.get('trailingPE', 'N/A')}")
            st.write(f"**52 Week High**: ₹{stock_info.get('fiftyTwoWeekHigh', 'N/A')}")
            st.write(f"**52 Week Low**: ₹{stock_info.get('fiftyTwoWeekLow', 'N/A')}")
            st.write(f"**Dividend Yield**: {stock_info.get('dividendYield', 'N/A')}%")
            st.write(f"**Country**: {stock_info.get('country', 'N/A')}")
            
            # If you want to display a brief description
            description = stock_info.get('longBusinessSummary', 'No description available')
            st.write(f"**Company Description**: {description}")

        except Exception as e:
            st.error(f"Error fetching stock information: {str(e)}")

elif page == "Stock Prediction":
    st.title("Stock Prediction and Analysis")

    # Dropdowns for selecting stock exchange and company
    exchange = st.selectbox("Select Stock Exchange", ["NSE", "BSE", "LSE", "NYSE", "NASDAQ"])
    company = st.text_input("Enter Stock Symbol", "AAPL")

    # Date inputs for historical data
    start_date = st.date_input("Start Date", datetime(2020, 1, 1))
    end_date = st.date_input("End Date", datetime.today())

    if st.button("Get Stock Data"):
        data = fetch_stock_data(exchange, company, start_date, end_date)
        if data.empty:
            st.error("No data found for the selected stock.")
        else:
            st.write(data.tail())

            # Plotting the stock's closing prices
            st.subheader("Stock Price Over Time")
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(data.index, data['Close'], label="Close Price")
            ax.set_xlabel("Date")
            ax.set_ylabel("Price")
            ax.set_title(f"{company} Stock Price Over Time")
            st.pyplot(fig)

            # Predictions
            future_dates, future_predictions = predict_stock_prices(data)
            future_dates = pd.to_datetime(future_dates, origin='julian', unit='D')

            st.subheader("Stock Price Predictions for Next 30 Days")
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(data.index, data['Close'], label="Historical Data")
            ax.plot(future_dates, future_predictions, label="Predicted Prices", linestyle="--")
            ax.set_xlabel("Date")
            ax.set_ylabel("Price")
            ax.set_title(f"{company} Future Price Prediction")
            st.pyplot(fig)

            # Display recommendation
            action = recommendation(data, future_predictions)
            st.write(f"Recommendation: {action}")
