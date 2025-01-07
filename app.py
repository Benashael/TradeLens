import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from datetime import datetime

# Predefined stock symbols for each exchange
stock_options = {
    "NSE": [
        "RELIANCE", "TCS", "INFY", "HDFCBANK", "ITC", "HINDUNILVR", "ICICIBANK", "SBIN", 
        "KOTAKBANK", "BAJFINANCE", "BHARTIARTL", "ADANIENT", "AXISBANK", "LT", "MARUTI", 
        "SUNPHARMA", "TITAN", "WIPRO", "ULTRACEMCO", "ASIANPAINT"
    ],
    "BSE": [
        "500325", "532540", "500112", "500180", "532898", "500875", "500209", "500124", 
        "532500", "532215", "532648", "500510", "500300", "500440", "500010", "500410", 
        "500520", "500295", "500182", "500696"
    ],
    "NYSE": [
        "AAPL", "MSFT", "GOOGL", "AMZN", "BRK-B", "JNJ", "NVDA", "XOM", "PG", "JPM", 
        "V", "UNH", "TSLA", "MA", "HD", "CVX", "PFE", "KO", "ABBV", "LLY"
    ],
    "NASDAQ": [
        "TSLA", "NVDA", "FB", "INTC", "AMZN", "GOOGL", "AAPL", "MSFT", "ADBE", "CMCSA", 
        "CSCO", "PEP", "PYPL", "NFLX", "COST", "TXN", "INTU", "QCOM", "AMGN", "MRNA"
    ],
    "LSE": [
        "VOD.L", "HSBA.L", "RDSB.L", "BP.L", "GLEN.L", "AZN.L", "ULVR.L", "RIO.L", 
        "BARC.L", "DGE.L", "BATS.L", "PRU.L", "SHEL.L", "TSCO.L", "LLOY.L", "REL.L", 
        "RR.L", "NG.L", "AAL.L", "STAN.L"
    ]
}

# Function to check stock availability across exchanges
def check_stock_availability(symbol):
    available_in = []
    for exchange, symbols in stock_options.items():
        if symbol in symbols:
            available_in.append(exchange)
    return available_in if available_in else ["Not Available"]

def fetch_stock_info(symbol, exchange):
    # Format ticker based on exchange
    exchange_formats = {
        "NSE": symbol,            # NSE: Direct symbol
        "BSE": f"{symbol}.BO",    # BSE: Append .BO
        "NYSE": symbol,           # NYSE: Direct symbol
        "NASDAQ": symbol,         # NASDAQ: Direct symbol
        "LSE": f"{symbol}.L"      # LSE: Append .L
    }

    # Get the formatted ticker
    formatted_symbol = exchange_formats.get(exchange, symbol)
    
    # Fetch stock information
    stock = yf.Ticker(formatted_symbol)
    stock_info = stock.info
    return stock_info

# Function to fetch historical stock data
def fetch_stock_data(exchange, symbol, start_date, end_date):
    ticker = f"{symbol}.{exchange}" if exchange != "NSE" else symbol  # For NSE, no need to prefix
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    return stock_data
    
# Function for stock price prediction (Linear Regression for demonstration)
def predict_stock_prices(data, end_date):
    # Ensure the index is a datetime object
    data['Date'] = pd.to_datetime(data.index)
    data['Date'] = data['Date'].map(pd.Timestamp.toordinal)

    # Prepare data for training
    X = data['Date'].values.reshape(-1, 1)
    y = data['Close'].values

    # Train the Linear Regression model
    model = LinearRegression()
    model.fit(X, y)

    # Generate future dates dynamically based on end_date
    last_date = pd.to_datetime(data.index[-1])  # Last date in the dataset
    business_days_diff = pd.bdate_range(last_date, end_date).size  # Number of business days
    
    # Limit the business days to 365 if it exceeds
    if business_days_diff > 365:
        st.warning("The period exceeds 365 business days. Limiting to 365 business days.")
        business_days_diff = 365

    # Generate future dates
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=business_days_diff, freq="B")
    future_dates_ordinal = future_dates.map(pd.Timestamp.toordinal).values.reshape(-1, 1)

    # Predict future prices
    future_predictions = model.predict(future_dates_ordinal)

    return future_dates, future_predictions

# Function to display stock information in a neat table format
def display_stock_information(stock_data_info):
    if stock_data_info is not None:
        st.write("### Additional Stock Information (In Tabular Format)")
        
        # Display stock information in table format
        st.write("**Basic Information**")
        basic_info = {
            "Symbol": stock_data_info.get('symbol', 'N/A'),
            "Company Name": stock_data_info.get('longName', 'N/A'),
            "Currency": stock_data_info.get('currency', 'N/A'),
            "Exchange": stock_data_info.get('exchange', 'N/A'),
        }
        st.table(pd.DataFrame(list(basic_info.items()), columns=["Metric", "Value"]))
        
        st.write("**Market Data**")
        market_data = {
            "Current Price": stock_data_info.get('currentPrice', 'N/A'),
            "Previous Close": stock_data_info.get('previousClose', 'N/A'),
            "Open": stock_data_info.get('open', 'N/A'),
            "Day Low": stock_data_info.get('dayLow', 'N/A'),
            "Day High": stock_data_info.get('dayHigh', 'N/A'),
            "52 Week Low": stock_data_info.get('fiftyTwoWeekLow', 'N/A'),
            "52 Week High": stock_data_info.get('fiftyTwoWeekHigh', 'N/A'),
        }
        st.table(pd.DataFrame(list(market_data.items()), columns=["Metric", "Value"]))
        
        st.write("**Volume and Shares**")
        volume_data = {
            "Volume": stock_data_info.get('volume', 'N/A'),
            "Market Volume": stock_data_info.get('regularMarketVolume', 'N/A'),
            "Average Volume": stock_data_info.get('averageVolume', 'N/A'),
            "Shares Outstanding": stock_data_info.get('sharesOutstanding', 'N/A'),
        }
        st.table(pd.DataFrame(list(volume_data.items()), columns=["Metric", "Value"]))
        
        st.write("**Dividends and Yield**")
        dividend_data = {
            "Dividend Rate": stock_data_info.get('dividendRate', 'N/A'),
            "Dividend Yield": stock_data_info.get('dividendYield', 'N/A'),
        }
        st.table(pd.DataFrame(list(dividend_data.items()), columns=["Metric", "Value"]))
        
    else:
        st.error("Stock data not available or symbol not found for the selected exchange.")

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

page = st.sidebar.radio("Select", ["Home", "Stock Information", "Stock Prediction"])

if page  == "Home":
    st.title("")

elif page == "Stock Information":
    st.title("Stock Information")
    st.write("Select a stock symbol and get details about its availability and key metrics.")

    # Dropdown for selecting exchange
    exchange = st.selectbox("Select Stock Exchange", list(stock_options.keys()))
    
    # Dropdown for selecting stock symbol
    symbol = st.selectbox("Select Stock Symbol", stock_options[exchange])

    # Button to get stock information
    if st.button("Check Availability and Get Stock Information"):
        # Check availability
        availability = check_stock_availability(symbol)
        if "Not Available" in availability:
            st.error(f"The stock '{symbol}' is not available in any exchange.")
        else:
            st.success(f"The stock '{symbol}' is available in: {', '.join(availability)}")

            # Fetch and display stock information
            try:
                stock_info = fetch_stock_info(symbol, exchange)
                st.write(f"### Stock Information for {symbol} ({exchange})")
                st.write(f"**Company Name**: {stock_info.get('longName', 'N/A')}")
                st.write(f"**Sector**: {stock_info.get('sector', 'N/A')}")
                st.write(f"**Industry**: {stock_info.get('industry', 'N/A')}")
                st.write(f"**Current Price**: {stock_info.get('currentPrice', 'N/A')}")
                st.write(f"**Market Cap**: {stock_info.get('marketCap', 'N/A')}")
                st.write(f"**PE Ratio**: {stock_info.get('trailingPE', 'N/A')}")
                st.write(f"**52 Week High**: {stock_info.get('fiftyTwoWeekHigh', 'N/A')}")
                st.write(f"**52 Week Low**: {stock_info.get('fiftyTwoWeekLow', 'N/A')}")
                st.write(f"**Beta**: {stock_info.get('beta', 'N/A')}")
                st.write(f"**Dividend Yield**: {stock_info.get('dividendYield', 'N/A')}")
                st.write(f"**Volume**: {stock_info.get('volume', 'N/A')}")
                st.write(f"**Country**: {stock_info.get('country', 'N/A')}")
                description = stock_info.get('longBusinessSummary', 'No description available')
                st.write(f"**Company Description**: {description}")
                stock_info = fetch_stock_info(symbol, exchange)
                display_stock_information(stock_info)
            except Exception as e:
                st.error(f"Failed to fetch stock information: {str(e)}")

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

            # Plotting the stock's price data (Open, High, Low, Close)
            st.subheader("Stock Prices Over Time")
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Plotting each price type with a unique color and label
            ax.plot(data.index, data['Open'], label="Open Price", color="blue")
            ax.plot(data.index, data['High'], label="High Price", color="green")
            ax.plot(data.index, data['Low'], label="Low Price", color="red")
            ax.plot(data.index, data['Close'], label="Close Price", color="orange")
            
            # Adding labels, title, and legend
            ax.set_xlabel("Date")
            ax.set_ylabel("Price")
            ax.set_title(f"{company} Stock Prices Over Time")
            ax.legend(loc="upper left")  # Position of legend
            
            # Display the plot
            st.pyplot(fig)

            # Plotting the stock's Open Price
            st.subheader("Open Price Over Time")
            fig_open, ax_open = plt.subplots(figsize=(10, 6))
            ax_open.plot(data.index, data['Open'], label="Open Price", color="blue")
            ax_open.set_xlabel("Date")
            ax_open.set_ylabel("Price")
            ax_open.set_title(f"{company} Open Price Over Time")
            ax_open.legend(loc="upper left")
            st.pyplot(fig_open)
            
            # Plotting the stock's High Price
            st.subheader("High Price Over Time")
            fig_high, ax_high = plt.subplots(figsize=(10, 6))
            ax_high.plot(data.index, data['High'], label="High Price", color="green")
            ax_high.set_xlabel("Date")
            ax_high.set_ylabel("Price")
            ax_high.set_title(f"{company} High Price Over Time")
            ax_high.legend(loc="upper left")
            st.pyplot(fig_high)
            
            # Plotting the stock's Low Price
            st.subheader("Low Price Over Time")
            fig_low, ax_low = plt.subplots(figsize=(10, 6))
            ax_low.plot(data.index, data['Low'], label="Low Price", color="red")
            ax_low.set_xlabel("Date")
            ax_low.set_ylabel("Price")
            ax_low.set_title(f"{company} Low Price Over Time")
            ax_low.legend(loc="upper left")
            st.pyplot(fig_low)
            
            # Plotting the stock's Close Price
            st.subheader("Close Price Over Time")
            fig_close, ax_close = plt.subplots(figsize=(10, 6))
            ax_close.plot(data.index, data['Close'], label="Close Price", color="orange")
            ax_close.set_xlabel("Date")
            ax_close.set_ylabel("Price")
            ax_close.set_title(f"{company} Close Price Over Time")
            ax_close.legend(loc="upper left")
            st.pyplot(fig_close)

            # Predictions
            future_dates, future_predictions = predict_stock_prices(data, end_date)
            
            # Plot the original closing prices and the predicted future prices
            st.subheader(f"Stock Price Prediction for {company}")
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Plot historical data
            ax.plot(data.index, data['Close'], label="Historical Close Price", color="blue")
            
            # Plot future predictions
            ax.plot(future_dates, future_predictions, label="Predicted Price", color="orange", linestyle="--")
            
            # Configure plot
            ax.set_xlabel("Date")
            ax.set_ylabel("Price")
            ax.set_title(f"{company} Stock Price Prediction")
            ax.legend()
            
            # Display the plot
            st.pyplot(fig)
            
            # Display predicted prices
            st.subheader("Future Predicted Prices")
            # Flatten the predictions array
            predictions_df = pd.DataFrame({
                "Date": future_dates, 
                "Predicted Price": future_predictions.flatten()
            })
            st.write(predictions_df)

            # Display recommendation
            action = recommendation(data, future_predictions)
            st.write(f"Recommendation: {action}")
