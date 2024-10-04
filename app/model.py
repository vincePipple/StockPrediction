import yfinance as yf
import pandas as pd


def fetch_stock_data(ticker: str, period = '1y', interval = '1d') -> pd.DataFrame:
    """
    Retrieving stock data

    Args:
        ticker: stock ticker about which predictions should be made
        period: the amount of time back data is retrieved
        interval: size of the time period
    """

    stock_data = yf.download(ticker, period=period, interval=interval)
    if stock_data.empty:
        return None
    
    stock_data.reset_index(inplace=True)

    return stock_data


def preprocess_stock_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates moving average with a time window of 10 days 
    Calculates percentage change for closing prices

    Args:
        data: Pandas data frame with stock data
    """
    data = data.dropna()

    data['MA10'] = data['Close'].rolling(window=10).mean()
    data['Pct_Change'] = data['Close'].pct_change()

    data = data.dropna()

    return data

