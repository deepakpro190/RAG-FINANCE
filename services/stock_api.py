import yfinance as yf

class StockAPIClient:
    def __init__(self):
        """
        Initializes the StockAPIClient.
        You can add any configuration settings or parameters if needed.
        """
        self.api_name = "Yahoo Finance API"
    
    def fetch_stock_data(self, query, period="1d", interval="1m"):
        """
        Fetch stock data for a given stock symbol.
        :param query: The stock symbol or company name to search.
        :param period: Period for fetching stock data (e.g., '1d', '1mo', '1y', etc.)
        :param interval: Data frequency (e.g., '1m' for 1-minute intervals, '5d' for 5 days)
        :return: A list of stock data points containing the date, open, high, low, close, and volume.
        """
        # Query the stock data from Yahoo Finance using yfinance
        stock = yf.Ticker(query)

        # Fetch historical market data
        try:
            data = stock.history(period=period, interval=interval)
            if data.empty:
                return None

            # Format the data into a list of dictionaries
            stock_data = []
            for date, row in data.iterrows():
                stock_data.append({
                    "date": date.strftime('%Y-%m-%d %H:%M:%S'),  # Date as string
                    "open": row["Open"],
                    "high": row["High"],
                    "low": row["Low"],
                    "close": row["Close"],
                    "volume": row["Volume"]
                })
            
            return stock_data

        except Exception as e:
            print(f"Error fetching stock data for {query}: {e}")
            return None
