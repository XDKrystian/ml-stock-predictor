import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta


def download_parts_data(start_date, end_date, ticker):
    try:
        df = yf.download(
            ticker,
            start=start_date.strftime("%Y-%m-%d"),
            end=end_date.strftime("%Y-%m-%d"),
            interval="1d",
            auto_adjust=True,
            progress=False
        )
        if df.empty:
            print("⚠️ Brak danych.")
        else:
            print(f"Pobrano dane dzienne {ticker}: {len(df)} wierszy od {df.index.min().date()} do {df.index.max().date()}")
        return df

    except Exception as e:
        print(f"Błąd pobierania danych: {e}")
    return pd.DataFrame()

data = download_parts_data(
    ticker="AAPL",
    start_date=datetime(2010, 1, 1),
    end_date=datetime(2025, 1, 2)
)
#print(data.head())
#print(f"Liczba wierszy i kolumn: {data.shape}")