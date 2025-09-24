import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta


def download_parts_data_with_buffer(start_date, end_date, ticker, extra_days=300):
    """
    extra_days  potrzebne do liczenia wskaźników + dni niehadlowe weekendy + święta
    """
    try:
        # Data z buforem
        start_date_with_buffer = start_date - pd.Timedelta(days=extra_days)

        df = yf.download(
            ticker,
            start=start_date_with_buffer.strftime("%Y-%m-%d"),
            end=end_date.strftime("%Y-%m-%d"),
            interval="1d",
            auto_adjust=True,
            progress=False
        )

        if df.empty:
            print("⚠️ Brak danych.")
            return pd.DataFrame()

        print(f"📊 Pobrano dane {ticker}: {len(df)} wierszy")
        print(f"   Okres z buforem: {df.index.min().date()} → {df.index.max().date()}")
        print(f"   Docelowy okres:  {start_date.date()} → {end_date.date()}")
        print(f"   Bufor: {extra_days} dni wstecz")

        return df

    except Exception as e:
        print(f"❌ Błąd pobierania danych: {e}")
        return pd.DataFrame()

# data = download_parts_data_with_buffer(
#     ticker="AAPL",
#     start_date=datetime(2000, 1, 1),
#     end_date=datetime(2025, 1, 2)
# )
#print(data.head())
#print(f"Liczba wierszy i kolumn: {data.shape}")