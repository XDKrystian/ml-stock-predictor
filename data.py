import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
from networkx.algorithms.threshold import creation_sequence

from  data_colecting import download_parts_data
from datetime import datetime, timedelta
import numpy as np
from torch.distributions.constraints import interval
from yfinance.utils import auto_adjust
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import joblib

# Dodaj wskaźniki
data_clear  = download_parts_data(
    ticker="AAPL",
    start_date=datetime(2022, 1, 1),
    end_date=datetime(2025, 1, 2)
    )

data_with_indicators = data_clear
#Średnia krocząca
data_with_indicators['SMA50'] = data_clear['Close'].rolling(window=50).mean()  # 50-dniowa średnia
data_with_indicators['SMA200'] = data_clear['Close'].rolling(window=200).mean()  # 200-dniowa średnia

# RSI
def calculate_rsi(data, window=14):
    delta = data['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1/window, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1/window, min_periods=window).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

data_with_indicators['RSI'] = calculate_rsi(data_clear)

# MACD
exp12 = data_clear['Close'].ewm(span=12, adjust=False).mean()
exp26 = data_clear['Close'].ewm(span=26, adjust=False).mean()
data_with_indicators['MACD'] = exp12 - exp26
data_with_indicators['MACD_signal'] = data_with_indicators['MACD'].ewm(span=9, adjust=False).mean()

# Bollinger Bands
rolling_mean = data_clear['Close'].rolling(window=20).mean()
rolling_std = data_clear['Close'].rolling(window=20).std()
data_with_indicators['UpperBand'] = rolling_mean + (rolling_std * 2)
data_with_indicators['LowerBand'] = rolling_mean - (rolling_std * 2)

# ATR
data_with_indicators['High-Low'] = data_clear['High'] - data_clear['Low']
data_with_indicators['High-Close'] = (data_clear['High'] - data_clear['Close'].shift()).abs()
data_with_indicators['Low-Close'] = (data_clear['Low'] - data_clear['Close'].shift()).abs()
data_with_indicators['TrueRange'] = data_with_indicators[['High-Low', 'High-Close', 'Low-Close']].max(axis=1)
data_with_indicators['ATR'] = data_with_indicators['TrueRange'].rolling(window=14).mean()

#Stochastic Oscillator
low_14 = data_with_indicators['Low'].rolling(window=14).min()
high_14 = data_with_indicators['High'].rolling(window=14).max()

data_with_indicators['%K'] = 100 * (data_with_indicators['Close'] - low_14) / (high_14 - low_14)
data_with_indicators['%D'] = data_with_indicators['%K'].rolling(window=3).mean()


#OBV

close = data_with_indicators['Close']
volume = data_with_indicators['Volume']

direction = np.sign(close.diff().fillna(0))

obv = (direction * volume).cumsum()

data_with_indicators['OBV'] = obv

#print(data_with_indicators[['Close', 'Volume', 'OBV']].head(10))

#ichimoku Cloud
high_9 = data_with_indicators['High'].rolling(window=9).max()
low_9 = data_with_indicators['Low'].rolling(window=9).min()
data_with_indicators['Tenkan'] = (high_9 + low_9) / 2

high_26 = data_with_indicators['High'].rolling(window=26).max()
low_26 = data_with_indicators['Low'].rolling(window=26).min()
data_with_indicators['Kijun'] = (high_26 + low_26) / 2

data_with_indicators['SenkouA'] = ((data_with_indicators['Tenkan'] + data_with_indicators['Kijun']) / 2).shift(26)
high_52 = data_with_indicators['High'].rolling(window=52).max()
low_52 = data_with_indicators['Low'].rolling(window=52).min()
data_with_indicators['SenkouB'] = ((high_52 + low_52) / 2).shift(26)

def create_sequences(data, sequence_length):
    """Tworzy sekwencje danych dla modeli czasowych"""
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:(i + sequence_length)])
        y.append(data[i + sequence_length])
    return np.array(X), np.array(y)



def prepare_data_for_ai_model(data_frame_with_indicators, sequence_length=30):
    # Kopia danych aby nie modyfikować oryginału

    if isinstance(data_frame_with_indicators.columns, pd.MultiIndex):
        data_frame_with_indicators.columns = [
            f'{col[0]}_{col[1]}' if col[1] else col[0]
            for col in data_frame_with_indicators.columns
        ]

        # Sprawdź nowe nazwy kolumn
    #print("Spłaszczone kolumny:", data_frame_with_indicators.columns.tolist())

    df = data_frame_with_indicators.copy()

    # 1. Usuwanie brakujących wartości
    df = df.dropna()

    # 2. Tworzenie targetu
    df['Target'] = df['Close_AAPL'].shift(-1)
    df = df[:-1]  # usuwamy ostatni wiersz z NaN w targetcie

    # 3. Definiowanie kolumn features
    feature_columns = [col for col in df.columns
                       if col not in ['Date', 'Target', 'Close_AAPL', 'Price_Change']]

    # 4. Normalizacja TYLKO cech (nie całego DataFrame!)
    scaler = StandardScaler()
    df_scaled = df.copy()  # tworzymy kopię
    df_scaled[feature_columns] = scaler.fit_transform(df[feature_columns])

    # 5. Tworzenie sekwencji (funkcja musi być zdefiniowana!)
    X, y_sequences = create_sequences(df_scaled[feature_columns].values, sequence_length)

    # 6. Dopasowanie targetu do sekwencji
    y_target = df['Target'].values[sequence_length:]

    return X, y_sequences, y_target, scaler, feature_columns


X, y_sequences, y_target, scaler, features = prepare_data_for_ai_model(data_with_indicators)

# print(f"X shape: {X.shape}")          # (próbki, sequence_length, cechy)
# print(f"y_sequences shape: {y_sequences.shape}")  # (próbki, cechy)
# print(f"y_target shape: {y_target.shape}")        # (próbki,)




