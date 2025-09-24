import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
from networkx.algorithms.threshold import creation_sequence

from  data_colecting import download_parts_data_with_buffer
from datetime import datetime, timedelta
import numpy as np
from torch.distributions.constraints import interval
from yfinance.utils import auto_adjust
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split
import joblib

# Dodaj wskaźniki
data_clear  = download_parts_data_with_buffer(
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

data_with_indicators.columns = [f"{c[0]}_{c[1]}" if c[1] else c[0] for c in data_with_indicators.columns]



data_with_indicators = data_with_indicators.iloc[200:]

data_with_indicators.to_excel('data_before.xlsx')

print(data_with_indicators.columns)



print(type(data_with_indicators))


def create_sequences(data, target_idx, time_steps=60):
    """
    data: numpy array (wszystkie dane z feature + target)
    target_idx: index kolumny targetu w tym array
    """
    X, y = [], []
    for i in range(time_steps, len(data)):
        X.append(data[i - time_steps:i])
        y.append(data[i, target_idx])
    return np.array(X), np.array(y)


def prepare_train_val_test_data(
        df,
        feature_columns=None,
        target_column="Close_AAPL",
        target_shift_days=[3, 7],
        sequence_length=30,
        train_ratio=0.7,
        val_ratio=0.1,
        scaler_type="standard",
        use_price_change=True
):
    df = df.copy()

    # 1️ tworzymy targety przesunięte
    target_cols = []
    for shift in target_shift_days:
        col_name = f"Target_{shift}d"
        if use_price_change:
            df.loc[:, col_name] = (df[target_column].shift(-shift) - df[target_column]) / df[target_column] * 100
        else:
            df[col_name] = df[target_column].shift(-shift)
        target_cols.append(col_name)

    max_shift = max(target_shift_days)
    if max_shift > 0:
        df = df.iloc[:-max_shift].reset_index(drop=True)

    if feature_columns is None:
        feature_columns = [col for col in df.columns if col not in ["Date", target_column] + target_cols]

    # 2️ podział na zbiory (kolejne segmenty, bez "ze środka")
    n_total = len(df)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    df_train = df.iloc[:n_train]
    df_val = df.iloc[n_train:n_train + n_val]
    df_test = df.iloc[n_train + n_val:]

    # 3️ skalowanie (fit tylko na train, potem transform na resztę)
    if scaler_type == "standard":
        scaler = StandardScaler()
    elif scaler_type == "minmax":
        scaler = MinMaxScaler()
    elif scaler_type == "robust":
        scaler = RobustScaler()
    else:
        raise ValueError("scaler_type must be 'standard', 'minmax' or 'robust'")

    df_train_scaled = df_train.copy()
    df_val_scaled = df_val.copy()
    df_test_scaled = df_test.copy()

    scaler.fit(df_train[feature_columns])
    df_train_scaled[feature_columns] = scaler.transform(df_train[feature_columns])
    df_val_scaled[feature_columns] = scaler.transform(df_val[feature_columns])
    df_test_scaled[feature_columns] = scaler.transform(df_test[feature_columns])

    # 4️ sekwencje – multi-target
    target_indices = [df.columns.get_loc(c) for c in target_cols]

    def create_multi_sequences(data, time_steps=60):
        X, y = [], []
        for i in range(time_steps, len(data)):
            X.append(data[i - time_steps:i])
            y.append(data[i, target_indices])  # wszystkie targety
        return np.array(X), np.array(y)

    X_train, y_train = create_multi_sequences(df_train_scaled.values, sequence_length)
    X_val, y_val = create_multi_sequences(df_val_scaled.values, sequence_length)
    X_test, y_test = create_multi_sequences(df_test_scaled.values, sequence_length)

    return X_train, y_train, X_val, y_val, X_test, y_test, scaler, feature_columns, target_cols


X_train, y_train, X_val, y_val, X_test, y_test, scaler, feature_columns, target_cols = prepare_train_val_test_data(data_with_indicators)


print(X_train.shape,'X_train.shape')
print(y_train.shape,'y_train.shape')
print(X_val.shape,'X_val.shape')
print(y_val.shape,'y_val.shape')
print(X_test.shape,'X_test.shape')


