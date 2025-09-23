#Average Directional Index
def ADX_indicators(df,ticker='AAPL' ,window=14):
   high = df[('High', ticker)]
    low = df[('Low', ticker)]
    close = df[('Close', ticker)]
    print('forma high',high)

    true_range = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    ],axis=1).max(axis=1)

    plus_dm = high.diff()
    minus_dm = low.diff() * -1

    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)

    # ATR
    atr = true_range.rolling(window=window).mean()

    # Directional Indicators
    plus_di = 100 * (plus_dm.rolling(window=window).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(window=window).mean() / atr)

    # DX i ADX
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.rolling(window=window).mean()
    # print('ADX TYPE',type(adx))
    # print(adx.head)
    return adx

ADX_TEMP = ADX_indicators(data_with_indicators)