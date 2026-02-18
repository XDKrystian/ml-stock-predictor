# src/data_pipeline.py
from __future__ import annotations

import os
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml
import pandas as pd
import numpy as np
import requests
import yfinance as yf
import pandas_ta as ta  # Dodano: biblioteka do wskaźników technicznych

from utils import (
    ensure_dir,
    coerce_datetime_index,
    to_business_daily,
    save_df,
    add_suffix,
)

from dotenv import load_dotenv

load_dotenv()

# Konfiguracja logowania
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

# ==============
# CONFIG (rozszerzony)
# ==============
DEFAULT_CONFIG = """
ticker: "AAPL"
currency: "USD"
start: "2010-01-01"
end: null
price_interval: "1d"
auto_adjust: true

# Zapobieganie look-ahead bias: o ile dni przesunąć publikację raportu (dni kalendarzowe)
fundamentals_lag_days: 60 

benchmarks:
  tickers: ["^GSPC", "^VIX", "UUP", "^WIG20", "PLN=X"]
  interval: "1d"

macro:
  provider: "fred"
  series: ["FEDFUNDS", "CPIAUCSL", "UNRATE", "DGS10"]
  fred_api_key: null

fundamentals:
  provider: "yfinance"
  fields:
    - "Total Revenue"
    - "Net Income"
    - "Total Assets"
    - "Total Liab"
    - "Operating Cash Flow"

output:
  dir_bronze: "data/bronze"
  dir_silver: "data/silver"
  save_format: "parquet"
""".strip()


# (Funkcja load_config pozostaje bez zmian jak w oryginale)
def load_config(path: Optional[str]) -> dict:
    explicit = path or os.environ.get("ML_CONFIG")
    if explicit:
        cfg_path = Path(explicit).expanduser().resolve()
        with open(cfg_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    script_dir = Path(__file__).resolve().parent
    candidate1 = (script_dir.parent / "config.yaml")
    if candidate1.exists():
        with open(candidate1, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    default_target = script_dir.parent / "config.yaml"
    default_target.parent.mkdir(parents=True, exist_ok=True)
    with open(default_target, "w", encoding="utf-8") as f:
        f.write(DEFAULT_CONFIG + "\n")
    logging.info(f"Utworzono domyślny config: {default_target}")
    return yaml.safe_load(DEFAULT_CONFIG)


# ==============
# FEATURE ENGINEERING
# ==============

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Dodaje wskaźniki techniczne i stopy zwrotu."""
    df = df.copy()

    # Logarytmiczne stopy zwrotu (stacjonarność)
    df["log_ret"] = np.log(df["close"] / df["close"].shift(1))

    # Wskaźniki z pandas_ta
    # RSI: momentum
    df.ta.rsi(length=14, append=True)
    # MACD: trend
    df.ta.macd(fast=12, slow=26, signal=9, append=True)
    # ATR: zmienność
    df.ta.atr(length=14, append=True)
    # Bollinger Bands
    df.ta.bbands(length=20, std=2, append=True)

    # Procentowa zmiana wolumenu
    df["vol_pct_change"] = df["volume"].pct_change()

    return df

# ==============
# POMOCNICZA: Standaryzacja czasu
# ==============
def strip_tz(df: pd.DataFrame) -> pd.DataFrame:
    """Usuwa informację o strefie czasowej z indeksu, czyniąc go 'naiwnym'."""
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    return df


def fetch_fred_series(series_id, api_key, start, end) -> pd.DataFrame:
    key = api_key or os.environ.get("FRED_API_KEY")
    if not key:
        raise RuntimeError("Brak klucza FRED API.")

    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": series_id,
        "api_key": key,
        "file_type": "json",
        "observation_start": start,
        "observation_end": end
    }

    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()

    df = pd.DataFrame(data["observations"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")[["value"]]
    df = df.rename(columns={"value": f"fred_{series_id}"})
    return strip_tz(df)  # Czyścimy strefę czasową
# ==============
# POBIERANIE DANYCH (OHLCV, FRED, FUNDAMENTY)
# ==============

def fetch_ohlcv_yf(ticker: str, start, end, interval="1d", auto_adjust=True) -> pd.DataFrame:
    df = yf.download(ticker, start=start, end=end, interval=interval,
                     auto_adjust=auto_adjust, progress=False)
    if df.empty:
        raise ValueError(f"Brak danych dla {ticker}")

    df = coerce_datetime_index(df)

    # Naprawa MultiIndex (yfinance 0.2.x często zwraca poziomy 'Price' i 'Ticker')
    if isinstance(df.columns, pd.MultiIndex):
        # Wybieramy poziom, który zawiera nazwy pól (Open, Close, itp.)
        if "Price" in df.columns.names:
            df.columns = df.columns.get_level_values("Price")
        else:
            df.columns = df.columns.get_level_values(0)

    # Standaryzacja nazw kolumn na małe litery i podkreślniki
    df.columns = [str(c).lower().replace(" ", "_") for c in df.columns]

    # Jeśli użyliśmy auto_adjust=True, yf może nazwać główną cenę 'adj_close' lub 'close'.
    # Mapujemy to na jedną nazwę 'close' dla spójności pipeline'u.
    if 'adj_close' in df.columns and 'close' not in df.columns:
        df = df.rename(columns={'adj_close': 'close'})

    # Upewnienie się, że mamy niezbędną kolumnę
    if 'close' not in df.columns:
        # Próba ratunkowa: jeśli jest tylko jedna kolumna, to prawdopodobnie nasza cena
        if len(df.columns) == 1:
            df.columns = ['close']
        else:
            raise KeyError(f"Nie znaleziono kolumny 'close' w danych dla {ticker}. Dostępne: {list(df.columns)}")

    return df


def fetch_fundamentals_yf(ticker: str, fields: List[str]) -> pd.DataFrame:
    tk = yf.Ticker(ticker)
    # Łączymy wszystkie dostępne sprawozdania
    frames = []
    for df in [tk.quarterly_financials, tk.quarterly_balance_sheet, tk.quarterly_cashflow]:
        if df is not None and not df.empty:
            frames.append(df.T)

    if not frames: return pd.DataFrame()

    fundamentals = pd.concat(frames, axis=1, sort=False) # Dodano sort=False
    fundamentals.index = pd.to_datetime(fundamentals.index, utc=True)

    if fields:
        available = [c for c in fields if c in fundamentals.columns]
        fundamentals = fundamentals[available]

    fundamentals.columns = [f"fund_{str(c).lower().replace(' ', '_')}" for c in fundamentals.columns]
    return fundamentals.sort_index()


# ==============
# SCALANIE I ELIMINACJA LOOK-AHEAD BIAS
# ==============

# ==============
# SCALANIE (Z poprawką join i tz)
# ==============
def align_and_merge(
        main_price: pd.DataFrame,
        benchmarks: Dict[str, pd.DataFrame],
        macro: pd.DataFrame,
        fundamentals: pd.DataFrame,
        lag_days: int = 60
) -> pd.DataFrame:
    # 1. Baza
    merged = to_business_daily(main_price)
    merged = strip_tz(merged)  # Na wszelki wypadek

    # Dodajemy wskaźniki techniczne
    merged = add_technical_indicators(merged)

    # 2. Benchmarks
    for t, df in benchmarks.items():
        bench_daily = to_business_daily(strip_tz(df))
        # Logarytmiczna stopa zwrotu dla benchmarku
        merged[f"bench_ret_{t}"] = np.log(bench_daily["close"] / bench_daily["close"].shift(1))

    # 3. Makro (FRED)
    if not macro.empty:
        macro_d = to_business_daily(strip_tz(macro))
        merged = merged.join(macro_d, how="left").ffill()

    # 4. Fundamenty (Z bezpiecznym przesunięciem)
    if not fundamentals.empty:
        fund_shifted = strip_tz(fundamentals.copy())
        # Przesuwamy daty o lag_days (dni kalendarzowe)
        fund_shifted.index = fund_shifted.index + pd.Timedelta(days=lag_days)

        # Join po usunięciu stref czasowych przejdzie bez błędu
        merged = merged.join(fund_shifted, how="left")
        merged[fund_shifted.columns] = merged[fund_shifted.columns].ffill()

    # 5. Generowanie TARGETU
    merged["target_next_5d"] = merged["log_ret"].shift(-5).rolling(window=5).sum()
    merged = merged.dropna(subset=["target_next_5d"])

    return merged.sort_index()


# ==============
# RUNNER
# ==============

def run_pipeline(config_path: Optional[str] = None):
    cfg = load_config(config_path)

    # Pobieranie danych (skrócone wywołania)
    main_price = fetch_ohlcv_yf(cfg["ticker"], cfg["start"], cfg["end"])

    benchmarks = {}
    for t in cfg["benchmarks"]["tickers"]:
        try:
            benchmarks[t] = fetch_ohlcv_yf(t, cfg["start"], cfg["end"])
        except:
            logging.warning(f"Pominięto benchmark {t}")

    macro = pd.DataFrame()
    if cfg["macro"]["provider"] == "fred":
        try:
            frames = [fetch_fred_series(s, cfg["macro"]["fred_api_key"], cfg["start"], cfg["end"])
                      for s in cfg["macro"]["series"]]
            macro = pd.concat(frames, axis=1, sort=False)  # Dodano sort=False
        except Exception as e:
            logging.warning(f"Błąd FRED: {e}")

    fundamentals = pd.DataFrame()
    if cfg["fundamentals"]["provider"] == "yfinance":
        fundamentals = fetch_fundamentals_yf(cfg["ticker"], cfg["fundamentals"]["fields"])

    # Scalanie z nową logiką
    logging.info("Scalanie danych i generowanie cech...")
    merged = align_and_merge(
        main_price, benchmarks, macro, fundamentals,
        lag_days=cfg.get("fundamentals_lag_days", 60)
    )

    # Zapis
    out_dir = Path(cfg["output"]["dir_silver"])
    ensure_dir(out_dir)
    save_path = out_dir / f"{cfg['ticker']}_final_ml_data.parquet"
    save_df(merged, str(save_path), "parquet")

    logging.info(f"Pipeline zakończony. Rozmiar końcowy: {merged.shape}")
    return merged


if __name__ == "__main__":
    run_pipeline()