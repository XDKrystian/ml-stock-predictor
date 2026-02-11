
# src/data_pipeline.py
from __future__ import annotations

import os
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml
import pandas as pd
import numpy as np
import requests
import yfinance as yf

from utils import (
    ensure_dir,
    coerce_datetime_index,
    to_business_daily,
    save_df,
    add_suffix,
)

from dotenv import load_dotenv

load_dotenv()

# ==============
# CONFIG (default)
# ==============
DEFAULT_CONFIG = """
ticker: "AAPL"
currency: "USD"
start: "2010-01-01"
end: null
price_interval: "1d"
auto_adjust: true

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
    - "Free Cash Flow"
    - "Earnings Per Share"

output:
  dir_bronze: "data/bronze"
  dir_silver: "data/silver"
  save_format: "parquet"
""".strip()


def load_config(path: Optional[str]) -> dict:
    """
    Priorytety:
    1) jawna ścieżka z argumentu --config lub env ML_CONFIG
    2) katalog skryptu/../config.yaml
    3) working directory ./config.yaml
    4) jeśli brak pliku -> zapis domyślnego do ../config.yaml i użycie
    """
    # 1) ENV / argument
    explicit = path or os.environ.get("ML_CONFIG")
    if explicit:
        cfg_path = Path(explicit).expanduser().resolve()
        if not cfg_path.exists():
            raise FileNotFoundError(f"Nie znaleziono configu pod ścieżką: {cfg_path}")
        with open(cfg_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    # 2) szukaj obok skryptu (../config.yaml względem src/)
    script_dir = Path(__file__).resolve().parent
    candidate1 = (script_dir.parent / "config.yaml")
    if candidate1.exists():
        with open(candidate1, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    # 3) working directory
    candidate2 = Path("config.yaml").resolve()
    if candidate2.exists():
        with open(candidate2, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    # 4) brak — zapisz domyślny do ../config.yaml
    default_target = script_dir.parent / "config.yaml"
    default_target.parent.mkdir(parents=True, exist_ok=True)
    with open(default_target, "w", encoding="utf-8") as f:
        f.write(DEFAULT_CONFIG + "\n")
    print(f"[INFO] Brak config.yaml — utworzyłem domyślny: {default_target}")

    return yaml.safe_load(DEFAULT_CONFIG)


# ==============
# OHLCV (główna spółka i benchmarki)
# ==============

def fetch_ohlcv_yf(
    ticker: str,
    start: Optional[str],
    end: Optional[str],
    interval: str = "1d",
    auto_adjust: bool = True,
) -> pd.DataFrame:
    """
    Pobiera OHLCV z yfinance.
    Zwraca kolumny: Open, High, Low, Close, Adj Close, Volume (nazwy znormalizowane do lowercase)
    Obsługuje MultiIndex zwracany przez yfinance (flatten).
    """
    df = yf.download(
        tickers=ticker,
        start=start,
        end=end,
        interval=interval,
        auto_adjust=auto_adjust,
        progress=False,
        threads=True,
        group_by="column",  # <-- kluczowe: spróbuj wymusić kolumny per pole, nie per ticker
    )

    if df is None or len(df) == 0:
        raise ValueError(f"Brak danych dla {ticker} w yfinance.")

    df = coerce_datetime_index(df)

    # ---- Obsługa MultiIndex w kolumnach (np. ('AAPL','Open')) ----
    if isinstance(df.columns, pd.MultiIndex):
        # Jeśli to MultiIndex (ticker, field), a jest tylko jeden ticker -> zrzucamy poziom tickera
        lvl0 = df.columns.get_level_values(0)
        if getattr(lvl0, "nunique", lambda: len(set(lvl0)))() == 1:
            df.columns = df.columns.get_level_values(-1)
        else:
            # Wielu tickerów – spłaszczamy "TICKER__FIELD"
            df.columns = ["__".join([str(x) for x in tup if x is not None]) for tup in df.columns]

    # Teraz powinna być zwykła Index z nazwami kolumn (string)
    df.columns = [str(c).lower().replace(" ", "_") for c in df.columns]

    return df


def fetch_benchmarks_yf(
    tickers: List[str],
    start: Optional[str],
    end: Optional[str],
    interval: str = "1d",
    auto_adjust: bool = True,
) -> Dict[str, pd.DataFrame]:
    out = {}
    for t in tickers:
        try:
            df = fetch_ohlcv_yf(t, start, end, interval, auto_adjust)
            out[t] = df
        except Exception as e:
            print(f"[WARN] Nie udało się pobrać benchmarku {t}: {e}")
    return out


# ==============
# Makro (FRED przez oficjalne API – requests)
# ==============
def fetch_fred_series(
    series_id: str,
    api_key: Optional[str],
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> pd.DataFrame:
    """
    Pobiera jedną serię FRED przez oficjalne API.
    Wymaga klucza API (parametr lub zmienna środowiskowa FRED_API_KEY).
    """
    key = api_key or os.environ.get("FRED_API_KEY")
    if not key:
        raise RuntimeError(
            "Brak klucza FRED API. Ustaw w config.yaml (macro.fred_api_key) lub zmiennej środowiskowej FRED_API_KEY."
        )

    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": series_id,
        "api_key": key,
        "file_type": "json",
    }
    if start:
        params["observation_start"] = start
    if end:
        params["observation_end"] = end

    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()

    if "observations" not in data:
        raise ValueError(f"Brak danych FRED dla {series_id}")

    df = pd.DataFrame(data["observations"])
    # Pola: date, value (string)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["date"] = pd.to_datetime(df["date"], utc=True)
    df = df.set_index("date")[["value"]]
    df = df.rename(columns={"value": f"fred_{series_id}"})

    df = coerce_datetime_index(df)

    return df


def fetch_macro_fred(
    series_list: List[str],
    start: Optional[str],
    end: Optional[str],
    fred_api_key: Optional[str] = None,
) -> pd.DataFrame:
    frames = []
    for s in series_list:
        try:
            df = fetch_fred_series(s, api_key=fred_api_key, start=start, end=end)
            frames.append(df)
        except Exception as e:
            print(f"[WARN] Błąd pobierania FRED {s}: {e}")
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, axis=1).sort_index()


# ==============
# Fundamenty (yfinance)
# ==============
def fetch_fundamentals_yf(ticker: str, fields: List[str]) -> pd.DataFrame:
    """
    Pobiera kwartalne fundamenty z yfinance i składa w jedną tabelę (as-of, w dni robocze).
    yfinance zwraca osobne ramki dla financials/balance_sheet/cashflow/earnings (kolumny=daty).
    """
    tk = yf.Ticker(ticker)

    q_fin = tk.quarterly_financials if tk.quarterly_financials is not None else pd.DataFrame()
    q_bs = tk.quarterly_balance_sheet if tk.quarterly_balance_sheet is not None else pd.DataFrame()
    q_cf = tk.quarterly_cashflow if tk.quarterly_cashflow is not None else pd.DataFrame()
    q_earn = tk.quarterly_earnings if tk.quarterly_earnings is not None else pd.DataFrame()

    frames = []
    for name, df in [("fin", q_fin), ("bs", q_bs), ("cf", q_cf), ("earn", q_earn)]:
        if df.empty:
            continue
        df = df.copy()
        if not isinstance(df.columns, pd.DatetimeIndex):
            try:
                df.columns = pd.to_datetime(df.columns, utc=True, errors="coerce")
            except Exception:
                pass
        df = df.T
        df.columns = [str(c) for c in df.columns]
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    fundamentals = pd.concat(frames, axis=1)
    fundamentals.index.name = "report_date"
    fundamentals = fundamentals.sort_index()

    if fields:
        available = [c for c in fields if c in fundamentals.columns]
        fundamentals = fundamentals[available] if available else fundamentals

    fundamentals_daily = to_business_daily(fundamentals, method="ffill")
    fundamentals_daily.columns = [f"fund_{c}" for c in fundamentals_daily.columns]
    return fundamentals_daily


# ==============
# Scalanie wszystkiego w jedną ramkę dzienną
# ==============
def align_and_merge(
    main_price: pd.DataFrame,
    benchmarks: Dict[str, pd.DataFrame],
    macro: pd.DataFrame,
    fundamentals_daily: pd.DataFrame,
) -> pd.DataFrame:
    main_d = to_business_daily(main_price)
    main_d = main_d.rename(columns={c: f"{c}" for c in main_d.columns})

    bench_cols = []
    for t, df in benchmarks.items():
        d = to_business_daily(df)
        d = add_suffix(d, f"__{t}")
        bench_cols.append(d)

    merged = main_d.copy()
    for b in bench_cols:
        merged = merged.join(b, how="left")

    if macro is not None and not macro.empty:
        macro_d = to_business_daily(macro)
        merged = merged.join(macro_d, how="left")

    if fundamentals_daily is not None and not fundamentals_daily.empty:
        merged = merged.join(fundamentals_daily, how="left")

    # === UZUPELNIANIE FUNDAMENTÓW ===
    fund_cols = [c for c in merged.columns if c.startswith("fund_")]

    # 1) Maska informująca model o braku raportu w danym dniu
    for c in fund_cols:
        merged[c + "_isna"] = merged[c].isna().astype(int)

    # 2) Forward fill (od publikacji do kolejnego dnia)
    merged[fund_cols] = merged[fund_cols].ffill()

    # 3) Backfill (uzupełnienie wcześniejszych okresów)
    merged[fund_cols] = merged[fund_cols].bfill()

    merged = merged.sort_index()

    # === UZUPELNIANIE MAKO ===
    macro_cols = [c for c in merged.columns if c.startswith("fred_")]
    if macro_cols:
        merged[macro_cols] = merged[macro_cols].ffill()

    # Możesz dodać bfill dla makro (opcjonalnie), ale raczej nie trzeba
    # merged[macro_cols] = merged[macro_cols].bfill()

    merged = merged.dropna(how="all")
    return merged


# ==============
# Główny runner
# ==============
def run_pipeline(config_path: Optional[str] = None) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cfg = load_config(config_path)

    ticker = cfg["ticker"]
    start = cfg.get("start")
    end = cfg.get("end")
    price_interval = cfg.get("price_interval", "1d")
    auto_adjust = cfg.get("auto_adjust", True)

    # Dane główne (OHLCV)
    print(f"[INFO] Pobieram OHLCV dla: {ticker}")
    main_price = fetch_ohlcv_yf(
        ticker=ticker,
        start=start,
        end=end,
        interval=price_interval,
        auto_adjust=auto_adjust,
    )

    # Benchmarks
    bm_cfg = cfg.get("benchmarks", {})
    bm_tickers = bm_cfg.get("tickers", [])
    bm_interval = bm_cfg.get("interval", "1d")
    print(f"[INFO] Pobieram benchmarki: {bm_tickers}")
    benchmarks = fetch_benchmarks_yf(
        tickers=bm_tickers,
        start=start,
        end=end,
        interval=bm_interval,
        auto_adjust=True,
    )

    # Makro (FRED)
    macro_cfg = cfg.get("macro", {})
    macro = pd.DataFrame()
    if macro_cfg.get("provider") == "fred":
        series = macro_cfg.get("series", [])
        fred_key = macro_cfg.get("fred_api_key")
        if series:
            print(f"[INFO] Pobieram FRED serie: {series}")
            try:
                macro = fetch_macro_fred(series_list=series, start=start, end=end, fred_api_key=fred_key)
            except RuntimeError as e:
                print(f"[WARN] {e} — pomijam makro FRED (ustaw klucz, aby pobierać).")

    # Fundamenty (yfinance)
    fund_cfg = cfg.get("fundamentals", {})
    fundamentals_daily = pd.DataFrame()
    if fund_cfg.get("provider") == "yfinance":
        fields = fund_cfg.get("fields", [])
        print(f"[INFO] Pobieram fundamenty yfinance (pola: {fields if fields else 'wszystkie dostępne'})")
        try:
            fundamentals_daily = fetch_fundamentals_yf(ticker, fields)
        except Exception as e:
            print(f"[WARN] Fundamentals yfinance dla {ticker}: {e}")

    # Scalanie
    print("[INFO] Scalanie danych do siatki dni roboczych…")
    merged = align_and_merge(main_price, benchmarks, macro, fundamentals_daily)

    # Zapisy
    out_cfg = cfg.get("output", {})
    dir_bronze = out_cfg.get("dir_bronze", "data/bronze")
    dir_silver = out_cfg.get("dir_silver", "data/silver")
    save_format = out_cfg.get("save_format", "parquet")

    ensure_dir(dir_bronze)
    ensure_dir(dir_silver)

    # Bronze (raw)
    save_df(main_price, os.path.join(dir_bronze, f"{ticker}_ohlcv.{save_format}"), save_format)
    for t, df in benchmarks.items():
        save_df(df, os.path.join(dir_bronze, f"benchmark_{t.replace('^','IDX_')}.{save_format}"), save_format)
    if macro is not None and not macro.empty:
        save_df(macro, os.path.join(dir_bronze, f"macro_fred.{save_format}"), save_format)
    if fundamentals_daily is not None and not fundamentals_daily.empty:
        save_df(fundamentals_daily, os.path.join(dir_bronze, f"{ticker}_fundamentals_daily.{save_format}"), save_format)

    # Silver (merged, business-daily)
    save_df(merged, os.path.join(dir_silver, f"{ticker}_merged_daily.{save_format}"), save_format)

    print("[OK] Zakończono. Kształty:")
    print(f"  OHLCV: {main_price.shape}")
    print(f"  Benchmarks: {[ (k, v.shape) for k, v in benchmarks.items() ]}")
    print(f"  Macro: {macro.shape}")
    print(f"  Fundamentals (daily): {fundamentals_daily.shape}")
    print(f"  Merged: {merged.shape}")

    return main_price, benchmarks, macro, fundamentals_daily, merged


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default=None, help="Ścieżka do config.yaml")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    _ = run_pipeline(args.config)
    print("Pipeline zakończony. Dane zapisane w data/bronze i data/silver.")
