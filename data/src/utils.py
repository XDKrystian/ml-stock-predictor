# src/utils.py
from __future__ import annotations
import os
from typing import Optional
import pandas as pd

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def coerce_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    """Zapewnia DatetimeIndex i USTALA indeks na 'naive UTC' (bez strefy)."""
    df = df.copy()
    # 1) Jeśli nie jest DatetimeIndex -> parsuj z utc=True
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True, errors="coerce")

    # 2) Jeśli jest tz-aware -> konwertuj do UTC i usuń strefę
    if getattr(df.index, "tz", None) is not None:
        df.index = df.index.tz_convert("UTC").tz_localize(None)
    else:
        # Jest tz-naive -> pozostaw jako naive (już OK)
        pass

    return df.sort_index()



def to_business_daily(df: pd.DataFrame, method: str = "ffill") -> pd.DataFrame:
    """Resampluje do biznesowych dni i wypełnia luki (makro/fundamenty)."""
    df = coerce_datetime_index(df)
    out = df.resample("B").last()
    if method:
        if method == "ffill":
            out = out.ffill()
        elif method == "bfill":
            out = out.bfill()
        else:
            # fallback – bezpieczne zachowanie
            out = out
    return out


def save_df(df: pd.DataFrame, path: str, fmt: str = "parquet") -> None:
    ensure_dir(os.path.dirname(path))
    if fmt == "parquet":
        df.to_parquet(path, engine="pyarrow")
    elif fmt == "csv":
        df.to_csv(path, index=True)
    else:
        raise ValueError(f"Nieobsługiwany format: {fmt}")

def add_suffix(df: pd.DataFrame, suffix: str) -> pd.DataFrame:
    df = df.copy()
    df.columns = [f"{c}{suffix}" for c in df.columns]
    return df