import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def validate_ml_data(file_path: str):
    if not Path(file_path).exists():
        print(f"[ERROR] Nie znaleziono pliku: {file_path}")
        return

    df = pd.read_parquet(file_path)

    print("=== PODSTAWOWE INFORMACJE ===")
    print(f"Liczba wierszy: {df.shape[0]}")
    print(f"Liczba kolumn: {df.shape[1]}")
    print(f"Zakres dat: {df.index.min()} do {df.index.max()}")

    print("\n=== BRAKUJĄCE DANE (Top 10) ===")
    nan_counts = df.isna().sum()
    print(nan_counts[nan_counts > 0].sort_values(ascending=False).head(10))

    # 1. Sprawdzenie korelacji z targetem
    print("\n=== KORELACJA CECH Z TARGETEM ===")
    # Wybieramy tylko kolumny numeryczne
    numeric_df = df.select_dtypes(include=[np.number])
    correlations = numeric_df.corr()['target_next_5d'].sort_values(ascending=False)

    print("Najmocniejsze dodatnie:")
    print(correlations.head(5))
    print("\nNajmocniejsze ujemne:")
    print(correlations.tail(5))

    # Wizualizacja
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Wykres 1: Rozkład Targetu
    sns.histplot(df['target_next_5d'], kde=True, ax=ax1, color='teal')
    ax1.set_title("Rozkład Targetu (Log Returns 5d)")
    ax1.axvline(0, color='red', linestyle='--')

    # Wykres 2: Heatmapa korelacji (wybrane cechy)
    top_features = correlations.abs().sort_values(ascending=False).head(15).index
    sns.heatmap(df[top_features].corr(), annot=True, fmt=".2f", cmap='coolwarm', ax=ax2)
    ax2.set_title("Top 15 najlepiej skorelowanych cech")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Podstaw ścieżkę do swojego pliku silver
    PATH_TO_DATA = "data/silver/AAPL_final_ml_data.parquet"
    validate_ml_data(PATH_TO_DATA)