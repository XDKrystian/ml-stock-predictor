import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


def train_stock_classification():
    # 1. Wczytanie danych (teraz z kolumną target_bin)
    df = pd.read_parquet("data/silver/AAPL_final_ml_data.parquet")

    # Sprawdźmy dostępne kolumny na wszelki wypadek
    target_col = 'target_bin'
    if target_col not in df.columns:
        raise KeyError(f"Nie znaleziono kolumny {target_col}. Dostępne kolumny: {df.columns.tolist()}")

    # 2. Przygotowanie X i y
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # 3. Podział chronologiczny (80% trening, 20% test)
    split_index = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

    print(f"[INFO] Trening na danych do: {X_train.index.max()}")
    print(f"[INFO] Test na danych od: {X_test.index.min()}")

    # 4. Model XGBoost Classifier (zamiast Regressor)
    model = xgb.XGBClassifier(
        n_estimators=1000,
        learning_rate=0.005,
        max_depth=3,
        scale_pos_weight=0.8,  # Dodaj to: wyrównuje wagę między spadkami a wzrostami
        subsample=0.6,  # Zmniejszamy, by bardziej urozmaicić naukę
        colsample_bytree=0.6,
        objective='binary:logistic',
        early_stopping_rounds=50,
        random_state=42
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=100
    )

    # 5. Ewaluacja
    probs = model.predict_proba(X_test)[:, 1]

    # DEBUG: Sprawdźmy co on właściwie wyliczył
    print(f"\nStatystyki prawdopodobieństw:")
    print(f"Min: {probs.min():.4f}, Max: {probs.max():.4f}, Średnia: {probs.mean():.4f}")

    # Wybieramy próg 0.50 jako bazowy do metryk
    preds = (probs > 0.50).astype(int)
    acc = accuracy_score(y_test, preds)  # Definiujemy 'acc' dla printa poniżej

    print(f"\n[WYNIKI - KLASYFIKACJA]")
    print(f"Accuracy (Celność ogólna dla progu 0.50): {acc:.2%}")

    # Tabela progów (zostaje jako informacja dodatkowa)
    print(f"\n{'Próg':<10} | {'Celność':<10} | {'Sygnały Kupna (1)':<20}")
    print("-" * 45)
    for thr in [0.45, 0.50, 0.55]:
        p = (probs > thr).astype(int)
        print(f"{thr:<10.2f} | {accuracy_score(y_test, p):<10.2%} | {p.sum():<20}")

    # 6. Wizualizacja Confusion Matrix (Macierz pomyłek)
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Macierz Pomyłek (0: Spadek/Brak zmian, 1: Wzrost)")
    plt.xlabel("Przewidziane")
    plt.ylabel("Rzeczywiste")
    plt.show()

    # 7. Ważność cech
    plt.figure(figsize=(10, 8))
    xgb.plot_importance(model, max_num_features=15)
    plt.show()


if __name__ == "__main__":
    train_stock_classification()