import numpy as np
import pandas as pd
import requests
import time
import os
import joblib

# Скрываем красные предупреждения TensorFlow
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Input

# --- КОНФИГУРАЦИЯ ---
SYMBOLS = ["BTCUSDT", "ETHUSDT"]

# Новые интервалы
INTERVALS = ["5m", "15m", "30m", "1h", "2h", "4h", "8h", "1d"]

# Новые периоды (добавили 2y)
PERIODS = ["3m", "6m", "1y", "2y"]

MODELS_DIR = "models"
if not os.path.exists(MODELS_DIR):
    os.makedirs(MODELS_DIR)


def get_candles_count(interval, period_str):
    """
    Переводит период (3m, 2y) в количество свечей.
    """
    minutes_in_period = 0
    if period_str.endswith("m"):  # Месяцы
        months = int(period_str.replace("m", ""))
        minutes_in_period = months * 30 * 24 * 60
    elif period_str.endswith("y"):  # Годы
        years = int(period_str.replace("y", ""))
        minutes_in_period = years * 365 * 24 * 60

    minutes_map = {
        "1m": 1,
        "5m": 5,
        "15m": 15,
        "30m": 30,
        "1h": 60,
        "2h": 120,
        "4h": 240,
        "8h": 480,
        "1d": 1440,
    }
    interval_minutes = minutes_map.get(interval, 60)

    # Итого свечей
    return int(minutes_in_period / interval_minutes)


def fetch_binance_data(symbol, interval, limit_total):
    print(f"   [Download] Качаем {limit_total} свечей для {symbol} ({interval})...")

    # Отрезаем последние 2 недели (Test Set)
    two_weeks_ms = 14 * 24 * 60 * 60 * 1000
    cutoff_time = int(time.time() * 1000) - two_weeks_ms

    end_time = cutoff_time
    all_data = []
    limit_per_req = 1000

    while len(all_data) < limit_total:
        remaining = limit_total - len(all_data)
        req_limit = min(limit_per_req, remaining)
        if req_limit < 100:
            req_limit = 100

        url = "https://api.binance.com/api/v3/klines"
        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": req_limit,
            "endTime": end_time,
        }
        try:
            r = requests.get(url, params=params)
            data = r.json()
            if not isinstance(data, list) or len(data) == 0:
                break

            end_time = data[0][0] - 1
            all_data = data + all_data
            time.sleep(0.05)
        except Exception as e:
            print(f"   [Error] {e}")
            break

    if len(all_data) > limit_total:
        all_data = all_data[-limit_total:]

    df = pd.DataFrame(
        all_data,
        columns=[
            "Open Time",
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
            "Close Time",
            "QAV",
            "NT",
            "TBB",
            "TBQ",
            "I",
        ],
    )
    return df["Close"].astype(float).values


def train_and_save(symbol, interval, period):
    filename_base = f"{MODELS_DIR}/{symbol}_{interval}_{period}"

    if os.path.exists(f"{filename_base}.keras"):
        print(f"-> Модель {symbol} {interval} {period} уже существует. Пропускаем.")
        return

    print(f"\n=== СТАРТ: {symbol} | Интервал: {interval} | Период: {period} ===")

    count = get_candles_count(interval, period)
    # Ограничиваем разумным пределом (чтобы 2 года 5-минуток не качать полчаса)
    # 200 000 свечей достаточно для любой нейронки
    if count > 200000:
        count = 200000

    dataset = fetch_binance_data(symbol, interval, count)

    # Если данных критически мало (меньше 60 свечей), пропускаем
    if len(dataset) < 60:
        print(f"   [Skip] Слишком мало данных ({len(dataset)}).")
        return

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset.reshape(-1, 1))

    # --- УМНЫЙ LOOKBACK ---
    # Если история длинная -> смотрим на 120 шагов назад
    # Если история короткая (например 1d за 3m) -> смотрим на 30 шагов
    lookback = 120
    if len(dataset) < 500:
        lookback = 30
        print(f"   [Config] Мало данных, уменьшаем окно обзора до {lookback}")

    # Проверка, хватает ли данных даже для маленького окна
    if len(dataset) <= lookback + 10:
        print(
            f"   [Skip] Недостаточно данных для обучения ({len(dataset)} vs {lookback})"
        )
        return

    x_train, y_train = [], []
    for i in range(lookback, len(scaled_data)):
        x_train.append(scaled_data[i - lookback : i, 0])
        y_train.append(scaled_data[i, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    model = Sequential()
    model.add(Input(shape=(x_train.shape[1], 1)))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(25))
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mean_squared_error")

    print(f"   [Training] Обучение (Lookback={lookback})...")
    # batch_size зависит от объема данных
    bs = 32 if len(dataset) < 1000 else 64
    model.fit(x_train, y_train, batch_size=bs, epochs=3, verbose=0)

    model.save(f"{filename_base}.keras")
    joblib.dump(scaler, f"{filename_base}_scaler.pkl")
    print(f"   [Saved] Модель сохранена!")


if __name__ == "__main__":
    print("Начинаем массовое обучение...")
    for symbol in SYMBOLS:
        for interval in INTERVALS:
            for period in PERIODS:
                train_and_save(symbol, interval, period)
    print("\nГОТОВО! Все периоды обучены. Запускай main_api.py")
