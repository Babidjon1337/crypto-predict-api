import numpy as np
import pandas as pd
import requests
import time
import datetime
import os
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import uvicorn

# Отключаем логи TF
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

MODELS_DIR = "models"
PREDICTION_STEPS = 5

app = FastAPI(title="Crypto Neuro API", version="Final-Extended")
loaded_models_cache = {}


class UserRequest(BaseModel):
    symbol: str = "BTCUSDT"
    interval: str = "5m"
    training_period: str = "1y"  # 3m, 6m, 1y, 2y


def get_interval_ms(interval_str):
    # Обновили карту времени
    map_time = {
        "1m": 60000,
        "5m": 300000,
        "15m": 900000,
        "30m": 1800000,
        "1h": 3600000,
        "2h": 7200000,
        "4h": 14400000,
        "8h": 28800000,
        "1d": 86400000,
    }
    return map_time.get(interval_str, 300000)


def fetch_fresh_data(symbol, interval, limit_total):
    """Качает свежие данные"""
    limit_per_req = 1000
    all_data = []
    current_end_time = int(time.time() * 1000)

    print(f"[Binance] Скачивание {limit_total} свечей для прогноза...")

    while len(all_data) < limit_total:
        remaining = limit_total - len(all_data)
        req_limit = min(limit_per_req, remaining)
        if req_limit <= 0:
            break

        url = "https://api.binance.com/api/v3/klines"
        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": req_limit,
            "endTime": current_end_time,
        }

        try:
            r = requests.get(url, params=params)
            data = r.json()
            if not isinstance(data, list) or len(data) == 0:
                break

            all_data = data + all_data
            oldest_open_time = data[0][0]
            current_end_time = oldest_open_time - 1
            if len(all_data) < limit_total:
                time.sleep(0.05)

        except Exception as e:
            print(f"Error fetching data: {e}")
            break

    if len(all_data) > limit_total:
        all_data = all_data[-limit_total:]
    if not all_data:
        return pd.DataFrame()

    df = pd.DataFrame(
        all_data,
        columns=[
            "Open Time",
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
            "CT",
            "QV",
            "NT",
            "TB",
            "TQ",
            "I",
        ],
    )
    return df


def load_model_pipeline(symbol, interval, period):
    model_key = f"{symbol}_{interval}_{period}"
    path_base = f"{MODELS_DIR}/{model_key}"

    if model_key in loaded_models_cache:
        return loaded_models_cache[model_key]

    if not os.path.exists(f"{path_base}.keras"):
        return None, None

    print(f"[Loading] {model_key}...")
    model = load_model(f"{path_base}.keras")
    scaler = joblib.load(f"{path_base}_scaler.pkl")
    loaded_models_cache[model_key] = (model, scaler)
    return model, scaler


# --- ENDPOINTS ---


@app.get("/")
def root():
    if not os.path.exists(MODELS_DIR):
        return {"status": "No models directory found"}
    files = os.listdir(MODELS_DIR)
    models_list = [f for f in files if f.endswith(".keras")]
    return {
        "status": "Running",
        "models_count": len(models_list),
        "available_models": models_list,
    }


@app.post("/predict")
def predict_crypto(req: UserRequest):
    # 1. Загрузка
    model, scaler = load_model_pipeline(req.symbol, req.interval, req.training_period)
    if model is None:
        raise HTTPException(
            status_code=404,
            detail=f"Модель {req.symbol} {req.interval} {req.training_period} не найдена. Обучите её.",
        )

    # 2. Определяем lookback из файла модели
    lookback = model.input_shape[1]

    # 3. Считаем данные
    interval_ms = get_interval_ms(req.interval)
    two_weeks_ms = 14 * 24 * 60 * 60 * 1000
    candles_for_2weeks = int(two_weeks_ms / interval_ms)

    min_needed_for_ai = lookback + 20
    fetch_limit = max(candles_for_2weeks, min_needed_for_ai)

    # 4. Качаем
    df = fetch_fresh_data(req.symbol, req.interval, fetch_limit)
    if df.empty or len(df) < lookback:
        raise HTTPException(status_code=502, detail="Недостаточно данных с Binance")

    dataset = df["Close"].astype(float).values
    current_price = float(dataset[-1])
    last_time = float(df.iloc[-1]["Open Time"])

    # 5. Прогноз
    last_window = dataset[-lookback:]
    current_batch = scaler.transform(last_window.reshape(-1, 1)).reshape(1, lookback, 1)

    predicted_prices = []

    for i in range(PREDICTION_STEPS):
        pred_scaled = model.predict(current_batch, verbose=0)
        val = pred_scaled[0, 0]
        predicted_prices.append(val)
        current_batch = np.append(current_batch[:, 1:, :], [[[val]]], axis=1)

    # Превращаем в Python список (ВАЖНО для JSON)
    result_prices = (
        scaler.inverse_transform(np.array(predicted_prices).reshape(-1, 1))
        .flatten()
        .tolist()
    )

    # 6. Ответ
    detailed_preds = []

    # --- ИСПРАВЛЕНИЕ: ДОБАВЛЯЕМ ТАЙМЗОНУ UTC+3 ---
    tz_offset = datetime.timezone(datetime.timedelta(hours=3))

    for i, price in enumerate(result_prices):
        future_ms = last_time + ((i + 1) * interval_ms)
        # Указываем tz=tz_offset при конвертации
        t_str = datetime.datetime.fromtimestamp(
            future_ms / 1000, tz=tz_offset
        ).strftime("%H:%M")
        detailed_preds.append({"step": i + 1, "time": t_str, "price": float(price)})

    trend = "UP" if result_prices[-1] > current_price else "DOWN"

    # Отдаем ровно 2 недели истории
    history_to_return = dataset
    if len(history_to_return) > candles_for_2weeks:
        history_to_return = history_to_return[-candles_for_2weeks:]

    return {
        "model_used": f"{req.symbol}_{req.interval}_{req.training_period}",
        "current_price": current_price,
        "trend": trend,
        "prediction_next_5_candles": detailed_preds,
        "history_last_candles": history_to_return.tolist(),
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
