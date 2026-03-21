from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[3]

RAW_STOCK_CSV = ROOT_DIR / "data" / "raw" / "stock_price_600519.csv"
SIM_MODEL_PATH = ROOT_DIR / "artifacts" / "models" / "sim_event_logreg.pkl"
SIM_SCALER_PATH = ROOT_DIR / "artifacts" / "models" / "sim_event_scaler.pkl"
SIM_META_PATH = ROOT_DIR / "artifacts" / "models" / "sim_event_meta.json"

POSITIVE_WORDS = [
    "增长", "预增", "分红", "回购", "中标", "合作", "利好", "创新高", "增持", "派息"
]

NEGATIVE_WORDS = [
    "亏损", "下滑", "减持", "风险", "处罚", "问询", "诉讼", "减值", "利空", "下跌"
]


def to_python_value(value: Any) -> Any:
    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
    except Exception:
        pass

    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    return value


def sanitize_for_json(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize_for_json(v) for v in obj]
    if isinstance(obj, tuple):
        return [sanitize_for_json(v) for v in obj]
    return to_python_value(obj)


@lru_cache(maxsize=1)
def load_stock_df() -> pd.DataFrame:
    if not RAW_STOCK_CSV.exists():
        raise FileNotFoundError(f"股票数据不存在: {RAW_STOCK_CSV}")

    df = pd.read_csv(RAW_STOCK_CSV)
    required_cols = ["date", "open", "close", "high", "low", "volume"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"股票数据缺少字段: {missing}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for c in ["open", "close", "high", "low", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["date", "open", "close", "high", "low", "volume"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


@lru_cache(maxsize=1)
def load_sim_model():
    if not SIM_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"模拟预测模型不存在: {SIM_MODEL_PATH}。请先训练并保存 sim_event_logreg.pkl"
        )
    return joblib.load(SIM_MODEL_PATH)


@lru_cache(maxsize=1)
def load_sim_scaler():
    if not SIM_SCALER_PATH.exists():
        raise FileNotFoundError(
            f"模拟预测 scaler 不存在: {SIM_SCALER_PATH}。请先训练并保存 sim_event_scaler.pkl"
        )
    return joblib.load(SIM_SCALER_PATH)


@lru_cache(maxsize=1)
def load_sim_meta():
    if not SIM_META_PATH.exists():
        raise FileNotFoundError(
            f"模拟预测 meta 文件不存在: {SIM_META_PATH}。请先生成 sim_event_meta.json"
        )
    return json.loads(SIM_META_PATH.read_text(encoding="utf-8"))


def count_keywords(text: str, keywords: list[str]) -> int:
    if not isinstance(text, str):
        return 0
    return sum(1 for word in keywords if word in text)


def build_title_features(title: str, event_type: int, event_source: str | None) -> dict[str, Any]:
    title = title if isinstance(title, str) else ""
    source_text = event_source if isinstance(event_source, str) else ""

    positive_count = count_keywords(title, POSITIVE_WORDS)
    negative_count = count_keywords(title, NEGATIVE_WORDS)
    sentiment_score = positive_count - negative_count

    return {
        "event_type": int(event_type),
        "title_length": len(title),
        "positive_keyword_count": positive_count,
        "negative_keyword_count": negative_count,
        "sentiment_score": sentiment_score,
        "is_report": int(int(event_type) == 1),
        "is_dividend": int(int(event_type) == 3),
        "is_governance": int(int(event_type) == 6),
        "is_manual_source": int(source_text.lower() == "manual"),
    }


def build_day_features(curr_row: pd.Series, prev_row: pd.Series | None) -> dict[str, float]:
    open_price = float(curr_row["open"])
    close_price = float(curr_row["close"])
    high_price = float(curr_row["high"])
    low_price = float(curr_row["low"])
    volume = float(curr_row["volume"])

    if prev_row is not None:
        prev_close = float(prev_row["close"])
        prev_volume = float(prev_row["volume"])
    else:
        prev_close = close_price
        prev_volume = volume

    intraday_return = (close_price - open_price) / open_price if open_price else 0.0
    amplitude = (high_price - low_price) / open_price if open_price else 0.0
    volume_change_vs_prev = (volume - prev_volume) / prev_volume if prev_volume else 0.0
    close_change_vs_prev_close = (close_price - prev_close) / prev_close if prev_close else 0.0

    return {
        "intraday_return": intraday_return,
        "amplitude": amplitude,
        "volume_change_vs_prev": volume_change_vs_prev,
        "close_change_vs_prev_close": close_change_vs_prev_close,
    }


def get_previous_context(event_date: str) -> dict[str, Any]:
    """
    取 event_date 之前最近两个交易日，作为新事件模拟预测上下文。
    注意：严格使用 date < event_date，不使用事件日当天数据。
    """
    stock_df = load_stock_df()
    event_dt = pd.to_datetime(event_date)

    prev_df = stock_df[stock_df["date"] < event_dt].copy()
    if len(prev_df) < 3:
        raise ValueError("给定日期之前可用交易日不足，无法构造模拟预测上下文")

    context_df = prev_df.tail(2).copy()
    prev_ref_df = prev_df.tail(3).copy()

    # prev_ref_df 的最后三天： [ref_for_t-2, t-2, t-1]
    ref_row = prev_ref_df.iloc[0]
    t_minus_2 = prev_ref_df.iloc[1]
    t_minus_1 = prev_ref_df.iloc[2]

    t_minus_2_feat = build_day_features(t_minus_2, ref_row)
    t_minus_1_feat = build_day_features(t_minus_1, t_minus_2)

    market_sequence = [
        {
            "date": t_minus_2["date"].strftime("%Y-%m-%d"),
            "open": float(t_minus_2["open"]),
            "close": float(t_minus_2["close"]),
            "high": float(t_minus_2["high"]),
            "low": float(t_minus_2["low"]),
            "volume": float(t_minus_2["volume"]),
            **t_minus_2_feat,
        },
        {
            "date": t_minus_1["date"].strftime("%Y-%m-%d"),
            "open": float(t_minus_1["open"]),
            "close": float(t_minus_1["close"]),
            "high": float(t_minus_1["high"]),
            "low": float(t_minus_1["low"]),
            "volume": float(t_minus_1["volume"]),
            **t_minus_1_feat,
        },
    ]

    feature_dict = {
        "t_minus_2_intraday_return": t_minus_2_feat["intraday_return"],
        "t_minus_2_amplitude": t_minus_2_feat["amplitude"],
        "t_minus_2_volume_change_vs_prev": t_minus_2_feat["volume_change_vs_prev"],
        "t_minus_2_close_change_vs_prev_close": t_minus_2_feat["close_change_vs_prev_close"],
        "t_minus_1_intraday_return": t_minus_1_feat["intraday_return"],
        "t_minus_1_amplitude": t_minus_1_feat["amplitude"],
        "t_minus_1_volume_change_vs_prev": t_minus_1_feat["volume_change_vs_prev"],
        "t_minus_1_close_change_vs_prev_close": t_minus_1_feat["close_change_vs_prev_close"],
    }

    return {
        "context_dates": [
            t_minus_2["date"].strftime("%Y-%m-%d"),
            t_minus_1["date"].strftime("%Y-%m-%d"),
        ],
        "market_sequence": market_sequence,
        "market_feature_dict": feature_dict,
    }


def get_simulate_context(event_date: str) -> dict[str, Any]:
    context = get_previous_context(event_date)
    return sanitize_for_json({
        "event_date": event_date,
        "simulation_mode": "pre_event_v1",
        "context_dates": context["context_dates"],
        "market_sequence": context["market_sequence"],
        "note": "该上下文仅使用事件日前最近两个交易日数据，不包含事件日及未来信息。"
    })


def predict_new_event_simulation(
    event_date: str,
    event_type: int,
    event_source: str | None,
    event_title: str,
) -> dict[str, Any]:
    context = get_previous_context(event_date)
    text_features = build_title_features(event_title, event_type, event_source)

    feature_dict = {}
    feature_dict.update(context["market_feature_dict"])
    feature_dict.update(text_features)

    meta = load_sim_meta()
    feature_columns = meta["feature_columns"]

    x_row = pd.DataFrame([{col: feature_dict.get(col, 0.0) for col in feature_columns}])
    scaler = load_sim_scaler()
    model = load_sim_model()

    x_scaled = scaler.transform(x_row.values)

    if hasattr(model, "predict_proba"):
        prob = float(model.predict_proba(x_scaled)[0, 1])
    elif hasattr(model, "decision_function"):
        score = float(model.decision_function(x_scaled)[0])
        prob = 1 / (1 + np.exp(-score))
    else:
        pred = int(model.predict(x_scaled)[0])
        prob = 0.75 if pred == 1 else 0.25

    label = "上涨" if prob >= 0.5 else "下跌"
    confidence = max(prob, 1 - prob)

    result = {
        "simulation_mode": "pre_event_v1",
        "event_date": event_date,
        "event_type": int(event_type),
        "event_source": event_source,
        "event_title": event_title,
        "model_name": meta.get("model_name", "Simulation V1 Model"),
        "prediction_label": label,
        "probability": round(prob, 4),
        "confidence": round(confidence, 4),
        "context_dates": context["context_dates"],
        "market_sequence": context["market_sequence"],
        "text_features": text_features,
        "note": "该结果为事件发生前模拟预测，不包含事件日及未来信息。"
    }
    return sanitize_for_json(result)