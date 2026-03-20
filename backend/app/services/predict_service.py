from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn


ROOT_DIR = Path(__file__).resolve().parents[3]

EVENTS_CSV = ROOT_DIR / "data" / "interim" / "events.csv"
EVENT_WINDOWS_CSV = ROOT_DIR / "data" / "interim" / "event_windows.csv"
SCALER_PATH = ROOT_DIR / "artifacts" / "models" / "lstm_scaler.pkl"
MODEL_PATH = ROOT_DIR / "artifacts" / "models" / "event_lstm_torch.pt"

RAW_STOCK_CSV = ROOT_DIR / "data" / "raw" / "stock_price_600519.csv"

# 如果你的 train_lstm.py 里实际参数不同，只改这里
DEFAULT_MODEL_CONFIG = {
    "input_size": 4,
    "hidden_size": 32,
    "num_layers": 1,
    "dropout": 0.0,
}


class EventLSTM(nn.Module):
    def __init__(
        self,
        input_size: int = 4,
        hidden_size: int = 32,
        num_layers: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        real_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=real_dropout,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.dropout(out)
        logits = self.classifier(out).squeeze(-1)
        return logits


def ensure_exists(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"文件不存在: {path}")
    
def to_python_value(value):
    if value is None:
        return None

    # pandas 的空值
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass

    # numpy 标量 -> python 标量
    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        return float(value)

    if isinstance(value, np.bool_):
        return bool(value)

    # pandas 时间
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")

    return value


def sanitize_for_json(obj):
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}

    if isinstance(obj, list):
        return [sanitize_for_json(v) for v in obj]

    if isinstance(obj, tuple):
        return [sanitize_for_json(v) for v in obj]

    return to_python_value(obj)


@lru_cache(maxsize=1)
def load_events_df() -> pd.DataFrame:
    ensure_exists(EVENTS_CSV)
    df = pd.read_csv(EVENTS_CSV)

    if "event_date" in df.columns:
        df["event_date"] = pd.to_datetime(df["event_date"], errors="coerce").dt.strftime("%Y-%m-%d")

    if "raw_event_date" in df.columns:
        df["raw_event_date"] = pd.to_datetime(df["raw_event_date"], errors="coerce").dt.strftime("%Y-%m-%d")

    if "event_date" in df.columns:
        df = df.sort_values(by="event_date", ascending=False)

    return df.where(pd.notnull(df), None)


@lru_cache(maxsize=1)
def load_event_windows_df() -> pd.DataFrame:
    ensure_exists(EVENT_WINDOWS_CSV)
    df = pd.read_csv(EVENT_WINDOWS_CSV)

    for col in ["date", "event_date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    numeric_cols = ["open", "close", "high", "low", "volume", "event_type"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.sort_values(by=["event_date", "date"], ascending=[False, True])
    return df.where(pd.notnull(df), None)


@lru_cache(maxsize=1)
def load_stock_df() -> pd.DataFrame:
    ensure_exists(RAW_STOCK_CSV)
    df = pd.read_csv(RAW_STOCK_CSV)

    if "date" not in df.columns or "close" not in df.columns:
        raise ValueError("stock_price_600519.csv 必须包含 date 和 close 列")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)
    return df


def get_actual_outcome(event_date: str, pred_label: str | None = None, horizon_days: int = 3) -> dict[str, Any]:
    """
    用真实股价回看 event_date 后第 horizon_days 个交易日的收益率与真实方向。
    """
    stock_df = load_stock_df()
    event_dt = pd.to_datetime(event_date)

    matched_idx = stock_df.index[stock_df["date"] == event_dt].tolist()
    if not matched_idx:
        return {
            "actual_is_available": False,
            "actual_horizon_days": horizon_days,
            "actual_label": None,
            "actual_post_return": None,
            "actual_event_close": None,
            "actual_future_close": None,
            "actual_future_date": None,
            "actual_is_correct": None,
        }

    idx = matched_idx[0]
    if idx + horizon_days >= len(stock_df):
        return {
            "actual_is_available": False,
            "actual_horizon_days": horizon_days,
            "actual_label": None,
            "actual_post_return": None,
            "actual_event_close": float(stock_df.iloc[idx]["close"]),
            "actual_future_close": None,
            "actual_future_date": None,
            "actual_is_correct": None,
        }

    event_close = float(stock_df.iloc[idx]["close"])
    future_row = stock_df.iloc[idx + horizon_days]
    future_close = float(future_row["close"])
    future_date = future_row["date"].strftime("%Y-%m-%d")

    post_return = (future_close - event_close) / event_close if event_close else None
    actual_label = "上涨" if post_return is not None and post_return >= 0 else "下跌"
    actual_is_correct = None if pred_label is None else (pred_label == actual_label)

    return {
        "actual_is_available": True,
        "actual_horizon_days": horizon_days,
        "actual_label": actual_label,
        "actual_post_return": None if post_return is None else float(post_return),
        "actual_event_close": event_close,
        "actual_future_close": future_close,
        "actual_future_date": future_date,
        "actual_is_correct": actual_is_correct,
    }


@lru_cache(maxsize=1)
def load_scaler():
    ensure_exists(SCALER_PATH)
    return joblib.load(SCALER_PATH)


@lru_cache(maxsize=1)
def load_model():
    ensure_exists(MODEL_PATH)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(MODEL_PATH, map_location=device)

    # 情况1：直接保存的是整个模型对象
    if isinstance(checkpoint, nn.Module):
        model = checkpoint
        model.to(device)
        model.eval()
        return model, device

    # 情况2：保存的是包含 model_state_dict 的 checkpoint
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        input_size = int(checkpoint.get("input_size", DEFAULT_MODEL_CONFIG["input_size"]))
        hidden_size = int(checkpoint.get("hidden_size", DEFAULT_MODEL_CONFIG["hidden_size"]))
        num_layers = int(checkpoint.get("num_layers", DEFAULT_MODEL_CONFIG["num_layers"]))
        dropout = float(checkpoint.get("dropout", DEFAULT_MODEL_CONFIG["dropout"]))

        model = EventLSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()
        return model, device

    # 情况3：保存的是 {"state_dict": ..., "model_config": ...}
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        model_config = DEFAULT_MODEL_CONFIG.copy()
        model_config.update(checkpoint.get("model_config", {}))
        model = EventLSTM(**model_config)
        model.load_state_dict(checkpoint["state_dict"])
        model.to(device)
        model.eval()
        return model, device

    # 情况4：保存的是纯 state_dict
    if isinstance(checkpoint, dict):
        model = EventLSTM(**DEFAULT_MODEL_CONFIG)
        model.load_state_dict(checkpoint)
        model.to(device)
        model.eval()
        return model, device

    raise ValueError("无法识别的模型保存格式，请检查 event_lstm_torch.pt")


def list_predict_options(limit: int = 200) -> list[dict[str, Any]]:
    df = load_events_df()
    cols = [c for c in ["event_date", "event_type", "event_source", "event_title", "event_count"] if c in df.columns]
    df = df[cols].head(limit)
    return df.to_dict(orient="records")


def _extract_sequence_rows(event_date: str) -> pd.DataFrame:
    windows_df = load_event_windows_df()
    event_dt = pd.to_datetime(event_date)

    event_df = windows_df[windows_df["event_date"] == event_dt].copy()
    if event_df.empty:
        raise ValueError(f"未找到 event_date={event_date} 对应的事件窗口数据")

    event_df = event_df.sort_values(by="date", ascending=True)

    # 无信息泄露：只使用 事件日前 + 事件日 这两行
    seq_df = event_df[event_df["date"] <= event_dt].tail(2).copy()

    # 兜底：如果因为数据缺失导致不足2行，就直接取前2行
    if len(seq_df) < 2:
        seq_df = event_df.head(2).copy()

    if len(seq_df) != 2:
        raise ValueError(f"event_date={event_date} 的窗口数据不足2行，无法构造 LSTM 输入")

    return seq_df


def _build_feature_sequence(seq_df: pd.DataFrame) -> pd.DataFrame:
    df = seq_df.copy()

    # 这里按你 prepare_lstm_data 日志里的 4 个特征构造
    df["intraday_return"] = (df["close"] - df["open"]) / df["open"]
    df["amplitude"] = (df["high"] - df["low"]) / df["open"]
    df["volume_change_vs_prev"] = df["volume"].pct_change().replace([np.inf, -np.inf], 0.0).fillna(0.0)
    df["close_change_vs_prev_close"] = df["close"].pct_change().replace([np.inf, -np.inf], 0.0).fillna(0.0)

    feature_cols = [
        "intraday_return",
        "amplitude",
        "volume_change_vs_prev",
        "close_change_vs_prev_close",
    ]
    return df[feature_cols].astype("float32")


def _scale_sequence(features_df: pd.DataFrame) -> np.ndarray:
    scaler = load_scaler()
    scaled = scaler.transform(features_df.values)
    return scaled.astype("float32")


def _get_event_meta(event_date: str) -> dict[str, Any]:
    events_df = load_events_df()
    row = events_df[events_df["event_date"] == event_date]

    if row.empty:
        return {
            "event_date": event_date,
            "event_title": None,
            "event_type": None,
            "event_source": None,
            "event_count": None,
        }

    item = row.iloc[0]
    return {
        "event_date": to_python_value(item.get("event_date")),
        "event_title": to_python_value(item.get("event_title")),
        "event_type": to_python_value(item.get("event_type")),
        "event_source": to_python_value(item.get("event_source")),
        "event_count": to_python_value(item.get("event_count")),
    }


def predict_existing_event(event_date: str) -> dict[str, Any]:
    seq_rows = _extract_sequence_rows(event_date)
    feature_df = _build_feature_sequence(seq_rows)
    scaled_seq = _scale_sequence(feature_df)

    model, device = load_model()

    x = torch.tensor(scaled_seq, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        prob = torch.sigmoid(logits).item()

    label = "上涨" if prob >= 0.5 else "下跌"
    confidence = max(prob, 1 - prob)

    meta = _get_event_meta(event_date)

    used_sequence_dates = seq_rows["date"].dt.strftime("%Y-%m-%d").tolist()
    raw_feature_records = []

    seq_rows_display = seq_rows.reset_index(drop=True)
    feature_display = feature_df.reset_index(drop=True)

    for i in range(len(seq_rows_display)):
        raw_feature_records.append(
            {
                "date": seq_rows_display.loc[i, "date"].strftime("%Y-%m-%d"),
                "intraday_return": round(float(feature_display.loc[i, "intraday_return"]), 6),
                "amplitude": round(float(feature_display.loc[i, "amplitude"]), 6),
                "volume_change_vs_prev": round(float(feature_display.loc[i, "volume_change_vs_prev"]), 6),
                "close_change_vs_prev_close": round(float(feature_display.loc[i, "close_change_vs_prev_close"]), 6),
            }
        )

    actual_info = get_actual_outcome(event_date=event_date, pred_label=label, horizon_days=3)
    result = {
        "event_date": meta["event_date"],
        "event_title": meta["event_title"],
        "event_type": meta["event_type"],
        "event_source": meta["event_source"],
        "event_count": meta["event_count"],
        "model_name": "LSTM (Event-driven, PyTorch)",
        "prediction_label": str(label),
        "probability": float(round(prob, 4)),
        "confidence": float(round(confidence, 4)),
        "used_sequence_dates": used_sequence_dates,
        "feature_sequence": raw_feature_records,
        **actual_info,
    }
    return sanitize_for_json(result)