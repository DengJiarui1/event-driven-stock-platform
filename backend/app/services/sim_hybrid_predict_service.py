from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch
from torch import nn

from .sim_predict_service import (
    build_title_features,
    get_previous_context,
    sanitize_for_json,
)


ROOT_DIR = Path(__file__).resolve().parents[3]

MODEL_PATH = ROOT_DIR / "artifacts" / "models" / "sim_hybrid_lstm.pt"
SEQ_SCALER_PATH = ROOT_DIR / "artifacts" / "models" / "sim_hybrid_seq_scaler.pkl"
STATIC_SCALER_PATH = ROOT_DIR / "artifacts" / "models" / "sim_hybrid_static_scaler.pkl"
META_PATH = ROOT_DIR / "artifacts" / "models" / "sim_hybrid_meta.json"


class HybridEventLSTM(nn.Module):
    def __init__(
        self,
        seq_input_size: int = 4,
        static_input_size: int = 9,
        hidden_size: int = 32,
        static_hidden_size: int = 16,
        fusion_hidden_size: int = 32,
        dropout: float = 0.10,
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=seq_input_size,
            hidden_size=hidden_size,
            batch_first=True,
        )

        self.static_mlp = nn.Sequential(
            nn.Linear(static_input_size, static_hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(static_hidden_size, static_hidden_size),
            nn.ReLU(),
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size + static_hidden_size, fusion_hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden_size, 1),
        )

    def forward(self, seq_x: torch.Tensor, static_x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(seq_x)
        seq_feat = lstm_out[:, -1, :]
        static_feat = self.static_mlp(static_x)
        fused = torch.cat([seq_feat, static_feat], dim=1)
        logits = self.classifier(fused).squeeze(-1)
        return logits


@lru_cache(maxsize=1)
def load_meta() -> dict[str, Any]:
    if not META_PATH.exists():
        raise FileNotFoundError(
            f"Hybrid LSTM meta 文件不存在: {META_PATH}，请先训练 sim_hybrid_lstm"
        )
    return json.loads(META_PATH.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def load_seq_scaler():
    if not SEQ_SCALER_PATH.exists():
        raise FileNotFoundError(
            f"Hybrid LSTM 序列 scaler 不存在: {SEQ_SCALER_PATH}"
        )
    return joblib.load(SEQ_SCALER_PATH)


@lru_cache(maxsize=1)
def load_static_scaler():
    if not STATIC_SCALER_PATH.exists():
        raise FileNotFoundError(
            f"Hybrid LSTM 静态 scaler 不存在: {STATIC_SCALER_PATH}"
        )
    return joblib.load(STATIC_SCALER_PATH)


@lru_cache(maxsize=1)
def load_hybrid_model() -> HybridEventLSTM:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Hybrid LSTM 模型不存在: {MODEL_PATH}，请先训练 sim_hybrid_lstm"
        )

    meta = load_meta()
    model_cfg = meta.get("model_config", {})

    model = HybridEventLSTM(
        seq_input_size=int(model_cfg.get("seq_input_size", 4)),
        static_input_size=int(model_cfg.get("static_input_size", 9)),
        hidden_size=int(model_cfg.get("hidden_size", 32)),
        static_hidden_size=int(model_cfg.get("static_hidden_size", 16)),
        fusion_hidden_size=int(model_cfg.get("fusion_hidden_size", 32)),
        dropout=float(model_cfg.get("dropout", 0.10)),
    )

    state_dict = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model


def build_seq_array_from_context(context: dict[str, Any]) -> np.ndarray:
    market_feature_dict = context["market_feature_dict"]

    seq_array = np.array(
        [[
            [
                float(market_feature_dict["t_minus_2_intraday_return"]),
                float(market_feature_dict["t_minus_2_amplitude"]),
                float(market_feature_dict["t_minus_2_volume_change_vs_prev"]),
                float(market_feature_dict["t_minus_2_close_change_vs_prev_close"]),
            ],
            [
                float(market_feature_dict["t_minus_1_intraday_return"]),
                float(market_feature_dict["t_minus_1_amplitude"]),
                float(market_feature_dict["t_minus_1_volume_change_vs_prev"]),
                float(market_feature_dict["t_minus_1_close_change_vs_prev_close"]),
            ],
        ]],
        dtype=np.float32,
    )
    return seq_array


def build_static_array_from_features(text_features: dict[str, Any], meta: dict[str, Any]) -> np.ndarray:
    static_feature_columns = meta.get("static_feature_columns", [])
    if not static_feature_columns:
        raise ValueError("sim_hybrid_meta.json 中缺少 static_feature_columns")

    static_row = [float(text_features.get(col, 0.0)) for col in static_feature_columns]
    return np.array([static_row], dtype=np.float32)


def predict_new_event_simulation_hybrid(
    event_date: str,
    event_type: int,
    event_source: str | None,
    event_title: str,
) -> dict[str, Any]:
    context = get_previous_context(event_date)
    meta = load_meta()
    seq_scaler = load_seq_scaler()
    static_scaler = load_static_scaler()
    model = load_hybrid_model()

    text_features = build_title_features(event_title, event_type, event_source)

    seq_array = build_seq_array_from_context(context)
    static_array = build_static_array_from_features(text_features, meta)

    # 标准化
    original_seq_shape = seq_array.shape
    seq_flat = seq_array.reshape(-1, original_seq_shape[-1])
    seq_scaled = seq_scaler.transform(seq_flat).reshape(original_seq_shape).astype(np.float32)
    static_scaled = static_scaler.transform(static_array).astype(np.float32)

    # 推理
    seq_tensor = torch.tensor(seq_scaled, dtype=torch.float32)
    static_tensor = torch.tensor(static_scaled, dtype=torch.float32)

    with torch.no_grad():
        logits = model(seq_tensor, static_tensor)
        prob = float(torch.sigmoid(logits).item())

    label = "上涨" if prob >= 0.5 else "下跌"
    confidence = max(prob, 1 - prob)

    result = {
        "simulation_mode": "pre_event_hybrid_lstm",
        "model_version": "hybrid_lstm",
        "event_date": event_date,
        "event_type": int(event_type),
        "event_source": event_source,
        "event_title": event_title,
        "model_name": meta.get("model_name", "Hybrid LSTM (Pre-event Simulation)"),
        "prediction_label": label,
        "probability": round(prob, 4),
        "confidence": round(confidence, 4),
        "context_dates": context["context_dates"],
        "market_sequence": context["market_sequence"],
        "text_features": text_features,
        "note": "该结果为事件发生前模拟预测，不包含事件日及未来信息；当前模型为 Hybrid LSTM（时序特征 + 静态特征融合）。",
    }
    return sanitize_for_json(result)