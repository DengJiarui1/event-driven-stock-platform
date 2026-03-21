from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset


ROOT_DIR = Path(__file__).resolve().parents[2]

DATASET_CSV = ROOT_DIR / "artifacts" / "datasets" / "simulation_train_data.csv"
MODEL_DIR = ROOT_DIR / "artifacts" / "models"
REPORT_DIR = ROOT_DIR / "artifacts" / "reports"

MODEL_PATH = MODEL_DIR / "sim_hybrid_lstm.pt"
SEQ_SCALER_PATH = MODEL_DIR / "sim_hybrid_seq_scaler.pkl"
STATIC_SCALER_PATH = MODEL_DIR / "sim_hybrid_static_scaler.pkl"
META_PATH = MODEL_DIR / "sim_hybrid_meta.json"

REPORT_PATH = REPORT_DIR / "sim_hybrid_report.json"
PRED_DETAIL_PATH = REPORT_DIR / "sim_hybrid_test_predictions.csv"

RANDOM_SEED = 42

# 时序特征：T-2 / T-1，各 4 个
SEQ_T_MINUS_2_COLUMNS = [
    "t_minus_2_intraday_return",
    "t_minus_2_amplitude",
    "t_minus_2_volume_change_vs_prev",
    "t_minus_2_close_change_vs_prev_close",
]
SEQ_T_MINUS_1_COLUMNS = [
    "t_minus_1_intraday_return",
    "t_minus_1_amplitude",
    "t_minus_1_volume_change_vs_prev",
    "t_minus_1_close_change_vs_prev_close",
]

SEQ_FEATURE_COLUMNS = [
    SEQ_T_MINUS_2_COLUMNS,
    SEQ_T_MINUS_1_COLUMNS,
]

# 静态事件特征
STATIC_FEATURE_COLUMNS = [
    "event_type",
    "title_length",
    "positive_keyword_count",
    "negative_keyword_count",
    "sentiment_score",
    "is_report",
    "is_dividend",
    "is_governance",
    "is_manual_source",
]

LABEL_COLUMN = "label"

# 时间顺序切分：70 / 10 / 20
TRAIN_RATIO = 0.70
VAL_RATIO = 0.10
TEST_RATIO = 0.20

# 训练参数
BATCH_SIZE = 16
EPOCHS = 80
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE = 10

SEQ_INPUT_SIZE = 4
STATIC_INPUT_SIZE = len(STATIC_FEATURE_COLUMNS)
HIDDEN_SIZE = 32
STATIC_HIDDEN_SIZE = 16
FUSION_HIDDEN_SIZE = 32
DROPOUT = 0.10


def set_seed(seed: int = RANDOM_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dirs() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)


def to_python_value(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def sanitize_for_json(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize_for_json(v) for v in obj]
    return to_python_value(obj)


@dataclass
class SplitBundle:
    df: pd.DataFrame
    seq_x: np.ndarray
    static_x: np.ndarray
    y: np.ndarray


class HybridSimulationDataset(Dataset):
    def __init__(self, seq_x: np.ndarray, static_x: np.ndarray, y: np.ndarray):
        self.seq_x = torch.tensor(seq_x, dtype=torch.float32)
        self.static_x = torch.tensor(static_x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return self.seq_x[idx], self.static_x[idx], self.y[idx]


class HybridEventLSTM(nn.Module):
    def __init__(
        self,
        seq_input_size: int = SEQ_INPUT_SIZE,
        static_input_size: int = STATIC_INPUT_SIZE,
        hidden_size: int = HIDDEN_SIZE,
        static_hidden_size: int = STATIC_HIDDEN_SIZE,
        fusion_hidden_size: int = FUSION_HIDDEN_SIZE,
        dropout: float = DROPOUT,
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


class EarlyStopping:
    def __init__(self, patience: int = PATIENCE):
        self.patience = patience
        self.best_loss = np.inf
        self.best_state_dict: dict[str, torch.Tensor] | None = None
        self.counter = 0
        self.should_stop = False

    def step(self, val_loss: float, model: nn.Module):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.best_state_dict = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True


def load_dataset() -> pd.DataFrame:
    if not DATASET_CSV.exists():
        raise FileNotFoundError(
            f"未找到模拟训练数据: {DATASET_CSV}，请先运行 build_simulation_features.py"
        )

    df = pd.read_csv(DATASET_CSV)
    if df.empty:
        raise ValueError("simulation_train_data.csv 为空")

    required_columns = {
        LABEL_COLUMN,
        *STATIC_FEATURE_COLUMNS,
        *SEQ_T_MINUS_2_COLUMNS,
        *SEQ_T_MINUS_1_COLUMNS,
    }
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise ValueError(f"simulation_train_data.csv 缺少字段: {missing}")

    if "event_date" not in df.columns:
        raise ValueError("simulation_train_data.csv 缺少 event_date 字段")

    df["event_date"] = pd.to_datetime(df["event_date"], errors="coerce")
    df = df.dropna(subset=["event_date"]).sort_values("event_date").reset_index(drop=True)

    numeric_columns = list(required_columns - {LABEL_COLUMN})
    numeric_columns.append(LABEL_COLUMN)
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=numeric_columns).reset_index(drop=True)
    df[LABEL_COLUMN] = df[LABEL_COLUMN].astype(int)

    return df


def build_seq_array(df: pd.DataFrame) -> np.ndarray:
    seq_rows = []
    for _, row in df.iterrows():
        seq_rows.append([
            [float(row[col]) for col in SEQ_T_MINUS_2_COLUMNS],
            [float(row[col]) for col in SEQ_T_MINUS_1_COLUMNS],
        ])
    return np.array(seq_rows, dtype=np.float32)


def build_static_array(df: pd.DataFrame) -> np.ndarray:
    return df[STATIC_FEATURE_COLUMNS].astype(np.float32).values


def build_label_array(df: pd.DataFrame) -> np.ndarray:
    return df[LABEL_COLUMN].astype(np.float32).values


def split_time_order(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n = len(df)
    if n < 20:
        raise ValueError(f"样本数过少（当前 {n}），不建议训练 Hybrid LSTM")

    train_end = int(n * TRAIN_RATIO)
    val_end = int(n * (TRAIN_RATIO + VAL_RATIO))

    train_df = df.iloc[:train_end].copy().reset_index(drop=True)
    val_df = df.iloc[train_end:val_end].copy().reset_index(drop=True)
    test_df = df.iloc[val_end:].copy().reset_index(drop=True)

    if len(train_df) == 0 or len(val_df) == 0 or len(test_df) == 0:
        raise ValueError("时间切分后 train/val/test 中存在空集，请检查数据量")

    return train_df, val_df, test_df


def fit_scalers(train_seq: np.ndarray, train_static: np.ndarray) -> tuple[StandardScaler, StandardScaler]:
    seq_scaler = StandardScaler()
    static_scaler = StandardScaler()

    seq_scaler.fit(train_seq.reshape(-1, train_seq.shape[-1]))
    static_scaler.fit(train_static)

    return seq_scaler, static_scaler


def transform_seq(seq_scaler: StandardScaler, seq_x: np.ndarray) -> np.ndarray:
    original_shape = seq_x.shape
    flat = seq_x.reshape(-1, original_shape[-1])
    flat_scaled = seq_scaler.transform(flat)
    return flat_scaled.reshape(original_shape).astype(np.float32)


def transform_static(static_scaler: StandardScaler, static_x: np.ndarray) -> np.ndarray:
    return static_scaler.transform(static_x).astype(np.float32)


def build_bundle(df: pd.DataFrame) -> SplitBundle:
    return SplitBundle(
        df=df,
        seq_x=build_seq_array(df),
        static_x=build_static_array(df),
        y=build_label_array(df),
    )


def create_dataloaders(
    train_bundle: SplitBundle,
    val_bundle: SplitBundle,
    test_bundle: SplitBundle,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_ds = HybridSimulationDataset(train_bundle.seq_x, train_bundle.static_x, train_bundle.y)
    val_ds = HybridSimulationDataset(val_bundle.seq_x, val_bundle.static_x, val_bundle.y)
    test_ds = HybridSimulationDataset(test_bundle.seq_x, test_bundle.static_x, test_bundle.y)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, val_loader, test_loader


def evaluate_loader(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_count = 0
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for seq_x, static_x, y in loader:
            seq_x = seq_x.to(device)
            static_x = static_x.to(device)
            y = y.to(device)

            logits = model(seq_x, static_x)
            loss = criterion(logits, y)

            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).float()

            batch_size = y.size(0)
            total_loss += loss.item() * batch_size
            total_count += batch_size

            all_probs.extend(probs.detach().cpu().numpy().tolist())
            all_labels.extend(y.detach().cpu().numpy().tolist())

    avg_loss = total_loss / total_count if total_count else 0.0
    if all_probs:
        preds_np = (np.array(all_probs) >= 0.5).astype(int)
        labels_np = np.array(all_labels).astype(int)
        acc = accuracy_score(labels_np, preds_np)
    else:
        acc = 0.0

    return avg_loss, acc


def predict_loader(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    all_probs = []
    all_preds = []
    all_true = []

    with torch.no_grad():
        for seq_x, static_x, y in loader:
            seq_x = seq_x.to(device)
            static_x = static_x.to(device)

            logits = model(seq_x, static_x)
            probs = torch.sigmoid(logits).detach().cpu().numpy()
            preds = (probs >= 0.5).astype(int)

            all_probs.extend(probs.tolist())
            all_preds.extend(preds.tolist())
            all_true.extend(y.numpy().astype(int).tolist())

    return (
        np.array(all_true, dtype=int),
        np.array(all_preds, dtype=int),
        np.array(all_probs, dtype=float),
    )


def main():
    print("A: device ok")
    df = load_dataset()
    print("B: load_dataset ok", len(df))
    train_df, val_df, test_df = split_time_order(df)
    print("C: split ok", len(train_df), len(val_df), len(test_df))
    set_seed()
    ensure_dirs()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    if hasattr(torch.backends, "mkldnn"):
        torch.backends.mkldnn.enabled = False

    print("D: backend config ok")

    df = load_dataset()
    train_df, val_df, test_df = split_time_order(df)
    print("E: reload dataset ok")

    train_bundle = build_bundle(train_df)
    print("F1: train bundle ok", train_bundle.seq_x.shape, train_bundle.static_x.shape, train_bundle.y.shape)

    val_bundle = build_bundle(val_df)
    print("F2: val bundle ok", val_bundle.seq_x.shape, val_bundle.static_x.shape, val_bundle.y.shape)

    test_bundle = build_bundle(test_df)
    print("F3: test bundle ok", test_bundle.seq_x.shape, test_bundle.static_x.shape, test_bundle.y.shape)

    seq_scaler, static_scaler = fit_scalers(train_bundle.seq_x, train_bundle.static_x)
    print("G: scalers ok")

    train_bundle.seq_x = transform_seq(seq_scaler, train_bundle.seq_x)
    val_bundle.seq_x = transform_seq(seq_scaler, val_bundle.seq_x)
    test_bundle.seq_x = transform_seq(seq_scaler, test_bundle.seq_x)
    print("H1: seq transform ok")

    train_bundle.static_x = transform_static(static_scaler, train_bundle.static_x)
    val_bundle.static_x = transform_static(static_scaler, val_bundle.static_x)
    test_bundle.static_x = transform_static(static_scaler, test_bundle.static_x)
    print("H2: static transform ok")

    train_loader, val_loader, test_loader = create_dataloaders(
        train_bundle, val_bundle, test_bundle
    )
    print("I: dataloaders ok")

    model = HybridEventLSTM().to(device)
    print("J: model ok")

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.01,
        momentum=0.9,
        foreach=False,
        fused=False,
    )
    print("K: optimizer ok")

    criterion = nn.BCEWithLogitsLoss()
    early_stopping = EarlyStopping(patience=PATIENCE)
    print("L: training objects ok")

    print("Train / Val / Test rows:")
    print({
        "train": len(train_df),
        "val": len(val_df),
        "test": len(test_df),
    })

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_train_loss = 0.0
        total_train_count = 0
        all_train_probs = []
        all_train_labels = []

        for seq_x, static_x, y in train_loader:
            seq_x = seq_x.to(device)
            static_x = static_x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            logits = model(seq_x, static_x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            probs = torch.sigmoid(logits)

            batch_size = y.size(0)
            total_train_loss += loss.item() * batch_size
            total_train_count += batch_size
            all_train_probs.extend(probs.detach().cpu().numpy().tolist())
            all_train_labels.extend(y.detach().cpu().numpy().tolist())

        train_loss = total_train_loss / total_train_count if total_train_count else 0.0
        train_preds = (np.array(all_train_probs) >= 0.5).astype(int)
        train_labels = np.array(all_train_labels).astype(int)
        train_acc = accuracy_score(train_labels, train_preds) if len(train_labels) else 0.0

        val_loss, val_acc = evaluate_loader(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch:02d}/{EPOCHS} - "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        early_stopping.step(val_loss, model)
        if early_stopping.should_stop:
            print("Early stopping triggered.")
            break

    if early_stopping.best_state_dict is not None:
        model.load_state_dict(early_stopping.best_state_dict)

    y_true, y_pred, y_prob = predict_loader(model, test_loader, device)

    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()
    report_text = classification_report(y_true, y_pred, zero_division=0)

    print("\nHybrid LSTM (Pre-event) Test Result")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-score : {f1:.4f}")
    print("Confusion Matrix:")
    print(cm)
    print(report_text)

    # 保存模型
    torch.save(model.state_dict(), MODEL_PATH)
    joblib.dump(seq_scaler, SEQ_SCALER_PATH)
    joblib.dump(static_scaler, STATIC_SCALER_PATH)

    meta = {
        "model_name": "Hybrid LSTM (Pre-event Simulation)",
        "task_type": "pre_event_simulation_classification",
        "seq_feature_columns": SEQ_FEATURE_COLUMNS,
        "static_feature_columns": STATIC_FEATURE_COLUMNS,
        "train_rows": int(len(train_df)),
        "val_rows": int(len(val_df)),
        "test_rows": int(len(test_df)),
        "split_method": "time_order_70_10_20",
        "model_config": {
            "seq_input_size": SEQ_INPUT_SIZE,
            "static_input_size": STATIC_INPUT_SIZE,
            "hidden_size": HIDDEN_SIZE,
            "static_hidden_size": STATIC_HIDDEN_SIZE,
            "fusion_hidden_size": FUSION_HIDDEN_SIZE,
            "dropout": DROPOUT,
        },
        "train_config": {
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "patience": PATIENCE,
            "device": str(device),
        },
        "metrics": {
            "accuracy": float(acc),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        },
    }
    META_PATH.write_text(
        json.dumps(sanitize_for_json(meta), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    report_json = {
        "model_name": "Hybrid LSTM (Pre-event Simulation)",
        "task_type": "pre_event_simulation_classification",
        "train_rows": int(len(train_df)),
        "val_rows": int(len(val_df)),
        "test_rows": int(len(test_df)),
        "metrics": {
            "accuracy": float(acc),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        },
        "confusion_matrix": cm,
        "class_labels": {
            "negative_class": 0,
            "positive_class": 1,
            "negative_label_text": "下跌",
            "positive_label_text": "上涨",
            "matrix_layout": "[[TN, FP], [FN, TP]]",
        },
        "classification_report_text": report_text,
        "seq_feature_columns": SEQ_FEATURE_COLUMNS,
        "static_feature_columns": STATIC_FEATURE_COLUMNS,
    }
    REPORT_PATH.write_text(
        json.dumps(sanitize_for_json(report_json), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # 保存测试集预测明细
    pred_detail = test_df.copy().reset_index(drop=True)
    pred_detail["y_true"] = y_true
    pred_detail["y_pred"] = y_pred
    pred_detail["pred_prob_up"] = y_prob
    pred_detail.to_csv(PRED_DETAIL_PATH, index=False, encoding="utf-8-sig")

    print("\n已保存文件：")
    print(MODEL_PATH)
    print(SEQ_SCALER_PATH)
    print(STATIC_SCALER_PATH)
    print(META_PATH)
    print(REPORT_PATH)
    print(PRED_DETAIL_PATH)


if __name__ == "__main__":
    main()