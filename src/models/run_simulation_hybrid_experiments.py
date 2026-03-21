from __future__ import annotations

import os

# Windows / PyTorch 稳定性设置：必须放在 import torch 前
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
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
REPORT_DIR = ROOT_DIR / "artifacts" / "reports"
TMP_DIR = REPORT_DIR / "sim_hybrid_experiment_tmp"

RUNS_CSV = REPORT_DIR / "sim_hybrid_experiment_runs.csv"
SUMMARY_CSV = REPORT_DIR / "sim_hybrid_experiment_summary.csv"
BEST_CONFIG_JSON = REPORT_DIR / "sim_hybrid_best_config_from_experiments.json"

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

TRAIN_RATIO = 0.70
VAL_RATIO = 0.10

# 你可以在这里改随机种子数量
SEEDS = [42, 7, 2024]

# 你可以在这里继续加实验组
EXPERIMENT_CONFIGS: list[dict[str, Any]] = [
    {
        "config_name": "baseline_like",
        "hidden_size": 16,
        "static_hidden_size": 8,
        "fusion_hidden_size": 16,
        "dropout": 0.20,
        "learning_rate": 5e-4,
        "weight_decay": 1e-3,
        "batch_size": 8,
        "patience": 15,
        "epochs": 80,
        "manual_pos_weight": 1.0,
        "threshold_candidates": list(np.round(np.arange(0.50, 0.59, 0.01), 2)),
        "threshold_select_metric": "balanced_accuracy",
    },
    {
        "config_name": "small_stable_v1",
        "hidden_size": 8,
        "static_hidden_size": 8,
        "fusion_hidden_size": 8,
        "dropout": 0.25,
        "learning_rate": 3e-4,
        "weight_decay": 1e-3,
        "batch_size": 8,
        "patience": 20,
        "epochs": 100,
        "manual_pos_weight": 1.0,
        "threshold_candidates": list(np.round(np.arange(0.50, 0.59, 0.01), 2)),
        "threshold_select_metric": "balanced_accuracy",
    },
    {
        "config_name": "small_stable_v2",
        "hidden_size": 8,
        "static_hidden_size": 8,
        "fusion_hidden_size": 8,
        "dropout": 0.25,
        "learning_rate": 3e-4,
        "weight_decay": 1e-3,
        "batch_size": 8,
        "patience": 20,
        "epochs": 100,
        "manual_pos_weight": 0.9,
        "threshold_candidates": list(np.round(np.arange(0.50, 0.59, 0.01), 2)),
        "threshold_select_metric": "balanced_accuracy",
    },
    {
        "config_name": "middle_balance_v1",
        "hidden_size": 12,
        "static_hidden_size": 8,
        "fusion_hidden_size": 16,
        "dropout": 0.20,
        "learning_rate": 5e-4,
        "weight_decay": 5e-4,
        "batch_size": 8,
        "patience": 20,
        "epochs": 90,
        "manual_pos_weight": 1.0,
        "threshold_candidates": list(np.round(np.arange(0.50, 0.59, 0.01), 2)),
        "threshold_select_metric": "balanced_accuracy",
    },
    {
        "config_name": "middle_balance_v2",
        "hidden_size": 12,
        "static_hidden_size": 8,
        "fusion_hidden_size": 16,
        "dropout": 0.20,
        "learning_rate": 5e-4,
        "weight_decay": 5e-4,
        "batch_size": 8,
        "patience": 20,
        "epochs": 90,
        "manual_pos_weight": 1.1,
        "threshold_candidates": list(np.round(np.arange(0.50, 0.59, 0.01), 2)),
        "threshold_select_metric": "balanced_accuracy",
    },
    {
        "config_name": "wider_threshold_scan",
        "hidden_size": 16,
        "static_hidden_size": 8,
        "fusion_hidden_size": 16,
        "dropout": 0.20,
        "learning_rate": 5e-4,
        "weight_decay": 1e-3,
        "batch_size": 8,
        "patience": 15,
        "epochs": 80,
        "manual_pos_weight": 1.0,
        "threshold_candidates": list(np.round(np.arange(0.45, 0.66, 0.02), 2)),
        "threshold_select_metric": "macro_f1",
    },
]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dirs() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    TMP_DIR.mkdir(parents=True, exist_ok=True)


def configure_torch_backend() -> None:
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    if hasattr(torch.backends, "mkldnn"):
        torch.backends.mkldnn.enabled = False


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
        seq_input_size: int,
        static_input_size: int,
        hidden_size: int,
        static_hidden_size: int,
        fusion_hidden_size: int,
        dropout: float,
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
    def __init__(self, patience: int):
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
            f"未找到训练数据: {DATASET_CSV}，请先运行 build_simulation_features.py"
        )

    df = pd.read_csv(DATASET_CSV)
    if df.empty:
        raise ValueError("simulation_train_data.csv 为空")

    required_columns = {
        LABEL_COLUMN,
        *STATIC_FEATURE_COLUMNS,
        *SEQ_T_MINUS_2_COLUMNS,
        *SEQ_T_MINUS_1_COLUMNS,
        "event_date",
    }
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise ValueError(f"缺少字段: {missing}")

    df["event_date"] = pd.to_datetime(df["event_date"], errors="coerce")
    df = df.dropna(subset=["event_date"]).sort_values("event_date").reset_index(drop=True)

    numeric_columns = [c for c in required_columns if c != "event_date"]
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=numeric_columns).reset_index(drop=True)
    df[LABEL_COLUMN] = df[LABEL_COLUMN].astype(int)

    return df


def split_time_order(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n = len(df)
    if n < 20:
        raise ValueError(f"样本数过少（当前 {n}），不建议批量试验")

    train_end = int(n * TRAIN_RATIO)
    val_end = int(n * (TRAIN_RATIO + VAL_RATIO))

    train_df = df.iloc[:train_end].copy().reset_index(drop=True)
    val_df = df.iloc[train_end:val_end].copy().reset_index(drop=True)
    test_df = df.iloc[val_end:].copy().reset_index(drop=True)

    return train_df, val_df, test_df


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


def build_bundle(df: pd.DataFrame) -> SplitBundle:
    return SplitBundle(
        df=df,
        seq_x=build_seq_array(df),
        static_x=build_static_array(df),
        y=build_label_array(df),
    )


def fit_scalers(train_seq: np.ndarray, train_static: np.ndarray) -> tuple[StandardScaler, StandardScaler]:
    seq_scaler = StandardScaler()
    static_scaler = StandardScaler()

    seq_scaler.fit(train_seq.reshape(-1, train_seq.shape[-1]))
    static_scaler.fit(train_static)

    return seq_scaler, static_scaler


def transform_seq(seq_scaler: StandardScaler, seq_x: np.ndarray) -> np.ndarray:
    shape = seq_x.shape
    flat = seq_x.reshape(-1, shape[-1])
    flat_scaled = seq_scaler.transform(flat)
    return flat_scaled.reshape(shape).astype(np.float32)


def transform_static(static_scaler: StandardScaler, static_x: np.ndarray) -> np.ndarray:
    return static_scaler.transform(static_x).astype(np.float32)


def create_dataloaders(
    train_bundle: SplitBundle,
    val_bundle: SplitBundle,
    test_bundle: SplitBundle,
    batch_size: int,
    seed: int,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_ds = HybridSimulationDataset(train_bundle.seq_x, train_bundle.static_x, train_bundle.y)
    val_ds = HybridSimulationDataset(val_bundle.seq_x, val_bundle.static_x, val_bundle.y)
    test_ds = HybridSimulationDataset(test_bundle.seq_x, test_bundle.static_x, test_bundle.y)

    generator = torch.Generator()
    generator.manual_seed(seed)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=generator)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def compute_pos_weight(train_y: np.ndarray, manual_pos_weight: float | None) -> float:
    if manual_pos_weight is not None:
        return float(manual_pos_weight)

    positive_count = int((train_y == 1).sum())
    negative_count = int((train_y == 0).sum())
    if positive_count == 0:
        return 1.0
    return float(negative_count / positive_count)


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

            batch_size = y.size(0)
            total_loss += loss.item() * batch_size
            total_count += batch_size

            all_probs.extend(probs.detach().cpu().numpy().tolist())
            all_labels.extend(y.detach().cpu().numpy().tolist())

    avg_loss = total_loss / total_count if total_count else 0.0
    preds_np = (np.array(all_probs) >= 0.5).astype(int)
    labels_np = np.array(all_labels).astype(int)
    acc = accuracy_score(labels_np, preds_np) if len(labels_np) else 0.0
    return avg_loss, acc


def predict_loader(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_probs = []
    all_true = []

    with torch.no_grad():
        for seq_x, static_x, y in loader:
            seq_x = seq_x.to(device)
            static_x = static_x.to(device)

            logits = model(seq_x, static_x)
            probs = torch.sigmoid(logits).detach().cpu().numpy()

            all_probs.extend(probs.tolist())
            all_true.extend(y.numpy().astype(int).tolist())

    return np.array(all_true, dtype=int), np.array(all_probs, dtype=float)


def calc_threshold_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict[str, Any]:
    y_pred = (y_prob >= threshold).astype(int)

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    negative_recall = recall_score(y_true, y_pred, pos_label=0, zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "negative_recall": float(negative_recall),
        "macro_f1": float(macro_f1),
        "balanced_accuracy": float(balanced_acc),
        "tn": int(cm[0, 0]),
        "fp": int(cm[0, 1]),
        "fn": int(cm[1, 0]),
        "tp": int(cm[1, 1]),
    }


def scan_thresholds(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold_candidates: list[float],
    select_metric: str,
) -> pd.DataFrame:
    rows = [calc_threshold_metrics(y_true, y_prob, th) for th in threshold_candidates]
    df = pd.DataFrame(rows)
    df = df.sort_values(
        by=[select_metric, "balanced_accuracy", "accuracy", "threshold"],
        ascending=[False, False, False, True]
    ).reset_index(drop=True)
    return df


def safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def run_single_experiment(
    base_df: pd.DataFrame,
    config: dict[str, Any],
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    set_seed(seed)

    train_df, val_df, test_df = split_time_order(base_df)

    train_bundle = build_bundle(train_df)
    val_bundle = build_bundle(val_df)
    test_bundle = build_bundle(test_df)

    seq_scaler, static_scaler = fit_scalers(train_bundle.seq_x, train_bundle.static_x)

    train_bundle.seq_x = transform_seq(seq_scaler, train_bundle.seq_x)
    val_bundle.seq_x = transform_seq(seq_scaler, val_bundle.seq_x)
    test_bundle.seq_x = transform_seq(seq_scaler, test_bundle.seq_x)

    train_bundle.static_x = transform_static(static_scaler, train_bundle.static_x)
    val_bundle.static_x = transform_static(static_scaler, val_bundle.static_x)
    test_bundle.static_x = transform_static(static_scaler, test_bundle.static_x)

    train_loader, val_loader, test_loader = create_dataloaders(
        train_bundle=train_bundle,
        val_bundle=val_bundle,
        test_bundle=test_bundle,
        batch_size=int(config["batch_size"]),
        seed=seed,
    )

    model = HybridEventLSTM(
        seq_input_size=4,
        static_input_size=len(STATIC_FEATURE_COLUMNS),
        hidden_size=int(config["hidden_size"]),
        static_hidden_size=int(config["static_hidden_size"]),
        fusion_hidden_size=int(config["fusion_hidden_size"]),
        dropout=float(config["dropout"]),
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
        foreach=False,
        fused=False,
    )

    pos_weight_value = compute_pos_weight(
        train_bundle.y,
        safe_float(config.get("manual_pos_weight")),
    )
    pos_weight_tensor = torch.tensor([pos_weight_value], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)

    early_stopping = EarlyStopping(patience=int(config["patience"]))
    epochs = int(config["epochs"])

    for epoch in range(1, epochs + 1):
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
            f"[{config['config_name']}] seed={seed} epoch={epoch:02d}/{epochs} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        early_stopping.step(val_loss, model)
        if early_stopping.should_stop:
            print(f"[{config['config_name']}] seed={seed} early stopping triggered.")
            break

    if early_stopping.best_state_dict is not None:
        model.load_state_dict(early_stopping.best_state_dict)

    val_y_true, val_y_prob = predict_loader(model, val_loader, device)
    threshold_scan_df = scan_thresholds(
        y_true=val_y_true,
        y_prob=val_y_prob,
        threshold_candidates=list(config["threshold_candidates"]),
        select_metric=str(config["threshold_select_metric"]),
    )

    threshold_scan_path = TMP_DIR / f"{config['config_name']}_seed{seed}_threshold_scan.csv"
    threshold_scan_df.to_csv(threshold_scan_path, index=False, encoding="utf-8-sig")

    best_threshold = float(threshold_scan_df.iloc[0]["threshold"])

    test_y_true, test_y_prob = predict_loader(model, test_loader, device)
    test_metrics = calc_threshold_metrics(test_y_true, test_y_prob, best_threshold)

    run_result = {
        "config_name": config["config_name"],
        "seed": seed,
        "hidden_size": int(config["hidden_size"]),
        "static_hidden_size": int(config["static_hidden_size"]),
        "fusion_hidden_size": int(config["fusion_hidden_size"]),
        "dropout": float(config["dropout"]),
        "learning_rate": float(config["learning_rate"]),
        "weight_decay": float(config["weight_decay"]),
        "batch_size": int(config["batch_size"]),
        "patience": int(config["patience"]),
        "epochs": int(config["epochs"]),
        "manual_pos_weight": float(pos_weight_value),
        "threshold_select_metric": str(config["threshold_select_metric"]),
        "threshold_candidates": json.dumps(list(config["threshold_candidates"]), ensure_ascii=False),
        "best_threshold": float(best_threshold),
        "accuracy": float(test_metrics["accuracy"]),
        "precision": float(test_metrics["precision"]),
        "recall": float(test_metrics["recall"]),
        "f1": float(test_metrics["f1"]),
        "macro_f1": float(test_metrics["macro_f1"]),
        "balanced_accuracy": float(test_metrics["balanced_accuracy"]),
        "negative_recall": float(test_metrics["negative_recall"]),
        "tn": int(test_metrics["tn"]),
        "fp": int(test_metrics["fp"]),
        "fn": int(test_metrics["fn"]),
        "tp": int(test_metrics["tp"]),
        "threshold_scan_csv": str(threshold_scan_path),
    }

    return run_result


def build_summary_df(runs_df: pd.DataFrame) -> pd.DataFrame:
    group_cols = [
        "config_name",
        "hidden_size",
        "static_hidden_size",
        "fusion_hidden_size",
        "dropout",
        "learning_rate",
        "weight_decay",
        "batch_size",
        "patience",
        "epochs",
        "manual_pos_weight",
        "threshold_select_metric",
        "threshold_candidates",
    ]

    metric_cols = [
        "best_threshold",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "macro_f1",
        "balanced_accuracy",
        "negative_recall",
        "tn",
        "fp",
        "fn",
        "tp",
    ]

    agg_dict: dict[str, list[str]] = {}
    for col in metric_cols:
        agg_dict[col] = ["mean", "std", "min", "max"]

    summary_df = runs_df.groupby(group_cols, dropna=False).agg(agg_dict).reset_index()
    summary_df.columns = [
        "_".join(c).strip("_") if isinstance(c, tuple) else c
        for c in summary_df.columns
    ]

    summary_df = summary_df.sort_values(
        by=["macro_f1_mean", "balanced_accuracy_mean", "accuracy_mean"],
        ascending=[False, False, False]
    ).reset_index(drop=True)

    return summary_df


def main():
    ensure_dirs()
    configure_torch_backend()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    base_df = load_dataset()
    print(f"Loaded dataset rows: {len(base_df)}")
    print(f"Experiment groups: {len(EXPERIMENT_CONFIGS)}")
    print(f"Seeds: {SEEDS}")

    all_run_results: list[dict[str, Any]] = []

    total_runs = len(EXPERIMENT_CONFIGS) * len(SEEDS)
    current_run = 0

    for config in EXPERIMENT_CONFIGS:
        print("\n" + "=" * 80)
        print(f"Running config: {config['config_name']}")
        print(json.dumps(config, ensure_ascii=False, indent=2))

        for seed in SEEDS:
            current_run += 1
            print("\n" + "-" * 80)
            print(f"Run progress: {current_run}/{total_runs} | config={config['config_name']} | seed={seed}")

            result = run_single_experiment(
                base_df=base_df,
                config=config,
                seed=seed,
                device=device,
            )
            all_run_results.append(result)

            print(
                f"[DONE] {config['config_name']} | seed={seed} | "
                f"acc={result['accuracy']:.4f} | macro_f1={result['macro_f1']:.4f} | "
                f"balanced_acc={result['balanced_accuracy']:.4f} | "
                f"neg_recall={result['negative_recall']:.4f} | "
                f"best_threshold={result['best_threshold']:.3f}"
            )

            pd.DataFrame(all_run_results).to_csv(RUNS_CSV, index=False, encoding="utf-8-sig")

    runs_df = pd.DataFrame(all_run_results)
    summary_df = build_summary_df(runs_df)

    runs_df.to_csv(RUNS_CSV, index=False, encoding="utf-8-sig")
    summary_df.to_csv(SUMMARY_CSV, index=False, encoding="utf-8-sig")

    best_row = summary_df.iloc[0].to_dict()
    BEST_CONFIG_JSON.write_text(
        json.dumps(best_row, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 80)
    print("All experiments finished.")
    print(f"Per-run results saved to: {RUNS_CSV}")
    print(f"Summary results saved to: {SUMMARY_CSV}")
    print(f"Best config summary saved to: {BEST_CONFIG_JSON}")

    print("\nTop 5 configs:")
    display_cols = [
        "config_name",
        "accuracy_mean",
        "macro_f1_mean",
        "balanced_accuracy_mean",
        "negative_recall_mean",
        "best_threshold_mean",
    ]
    print(summary_df[display_cols].head(5).to_string(index=False))


if __name__ == "__main__":
    main()