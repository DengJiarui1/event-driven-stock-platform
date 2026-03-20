from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.config import DATA_PROCESSED, ARTIFACTS_MODELS, ARTIFACTS_REPORTS


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class EventLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 32):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.classifier = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        last_hidden = out[:, -1, :]
        logits = self.classifier(last_hidden)
        return logits.squeeze(-1)


def _split_by_time(n_rows: int, test_ratio: float = 0.2, val_ratio_within_train: float = 0.2) -> tuple[slice, slice, slice]:
    if n_rows < 10:
        raise ValueError(f"LSTM 样本量过少: {n_rows}，无法进行稳定的时间切分。")

    train_end = max(1, int(n_rows * (1 - test_ratio)))
    train_end = min(train_end, n_rows - 1)

    train_pool_size = train_end
    val_size = max(1, int(train_pool_size * val_ratio_within_train))
    val_start = max(1, train_end - val_size)

    if val_start <= 0 or val_start >= train_end:
        raise ValueError(
            f"无法生成有效的训练/验证切分: n_rows={n_rows}, train_end={train_end}, val_start={val_start}"
        )

    return slice(0, val_start), slice(val_start, train_end), slice(train_end, n_rows)


def _build_loader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def _evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total = 0
    correct = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).float()
            total_loss += loss.item() * len(xb)
            correct += (preds == yb).sum().item()
            total += len(xb)
    return total_loss / max(total, 1), correct / max(total, 1)


def main() -> None:
    set_seed(42)
    ARTIFACTS_MODELS.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_REPORTS.mkdir(parents=True, exist_ok=True)

    X = np.load(DATA_PROCESSED / "X_lstm_scaled.npy")
    y = np.load(DATA_PROCESSED / "y_lstm.npy")

    meta = pd.read_csv(DATA_PROCESSED / "dataset.csv", parse_dates=["event_date"])
    meta = meta.sort_values("event_date").reset_index(drop=True)

    if len(meta) != len(X) or len(meta) != len(y):
        raise ValueError(
            "LSTM 输入与 dataset.csv 行数不一致。请先重新运行: "
            "build_features.py、prepare_lstm_data.py、normalize_lstm_data.py"
        )

    train_slice, val_slice, test_slice = _split_by_time(len(meta), test_ratio=0.2, val_ratio_within_train=0.2)
    X_train, y_train = X[train_slice], y[train_slice]
    X_val, y_val = X[val_slice], y[val_slice]
    X_test, y_test = X[test_slice], y[test_slice]

    train_loader = _build_loader(X_train, y_train, batch_size=8, shuffle=True)
    val_loader = _build_loader(X_val, y_val, batch_size=8, shuffle=False)
    test_loader = _build_loader(X_test, y_test, batch_size=8, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EventLSTM(input_size=X.shape[2], hidden_size=32).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    history: list[dict[str, float]] = []
    best_val_loss = float("inf")
    patience = 5
    wait = 0
    best_path = ARTIFACTS_MODELS / "event_lstm_torch_best.pt"

    for epoch in range(1, 31):
        model.train()
        running_loss = 0.0
        running_correct = 0
        total = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).float()
            running_loss += loss.item() * len(xb)
            running_correct += (preds == yb).sum().item()
            total += len(xb)

        train_loss = running_loss / max(total, 1)
        train_acc = running_correct / max(total, 1)
        val_loss, val_acc = _evaluate(model, val_loader, criterion, device)

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
            }
        )

        print(
            f"Epoch {epoch:02d}/30 - train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            wait = 0
            torch.save(model.state_dict(), best_path)
        else:
            wait += 1
            if wait >= patience:
                print("Early stopping triggered.")
                break

    model.load_state_dict(torch.load(best_path, map_location=device))
    test_loss, test_acc = _evaluate(model, test_loader, criterion, device)

    final_path = ARTIFACTS_MODELS / "event_lstm_torch.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_size": X.shape[2],
            "hidden_size": 32,
            "seq_len": X.shape[1],
        },
        final_path,
    )

    pd.DataFrame(history).to_csv(ARTIFACTS_REPORTS / "lstm_history.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"Model": ["LSTM (Event-driven, PyTorch)"], "Accuracy": [test_acc]}).to_csv(
        ARTIFACTS_REPORTS / "lstm_result.csv", index=False, encoding="utf-8-sig"
    )

    split_info = pd.DataFrame(
        {
            "split": ["train", "val", "test"],
            "rows": [len(X_train), len(X_val), len(X_test)],
            "start_date": [
                meta.iloc[train_slice]["event_date"].min(),
                meta.iloc[val_slice]["event_date"].min(),
                meta.iloc[test_slice]["event_date"].min(),
            ],
            "end_date": [
                meta.iloc[train_slice]["event_date"].max(),
                meta.iloc[val_slice]["event_date"].max(),
                meta.iloc[test_slice]["event_date"].max(),
            ],
        }
    )
    split_info.to_csv(ARTIFACTS_REPORTS / "lstm_time_split.csv", index=False, encoding="utf-8-sig")

    print("LSTM 模型结果（PyTorch + 时间顺序切分版）")
    print(split_info)
    print(f"LSTM 测试准确率: {test_acc:.4f}")
    print(f"模型已保存: {final_path}")


if __name__ == "__main__":
    main()
