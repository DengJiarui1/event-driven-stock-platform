import numpy as np
import pandas as pd

from src.config import DATA_INTERIM, DATA_PROCESSED


SEQ_FEATURES = [
    "intraday_return",
    "amplitude",
    "volume_change_vs_prev",
    "close_change_vs_prev_close",
]


def _safe_div(numerator: float, denominator: float) -> float:
    if denominator is None or denominator == 0 or pd.isna(denominator):
        return 0.0
    return float(numerator) / float(denominator)


def main() -> None:
    df = pd.read_csv(DATA_INTERIM / "event_windows.csv", parse_dates=["date", "event_date"])

    X, y = [], []
    for event_date, group in df.groupby("event_date"):
        group = group.sort_values("date").reset_index(drop=True)

        # 当前窗口为: t-1, t, t+1, t+2, t+3
        if len(group) < 5:
            continue

        prev_row = group.iloc[0]
        event_row = group.iloc[1]
        future_end_row = group.iloc[4]

        prev_step = [
            _safe_div(prev_row["close"] - prev_row["open"], prev_row["open"]),
            _safe_div(prev_row["high"] - prev_row["low"], prev_row["open"]),
            0.0,
            0.0,
        ]
        event_step = [
            _safe_div(event_row["close"] - event_row["open"], event_row["open"]),
            _safe_div(event_row["high"] - event_row["low"], event_row["open"]),
            _safe_div(event_row["volume"] - prev_row["volume"], prev_row["volume"]),
            _safe_div(event_row["close"] - prev_row["close"], prev_row["close"]),
        ]

        seq = np.array([prev_step, event_step], dtype=float)
        label = int(_safe_div(future_end_row["close"] - event_row["close"], event_row["close"]) > 0)

        X.append(seq)
        y.append(label)

    if not X:
        raise ValueError("没有构造出有效的 LSTM 输入，请检查 event_windows.csv 是否为空。")

    X = np.array(X, dtype=float)
    y = np.array(y, dtype=int)

    np.save(DATA_PROCESSED / "X_lstm.npy", X)
    np.save(DATA_PROCESSED / "y_lstm.npy", y)

    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print("LSTM 数据准备完成（无信息泄露版）")
    print(f"LSTM 序列特征: {SEQ_FEATURES}")


if __name__ == "__main__":
    main()
