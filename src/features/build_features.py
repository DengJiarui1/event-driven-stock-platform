import numpy as np
import pandas as pd

from src.config import DATA_INTERIM, DATA_PROCESSED


FEATURE_COLUMNS = [
    "event_type",
    "pre_day_return",
    "event_day_gap",
    "event_day_return",
    "event_day_amplitude",
    "volume_change",
    "close_change_from_prev",
]


def _safe_div(numerator: float, denominator: float) -> float:
    if denominator is None or denominator == 0 or pd.isna(denominator):
        return 0.0
    return float(numerator) / float(denominator)


def main() -> None:
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_INTERIM / "event_windows.csv", parse_dates=["date", "event_date"])

    features = []
    for event_date, group in df.groupby("event_date"):
        group = group.sort_values("date").reset_index(drop=True)

        # 当前事件窗口约定为: t-1, t, t+1, t+2, t+3
        if len(group) < 5:
            continue

        prev_row = group.iloc[0]
        event_row = group.iloc[1]
        future_end_row = group.iloc[4]

        pre_day_return = _safe_div(prev_row["close"] - prev_row["open"], prev_row["open"])
        event_day_gap = _safe_div(event_row["open"] - prev_row["close"], prev_row["close"])
        event_day_return = _safe_div(event_row["close"] - event_row["open"], event_row["open"])
        event_day_amplitude = _safe_div(event_row["high"] - event_row["low"], event_row["open"])
        volume_change = _safe_div(event_row["volume"] - prev_row["volume"], prev_row["volume"])
        close_change_from_prev = _safe_div(event_row["close"] - prev_row["close"], prev_row["close"])

        target_post_3d_return = _safe_div(
            future_end_row["close"] - event_row["close"], event_row["close"]
        )

        feat = {
            "event_date": event_date,
            "event_type": int(event_row.get("event_type", 1)),
            "pre_day_return": pre_day_return,
            "event_day_gap": event_day_gap,
            "event_day_return": event_day_return,
            "event_day_amplitude": event_day_amplitude,
            "volume_change": volume_change,
            "close_change_from_prev": close_change_from_prev,
            "target_post_3d_return": target_post_3d_return,
            "label": int(target_post_3d_return > 0),
        }
        features.append(feat)

    if not features:
        raise ValueError("没有构造出有效特征，请检查 event_windows.csv 是否为空。")

    feature_df = pd.DataFrame(features).sort_values("event_date").reset_index(drop=True)

    # 完整特征表: 保留未来收益列, 便于论文分析和结果核对
    feature_df.to_csv(DATA_PROCESSED / "features.csv", index=False, encoding="utf-8-sig")

    # 训练数据集: 只保留事件当下可见特征 + 标签, 避免信息泄露
    dataset_df = feature_df[["event_date", *FEATURE_COLUMNS, "label"]].copy()
    dataset_df.to_csv(DATA_PROCESSED / "dataset.csv", index=False, encoding="utf-8-sig")

    print("无信息泄露特征构建完成")
    print(feature_df.head())
    print("\nBaseline 使用特征列:")
    print(FEATURE_COLUMNS)
    print("\n标签分布:")
    print(feature_df["label"].value_counts())


if __name__ == "__main__":
    main()
