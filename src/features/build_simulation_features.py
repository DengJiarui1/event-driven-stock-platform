from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[2]

RAW_STOCK_CSV = ROOT_DIR / "data" / "raw" / "stock_price_600519.csv"
EVENTS_CSV = ROOT_DIR / "data" / "interim" / "events.csv"
OUTPUT_DIR = ROOT_DIR / "artifacts" / "datasets"
OUTPUT_CSV = OUTPUT_DIR / "simulation_train_data.csv"

POSITIVE_WORDS = [
    "增长", "预增", "分红", "回购", "中标", "合作", "利好", "创新高", "增持", "派息"
]

NEGATIVE_WORDS = [
    "亏损", "下滑", "减持", "风险", "处罚", "问询", "诉讼", "减值", "利空", "下跌"
]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


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


def load_events_df() -> pd.DataFrame:
    if not EVENTS_CSV.exists():
        raise FileNotFoundError(f"事件数据不存在: {EVENTS_CSV}")

    df = pd.read_csv(EVENTS_CSV)
    required_cols = ["event_date", "event_type", "event_source", "event_title"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"事件数据缺少字段: {missing}")

    df["event_date"] = pd.to_datetime(df["event_date"], errors="coerce")
    df = df.dropna(subset=["event_date"]).sort_values("event_date").reset_index(drop=True)
    return df


def count_keywords(text: str, keywords: list[str]) -> int:
    if not isinstance(text, str):
        return 0
    return sum(1 for word in keywords if word in text)


def build_title_features(title: str, event_type: Any, event_source: Any) -> dict[str, Any]:
    title = title if isinstance(title, str) else ""

    positive_count = count_keywords(title, POSITIVE_WORDS)
    negative_count = count_keywords(title, NEGATIVE_WORDS)
    sentiment_score = positive_count - negative_count

    event_type_num = int(event_type) if pd.notna(event_type) else -1
    source_text = str(event_source) if pd.notna(event_source) else ""

    return {
        "event_type": event_type_num,
        "title_length": len(title),
        "positive_keyword_count": positive_count,
        "negative_keyword_count": negative_count,
        "sentiment_score": sentiment_score,
        "is_report": int(event_type_num == 1),
        "is_dividend": int(event_type_num == 3),
        "is_governance": int(event_type_num == 6),
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


def build_label(stock_df: pd.DataFrame, event_idx: int, horizon_days: int = 3) -> tuple[float | None, int | None]:
    if event_idx + horizon_days >= len(stock_df):
        return None, None

    event_close = float(stock_df.iloc[event_idx]["close"])
    future_close = float(stock_df.iloc[event_idx + horizon_days]["close"])

    target_post_return = (future_close - event_close) / event_close if event_close else None
    if target_post_return is None:
        return None, None

    label = 1 if target_post_return >= 0 else 0
    return target_post_return, label


def build_dataset() -> pd.DataFrame:
    stock_df = load_stock_df()
    events_df = load_events_df()

    rows: list[dict[str, Any]] = []

    stock_date_to_idx = {
        row["date"].strftime("%Y-%m-%d"): idx
        for idx, row in stock_df.iterrows()
    }

    for _, event in events_df.iterrows():
        event_date = event["event_date"]
        event_date_str = event_date.strftime("%Y-%m-%d")

        if event_date_str not in stock_date_to_idx:
            continue

        event_idx = stock_date_to_idx[event_date_str]

        # 需要：
        # t-2 本身要有前一天参考，因此最低要求 event_idx >= 3
        # 还需要 event_idx+3 才能生成标签
        if event_idx < 3:
            continue

        target_post_return, label = build_label(stock_df, event_idx, horizon_days=3)
        if label is None:
            continue

        t_minus_2 = stock_df.iloc[event_idx - 2]
        t_minus_3 = stock_df.iloc[event_idx - 3]
        t_minus_1 = stock_df.iloc[event_idx - 1]

        t_minus_2_feat = build_day_features(t_minus_2, t_minus_3)
        t_minus_1_feat = build_day_features(t_minus_1, t_minus_2)

        title_feat = build_title_features(
            title=event["event_title"],
            event_type=event["event_type"],
            event_source=event["event_source"],
        )

        row = {
            "event_date": event_date_str,
            "event_type_raw": int(event["event_type"]) if pd.notna(event["event_type"]) else None,
            "event_source": event["event_source"],
            "event_title": event["event_title"],

            "t_minus_2_date": t_minus_2["date"].strftime("%Y-%m-%d"),
            "t_minus_1_date": t_minus_1["date"].strftime("%Y-%m-%d"),

            "t_minus_2_intraday_return": t_minus_2_feat["intraday_return"],
            "t_minus_2_amplitude": t_minus_2_feat["amplitude"],
            "t_minus_2_volume_change_vs_prev": t_minus_2_feat["volume_change_vs_prev"],
            "t_minus_2_close_change_vs_prev_close": t_minus_2_feat["close_change_vs_prev_close"],

            "t_minus_1_intraday_return": t_minus_1_feat["intraday_return"],
            "t_minus_1_amplitude": t_minus_1_feat["amplitude"],
            "t_minus_1_volume_change_vs_prev": t_minus_1_feat["volume_change_vs_prev"],
            "t_minus_1_close_change_vs_prev_close": t_minus_1_feat["close_change_vs_prev_close"],

            "target_post_3d_return": target_post_return,
            "label": label,
        }

        row.update(title_feat)
        rows.append(row)

    out_df = pd.DataFrame(rows)
    if out_df.empty:
        raise ValueError("构造后的 simulation_train_data 为空，请检查事件数据与股票数据")

    out_df = out_df.sort_values("event_date").reset_index(drop=True)
    return out_df


def main():
    ensure_dir(OUTPUT_DIR)
    df = build_dataset()
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    print(f"已保存模拟预测训练数据: {OUTPUT_CSV}")
    print(f"样本数: {len(df)}")
    print("字段列表:")
    print(df.columns.tolist())
    print(df.head())


if __name__ == "__main__":
    main()