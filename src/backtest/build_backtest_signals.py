# 作用：

# 从模型预测结果 CSV 里提取回测信号
# 默认支持：
# simulation_v1
# hybrid_lstm
# 输出：
# artifacts/backtest/backtest_signals_<model>.csv

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[2]
REPORT_DIR = ROOT_DIR / "artifacts" / "reports"
BACKTEST_DIR = ROOT_DIR / "artifacts" / "backtest"

DEFAULT_INPUT_MAP = {
    "simulation_v1": REPORT_DIR / "simulation_test_predictions.csv",
    "hybrid_lstm": REPORT_DIR / "sim_hybrid_test_predictions.csv",
}

DEFAULT_OUTPUT_MAP = {
    "simulation_v1": BACKTEST_DIR / "backtest_signals_simulation_v1.csv",
    "hybrid_lstm": BACKTEST_DIR / "backtest_signals_hybrid_lstm.csv",
}


def ensure_dirs() -> None:
    BACKTEST_DIR.mkdir(parents=True, exist_ok=True)


def detect_pred_prob_col(df: pd.DataFrame) -> str:
    candidates = ["pred_prob_up", "probability"]
    for col in candidates:
        if col in df.columns:
            return col
    raise ValueError(f"预测结果文件中未找到概率列，候选列: {candidates}")


def detect_signal_threshold_col(df: pd.DataFrame) -> str | None:
    candidates = ["used_threshold", "decision_threshold", "best_threshold"]
    for col in candidates:
        if col in df.columns:
            return col
    return None


def build_signal_col(df: pd.DataFrame) -> pd.Series:
    if "y_pred" in df.columns:
        return pd.to_numeric(df["y_pred"], errors="coerce").fillna(0).astype(int)

    if "pred_label" in df.columns:
        return df["pred_label"].astype(str).map(lambda x: 1 if x == "上涨" else 0).astype(int)

    if "prediction_label" in df.columns:
        return df["prediction_label"].astype(str).map(lambda x: 1 if x == "上涨" else 0).astype(int)

    raise ValueError("预测结果文件中未找到 y_pred / pred_label / prediction_label 字段")


def load_prediction_df(input_csv: Path) -> pd.DataFrame:
    if not input_csv.exists():
        raise FileNotFoundError(f"未找到预测结果文件: {input_csv}")

    df = pd.read_csv(input_csv)
    if df.empty:
        raise ValueError(f"预测结果文件为空: {input_csv}")

    if "event_date" not in df.columns:
        raise ValueError("预测结果文件缺少 event_date 字段")

    df["event_date"] = pd.to_datetime(df["event_date"], errors="coerce")
    df = df.dropna(subset=["event_date"]).copy()
    df = df.sort_values("event_date").reset_index(drop=True)

    prob_col = detect_pred_prob_col(df)
    df[prob_col] = pd.to_numeric(df[prob_col], errors="coerce").fillna(0.0)

    threshold_col = detect_signal_threshold_col(df)
    if threshold_col:
        df[threshold_col] = pd.to_numeric(df[threshold_col], errors="coerce")

    df["event_signal"] = build_signal_col(df)

    # 兼容字段
    if "event_type_raw" not in df.columns and "event_type" in df.columns:
        df["event_type_raw"] = df["event_type"]

    if "event_title" not in df.columns:
        df["event_title"] = ""

    if "event_source" not in df.columns:
        df["event_source"] = ""

    return df


def aggregate_daily_signals(
    df: pd.DataFrame,
    model_name: str,
    default_threshold: float,
) -> pd.DataFrame:
    prob_col = detect_pred_prob_col(df)
    threshold_col = detect_signal_threshold_col(df)

    # 同一交易日如存在多条事件，优先保留“上涨概率最高”的那条
    pick_idx = df.groupby("event_date")[prob_col].idxmax()
    daily_df = df.loc[pick_idx].copy().sort_values("event_date").reset_index(drop=True)

    daily_df["signal_date"] = daily_df["event_date"].dt.strftime("%Y-%m-%d")
    daily_df["signal_prob"] = daily_df[prob_col].astype(float)
    daily_df["signal_threshold"] = (
        daily_df[threshold_col].fillna(default_threshold).astype(float)
        if threshold_col
        else float(default_threshold)
    )

    output_df = pd.DataFrame(
        {
            "signal_date": daily_df["signal_date"],
            "event_signal": daily_df["event_signal"].astype(int),
            "signal_prob": daily_df["signal_prob"].astype(float),
            "signal_threshold": daily_df["signal_threshold"].astype(float),
            "model_name": model_name,
            "event_title": daily_df["event_title"].astype(str),
            "event_source": daily_df["event_source"].astype(str),
            "event_type_raw": pd.to_numeric(daily_df["event_type_raw"], errors="coerce").fillna(0).astype(int),
        }
    )

    return output_df


def main() -> None:
    parser = argparse.ArgumentParser(description="构造 Backtrader 回测信号表")
    parser.add_argument(
        "--model-name",
        default="simulation_v1",
        choices=["simulation_v1", "hybrid_lstm"],
        help="选择回测信号来源模型",
    )
    parser.add_argument(
        "--input-csv",
        default=None,
        help="自定义预测结果 CSV 路径，默认按 model_name 自动匹配",
    )
    parser.add_argument(
        "--output-csv",
        default=None,
        help="自定义信号输出 CSV 路径",
    )
    parser.add_argument(
        "--default-threshold",
        type=float,
        default=0.50,
        help="当预测结果文件中没有 used_threshold / decision_threshold 时使用该默认值",
    )

    args = parser.parse_args()

    ensure_dirs()

    input_csv = Path(args.input_csv) if args.input_csv else DEFAULT_INPUT_MAP[args.model_name]
    output_csv = Path(args.output_csv) if args.output_csv else DEFAULT_OUTPUT_MAP[args.model_name]

    pred_df = load_prediction_df(input_csv)
    signal_df = aggregate_daily_signals(
        df=pred_df,
        model_name=args.model_name,
        default_threshold=args.default_threshold,
    )

    signal_df.to_csv(output_csv, index=False, encoding="utf-8-sig")

    print(f"模型名称: {args.model_name}")
    print(f"输入文件: {input_csv}")
    print(f"输出文件: {output_csv}")
    print(f"信号总数: {len(signal_df)}")
    print(signal_df.head())


if __name__ == "__main__":
    main()