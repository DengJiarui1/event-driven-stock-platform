# 作用：

# 读取股票日线 + 信号表
# 合并成 Backtrader 数据源
# 跑回测
# 输出：
# 资金曲线
# 回撤曲线
# 交易记录
# 风险收益报告 JSON

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import backtrader as bt
import numpy as np
import pandas as pd

from src.backtest.event_signal_strategy import EventPredictionStrategy, EventSignalData


ROOT_DIR = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT_DIR / "data" / "raw"
BACKTEST_DIR = ROOT_DIR / "artifacts" / "backtest"

DEFAULT_STOCK_CSV = RAW_DIR / "stock_price_600519.csv"
DEFAULT_SIGNAL_MAP = {
    "simulation_v1": BACKTEST_DIR / "backtest_signals_simulation_v1.csv",
    "hybrid_lstm": BACKTEST_DIR / "backtest_signals_hybrid_lstm.csv",
}


def ensure_dirs() -> None:
    BACKTEST_DIR.mkdir(parents=True, exist_ok=True)


def load_stock_df(stock_csv: Path) -> pd.DataFrame:
    if not stock_csv.exists():
        raise FileNotFoundError(f"未找到股票数据文件: {stock_csv}")

    df = pd.read_csv(stock_csv)
    required_cols = ["date", "open", "high", "low", "close", "volume"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"股票数据缺少字段: {missing}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).copy()

    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["open", "high", "low", "close", "volume"]).copy()
    df = df.sort_values("date").reset_index(drop=True)

    return df


def load_signal_df(signal_csv: Path) -> pd.DataFrame:
    if not signal_csv.exists():
        raise FileNotFoundError(f"未找到回测信号文件: {signal_csv}")

    df = pd.read_csv(signal_csv)
    required_cols = ["signal_date", "event_signal", "signal_prob", "signal_threshold"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"信号文件缺少字段: {missing}")

    df["signal_date"] = pd.to_datetime(df["signal_date"], errors="coerce")
    df = df.dropna(subset=["signal_date"]).copy()

    df["event_signal"] = pd.to_numeric(df["event_signal"], errors="coerce").fillna(0).astype(int)
    df["signal_prob"] = pd.to_numeric(df["signal_prob"], errors="coerce").fillna(0.0)
    df["signal_threshold"] = pd.to_numeric(df["signal_threshold"], errors="coerce").fillna(0.5)

    if "event_type_raw" not in df.columns:
        df["event_type_raw"] = 0
    df["event_type_raw"] = pd.to_numeric(df["event_type_raw"], errors="coerce").fillna(0).astype(int)

    if "event_title" not in df.columns:
        df["event_title"] = ""
    if "event_source" not in df.columns:
        df["event_source"] = ""

    df = df.sort_values("signal_date").reset_index(drop=True)
    return df


def build_signal_meta_map(signal_df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    meta_map: dict[str, dict[str, Any]] = {}

    for _, row in signal_df.iterrows():
        date_str = row["signal_date"].strftime("%Y-%m-%d")
        meta_map[date_str] = {
            "event_title": str(row.get("event_title", "")),
            "event_source": str(row.get("event_source", "")),
            "event_type_raw": int(row.get("event_type_raw", 0)),
            "signal_prob": float(row.get("signal_prob", 0.0)),
            "signal_threshold": float(row.get("signal_threshold", 0.5)),
        }

    return meta_map


def merge_stock_and_signals(stock_df: pd.DataFrame, signal_df: pd.DataFrame) -> pd.DataFrame:
    daily_signal_df = signal_df.rename(columns={"signal_date": "date"}).copy()

    merged_df = stock_df.merge(
        daily_signal_df[["date", "event_signal", "signal_prob", "signal_threshold", "event_type_raw"]],
        on="date",
        how="left",
    )

    merged_df["event_signal"] = merged_df["event_signal"].fillna(0).astype(int)
    merged_df["signal_prob"] = merged_df["signal_prob"].fillna(0.0).astype(float)
    merged_df["signal_threshold"] = merged_df["signal_threshold"].fillna(0.5).astype(float)
    merged_df["event_type_num"] = merged_df["event_type_raw"].fillna(0).astype(float)

    merged_df = merged_df.set_index("date")
    return merged_df


def compute_sharpe_ratio(equity_df: pd.DataFrame) -> float | None:
    if equity_df.empty or len(equity_df) < 2:
        return None

    equity_df = equity_df.copy()
    equity_df["daily_return"] = equity_df["portfolio_value"].pct_change()
    daily_ret = equity_df["daily_return"].dropna()

    if daily_ret.empty or float(daily_ret.std()) == 0.0:
        return None

    sharpe = (daily_ret.mean() / daily_ret.std()) * np.sqrt(252)
    return float(sharpe)


def compute_drawdown_curve(equity_df: pd.DataFrame) -> pd.DataFrame:
    if equity_df.empty:
        return pd.DataFrame(columns=["date", "portfolio_value", "rolling_peak", "drawdown_pct"])

    df = equity_df.copy()
    df["rolling_peak"] = df["portfolio_value"].cummax()
    df["drawdown_pct"] = (df["portfolio_value"] - df["rolling_peak"]) / df["rolling_peak"]
    return df


def build_backtest_report(
    equity_df: pd.DataFrame,
    drawdown_df: pd.DataFrame,
    trade_df: pd.DataFrame,
    initial_cash: float,
    final_value: float,
    model_name: str,
    params_dict: dict[str, Any],
) -> dict[str, Any]:
    total_return = (final_value - initial_cash) / initial_cash if initial_cash else 0.0

    if len(equity_df) > 1:
        n_days = len(equity_df)
        annual_return = (final_value / initial_cash) ** (252 / n_days) - 1 if initial_cash > 0 else 0.0
    else:
        annual_return = 0.0

    max_drawdown = float(drawdown_df["drawdown_pct"].min()) if not drawdown_df.empty else 0.0
    sharpe_ratio = compute_sharpe_ratio(equity_df)

    trade_count = int(len(trade_df))
    win_trade_df = trade_df[trade_df["pnl_comm"] > 0].copy() if not trade_df.empty else trade_df
    loss_trade_df = trade_df[trade_df["pnl_comm"] <= 0].copy() if not trade_df.empty else trade_df

    win_rate = float(len(win_trade_df) / trade_count) if trade_count > 0 else 0.0
    avg_win = float(win_trade_df["return_pct"].mean()) if not win_trade_df.empty else 0.0
    avg_loss = float(loss_trade_df["return_pct"].mean()) if not loss_trade_df.empty else 0.0
    avg_hold_days = float(trade_df["hold_days"].mean()) if trade_count > 0 else 0.0

    gross_profit = float(win_trade_df["pnl_comm"].sum()) if not win_trade_df.empty else 0.0
    gross_loss = float(loss_trade_df["pnl_comm"].sum()) if not loss_trade_df.empty else 0.0
    profit_factor = (gross_profit / abs(gross_loss)) if gross_loss < 0 else None

    report = {
        "model_name": model_name,
        "strategy_name": "EventPredictionStrategy",
        "params": params_dict,
        "summary": {
            "initial_cash": float(initial_cash),
            "final_value": float(final_value),
            "total_return": float(total_return),
            "annual_return": float(annual_return),
            "max_drawdown": float(max_drawdown),
            "sharpe_ratio": float(sharpe_ratio) if sharpe_ratio is not None else None,
            "trade_count": trade_count,
            "win_rate": float(win_rate),
            "avg_win": float(avg_win),
            "avg_loss": float(avg_loss),
            "avg_hold_days": float(avg_hold_days),
            "profit_factor": float(profit_factor) if profit_factor is not None else None,
        },
    }

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="运行 Backtrader 事件驱动策略回测")
    parser.add_argument(
        "--model-name",
        default="simulation_v1",
        choices=["simulation_v1", "hybrid_lstm"],
        help="选择信号来源模型",
    )
    parser.add_argument(
        "--stock-csv",
        default=str(DEFAULT_STOCK_CSV),
        help="股票日线 CSV 路径",
    )
    parser.add_argument(
        "--signal-csv",
        default=None,
        help="回测信号 CSV 路径，默认按 model_name 自动匹配",
    )
    parser.add_argument("--initial-cash", type=float, default=100000.0, help="初始资金")
    parser.add_argument("--commission", type=float, default=0.001, help="单边手续费率")
    parser.add_argument("--hold-days", type=int, default=3, help="固定持有天数")
    parser.add_argument("--prob-threshold", type=float, default=None, help="手动覆盖概率阈值")
    parser.add_argument("--stop-loss-pct", type=float, default=0.03, help="止损比例，例如 0.03 表示 -3%")
    parser.add_argument("--take-profit-pct", type=float, default=0.05, help="止盈比例，例如 0.05 表示 +5%")

    parser.add_argument(
    "--target-percent",
    type=float,
    default=0.95,
    help="目标仓位比例，例如 0.95 表示使用 95%% 资金建仓",
)

    args = parser.parse_args()

    ensure_dirs()

    stock_csv = Path(args.stock_csv)
    signal_csv = Path(args.signal_csv) if args.signal_csv else DEFAULT_SIGNAL_MAP[args.model_name]

    stock_df = load_stock_df(stock_csv)
    signal_df = load_signal_df(signal_csv)

    signal_meta_map = build_signal_meta_map(signal_df)
    merged_df = merge_stock_and_signals(stock_df, signal_df)

    datafeed = EventSignalData(dataname=merged_df)

    cerebro = bt.Cerebro(stdstats=False)
    cerebro.broker.setcash(args.initial_cash)
    cerebro.broker.setcommission(commission=args.commission)
    cerebro.adddata(datafeed)

    cerebro.addstrategy(
        EventPredictionStrategy,
        hold_days=args.hold_days,
        prob_threshold=args.prob_threshold,
        stop_loss_pct=args.stop_loss_pct,
        take_profit_pct=args.take_profit_pct,
        target_percent=args.target_percent,
        signal_meta_map=signal_meta_map,
    )

    print(f"模型名称: {args.model_name}")
    print(f"股票数据: {stock_csv}")
    print(f"信号文件: {signal_csv}")
    print(f"初始资金: {args.initial_cash}")
    print(f"手续费率: {args.commission}")
    print(f"持有天数: {args.hold_days}")
    print(f"阈值覆盖: {args.prob_threshold}")
    print(f"止损比例: {args.stop_loss_pct}")
    print(f"止盈比例: {args.take_profit_pct}")
    print(f"目标仓位: {args.target_percent}")

    results = cerebro.run()
    strat: EventPredictionStrategy = results[0]

    final_value = float(cerebro.broker.getvalue())

    equity_df = pd.DataFrame(strat.equity_curve)
    drawdown_df = compute_drawdown_curve(equity_df)

    trade_df = pd.DataFrame(strat.trade_log)
    if not trade_df.empty:
        trade_df = trade_df.sort_values(["entry_date", "exit_date"]).reset_index(drop=True)

    report = build_backtest_report(
        equity_df=equity_df,
        drawdown_df=drawdown_df,
        trade_df=trade_df,
        initial_cash=float(args.initial_cash),
        final_value=final_value,
        model_name=args.model_name,
        params_dict={
            "hold_days": args.hold_days,
            "prob_threshold": args.prob_threshold,
            "stop_loss_pct": args.stop_loss_pct,
            "take_profit_pct": args.take_profit_pct,
            "commission": args.commission,
            "target_percent": args.target_percent,
        },
    )

    # 输出文件
    suffix = args.model_name

    equity_csv = BACKTEST_DIR / f"equity_curve_{suffix}.csv"
    drawdown_csv = BACKTEST_DIR / f"drawdown_curve_{suffix}.csv"
    trade_csv = BACKTEST_DIR / f"trade_log_{suffix}.csv"
    report_json = BACKTEST_DIR / f"backtest_report_{suffix}.json"

    equity_df.to_csv(equity_csv, index=False, encoding="utf-8-sig")
    drawdown_df.to_csv(drawdown_csv, index=False, encoding="utf-8-sig")
    trade_df.to_csv(trade_csv, index=False, encoding="utf-8-sig")
    report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n回测完成。")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("\n输出文件：")
    print(equity_csv)
    print(drawdown_csv)
    print(trade_csv)
    print(report_json)


if __name__ == "__main__":
    main()