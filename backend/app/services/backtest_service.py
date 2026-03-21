from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[3]
BACKTEST_DIR = ROOT_DIR / "artifacts" / "backtest"
LATEST_RUN_META = BACKTEST_DIR / "latest_backtest_run.json"

SUPPORTED_MODELS = {"simulation_v1", "hybrid_lstm"}

FILE_MAP = {
    "simulation_v1": {
        "signal_csv": BACKTEST_DIR / "backtest_signals_simulation_v1.csv",
        "equity_csv": BACKTEST_DIR / "equity_curve_simulation_v1.csv",
        "drawdown_csv": BACKTEST_DIR / "drawdown_curve_simulation_v1.csv",
        "trade_csv": BACKTEST_DIR / "trade_log_simulation_v1.csv",
        "report_json": BACKTEST_DIR / "backtest_report_simulation_v1.json",
    },
    "hybrid_lstm": {
        "signal_csv": BACKTEST_DIR / "backtest_signals_hybrid_lstm.csv",
        "equity_csv": BACKTEST_DIR / "equity_curve_hybrid_lstm.csv",
        "drawdown_csv": BACKTEST_DIR / "drawdown_curve_hybrid_lstm.csv",
        "trade_csv": BACKTEST_DIR / "trade_log_hybrid_lstm.csv",
        "report_json": BACKTEST_DIR / "backtest_report_hybrid_lstm.json",
    },
}


def ensure_dirs() -> None:
    BACKTEST_DIR.mkdir(parents=True, exist_ok=True)


def validate_model_name(model_name: str) -> str:
    model_name = (model_name or "").strip().lower()
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(f"不支持的 model_name: {model_name}，可选值为 {sorted(SUPPORTED_MODELS)}")
    return model_name


def run_subprocess(cmd: list[str]) -> dict[str, Any]:
    result = subprocess.run(
        cmd,
        cwd=str(ROOT_DIR),
        capture_output=True,
        text=True,
        encoding="utf-8",
    )

    payload = {
        "cmd": cmd,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }

    if result.returncode != 0:
        raise RuntimeError(
            f"命令执行失败\nCMD: {' '.join(cmd)}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )

    return payload


def read_json_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"未找到文件: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv_rows(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    import pandas as pd

    if not path.exists():
        raise FileNotFoundError(f"未找到文件: {path}")

    df = pd.read_csv(path)
    if limit is not None and limit > 0:
        df = df.tail(limit).copy()

    df = df.where(df.notnull(), None)
    return df.to_dict(orient="records")


def save_latest_run_meta(meta: dict[str, Any]) -> None:
    ensure_dirs()
    LATEST_RUN_META.write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def get_file_bundle(model_name: str) -> dict[str, Path]:
    model_name = validate_model_name(model_name)
    return FILE_MAP[model_name]


def build_backtest_signals(model_name: str) -> dict[str, Any]:
    model_name = validate_model_name(model_name)

    cmd = [
        sys.executable,
        "-m",
        "src.backtest.build_backtest_signals",
        "--model-name",
        model_name,
    ]
    return run_subprocess(cmd)


def run_backtest_pipeline(
    model_name: str,
    initial_cash: float = 100000.0,
    commission: float = 0.001,
    hold_days: int = 3,
    prob_threshold: float | None = None,
    stop_loss_pct: float = 0.03,
    take_profit_pct: float = 0.05,
    target_percent: float = 0.95,
) -> dict[str, Any]:
    model_name = validate_model_name(model_name)
    ensure_dirs()

    signal_build_result = build_backtest_signals(model_name)

    cmd = [
        sys.executable,
        "-m",
        "src.backtest.run_backtest",
        "--model-name",
        model_name,
        "--initial-cash",
        str(initial_cash),
        "--commission",
        str(commission),
        "--hold-days",
        str(hold_days),
        "--stop-loss-pct",
        str(stop_loss_pct),
        "--take-profit-pct",
        str(take_profit_pct),
        "--target-percent",
        str(target_percent),
    ]

    if prob_threshold is not None:
        cmd.extend(["--prob-threshold", str(prob_threshold)])

    backtest_run_result = run_subprocess(cmd)

    files = get_file_bundle(model_name)
    report = read_json_file(files["report_json"])

    latest_meta = {
        "model_name": model_name,
        "params": {
            "initial_cash": float(initial_cash),
            "commission": float(commission),
            "hold_days": int(hold_days),
            "prob_threshold": float(prob_threshold) if prob_threshold is not None else None,
            "stop_loss_pct": float(stop_loss_pct),
            "take_profit_pct": float(take_profit_pct),
            "target_percent": float(target_percent),
        },
        "artifacts": {k: str(v) for k, v in files.items()},
        "summary": report.get("summary", {}),
    }
    save_latest_run_meta(latest_meta)

    return {
        "message": "回测运行完成",
        "model_name": model_name,
        "params": latest_meta["params"],
        "summary": report.get("summary", {}),
        "report": report,
        "artifacts": latest_meta["artifacts"],
        "signal_build_stdout": signal_build_result.get("stdout", ""),
        "backtest_run_stdout": backtest_run_result.get("stdout", ""),
    }


def get_latest_backtest_report(model_name: str | None = None) -> dict[str, Any]:
    if model_name:
        model_name = validate_model_name(model_name)
        files = get_file_bundle(model_name)
        report = read_json_file(files["report_json"])
        return {
            "model_name": model_name,
            "report": report,
            "artifacts": {k: str(v) for k, v in files.items()},
        }

    if not LATEST_RUN_META.exists():
        raise FileNotFoundError("尚无最近一次回测记录，请先调用 /api/backtest/run")

    latest_meta = read_json_file(LATEST_RUN_META)
    latest_model_name = validate_model_name(latest_meta["model_name"])
    files = get_file_bundle(latest_model_name)
    report = read_json_file(files["report_json"])

    return {
        "model_name": latest_model_name,
        "latest_meta": latest_meta,
        "report": report,
        "artifacts": {k: str(v) for k, v in files.items()},
    }


def get_backtest_trades(model_name: str, limit: int = 200) -> dict[str, Any]:
    model_name = validate_model_name(model_name)
    files = get_file_bundle(model_name)
    rows = read_csv_rows(files["trade_csv"], limit=limit)

    return {
        "model_name": model_name,
        "total_rows": len(rows),
        "rows": rows,
        "trade_csv": str(files["trade_csv"]),
    }


def get_backtest_equity(model_name: str, limit: int = 2000) -> dict[str, Any]:
    model_name = validate_model_name(model_name)
    files = get_file_bundle(model_name)
    rows = read_csv_rows(files["equity_csv"], limit=limit)

    return {
        "model_name": model_name,
        "total_rows": len(rows),
        "rows": rows,
        "equity_csv": str(files["equity_csv"]),
    }


def get_backtest_drawdown(model_name: str, limit: int = 2000) -> dict[str, Any]:
    model_name = validate_model_name(model_name)
    files = get_file_bundle(model_name)
    rows = read_csv_rows(files["drawdown_csv"], limit=limit)

    return {
        "model_name": model_name,
        "total_rows": len(rows),
        "rows": rows,
        "drawdown_csv": str(files["drawdown_csv"]),
    }