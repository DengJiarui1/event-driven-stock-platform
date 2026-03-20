from __future__ import annotations

import json
import subprocess
import sys
import threading
import traceback
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[3]
INFORMER_DIR = ROOT_DIR / "Informer2020"
RAW_STOCK_CSV = ROOT_DIR / "data" / "raw" / "stock_price_600519.csv"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
DEFAULT_INPUT_CSV = PROCESSED_DIR / "informer_600519.csv"
ARTIFACT_DIR = ROOT_DIR / "artifacts" / "informer"
RESULTS_DIR = INFORMER_DIR / "results"

JOBS: dict[str, dict[str, Any]] = {}
JOB_LOCK = threading.Lock()


def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def to_rel_str(path: Path) -> str:
    return str(path.relative_to(ROOT_DIR)).replace("\\", "/")


def to_python_value(value: Any) -> Any:
    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
    except Exception:
        pass

    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")

    return value


def sanitize_for_json(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize_for_json(v) for v in obj]
    if isinstance(obj, tuple):
        return [sanitize_for_json(v) for v in obj]
    return to_python_value(obj)


def prepare_default_informer_csv() -> Path:
    """
    从 stock_price_600519.csv 自动生成 Informer 可用的单变量收益率数据：
    列格式：date, return
    """
    ensure_dir(PROCESSED_DIR)

    if not RAW_STOCK_CSV.exists():
        raise FileNotFoundError(f"原始股票文件不存在: {RAW_STOCK_CSV}")

    df = pd.read_csv(RAW_STOCK_CSV)
    if "date" not in df.columns or "close" not in df.columns:
        raise ValueError("stock_price_600519.csv 必须包含 date 和 close 列")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    df["return"] = pd.to_numeric(df["close"], errors="coerce").pct_change().fillna(0.0)

    out_df = df[["date", "return"]].copy()
    out_df["date"] = out_df["date"].dt.strftime("%Y-%m-%d")
    out_df.to_csv(DEFAULT_INPUT_CSV, index=False, encoding="utf-8-sig")

    return DEFAULT_INPUT_CSV


def _update_job(job_id: str, **kwargs) -> None:
    with JOB_LOCK:
        if job_id not in JOBS:
            return
        JOBS[job_id].update(kwargs)


def list_jobs() -> list[dict[str, Any]]:
    with JOB_LOCK:
        items = list(JOBS.values())

    items.sort(key=lambda x: x.get("created_at", ""), reverse=True)
    return sanitize_for_json(items)


def get_job(job_id: str) -> dict[str, Any] | None:
    with JOB_LOCK:
        item = JOBS.get(job_id)
    return sanitize_for_json(item) if item else None


def get_latest_success_job() -> dict[str, Any] | None:
    jobs = list_jobs()
    success_jobs = [job for job in jobs if job.get("status") == "success"]
    if not success_jobs:
        return None
    success_jobs.sort(key=lambda x: x.get("finished_at", ""), reverse=True)
    return success_jobs[0]


def _locate_new_result_dir(before_names: set[str]) -> Path:
    if not RESULTS_DIR.exists():
        raise FileNotFoundError(f"Informer results 目录不存在: {RESULTS_DIR}")

    all_dirs = [p for p in RESULTS_DIR.iterdir() if p.is_dir()]
    new_dirs = [p for p in all_dirs if p.name not in before_names]

    candidates = new_dirs if new_dirs else all_dirs
    if not candidates:
        raise FileNotFoundError("未找到 Informer 结果目录，请检查 main_informer.py 是否成功运行")

    return max(candidates, key=lambda p: p.stat().st_mtime)


def _save_prediction_chart(pred: np.ndarray, true: np.ndarray, output_png: Path) -> None:
    pred_arr = np.array(pred)
    true_arr = np.array(true)

    if pred_arr.ndim == 3:
        pred_line = pred_arr[0, :, 0]
    elif pred_arr.ndim == 2:
        pred_line = pred_arr[0]
    else:
        pred_line = pred_arr.reshape(-1)

    if true_arr.ndim == 3:
        true_line = true_arr[0, :, 0]
    elif true_arr.ndim == 2:
        true_line = true_arr[0]
    else:
        true_line = true_arr.reshape(-1)

    plt.figure(figsize=(10, 4.5))
    plt.plot(true_line, label="True")
    plt.plot(pred_line, label="Pred")
    plt.title("Informer Prediction vs True")
    plt.xlabel("Forecast Step")
    plt.ylabel("Return")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_png, dpi=150)
    plt.close()


def _build_summary(
    job_id: str,
    work_dir: Path,
    result_dir: Path,
    config: dict[str, Any],
    input_csv: Path,
) -> dict[str, Any]:
    metrics_path = result_dir / "metrics.npy"
    pred_path = result_dir / "pred.npy"
    true_path = result_dir / "true.npy"

    metrics = None
    metrics_map = {}

    if metrics_path.exists():
        metrics = np.load(metrics_path).tolist()
        # official order: mae, mse, rmse, mape, mspe
        if len(metrics) >= 5:
            metrics_map = {
                "mae": float(metrics[0]),
                "mse": float(metrics[1]),
                "rmse": float(metrics[2]),
                "mape": float(metrics[3]),
                "mspe": float(metrics[4]),
            }

    preview = None
    chart_url = None
    pred_shape = None
    true_shape = None

    if pred_path.exists() and true_path.exists():
        pred = np.load(pred_path)
        true = np.load(true_path)

        pred_shape = list(pred.shape)
        true_shape = list(true.shape)

        preview_pred = pred[0].reshape(-1).tolist()[:10] if pred.size else []
        preview_true = true[0].reshape(-1).tolist()[:10] if true.size else []

        preview = {
            "pred_head": [float(x) for x in preview_pred],
            "true_head": [float(x) for x in preview_true],
        }

        chart_path = work_dir / "prediction_compare.png"
        _save_prediction_chart(pred, true, chart_path)
        chart_url = f"/static/informer/jobs/{job_id}/prediction_compare.png"

    summary = {
        "job_id": job_id,
        "status": "success",
        "created_at": config.get("created_at"),
        "finished_at": now_str(),
        "input_csv": to_rel_str(input_csv),
        "result_dir": to_rel_str(result_dir),
        "chart_url": chart_url,
        "metrics": metrics_map,
        "metrics_raw": metrics,
        "pred_shape": pred_shape,
        "true_shape": true_shape,
        "preview": preview,
        "config": config,
    }
    return sanitize_for_json(summary)


def _run_informer_job(job_id: str, config: dict[str, Any]) -> None:
    work_dir = ARTIFACT_DIR / "jobs" / job_id
    ensure_dir(work_dir)

    try:
        if not INFORMER_DIR.exists():
            raise FileNotFoundError(f"Informer2020 目录不存在: {INFORMER_DIR}")

        input_csv = config.get("data_path")
        if input_csv:
            input_csv_path = Path(input_csv)
            if not input_csv_path.is_absolute():
                input_csv_path = ROOT_DIR / input_csv_path
        else:
            input_csv_path = prepare_default_informer_csv()

        if not input_csv_path.exists():
            raise FileNotFoundError(f"Informer 输入文件不存在: {input_csv_path}")

        ensure_dir(RESULTS_DIR)

        before_names = {p.name for p in RESULTS_DIR.iterdir() if p.is_dir()} if RESULTS_DIR.exists() else set()

        cmd = [
            sys.executable,
            "main_informer.py",
            "--model", "informer",
            "--data", "custom",
            "--features", "S",
            "--target", str(config["target"]),
            "--root_path", str(input_csv_path.parent),
            "--data_path", input_csv_path.name,
            "--seq_len", str(config["seq_len"]),
            "--label_len", str(config["label_len"]),
            "--pred_len", str(config["pred_len"]),
            "--enc_in", "1",
            "--dec_in", "1",
            "--c_out", "1",
            "--itr", "1",
            "--train_epochs", str(config["train_epochs"]),
            "--batch_size", str(config["batch_size"]),
            "--patience", str(config["patience"]),
            "--learning_rate", str(config["learning_rate"]),
            "--d_model", str(config["d_model"]),
            "--n_heads", str(config["n_heads"]),
            "--e_layers", str(config["e_layers"]),
            "--d_layers", str(config["d_layers"]),
            "--d_ff", str(config["d_ff"]),
            "--dropout", str(config["dropout"]),
            "--des", str(config["run_name"]),
        ]

        _update_job(job_id, status="running", started_at=now_str(), command=" ".join(cmd))

        proc = subprocess.run(
            cmd,
            cwd=INFORMER_DIR,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="ignore",
        )

        log_text = (
            f"$ {' '.join(cmd)}\n\n"
            f"[STDOUT]\n{proc.stdout}\n\n"
            f"[STDERR]\n{proc.stderr}\n"
        )
        (work_dir / "run.log").write_text(log_text, encoding="utf-8")

        if proc.returncode != 0:
            raise RuntimeError(proc.stderr.strip() or proc.stdout.strip() or f"Informer 运行失败，退出码={proc.returncode}")

        result_dir = _locate_new_result_dir(before_names)
        summary = _build_summary(
            job_id=job_id,
            work_dir=work_dir,
            result_dir=result_dir,
            config=config,
            input_csv=input_csv_path,
        )

        (work_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        _update_job(
            job_id,
            status="success",
            finished_at=summary["finished_at"],
            result=summary,
            log_url=f"/static/informer/jobs/{job_id}/run.log",
        )

    except Exception as e:
        traceback.print_exc()
        _update_job(
            job_id,
            status="failed",
            finished_at=now_str(),
            error=str(e),
            log_url=f"/static/informer/jobs/{job_id}/run.log" if (work_dir / "run.log").exists() else None,
        )


def create_job(payload: dict[str, Any]) -> dict[str, Any]:
    job_id = uuid.uuid4().hex[:12]

    config = {
        "data_path": payload.get("data_path"),
        "target": payload.get("target", "return"),
        "seq_len": int(payload.get("seq_len", 60)),
        "label_len": int(payload.get("label_len", 30)),
        "pred_len": int(payload.get("pred_len", 5)),
        "train_epochs": int(payload.get("train_epochs", 6)),
        "batch_size": int(payload.get("batch_size", 16)),
        "patience": int(payload.get("patience", 3)),
        "learning_rate": float(payload.get("learning_rate", 0.0001)),
        "d_model": int(payload.get("d_model", 128)),
        "n_heads": int(payload.get("n_heads", 4)),
        "e_layers": int(payload.get("e_layers", 2)),
        "d_layers": int(payload.get("d_layers", 1)),
        "d_ff": int(payload.get("d_ff", 256)),
        "dropout": float(payload.get("dropout", 0.05)),
        "run_name": payload.get("run_name", "maotai_return_only"),
        "created_at": now_str(),
    }

    job_record = {
        "job_id": job_id,
        "status": "queued",
        "created_at": config["created_at"],
        "config": config,
        "result": None,
        "error": None,
        "log_url": None,
    }

    with JOB_LOCK:
        JOBS[job_id] = job_record

    thread = threading.Thread(target=_run_informer_job, args=(job_id, config), daemon=True)
    thread.start()

    return sanitize_for_json(job_record)