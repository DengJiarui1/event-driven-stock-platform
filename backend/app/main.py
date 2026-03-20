from pathlib import Path
from typing import Optional

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend.app.services.predict_service import (
    list_predict_options,
    predict_existing_event,
)

from fastapi.staticfiles import StaticFiles

from backend.app.services.informer_service import (
    create_job as create_informer_job,
    get_job as get_informer_job,
    get_latest_success_job,
    list_jobs as list_informer_jobs,
    prepare_default_informer_csv,
)

app = FastAPI(
    title="Event-driven Stock Analysis API",
    version="1.0.0"
)

# 允许前端访问
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 项目根目录
ROOT_DIR = Path(__file__).resolve().parents[2]

EVENTS_CSV = ROOT_DIR / "data" / "interim" / "events.csv"
MODEL_COMPARISON_CSV = ROOT_DIR / "artifacts" / "reports" / "model_comparison.csv"
EVENT_WINDOWS_CSV = ROOT_DIR / "data" / "interim" / "event_windows.csv"

ARTIFACTS_DIR = ROOT_DIR / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

app.mount("/static", StaticFiles(directory=ARTIFACTS_DIR), name="static")


class PredictRequest(BaseModel):
    event_date: str
    event_type: int = 1
    event_title: str

class InformerJobRequest(BaseModel):
    data_path: Optional[str] = None
    target: str = "return"
    seq_len: int = 60
    label_len: int = 30
    pred_len: int = 5
    train_epochs: int = 6
    batch_size: int = 16
    patience: int = 3
    learning_rate: float = 0.0001
    d_model: int = 128
    n_heads: int = 4
    e_layers: int = 2
    d_layers: int = 1
    d_ff: int = 256
    dropout: float = 0.05
    run_name: str = "maotai_return_only"


def ensure_file_exists(path: Path):
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"文件不存在: {path}")


def load_events_df() -> pd.DataFrame:
    ensure_file_exists(EVENTS_CSV)
    df = pd.read_csv(EVENTS_CSV)

    # 日期字段格式化
    if "event_date" in df.columns:
        df["event_date"] = pd.to_datetime(df["event_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    if "raw_event_date" in df.columns:
        df["raw_event_date"] = pd.to_datetime(df["raw_event_date"], errors="coerce").dt.strftime("%Y-%m-%d")

    # 按事件日期倒序
    if "event_date" in df.columns:
        df = df.sort_values(by="event_date", ascending=False)

    # 空值处理
    df = df.where(pd.notnull(df), None)

    return df


def load_model_df() -> pd.DataFrame:
    ensure_file_exists(MODEL_COMPARISON_CSV)
    df = pd.read_csv(MODEL_COMPARISON_CSV)

    # 统一字段名
    rename_map = {
        "Model": "model",
        "Accuracy": "accuracy",
    }
    df = df.rename(columns=rename_map)

    # 只保留前端要用的字段
    keep_cols = [col for col in ["model", "accuracy"] if col in df.columns]
    df = df[keep_cols]

    # 转数值
    if "accuracy" in df.columns:
        df["accuracy"] = pd.to_numeric(df["accuracy"], errors="coerce")

    # 按准确率倒序
    if "accuracy" in df.columns:
        df = df.sort_values(by="accuracy", ascending=False)

    df = df.where(pd.notnull(df), None)

    return df

def load_event_windows_df() -> pd.DataFrame:
    ensure_file_exists(EVENT_WINDOWS_CSV)
    df = pd.read_csv(EVENT_WINDOWS_CSV)

    # 日期字段格式化
    for col in ["date", "event_date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce").dt.strftime("%Y-%m-%d")

    # 数值字段转成标准数值类型
    numeric_cols = ["open", "close", "high", "low", "volume", "event_type"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # 排序：先按事件日期，再按窗口内日期
    sort_cols = [col for col in ["event_date", "date"] if col in df.columns]
    if sort_cols:
        df = df.sort_values(by=sort_cols, ascending=[False, True] if len(sort_cols) == 2 else False)

    df = df.where(pd.notnull(df), None)
    return df


def get_best_model_name() -> str:
    df = load_model_df()
    if df.empty:
        return "N/A"
    return str(df.iloc[0]["model"])


def simple_predict_logic(event_title: str) -> tuple[str, float]:
    """
    这是演示版预测逻辑：
    后面你可以替换成真正的模型推理。
    """
    positive_keywords = [
        "增长", "预增", "盈利", "分红", "回购", "中标", "合作", "业绩", "创新高", "利好"
    ]
    negative_keywords = [
        "下滑", "亏损", "减持", "处罚", "风险", "问询", "暴跌", "诉讼", "减值", "利空"
    ]

    title = event_title or ""
    score = 0.50

    for word in positive_keywords:
        if word in title:
            score += 0.08

    for word in negative_keywords:
        if word in title:
            score -= 0.08

    score = max(0.05, min(0.95, score))

    label = "上涨" if score >= 0.5 else "下跌"
    return label, round(score, 4)


@app.get("/")
def root():
    return {
        "message": "FastAPI backend is running."
    }


@app.get("/api/health")
def health():
    return {
        "status": "ok"
    }


@app.get("/api/events")
def get_events(limit: int = Query(default=100, ge=1, le=500)):
    df = load_events_df()
    records = df.head(limit).to_dict(orient="records")
    return records


@app.get("/api/model-comparison")
def get_model_comparison():
    df = load_model_df()
    records = df.to_dict(orient="records")
    return records


@app.get("/api/latest-prediction")
def get_latest_prediction():
    events_df = load_events_df()

    if events_df.empty:
        return {
            "event_date": None,
            "event_title": None,
            "prediction_label": None,
            "probability": None,
            "model_name": "LSTM (Event-driven, PyTorch)",
        }

    latest_event_date = str(events_df.iloc[0]["event_date"])

    try:
        return predict_existing_event(latest_event_date)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"最新事件真实预测失败: {e}")

@app.get("/api/event-windows")
def get_event_windows(
    event_date: Optional[str] = Query(default=None, description="按事件日期筛选，如 2025-12-11"),
    limit: int = Query(default=500, ge=1, le=5000)
):
    """
    返回事件窗口数据：
    - 不传 event_date：返回全部窗口记录
    - 传 event_date：返回某个事件对应的窗口记录
    """
    df = load_event_windows_df()

    if event_date:
        df = df[df["event_date"] == event_date]

    records = df.head(limit).to_dict(orient="records")
    return records

@app.get("/api/event-window-dates")
def get_event_window_dates():
    df = load_event_windows_df()

    if "event_date" not in df.columns:
        return []

    dates = (
        df["event_date"]
        .dropna()
        .astype(str)
        .drop_duplicates()
        .sort_values(ascending=False)
        .tolist()
    )
    return dates

@app.get("/api/predict-options")
def get_predict_options(limit: int = Query(default=200, ge=1, le=500)):
    return list_predict_options(limit=limit)

@app.get("/api/predict-by-date")
def predict_by_date(event_date: str = Query(..., description="事件日期，如 2025-12-11")):
    try:
        return predict_existing_event(event_date)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/predict")
def predict(payload: PredictRequest):
    """
    真实预测版：
    当前只支持对历史事件库中已有 event_date 的事件做真实模型推理。
    """
    try:
        return predict_existing_event(payload.event_date)
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"真实预测失败：{e}。当前版本只支持对 events.csv 中已有事件做预测。"
        )
    

@app.get("/api/informer/default-input")
def get_informer_default_input():
    path = prepare_default_informer_csv()
    return {
        "data_path": str(path.relative_to(ROOT_DIR)).replace("\\", "/")
    }


@app.post("/api/informer/jobs")
def create_informer_experiment(payload: InformerJobRequest):
    return create_informer_job(payload.model_dump())


@app.get("/api/informer/jobs")
def get_informer_jobs():
    return list_informer_jobs()


@app.get("/api/informer/jobs/{job_id}")
def get_informer_job_detail(job_id: str):
    job = get_informer_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"未找到 job_id={job_id}")
    return job


@app.get("/api/informer/latest")
def get_latest_informer_result():
    latest = get_latest_success_job()
    if not latest:
        return None
    return latest