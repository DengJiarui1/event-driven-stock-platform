import json

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

from backend.app.services.sim_predict_service import (
    get_simulate_context,
    predict_new_event_simulation,
)

from backend.app.services.sim_hybrid_predict_service import (
    predict_new_event_simulation_hybrid,
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

SIM_REPORT_JSON = ROOT_DIR / "artifacts" / "reports" / "simulation_model_report.json"
SIM_TEST_PRED_CSV = ROOT_DIR / "artifacts" / "reports" / "simulation_test_predictions.csv"

SIM_POSITIVE_WORDS = [
    "增长", "预增", "分红", "回购", "中标", "合作", "利好", "创新高", "增持", "派息"
]

SIM_NEGATIVE_WORDS = [
    "亏损", "下滑", "减持", "风险", "处罚", "问询", "诉讼", "减值", "利空", "下跌"
]

SIM_NEUTRAL_WORDS = [
    "年报", "半年报", "季报", "公告", "董事会", "股东大会", "议案", "交易", "收购", "治理"
]

SIM_KEYWORD_POOL = list(dict.fromkeys(
    SIM_POSITIVE_WORDS + SIM_NEGATIVE_WORDS + SIM_NEUTRAL_WORDS
))

SIM_HYBRID_REPORT_JSON = ROOT_DIR / "artifacts" / "reports" / "sim_hybrid_report.json"
SIM_HYBRID_TEST_PRED_CSV = ROOT_DIR / "artifacts" / "reports" / "sim_hybrid_test_predictions.csv"

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

class SimulatePredictRequest(BaseModel):
    event_date: str
    event_type: int
    event_source: Optional[str] = "manual"
    event_title: str
    # model_version: Optional[str] = "simulation_v1"
    model_version: Optional[str] = "hybrid_lstm"         # 默认模型调整

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


def analyze_wrong_title_keywords(wrong_df: pd.DataFrame) -> dict:
    if wrong_df.empty or "event_title" not in wrong_df.columns:
        return {
            "total_wrong_titles": 0,
            "titles_with_keyword_count": 0,
            "no_keyword_match_count": 0,
            "top_keywords": [],
            "positive_hits": [],
            "negative_hits": [],
            "summary_text": "当前没有误判样本，无法进行标题关键词分析。"
        }

    titles = wrong_df["event_title"].fillna("").astype(str).tolist()

    keyword_counter = {}
    positive_counter = {}
    negative_counter = {}

    titles_with_keyword_count = 0
    no_keyword_match_count = 0

    for title in titles:
        matched_any = False

        for kw in SIM_KEYWORD_POOL:
            if kw and kw in title:
                keyword_counter[kw] = keyword_counter.get(kw, 0) + 1
                matched_any = True

        for kw in SIM_POSITIVE_WORDS:
            if kw and kw in title:
                positive_counter[kw] = positive_counter.get(kw, 0) + 1

        for kw in SIM_NEGATIVE_WORDS:
            if kw and kw in title:
                negative_counter[kw] = negative_counter.get(kw, 0) + 1

        if matched_any:
            titles_with_keyword_count += 1
        else:
            no_keyword_match_count += 1

    top_keywords = sorted(
        [{"keyword": k, "count": v} for k, v in keyword_counter.items()],
        key=lambda x: x["count"],
        reverse=True
    )[:10]

    positive_hits = sorted(
        [{"keyword": k, "count": v} for k, v in positive_counter.items()],
        key=lambda x: x["count"],
        reverse=True
    )[:10]

    negative_hits = sorted(
        [{"keyword": k, "count": v} for k, v in negative_counter.items()],
        key=lambda x: x["count"],
        reverse=True
    )[:10]

    if not top_keywords:
        summary_text = "误判样本标题中未命中预设关键词，说明当前规则关键词体系对这部分样本的覆盖仍然不足。"
    else:
        top_kw = top_keywords[0]["keyword"]
        top_kw_count = top_keywords[0]["count"]

        if len(positive_hits) > len(negative_hits):
            sentiment_text = "误判样本标题中利好类关键词更常出现"
        elif len(negative_hits) > len(positive_hits):
            sentiment_text = "误判样本标题中利空类关键词更常出现"
        else:
            sentiment_text = "误判样本标题中的利好和利空关键词分布较为接近"

        summary_text = (
            f"误判样本标题中高频关键词以“{top_kw}”最为突出，共出现 {top_kw_count} 次；"
            f"{sentiment_text}；"
            f"共有 {no_keyword_match_count} 条误判样本未命中预设关键词。"
        )

    return {
        "total_wrong_titles": int(len(titles)),
        "titles_with_keyword_count": int(titles_with_keyword_count),
        "no_keyword_match_count": int(no_keyword_match_count),
        "top_keywords": top_keywords,
        "positive_hits": positive_hits,
        "negative_hits": negative_hits,
        "summary_text": summary_text,
    }

def analyze_title_sentiment(all_df: pd.DataFrame, wrong_df: pd.DataFrame) -> dict:
    if all_df.empty:
        return {
            "avg_wrong_sentiment": None,
            "avg_correct_sentiment": None,
            "wrong_positive_count": 0,
            "wrong_neutral_count": 0,
            "wrong_negative_count": 0,
            "dominant_wrong_sentiment_label": "暂无数据",
            "sentiment_gap": None,
            "summary_text": "当前没有可用样本，无法进行标题情感得分分析。"
        }

    work_df = all_df.copy()

    # 统一构造 sentiment_score
    if "sentiment_score" in work_df.columns:
        work_df["sentiment_score"] = pd.to_numeric(work_df["sentiment_score"], errors="coerce").fillna(0)
    else:
        pos_col = pd.to_numeric(work_df.get("positive_keyword_count", 0), errors="coerce").fillna(0)
        neg_col = pd.to_numeric(work_df.get("negative_keyword_count", 0), errors="coerce").fillna(0)
        work_df["sentiment_score"] = pos_col - neg_col

    correct_df = work_df[work_df["is_correct"] == True].copy()
    wrong_sent_df = work_df[work_df["is_correct"] == False].copy()

    avg_wrong_sentiment = None
    if not wrong_sent_df.empty:
        avg_wrong_sentiment = wrong_sent_df["sentiment_score"].mean()
        avg_wrong_sentiment = None if pd.isna(avg_wrong_sentiment) else float(avg_wrong_sentiment)

    avg_correct_sentiment = None
    if not correct_df.empty:
        avg_correct_sentiment = correct_df["sentiment_score"].mean()
        avg_correct_sentiment = None if pd.isna(avg_correct_sentiment) else float(avg_correct_sentiment)

    wrong_positive_count = 0
    wrong_neutral_count = 0
    wrong_negative_count = 0

    if not wrong_sent_df.empty:
        wrong_positive_count = int((wrong_sent_df["sentiment_score"] > 0).sum())
        wrong_neutral_count = int((wrong_sent_df["sentiment_score"] == 0).sum())
        wrong_negative_count = int((wrong_sent_df["sentiment_score"] < 0).sum())

    dominant_wrong_sentiment_label = "暂无数据"
    max_count = max(wrong_positive_count, wrong_neutral_count, wrong_negative_count)

    if max_count > 0:
        if max_count == wrong_positive_count:
            dominant_wrong_sentiment_label = "利好倾向"
        elif max_count == wrong_negative_count:
            dominant_wrong_sentiment_label = "利空倾向"
        else:
            dominant_wrong_sentiment_label = "中性倾向"

    sentiment_gap = None
    if avg_wrong_sentiment is not None and avg_correct_sentiment is not None:
        sentiment_gap = float(avg_wrong_sentiment - avg_correct_sentiment)

    if wrong_sent_df.empty:
        summary_text = "当前没有误判样本，无法分析误判样本的标题情感特征。"
    else:
        if dominant_wrong_sentiment_label == "利好倾向":
            dominant_text = "误判样本主要集中在利好倾向标题"
        elif dominant_wrong_sentiment_label == "利空倾向":
            dominant_text = "误判样本主要集中在利空倾向标题"
        elif dominant_wrong_sentiment_label == "中性倾向":
            dominant_text = "误判样本主要集中在中性标题"
        else:
            dominant_text = "误判样本的情感分布暂不明显"

        if sentiment_gap is None:
            gap_text = "当前无法比较误判样本与正确样本的情感分差异"
        else:
            if sentiment_gap > 0:
                gap_text = "误判样本整体情感分高于正确样本"
            elif sentiment_gap < 0:
                gap_text = "误判样本整体情感分低于正确样本"
            else:
                gap_text = "误判样本与正确样本整体情感分接近"

        summary_text = f"{dominant_text}；{gap_text}。"

    return {
        "avg_wrong_sentiment": avg_wrong_sentiment,
        "avg_correct_sentiment": avg_correct_sentiment,
        "wrong_positive_count": wrong_positive_count,
        "wrong_neutral_count": wrong_neutral_count,
        "wrong_negative_count": wrong_negative_count,
        "dominant_wrong_sentiment_label": dominant_wrong_sentiment_label,
        "sentiment_gap": sentiment_gap,
        "summary_text": summary_text,
    }

def build_simulation_prediction_payload(csv_path: Path, limit: int, only_error: bool):
    if not csv_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"{csv_path.name} 不存在，请先训练对应模型"
        )

    df = pd.read_csv(csv_path)

    if df.empty:
        return {
            "total_rows": 0,
            "correct_rows": 0,
            "wrong_rows": 0,
            "error_analysis": {},
            "rows": []
        }

    if "y_true" not in df.columns or "y_pred" not in df.columns:
        raise HTTPException(
            status_code=500,
            detail=f"{csv_path.name} 缺少 y_true 或 y_pred 字段"
        )

    df["y_true"] = pd.to_numeric(df["y_true"], errors="coerce").fillna(0).astype(int)
    df["y_pred"] = pd.to_numeric(df["y_pred"], errors="coerce").fillna(0).astype(int)
    df["is_correct"] = df["y_true"] == df["y_pred"]
    df["actual_label"] = df["y_true"].map({0: "下跌", 1: "上涨"})
    df["pred_label"] = df["y_pred"].map({0: "下跌", 1: "上涨"})

    if "pred_prob_up" in df.columns:
        df["pred_prob_up"] = pd.to_numeric(df["pred_prob_up"], errors="coerce")

    total_rows = int(len(df))
    correct_rows = int(df["is_correct"].sum())
    wrong_rows = int(total_rows - correct_rows)

    wrong_df = df[df["is_correct"] == False].copy()

    false_positive_count = int(((df["y_true"] == 0) & (df["y_pred"] == 1)).sum())
    false_negative_count = int(((df["y_true"] == 1) & (df["y_pred"] == 0)).sum())

    top_error_event_types = []
    if not wrong_df.empty and "event_type_raw" in wrong_df.columns:
        event_type_stat = (
            wrong_df.groupby("event_type_raw")
            .size()
            .reset_index(name="error_count")
            .sort_values("error_count", ascending=False)
            .head(5)
        )
        top_error_event_types = event_type_stat.to_dict(orient="records")

    avg_wrong_prob_up = None
    if not wrong_df.empty and "pred_prob_up" in wrong_df.columns:
        avg_wrong_prob_up = wrong_df["pred_prob_up"].dropna().mean()
        avg_wrong_prob_up = None if pd.isna(avg_wrong_prob_up) else float(avg_wrong_prob_up)

    low_confidence_wrong_count = 0
    high_confidence_wrong_count = 0
    if not wrong_df.empty and "pred_prob_up" in wrong_df.columns:
        wrong_df["confidence"] = wrong_df["pred_prob_up"].apply(
            lambda p: max(p, 1 - p) if pd.notna(p) else None
        )
        low_confidence_wrong_count = int((wrong_df["confidence"] < 0.60).sum())
        high_confidence_wrong_count = int((wrong_df["confidence"] >= 0.60).sum())

    if wrong_rows == 0:
        summary_text = "测试集中暂无误判样本，当前模型在测试集上的分类结果全部正确。"
    else:
        if false_positive_count > false_negative_count:
            bias_text = "模型更容易把实际下跌样本误判为上涨"
        elif false_negative_count > false_positive_count:
            bias_text = "模型更容易把实际上涨样本误判为下跌"
        else:
            bias_text = "模型在两类误判上的分布较为接近"

        if low_confidence_wrong_count >= high_confidence_wrong_count:
            conf_text = "多数误判样本出现在低置信度区间"
        else:
            conf_text = "存在一定数量的高置信度误判样本，说明模型在部分样本上存在较强误判"

        if top_error_event_types:
            top_type = top_error_event_types[0]["event_type_raw"]
            top_type_count = top_error_event_types[0]["error_count"]
            type_text = f"误判最多的事件类型为 {top_type}，共 {top_type_count} 条"
        else:
            type_text = "当前无法统计误判事件类型分布"

        summary_text = f"{bias_text}；{conf_text}；{type_text}。"

    keyword_analysis = analyze_wrong_title_keywords(wrong_df)
    sentiment_analysis = analyze_title_sentiment(df, wrong_df)

    error_analysis = {
        "false_positive_count": false_positive_count,
        "false_negative_count": false_negative_count,
        "top_error_event_types": top_error_event_types,
        "avg_wrong_prob_up": avg_wrong_prob_up,
        "low_confidence_wrong_count": low_confidence_wrong_count,
        "high_confidence_wrong_count": high_confidence_wrong_count,
        "summary_text": summary_text,
        "keyword_analysis": keyword_analysis,
        "sentiment_analysis": sentiment_analysis,
    }

    display_df = df.copy()
    if only_error:
        display_df = display_df[display_df["is_correct"] == False].copy()

    display_df = display_df.tail(limit).copy().reset_index(drop=True)
    display_df = display_df.where(pd.notnull(display_df), None)

    rows = display_df.to_dict(orient="records")

    return {
        "total_rows": total_rows,
        "correct_rows": correct_rows,
        "wrong_rows": wrong_rows,
        "error_analysis": error_analysis,
        "rows": rows
    }

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

@app.get("/api/simulate/context")
def api_get_simulate_context(
    event_date: str = Query(..., description="新事件日期，如 2026-03-25")
):
    try:
        return get_simulate_context(event_date)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/predict-simulate")
def api_predict_simulate(payload: SimulatePredictRequest):
    try:
        model_version = (payload.model_version or "simulation_v1").strip().lower()

        if model_version == "simulation_v1":
            return predict_new_event_simulation(
                event_date=payload.event_date,
                event_type=payload.event_type,
                event_source=payload.event_source,
                event_title=payload.event_title,
            )

        if model_version == "hybrid_lstm":
            return predict_new_event_simulation_hybrid(
                event_date=payload.event_date,
                event_type=payload.event_type,
                event_source=payload.event_source,
                event_title=payload.event_title,
            )

        raise HTTPException(
            status_code=400,
            detail=f"不支持的 model_version: {payload.model_version}，可选值为 simulation_v1 / hybrid_lstm"
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    

@app.get("/api/simulation-model-report")
def get_simulation_model_report():
    if not SIM_REPORT_JSON.exists():
        raise HTTPException(status_code=404, detail="simulation_model_report.json 不存在，请先训练 Simulation V1 模型")
    return json.loads(SIM_REPORT_JSON.read_text(encoding="utf-8"))

@app.get("/api/sim-hybrid-report")
def get_sim_hybrid_report():
    if not SIM_HYBRID_REPORT_JSON.exists():
        raise HTTPException(
            status_code=404,
            detail="sim_hybrid_report.json 不存在，请先训练 Hybrid LSTM 模型"
        )
    return json.loads(SIM_HYBRID_REPORT_JSON.read_text(encoding="utf-8"))

@app.get("/api/simulation-test-predictions")
def get_simulation_test_predictions(
    limit: int = Query(50, ge=1, le=500),
    only_error: bool = Query(False),
):
    return build_simulation_prediction_payload(
        csv_path=SIM_TEST_PRED_CSV,
        limit=limit,
        only_error=only_error,
    )


@app.get("/api/sim-hybrid-test-predictions")
def get_sim_hybrid_test_predictions(
    limit: int = Query(50, ge=1, le=500),
    only_error: bool = Query(False),
):
    return build_simulation_prediction_payload(
        csv_path=SIM_HYBRID_TEST_PRED_CSV,
        limit=limit,
        only_error=only_error,
    )