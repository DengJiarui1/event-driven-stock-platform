from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import StandardScaler


ROOT_DIR = Path(__file__).resolve().parents[2]

DATASET_CSV = ROOT_DIR / "artifacts" / "datasets" / "simulation_train_data.csv"
MODEL_DIR = ROOT_DIR / "artifacts" / "models"
REPORT_DIR = ROOT_DIR / "artifacts" / "reports"

SIM_MODEL_PATH = MODEL_DIR / "sim_event_logreg.pkl"
SIM_SCALER_PATH = MODEL_DIR / "sim_event_scaler.pkl"
SIM_META_PATH = MODEL_DIR / "sim_event_meta.json"
SIM_REPORT_PATH = REPORT_DIR / "simulation_model_report.json"
SIM_PRED_DETAIL_PATH = REPORT_DIR / "simulation_test_predictions.csv"

FEATURE_COLUMNS = [
    "t_minus_2_intraday_return",
    "t_minus_2_amplitude",
    "t_minus_2_volume_change_vs_prev",
    "t_minus_2_close_change_vs_prev_close",
    "t_minus_1_intraday_return",
    "t_minus_1_amplitude",
    "t_minus_1_volume_change_vs_prev",
    "t_minus_1_close_change_vs_prev_close",
    "event_type",
    "title_length",
    "positive_keyword_count",
    "negative_keyword_count",
    "sentiment_score",
    "is_report",
    "is_dividend",
    "is_governance",
    "is_manual_source",
]


def ensure_dirs():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    ensure_dirs()

    if not DATASET_CSV.exists():
        raise FileNotFoundError(
            f"未找到模拟训练数据: {DATASET_CSV}，请先运行 build_simulation_features.py"
        )

    df = pd.read_csv(DATASET_CSV)
    if df.empty:
        raise ValueError("simulation_train_data.csv 为空")

    df = df.sort_values("event_date").reset_index(drop=True)

    X = df[FEATURE_COLUMNS].copy()
    y = df["label"].astype(int)

    # 时间顺序切分，避免未来信息进入训练集
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(max_iter=2000)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test_scaled)[:, 1]
    else:
        y_prob = None

    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1]).tolist()
    report_text = classification_report(y_test, y_pred, zero_division=0)

    print("Simulation V1 Model Result")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-score : {f1:.4f}")
    print("Confusion Matrix:")
    print(cm)
    print(report_text)

    # 保存模型与 scaler
    joblib.dump(model, SIM_MODEL_PATH)
    joblib.dump(scaler, SIM_SCALER_PATH)

    meta = {
        "model_name": "Simulation V1 (Pre-event Logistic Regression)",
        "feature_columns": FEATURE_COLUMNS,
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "split_method": "time_order_80_20",
        "metrics": {
            "accuracy": float(acc),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        },
    }
    SIM_META_PATH.write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    report_json = {
        "model_name": "Simulation V1 (Pre-event Logistic Regression)",
        "task_type": "pre_event_simulation_classification",
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "metrics": {
            "accuracy": float(acc),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        },
        "confusion_matrix": cm,
        "class_labels": {
            "negative_class": 0,
            "positive_class": 1,
            "negative_label_text": "下跌",
            "positive_label_text": "上涨",
            "matrix_layout": "[[TN, FP], [FN, TP]]"
        },
        "classification_report_text": report_text,
        "feature_columns": FEATURE_COLUMNS,
    }
    SIM_REPORT_PATH.write_text(
        json.dumps(report_json, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    pred_detail = df.iloc[split_idx:].copy().reset_index(drop=True)
    pred_detail["y_true"] = y_test.reset_index(drop=True)
    pred_detail["y_pred"] = y_pred
    if y_prob is not None:
      pred_detail["pred_prob_up"] = y_prob
    pred_detail.to_csv(SIM_PRED_DETAIL_PATH, index=False, encoding="utf-8-sig")

    print(f"已保存模型: {SIM_MODEL_PATH}")
    print(f"已保存 scaler: {SIM_SCALER_PATH}")
    print(f"已保存 meta: {SIM_META_PATH}")
    print(f"已保存评估报告: {SIM_REPORT_PATH}")
    print(f"已保存测试预测明细: {SIM_PRED_DETAIL_PATH}")


if __name__ == "__main__":
    main()