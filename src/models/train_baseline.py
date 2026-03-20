import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.config import DATA_PROCESSED, ARTIFACTS_REPORTS


NUMERIC_FEATURES = [
    "pre_day_return",
    "event_day_gap",
    "event_day_return",
    "event_day_amplitude",
    "volume_change",
    "close_change_from_prev",
]

CATEGORICAL_FEATURES = ["event_type"]


def _time_split(df: pd.DataFrame, test_ratio: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df.sort_values("event_date").reset_index(drop=True)
    split_idx = max(1, int(len(df) * (1 - test_ratio)))
    split_idx = min(split_idx, len(df) - 1)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    return train_df, test_df


def _encode_features(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_num = train_df[NUMERIC_FEATURES].copy()
    test_num = test_df[NUMERIC_FEATURES].copy()

    train_cat = pd.get_dummies(train_df[CATEGORICAL_FEATURES].astype(str), prefix=CATEGORICAL_FEATURES)
    test_cat = pd.get_dummies(test_df[CATEGORICAL_FEATURES].astype(str), prefix=CATEGORICAL_FEATURES)
    test_cat = test_cat.reindex(columns=train_cat.columns, fill_value=0)

    train_X = pd.concat([train_num, train_cat], axis=1)
    test_X = pd.concat([test_num, test_cat], axis=1)
    return train_X, test_X


def main() -> None:
    ARTIFACTS_REPORTS.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PROCESSED / "dataset.csv", parse_dates=["event_date"])
    train_df, test_df = _time_split(df, test_ratio=0.2)

    X_train_raw, X_test_raw = _encode_features(train_df, test_df)
    y_train = train_df["label"]
    y_test = test_df["label"]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_test_scaled = scaler.transform(X_test_raw)

    results = {}

    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train_scaled, y_train)
    results["Logistic Regression"] = accuracy_score(y_test, lr.predict(X_test_scaled))

    svm = SVC(kernel="rbf")
    svm.fit(X_train_scaled, y_train)
    results["SVM"] = accuracy_score(y_test, svm.predict(X_test_scaled))

    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf.fit(X_train_raw, y_train)
    rf_preds = rf.predict(X_test_raw)
    results["Random Forest"] = accuracy_score(y_test, rf_preds)

    result_df = pd.DataFrame({"Model": list(results.keys()), "Accuracy": list(results.values())})
    result_df.to_csv(ARTIFACTS_REPORTS / "baseline_results.csv", index=False, encoding="utf-8-sig")

    split_info = pd.DataFrame(
        {
            "split": ["train", "test"],
            "rows": [len(train_df), len(test_df)],
            "start_date": [train_df["event_date"].min(), test_df["event_date"].min()],
            "end_date": [train_df["event_date"].max(), test_df["event_date"].max()],
        }
    )
    split_info.to_csv(ARTIFACTS_REPORTS / "baseline_time_split.csv", index=False, encoding="utf-8-sig")

    print("Baseline 模型结果（无信息泄露版）:")
    print(result_df)
    print("\n时间划分:")
    print(split_info)
    print("\nRandom Forest 分类报告：")
    print(classification_report(y_test, rf_preds))


if __name__ == "__main__":
    main()
