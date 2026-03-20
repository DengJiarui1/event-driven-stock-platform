import pandas as pd

from src.config import ARTIFACTS_REPORTS


def main() -> None:
    baseline_path = ARTIFACTS_REPORTS / "baseline_results.csv"
    lstm_path = ARTIFACTS_REPORTS / "lstm_result.csv"

    baseline_df = pd.read_csv(baseline_path)
    lstm_df = pd.read_csv(lstm_path)

    comparison_df = pd.concat([baseline_df, lstm_df], ignore_index=True)
    comparison_path = ARTIFACTS_REPORTS / "model_comparison.csv"
    comparison_df.to_csv(comparison_path, index=False, encoding="utf-8-sig")

    print("模型对比完成：")
    print(comparison_df)
    print(f"已保存: {comparison_path}")


if __name__ == "__main__":
    main()
