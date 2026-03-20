import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODULES = [
    "src.data.fetch_stock_price",
    "src.data.build_events",
    "src.data.build_event_windows",
    "src.features.build_features",
    "src.features.prepare_lstm_data",
    "src.features.normalize_lstm_data",
    "src.models.train_baseline",
    "src.models.train_lstm",
    "src.visualization.compare_models",
]


def main() -> None:
    for module in MODULES:
        print(f"\n{'=' * 20} 正在运行: {module} {'=' * 20}")
        result = subprocess.run([sys.executable, "-m", module], cwd=ROOT)
        if result.returncode != 0:
            raise SystemExit(f"脚本运行失败: {module}")
    print("\n全部流程运行完成。")


if __name__ == "__main__":
    main()
