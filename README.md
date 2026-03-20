# 事件驱动股票短期波动分析平台（全新重建版）

这是一个重新整理后的 VS Code 工程版项目，用于对贵州茅台（600519）的真实公告/新闻事件进行短期波动预测实验。

## 本版特点

- 使用真实公告/新闻事件 CSV
- 使用无信息泄露特征
- Baseline 使用时间顺序切分
- LSTM 使用 **PyTorch** 实现，避免 TensorFlow 在 Windows 原生环境下的 DLL 加载问题
- 自带 PowerShell 环境重建脚本

## 目录结构

```text
event-driven-stock-platform-fresh/
├─ .vscode/
├─ artifacts/
├─ data/
│  ├─ raw/
│  ├─ interim/
│  └─ processed/
├─ docs/
├─ notebooks/
├─ scripts/
├─ src/
├─ requirements-core.txt
├─ requirements-lstm.txt
├─ requirements.txt
└─ run_pipeline.py
```

## Windows 重建环境

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\recreate_env.ps1
```

验证环境：

```powershell
.\scripts\verify_env.ps1
```

运行全流程：

```powershell
.\scripts\run_pipeline.ps1
```

## 手动安装

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements-core.txt -i https://pypi.org/simple
python -m pip install -r requirements-lstm.txt -i https://pypi.org/simple
```

## 数据准备

至少放入：

- `data/raw/news_events_600519.csv`

如果你已有旧项目跑出来的股价缓存，直接复制：

- `data/raw/stock_price_600519.csv`

这样可以避免代理或网络问题。

## 主要脚本

- `src/data/fetch_stock_price.py`
- `src/data/build_events.py`
- `src/data/build_event_windows.py`
- `src/features/build_features.py`
- `src/features/prepare_lstm_data.py`
- `src/features/normalize_lstm_data.py`
- `src/models/train_baseline.py`
- `src/models/train_lstm.py`
- `src/visualization/compare_models.py`
