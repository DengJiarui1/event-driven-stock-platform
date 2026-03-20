# 全新重建版说明

这版重新整理了项目和环境，重点变化：

- 保留真实事件读取与清洗逻辑
- 保留无信息泄露特征设计
- Baseline 改为时间顺序切分
- LSTM 改为 **PyTorch 实现**，避免 TensorFlow 在 Windows 原生环境下的 DLL 问题
- 提供 Windows PowerShell 环境重建脚本

## 数据文件

必须准备：

- `data/raw/news_events_600519.csv`

建议准备：

- `data/raw/stock_price_600519.csv`

如果没有股价缓存，脚本会尝试用 AkShare 抓取。

## 环境

- `requirements-core.txt`：基础依赖
- `requirements-lstm.txt`：LSTM 依赖（PyTorch）
- `scripts/recreate_env.ps1`：一键重建 `.venv`
- `scripts/verify_env.ps1`：验证依赖是否成功导入

## 模型输出

- `artifacts/reports/baseline_results.csv`
- `artifacts/reports/lstm_result.csv`
- `artifacts/reports/model_comparison.csv`
- `artifacts/reports/baseline_time_split.csv`
- `artifacts/reports/lstm_time_split.csv`
