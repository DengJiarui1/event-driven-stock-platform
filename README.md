# 基于事件驱动的股票短期价格波动数据分析平台

一个面向毕业设计的股票短期波动分析与预测平台。项目结合股票历史行情数据与外部事件信息，对股价短期波动进行分析、预测、回测与可视化展示，形成“数据处理 + 模型训练 + 接口服务 + 前端展示”的完整系统。

## 项目简介

传统股价预测大多依赖历史价格、成交量和技术指标等结构化数据，但在实际市场中，新闻、公告、政策变化等事件信息往往会对股票短期波动产生明显影响。  
本项目围绕“事件驱动”思想，构建了一个集数据采集、数据处理、事件分析、模型预测、结果对比和前端可视化于一体的分析平台。

平台主要实现以下目标：

- 对股票历史行情数据进行清洗与特征构建
- 融合事件信息进行事件窗口分析
- 使用深度学习模型进行短期价格波动预测
- 对不同模型效果进行对比分析
- 通过 React 前端进行结果展示与交互

## 技术栈

### 后端
- Python
- FastAPI
- Pandas
- NumPy
- PyTorch

### 前端
- React
- Vite
- ECharts / Recharts（按你项目实际使用为准）
- Axios
- React Router

### 模型与分析
- LSTM
- Informer
- 时间序列特征工程
- 事件窗口分析
- 模型效果对比

## 项目功能

- 首页展示项目背景、研究意义与系统简介
- 事件分析页展示事件类型、事件窗口及价格变化
- 预测结果页展示模型预测值与真实值对比
- 模型对比页展示不同模型评估指标
- 后端接口读取处理后的 CSV 数据并提供给前端调用
- 支持数据处理、模型训练、预测输出与结果可视化

## 项目结构

```text
event-driven-stock-platform/
├─ data/
│  ├─ raw/                  # 原始数据
│  ├─ processed/            # 处理后数据
│  └─ results/              # 预测结果、对比结果等
├─ src/
│  ├─ data/                 # 数据采集与预处理
│  ├─ features/             # 特征工程
│  ├─ models/               # 模型定义
│  ├─ training/             # 模型训练
│  └─ evaluation/           # 评估与分析
├─ frontend/                # React 前端
│  ├─ src/
│  ├─ public/
│  └─ package.json
├─ app/
│  └─ main.py               # FastAPI 入口（按实际文件名调整）
├─ run_pipeline.py          # 数据处理主流程
├─ main_informer.py         # Informer 模型训练/预测入口
├─ requirements-core.txt
├─ requirements-lstm.txt
└─ README.md
运行环境

建议环境如下：

Python 3.10 / 3.11

Node.js 18+

npm 9+

Windows 10/11

安装与运行
1. 克隆项目
git clone https://github.com/DengJiarui1/event-driven-stock-platform.git
cd event-driven-stock-platform
2. 创建 Python 虚拟环境
python -m venv .venv

Windows 激活：

.venv\Scripts\activate
3. 安装后端依赖
pip install --upgrade pip setuptools wheel
pip install -r requirements-core.txt
pip install -r requirements-lstm.txt
4. 运行数据处理流程
python run_pipeline.py
5. 启动模型训练或预测

例如运行 Informer：

python main_informer.py --model informer --data custom --features S --target return --root_path data/processed --data_path informer_600519.csv --seq_len 60 --label_len 30 --pred_len 5 --enc_in 1 --dec_in 1 --c_out 1 --itr 1 --train_epochs 10

注：命令参数可根据实际数据文件和实验设置调整。

6. 启动 FastAPI 后端

如果你的后端入口文件是 app/main.py：

uvicorn app.main:app --reload

如果你的入口文件名不同，请按实际路径修改。

启动后默认访问：

http://127.0.0.1:8000

接口文档：

http://127.0.0.1:8000/docs
7. 启动 React 前端
cd frontend
npm install
npm run dev

启动后默认访问：

http://localhost:5173
主要数据说明

项目中涉及的数据主要包括：

股票历史行情数据

事件数据（如新闻、公告、政策类事件）

模型预测结果数据

模型对比结果数据

常见结果文件包括：

events.csv

model_comparison.csv

event_windows.csv

预测结果相关 CSV 文件

这些数据由后端接口读取后提供给前端展示。

系统页面说明
1. 首页

展示课题背景、平台简介、研究意义和系统主要功能。

2. 事件分析页

支持查看不同事件类型对股价短期变化的影响，并通过事件窗口展示事件前后价格波动情况。

3. 预测结果页

展示模型预测值与真实值对比，反映模型对短期价格波动的预测能力。

4. 模型对比页

展示 LSTM、Informer 等模型在 MAE、RMSE、方向准确率等指标上的效果差异。

项目亮点

将事件信息引入股票短期波动分析

结合时间序列建模与事件驱动思想

前后端分离，具有完整平台形态

支持预测、分析、对比与可视化展示

适合作为本科毕业设计项目

后续可扩展方向

引入更多事件文本特征，如情感分数、关键词权重

增加更多预测模型，如 Transformer、GRU、XGBoost

增加用户登录、个股切换、结果导出等功能

引入量化回测模块，提升系统完整性

增强前端交互与图表联动效果

说明

本项目为毕业设计实践项目，主要用于研究“事件驱动”方法在股票短期价格波动分析与预测中的应用价值。
部分数据与模型结果会根据实验设置、时间范围和参数不同而有所变化。

作者

邓家瑞

仓库地址
https://github.com/DengJiarui1/event-driven-stock-platform