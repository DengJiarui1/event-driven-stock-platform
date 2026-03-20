# 基于事件驱动的股票短期价格波动数据分析平台

## 项目简介

本项目围绕“股票短期价格波动预测”展开，结合股票历史行情数据与新闻/公告事件信息，构建了一个集 **数据处理、事件构建、模型训练、真实推理、前后端可视化展示** 于一体的分析平台。

项目以贵州茅台（600519）为实验对象，完成了以下核心工作：

- 股票历史数据采集与缓存
- 事件表构建与事件窗口生成
- 无信息泄露特征工程
- 传统机器学习基线模型训练与对比
- 基于 PyTorch 的事件驱动 LSTM 模型训练与真实推理
- 基于 Informer 的单变量收益率预测扩展实验
- FastAPI 后端接口封装
- React + Vite 前端可视化展示

当前系统已经实现从 **数据处理 → 模型训练 → 结果保存 → 后端服务 → 前端页面展示** 的完整闭环。

---

## 项目特色

- **事件驱动建模**：不仅使用历史价格序列，还结合公告/新闻事件构建事件窗口特征。
- **多模型对比**：支持 Logistic Regression、SVM、Random Forest、LSTM、Informer 等模型实验。
- **真实预测接口**：前端可调用后端真实 LSTM 模型，对历史事件进行推理并回看真实结果。
- **Informer 扩展实验**：工程化接入 Informer，支持任务提交、状态轮询、指标读取与预测图展示。
- **可视化平台实现**：前端已实现首页、事件分析、预测结果、模型对比、Informer 实验等页面。

---

## 主要功能

### 1. 数据处理与特征工程
- 股票历史数据获取与本地缓存
- 新闻/公告事件表构建
- 事件日期与交易日自动对齐
- 事件窗口生成
- 无信息泄露特征工程
- LSTM 输入序列构造与标准化

### 2. 模型训练与实验
- Logistic Regression 基线实验
- SVM 基线实验
- Random Forest 基线实验
- 事件驱动 LSTM 模型训练与保存
- Informer 单变量收益率预测扩展实验

### 3. 后端接口
- 事件列表接口
- 事件窗口接口
- 模型对比接口
- 最新预测接口
- 真实预测接口
- Informer 实验任务接口

### 4. 前端页面
- 首页仪表盘
- 事件分析页
- 预测结果页
- 模型对比页
- Informer 实验页

---

## 项目结构

```text
event-driven-stock-platform/
├─ Informer2020/              # Informer 官方工程与扩展实验代码
├─ artifacts/                 # 模型、报告、Informer 实验结果等产物
├─ backend/
│  └─ app/
│     ├─ main.py              # FastAPI 后端入口
│     └─ services/            # LSTM / Informer / 预测等服务逻辑
├─ data/
│  ├─ raw/                    # 原始数据
│  ├─ interim/                # 中间结果（events、event_windows 等）
│  └─ processed/              # Informer 等模型使用的处理后数据
├─ docs/                      # 文档说明
├─ frontend/                  # React + Vite 前端
│  ├─ src/
│  ├─ public/
│  └─ package.json
├─ notebooks/                 # 实验与分析记录
├─ scripts/                   # 辅助脚本
├─ src/                       # 数据处理、特征工程、模型训练主代码
├─ requirements-core.txt
├─ requirements-lstm.txt
├─ requirements.txt
├─ run_pipeline.py            # 数据处理主流程入口
└─ README.md
```

---

## 技术栈

### 后端
- Python
- FastAPI
- Pandas
- NumPy
- scikit-learn
- PyTorch
- joblib
- matplotlib

### 前端
- React
- Vite
- Axios
- ECharts
- React Router

### 模型与实验
- Logistic Regression
- SVM
- Random Forest
- LSTM（事件驱动）
- Informer（收益率单变量扩展实验）

---

## 运行环境

建议环境如下：

- Python 3.10 / 3.11
- Node.js 18+
- npm 9+
- Windows 10 / 11

---

## 安装与运行

### 1. 克隆项目

```bash
git clone https://github.com/DengJiarui1/event-driven-stock-platform.git
cd event-driven-stock-platform
```

### 2. 创建并激活 Python 虚拟环境

```bash
python -m venv .venv
```

Windows 下激活：

```powershell
.venv\Scripts\activate
```

### 3. 安装依赖

```bash
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements-core.txt
python -m pip install -r requirements-lstm.txt
```

---

## 数据处理主流程

运行：

```bash
python run_pipeline.py
```

该命令会依次完成：

- 股票数据读取/抓取
- 事件表生成
- 事件窗口生成
- 特征工程构建
- LSTM 数据准备与标准化
- Baseline 模型训练
- LSTM 模型训练
- 模型对比结果保存

常见输出文件包括：

- `data/interim/events.csv`
- `data/interim/event_windows.csv`
- `artifacts/models/lstm_scaler.pkl`
- `artifacts/models/event_lstm_torch.pt`
- `artifacts/reports/model_comparison.csv`

---

## 启动后端

当前 FastAPI 后端入口为：

```bash
python -m uvicorn backend.app.main:app --reload --port 8000
```

启动后访问：

- 后端服务地址：`http://127.0.0.1:8000`
- 接口文档：`http://127.0.0.1:8000/docs`

---

## 启动前端

```bash
cd frontend
npm install
npm run dev
```

启动后访问：

- 前端地址：`http://localhost:5173`

---

## Informer 扩展实验（可选）

Informer 官方工程位于 `Informer2020/` 目录下。  
当前项目已经支持通过前端 **Informer实验** 页面与后端接口进行工程化调用。

如需手动运行，可参考：

```bash
python Informer2020/main_informer.py \
  --model informer \
  --data custom \
  --features S \
  --target return \
  --root_path data/processed \
  --data_path informer_600519.csv \
  --seq_len 60 \
  --label_len 30 \
  --pred_len 5 \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --itr 1 \
  --train_epochs 6
```

说明：

- 当前 Informer 实验主要用于 **单变量收益率预测扩展实验**
- 其定位是 **对比实验与研究补充**，不替代事件驱动 LSTM 主流程

---

## 系统页面说明

### 1. 首页仪表盘
展示平台概览、事件统计、最优分类模型、最新预测结果及系统结论说明。

### 2. 事件分析页
支持选择事件日期，联动展示：
- 事件信息
- 事件窗口内价格走势
- 成交量变化
- 关键统计指标
- 事件窗口明细表

### 3. 预测结果页
支持选择历史事件，调用真实 LSTM 模型进行推理，并展示：
- 预测方向
- 预测概率
- 预测置信度
- 真实值回看
- 特征序列输入

### 4. 模型对比页
将模型结果分为两部分展示：

- **分类模型对比**：Logistic Regression、SVM、Random Forest、LSTM
- **时序回归模型对比**：Informer

避免将 Accuracy 与 MAE/MSE/RMSE 等不同指标混在同一图表中。

### 5. Informer 实验页
支持：
- 提交 Informer 实验任务
- 查看任务状态
- 查看误差指标
- 查看预测图
- 查看结果预览

---

## 当前实验结论

### 分类模型结论
在当前小样本事件驱动分类任务中，简单模型表现更稳定。  
从现有实验结果看，Logistic Regression 在 Accuracy 上表现较优，说明在事件样本规模有限的条件下，传统机器学习方法仍具有较强可用性。

### LSTM 结论
事件驱动 LSTM 能够利用事件窗口内的时序特征完成真实推理，并已经成功接入系统前后端。  
在当前数据规模下，其表现未明显超过最优简单模型，但其优势在于可扩展为更丰富的事件驱动特征融合框架。

### Informer 结论
Informer 在仅使用历史收益率作为单变量输入时，预测结果整体更平滑，对股票短期剧烈波动的刻画能力有限。  
这说明单纯依赖价格时序信息不足以支撑短期波动预测，引入公告/新闻等事件信息具有必要性。

---

## 推荐体验流程

建议按以下顺序体验项目：

1. 运行 `python run_pipeline.py`
2. 启动 FastAPI 后端
3. 启动 React 前端
4. 打开前端页面依次查看：
   - 首页仪表盘
   - 事件分析页
   - 预测结果页
   - 模型对比页
   - Informer 实验页

---

## 注意事项

- 当前“真实预测”主要针对历史事件库中的已有事件进行推理与回看。
- Informer 模块当前作为扩展实验模块使用。
- 如果重新创建虚拟环境，请重新安装依赖。
- 若 Informer 官方代码在新环境下运行报错，需注意旧版 NumPy 接口兼容性问题，例如 `np.Inf` 应修改为 `np.inf`。

---

## 后续可扩展方向

- 支持新事件模拟预测
- 引入事件文本情感特征
- 支持多股票切换
- 增加回测模块
- 增加实验记录持久化
- 增加结果导出与实验报告下载

---

## 项目作者

邓家瑞

本科毕业设计项目：  
**基于事件驱动的股票短期价格波动数据分析平台**