<!--
  Comprehensive project report and user guide for the Bitcoin Investment
  Decision System. Written in simplified Chinese as requested.
-->

# 比特币多智能体决策系统 – 项目报告与使用说明

## 1. 执行摘要

本项目构建了一个端到端的比特币投资决策平台，结合 **传统量化模型 + 多智能体分析 + 自动化工作流 (N8N) + 桌面级 GUI**。核心价值：

1. **三模型集成预测**：线性回归（Elastic Net）、ARIMA、季节性模型（Prophet-like）+ 加权集成，形成稳健的趋势判断。
2. **多智能体研究委员会**：5 位 AI 分析师（Technical / Industry / Financial / Market / Risk）各自独立分析，随后进行匿名互评与投票，缓解群体思维偏差。
3. **验证与挑战机制**：Devil’s Advocate、Red-Blue Team、Bayesian Signal Fusion、Confidence Calibration，保证结论经过多轮“挑战—防守”与概率校准。
4. **风险预算管理**：结合持仓比例、止盈止损、最大亏损预算，输出完整执行方案。
5. **自动化与可视化**：N8N 工作流统筹全部 API；`setup_and_run.py` 一键执行；`gui_investment_system.py` 提供现代化桌面界面并可加载最新工作流快照。

---

## 2. 系统架构

```
┌──────────────────────────────────────────────────────────────┐
│ 数据/模型层                                                  │
│ - Bitcoin price prediction Project                           │
│ - core_prediction_models.py（Linear/ARIMA/Seasonal）          │
│ - advanced_features.py（Multi-timeframe, Scenario, etc.）     │
└──────────────────────────────────────────────────────────────┘
                 │
                 ▼
┌──────────────────────────────────────────────────────────────┐
│ 服务层                                                       │
│ n8n_workflows/n8n_api_server.py                              │
│ 18 个 REST API：数据获取、指标计算、分析师、匿名互评、        │
│ Devil's Advocate、Red-Blue、Bayesian Fusion、Risk Budget 等  │
└──────────────────────────────────────────────────────────────┘
                 │
      ┌──────────┴──────────┐
      ▼                     ▼
┌───────────────┐   ┌─────────────────────────────────────────┐
│ 自动化工作流   │   │ 桌面 GUI                                │
│ N8N + workflow │   │ gui_investment_system.py               │
│ setup_and_run  │   │ - 16 步流程进度                        │
│                │   │ - 分析师卡片/多面板输出                │
└───────────────┘   │ - “加载最新结果”按钮（读取 workflow   │
                    │   快照）                                │
                    └─────────────────────────────────────────┘
```

---

## 3. 组件概览

### 3.1 预测模型
- **LinearRegressionModel (Elastic Net)**：82.12% 样本内准确率，权重 0.50。
- **ARIMAModel (2,1,2)**：序列趋势/季节捕捉，权重 0.30。
- **SeasonalModel (Prophet-like)**：节日/周期特征，权重 0.20。
- **EnsemblePredictorForAnalysts**：按权重合成，产出当前价格、预测价格、变动百分比与置信度。

### 3.2 分析与验证模块
- `advanced_features.py`：Multi-Timeframe、Scenario Analysis、Bayesian Fusion、Red-Blue、Confidence Calibration、Risk Budget 等。
- `anonymous_peer_review.py`：匿名化分析、互评维度（accuracy/insight/logic/risk/actionability）、质量排名。
- `investment_committee.py`：多智能体工作流骨架，管理分析师记忆、辩论、投票、决策归档。

### 3.3 API & 工作流
- `n8n_api_server.py`：18 个端点（见下节）。
- `setup_and_run.py`：交互式菜单与命令行参数 (`--server`, `--workflow`, `--test` 等)，并生成 `n8n_workflows/latest_workflow_result.json`。
- `n8n_workflows/*.json`：基础版 + 增强版，27 节点覆盖 6 大阶段。

### 3.4 GUI
- `gui_investment_system.py`：现代暗色主题，16 步进流程 + 四大输出面板。
- “⬇ Load Latest Result” 根据 `latest_workflow_result.json` 将 API 最新结果映射到 GUI（无需重新跑全部算法）。

---

## 4. 关键 API（n8n_api_server.py）

| 类别 | 端点 | 功能 |
| --- | --- | --- |
| 健康/基础 | `GET /api/health` | 状态与模型载入情况 |
| 数据 & 指标 | `POST /api/fetch-price`、`/calculate-indicators` | 比特币价数据 & 技术指标 |
| 预测模型 | `/linear-regression-predict` `/arima-predict` `/seasonal-predict` | 三个基础模型输出 |
| 高阶分析 | `/multi-timeframe-analysis` `/scenario-analysis` | 多周期 & 场景模拟 |
| 分析师/共识 | `/analyst/<type>` `/anonymize-analyses` `/peer-review` `/calculate-rankings` | 5 位分析师 -> 匿名互评 -> 排名 |
| 验证/挑战 | `/devils-advocate` `/red-blue-team` `/bayesian-fusion` `/confidence-calibration` | Devil、Red-Blue、贝叶斯融合、置信校准 |
| 决策输出 | `/chairman-synthesis` `/risk-budget-calculation` `/save-decision` | 主席合成（含完整报告）、仓位预算、最终存档 |

> **富文本输出**：`chairman-synthesis` / `red-blue-team` / `risk-budget-calculation` 均附带报告字段，格式与用户示例一致，可直接呈现在 GUI / Markdown / PDF 中。

---

## 5. 数据流与执行流程

1. **数据阶段**：`fetch-price` → `calculate-indicators` （历史价序列 + RSI/MACD/Bollinger/SMA/EMA）。
2. **预测阶段**：三个模型并行 → `Ensemble` 合成趋势、涨跌概率。
3. **高阶分析**：`multi-timeframe`（短/中/长）、`scenario-analysis`（牛/熊/震荡/黑天鹅）。
4. **多智能体**：5 位分析师独立分析 → 匿名化 → 互评/排名 → 投票（BUY/HOLD/SELL）。
5. **验证校准**：Devil’s Advocate → Red-Blue 对抗 → Bayesian Fusion → Confidence Calibration。
6. **最终决策**：`chairman_synthesis` 汇总全部输入并输出包含“FINAL DECISION & DECISION REVIEW”的详细文本；`risk-budget` 给出仓位、止盈止损、风险收益比；`save-decision` 写入 `investment_history/decision_*.json`，供版本追踪与 GUI 加载。

---

## 6. 环境与部署

### 6.1 依赖安装

```powershell
python -m venv .venv
. .venv/Scripts/Activate.ps1
pip install -r requirements.txt
pip install -r requirements_gui.txt          # 若需 GUI
```

> 若缺乏 `pdflatex`，请使用 MiKTeX 或 Overleaf 编译 LaTeX 报告。

### 6.2 API 服务器

```powershell
python n8n_workflows/run_n8n_server.py
# 或用 setup_and_run 菜单的「3. Start API Server」
```

### 6.3 一体化启动（推荐）

```powershell
python n8n_workflows/setup_and_run.py
# 菜单：
# 1 安装指南  2 导入指南  3 启动 API  4 测试 API
# 5 完整工作流  6 退出
```

- `--workflow`：直接执行完整分析（含自动快照）。
- `--server`：仅启动 API。
- `--test`：批量调用全部端点验证。

### 6.4 N8N 工作流

1. 安装 Node.js + `npm install -g n8n` 或使用 Docker。
2. 导入 `n8n_workflows/bitcoin_investment_workflow_enhanced.json`。
3. HTTP 节点指向 `http://localhost:5000`（如同机运行无需修改）。
4. 可设置 Webhook/定时触发器，实现无人值守执行。

### 6.5 GUI

```powershell
python gui_investment_system.py
```

流程：
1. 点击 `▶ Start Analysis` 可运行内置模拟流程。
2. 若想展示真实工作流结果，先运行 `setup_and_run.py --workflow`，然后在 GUI 中点击 **“⬇ Load Latest Result”**，即可加载 `n8n_workflows/latest_workflow_result.json` 所描述的全过程输出。

---

## 7. 使用场景

| 角色 | 场景 | 建议操作 |
| --- | --- | --- |
| 研究员 | 需要快速回看完整工作流结果 | `setup_and_run.py --workflow` → GUI 加载快照 |
| 开发/运维 | 维护 API & 工作流 | 运行 `run_n8n_server.py`，在 Postman / N8N 中调试各端点 |
| 投资委员会 | 会议展示 | 启动 GUI 显示 16 步进度 + 富文本报告；或者导出 `latest_workflow_result.json` 生成汇报 PPT |
| 自动化 | 计划定时生成建议 | 使用 N8N “Schedule Trigger” 每 X 小时执行一次，并把 `save-decision` 结果推送 Slack / 邮件 |

---

## 8. 常见问题

1. **PowerShell 不识别 `&&`**：请使用分号 `;` 或逐条执行命令。
2. **`pdflatex` 未安装**：请安装 MiKTeX（Windows）或使用 Overleaf。
3. **API 连接失败**：先运行 `setup_and_run.py --server` 或菜单 3，确保 5000 端口可用。
4. **GUI 中文乱码**：Windows 终端请执行 `chcp 65001`；脚本已包含 ANSI/UTF-8 处理。
5. **N8N 节点 401/404**：检查 HTTP 节点 URL、API server 是否启动、同网络是否可访问。

---

## 9. 进一步扩展

- **更多数据源**：可在 `fetch-price` 中接入 CoinGecko、Binance 等实时 API。
- **模型升级**：加入 LSTM/Transformer、链上指标；或对现有模型做在线学习。
- **自动交易执行**：将 `risk_budget` 输出对接交易所 API，实现自动下单与风控。
- **团队协作**：利用 `investment_history/*.json` 与 `workflow_runs/*.json` 做回溯分析、绩效考核。

---

## 10. 附录：核心路径总览

| 路径 | 说明 |
| --- | --- |
| `gui_investment_system.py` | 桌面 GUI |
| `n8n_workflows/setup_and_run.py` | 主启动器，含交互菜单 |
| `n8n_workflows/run_n8n_server.py` | API 服务器启动脚本 |
| `n8n_workflows/n8n_api_server.py` | 18 个 REST API |
| `n8n_workflows/latest_workflow_result.json` | 最近一次工作流快照（GUI 可加载） |
| `n8n_workflows/workflow_runs/` | 历史快照归档 |
| `investment_history/decision_*.json` | 决策执行档案 |
| `README.md`（根目录） | 项目简介、快速入门 |
| `n8n_workflows/README.md` | N8N 专用说明 |

---

这份报告可作为团队内部交接文档、客户演示资料或二次开发参考。若需生成 PDF，可用 Markdown 转换工具（如 Typora、Pandoc）或直接纳入 LaTeX 模板。祝使用顺利！ 🙌

