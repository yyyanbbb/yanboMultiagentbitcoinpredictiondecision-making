# Intelligent Investment Decision System v3.1

A multi-agent Bitcoin investment analysis system with advanced decision-making capabilities and **Anonymous Peer Review** innovation.

> 📘 **完整项目报告 & 使用说明**  
> 请参阅 [`docs/project_report.md`](docs/project_report.md)，包含系统架构、API、工作流、GUI、部署与常见问题的详尽说明。

## ⭐ Core Feature: Real Three-Model Prediction System

Analysts can now **actually use** three prediction algorithms for analysis, not just simulations:

| Model | Algorithm | Direction Accuracy | Weight | Use Case |
|-------|-----------|-------------------|--------|----------|
| **Linear Regression** | Elastic Net Regularization | **82.12%** | 50% | Short-term price prediction |
| **ARIMA(2,1,2)** | AutoRegressive Integrated Moving Average | 54.20% | 30% | Trend analysis |
| **Seasonal** | Prophet-like Seasonality | 53.47% | 20% | Cyclical analysis |

## System Architecture

```
Investment Committee
├── Technical Analyst - Uses three-model ensemble prediction (Linear Regression, ARIMA, Seasonal)
│   ├── LinearRegressionModel - Real trained Elastic Net model
│   ├── ARIMAModel - Real trained ARIMA(2,1,2) model
│   └── SeasonalModel - Real trained seasonal model
├── Industry Analyst - Analyzes industry trends and policy environment
├── Financial Analyst - Evaluates risk-return ratio and position recommendations
├── Market Expert - Analyzes market sentiment and fund flows
├── Risk Analyst - Identifies and quantifies various risks
└── Investment Manager - Makes final decisions based on all opinions
```

## Features

### Core Features

| Feature | Description |
|---------|-------------|
| Analyst Personalization | Each analyst has unique style, nickname, and catchphrase |
| Adversarial Debate | Analysts debate on differing viewpoints |
| Decision Review | Automatic analysis of decision quality and risk points |
| Historical Tracking | Track and validate historical decision accuracy |

### Advanced Features (v3.0)

| Feature | Description | File |
|---------|-------------|------|
| **Multi-Timeframe Analysis** | Short(1-7 days)/Medium(1-4 weeks)/Long(1-3 months) comprehensive analysis | `advanced_features.py` |
| **Scenario Analysis Engine** | Bull/Bear/Sideways/Black Swan scenario simulation with expected returns | `advanced_features.py` |
| **Bayesian Signal Fusion** | Intelligent weighted fusion of prediction models, technical indicators, analyst consensus | `advanced_features.py` |
| **Red-Blue Team Validation** | Red team challenges, Blue team defends, Judge gives final score | `advanced_features.py` |
| **Risk Budget Management** | Position sizing based on maximum acceptable risk (VaR) | `advanced_features.py` |
| **Confidence Calibration** | Calibrate AI model overconfidence, output calibrated confidence interval | `advanced_features.py` |

### 🆕 Anonymous Peer Review (v3.1 Innovation)

| Feature | Description | File |
|---------|-------------|------|
| **Anonymous Peer Review** | Anonymize all analyses (A/B/C/D/E), each analyst rates others on Accuracy + Insight | `anonymous_peer_review.py` |
| **Quality-Weighted Ranking** | Rank analyses by peer review scores, top-ranked opinions carry more weight | `anonymous_peer_review.py` |
| **Devil's Advocate** | Dedicated challenger forcefully argues against consensus to expose blind spots | `anonymous_peer_review.py` |
| **Dynamic Weight Adjustment** | Analysts who historically predict well gain more influence over time | `anonymous_peer_review.py` |
| **Chairman Synthesis** | Final synthesis integrating quality rankings + Devil's Advocate + all reviews | `anonymous_peer_review.py` |

**Why Anonymous Peer Review?**
- 🎭 **Reduces Authority Bias**: Senior analysts don't have undue influence
- 🧠 **Prevents Groupthink**: Forced critical evaluation of peers
- ⚖️ **Objective Quality Ranking**: Scores based on merit, not reputation
- 👹 **Devil's Advocate**: Ensures contrarian views are always considered

## Decision Process

```
v3.1 Complete Process (13 phases):

┌─────────────────────────────────────────────────────────────────────┐
│ 1. Data Preparation    → Load price data, run three-model ensemble │
│ 2. Advanced Analysis   → Multi-timeframe + Scenario + Calibration  │
│ 3. Independent Analysis→ 5 analysts conduct independent analysis   │
│ 4. Debate Phase        → Analysts debate on differing viewpoints   │
│ 5. Discussion Phase    → Analysts exchange views, influence each   │
│ 6. Signal Fusion       → Bayesian weighted fusion of signals       │
│ 7. Voting              → Each analyst gives recommendation         │
├─────────────────────────────────────────────────────────────────────┤
│ 8. ★ ANONYMOUS PEER REVIEW ★ (v3.1 NEW!)                           │
│    ├── Anonymize all opinions (A/B/C/D/E)                          │
│    ├── Each analyst rates others on Accuracy + Insight             │
│    ├── Calculate quality-weighted rankings                          │
│    └── Devil's Advocate challenges consensus                        │
├─────────────────────────────────────────────────────────────────────┤
│ 9. Final Decision      → Quality-weighted comprehensive decision   │
│ 10. Red-Blue Validation→ Adversarial decision validation           │
│ 11. Risk Budget        → Calculate position based on risk budget   │
│ 12. Save Records       → Save decisions and analyst experience     │
│ 13. Decision Review    → Analyze decision quality and risk points  │
└─────────────────────────────────────────────────────────────────────┘
```

## Installation

```bash
pip install -r requirements.txt
```

Main dependencies:
- `numpy`, `pandas` - Data processing
- `python-dotenv` - Environment variables
- `colorama` - Terminal styling
- `openai` - API calls
- `scipy` - Scientific computing

## API Configuration

1. Copy the environment variable example file:
```bash
cp .env.example .env
```

2. Edit `.env` file, enter your DeepSeek API Key:
```
DEEPSEEK_API_KEY=your_api_key_here
```

## Usage

### 🖥️ GUI Mode (Recommended - No Browser Required)

Launch the beautiful desktop GUI interface:

```bash
# Windows - Double-click run_gui.bat or:
python gui_investment_system.py

# Or using PowerShell:
.\run_gui.ps1
```

**GUI Features:**
- 🎨 Modern dark theme interface
- 📊 Real-time progress tracking for all 16 analysis steps
- 👥 Live analyst recommendation cards
- 📝 Tabbed output panels (Main Output, Analysis Details, Peer Review, Final Decision)
- 🔄 Step-by-step visualization of the entire analysis pipeline
- ⚡ No browser required - runs directly as a desktop application

### Interactive Mode (Terminal - Full Features)
```bash
python investment_committee.py
```

### Auto Mode (Terminal - Full Features)
```bash
python investment_committee.py --auto
```

### Fast Mode (Terminal - Skip Debate and Advanced Analysis)
```bash
python investment_committee.py --auto --fast
```

### Custom Mode
```bash
# Disable debate phase
python investment_committee.py --no-debate

# Disable review analysis
python investment_committee.py --no-analysis

# Disable advanced analysis features
python investment_committee.py --no-advanced

# Combined usage
python investment_committee.py --auto --no-debate --no-advanced
```

### Test Advanced Features Module
```bash
python advanced_features.py
```

### 🔄 N8N Workflow Integration (v3.2 NEW!)

Automate the entire investment decision workflow:

```bash
# Option 1: Standalone Mode (No N8N needed - Easiest!)
python n8n_workflows/setup_and_run.py --standalone

# Option 2: Interactive Menu
python n8n_workflows/setup_and_run.py

# Option 3: With N8N (Full automation)
python n8n_workflows/setup_and_run.py --full
```

**Workflow Features:**
- 🚀 **Standalone Mode**: Run complete analysis without N8N installation
- 🔄 23-node N8N workflow for full automation
- ⏰ Scheduled execution (every 6 hours)
- 📊 Full 10-phase multi-agent analysis pipeline
- 🔔 Slack notifications for buy/sell signals
- 💾 Automatic decision history storage

See [n8n_workflows/README.md](n8n_workflows/README.md) for detailed documentation.

## File Structure

```
bitcoinprediction_decision-making/
├── gui_investment_system.py     # GUI Application (v3.1 NEW!) ★★★
├── run_gui.bat                  # Windows batch launcher
├── run_gui.ps1                  # PowerShell launcher
├── investment_committee.py      # Main program (v3.1 enhanced)
├── core_prediction_models.py    # Core prediction models (v3.0) ★
│   ├── LinearRegressionModel    # Linear regression model (Elastic Net)
│   ├── ARIMAModel               # ARIMA time series model
│   ├── SeasonalModel            # Seasonal model (Prophet-like)
│   └── EnsemblePredictorForAnalysts  # Ensemble predictor
├── anonymous_peer_review.py     # Anonymous Peer Review system (v3.1 new) ★★
│   ├── AnonymousPeerReviewSystem# Main peer review orchestrator
│   ├── AnonymousAnalysis        # Anonymized analysis container
│   ├── PeerReview               # Peer review data structure
│   ├── DevilsAdvocateChallenge  # Devil's Advocate challenge
│   ├── RankedAnalysis           # Quality-ranked analysis
│   └── DynamicWeightAdjuster    # Historical accuracy weight adjuster
├── advanced_features.py         # Advanced features module (v3.0)
│   ├── MultiTimeframeAnalyzer   # Multi-timeframe analyzer
│   ├── ScenarioAnalysisEngine   # Scenario analysis engine
│   ├── BayesianSignalFusion     # Bayesian signal fusion
│   ├── RedBlueTeamValidator     # Red-Blue team validator
│   ├── RiskBudgetManager        # Risk budget manager
│   └── ConfidenceCalibrator     # Confidence calibrator
├── n8n_workflows/               # N8N Workflow Integration (v3.2 NEW!) ★★★
│   ├── setup_and_run.py         # ★★★ Main launcher (standalone + N8N)
│   ├── run_n8n_server.py        # API server launcher
│   ├── bitcoin_investment_workflow.json  # N8N workflow (23 nodes)
│   ├── n8n_api_server.py        # Flask API server
│   ├── README.md                # N8N workflow documentation
│   └── workflow_diagram.md      # Visual workflow diagram
├── bitcoin_predictor_ensemble.py # Bitcoin predictor ensemble
├── decision_analyzer.py         # Decision analyzer
├── enhanced_analysts.py         # Enhanced analysts
├── output_formatter.py          # Output formatter
├── requirements.txt             # Dependencies
├── .env                         # Environment variables
└── Bitcoin price prediction Project/  # Original prediction model data
    ├── data/                    # Preprocessed training data
    ├── models.py
    ├── preprocess.py
    └── ...
```

## Three-Model Prediction System Details

### 1. Linear Regression (Elastic Net)

Most accurate prediction model with **82.12%** direction accuracy.

```python
# Model Configuration
LinearRegressionModel(
    learning_rate=0.04,
    max_iterations=800,
    batch_size=32,
    reg_type='elastic_net',  # L1+L2 regularization
    reg_lambda=0.05,
    l1_ratio=0.3,
    learning_rate_decay=0.012
)
```

Features:
- Elastic Net regularization prevents overfitting
- Mini-batch gradient descent training
- Learning rate decay strategy
- Feature importance analysis output

### 2. ARIMA(2,1,2)

Classic time series prediction model.

```python
# Model Configuration
ARIMAModel(p=2, d=1, q=2)
# p=2: Autoregressive order
# d=1: Differencing order (makes data stationary)
# q=2: Moving average order
```

Features:
- AR parameters estimated using Yule-Walker equations
- Automatic differencing for non-stationary data
- 7-day forecast path output

### 3. Seasonal Model (Prophet-like)

Captures cyclical patterns in prices.

```python
# Model Configuration
SeasonalModel(
    yearly_seasonality=True,  # Yearly seasonality
    weekly_seasonality=True   # Weekly seasonality
)
```

Features:
- Decomposes trend, seasonality, residuals
- Analyzes day-of-week effects (Monday to Sunday)
- Smoothed seasonal patterns

## v3.0 Highlights

### 1. Multi-Timeframe Analysis
Investment requires multiple time dimensions:
- **Short-term (1-7 days)**: Technical analysis driven, focus on overbought/oversold, breakout signals
- **Medium-term (1-4 weeks)**: Sentiment driven, focus on market sentiment, institutional movements
- **Long-term (1-3 months)**: Fundamentals driven, focus on adoption rate, policy changes

### 2. Scenario Analysis
Consider different market scenarios with probabilities:
- Bull market scenario (30%) - Expected surge
- Bear market scenario (20%) - Expected decline
- Sideways scenario (40%) - Range-bound
- Black Swan (10%) - Extreme decline

### 3. Bayesian Signal Fusion
Intelligent fusion of multiple signal sources:
- Dynamically adjust weights based on historical accuracy
- Consider correlations between signals
- Output probabilistic predictions

### 4. Red-Blue Team Validation
Adversarial thinking for decision validation:
- **Red Team**: Challenges decision weaknesses
- **Blue Team**: Defends decision validity
- **Judge**: Comprehensive evaluation with final score

### 5. Risk Budget Management
Position sizing based on risk budget:
- Set maximum acceptable risk exposure
- Dynamically adjust position based on stop-loss distance
- VaR (Value at Risk) control

### 6. Confidence Calibration
Address AI model overconfidence:
- Analyze historical prediction calibration curves
- Adjust current prediction confidence
- Output calibrated confidence intervals

### 7. Anonymous Peer Review (v3.1 Innovation)

**The Problem:** Traditional multi-agent systems suffer from:
- **Authority Bias**: Junior agents defer to senior agents
- **Groupthink**: Agents converge on consensus too quickly
- **Anchoring**: First speaker unduly influences others

**The Solution:** Academic-style blind peer review:

```
┌────────────────────────────────────────────────────────────────┐
│         ANONYMOUS PEER REVIEW WORKFLOW                         │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  Step 1: ANONYMIZATION                                         │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐     │
│  │ Technical    │    │ Financial    │    │ Risk         │     │
│  │ Analyst      │ => │ Analyst      │ => │ Analyst      │     │
│  │ (hidden)     │    │ (hidden)     │    │ (hidden)     │     │
│  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘     │
│         │                   │                   │              │
│         ▼                   ▼                   ▼              │
│     Analyst A           Analyst B           Analyst C         │
│                                                                │
│  Step 2: CROSS-EVALUATION (Accuracy + Insight + Logic)         │
│                                                                │
│     A reviews B,C    B reviews A,C    C reviews A,B           │
│         │                 │                │                   │
│         └─────────────────┼────────────────┘                   │
│                           ▼                                    │
│  Step 3: QUALITY RANKING                                       │
│  ┌────────────────────────────────────────┐                    │
│  │ #1. Analyst B - Score: 8.2/10 ⭐        │                    │
│  │ #2. Analyst A - Score: 7.5/10          │                    │
│  │ #3. Analyst C - Score: 6.8/10          │                    │
│  └────────────────────────────────────────┘                    │
│                                                                │
│  Step 4: DEVIL'S ADVOCATE                                      │
│  ┌────────────────────────────────────────┐                    │
│  │ 😈 "But what if the consensus is      │                    │
│  │    completely WRONG? Consider:         │                    │
│  │    - Black swan risks                  │                    │
│  │    - Contrary indicators               │                    │
│  │    - Historical failures               │                    │
│  └────────────────────────────────────────┘                    │
│                                                                │
│  Step 5: CHAIRMAN SYNTHESIS                                    │
│  ┌────────────────────────────────────────┐                    │
│  │ Final decision weighted by:            │                    │
│  │ • Quality rankings (merit-based)       │                    │
│  │ • Devil's Advocate warnings            │                    │
│  │ • Cross-validation insights            │                    │
│  └────────────────────────────────────────┘                    │
└────────────────────────────────────────────────────────────────┘
```

**Evaluation Dimensions:**
| Dimension | Weight | Description |
|-----------|--------|-------------|
| Accuracy | 30% | How correct is the analysis based on data? |
| Insight | 25% | How novel and valuable are the perspectives? |
| Logic | 20% | How coherent is the reasoning chain? |
| Risk Awareness | 15% | Are potential downsides properly identified? |
| Actionability | 10% | Can the advice be practically implemented? |

**Dynamic Weight Adjustment:**
- Analysts who consistently predict well gain more influence
- Historical accuracy tracked and validated
- Weights updated after each market outcome is known

## Output Example

```
================================================================================
           Multi-Timeframe Analysis Report
================================================================================

[Short-term Analysis] (1-7 days)
  Focus: Technical
  Predicted Change: +2.60%
  Direction: Bullish
  Signal: Light position buy

[Medium-term Analysis] (1-4 weeks)
  Predicted Change: +5.20%
  Direction: Bullish
  Signal: Buy

[Long-term Analysis] (1-3 months)
  Predicted Change: +7.80%
  Direction: Bullish
  Signal: Buy

================================================================================
                 Comprehensive Assessment
================================================================================
  Weighted Predicted Change: +5.20%
  Timeframe Consistency: Highly consistent
  Final Signal: Buy
  Recommendation: All three timeframes bullish, recommend active buying, expected return 5.2%
```

## Architecture Comparison

| Feature | Traditional Multi-Agent | Investment Decision System v3.1 |
|---------|-------------------------|--------------------------------|
| Decision Dimension | Single round voting | Multi-timeframe |
| Scenario Consideration | Binary outcome | Four probability scenarios |
| Signal Fusion | Simple majority voting | Bayesian weighted fusion |
| Validation Mechanism | None | Red-Blue team validation |
| Risk Management | None | Risk budget system |
| Confidence | Subjective judgment | Scientific calibration |
| **Bias Mitigation** | None | **Anonymous Peer Review** |
| **Quality Weighting** | Equal weight | **Merit-based ranking** |
| **Contrarian Views** | Ignored | **Devil's Advocate** |
| **Historical Learning** | Static | **Dynamic weight adjustment** |

## 🔄 N8N Workflow Architecture

The system now supports full automation via N8N workflow:

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    N8N WORKFLOW (23 Nodes, 6 Phases)                            │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  PHASE 1: DATA COLLECTION                                                       │
│  [Webhook/Schedule] → [Fetch Price] → [Calculate Indicators]                    │
│                                                                                 │
│  PHASE 2: MODEL PREDICTIONS (Parallel)                                          │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │ [Linear Regression 82%] ──┐                                              │   │
│  │ [ARIMA 54%] ──────────────┼─→ [Merge] → [Ensemble Calculation]          │   │
│  │ [Seasonal 53%] ───────────┘                                              │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
│  PHASE 3: ADVANCED ANALYSIS                                                     │
│  [Multi-Timeframe] → [Scenario Analysis]                                        │
│                                                                                 │
│  PHASE 4: MULTI-AGENT ANALYSIS (Parallel)                                       │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │ [Technical] [Industry] [Financial] [Market] [Risk]                       │   │
│  │      │          │          │          │        │                         │   │
│  │      └──────────┴──────────┼──────────┴────────┘                         │   │
│  │                            ▼                                              │   │
│  │ [Anonymize] → [Peer Review] → [Rankings]                                 │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
│  PHASE 5: VALIDATION                                                            │
│  [Devil's Advocate] → [Red-Blue Team] → [Bayesian Fusion] → [Calibration]       │
│                                                                                 │
│  PHASE 6: FINAL DECISION                                                        │
│  [Chairman Synthesis] → [Risk Budget] → [Save] → [Slack Alert]                  │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

**API Endpoints (Flask Server @ localhost:5000):**

| Endpoint | Description |
|----------|-------------|
| `/api/fetch-price` | Fetch Bitcoin price data |
| `/api/calculate-indicators` | Calculate technical indicators |
| `/api/linear-regression-predict` | Linear Regression prediction |
| `/api/arima-predict` | ARIMA prediction |
| `/api/seasonal-predict` | Seasonal model prediction |
| `/api/multi-timeframe-analysis` | Multi-timeframe analysis |
| `/api/scenario-analysis` | Scenario analysis |
| `/api/analyst/<type>` | Individual analyst analysis |
| `/api/anonymize-analyses` | Anonymize analyst outputs |
| `/api/peer-review` | Conduct peer reviews |
| `/api/calculate-rankings` | Calculate quality rankings |
| `/api/devils-advocate` | Devil's Advocate challenge |
| `/api/red-blue-team` | Red-Blue team validation |
| `/api/bayesian-fusion` | Bayesian signal fusion |
| `/api/confidence-calibration` | Confidence calibration |
| `/api/chairman-synthesis` | Chairman final synthesis |
| `/api/risk-budget-calculation` | Risk budget position sizing |
| `/api/save-decision` | Save decision to history |

## License

MIT License
