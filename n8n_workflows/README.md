# N8N Workflow for Bitcoin Multi-Agent Investment Decision System

This directory contains the N8N workflow configuration and supporting files for automating the Bitcoin investment decision-making process using a multi-agent architecture.

## 📁 Directory Structure

```
n8n_workflows/
├── setup_and_run.py                    # ★★★ Main launcher (recommended)
├── run_n8n_server.py                   # API server launcher
├── bitcoin_investment_workflow.json    # Basic N8N workflow (23 nodes)
├── bitcoin_investment_workflow_enhanced.json  # Enhanced workflow (27 nodes) ★
├── n8n_api_server.py                   # Flask API server
├── README.md                           # This documentation
└── workflow_diagram.md                 # Visual workflow diagram
```

## 🚀 Quick Start

### Option 1: Standalone Mode (Easiest - No N8N Required!)

Run the complete workflow directly in Python:

```bash
python n8n_workflows/setup_and_run.py --standalone
```

Or use the interactive menu:

```bash
python n8n_workflows/setup_and_run.py
# Then select option [1]
```

This runs the full 10-phase investment analysis without needing N8N!

### Option 2: With N8N (Full Automation)

**Step 1: Install Node.js** (if not installed)
- Download from: https://nodejs.org/
- Recommended: LTS version (18.x or 20.x)

**Step 2: Install N8N**
```bash
npm install -g n8n
```

**Step 3: Start Services**
```bash
# Terminal 1: Start API Server
python n8n_workflows/run_n8n_server.py

# Terminal 2: Start N8N
npx n8n start
# Open http://localhost:5678
```

**Step 4: Import Workflow**
1. Open N8N at http://localhost:5678
2. Click **Add Workflow** → **Import from File**
3. Select `n8n_workflows/bitcoin_investment_workflow.json`
4. Click **Execute Workflow**

### Option 3: One-Command Full Setup

```bash
python n8n_workflows/setup_and_run.py --full
```

This starts both API server and N8N automatically.

## 🔄 Workflow Overview

The workflow consists of **23 nodes** organized into **6 phases**:

### Phase 1: Data Collection & Preprocessing (Steps 1-2)
```
[Webhook/Schedule] → [Fetch Price Data] → [Calculate Indicators]
```
- Fetches Bitcoin price data from Bitfinex
- Calculates technical indicators (RSI, MACD, Bollinger Bands, SMA, EMA)

### Phase 2: Model Predictions (Steps 3-5)
```
                    ┌→ [Linear Regression] ─┐
[Indicators] ──────→├→ [ARIMA Model] ───────├→ [Merge] → [Ensemble Calculation]
                    └→ [Seasonal Model] ────┘
```
- **Linear Regression (Elastic Net)**: 82.12% accuracy, 50% weight
- **ARIMA(2,1,2)**: 54.20% accuracy, 30% weight
- **Seasonal (Prophet-like)**: 53.47% accuracy, 20% weight

### Phase 3: Advanced Analysis (Steps 6-7)
```
[Ensemble] → [Multi-Timeframe Analysis] → [Scenario Analysis]
```
- **Multi-Timeframe**: Short-term (1-7 days), Medium-term (1-4 weeks), Long-term (1-3 months)
- **Scenario Analysis**: Bull/Bear/Sideways/Black Swan scenarios with probabilities

### Phase 4: Multi-Agent Analysis (Steps 8-12)
```
                    ┌→ [Technical Analyst] ─┐
                    ├→ [Industry Analyst] ──┤
[Scenarios] ───────→├→ [Financial Analyst] ─├→ [Merge] → [Anonymize] → [Peer Review] → [Rankings]
                    ├→ [Market Expert] ─────┤
                    └→ [Risk Analyst] ──────┘
```
- 5 specialized AI analysts provide independent opinions
- Anonymous peer review on 5 dimensions
- Quality-weighted rankings

### Phase 5: Validation & Calibration (Steps 13-16)
```
[Rankings] → [Devil's Advocate] → [Red-Blue Team] → [Bayesian Fusion] → [Confidence Calibration]
```
- **Devil's Advocate**: Challenges consensus view
- **Red-Blue Team**: Adversarial validation
- **Bayesian Fusion**: Probabilistic signal combination
- **Confidence Calibration**: Adjusts for overconfidence/underconfidence

### Phase 6: Final Decision & Output (Steps 17-23)
```
[Calibration] → [Chairman Synthesis] → [Risk Budget] → [Compile Decision] → [Save] → [Alert]
```
- Chairman synthesizes all inputs
- Risk-based position sizing
- Saves to history
- Sends Slack alerts

## 📊 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/health` | GET | Health check |
| `/api/fetch-price` | POST | Fetch Bitcoin price data |
| `/api/calculate-indicators` | POST | Calculate technical indicators |
| `/api/linear-regression-predict` | POST | Linear Regression prediction |
| `/api/arima-predict` | POST | ARIMA prediction |
| `/api/seasonal-predict` | POST | Seasonal model prediction |
| `/api/multi-timeframe-analysis` | POST | Multi-timeframe analysis |
| `/api/scenario-analysis` | POST | Scenario analysis |
| `/api/analyst/<type>` | POST | Individual analyst analysis |
| `/api/anonymize-analyses` | POST | Anonymize analyst outputs |
| `/api/peer-review` | POST | Conduct peer reviews |
| `/api/calculate-rankings` | POST | Calculate quality rankings |
| `/api/devils-advocate` | POST | Generate Devil's Advocate challenge |
| `/api/red-blue-team` | POST | Red-Blue team validation |
| `/api/bayesian-fusion` | POST | Bayesian signal fusion |
| `/api/confidence-calibration` | POST | Calibrate confidence |
| `/api/chairman-synthesis` | POST | Chairman final synthesis |
| `/api/risk-budget-calculation` | POST | Risk budget position sizing |
| `/api/save-decision` | POST | Save decision to history |

## 🧮 Mathematical Formulas

### Ensemble Prediction
$$\hat{y}_{final} = \sum_{m \in M} w_m \cdot \hat{y}_m$$

Where weights are:
- Linear Regression: $w_{LR} = 0.50$
- ARIMA: $w_{ARIMA} = 0.30$
- Seasonal: $w_{Seasonal} = 0.20$

### Peer Review Weight Adjustment
$$W_i^{final} = W_i^{base} \times \frac{Q_i}{\sum_j Q_j}$$

Where $Q_i$ is the quality score from peer review.

### Risk Budget Position Sizing
$$\text{Position} = \frac{\text{Total Capital} \times \text{Risk \%}}{|\text{Entry} - \text{StopLoss}|}$$

## 🔧 Configuration

### Environment Variables

Create a `.env` file in the project root:

```env
DEEPSEEK_API_KEY=your_api_key
DEEPSEEK_BASE_URL=https://api.deepseek.com
MODEL_NAME=deepseek-chat
```

### N8N Credentials

Configure the following in N8N:
- **Slack**: For alert notifications
- **HTTP Request**: Default settings work

### Workflow Settings

In the workflow JSON:
- `executionOrder`: "v1"
- `saveManualExecutions`: true
- Scheduled trigger: Every 6 hours

## 📈 Output Format

The final decision output includes:

```json
{
  "timestamp": "2024-12-01T12:00:00.000Z",
  "decision_id": "DEC_1701432000000",
  "price_info": {
    "current_price": 67500,
    "predicted_price": 71000,
    "change_percent": 5.2,
    "trend": "upward"
  },
  "recommendation": {
    "action": "buy",
    "confidence": 0.72,
    "position_percent": 45,
    "entry_price": 67500,
    "stop_loss": 62100,
    "take_profit": 75600
  },
  "analyst_rankings": [...],
  "risk_assessment": {
    "devils_advocate_confidence": 6,
    "red_blue_verdict": "Decision passed validation",
    "max_loss": 5000
  },
  "model_performance": {
    "linear_regression": { "accuracy": 0.8212, "weight": 0.50 },
    "arima": { "accuracy": 0.542, "weight": 0.30 },
    "seasonal": { "accuracy": 0.535, "weight": 0.20 }
  }
}
```

## 🖥️ GUI Integration Snapshot

- Latest run snapshot: `n8n_workflows/latest_workflow_result.json`
- Historical runs: `n8n_workflows/workflow_runs/workflow_run_<timestamp>.json`

After executing the workflow (menu option **5** or `--workflow`), open `gui_investment_system.py` and click **“⬇ Load Latest Result”** to inject:

1. **📊 Main Output** – price data, indicators, ensemble predictions
2. **🔍 Analysis Details** – multi-timeframe signals, scenarios, analyst notes
3. **📝 Peer Review** – anonymized reviews, rankings, Devil’s Advocate, Red-Blue team
4. **🎯 Final Decision** – chairman synthesis, risk budget, saved decision path

This bridges the automated workflow and the desktop GUI without re-running analyses inside the UI.

## 🔒 Security Considerations

1. **API Authentication**: Add API key authentication for production
2. **Rate Limiting**: Implement rate limiting on API endpoints
3. **Data Encryption**: Use HTTPS for all communications
4. **Credential Storage**: Use N8N's credential management

## 🐛 Troubleshooting

### API Server Issues

```bash
# Check if server is running
curl http://localhost:5000/api/health

# Check logs
python n8n_api_server.py 2>&1 | tee server.log
```

### N8N Workflow Issues

1. Check node execution order
2. Verify HTTP request URLs
3. Check credential configurations
4. Review execution logs in N8N

### Model Loading Issues

```bash
# Ensure dependencies are installed
pip install -r ../requirements.txt

# Check if data files exist
ls ../Bitcoin\ price\ prediction\ Project/data/
```

## 📚 References

- [N8N Documentation](https://docs.n8n.io/)
- [Flask Documentation](https://flask.palletsprojects.com/)
- [Project Main README](../README.md)

## 📄 License

This project is part of the Bitcoin Investment Decision System. See the main project for license information.

