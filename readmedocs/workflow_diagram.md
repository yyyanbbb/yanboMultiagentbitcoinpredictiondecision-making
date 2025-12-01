# Bitcoin Multi-Agent Investment Decision Workflow Diagram

## 🎯 Complete Workflow Visualization

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                    BITCOIN MULTI-AGENT INVESTMENT DECISION WORKFLOW                              │
│                                              N8N Automation System                                               │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘

═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════
                                              PHASE 1: DATA COLLECTION
═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════

    ┌──────────────────┐         ┌──────────────────────────┐         ┌──────────────────────────────┐
    │   🔔 TRIGGERS    │         │   📊 FETCH PRICE DATA    │         │   📈 CALCULATE INDICATORS    │
    │                  │         │                          │         │                              │
    │  ┌────────────┐  │         │  • Current BTC/USD       │         │  • RSI (14)                  │
    │  │  Webhook   │──┼────────▶│  • Historical prices     │────────▶│  • MACD                      │
    │  │  Trigger   │  │         │  • 100-day series        │         │  • Bollinger Bands           │
    │  └────────────┘  │         │  • Bitfinex API          │         │  • SMA (5, 20, 50)           │
    │                  │         │                          │         │  • EMA (12, 26)              │
    │  ┌────────────┐  │         │  Source: Bitfinex        │         │                              │
    │  │ Scheduled  │──┤         │  Output: price_series[]  │         │  Output: features{}          │
    │  │ (6 hours)  │  │         └──────────────────────────┘         └──────────────────────────────┘
    │  └────────────┘  │                     │                                      │
    └──────────────────┘                     │                                      │
                                             ▼                                      ▼
═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════
                                          PHASE 2: MODEL PREDICTIONS
═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════

                              ┌─────────────────────────────────────────────────────────────┐
                              │                     PARALLEL EXECUTION                       │
                              └─────────────────────────────────────────────────────────────┘
                                                         │
                    ┌────────────────────────────────────┼────────────────────────────────────┐
                    │                                    │                                    │
                    ▼                                    ▼                                    ▼
    ┌───────────────────────────────┐  ┌───────────────────────────────┐  ┌───────────────────────────────┐
    │  🔵 LINEAR REGRESSION         │  │  🟢 ARIMA(2,1,2)              │  │  🟡 SEASONAL MODEL            │
    │     (Elastic Net)             │  │                               │  │     (Prophet-like)            │
    │                               │  │                               │  │                               │
    │  Accuracy: 82.12%             │  │  Accuracy: 54.20%             │  │  Accuracy: 53.47%             │
    │  Weight: 50%                  │  │  Weight: 30%                  │  │  Weight: 20%                  │
    │                               │  │                               │  │                               │
    │  Formula:                     │  │  Formula:                     │  │  Formula:                     │
    │  min ||Xθ-y||² + λ(α||θ||₁    │  │  (1-Σφᵢ Lⁱ)(1-L)ᵈXₜ =        │  │  y(t) = T(t) + S(t) + εₜ     │
    │      + (1-α)/2||θ||²₂)       │  │  (1+Σθᵢ Lⁱ)εₜ                 │  │                               │
    │                               │  │                               │  │  • Trend component            │
    │  • Feature-based prediction   │  │  • Time series autocorrelation│  │  • Weekly seasonality         │
    │  • Regularization prevents    │  │  • Short-term forecasting     │  │  • Yearly patterns            │
    │    overfitting                │  │  • 7-day forecast path        │  │  • Decomposition              │
    └───────────────────────────────┘  └───────────────────────────────┘  └───────────────────────────────┘
                    │                                    │                                    │
                    └────────────────────────────────────┼────────────────────────────────────┘
                                                         │
                                                         ▼
                              ┌─────────────────────────────────────────────────────────────┐
                              │               📊 ENSEMBLE PREDICTION CALCULATION             │
                              │                                                             │
                              │   ŷ_final = 0.50×LR + 0.30×ARIMA + 0.20×Seasonal           │
                              │                                                             │
                              │   Output:                                                   │
                              │   • predicted_price: $71,000                               │
                              │   • change_percent: +5.2%                                  │
                              │   • trend: "upward"                                        │
                              │   • model_consensus: "3/3 bullish"                         │
                              │   • confidence: 0.68                                       │
                              └─────────────────────────────────────────────────────────────┘
                                                         │
                                                         ▼
═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════
                                          PHASE 3: ADVANCED ANALYSIS
═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════

    ┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
    │                                     🕐 MULTI-TIMEFRAME ANALYSIS                                              │
    │                                                                                                             │
    │   ┌─────────────────────────┐   ┌─────────────────────────┐   ┌─────────────────────────┐                   │
    │   │     SHORT-TERM          │   │     MEDIUM-TERM         │   │     LONG-TERM           │                   │
    │   │     (1-7 days)          │   │     (1-4 weeks)         │   │     (1-3 months)        │                   │
    │   │                         │   │                         │   │                         │                   │
    │   │  Focus: Technical       │   │  Focus: Sentiment       │   │  Focus: Fundamentals    │                   │
    │   │  Weight: 30%            │   │  Weight: 40%            │   │  Weight: 30%            │                   │
    │   │  Change: +2.6%          │   │  Change: +5.2%          │   │  Change: +7.8%          │                   │
    │   │  Signal: BUY            │   │  Signal: BUY            │   │  Signal: BUY            │                   │
    │   └─────────────────────────┘   └─────────────────────────┘   └─────────────────────────┘                   │
    │                                                                                                             │
    │   Synthesis: "Mostly bullish across all timeframes"                                                         │
    └─────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
                                                         │
                                                         ▼
    ┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
    │                                       🎲 SCENARIO ANALYSIS ENGINE                                            │
    │                                                                                                             │
    │   ┌────────────────────┐  ┌────────────────────┐  ┌────────────────────┐  ┌────────────────────┐            │
    │   │  🐂 BULL MARKET    │  │  🐻 BEAR MARKET    │  │  📊 SIDEWAYS       │  │  🦢 BLACK SWAN     │            │
    │   │                    │  │                    │  │                    │  │                    │            │
    │   │  Prob: 30%         │  │  Prob: 20%         │  │  Prob: 40%         │  │  Prob: 10%         │            │
    │   │  Change: +15%      │  │  Change: -15%      │  │  Change: +1.5%     │  │  Change: -35%      │            │
    │   │  Price: $77,625    │  │  Price: $57,375    │  │  Price: $68,513    │  │  Price: $43,875    │            │
    │   │  Risk: Medium      │  │  Risk: High        │  │  Risk: Low         │  │  Risk: Extreme     │            │
    │   └────────────────────┘  └────────────────────┘  └────────────────────┘  └────────────────────┘            │
    │                                                                                                             │
    │   Expected Value: +3.12%  |  Downside Probability: 30%  |  Risk-Adjusted Return: 0.45                       │
    └─────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
                                                         │
                                                         ▼
═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════
                                        PHASE 4: MULTI-AGENT ANALYSIS
═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════

                              ┌─────────────────────────────────────────────────────────────┐
                              │                     5 AI ANALYSTS (LLM)                      │
                              │                    PARALLEL EXECUTION                        │
                              └─────────────────────────────────────────────────────────────┘
                                                         │
        ┌────────────────┬───────────────┬───────────────┼───────────────┬───────────────┐
        │                │               │               │               │               │
        ▼                ▼               ▼               ▼               ▼               │
   ┌─────────┐     ┌─────────┐     ┌─────────┐     ┌─────────┐     ┌─────────┐          │
   │TECHNICAL│     │INDUSTRY │     │FINANCIAL│     │ MARKET  │     │  RISK   │          │
   │ ANALYST │     │ ANALYST │     │ ANALYST │     │ EXPERT  │     │ ANALYST │          │
   │         │     │         │     │         │     │         │     │         │          │
   │📊 Charts│     │📰 News  │     │💰 R/R   │     │😰 Fear/ │     │⚠️ Risks │          │
   │RSI,MACD │     │Adoption │     │Ratios   │     │ Greed   │     │Exposure │          │
   │         │     │         │     │         │     │         │     │         │          │
   │Vote: BUY│     │Vote: BUY│     │Vote:HOLD│     │Vote: BUY│     │Vote:HOLD│          │
   │Conf: 8  │     │Conf: 7  │     │Conf: 6  │     │Conf: 7  │     │Conf: 5  │          │
   └─────────┘     └─────────┘     └─────────┘     └─────────┘     └─────────┘          │
        │                │               │               │               │               │
        └────────────────┴───────────────┴───────────────┼───────────────┴───────────────┘
                                                         │
                                                         ▼
                              ┌─────────────────────────────────────────────────────────────┐
                              │                    🔒 ANONYMIZATION                          │
                              │                                                             │
                              │   Technical Analyst  →  Analyst A                          │
                              │   Industry Analyst   →  Analyst B                          │
                              │   Financial Analyst  →  Analyst C                          │
                              │   Market Expert      →  Analyst D                          │
                              │   Risk Analyst       →  Analyst E                          │
                              │                                                             │
                              │   Purpose: Eliminate authority bias, enable objective review│
                              └─────────────────────────────────────────────────────────────┘
                                                         │
                                                         ▼
                              ┌─────────────────────────────────────────────────────────────┐
                              │                  📝 ANONYMOUS PEER REVIEW                    │
                              │                                                             │
                              │   Each analyst reviews others on 5 dimensions:             │
                              │                                                             │
                              │   ┌──────────────────┬────────────────────────────┐        │
                              │   │    Dimension     │         Weight             │        │
                              │   ├──────────────────┼────────────────────────────┤        │
                              │   │   Accuracy       │           30%              │        │
                              │   │   Insight        │           25%              │        │
                              │   │   Logic          │           20%              │        │
                              │   │   Risk Awareness │           15%              │        │
                              │   │   Actionability  │           10%              │        │
                              │   └──────────────────┴────────────────────────────┘        │
                              │                                                             │
                              │   Total Reviews: 20 (5 analysts × 4 others each)           │
                              └─────────────────────────────────────────────────────────────┘
                                                         │
                                                         ▼
                              ┌─────────────────────────────────────────────────────────────┐
                              │                    🏆 QUALITY RANKINGS                       │
                              │                                                             │
                              │   #1. Analyst A (Technical) - Score: 8.2/10                │
                              │   #2. Analyst D (Market)    - Score: 7.5/10                │
                              │   #3. Analyst B (Industry)  - Score: 7.1/10                │
                              │   #4. Analyst C (Financial) - Score: 6.8/10                │
                              │   #5. Analyst E (Risk)      - Score: 6.2/10                │
                              │                                                             │
                              │   Weight Adjustment: W_final = W_base × (Q_i / ΣQ_j)       │
                              │   Consensus: BUY (3/5 analysts)                            │
                              └─────────────────────────────────────────────────────────────┘
                                                         │
                                                         ▼
═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════
                                      PHASE 5: VALIDATION & CALIBRATION
═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════

    ┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
    │                                       😈 DEVIL'S ADVOCATE CHALLENGE                                          │
    │                                                                                                             │
    │   Challenging Consensus: BUY (3/5 analysts agree)                                                           │
    │                                                                                                             │
    │   Counter-Arguments:                                                                                        │
    │   ⚡ The market may be in a bull trap - recent gains could reverse sharply                                  │
    │   ⚡ Institutional interest might be peaking, suggesting smart money is selling                             │
    │   ⚡ Technical indicators may lag behind fundamental deterioration                                          │
    │                                                                                                             │
    │   Alternative Scenarios:                                                                                    │
    │   🔄 Black swan event (exchange hack, regulatory ban) - Could cause 30-50% drop                            │
    │   🔄 Major institutional adoption announcement - Could trigger 20-40% rally                                │
    │                                                                                                             │
    │   Challenge Confidence: 6/10                                                                                │
    └─────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
                                                         │
                                                         ▼
    ┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
    │                                       🔴🔵 RED-BLUE TEAM VALIDATION                                          │
    │                                                                                                             │
    │   ┌─────────────────────────────────────┐     ┌─────────────────────────────────────┐                       │
    │   │         🔴 RED TEAM                 │     │         🔵 BLUE TEAM                │                       │
    │   │         (Attackers)                 │     │         (Defenders)                 │                       │
    │   │                                     │     │                                     │                       │
    │   │  Challenges:                        │     │  Defenses:                          │                       │
    │   │  • Insufficient model confidence    │     │  • Multi-model ensemble reliability │                       │
    │   │  • Downside risk underestimated     │     │  • Decision aligns with prediction  │                       │
    │   │  • Tail risk not considered         │     │  • Reasonable position control      │                       │
    │   │                                     │     │  • Strong historical performance    │                       │
    │   │  Risk Rating: 6/10                  │     │  Defense Strength: 7/10             │                       │
    │   └─────────────────────────────────────┘     └─────────────────────────────────────┘                       │
    │                                                                                                             │
    │   ⚖️ JUDGE VERDICT: Decision Passed Validation                                                              │
    │   Final Score: 6.6/10 | Recommendation: Can execute original decision                                       │
    └─────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
                                                         │
                                                         ▼
    ┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
    │                                       📊 BAYESIAN SIGNAL FUSION                                              │
    │                                                                                                             │
    │   Signal Sources & Prior Accuracy:                                                                          │
    │   ┌────────────────────────┬──────────────┬──────────────┐                                                  │
    │   │      Signal            │   Accuracy   │    Weight    │                                                  │
    │   ├────────────────────────┼──────────────┼──────────────┤                                                  │
    │   │   Linear Regression    │    82%       │    0.35      │                                                  │
    │   │   ARIMA                │    54%       │    0.18      │                                                  │
    │   │   Seasonal             │    53%       │    0.15      │                                                  │
    │   │   RSI Signal           │    60%       │    0.12      │                                                  │
    │   │   MACD Signal          │    58%       │    0.10      │                                                  │
    │   │   MA Crossover         │    55%       │    0.10      │                                                  │
    │   └────────────────────────┴──────────────┴──────────────┘                                                  │
    │                                                                                                             │
    │   Fusion Result:                                                                                            │
    │   • Up Probability: 68%                                                                                     │
    │   • Down Probability: 32%                                                                                   │
    │   • Direction: UP                                                                                           │
    │   • Confidence: 0.68                                                                                        │
    └─────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
                                                         │
                                                         ▼
    ┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
    │                                      🎯 CONFIDENCE CALIBRATION                                               │
    │                                                                                                             │
    │   Calibration Curve (Historical):                                                                           │
    │                                                                                                             │
    │   Predicted │ 50%  60%  70%  80%  90%  100%                                                                 │
    │   Actual    │ 52%  58%  65%  72%  78%   82%                                                                 │
    │                                                                                                             │
    │   Raw Confidence: 72%                                                                                       │
    │   Calibrated Confidence: 65%                                                                                │
    │   Calibration Gap: +7% (Slightly Overconfident)                                                             │
    │                                                                                                             │
    │   Recommendation: Confidence assessment reasonable, minor adjustment applied                                │
    └─────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
                                                         │
                                                         ▼
═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════
                                       PHASE 6: FINAL DECISION & OUTPUT
═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════

    ┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
    │                                      👔 CHAIRMAN FINAL SYNTHESIS                                             │
    │                                                                                                             │
    │   Inputs Considered:                                                                                        │
    │   ✓ Quality-weighted analyst rankings (#1 Technical Analyst: 8.2/10)                                       │
    │   ✓ Devil's Advocate challenge (Confidence: 6/10)                                                          │
    │   ✓ Red-Blue Team verdict (Passed, Score: 6.6/10)                                                          │
    │   ✓ Bayesian fusion (68% up probability)                                                                   │
    │   ✓ Calibrated confidence (65%)                                                                            │
    │                                                                                                             │
    │   Decision Formula:                                                                                         │
    │   buy_score = up_prob × 0.4 + verdict_score × 0.3 + (1 - DA_conf) × 0.3                                    │
    │   buy_score = 0.68 × 0.4 + 0.66 × 0.3 + 0.4 × 0.3 = 0.59                                                   │
    │                                                                                                             │
    │   ┌─────────────────────────────────────────────────────────────────────────────────────────────────────┐  │
    │   │                              📋 FINAL RECOMMENDATION                                                │  │
    │   │                                                                                                     │  │
    │   │   Action: BUY                                                                                       │  │
    │   │   Confidence: 65%                                                                                   │  │
    │   │   Position: 45%                                                                                     │  │
    │   └─────────────────────────────────────────────────────────────────────────────────────────────────────┘  │
    └─────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
                                                         │
                                                         ▼
    ┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
    │                                      💰 RISK BUDGET POSITION SIZING                                          │
    │                                                                                                             │
    │   Account Information:                                                                                      │
    │   • Total Capital: $100,000                                                                                 │
    │   • Max Risk Budget: 5% ($5,000)                                                                            │
    │                                                                                                             │
    │   Trade Parameters:                                                                                         │
    │   • Entry Price: $67,500                                                                                    │
    │   • Stop-Loss: $62,100 (-8%)                                                                                │
    │   • Risk per Unit: $5,400                                                                                   │
    │                                                                                                             │
    │   Position Calculation:                                                                                     │
    │   Position = (Capital × Risk%) / |Entry - StopLoss|                                                        │
    │   Position = ($100,000 × 5%) / $5,400 = 0.93 BTC                                                           │
    │                                                                                                             │
    │   Confidence Adjustment: 0.93 × 0.82 = 0.76 BTC                                                            │
    │                                                                                                             │
    │   ┌─────────────────────────────────────────────────────────────────────────────────────────────────────┐  │
    │   │   Recommended Units: 0.76 BTC                                                                       │  │
    │   │   Investment Amount: $45,000                                                                        │  │
    │   │   Position Ratio: 45%                                                                               │  │
    │   │   Maximum Loss: $5,000                                                                              │  │
    │   └─────────────────────────────────────────────────────────────────────────────────────────────────────┘  │
    └─────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
                                                         │
                                                         ▼
    ┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
    │                                         📤 OUTPUT & NOTIFICATIONS                                            │
    │                                                                                                             │
    │   ┌─────────────────────┐    ┌─────────────────────┐    ┌─────────────────────┐                            │
    │   │  💾 SAVE TO HISTORY │    │   📱 SLACK ALERT    │    │  🔗 WEBHOOK RESPONSE │                            │
    │   │                     │    │                     │    │                     │                            │
    │   │  investment_history/│    │  #bitcoin-alerts    │    │  JSON Response      │                            │
    │   │  decision_YYYYMMDD_ │    │                     │    │                     │                            │
    │   │  HHMMSS.json        │    │  🚀 BUY SIGNAL 🚀   │    │  {                  │                            │
    │   │                     │    │  Price: $67,500     │    │    decision_id,     │                            │
    │   │  For future         │    │  Predicted: $71,000 │    │    price_info,      │                            │
    │   │  learning &         │    │  Position: 45%      │    │    recommendation,  │                            │
    │   │  validation         │    │  Confidence: 65%    │    │    risk_assessment  │                            │
    │   │                     │    │                     │    │  }                  │                            │
    │   └─────────────────────┘    └─────────────────────┘    └─────────────────────┘                            │
    └─────────────────────────────────────────────────────────────────────────────────────────────────────────────┘


═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════
                                              WORKFLOW SUMMARY
═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════

┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                                                                                                 │
│   Total Nodes: 23                                                                                               │
│   Total Phases: 6                                                                                               │
│   Execution Time: ~2-5 minutes                                                                                  │
│                                                                                                                 │
│   Key Innovations:                                                                                              │
│   ✓ Three-model ensemble prediction with accuracy-based weighting                                              │
│   ✓ Anonymous peer review to eliminate authority bias                                                          │
│   ✓ Devil's Advocate challenge for stress-testing consensus                                                    │
│   ✓ Red-Blue Team adversarial validation                                                                       │
│   ✓ Bayesian signal fusion for probabilistic decision-making                                                   │
│   ✓ Confidence calibration to address overconfidence                                                           │
│   ✓ Risk budget-based position sizing                                                                          │
│                                                                                                                 │
│   Model Performance:                                                                                            │
│   • Linear Regression: 82.12% accuracy (Primary signal)                                                        │
│   • ARIMA: 54.20% accuracy (Time series component)                                                             │
│   • Seasonal: 53.47% accuracy (Pattern recognition)                                                            │
│                                                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

## 📊 Node Connection Map

```
[1] Webhook ─────────────────────────────────────────────────────────────────────────────────────────────────────┐
[2] Schedule ────────────────────────────────────────────────────────────────────────────────────────────────────┤
                                                                                                                 │
                                                                                                                 ▼
[3] Fetch Price ─────────────────────────────────────────────────────────────────────────────────────────────────┤
                                                                                                                 │
                                                                                                                 ▼
[4] Calculate Indicators ────────────────────────────────────────────────────────────────────────────────────────┤
                                                                                                                 │
                           ┌─────────────────────────────────────────────────────────────────────────────────────┤
                           │                                                                                     │
                           ▼                                   ▼                                   ▼             │
[5a] Linear Regression ────┤            [5b] ARIMA ───────────┤            [5c] Seasonal ─────────┤             │
                           │                                   │                                   │             │
                           └───────────────────────────────────┼───────────────────────────────────┘             │
                                                               │                                                 │
                                                               ▼                                                 │
[6] Merge Predictions ───────────────────────────────────────────────────────────────────────────────────────────┤
                                                               │                                                 │
                                                               ▼                                                 │
[7] Ensemble Calculation ────────────────────────────────────────────────────────────────────────────────────────┤
                                                               │                                                 │
                                                               ▼                                                 │
[8] Multi-Timeframe ─────────────────────────────────────────────────────────────────────────────────────────────┤
                                                               │                                                 │
                                                               ▼                                                 │
[9] Scenario Analysis ───────────────────────────────────────────────────────────────────────────────────────────┤
                                                               │                                                 │
                           ┌───────────────────────────────────┼───────────────────────────────────┐             │
                           │               │               │               │               │       │             │
                           ▼               ▼               ▼               ▼               ▼       │             │
[10a] Technical ──┤  [10b] Industry ──┤  [10c] Financial ─┤  [10d] Market ───┤  [10e] Risk ───┤   │             │
                  │                   │                   │                  │                │   │             │
                  └───────────────────┴───────────────────┼──────────────────┴────────────────┘   │             │
                                                          │                                       │             │
                                                          ▼                                       │             │
[11] Merge Analysts ─────────────────────────────────────────────────────────────────────────────────────────────┤
                                                          │                                                      │
                                                          ▼                                                      │
[12] Anonymize ──────────────────────────────────────────────────────────────────────────────────────────────────┤
                                                          │                                                      │
                                                          ▼                                                      │
[13] Peer Review ────────────────────────────────────────────────────────────────────────────────────────────────┤
                                                          │                                                      │
                                                          ▼                                                      │
[14] Calculate Rankings ─────────────────────────────────────────────────────────────────────────────────────────┤
                                                          │                                                      │
                                                          ▼                                                      │
[15] Devil's Advocate ───────────────────────────────────────────────────────────────────────────────────────────┤
                                                          │                                                      │
                                                          ▼                                                      │
[16] Red-Blue Team ──────────────────────────────────────────────────────────────────────────────────────────────┤
                                                          │                                                      │
                                                          ▼                                                      │
[17] Bayesian Fusion ────────────────────────────────────────────────────────────────────────────────────────────┤
                                                          │                                                      │
                                                          ▼                                                      │
[18] Confidence Calibration ─────────────────────────────────────────────────────────────────────────────────────┤
                                                          │                                                      │
                                                          ▼                                                      │
[19] Chairman Synthesis ─────────────────────────────────────────────────────────────────────────────────────────┤
                                                          │                                                      │
                                                          ▼                                                      │
[20] Risk Budget ────────────────────────────────────────────────────────────────────────────────────────────────┤
                                                          │                                                      │
                                                          ▼                                                      │
[21] Compile Decision ───────────────────────────────────────────────────────────────────────────────────────────┤
                                                          │                                                      │
                                                          ▼                                                      │
[22] Save to History ────────────────────────────────────────────────────────────────────────────────────────────┤
                                                          │                                                      │
                                                          ▼                                                      │
[23] Check Action ───────────────────────────────────────────────────────────────────────────────────────────────┤
                           │                                                               │                     │
                           ▼                                                               ▼                     │
[24a] Slack Buy Alert ─────┤                                             [24b] Slack Sell Alert ────────────────┤
                           │                                                               │                     │
                           └───────────────────────────────┬───────────────────────────────┘                     │
                                                           │                                                     │
                                                           ▼                                                     │
[25] Webhook Response ───────────────────────────────────────────────────────────────────────────────────────────┘
```

## 🔗 API Endpoint Flow

```
Webhook/Schedule
       │
       ▼
POST /api/fetch-price
       │
       ▼
POST /api/calculate-indicators
       │
       ├──────────────────────────────────────────┐
       │                                          │
       ▼                                          ▼
POST /api/linear-regression-predict    POST /api/arima-predict
       │                                          │
       │                                          ▼
       │                              POST /api/seasonal-predict
       │                                          │
       └──────────────────┬───────────────────────┘
                          │
                          ▼
              [JavaScript: Ensemble Calc]
                          │
                          ▼
            POST /api/multi-timeframe-analysis
                          │
                          ▼
              POST /api/scenario-analysis
                          │
       ┌──────────────────┼──────────────────┐
       │                  │                  │
       ▼                  ▼                  ▼
POST /api/analyst/    POST /api/analyst/  POST /api/analyst/
   technical            industry           financial
       │                  │                  │
       │                  │                  │
       ▼                  ▼                  ▼
POST /api/analyst/    POST /api/analyst/
    market               risk
       │                  │
       └──────────────────┼──────────────────┘
                          │
                          ▼
            POST /api/anonymize-analyses
                          │
                          ▼
              POST /api/peer-review
                          │
                          ▼
            POST /api/calculate-rankings
                          │
                          ▼
            POST /api/devils-advocate
                          │
                          ▼
              POST /api/red-blue-team
                          │
                          ▼
             POST /api/bayesian-fusion
                          │
                          ▼
          POST /api/confidence-calibration
                          │
                          ▼
            POST /api/chairman-synthesis
                          │
                          ▼
          POST /api/risk-budget-calculation
                          │
                          ▼
             POST /api/save-decision
                          │
                          ▼
              [Slack Notification]
                          │
                          ▼
              [Webhook Response]
```

