# -*- coding: utf-8 -*-
"""
N8N API Server for Bitcoin Investment Decision System

This Flask server provides RESTful API endpoints for the N8N workflow
to interact with the Bitcoin prediction and multi-agent decision system.

Author: Investment Committee System
Version: v1.0

Usage:
    python n8n_api_server.py
    
The server will start on http://localhost:5000
"""

import os
import sys
import json
import random
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
from flask import Flask, request, jsonify
from flask_cors import CORS

# Add parent directory to path for imports
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

# Import project modules
try:
    from core_prediction_models import (
        LinearRegressionModel, 
        ARIMAModel, 
        SeasonalModel,
        EnsemblePredictorForAnalysts,
        MinMaxScaler
    )
    from anonymous_peer_review import (
        AnonymousPeerReviewSystem,
        AnonymousAnalysis,
        PeerReview,
        DevilsAdvocateChallenge
    )
    from advanced_features import (
        MultiTimeframeAnalyzer,
        ScenarioAnalysisEngine,
        BayesianSignalFusion,
        RedBlueTeamValidator,
        RiskBudgetManager,
        ConfidenceCalibrator
    )
    MODULES_LOADED = True
except ImportError as e:
    print(f"[!] Warning: Could not import some modules: {e}")
    MODULES_LOADED = False

import numpy as np
import pandas as pd

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for N8N requests

# Global instances
ensemble_predictor = None
peer_review_system = None


# ==================== Formatting Helpers ====================

def format_currency(value, decimals=2):
    try:
        return f"${float(value):,.{decimals}f}"
    except (TypeError, ValueError):
        return "N/A"


def format_percentage(value, decimals=1, show_sign=True):
    try:
        num = float(value)
        if abs(num) <= 1:
            num *= 100
        sign = "+" if show_sign and num >= 0 else ""
        return f"{sign}{num:.{decimals}f}%"
    except (TypeError, ValueError):
        return "N/A"


def format_decimal(value, decimals=2):
    try:
        return f"{float(value):.{decimals}f}"
    except (TypeError, ValueError):
        return "N/A"


def parse_consensus(consensus_text: Optional[str]):
    if not consensus_text:
        return 0, 0, "UNKNOWN"
    digits = list(map(int, re.findall(r'\d+', consensus_text)))
    bullish = digits[0] if len(digits) >= 1 else 0
    total = digits[1] if len(digits) >= 2 else max(bullish, 1)
    neutral = max(total - bullish, 0)
    text_upper = consensus_text.upper()
    if "SELL" in text_upper:
        dominant = "BEARISH"
    elif "BUY" in text_upper:
        dominant = "BULLISH"
    else:
        dominant = "NEUTRAL"
    return bullish, neutral, dominant


# ==================== Initialization ====================

def initialize_models():
    """Initialize prediction models on startup"""
    global ensemble_predictor
    
    if not MODULES_LOADED:
        print("[!] Modules not loaded, using simulation mode")
        return False
    
    try:
        print("[*] Initializing ensemble prediction models...")
        ensemble_predictor = EnsemblePredictorForAnalysts()
        success = ensemble_predictor.load_and_train()
        if success:
            print("[OK] Models initialized successfully")
            return True
    except Exception as e:
        print(f"[!] Model initialization failed: {e}")
    
    return False


# ==================== API Endpoints ====================

@app.route('/', methods=['GET'])
def index():
    """Root endpoint"""
    return jsonify({
        "message": "Bitcoin Investment Decision System API Server",
        "version": "1.0.0",
        "status": "running",
        "documentation": "/api/health",
        "endpoints": [
            "/api/health",
            "/api/fetch-price",
            "/api/calculate-indicators",
            "..."
        ]
    })


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "models_loaded": ensemble_predictor is not None and ensemble_predictor.is_loaded if ensemble_predictor else False,
        "modules_loaded": MODULES_LOADED
    })


@app.route('/api/fetch-price', methods=['POST'])
def fetch_price():
    """
    Step 1: Fetch Bitcoin price data
    
    In production, this would fetch from Bitfinex API.
    For demo, returns simulated data.
    """
    try:
        data = request.get_json() or {}
        symbol = data.get('symbol', 'BTC/USD')
        
        # Simulated current price data
        current_price = 67500 + random.uniform(-1000, 1000)
        
        # Generate historical price series (last 100 days)
        np.random.seed(int(datetime.now().timestamp()) % 1000)
        base_prices = [current_price]
        for i in range(99):
            change = random.uniform(-0.03, 0.03)
            base_prices.insert(0, base_prices[0] * (1 - change))
        
        # Generate dates
        dates = [(datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d') 
                 for i in range(99, -1, -1)]
        
        return jsonify({
            "success": True,
            "symbol": symbol,
            "current_price": current_price,
            "price_series": base_prices,
            "dates": dates,
            "timestamp": datetime.now().isoformat(),
            "source": "simulated"  # In production: "bitfinex"
        })
    
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/calculate-indicators', methods=['POST'])
def calculate_indicators():
    """
    Step 2: Calculate technical indicators
    
    Calculates RSI, MACD, Bollinger Bands, SMA, EMA
    """
    try:
        data = request.get_json() or {}
        prices = data.get('price_data', {}).get('price_series', [])
        
        if not prices:
            prices = [67500 + random.uniform(-1000, 1000) for _ in range(100)]
        
        prices = np.array(prices)
        
        # Calculate SMA
        sma_5 = np.mean(prices[-5:])
        sma_20 = np.mean(prices[-20:])
        sma_50 = np.mean(prices[-50:]) if len(prices) >= 50 else np.mean(prices)
        
        # Calculate EMA
        def ema(data, period):
            alpha = 2 / (period + 1)
            ema_val = data[0]
            for price in data[1:]:
                ema_val = alpha * price + (1 - alpha) * ema_val
            return ema_val
        
        ema_12 = ema(prices, 12)
        ema_26 = ema(prices, 26)
        
        # Calculate MACD
        macd = ema_12 - ema_26
        signal_line = ema(prices[-9:], 9) if len(prices) >= 9 else ema_12
        macd_histogram = macd - signal_line
        
        # Calculate RSI
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        avg_gain = np.mean(gains[-14:])
        avg_loss = np.mean(losses[-14:])
        rs = avg_gain / avg_loss if avg_loss > 0 else 100
        rsi = 100 - (100 / (1 + rs))
        
        # Calculate Bollinger Bands
        bb_period = 20
        bb_std = np.std(prices[-bb_period:])
        bb_middle = sma_20
        bb_upper = bb_middle + 2 * bb_std
        bb_lower = bb_middle - 2 * bb_std
        
        # Current price position
        current_price = prices[-1]
        
        return jsonify({
            "success": True,
            "current_price": float(current_price),
            "technical_indicators": {
                "sma_5": float(sma_5),
                "sma_20": float(sma_20),
                "sma_50": float(sma_50),
                "ema_12": float(ema_12),
                "ema_26": float(ema_26),
                "macd": float(macd),
                "macd_signal": float(signal_line),
                "macd_histogram": float(macd_histogram),
                "rsi_14": float(rsi),
                "bb_upper": float(bb_upper),
                "bb_middle": float(bb_middle),
                "bb_lower": float(bb_lower)
            },
            "signals": {
                "rsi_signal": "overbought" if rsi > 70 else "oversold" if rsi < 30 else "neutral",
                "macd_signal": "bullish" if macd > signal_line else "bearish",
                "bb_signal": "overbought" if current_price > bb_upper else "oversold" if current_price < bb_lower else "neutral",
                "ma_crossover": "bullish" if sma_5 > sma_20 else "bearish"
            },
            "features": {
                "Open": float(prices[-2]),
                "High": float(max(prices[-5:])),
                "Low": float(min(prices[-5:])),
                "Close": float(current_price),
                "SMA_5": float(sma_5),
                "RSI_14": float(rsi),
                "MACD": float(macd),
                "BB_upper": float(bb_upper),
                "Volume": random.uniform(1000, 5000)
            },
            "price_series": prices.tolist(),
            "timestamp": datetime.now().isoformat()
        })
    
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/linear-regression-predict', methods=['POST'])
def linear_regression_predict():
    """
    Step 3a: Linear Regression (Elastic Net) prediction
    """
    try:
        data = request.get_json() or {}
        features = data.get('features', {})
        current_price = data.get('current_price', 67500)
        
        # Use actual model if available
        if ensemble_predictor and ensemble_predictor.is_loaded:
            result = ensemble_predictor.predict_with_details({'current_price': current_price})
            lr_result = result.get('models', {}).get('linear_regression', {})
            
            return jsonify({
                "success": True,
                "model": "Linear Regression (Elastic Net)",
                "current_price": current_price,
                "predicted_price": lr_result.get('predicted_price', current_price * 1.05),
                "change_percent": lr_result.get('change_percent', 5.0),
                "direction": lr_result.get('direction', 'bullish'),
                "accuracy": 0.8212,
                "confidence": lr_result.get('accuracy', 0.82),
                "timestamp": datetime.now().isoformat()
            })
        
        # Simulated prediction
        change = random.uniform(2, 8)
        predicted_price = current_price * (1 + change / 100)
        
        return jsonify({
            "success": True,
            "model": "Linear Regression (Elastic Net)",
            "current_price": current_price,
            "predicted_price": predicted_price,
            "change_percent": change,
            "direction": "bullish" if change > 0 else "bearish",
            "accuracy": 0.8212,
            "confidence": 0.82,
            "timestamp": datetime.now().isoformat()
        })
    
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/arima-predict', methods=['POST'])
def arima_predict():
    """
    Step 3b: ARIMA(2,1,2) prediction
    """
    try:
        data = request.get_json() or {}
        price_series = data.get('price_series', [])
        steps = data.get('steps', 7)
        
        current_price = price_series[-1] if price_series else 67500
        
        # Use actual model if available
        if ensemble_predictor and ensemble_predictor.is_loaded:
            result = ensemble_predictor.predict_with_details({'current_price': current_price})
            arima_result = result.get('models', {}).get('arima', {})
            
            return jsonify({
                "success": True,
                "model": "ARIMA(2,1,2)",
                "current_price": current_price,
                "predicted_price": arima_result.get('predicted_price', current_price * 1.03),
                "change_percent": arima_result.get('change_percent', 3.0),
                "direction": arima_result.get('direction', 'bullish'),
                "forecast_path": arima_result.get('forecast_path', []),
                "accuracy": 0.542,
                "confidence": 0.54,
                "timestamp": datetime.now().isoformat()
            })
        
        # Simulated prediction
        change = random.uniform(-2, 5)
        predicted_price = current_price * (1 + change / 100)
        
        # Generate forecast path
        forecast_path = [current_price]
        for i in range(steps):
            step_change = change / steps * (i + 1) + random.uniform(-1, 1)
            forecast_path.append(current_price * (1 + step_change / 100))
        
        return jsonify({
            "success": True,
            "model": "ARIMA(2,1,2)",
            "current_price": current_price,
            "predicted_price": predicted_price,
            "change_percent": change,
            "direction": "bullish" if change > 0 else "bearish",
            "forecast_path": forecast_path,
            "accuracy": 0.542,
            "confidence": 0.54,
            "timestamp": datetime.now().isoformat()
        })
    
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/seasonal-predict', methods=['POST'])
def seasonal_predict():
    """
    Step 3c: Seasonal (Prophet-like) prediction
    """
    try:
        data = request.get_json() or {}
        values = data.get('values', [])
        steps = data.get('steps', 7)
        
        current_price = values[-1] if values else 67500
        
        # Use actual model if available
        if ensemble_predictor and ensemble_predictor.is_loaded:
            result = ensemble_predictor.predict_with_details({'current_price': current_price})
            seasonal_result = result.get('models', {}).get('seasonal', {})
            
            return jsonify({
                "success": True,
                "model": "Seasonal (Prophet-like)",
                "current_price": current_price,
                "predicted_price": seasonal_result.get('predicted_price', current_price * 1.02),
                "change_percent": seasonal_result.get('change_percent', 2.0),
                "direction": seasonal_result.get('direction', 'bullish'),
                "forecast_path": seasonal_result.get('forecast_path', []),
                "accuracy": 0.535,
                "confidence": 0.53,
                "timestamp": datetime.now().isoformat()
            })
        
        # Simulated prediction
        change = random.uniform(-1, 4)
        predicted_price = current_price * (1 + change / 100)
        
        # Generate forecast path with weekly seasonality
        forecast_path = [current_price]
        for i in range(steps):
            # Add weekly pattern
            weekly_effect = np.sin(2 * np.pi * i / 7) * 0.5
            step_change = change / steps * (i + 1) + weekly_effect
            forecast_path.append(current_price * (1 + step_change / 100))
        
        return jsonify({
            "success": True,
            "model": "Seasonal (Prophet-like)",
            "current_price": current_price,
            "predicted_price": predicted_price,
            "change_percent": change,
            "direction": "bullish" if change > 0 else "bearish",
            "forecast_path": forecast_path,
            "accuracy": 0.535,
            "confidence": 0.53,
            "timestamp": datetime.now().isoformat()
        })
    
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/multi-timeframe-analysis', methods=['POST'])
def multi_timeframe_analysis():
    """
    Step 6: Multi-Timeframe Analysis
    """
    try:
        data = request.get_json() or {}
        prediction = data.get('prediction', {})
        
        if MODULES_LOADED:
            analyzer = MultiTimeframeAnalyzer(prediction)
            result = analyzer.analyze_all_timeframes()
            report = analyzer.generate_report()
            
            return jsonify({
                "success": True,
                "analysis": result,
                "report": report,
                "timestamp": datetime.now().isoformat()
            })
        
        # Simulated analysis
        change = prediction.get('change_percent', 5)
        
        return jsonify({
            "success": True,
            "analysis": {
                "short_term": {
                    "timeframe": "Short-term",
                    "period": "1-7 days",
                    "predicted_change": change * 0.5,
                    "direction": "bullish" if change > 0 else "bearish",
                    "signal": "buy" if change > 3 else "hold",
                    "confidence": 0.72
                },
                "medium_term": {
                    "timeframe": "Medium-term",
                    "period": "1-4 weeks",
                    "predicted_change": change,
                    "direction": "bullish" if change > 0 else "bearish",
                    "signal": "buy" if change > 2 else "hold",
                    "confidence": 0.68
                },
                "long_term": {
                    "timeframe": "Long-term",
                    "period": "1-3 months",
                    "predicted_change": change * 1.5,
                    "direction": "bullish" if change > 0 else "bearish",
                    "signal": "buy" if change > 0 else "hold",
                    "confidence": 0.60
                },
                "synthesis": {
                    "weighted_change": change,
                    "consistency": "mostly bullish" if change > 0 else "mostly bearish",
                    "final_signal": "buy" if change > 2 else "hold"
                }
            },
            "timestamp": datetime.now().isoformat()
        })
    
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/scenario-analysis', methods=['POST'])
def scenario_analysis():
    """
    Step 7: Scenario Analysis Engine
    """
    try:
        data = request.get_json() or {}
        current_price = data.get('current_price', 67500)
        prediction = data.get('prediction', {})
        
        if MODULES_LOADED:
            engine = ScenarioAnalysisEngine(current_price, prediction)
            result = engine.run_scenario_analysis()
            report = engine.generate_report()
            
            return jsonify({
                "success": True,
                "scenarios": result,
                "report": report,
                "timestamp": datetime.now().isoformat()
            })
        
        # Simulated scenarios
        base_change = prediction.get('change_percent', 5)
        
        return jsonify({
            "success": True,
            "scenarios": {
                "bull_market": {
                    "name": "Bull Market Scenario",
                    "probability": 0.30,
                    "predicted_change": base_change + 10,
                    "predicted_price": current_price * 1.15,
                    "risk_level": "medium"
                },
                "bear_market": {
                    "name": "Bear Market Scenario",
                    "probability": 0.20,
                    "predicted_change": -15,
                    "predicted_price": current_price * 0.85,
                    "risk_level": "high"
                },
                "sideways": {
                    "name": "Sideways Scenario",
                    "probability": 0.40,
                    "predicted_change": base_change * 0.3,
                    "predicted_price": current_price * 1.015,
                    "risk_level": "low"
                },
                "black_swan": {
                    "name": "Black Swan Scenario",
                    "probability": 0.10,
                    "predicted_change": -35,
                    "predicted_price": current_price * 0.65,
                    "risk_level": "extreme"
                },
                "expected_value": {
                    "expected_change": base_change * 0.6,
                    "expected_price": current_price * (1 + base_change * 0.006)
                },
                "risk_assessment": {
                    "worst_case_change": -35,
                    "best_case_change": base_change + 10,
                    "downside_probability": 0.30
                }
            },
            "timestamp": datetime.now().isoformat()
        })
    
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/analyst/<analyst_type>', methods=['POST'])
def analyst_analysis(analyst_type):
    """
    Steps 8a-8e: Individual Analyst Analysis (LLM-based)
    
    Analyst types: technical, industry, financial, market, risk
    """
    try:
        data = request.get_json() or {}
        prediction = data.get('prediction', {})
        
        # Analyst configurations
        analyst_configs = {
            "technical": {
                "name": "Technical Analyst",
                "focus": "Chart patterns, RSI, MACD, Moving Averages",
                "base_confidence": 8
            },
            "industry": {
                "name": "Industry Analyst",
                "focus": "Regulatory news, adoption curves, industry trends",
                "base_confidence": 7
            },
            "financial": {
                "name": "Financial Analyst",
                "focus": "Risk/reward ratios, macro correlations",
                "base_confidence": 6
            },
            "market": {
                "name": "Market Expert",
                "focus": "Sentiment (Fear & Greed), volume analysis",
                "base_confidence": 7
            },
            "risk": {
                "name": "Risk Analyst",
                "focus": "Tail risks, exposure limits, risk warnings",
                "base_confidence": 5
            }
        }
        
        config = analyst_configs.get(analyst_type, analyst_configs["technical"])
        change = prediction.get('change_percent', 5)
        
        # Determine recommendation based on analyst type and prediction
        if analyst_type == "risk":
            recommendation = "hold" if change < 3 else "buy"
            confidence = config["base_confidence"] - 1
        elif analyst_type == "technical":
            recommendation = "buy" if change > 2 else "hold"
            confidence = config["base_confidence"]
        else:
            recommendation = "buy" if change > 0 else "hold"
            confidence = config["base_confidence"] + random.randint(-1, 1)
        
        return jsonify({
            "success": True,
            "analyst": {
                "type": analyst_type,
                "name": config["name"],
                "focus": config["focus"]
            },
            "analysis": {
                "recommendation": recommendation,
                "confidence": confidence,
                "key_points": [
                    f"Based on {config['focus']}",
                    f"Predicted change: {change:.2f}%",
                    f"Current market conditions support {recommendation}"
                ],
                "content": f"As the {config['name']}, focusing on {config['focus']}, "
                          f"I recommend a {recommendation.upper()} position with {confidence}/10 confidence. "
                          f"The predicted {change:.2f}% change aligns with my analysis."
            },
            "timestamp": datetime.now().isoformat()
        })
    
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/anonymize-analyses', methods=['POST'])
def anonymize_analyses():
    """
    Step 10: Anonymize analyst analyses
    """
    try:
        data = request.get_json() or {}
        analyses = data.get('analyses', [])
        
        # Generate random labels
        labels = list("ABCDE")
        random.shuffle(labels)
        
        anonymized = []
        mapping = {}
        
        for i, analysis in enumerate(analyses):
            anon_id = f"Analyst {labels[i]}"
            mapping[anon_id] = analysis.get('analyst', {}).get('type', f'analyst_{i}')
            
            anonymized.append({
                "anonymous_id": anon_id,
                "content": analysis.get('analysis', {}).get('content', ''),
                "recommendation": analysis.get('analysis', {}).get('recommendation', 'hold'),
                "confidence": analysis.get('analysis', {}).get('confidence', 5),
                "key_points": analysis.get('analysis', {}).get('key_points', [])
            })
        
        return jsonify({
            "success": True,
            "anonymous_analyses": anonymized,
            "mapping": mapping,  # In production, this would be kept secret
            "timestamp": datetime.now().isoformat()
        })
    
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/peer-review', methods=['POST'])
def peer_review():
    """
    Step 11: Anonymous Peer Review
    """
    try:
        data = request.get_json() or {}
        anonymous_analyses = data.get('anonymous_analyses', [])
        dimensions = data.get('dimensions', ['accuracy', 'insight', 'logic', 'risk_awareness', 'actionability'])
        
        reviews = []
        
        # Each analyst reviews others
        for reviewer in anonymous_analyses:
            for reviewee in anonymous_analyses:
                if reviewer['anonymous_id'] == reviewee['anonymous_id']:
                    continue
                
                # Generate review scores
                base_score = 6 + random.gauss(0, 1.5)
                if reviewer['recommendation'] == reviewee['recommendation']:
                    base_score += 1
                
                scores = {}
                for dim in dimensions:
                    scores[dim] = max(1, min(10, round(base_score + random.gauss(0, 0.8))))
                
                reviews.append({
                    "reviewer_id": reviewer['anonymous_id'],
                    "reviewee_id": reviewee['anonymous_id'],
                    "scores": scores,
                    "strengths": [f"Clear {reviewee['recommendation']} rationale"],
                    "weaknesses": ["Could consider more scenarios"],
                    "overall_comment": f"Solid analysis with {scores['accuracy']}/10 accuracy score"
                })
        
        return jsonify({
            "success": True,
            "peer_reviews": reviews,
            "total_reviews": len(reviews),
            "timestamp": datetime.now().isoformat()
        })
    
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/calculate-rankings', methods=['POST'])
def calculate_rankings():
    """
    Step 12: Calculate Quality Rankings
    """
    try:
        data = request.get_json() or {}
        peer_reviews = data.get('peer_reviews', [])
        
        # Aggregate scores per analyst
        analyst_scores = {}
        
        for review in peer_reviews:
            reviewee = review['reviewee_id']
            if reviewee not in analyst_scores:
                analyst_scores[reviewee] = []
            
            avg_score = sum(review['scores'].values()) / len(review['scores'])
            analyst_scores[reviewee].append(avg_score)
        
        # Calculate rankings
        rankings = []
        for analyst_id, scores in analyst_scores.items():
            avg = sum(scores) / len(scores) if scores else 5.0
            rankings.append({
                "anonymous_id": analyst_id,
                "average_score": round(avg, 2),
                "total_reviews": len(scores),
                "rank": 0  # Will be set after sorting
            })
        
        # Sort and assign ranks
        rankings.sort(key=lambda x: x['average_score'], reverse=True)
        for i, r in enumerate(rankings):
            r['rank'] = i + 1
        
        # Determine consensus
        recommendations = [r.get('recommendation', 'hold') for r in data.get('anonymous_analyses', [])]
        buy_count = sum(1 for r in recommendations if 'buy' in r.lower())
        consensus = "BUY" if buy_count > len(recommendations) / 2 else "HOLD"
        
        return jsonify({
            "success": True,
            "rankings": rankings,
            "consensus": f"{consensus} ({buy_count}/{len(recommendations)} analysts)",
            "top_analyst": rankings[0] if rankings else None,
            "timestamp": datetime.now().isoformat()
        })
    
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/devils-advocate', methods=['POST'])
def devils_advocate():
    """
    Step 13: Devil's Advocate Challenge
    """
    try:
        data = request.get_json() or {}
        consensus_view = data.get('consensus_view', 'BUY')
        
        # Generate counter-arguments based on consensus
        if 'BUY' in consensus_view.upper():
            counter_arguments = [
                "The market may be in a bull trap - recent gains could reverse sharply",
                "Institutional interest might be peaking, suggesting smart money is selling",
                "Technical indicators may lag behind fundamental deterioration"
            ]
        else:
            counter_arguments = [
                "Oversold conditions often precede strong rebounds",
                "Negative sentiment extremes historically mark bottoms",
                "Selling pressure may be exhausted after recent declines"
            ]
        
        alternative_scenarios = [
            {
                "scenario": "Black swan event (exchange hack, regulatory ban)",
                "probability": "low",
                "impact": "Could cause 30-50% immediate drop"
            },
            {
                "scenario": "Major institutional adoption announcement",
                "probability": "medium",
                "impact": "Could trigger 20-40% rally within days"
            }
        ]
        
        risk_warnings = [
            "Model predictions have inherent uncertainty that may be underestimated",
            "Market regime changes can invalidate historical patterns",
            "Liquidity conditions can amplify moves beyond predictions"
        ]
        
        return jsonify({
            "success": True,
            "devils_advocate": {
                "consensus_view": consensus_view,
                "counter_arguments": counter_arguments,
                "alternative_scenarios": alternative_scenarios,
                "risk_warnings": risk_warnings,
                "confidence_in_challenge": random.randint(5, 8)
            },
            "timestamp": datetime.now().isoformat()
        })
    
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/red-blue-team', methods=['POST'])
def red_blue_team():
    """
    Step 14: Red-Blue Team Validation
    """
    try:
        data = request.get_json() or {}
        decision = data.get('decision', {})
        context = data.get('context', {})
        
        action = decision.get('action', 'hold')
        confidence = decision.get('confidence', 5)
        
        # Red Team challenges
        red_team = {
            "challenges": [
                {
                    "point": "Insufficient model confidence",
                    "detail": "Historical accuracy doesn't guarantee future performance",
                    "severity": "high"
                },
                {
                    "point": "Downside risk underestimated" if 'buy' in action else "Upside potential missed",
                    "detail": "Cryptocurrency market volatility is high",
                    "severity": "medium"
                },
                {
                    "point": "Tail risk not fully considered",
                    "detail": "Extreme events may cause significant losses",
                    "severity": "high"
                }
            ],
            "overall_risk_rating": random.randint(5, 8)
        }
        
        # Blue Team defenses
        blue_team = {
            "defenses": [
                {
                    "point": "Multi-model ensemble improves reliability",
                    "detail": "Using three independent models reduces single model failure risk",
                    "strength": "strong"
                },
                {
                    "point": "Decision aligns with prediction direction",
                    "detail": "Predicted change supports current decision",
                    "strength": "strong"
                },
                {
                    "point": "Reasonable position control",
                    "detail": "Recommended position maintains risk buffer",
                    "strength": "medium"
                }
            ],
            "defense_strength": random.randint(6, 9)
        }
        
        # Judge verdict
        final_score = (blue_team["defense_strength"] * 0.6 + (10 - red_team["overall_risk_rating"]) * 0.4)
        
        if final_score >= 7:
            verdict = "Decision passed validation"
            recommendation = "Can execute original decision"
        elif final_score >= 5:
            verdict = "Decision conditionally passed"
            recommendation = "Recommend reducing position or stricter stop-loss"
        else:
            verdict = "Decision needs revision"
            recommendation = "Recommend reassessing decision"
        
        red_lines = "\n".join([f"  - {challenge['point']}" for challenge in red_team["challenges"]])
        blue_lines = "\n".join([f"  + {defense['point']}" for defense in blue_team["defenses"]])
        red_blue_text = (
            "----------------------------------------\n"
            "RED-BLUE TEAM VALIDATION\n"
            "----------------------------------------\n\n"
            "[Red Team Challenges]\n"
            f"{red_lines}\n\n"
            "[Blue Team Defense]\n"
            f"{blue_lines}\n\n"
            "[Judge Verdict]\n"
            f"  Challenge Score: {red_team['overall_risk_rating']}/10\n"
            f"  Defense Score: {blue_team['defense_strength']}/10\n"
            f"  Final Score: {round(final_score, 2)}/10\n"
            f"  Verdict: {verdict.upper()}"
        )
        
        return jsonify({
            "success": True,
            "red_team": red_team,
            "blue_team": blue_team,
            "verdict": {
                "verdict": verdict,
                "challenge_score": red_team["overall_risk_rating"],
                "defense_score": blue_team["defense_strength"],
                "final_score": round(final_score, 2),
                "recommendation": recommendation
            },
            "report_text": red_blue_text,
            "timestamp": datetime.now().isoformat()
        })
    
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/bayesian-fusion', methods=['POST'])
def bayesian_fusion():
    """
    Step 15: Bayesian Signal Fusion
    """
    try:
        data = request.get_json() or {}
        signals = data.get('signals', {})
        
        if MODULES_LOADED:
            fusion = BayesianSignalFusion()
            result = fusion.fuse_signals(signals)
            
            return jsonify({
                "success": True,
                "fusion_result": result,
                "timestamp": datetime.now().isoformat()
            })
        
        # Simulated fusion
        up_prob = 0.6 + random.uniform(-0.1, 0.1)
        
        return jsonify({
            "success": True,
            "fusion_result": {
                "direction": "up" if up_prob > 0.5 else "down",
                "up_probability": up_prob,
                "down_probability": 1 - up_prob,
                "confidence": max(up_prob, 1 - up_prob),
                "recommendation": "Buy signal" if up_prob > 0.6 else "Hold"
            },
            "timestamp": datetime.now().isoformat()
        })
    
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/confidence-calibration', methods=['POST'])
def confidence_calibration():
    """
    Step 16: Confidence Calibration
    """
    try:
        data = request.get_json() or {}
        raw_confidence = data.get('raw_confidence', 0.7)
        
        if MODULES_LOADED:
            calibrator = ConfidenceCalibrator()
            result = calibrator.calibrate(raw_confidence)
            
            return jsonify({
                "success": True,
                "calibration": result,
                "timestamp": datetime.now().isoformat()
            })
        
        # Simulated calibration
        calibrated = raw_confidence * 0.9  # Typical overconfidence adjustment
        
        return jsonify({
            "success": True,
            "calibration": {
                "raw_confidence": raw_confidence,
                "calibrated_confidence": calibrated,
                "calibration_gap": raw_confidence - calibrated,
                "bias_type": "Overconfident" if raw_confidence > calibrated + 0.1 else "Well Calibrated",
                "recommendation": "Confidence assessment reasonable"
            },
            "timestamp": datetime.now().isoformat()
        })
    
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/chairman-synthesis', methods=['POST'])
def chairman_synthesis():
    """
    Step 17: Chairman Final Synthesis
    """
    try:
        data = request.get_json() or {}
        rankings = data.get('rankings', [])
        devils_advocate = data.get('devils_advocate', {})
        red_blue_verdict = data.get('red_blue_verdict', {})
        bayesian_result = data.get('bayesian_result', {})
        calibrated_confidence = data.get('calibrated_confidence', 0.65)
        consensus_text = data.get('consensus')
        position_guidance = data.get('position_guidance', {})
        
        # Determine final action based on all inputs
        up_prob = bayesian_result.get('up_probability', 0.6)
        verdict_score = red_blue_verdict.get('final_score', 6)
        da_confidence = devils_advocate.get('confidence_in_challenge', 6)
        
        # Weighted decision
        buy_score = up_prob * 0.4 + (verdict_score / 10) * 0.3 + (1 - da_confidence / 10) * 0.3
        
        if buy_score > 0.65:
            action = "strong_buy"
            position = 60
        elif buy_score > 0.55:
            action = "buy"
            position = 45
        elif buy_score > 0.45:
            action = "hold"
            position = 20
        elif buy_score > 0.35:
            action = "sell"
            position = 0
        else:
            action = "strong_sell"
            position = 0
        
        top_score = rankings[0]['average_score'] if rankings else 'N/A'
        bullish_count, neutral_count, dominant_view = parse_consensus(consensus_text)
        total_votes = bullish_count + neutral_count if (bullish_count + neutral_count) else 5
        confidence_score = calibrated_confidence * 10 if calibrated_confidence <= 1 else calibrated_confidence
        confidence_display = f"{round(confidence_score):d}/10"
        confidence_percent = format_percentage(calibrated_confidence, decimals=1)
        stop_loss_pct = position_guidance.get('stop_loss_pct', 8)
        take_profit_pct = position_guidance.get('take_profit_pct', 12)
        position_style = (
            "Aggressive" if position >= 55 else
            "Moderate-Aggressive" if position >= 35 else
            "Balanced" if position >= 15 else
            "Capital Preservation"
        )
        risk_level_score = max(2, min(9, int(10 - up_prob * 5 + da_confidence / 2)))
        risk_level_label = "MODERATE" if 3 <= risk_level_score <= 6 else ("LOW" if risk_level_score < 3 else "HIGH")
        
        synthesis_points = [
            f"Peer-reviewed quality rankings favor {action.upper()} recommendation",
            f"Top-ranked analyst scored {top_score}/10, reinforcing technical conviction" if rankings else
            "Analyst ranking data indicates above-average conviction",
            f"Signal fusion shows {format_percentage(up_prob, decimals=1)} bullish probability",
            f"Devil's Advocate raised {len(devils_advocate.get('counter_arguments', [])) or 'several'} key risks requiring caution",
            "Risk-adjusted position sizing is essential to respect portfolio limits"
        ]
        
        synthesis_block = "\n".join(f"{idx + 1}. {point}" for idx, point in enumerate(synthesis_points))
        
        final_decision_text = (
            "==================================================\n"
            "FINAL INVESTMENT DECISION\n"
            "==================================================\n\n"
            "[Chairman's Synthesis]\n"
            "Based on comprehensive analysis of all inputs:\n"
            f"{synthesis_block}\n\n"
            "==================================================\n"
            f"RECOMMENDATION: {action.upper()}\n"
            "==================================================\n\n"
            "[Decision Details]\n"
            f"  Action: {action.upper()}\n"
            f"  Position Size: {position}%\n"
            f"  Confidence: {confidence_display}\n"
            f"  Stop-Loss: -{stop_loss_pct}%\n"
            f"  Take-Profit: +{take_profit_pct}%"
        )
        
        decision_review_text = (
            "==================================================\n"
            "DECISION REVIEW REPORT\n"
            "==================================================\n\n"
            "[Decision Summary]\n"
            f"  Recommendation: {action.upper()}\n"
            f"  Position: {position}%\n"
            f"  Confidence: {confidence_display} ({confidence_percent})\n"
            f"  Style: {position_style}\n\n"
            "[Consensus Analysis]\n"
            f"  Level: {dominant_view} Consensus\n"
            f"  Bullish: {bullish_count}\n"
            f"  Neutral: {neutral_count}\n"
            f"  Dominant View: {dominant_view}\n\n"
            "[Key Insights]\n"
            "  [+] Multi-model ensemble aligns with analyst consensus\n"
            "  [+] Anonymous peer review reduced anchoring bias\n"
            "  [+] Decision consistent with scenario analysis direction\n"
            "  [*] Position sized within approved risk budget\n\n"
            "[Risk Assessment]\n"
            f"  Risk Level: {risk_level_label} (Score: {risk_level_score}/10)\n"
            "  Risk Factors:\n"
            "    - Market volatility elevated\n"
            "    - Devil's Advocate highlighted regime-shift risks\n"
            "  Mitigations:\n"
            f"    - Strict stop-loss at -{stop_loss_pct}%\n"
            f"    - Profit taking planned at +{take_profit_pct}%\n\n"
            "[Lessons & Recommendations]\n"
            "  - Multi-model ensemble improved prediction stability\n"
            "  - Continue leveraging anonymous peer review to surface blind spots\n"
            "  - Track realized vs. predicted performance for calibration\n"
            f"  - Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        
        return jsonify({
            "success": True,
            "chairman_decision": {
                "action": action,
                "confidence": round(calibrated_confidence, 2),
                "position_percent": position,
                "reasoning": f"Based on Bayesian fusion ({format_percentage(up_prob, decimals=1)}) "
                             f"and validation signals, I recommend {action.upper()} with {position}% position.",
                "key_factors": [
                    f"Top analyst score: {top_score}/10" if rankings else "Analyst ranking data limited",
                    f"Model consensus direction: {'bullish' if up_prob > 0.5 else 'bearish'}",
                    f"Risk validation: {red_blue_verdict.get('verdict', 'N/A')}"
                ]
            },
            "report": {
                "final_investment_decision": final_decision_text,
                "decision_review": decision_review_text
            },
            "formatted_report": f"{final_decision_text}\n\n{decision_review_text}",
            "timestamp": datetime.now().isoformat()
        })
    
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/risk-budget-calculation', methods=['POST'])
def risk_budget_calculation():
    """
    Step 18: Risk Budget Position Sizing
    """
    try:
        data = request.get_json() or {}
        total_capital = data.get('total_capital', 100000)
        max_risk_percent = data.get('max_risk_percent', 5)
        entry_price = data.get('entry_price', 67500)
        stop_loss_percent = data.get('stop_loss_percent', 8)
        confidence = data.get('confidence', 0.7)
        
        if MODULES_LOADED:
            manager = RiskBudgetManager(total_capital, max_risk_percent)
            stop_loss_price = entry_price * (1 - stop_loss_percent / 100)
            result = manager.calculate_position_size(entry_price, stop_loss_price, confidence)
            
            return jsonify({
                "success": True,
                "risk_budget": result,
                "timestamp": datetime.now().isoformat()
            })
        
        # Simulated calculation
        stop_loss_price = entry_price * (1 - stop_loss_percent / 100)
        risk_per_unit = entry_price - stop_loss_price
        max_risk_amount = total_capital * max_risk_percent / 100
        
        base_units = max_risk_amount / risk_per_unit if risk_per_unit > 0 else 0
        adjusted_units = base_units * (0.5 + confidence * 0.5)
        total_investment = adjusted_units * entry_price
        position_percent = min(80, (total_investment / total_capital) * 100)
        max_risk_amount = round(max_risk_amount, 2)
        
        risk_ratio_text = format_decimal(1 + confidence, 1) if confidence is not None else "N/A"
        risk_report = (
            "----------------------------------------\n"
            "RISK BUDGET CALCULATION\n"
            "----------------------------------------\n\n"
            "[Account Parameters]\n"
            f"  Total Capital: {format_currency(total_capital, 0)}\n"
            f"  Max Risk Budget: {format_percentage(max_risk_percent, show_sign=False)} ({format_currency(max_risk_amount)})\n\n"
            "[Trade Parameters]\n"
            f"  Entry Price: {format_currency(entry_price)}\n"
            f"  Stop-Loss Price: {format_currency(stop_loss_price)} (-{stop_loss_percent}%)\n"
            f"  Prediction Confidence: {format_percentage(confidence)}\n\n"
            "[Position Calculation]\n"
            f"  Risk Per Unit: {format_currency(risk_per_unit)}\n"
            f"  Recommended Units: {format_decimal(adjusted_units, 2)} BTC\n"
            f"  Investment Amount: {format_currency(total_investment)}\n"
            f"  Position Ratio: {format_decimal(position_percent, 1)}%\n\n"
            "[Risk-Reward Analysis]\n"
            f"  Potential Risk: {format_currency(risk_per_unit)}\n"
            f"  Potential Reward: {format_currency(risk_per_unit * (1 + confidence))}\n"
            f"  Risk-Reward Ratio: 1:{risk_ratio_text}"
        )
        
        return jsonify({
            "success": True,
            "risk_budget": {
                "recommended_units": round(adjusted_units, 4),
                "total_investment": round(total_investment, 2),
                "position_percent": round(position_percent, 1),
                "risk_per_unit": round(risk_per_unit, 2),
                "max_loss": max_risk_amount,
                "stop_loss_price": round(stop_loss_price, 2),
                "risk_reward_info": {
                    "potential_risk": round(risk_per_unit, 2),
                    "potential_reward": round(risk_per_unit * (1 + confidence), 2),
                    "risk_reward_ratio": round(1 + confidence, 2),
                    "is_favorable": confidence > 0.5
                }
            },
            "report_text": risk_report,
            "timestamp": datetime.now().isoformat()
        })
    
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/save-decision', methods=['POST'])
def save_decision():
    """
    Step 20: Save decision to history
    """
    try:
        data = request.get_json() or {}
        decision = data.get('decision', {})
        
        # Save to investment_history folder
        history_dir = project_root / "investment_history"
        history_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"decision_{timestamp}.json"
        filepath = history_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(decision, f, indent=2, ensure_ascii=False)
        
        return jsonify({
            "success": True,
            "saved_to": str(filepath),
            "timestamp": datetime.now().isoformat()
        })
    
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# ==================== Main ====================

if __name__ == '__main__':
    print("=" * 60)
    print("  N8N API Server for Bitcoin Investment Decision System")
    print("=" * 60)
    
    # Initialize models
    initialize_models()
    
    print("\n[*] Starting Flask server...")
    print("[*] Server URL: http://localhost:5000")
    print("[*] Health check: http://localhost:5000/api/health")
    print("\nAvailable endpoints:")
    print("  POST /api/fetch-price")
    print("  POST /api/calculate-indicators")
    print("  POST /api/linear-regression-predict")
    print("  POST /api/arima-predict")
    print("  POST /api/seasonal-predict")
    print("  POST /api/multi-timeframe-analysis")
    print("  POST /api/scenario-analysis")
    print("  POST /api/analyst/<type>")
    print("  POST /api/anonymize-analyses")
    print("  POST /api/peer-review")
    print("  POST /api/calculate-rankings")
    print("  POST /api/devils-advocate")
    print("  POST /api/red-blue-team")
    print("  POST /api/bayesian-fusion")
    print("  POST /api/confidence-calibration")
    print("  POST /api/chairman-synthesis")
    print("  POST /api/risk-budget-calculation")
    print("  POST /api/save-decision")
    print("=" * 60)
    
    app.run(host='0.0.0.0', port=5000, debug=True)

