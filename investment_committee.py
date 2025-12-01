# -*- coding: utf-8 -*-
"""
Intelligent Investment Decision System - Investment Committee Decision (Enhanced)
Multi-Agent Architecture for Bitcoin Investment Analysis

This system simulates an investment committee where multiple AI analysts
analyze Bitcoin investment opportunities from different perspectives,
and the investment manager makes the final decision based on all opinions.

Enhanced Features (v3.0):
- Analyst personalization system
- Adversarial debate phase
- Decision review analysis
- Historical decision tracking
- Command-line argument support
- Multi-timeframe analysis
- Scenario analysis engine
- Bayesian signal fusion
- Red-blue team validation
- Risk budget management
"""

import os
import sys
import json
import random
import time
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from colorama import Fore, Style, init

# Initialize colorama
init(autoreset=True)

# Load environment variables
project_root = Path(__file__).resolve().parent
load_dotenv(dotenv_path=project_root / ".env", override=True)

# Add Bitcoin prediction project path
btc_project_path = project_root / "Bitcoin price prediction Project"
sys.path.insert(0, str(btc_project_path))


# ==================== Configuration Class ====================

class InvestmentConfig:
    """Investment Decision System Configuration"""
    
    # API Configuration
    DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
    DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    MODEL_NAME = os.getenv("MODEL_NAME", "deepseek-chat")
    
    # Analyst Configuration
    ANALYST_CONFIG = {
        "technical_analyst": 1,      # Technical Analyst (Core)
        "industry_analyst": 1,       # Industry Analyst
        "financial_analyst": 1,      # Financial Analyst
        "market_expert": 1,          # Market Expert
        "risk_analyst": 1,           # Risk Analyst
        "investment_manager": 1,     # Investment Manager
    }
    
    # Role Display Names
    ANALYST_NAMES = {
        "technical_analyst": "Technical Analyst",
        "industry_analyst": "Industry Analyst",
        "financial_analyst": "Financial Analyst",
        "market_expert": "Market Expert",
        "risk_analyst": "Risk Analyst",
        "investment_manager": "Investment Manager",
    }
    
    # Role Descriptions
    ANALYST_DESCRIPTIONS = {
        "technical_analyst": """You are a senior technical analyst specializing in Bitcoin price trends and technical indicators.
Your responsibilities:
1. Analyze price prediction model outputs
2. Interpret technical indicators (RSI, MACD, Moving Averages, Bollinger Bands, etc.)
3. Identify price trends and key support/resistance levels
4. Provide trading signals based on technical analysis""",
        
        "industry_analyst": """You are a cryptocurrency industry analyst focusing on industry trends and fundamental research.
Your responsibilities:
1. Analyze overall cryptocurrency industry development trends
2. Assess regulatory policy impacts on the market
3. Track important industry news and events
4. Evaluate Bitcoin's competitive position in the industry""",
        
        "financial_analyst": """You are a financial analyst focusing on investment feasibility and risk-return analysis.
Your responsibilities:
1. Calculate and evaluate investment returns
2. Analyze risk-reward ratios
3. Provide capital management recommendations
4. Set reasonable stop-loss and take-profit levels""",
        
        "market_expert": """You are a market expert focusing on market sentiment and liquidity analysis.
Your responsibilities:
1. Analyze market sentiment indicators
2. Assess market liquidity conditions
3. Track movements of large institutions
4. Identify market fear and greed sentiment""",
        
        "risk_analyst": """You are a risk analyst focusing on identifying and quantifying investment risks.
Your responsibilities:
1. Identify various investment risks (market risk, liquidity risk, policy risk, etc.)
2. Quantify risk exposure
3. Propose risk mitigation measures
4. Set risk alert thresholds""",
        
        "investment_manager": """You are the investment manager of the committee, responsible for making final investment decisions.
Your responsibilities:
1. Listen to analysis from all analysts
2. Weigh different perspectives
3. Consider both risks and returns comprehensively
4. Make final investment decisions (buy/sell/hold)
5. Determine position sizes and execution strategies""",
    }
    
    # Decision Parameters
    TEMPERATURE = 0.7
    MAX_ANALYSIS_WORDS = 500
    MAX_DISCUSSION_WORDS = 300
    DECISION_OPTIONS = ["strong_buy", "buy", "hold", "sell", "strong_sell"]
    
    # Memory System
    ENABLE_LONG_TERM_MEMORY = True
    MEMORY_FILE_DIR = "investment_memory"
    HISTORY_FILE_DIR = "investment_history"
    MAX_HISTORY_RECORDS = 50
    
    @classmethod
    def validate(cls):
        """Validate configuration"""
        if not cls.DEEPSEEK_API_KEY:
            raise ValueError("[ERROR] DEEPSEEK_API_KEY not found! Please configure it in .env file.")
        return True


# ==================== API Client ====================

class DeepSeekClient:
    """DeepSeek API Client"""
    
    def __init__(self):
        from openai import OpenAI
        
        self.client = OpenAI(
            api_key=InvestmentConfig.DEEPSEEK_API_KEY,
            base_url=InvestmentConfig.DEEPSEEK_BASE_URL
        )
        self.model = InvestmentConfig.MODEL_NAME
    
    def chat(self, system_prompt: str, user_message: str, temperature: float = None) -> str:
        """Send chat request"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                temperature=temperature or InvestmentConfig.TEMPERATURE
            )
            return response.choices[0].message.content
        except Exception as e:
            error_str = str(e)
            if "402" in error_str or "Insufficient Balance" in error_str:
                raise Exception(f"[X] DeepSeek API insufficient balance! Please recharge.\nError: {error_str}")
            raise Exception(f"[X] DeepSeek API call failed: {error_str}")
    
    def chat_with_context(self, system_prompt: str, messages: list, temperature: float = None) -> str:
        """Chat request with context"""
        try:
            full_messages = [{"role": "system", "content": system_prompt}] + messages
            response = self.client.chat.completions.create(
                model=self.model,
                messages=full_messages,
                temperature=temperature or InvestmentConfig.TEMPERATURE
            )
            return response.choices[0].message.content
        except Exception as e:
            error_str = str(e)
            if "402" in error_str or "Insufficient Balance" in error_str:
                raise Exception(f"[X] DeepSeek API insufficient balance! Please recharge.\nError: {error_str}")
            raise Exception(f"[X] DeepSeek API call failed: {error_str}")


# ==================== Price Prediction Model Interface ====================

class BitcoinPredictor:
    """
    Bitcoin Price Prediction Model Interface - Multi-Model Ensemble
    
    Uses three prediction algorithms:
    1. Linear Regression (Elastic Net) - Direction accuracy 82.12%
    2. ARIMA(2,1,2) - Direction accuracy 54.20%
    3. Seasonal Model (Prophet-like) - Direction accuracy 53.47%
    """
    
    def __init__(self):
        """Initialize predictor, load ensemble models"""
        self.models_loaded = False
        self.core_predictor = None
        self.integrated_predictor = None
        
        # Try loading core prediction models first (no matplotlib dependency)
        self._load_core_predictor()
        
        # If core models fail, try loading original ensemble predictor
        if not self.models_loaded:
            self._load_integrated_predictor()
    
    def _load_core_predictor(self):
        """Load core prediction models (no matplotlib dependency)"""
        try:
            from core_prediction_models import EnsemblePredictorForAnalysts
            
            print(f"{Fore.CYAN}[*] Loading core prediction models...{Style.RESET_ALL}")
            self.core_predictor = EnsemblePredictorForAnalysts()
            
            # Load and train models
            success = self.core_predictor.load_and_train()
            
            if success:
                self.models_loaded = True
                print(f"{Fore.GREEN}[OK] Three-model prediction system loaded successfully!{Style.RESET_ALL}")
                print(f"{Fore.CYAN}    Model weights:{Style.RESET_ALL}")
                print(f"{Fore.CYAN}    - Linear Regression (Elastic Net): 50% weight, 82.12% accuracy{Style.RESET_ALL}")
                print(f"{Fore.CYAN}    - ARIMA(2,1,2): 30% weight, 54.20% accuracy{Style.RESET_ALL}")
                print(f"{Fore.CYAN}    - Seasonal (Prophet-like): 20% weight, 53.47% accuracy{Style.RESET_ALL}")
        except Exception as e:
            print(f"{Fore.YELLOW}[!] Core prediction model loading failed: {e}{Style.RESET_ALL}")
    
    def _load_integrated_predictor(self):
        """Load original ensemble predictor (requires matplotlib)"""
        try:
            from bitcoin_predictor_ensemble import BitcoinPredictorEnsemble
            self.integrated_predictor = BitcoinPredictorEnsemble()
            self.models_loaded = self.integrated_predictor.models_loaded
            if self.models_loaded:
                print(f"{Fore.GREEN}[OK] Backup ensemble predictor loaded successfully!{Style.RESET_ALL}")
        except Exception as e:
            print(f"{Fore.YELLOW}[!] Backup ensemble predictor loading failed: {e}{Style.RESET_ALL}")
            self.models_loaded = False
    
    def predict(self, price_data: dict = None) -> dict:
        """
        Comprehensive Bitcoin price prediction
        
        Returns:
            Prediction result dict with multi-model ensemble prediction
        """
        # Use core predictor first
        if self.core_predictor is not None and self.core_predictor.is_loaded:
            try:
                detailed_results = self.core_predictor.predict_with_details(price_data)
                return self._convert_to_standard_format(detailed_results, price_data)
            except Exception as e:
                print(f"{Fore.YELLOW}[!] Core prediction failed: {e}{Style.RESET_ALL}")
        
        # Use original ensemble predictor
        if self.integrated_predictor is not None and self.models_loaded:
            try:
                result = self.integrated_predictor.predict(price_data)
                return result
            except Exception as e:
                print(f"{Fore.YELLOW}[!] Ensemble prediction failed: {e}, using simulated prediction{Style.RESET_ALL}")
        
        # Simulated prediction
        return self._simulate_prediction(price_data.get('current_price', 45000) if price_data else 45000)
    
    def _convert_to_standard_format(self, detailed_results: dict, price_data: dict) -> dict:
        """Convert detailed prediction results to standard format"""
        current_price = price_data.get('current_price', 67500) if price_data else 67500
        
        ensemble = detailed_results.get('ensemble', {})
        models = detailed_results.get('models', {})
        
        predicted_price = ensemble.get('predicted_price', current_price)
        change_percent = ensemble.get('change_percent', 0)
        
        # Determine trend
        if change_percent > 5:
            trend = "strong_upward"
        elif change_percent > 2:
            trend = "upward"
        elif change_percent > -2:
            trend = "sideways"
        elif change_percent > -5:
            trend = "downward"
        else:
            trend = "strong_downward"
        
        # Build individual model prediction list
        individual_models = []
        for model_name, model_result in models.items():
            individual_models.append({
                'model': model_name,
                'predicted_price': model_result.get('predicted_price', current_price),
                'change_percent': model_result.get('change_percent', 0),
                'direction_accuracy': model_result.get('accuracy', 0.5),
                'weight': model_result.get('weight', 0.33)
            })
        
        return {
            "predicted_price": predicted_price,
            "current_price": current_price,
            "change_percent": change_percent,
            "confidence": ensemble.get('confidence', 0.5),
            "time_horizon": "7_days",
            "trend": trend,
            "trend_score": change_percent / 10,
            "model_used": "three_model_ensemble",
            "individual_models": individual_models,
            "model_consensus": ensemble.get('model_consensus', 'N/A'),
            "price_range": {
                "min": predicted_price * 0.95,
                "max": predicted_price * 1.05
            },
            "detailed_results": detailed_results  # Save detailed results for analyst use
        }
    
    def _simulate_prediction(self, current_price: float) -> dict:
        """Simulated prediction (when model unavailable)"""
        change_percent = random.uniform(-5, 10)
        predicted_price = current_price * (1 + change_percent / 100)
        
        if change_percent > 5:
            trend = "strong_upward"
        elif change_percent > 2:
            trend = "upward"
        elif change_percent > -2:
            trend = "sideways"
        elif change_percent > -5:
            trend = "downward"
        else:
            trend = "strong_downward"
        
        return {
            "predicted_price": predicted_price,
            "current_price": current_price,
            "change_percent": change_percent,
            "confidence": 0.5,
            "time_horizon": "7_days",
            "trend": trend,
            "trend_score": change_percent / 10,
            "model_used": "simulation",
            "individual_models": [],
            "price_range": {
                "min": predicted_price * 0.95,
                "max": predicted_price * 1.05
            }
        }
    
    def get_technical_indicators(self, price_data: dict = None) -> dict:
        """Get technical indicators analysis"""
        if self.integrated_predictor is not None:
            try:
                return self.integrated_predictor.get_technical_indicators(price_data)
            except:
                pass
        
        # Default values
        return {
            "rsi": {"value": 50, "signal": "neutral"},
            "macd": {"value": 0, "signal_line": 0, "histogram": 0, "signal": "neutral"},
            "moving_averages": {"sma_5": 0, "sma_20": 0, "signal": "neutral"},
            "bollinger_bands": {"upper": 0, "middle": 0, "lower": 0, "signal": "neutral"}
        }
    
    def get_model_comparison(self) -> dict:
        """Get model performance comparison"""
        if self.integrated_predictor is not None:
            try:
                return self.integrated_predictor.get_model_comparison()
            except:
                pass
        return {}
    
    def get_full_analysis_report(self, price_data: dict = None) -> str:
        """
        Get complete three-model analysis report
        For technical analyst use
        """
        if self.core_predictor is not None and self.core_predictor.is_loaded:
            try:
                return self.core_predictor.generate_full_analysis_report(price_data)
            except Exception as e:
                return f"[!] Failed to generate analysis report: {e}"
        
        return "[!] Prediction model not loaded, cannot generate analysis report"


# ==================== Memory Management System ====================

class InvestmentMemoryManager:
    """Investment Decision Memory Manager"""
    
    def __init__(self):
        self.memory_dir = project_root / InvestmentConfig.MEMORY_FILE_DIR
        self.history_dir = project_root / InvestmentConfig.HISTORY_FILE_DIR
        self.memory_dir.mkdir(exist_ok=True)
        self.history_dir.mkdir(exist_ok=True)
    
    def load_role_experience(self, role: str) -> str:
        """Load role historical experience"""
        memory_file = self.memory_dir / f"{role}_memory.json"
        
        if not memory_file.exists():
            return ""
        
        try:
            with open(memory_file, 'r', encoding='utf-8') as f:
                memories = json.load(f)
            
            if not memories:
                return ""
            
            # Get recent experience
            recent_memories = memories[-5:]
            experience_parts = []
            
            for mem in recent_memories:
                experience_parts.append(f"- {mem.get('date', 'Unknown')}: {mem.get('experience', '')}")
            
            return "\n".join(experience_parts)
        except Exception as e:
            print(f"Failed to load experience: {e}")
            return ""
    
    def save_decision_experience(self, role: str, experience: str, decision_id: str):
        """Save investment decision experience"""
        memory_file = self.memory_dir / f"{role}_memory.json"
        
        memories = []
        if memory_file.exists():
            try:
                with open(memory_file, 'r', encoding='utf-8') as f:
                    memories = json.load(f)
            except:
                memories = []
        
        memories.append({
            "decision_id": decision_id,
            "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "experience": experience
        })
        
        # Keep recent records
        if len(memories) > InvestmentConfig.MAX_HISTORY_RECORDS:
            memories = memories[-InvestmentConfig.MAX_HISTORY_RECORDS:]
        
        with open(memory_file, 'w', encoding='utf-8') as f:
            json.dump(memories, f, ensure_ascii=False, indent=2)
    
    def save_decision_record(self, decision_data: dict):
        """Save complete decision record"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        record_file = self.history_dir / f"decision_{timestamp}.json"
        
        with open(record_file, 'w', encoding='utf-8') as f:
            json.dump(decision_data, f, ensure_ascii=False, indent=2)
        
        return record_file


# ==================== Base Analyst Class ====================

class BaseAnalyst(ABC):
    """Base Analyst Class"""
    
    def __init__(self, analyst_id: int, role: str, historical_experience: str = ""):
        self.analyst_id = analyst_id
        self.role = role
        self.role_name = InvestmentConfig.ANALYST_NAMES.get(role, role)
        self.client = DeepSeekClient()
        self.historical_experience = historical_experience
        self.analysis_history = []
        self.recommendation = None
        self.system_prompt = self._build_system_prompt()
    
    def _build_system_prompt(self) -> str:
        """Build role system prompt"""
        base_prompt = f"""You are a professional investment analyst serving as {self.role_name} on the investment committee.

[Your Identity]
Analyst ID: {self.analyst_id}
Role: {self.role_name}

[Role Responsibilities]
{InvestmentConfig.ANALYST_DESCRIPTIONS.get(self.role, "")}

[Analysis Requirements]
1. Conduct objective analysis based on provided data
2. Provide clear opinions and reasoning
3. Analysis should not exceed {InvestmentConfig.MAX_ANALYSIS_WORDS} words
4. Give final investment recommendation: Strong Buy/Buy/Hold/Sell/Strong Sell"""

        if self.historical_experience:
            base_prompt += f"""

[Historical Experience Reference]
{self.historical_experience}

Please refer to historical experience but adjust analysis flexibly based on current market conditions."""

        return base_prompt
    
    def analyze(self, context: dict) -> str:
        """Conduct analysis"""
        prompt = self._build_analysis_prompt(context)
        response = self.client.chat(
            system_prompt=self.system_prompt,
            user_message=prompt,
            temperature=InvestmentConfig.TEMPERATURE
        )
        
        self.analysis_history.append({
            "timestamp": datetime.now().isoformat(),
            "context": context,
            "analysis": response
        })
        
        return response
    
    @abstractmethod
    def _build_analysis_prompt(self, context: dict) -> str:
        """Build analysis prompt (implemented by subclass)"""
        pass
    
    def discuss(self, discussion_context: dict) -> str:
        """Participate in discussion"""
        prompt = f"""This is the investment committee discussion phase.

[Other Analysts' Opinions]
{discussion_context.get('other_analyses', 'None')}

[Current Market Data]
{discussion_context.get('market_summary', 'None')}

Please share your views on other analysts' opinions. You may:
1. Add your analytical perspective
2. Question or support other viewpoints
3. Emphasize risks or opportunities you consider most important

Discussion should not exceed {InvestmentConfig.MAX_DISCUSSION_WORDS} words."""
        
        return self.client.chat(
            system_prompt=self.system_prompt,
            user_message=prompt,
            temperature=InvestmentConfig.TEMPERATURE
        )
    
    def make_recommendation(self, context: dict) -> dict:
        """Give investment recommendation"""
        prompt = f"""Based on your previous analysis, please give your final investment recommendation.

【market data summary】
{context.get('market_summary', 'None')}

【you analysis history】
{self.analysis_history[-1]['analysis'] if self.analysis_history else 'None'}

Please reply in the following format:
1. Investment recommendation: [strong buy/buy/hold/sell/strong sell]
2. Confidence index: [1-10 number, 10 is highest confidence]
3. Core reasoning: [One sentence summary]"""

        response = self.client.chat(
            system_prompt=self.system_prompt,
            user_message=prompt,
            temperature=0.5  # lower temperature for more stable output
        )
        
        # parse recommendation
        recommendation = self._parse_recommendation(response)
        self.recommendation = recommendation
        return recommendation
    
    def _parse_recommendation(self, response: str) -> dict:
        """parse investment recommendation"""
        # default value
        action = "hold"
        confidence = 5
        reasoning = response
        
        response_lower = response.lower()
        
        # parse investment recommendation
        if "strong buy" in response:
            action = "strong_buy"
        elif "strong sell" in response:
            action = "strong_sell"
        elif "buy" in response:
            action = "buy"
        elif "sell" in response:
            action = "sell"
        else:
            action = "hold"
        
        # parse confidence index
        import re
        confidence_match = re.search(r'confidence index[：:]\s*(\d+)', response)
        if confidence_match:
            confidence = int(confidence_match.group(1))
            confidence = max(1, min(10, confidence))
        
        return {
            "action": action,
            "confidence": confidence,
            "reasoning": reasoning,
            "analyst_id": self.analyst_id,
            "role": self.role
        }


# ==================== Prediction Results Formatting Tool ====================

def format_prediction_summary(prediction: dict, indicators: dict) -> str:
    """
    Format prediction results summary，for all analysts reference
    
    includeincluding：
    1. threeprediction model independentprediction results
    2. Ensemble Prediction Results
    3. technical indicators
    4. model performanceNote
    """
    # basic prediction data
    current_price = prediction.get('current_price', 0)
    predicted_price = prediction.get('predicted_price', 0)
    change_percent = prediction.get('change_percent', 0)
    confidence = prediction.get('confidence', 0)
    trend = prediction.get('trend', 'unknown')
    model_used = prediction.get('model_used', 'unknown')
    price_range = prediction.get('price_range', {})
    
    # individual model prediction details
    individual_models = prediction.get('individual_models', [])
    models_detail = ""
    
    if individual_models:
        if isinstance(individual_models, list):
            for model_pred in individual_models:
                model_name = model_pred.get('model', 'N/A')
                model_change = model_pred.get('change_percent', 0)
                model_price = model_pred.get('predicted_price', 0)
                model_accuracy = model_pred.get('direction_accuracy', 0)
                model_weight = model_pred.get('weight', 0)
                
                direction = "bullish" if model_change > 0 else ("bearish" if model_change < 0 else "neutral")
                models_detail += f"""
  [{model_name}]
    - prediction direction: {direction}
    - predicted change: {model_change:+.2f}%
    - predicted price: ${model_price:,.2f}
    - historical accuracy: {model_accuracy:.1%}
    - voting weight: {model_weight:.0%}"""
        elif isinstance(individual_models, dict):
            for model_name, model_pred in individual_models.items():
                if isinstance(model_pred, dict):
                    model_change = model_pred.get('change_percent', 0)
                    direction = "bullish" if model_change > 0 else ("bearish" if model_change < 0 else "neutral")
                    models_detail += f"\n  [{model_name}]: {direction} ({model_change:+.2f}%)"
    
    # parse technical indicators
    rsi_raw = indicators.get('rsi', 50)
    if isinstance(rsi_raw, dict):
        rsi_value = rsi_raw.get('value', 50)
    else:
        rsi_value = float(rsi_raw) if rsi_raw else 50
    
    rsi_signal = "overbought(sell signal)" if rsi_value > 70 else ("oversold(buy signal)" if rsi_value < 30 else "neutral")
    
    macd_raw = indicators.get('macd', {})
    if isinstance(macd_raw, dict):
        macd_value = macd_raw.get('value', 0)
        macd_signal_line = macd_raw.get('signal_line', macd_raw.get('signal', 0))
        macd_hist = macd_raw.get('histogram', 0)
    else:
        macd_value = 0
        macd_signal_line = 0
        macd_hist = 0
    
    # Ensure numeric types for MACD values
    def safe_float(val, default=0.0):
        """Safely convert value to float"""
        if val is None:
            return default
        if isinstance(val, (int, float)):
            return float(val)
        if isinstance(val, str):
            try:
                return float(val)
            except ValueError:
                return default
        return default
    
    macd_value = safe_float(macd_value)
    macd_signal_line = safe_float(macd_signal_line)
    macd_hist = safe_float(macd_hist)
    
    macd_signal = "bullish(MACD > Signal)" if macd_hist > 0 else "bearish(MACD < Signal)"
    
    ma_data = indicators.get('moving_averages', {})
    sma_5 = float(ma_data.get('sma_5', 0)) if isinstance(ma_data, dict) else 0
    sma_20 = float(ma_data.get('sma_20', 0)) if isinstance(ma_data, dict) else 0
    ma_signal = "short-term bullish(SMA5 > SMA20)" if sma_5 > sma_20 else "short-term bearish(SMA5 < SMA20)"
    
    bb_data = indicators.get('bollinger_bands', {})
    bb_upper = float(bb_data.get('upper', 0)) if isinstance(bb_data, dict) else 0
    bb_middle = float(bb_data.get('middle', 0)) if isinstance(bb_data, dict) else 0
    bb_lower = float(bb_data.get('lower', 0)) if isinstance(bb_data, dict) else 0
    
    # comprehensive trend judgment
    trend_cn = {
        "strong_upward": "strong upward",
        "upward": "moderate upward",
        "sideways": "sideways",
        "downward": "moderate downward",
        "strong_downward": "strong downward"
    }.get(trend, trend)
    
    return f"""
================================================================================
                    AI Prediction Model Analysis Report
================================================================================

I. Ensemble Prediction Results (three-model weighted average)
--------------------------------------------------------------------------------
  Current Price:     ${current_price:,.2f}
  Predicted Price:   ${predicted_price:,.2f}
  Predicted Change:  {change_percent:+.2f}%
  Trend Judgment:    {trend_cn}
  Overall Confidence:{confidence:.1%}
  Price Range:       ${price_range.get('min', 0):,.2f} ~ ${price_range.get('max', 0):,.2f}

II. Individual Prediction Model Results
--------------------------------------------------------------------------------{models_detail if models_detail else '''
  (Model details unavailable)'''}

III. Model Performance Reference (historical validation data)
--------------------------------------------------------------------------------
  +-----------------------------+--------------+--------+----------+
  | Model Name                  | Dir Accuracy | Weight | Use Case |
  +-----------------------------+--------------+--------+----------+
  | Linear Regression (best)    | 82.12%       | 50%    | Short-term|
  | ARIMA(2,1,2)                | 54.20%       | 30%    | Trend    |
  | Prophet                     | 53.47%       | 20%    | Seasonal |
  +-----------------------------+--------------+--------+----------+
  
  Note: Linear Regression has highest direction accuracy (82.12%), use as primary reference

IV. Technical Indicators Analysis
--------------------------------------------------------------------------------
  [RSI(14)] Relative Strength Index
    - Current value: {rsi_value:.2f}
    - Signal: {rsi_signal}
    - Interpretation: RSI > 70 indicates overbought, < 30 indicates oversold
  
  [MACD] Moving Average Convergence/Divergence
    - MACD value: {macd_value:.4f}
    - Signal Line: {macd_signal_line:.4f}
    - Histogram: {macd_hist:.4f}
    - Signal: {macd_signal}
  
  [Moving Averages]
    - SMA(5): {sma_5:.4f}
    - SMA(20): {sma_20:.4f}
    - Signal: {ma_signal}
  
  [Bollinger Bands]
    - Upper Band: {bb_upper:.4f}
    - Middle Band: {bb_middle:.4f}
    - Lower Band: {bb_lower:.4f}

V. Comprehensive Signal
--------------------------------------------------------------------------------
  Prediction Direction: {'bullish' if change_percent > 0 else ('bearish' if change_percent < 0 else 'neutral')}
  Signal Strength: {'strong' if abs(change_percent) > 5 else ('moderate' if abs(change_percent) > 2 else 'weak')}
  Reference: Use Linear Regression model (82.12% accuracy) as primary, combine with technical indicators

================================================================================
"""


# ==================== Professional Analyst Classes ====================

class TechnicalAnalyst(BaseAnalyst):
    """
    Technical Analyst - Actually uses three prediction models for analysis
    
    Integrated models:
    1. Linear Regression (Elastic Net) - direction accuracy 82.12%
    2. ARIMA(2,1,2) - direction accuracy 54.20%
    3. Seasonal (Prophet-like) - direction accuracy 53.47%
    """
    
    def __init__(self, analyst_id: int, predictor: BitcoinPredictor, historical_experience: str = ""):
        self.predictor = predictor
        super().__init__(analyst_id, "technical_analyst", historical_experience)
    
    def analyze(self, context: dict) -> str:
        """
        Override analysis method - actually uses three prediction models
        """
        prediction = context.get("prediction", {})
        
        # Get detailed analysis report from three models
        model_analysis_report = ""
        if self.predictor is not None:
            try:
                # Get complete three-model analysis report
                model_analysis_report = self.predictor.get_full_analysis_report(
                    {"current_price": prediction.get("current_price", 67500)}
                )
            except Exception as e:
                model_analysis_report = f"[!] Failed to get model analysis report: {e}"
        
        # Build analysis prompt
        prompt = self._build_analysis_prompt(context, model_analysis_report)
        
        # Call AI for analysis
        analysis = self.client.chat(
            system_prompt=self.system_prompt,
            user_message=prompt,
            temperature=InvestmentConfig.TEMPERATURE
        )
        
        # Record analysis history
        self.analysis_history.append({
            "context": str(context)[:500],
            "analysis": analysis,
            "timestamp": datetime.now().isoformat(),
            "model_report_included": bool(model_analysis_report)
        })
        
        return analysis
    
    def _build_analysis_prompt(self, context: dict, model_report: str = "") -> str:
        """Build technical analysis prompt - Deep reference to three prediction model results"""
        prediction = context.get("prediction", {})
        indicators = context.get("technical_indicators", {})
        
        # Get basic prediction summary
        prediction_summary = format_prediction_summary(prediction, indicators)
        
        # Get detailed predictions from each model
        individual_models = prediction.get("individual_models", [])
        model_details = ""
        if individual_models:
            model_details = "\n【Individual Model Prediction Details】\n"
            for model in individual_models:
                if isinstance(model, dict):
                    model_name = model.get('model', 'unknown')
                    pred_price = model.get('predicted_price', 0)
                    change_pct = model.get('change_percent', 0)
                    accuracy = model.get('direction_accuracy', 0)
                    weight = model.get('weight', 0)
                    direction = "bullish" if change_pct > 0 else ("bearish" if change_pct < 0 else "neutral")
                    
                    model_details += f"""
  [{model_name.upper()}]
    predicted price: ${pred_price:,.2f}
    predicted change: {change_pct:+.2f}%
    prediction direction: {direction}
    historical accuracy: {accuracy:.2%}
    ensemble weight: {weight:.0%}
"""
        
        # Get model consensus information
        model_consensus = prediction.get("model_consensus", "N/A")
        
        return f"""As the Technical Analyst，you can now usethree real prediction algorithms for analysis：

================================================================================
                    Real Prediction Model Output
================================================================================

{prediction_summary}
{model_details}

【model consensus】: {model_consensus}

================================================================================
                    Complete Model Analysis Report
================================================================================

{model_report if model_report else "(Model analysis report not generated)"}

================================================================================
                    Your Analysis Task
================================================================================

As the Technical Analyst，you have obtainedoutput results from three real prediction models：
- Linear Regression (Elastic Net): highest accuracy(82.12%)，primary reference
- ARIMA(2,1,2): time series model，specializes in trend analysis
- Seasonal (Prophet-like): seasonal model，captures cyclical patterns

Based on these real model predictions，conduct professional technical analysis：

1. [Three Model Comparison Analysis]
   - Do the three models agree on prediction direction?
   - How large is the difference in predicted prices?
   - Which model prediction is most trustworthy? Why?

2. [Model and Technical Indicator Cross-Validation]
   - Does the model prediction align with RSI, MACD and other technical indicators?
   - What are the key support/resistance levels?
   - Are there any divergence signals?

3. [Comprehensive Trading Recommendation]
   - Based on three-model ensemble prediction, should we buy, sell, or wait?
   - What are the recommended entry point, stop-loss, and take-profit levels?
   - What is the recommended position ratio?

4. [Risk Assessment]
   - How reliable is the model prediction?
   - What risk factors could cause prediction failure?

Please provide your professional analysis conclusion based on three real prediction models (not exceed 500 characters)."""


class IndustryAnalyst(BaseAnalyst):
    """Industry Analyst"""
    
    def __init__(self, analyst_id: int, historical_experience: str = ""):
        super().__init__(analyst_id, "industry_analyst", historical_experience)
    
    def _build_analysis_prompt(self, context: dict) -> str:
        """buildindustryanalysishintword - referenceprediction results"""
        prediction = context.get("prediction", {})
        indicators = context.get("technical_indicators", {})
        
        # getpredictionsummary
        prediction_summary = format_prediction_summary(prediction, indicators)
        
        current_price = prediction.get('current_price', 45000)
        change_percent = prediction.get('change_percent', 0)
        trend = prediction.get('trend', 'unknown')
        confidence = prediction.get('confidence', 0)
        
        trend_cn = {
            "strong_upward": "strong upward",
            "upward": "moderate upward",
            "sideways": "sideways",
            "downward": "moderate downward",
            "strong_downward": "strong downward"
        }.get(trend, trend)
        
        return f"""As the Industry Analyst, please reference AI prediction model results and analyze from industry fundamentals perspective:

{prediction_summary}

[Industry Background Information]
- Global cryptocurrency market cap is approximately $2.5 trillion
- Bitcoin market cap accounts for about 50%, still the cryptocurrency leader
- Institutional investors (like BlackRock, Fidelity) continue to enter
- Bitcoin ETF has been approved, bringing new fund inflow channels
- Regulatory policies in various countries are becoming clearer

[Your Analysis Task]

Based on prediction model showing {trend_cn} trend (change {change_percent:+.2f}%), please analyze:

1. [Industry Trend Validation]
   - Does current industry fundamentals support prediction model direction?
   - What recent industry events could affect Bitcoin price?
   - What are institutional investor movements? Any large-scale buy/sell signals?

2. [Policy Environment Analysis]
   - Any major regulatory policy changes recently?
   - Any changes in cryptocurrency stance by various countries?
   - Does policy environment favor prediction model trend judgment?

3. [Competitive Landscape]
   - Is Bitcoin's competitive advantage over other cryptocurrencies stable?
   - Any new technology or projects that could affect Bitcoin's position?

4. [Investment Recommendation]
   - From industry fundamentals perspective, do you agree with prediction model direction?
   - What is your investment recommendation (buy/hold/sell)?

Please provide your industry analysis conclusion (not exceed 500 characters)."""


class FinancialAnalyst(BaseAnalyst):
    """Financial Analyst"""
    
    def __init__(self, analyst_id: int, historical_experience: str = ""):
        super().__init__(analyst_id, "financial_analyst", historical_experience)
    
    def _build_analysis_prompt(self, context: dict) -> str:
        """buildfinancialanalysishintword - based onprediction resultscalculatereturnrisk"""
        prediction = context.get("prediction", {})
        indicators = context.get("technical_indicators", {})
        
        # getpredictionsummary
        prediction_summary = format_prediction_summary(prediction, indicators)
        
        current_price = prediction.get('current_price', 45000)
        predicted_price = prediction.get('predicted_price', 45000)
        change_percent = prediction.get('change_percent', 0)
        confidence = prediction.get('confidence', 0.5)
        price_range = prediction.get('price_range', {})
        
        # Calculate financial metrics
        investment_amount = 10000
        expected_return = investment_amount * change_percent / 100
        max_loss = investment_amount * 0.15  # Assume maximum drawdown 15%
        risk_reward_ratio = abs(change_percent / 15) if change_percent != 0 else 0
        
        # Annualized return (assuming 7-day prediction period)
        annualized_return = change_percent * (365 / 7)
        
        return f"""As the Financial Analyst, please conduct investment financial feasibility analysis based on AI prediction model results:

{prediction_summary}

[Financial Analysis Data]
================================================================================
Assumed Investment Amount: ${investment_amount:,.2f}

I. Return Prediction
  - Predicted Price Change: {change_percent:+.2f}%
  - Expected Return: ${expected_return:+,.2f}
  - Predicted Price Range: ${price_range.get('min', 0):,.2f} ~ ${price_range.get('max', 0):,.2f}
  - Best Case Return: ${investment_amount * (price_range.get('max', predicted_price) / current_price - 1):+,.2f}
  - Worst Case Return: ${investment_amount * (price_range.get('min', predicted_price) / current_price - 1):+,.2f}
  - Annualized Return (based on 7 days): {annualized_return:+.2f}%

II. Risk Assessment
  - Model Confidence: {confidence:.1%}
  - Assumed Maximum Drawdown: 15%
  - Maximum Potential Loss: ${max_loss:,.2f}
  - Risk-Return Ratio: 1:{risk_reward_ratio:.2f}

III. Stop-Loss/Take-Profit Recommendation
  - Recommended Stop-Loss Price: ${current_price * 0.95:,.2f} (-5%)
  - Recommended Take-Profit Price: ${current_price * (1 + abs(change_percent) * 1.5 / 100):,.2f} (+{abs(change_percent) * 1.5:.2f}%)
================================================================================

[Your Analysis Task]

Based on the above prediction model results and financial data, please analyze:

1. [Return Analysis]
   - Is the predicted {change_percent:+.2f}% return attractive?
   - Compared to other investment channels, how is the risk-adjusted return?
   - Is the annualized return of {annualized_return:+.2f}% reasonable?

2. [Risk Assessment]
   - Is the model confidence of {confidence:.1%} high enough?
   - Is the risk-return ratio of 1:{risk_reward_ratio:.2f} worth investing?
   - Is the maximum drawdown risk acceptable?

3. [Position Recommendation]
   - Based on risk tolerance, what position size do you recommend (0-100%)?
   - Do you recommend building positions in batches? How to allocate?

4. [Stop-Loss/Take-Profit Strategy]
   - What stop-loss point do you recommend?
   - What take-profit strategy do you recommend (one-time vs. batch)?

Please provide your financial analysis conclusion and specific recommendations (not exceed 500 characters)."""


class MarketExpert(BaseAnalyst):
    """Market Expert"""
    
    def __init__(self, analyst_id: int, historical_experience: str = ""):
        super().__init__(analyst_id, "market_expert", historical_experience)
    
    def _build_analysis_prompt(self, context: dict) -> str:
        """buildmarketanalysishintword - combineprediction resultsanalysismarket sentiment"""
        prediction = context.get("prediction", {})
        indicators = context.get("technical_indicators", {})
        
        # getpredictionsummary
        prediction_summary = format_prediction_summary(prediction, indicators)
        
        change_percent = prediction.get('change_percent', 0)
        confidence = prediction.get('confidence', 0.5)
        trend = prediction.get('trend', 'unknown')
        
        # parseRSI
        rsi_raw = indicators.get('rsi', 50)
        if isinstance(rsi_raw, dict):
            rsi = rsi_raw.get('value', 50)
        else:
            rsi = float(rsi_raw) if rsi_raw else 50
        
        # Judge market sentiment based on RSI
        if rsi > 70:
            sentiment = "Extreme Greed"
            sentiment_advice = "Market overheated, beware of pullback risk"
        elif rsi > 60:
            sentiment = "Greed"
            sentiment_advice = "Market sentiment optimistic, but stay vigilant"
        elif rsi > 40:
            sentiment = "Neutral"
            sentiment_advice = "Market sentiment stable, wait and see"
        elif rsi > 30:
            sentiment = "Fear"
            sentiment_advice = "Market sentiment slightly bearish, may be buying opportunity"
        else:
            sentiment = "Extreme Fear"
            sentiment_advice = "Panic selling, contrarian investment opportunity"
        
        # Prediction direction and market sentiment consistency
        model_direction = "bullish" if change_percent > 0 else ("bearish" if change_percent < 0 else "neutral")
        sentiment_direction = "bullish" if rsi < 50 else "bearish"  # Contrarian sentiment
        consistency = "consistent" if model_direction == sentiment_direction else "divergent"
        
        return f"""As the Market Expert, please analyze current market sentiment and investment opportunities based on AI prediction model results:

{prediction_summary}

[Market Sentiment Analysis]
================================================================================
  RSI Indicator:        {rsi:.2f}
  Sentiment Judgment:   {sentiment}
  Sentiment Advice:     {sentiment_advice}
  
  Model Prediction:     {model_direction} ({change_percent:+.2f}%)
  Contrarian Indicator: {sentiment_direction}
  Direction Consistency: {consistency}
================================================================================

[Your Analysis Task]

Based on prediction model showing {model_direction} trend, combined with market sentiment analysis:

1. [Market Sentiment Interpretation]
   - With RSI={rsi:.2f}, market is in "{sentiment}" state, how do you interpret this?
   - Prediction model shows {model_direction}, {consistency} with market sentiment, what does this mean?
   - Are there contrarian investment opportunities?

2. [Fund Flow Analysis]
   - Based on market sentiment, what might large funds be doing?
   - What are retail and institutional behaviors?
   - Any abnormal fund flow signals?

3. [Market Rhythm Judgment]
   - What phase is the market in (accumulation, rising, distribution, decline)?
   - Does prediction model trend judgment fit market rhythm?
   - What is the best entry/exit timing?

4. [Investment Recommendation]
   - Combining market sentiment and model prediction, what is your investment recommendation?
   - If divergent from model prediction, which do you trust more? Why?

Please provide your market analysis conclusion (not exceed 500 characters)."""


class RiskAnalyst(BaseAnalyst):
    """Risk Analyst"""
    
    def __init__(self, analyst_id: int, historical_experience: str = ""):
        super().__init__(analyst_id, "risk_analyst", historical_experience)
    
    def _build_analysis_prompt(self, context: dict) -> str:
        """buildriskanalysishintword - based onprediction resultsassessmentrisk"""
        prediction = context.get("prediction", {})
        indicators = context.get("technical_indicators", {})
        
        # Get prediction summary
        prediction_summary = format_prediction_summary(prediction, indicators)
        
        change_percent = prediction.get('change_percent', 0)
        confidence = prediction.get('confidence', 0.5)
        trend = prediction.get('trend', 'unknown')
        price_range = prediction.get('price_range', {})
        current_price = prediction.get('current_price', 45000)
        
        # Calculate risk metrics
        prediction_uncertainty = 1 - confidence
        price_volatility = (price_range.get('max', current_price) - price_range.get('min', current_price)) / current_price * 100
        
        # Model consistency risk
        individual_models = prediction.get('individual_models', [])
        if isinstance(individual_models, list) and len(individual_models) >= 2:
            directions = [m.get('change_percent', 0) > 0 for m in individual_models]
            model_consistency = sum(directions) / len(directions)
            consistency_risk = "low" if model_consistency in [0, 1] else ("medium" if model_consistency in [0.33, 0.67] else "high")
        else:
            consistency_risk = "unknown"
        
        return f"""As the Risk Analyst, please conduct comprehensive risk assessment based on AI prediction model results:

{prediction_summary}

[Risk Assessment Data]
================================================================================
I. Model Risk
  - Model Confidence: {confidence:.1%}
  - Prediction Uncertainty: {prediction_uncertainty:.1%}
  - Model Consistency Risk: {consistency_risk}
  
II. Price Risk
  - Predicted Change: {change_percent:+.2f}%
  - Price Volatility Range: {price_volatility:.2f}%
  - Maximum Potential Decline: {abs(min(change_percent, -15)):.2f}%
  
III. Identified Risk Factors
  1. Model Risk: Prediction models have inherent errors, historical accuracy doesn't represent future
  2. Market Risk: Cryptocurrency market volatility is high, extreme conditions may exceed prediction range
  3. Liquidity Risk: Liquidity may dry up during severe volatility
  4. Policy Risk: Regulatory policy changes may cause severe market volatility
  5. Technical Risk: Network attacks, exchange failures and other technical issues
  6. Black Swan Risk: Unpredictable sudden events
================================================================================

[Your Analysis Task]

Based on prediction model showing {trend} trend (confidence {confidence:.1%}), please conduct risk assessment:

1. [Model Risk Assessment]
   - Is model confidence of {confidence:.1%} sufficiently reliable?
   - Are there significant divergences between three models? What does this mean?
   - Where do you think the main risk points of prediction model are?

2. [Market Risk Assessment]
   - In current market environment, how high is the probability of prediction failure?
   - If prediction is wrong, what is the maximum potential loss?
   - Are there high-probability scenarios that could trigger stop-loss?

3. [Extreme Scenario Analysis]
   - If a black swan event occurs, what is the worst case scenario?
   - What is the recommended risk exposure limit?
   - Is there a need to set up risk warning mechanisms?

4. [Risk Mitigation Recommendations]
   - How to reduce investment risk? (Position control, stop-loss settings, etc.)
   - Do you recommend hedging strategies?
   - What is your risk rating (low/medium/high/extremely high)?

Please provide your risk assessment conclusion (not exceed 500 characters).
Note: As the Risk Analyst, you should remain cautious - better to overestimate risk than underestimate it."""


class InvestmentManager(BaseAnalyst):
    """Investment Manager - Makes Final Decision"""
    
    def __init__(self, analyst_id: int, historical_experience: str = ""):
        super().__init__(analyst_id, "investment_manager", historical_experience)
    
    def _build_analysis_prompt(self, context: dict) -> str:
        """Build Investment Manager analysis prompt"""
        return f"""As the Investment Manager, please provide preliminary judgment based on comprehensive analysis:

[Market Data]
- Current Price: ${context.get('prediction', {}).get('current_price', 45000):,.2f}
- Predicted Price: ${context.get('prediction', {}).get('predicted_price', 45000):,.2f}
- Predicted Change: {context.get('prediction', {}).get('change_percent', 0):.2f}%

Please provide your preliminary judgment. You will make the final decision after hearing from all analysts."""
    
    def make_final_decision(self, state: 'InvestmentState') -> dict:
        """Make final decision based on all analyses"""
        # Collect all analysis recommendations
        all_recommendations = state.get_all_recommendations()
        all_analyses = state.get_all_analyses()
        
        # builddecisionhintword
        recommendations_text = "\n".join([
            f"- {InvestmentConfig.ANALYST_NAMES.get(rec['role'], rec['role'])}: "
            f"{rec['action']} (confidence: {rec['confidence']}/10)"
            for rec in all_recommendations
        ])
        
        analyses_summary = "\n\n".join([
            f"【{InvestmentConfig.ANALYST_NAMES.get(analysis['role'], analysis['role'])}】\n{analysis['content'][:300]}..."
            for analysis in all_analyses
        ])
        
        prompt = f"""As Investment Manager, please make final investment decision based on all analysts' input.

[All Analyst Recommendations Summary]
{recommendations_text}

[All Analyst Analysis Summary]
{analyses_summary}

[Market Data]
- Current Price: ${state.prediction.get('current_price', 45000):,.2f}
- Predicted Price: ${state.prediction.get('predicted_price', 45000):,.2f}
- Predicted Change: {state.prediction.get('change_percent', 0):.2f}%

Please make final decision and reply in the following format:

[Final Decision]
Investment Recommendation: [strong buy/buy/hold/sell/strong sell]
Recommended Position: [0-100%]
Confidence Index: [1-10]

[Decision Reasoning]
[Comprehensive reasoning considering all perspectives, within 200 characters]

[Execution Recommendation]
- Entry Price: $[price]
- Stop-Loss Price: $[price]
- Take-Profit Price: $[price]"""

        response = self.client.chat(
            system_prompt=self.system_prompt,
            user_message=prompt,
            temperature=0.5
        )
        
        # parseFinal Decision
        decision = self._parse_final_decision(response, state)
        return decision
    
    def _parse_final_decision(self, response: str, state: 'InvestmentState') -> dict:
        """parseFinal Decision"""
        import re
        
        # default value
        action = "hold"
        position = 0
        confidence = 5
        
        # parse investment recommendation
        if "strong buy" in response:
            action = "strong_buy"
            position = 80
        elif "strong sell" in response:
            action = "strong_sell"
            position = 0
        elif "buy" in response:
            action = "buy"
            position = 50
        elif "sell" in response:
            action = "sell"
            position = 20
        else:
            action = "hold"
            position = 30
        
        # Parse position
        position_match = re.search(r'Recommended Position[：:]\s*(\d+)', response, re.IGNORECASE)
        if position_match:
            position = int(position_match.group(1))
            position = max(0, min(100, position))
        
        # parse confidence index
        confidence_match = re.search(r'confidence index[：:]\s*(\d+)', response)
        if confidence_match:
            confidence = int(confidence_match.group(1))
            confidence = max(1, min(10, confidence))
        
        return {
            "action": action,
            "action_cn": self._action_to_chinese(action),
            "position": position,
            "confidence": confidence,
            "reasoning": response,
            "timestamp": datetime.now().isoformat(),
            "prediction": state.prediction,
            "analyst_recommendations": state.get_all_recommendations()
        }
    
    def _action_to_chinese(self, action: str) -> str:
        """Convert action to display text"""
        mapping = {
            "strong_buy": "strong buy",
            "buy": "buy",
            "hold": "hold",
            "sell": "sell",
            "strong_sell": "strong sell"
        }
        return mapping.get(action, "hold")


# ==================== Investment State Management ====================

class InvestmentState:
    """Investment decision state management (corresponds to GameState)"""
    
    def __init__(self, analysts: Dict[int, BaseAnalyst]):
        self.analysts = analysts
        self.current_round = 0
        self.analyses = []
        self.recommendations = []
        self.discussions = []
        self.final_decision = None
        self.prediction = {}
        self.technical_indicators = {}
        self.start_time = datetime.now()
    
    def record_analysis(self, analyst_id: int, role: str, analysis: str):
        """Record analyst analysis"""
        self.analyses.append({
            "analyst_id": analyst_id,
            "role": role,
            "content": analysis,
            "timestamp": datetime.now().isoformat()
        })
    
    def record_recommendation(self, recommendation: dict):
        """recordinvestment recommendation"""
        self.recommendations.append(recommendation)
    
    def record_discussion(self, analyst_id: int, role: str, discussion: str):
        """recorddiscussionspeak"""
        self.discussions.append({
            "analyst_id": analyst_id,
            "role": role,
            "content": discussion,
            "timestamp": datetime.now().isoformat()
        })
    
    def record_final_decision(self, decision: dict):
        """recordFinal Decision"""
        self.final_decision = decision
    
    def get_all_analyses(self) -> List[dict]:
        """gethasanalysis"""
        return self.analyses
    
    def get_all_recommendations(self) -> List[dict]:
        """gethasrecommendation"""
        return self.recommendations
    
    def get_analyses_summary(self) -> str:
        """getanalysissummary"""
        if not self.analyses:
            return "Noneanalysis"
        
        summary_parts = []
        for analysis in self.analyses:
            role_name = InvestmentConfig.ANALYST_NAMES.get(analysis['role'], analysis['role'])
            summary_parts.append(f"【{role_name}】\n{analysis['content'][:200]}...")
        
        return "\n\n".join(summary_parts)
    
    def export_to_dict(self) -> dict:
        """Export state as dictionary (safe serialization, avoid circular reference)"""
        import copy
        
        def safe_serialize(obj, seen=None):
            """Safe serialization, prevent circular reference"""
            if seen is None:
                seen = set()
            
            obj_id = id(obj)
            if obj_id in seen:
                return "[Circular Reference]"
            
            if isinstance(obj, dict):
                seen.add(obj_id)
                return {k: safe_serialize(v, seen.copy()) for k, v in obj.items()}
            elif isinstance(obj, list):
                seen.add(obj_id)
                return [safe_serialize(item, seen.copy()) for item in obj]
            elif isinstance(obj, (str, int, float, bool, type(None))):
                return obj
            elif hasattr(obj, '__dict__'):
                return str(obj)
            else:
                try:
                    return str(obj)
                except:
                    return "[Unserializable]"
        
        # Create safe copy of final decision
        safe_final_decision = None
        if self.final_decision:
            safe_final_decision = {
                "action": self.final_decision.get("action"),
                "action_cn": self.final_decision.get("action_cn"),
                "position": self.final_decision.get("position"),
                "confidence": self.final_decision.get("confidence"),
                "reasoning": self.final_decision.get("reasoning", "")[:500],  # Truncate long text
                "timestamp": self.final_decision.get("timestamp"),
            }
        
        return {
            "round": self.current_round,
            "start_time": self.start_time.isoformat(),
            "end_time": datetime.now().isoformat(),
            "prediction": safe_serialize(self.prediction),
            "technical_indicators": safe_serialize(self.technical_indicators),
            "analyses": safe_serialize(self.analyses),
            "recommendations": safe_serialize(self.recommendations),
            "discussions": safe_serialize(self.discussions),
            "final_decision": safe_final_decision
        }


# ==================== Investment Committeemainprocess ====================

class InvestmentCommittee:
    """Investment Committeemainprocess（forshould WerewolfGame）"""
    
    def __init__(self):
        self.analysts: Dict[int, BaseAnalyst] = {}
        self.state: Optional[InvestmentState] = None
        self.predictor = BitcoinPredictor()
        self.memory_manager = InvestmentMemoryManager()
    
    def print_section(self, title: str, color: str = Fore.YELLOW):
        """Print section title"""
        separator = "=" * 78
        print(f"\n{color}+{separator}+")
        print(f"|{title:^78}|")
        print(f"+{separator}+{Style.RESET_ALL}\n")
    
    def print_info(self, message: str, color: str = Fore.WHITE):
        """printinformation"""
        print(f"{color}{message}{Style.RESET_ALL}")
    
    def initialize_committee(self):
        """initializationInvestment Committee"""
        self.print_section("[Committee] Investment Committee Initialization", Fore.CYAN)
        
        InvestmentConfig.validate()
        self.print_info("[OK] Configuration validation passed", Fore.GREEN)
        
        analyst_id = 1
        
        # Create Technical Analyst (core role)
        experience = self.memory_manager.load_role_experience("technical_analyst")
        self.analysts[analyst_id] = TechnicalAnalyst(analyst_id, self.predictor, experience)
        self.print_info(f"  Analyst {analyst_id} -> Technical Analyst [Chart]", Fore.CYAN)
        analyst_id += 1
        
        # Create other analysts
        analyst_classes = [
            ("industry_analyst", IndustryAnalyst),
            ("financial_analyst", FinancialAnalyst),
            ("market_expert", MarketExpert),
            ("risk_analyst", RiskAnalyst),
        ]
        
        for role, AnalystClass in analyst_classes:
            experience = self.memory_manager.load_role_experience(role)
            self.analysts[analyst_id] = AnalystClass(analyst_id, experience)
            self.print_info(f"  Analyst {analyst_id} -> {InvestmentConfig.ANALYST_NAMES[role]}", Fore.CYAN)
            analyst_id += 1
        
        # Create Investment Manager
        experience = self.memory_manager.load_role_experience("investment_manager")
        self.analysts[analyst_id] = InvestmentManager(analyst_id, experience)
        self.print_info(f"  Analyst {analyst_id} -> Investment Manager [Manager]", Fore.YELLOW)
        
        self.state = InvestmentState(self.analysts)
        self.print_info("\n[OK] Investment Committee initialization completed!", Fore.GREEN)
        time.sleep(1)
    
    def run_decision_process(self, price_data: dict = None):
        """Run complete decision process"""
        try:
            # Phase 1: Data preparation
            context = self._prepare_context(price_data)
            
            # phase2：independentanalysis
            self._independent_analysis_phase(context)
            
            # phase3：discussionphase
            self._discussion_phase()
            
            # phase4：votingrecommendation
            self._recommendation_phase(context)
            
            # phase5：Final Decision
            self._final_decision_phase()
            
            # phase6：savingrecord
            self._save_decision_record()
            
            return self.state.final_decision
            
        except Exception as e:
            self.print_info(f"[X] Decision process error: {e}", Fore.RED)
            raise
    
    def _prepare_context(self, price_data: dict = None) -> dict:
        """Prepare analysis context"""
        self.print_section("[Chart] Data Preparation Phase", Fore.BLUE)
        
        # Run price prediction (ensemble prediction will automatically use trained data)
        self.print_info("[*] Running multi-model ensemble prediction...", Fore.YELLOW)
        prediction = self.predictor.predict(price_data)
        self.state.prediction = prediction
        
        # showprediction results
        current_price = prediction.get('current_price', 0)
        self.print_info(f"  Current Price: {current_price:.6f}", Fore.WHITE)
        self.print_info(f"  Ensemble Prediction Trend: {prediction.get('trend', 'N/A')}", Fore.WHITE)
        self.print_info(f"  Trend Score: {prediction.get('trend_score', 0):.4f}", Fore.WHITE)
        self.print_info(f"  Predicted Change: {prediction.get('change_percent', 0):.2f}%", Fore.WHITE)
        self.print_info(f"  Overall Confidence: {prediction.get('confidence', 0):.2%}", Fore.WHITE)
        
        # Show each model prediction
        individual_models = prediction.get('individual_models', [])
        if individual_models:
            self.print_info("\n  Individual model predictions:", Fore.CYAN)
            # Handle both list and dict formats
            if isinstance(individual_models, list):
                for model_pred in individual_models:
                    model_name = model_pred.get('model', 'Unknown')
                    change = model_pred.get('change_percent', 0)
                    accuracy = model_pred.get('direction_accuracy', 0)
                    self.print_info(f"    {model_name}: {change:+.2f}% (accuracy: {accuracy:.1%})", Fore.WHITE)
            elif isinstance(individual_models, dict):
                for model_name, model_pred in individual_models.items():
                    self.print_info(f"    {model_name}: {model_pred.get('trend', 'N/A')} (confidence: {model_pred.get('confidence', 0):.0%})", Fore.WHITE)
        
        # Get technical indicators
        self.print_info("\n[+] Getting Technical Indicators Analysis...", Fore.YELLOW)
        technical_indicators = self.predictor.get_technical_indicators(price_data)
        self.state.technical_indicators = technical_indicators
        
        # Show technical indicators
        rsi_info = technical_indicators.get('rsi', 50)
        macd_info = technical_indicators.get('macd', {})
        
        # Handle different RSI return value formats
        if isinstance(rsi_info, dict):
            rsi_value = rsi_info.get('value', 50)
            rsi_signal = rsi_info.get('signal', 'N/A')
        else:
            rsi_value = float(rsi_info) if rsi_info else 50
            rsi_signal = "overbought" if rsi_value > 70 else ("oversold" if rsi_value < 30 else "neutral")
        
        # Handle different MACD return value formats
        if isinstance(macd_info, dict):
            macd_signal = macd_info.get('signal', macd_info.get('histogram', 0))
            if isinstance(macd_signal, (int, float)):
                macd_signal = "bullish" if macd_signal > 0 else "bearish"
        else:
            macd_signal = "N/A"
        
        self.print_info(f"  RSI: {rsi_value:.2f} - {rsi_signal}", Fore.WHITE)
        self.print_info(f"  MACD: {macd_signal}", Fore.WHITE)
        
        context = {
            "price_data": price_data,
            "prediction": prediction,
            "technical_indicators": technical_indicators,
            "market_summary": f"trend: {prediction.get('trend', 'N/A')}, "
                            f"trend score: {prediction.get('trend_score', 0):.4f}, "
                            f"predicted change: {prediction.get('change_percent', 0):.2f}%"
        }
        
        self.print_info("\n[OK] Data preparation completed!", Fore.GREEN)
        time.sleep(1)
        
        return context
    
    def _independent_analysis_phase(self, context: dict):
        """Independent analysis phase"""
        self.print_section("[Search] Independent Analysis Phase", Fore.MAGENTA)
        
        for analyst_id, analyst in self.analysts.items():
            if analyst.role == "investment_manager":
                continue  # Investment Manager does not participate in independent analysis
            
            self.print_info(f"\n{analyst.role_name} is analyzing...", Fore.CYAN)
            
            analysis = analyst.analyze(context)
            self.state.record_analysis(analyst_id, analyst.role, analysis)
            
            # Show analysis results summary
            self.print_info(f"{Fore.WHITE}{analysis[:300]}...{Style.RESET_ALL}")
            time.sleep(0.5)
        
        self.print_info("\n[OK] independentanalysisphasecompleted!", Fore.GREEN)
        time.sleep(1)
    
    def _discussion_phase(self):
        """discussionphase"""
        self.print_section("[Talk] discussionphase", Fore.YELLOW)
        
        # Build discussion context
        discussion_context = {
            "other_analyses": self.state.get_analyses_summary(),
            "market_summary": f"current price: ${self.state.prediction.get('current_price', 0):,.2f}, "
                            f"predicted change: {self.state.prediction.get('change_percent', 0):.2f}%"
        }
        
        for analyst_id, analyst in self.analysts.items():
            if analyst.role == "investment_manager":
                continue
            
            self.print_info(f"\n{analyst.role_name} speak:", Fore.CYAN)
            
            discussion = analyst.discuss(discussion_context)
            self.state.record_discussion(analyst_id, analyst.role, discussion)
            
            self.print_info(f"{Fore.WHITE}{discussion[:200]}...{Style.RESET_ALL}")
            time.sleep(0.5)
        
        self.print_info("\n[OK] discussionphasecompleted!", Fore.GREEN)
        time.sleep(1)
    
    def _recommendation_phase(self, context: dict):
        """votingrecommendationphase"""
        self.print_section("[Vote] votingrecommendationphase", Fore.BLUE)
        
        for analyst_id, analyst in self.analysts.items():
            if analyst.role == "investment_manager":
                continue
            
            self.print_info(f"\n{analyst.role_name} providerecommendation...", Fore.CYAN)
            
            recommendation = analyst.make_recommendation(context)
            self.state.record_recommendation(recommendation)
            
            action_cn = {
                "strong_buy": "strong buy",
                "buy": "buy",
                "hold": "hold",
                "sell": "sell",
                "strong_sell": "strong sell"
            }.get(recommendation['action'], "hold")
            
            self.print_info(f"  recommendation: {action_cn} | confidence: {recommendation['confidence']}/10", Fore.WHITE)
            time.sleep(0.3)
        
        self.print_info("\n[OK] votingrecommendationphasecompleted!", Fore.GREEN)
        time.sleep(1)
    
    def _final_decision_phase(self):
        """Final Decisionphase"""
        self.print_section("[Manager] Final Decisionphase", Fore.RED)
        
        # findtoInvestment Manager
        investment_manager = None
        for analyst in self.analysts.values():
            if analyst.role == "investment_manager":
                investment_manager = analyst
                break
        
        if investment_manager is None:
            raise ValueError("not yetfindtoInvestment Manager!")
        
        self.print_info("Investment Manager is synthesizing all perspectives...", Fore.YELLOW)
        
        final_decision = investment_manager.make_final_decision(self.state)
        self.state.record_final_decision(final_decision)
        
        # showFinal Decision
        self.print_section("[Report] finalinvestmentdecision", Fore.GREEN)
        
        self.print_info(f"investment recommendation: {final_decision['action_cn']}", Fore.YELLOW)
        self.print_info(f"recommendationposition: {final_decision['position']}%", Fore.WHITE)
        self.print_info(f"confidence index: {final_decision['confidence']}/10", Fore.WHITE)
        self.print_info(f"\nDecision Reasoning:", Fore.CYAN)
        self.print_info(f"{final_decision['reasoning'][:500]}", Fore.WHITE)
        
        time.sleep(1)
    
    def _save_decision_record(self):
        """savingdecisionrecord"""
        self.print_section("[Save] savingdecisionrecord", Fore.CYAN)
        
        decision_data = self.state.export_to_dict()
        record_file = self.memory_manager.save_decision_record(decision_data)
        
        self.print_info(f"[OK] decisionrecordalreadysaving: {record_file}", Fore.GREEN)
        
        # aseachanalysissavingexperience
        decision_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for analyst_id, analyst in self.analysts.items():
            if analyst.recommendation:
                experience = f"recommendation{analyst.recommendation['action']}，confidence{analyst.recommendation['confidence']}/10"
                self.memory_manager.save_decision_experience(
                    analyst.role, experience, decision_id
                )
        
        self.print_info("[OK] analysisexperiencealreadysaving!", Fore.GREEN)


# ==================== increasestrongfunction：debatephase + advancedfunction ====================

class EnhancedInvestmentCommittee(InvestmentCommittee):
    """
    increasestrongversionInvestment Committee v3.0
    
    Core Features:
    1. Analyst personalization display
    2. Adversarial debate phase
    3. Decision review analysis
    
    Advanced Features:
    4. Multi-timeframe analysis - Short/Medium/Long-term 3D analysis
    5. Scenario analysis engine - Bull/Bear/Volatile/Black Swan scenario simulation
    6. Bayesiansignal fusion - canfitmanysignalsource
    7. Red-blue team validation - Decision quality adversarial validation
    8. riskbudgetmanagement - based onrisk positioncalculate
    9. Confidence calibration - Calibrate overconfidence
    """
    
    def __init__(self, enable_debate: bool = True, enable_analysis: bool = True,
                 enable_advanced: bool = True):
        super().__init__()
        self.enable_debate = enable_debate
        self.enable_analysis = enable_analysis
        self.enable_advanced = enable_advanced
        self.debate_manager = None
        
        # advancedfunctionmodule
        self.mtf_analyzer = None  # Multi-timeframe analyzer
        self.scenario_engine = None  # Scenario analysis engine
        self.signal_fusion = None  # signal fusion
        self.red_blue_validator = None  # red-blue team
        self.risk_manager = None  # riskmanagement
        self.confidence_calibrator = None  # Confidence calibrator
    
    def initialize_committee(self):
        """increasestrongversioninitialization - showanalysis"""
        super().initialize_committee()
        
        # showanalysispersonalizationinformation
        try:
            from enhanced_analysts import ANALYST_PERSONALITIES
            
            self.print_section("[Personalities] Analyst Team Introduction", Fore.MAGENTA)
            
            for analyst_id, analyst in self.analysts.items():
                personality = ANALYST_PERSONALITIES.get(analyst.role, {})
                if personality:
                    print(f"{Fore.CYAN}  [{analyst_id}] {personality.get('name', analyst.role_name)}")
                    print(f"{Fore.WHITE}      nickname: {personality.get('nickname', 'N/A')}")
                    print(f"{Fore.WHITE}      style: {personality.get('style', 'N/A')}")
                    print(f"{Fore.WHITE}      catchphrase: \"{personality.get('catchphrase', 'N/A')}\"{Style.RESET_ALL}")
            
            self.print_info("\n[OK] Analyst team is ready!", Fore.GREEN)
            
        except ImportError:
            pass
    
    def run_decision_process(self, price_data: dict = None):
        """Enhanced decision process v3.0 with Anonymous Peer Review"""
        try:
            # ========== onephase：data preparation ==========
            context = self._prepare_context(price_data)
            
            # ========== Phase 2: Advanced Analysis (New Feature) ==========
            if self.enable_advanced:
                self._advanced_analysis_phase(context)
            
            # ========== Phase 3: Independent Analysis ==========
            self._independent_analysis_phase(context)
            
            # ========== Phase 4: Debate Phase ==========
            if self.enable_debate:
                self._debate_phase(context)
            
            # ========== Phase 5: Discussion Phase ==========
            self._discussion_phase()
            
            # ========== Phase 6: Signal Fusion (New Feature) ==========
            if self.enable_advanced:
                self._signal_fusion_phase(context)
            
            # ========== Phase 7: Voting Recommendation ==========
            self._recommendation_phase(context)
            
            # ========== Phase 8: Anonymous Peer Review (v3.1 New!) ==========
            if self.enable_advanced:
                self._anonymous_peer_review_phase()
            
            # ========== Phase 9: Final Decision ==========
            self._final_decision_phase()
            
            # ========== Phase 10: Red-Blue Team Validation (New Feature) ==========
            if self.enable_advanced:
                self._red_blue_validation_phase()
            
            # ========== Phase 11: Risk Budget Calculation (New Feature) ==========
            if self.enable_advanced:
                self._risk_budget_phase()
            
            # ========== Phase 12: Save Record ==========
            self._save_decision_record()
            
            # ========== Phase 13: Decision Review ==========
            if self.enable_analysis:
                self._analysis_phase()
            
            return self.state.final_decision
            
        except Exception as e:
            self.print_info(f"[X] Decision process error: {e}", Fore.RED)
            raise
    
    def _advanced_analysis_phase(self, context: dict):
        """Advanced analysis phase - multi-timeframe + scenario analysis"""
        self.print_section("[Advanced] Advanced Analysis Phase", Fore.MAGENTA)
        
        try:
            from advanced_features import (
                MultiTimeframeAnalyzer, 
                ScenarioAnalysisEngine,
                ConfidenceCalibrator
            )
            
            prediction = context.get("prediction", {})
            current_price = prediction.get("current_price", 67500)
            
            # 1. Multi-timeframe analysis
            self.print_info("\n[1/3] Multi-timeframe analysis...", Fore.CYAN)
            self.mtf_analyzer = MultiTimeframeAnalyzer(prediction)
            mtf_report = self.mtf_analyzer.generate_report()
            print(f"{Fore.WHITE}{mtf_report}{Style.RESET_ALL}")
            
            # savingtocontextforanalysisreference
            context["multi_timeframe"] = self.mtf_analyzer.analyze_all_timeframes()
            
            time.sleep(1)
            
            # 2. Scenario analysis
            self.print_info("\n[2/3] Scenario analysis...", Fore.CYAN)
            self.scenario_engine = ScenarioAnalysisEngine(current_price, prediction)
            scenario_report = self.scenario_engine.generate_report()
            print(f"{Fore.WHITE}{scenario_report}{Style.RESET_ALL}")
            
            # savingtocontext
            context["scenario_analysis"] = self.scenario_engine.run_scenario_analysis()
            
            time.sleep(1)
            
            # 3. Confidence calibration
            self.print_info("\n[3/3] Confidence calibration...", Fore.CYAN)
            self.confidence_calibrator = ConfidenceCalibrator()
            raw_confidence = prediction.get("confidence", 0.5)
            calibration = self.confidence_calibrator.calibrate(raw_confidence)
            calibration_report = self.confidence_calibrator.generate_report(raw_confidence)
            print(f"{Fore.WHITE}{calibration_report}{Style.RESET_ALL}")
            
            # updateprediction confidence
            context["calibrated_confidence"] = calibration
            
            self.print_info("\n[OK] Advanced analysis phase completed!", Fore.GREEN)
            time.sleep(1)
            
        except ImportError as e:
            self.print_info(f"[!] Advanced analysis module not loaded: {e}", Fore.YELLOW)
        except Exception as e:
            self.print_info(f"[!] Advanced analysis error: {e}", Fore.YELLOW)
    
    def _signal_fusion_phase(self, context: dict):
        """signal fusionphase - Bayesianweighted fusion"""
        self.print_section("[Fusion] cansignal fusion", Fore.CYAN)
        
        try:
            from advanced_features import BayesianSignalFusion
            
            self.signal_fusion = BayesianSignalFusion()
            
            # Collect all class signals
            prediction = context.get("prediction", {})
            indicators = context.get("technical_indicators", {})
            recommendations = self.state.get_all_recommendations()
            
            # buildsignaldictionary
            signals = {}
            
            # 1. prediction modelsignal
            change_percent = prediction.get("change_percent", 0)
            confidence = prediction.get("confidence", 0.5)
            signals["linear_regression"] = {
                "direction": "up" if change_percent > 0 else "down",
                "strength": min(1, abs(change_percent) / 10),
                "confidence": 0.82  # historical accuracy
            }
            
            # 2. technical indicatorssignal
            rsi = indicators.get("rsi", 50)
            if isinstance(rsi, dict):
                rsi = rsi.get("value", 50)
            signals["rsi_signal"] = {
                "direction": "up" if rsi < 30 else ("down" if rsi > 70 else "neutral"),
                "strength": abs(rsi - 50) / 50,
                "confidence": 0.60
            }
            
            macd = indicators.get("macd", {})
            if isinstance(macd, dict):
                macd_hist = macd.get("histogram", 0)
            else:
                macd_hist = 0
            signals["macd_signal"] = {
                "direction": "up" if macd_hist > 0 else "down",
                "strength": min(1, abs(macd_hist) * 100),
                "confidence": 0.58
            }
            
            # 3. analysistotalidentifysignal
            if recommendations:
                bullish_count = sum(1 for r in recommendations 
                                   if r.get("action") in ["buy", "strong_buy"])
                total_count = len(recommendations)
                signals["analyst_consensus"] = {
                    "direction": "up" if bullish_count > total_count / 2 else "down",
                    "strength": abs(bullish_count / total_count - 0.5) * 2,
                    "confidence": 0.65
                }
            
            # fitsignal
            fusion_result = self.signal_fusion.fuse_signals(signals)
            
            # showresults
            self.print_info("\nsignal fusionresults:", Fore.YELLOW)
            self.print_info(f"  risingprobability: {fusion_result['up_probability']:.1%}", Fore.WHITE)
            self.print_info(f"  decline probability: {fusion_result['down_probability']:.1%}", Fore.WHITE)
            self.print_info(f"  comprehensivedirection: {'bullish' if fusion_result['direction'] == 'up' else ('bearish' if fusion_result['direction'] == 'down' else 'neutral')}", 
                          Fore.GREEN if fusion_result['direction'] == 'up' else Fore.RED)
            self.print_info(f"  fitconfidence: {fusion_result['confidence']:.1%}", Fore.WHITE)
            self.print_info(f"\n  recommendation: {fusion_result['recommendation']}", Fore.CYAN)
            
            # showeachsignalWeight
            self.print_info("\neachsignalWeight:", Fore.YELLOW)
            for signal_name, weight in fusion_result['signal_weights'].items():
                self.print_info(f"  {signal_name}: {weight:.1%}", Fore.WHITE)
            
            # savingfitresults
            context["signal_fusion"] = fusion_result
            
            self.print_info("\n[OK] signal fusioncompleted!", Fore.GREEN)
            time.sleep(1)
            
        except ImportError:
            self.print_info("[!] Signal fusion module not loaded", Fore.YELLOW)
        except Exception as e:
            self.print_info(f"[!] Signal fusion error: {e}", Fore.YELLOW)
    
    def _anonymous_peer_review_phase(self):
        """
        Anonymous Peer Review Phase (v3.1 Innovation)
        
        This phase implements:
        1. Anonymize all analyst analyses with labels (A/B/C/D/E)
        2. Each analyst reviews other analyses on Accuracy + Insight
        3. Devil's Advocate challenges consensus
        4. Chairman synthesizes quality-weighted decision
        """
        self.print_section("[PeerReview] Anonymous Peer Review Phase", Fore.MAGENTA)
        
        try:
            from anonymous_peer_review import (
                AnonymousPeerReviewSystem,
                DynamicWeightAdjuster
            )
            
            # Initialize peer review system with LLM client
            self.peer_review_system = AnonymousPeerReviewSystem(self.llm_client)
            
            # Collect all analyst analyses
            analyses = []
            recommendations = self.state.get_all_recommendations()
            
            for rec in recommendations:
                analyst_id = rec.get("analyst_id", 0)
                analyst = self.analysts.get(analyst_id)
                if analyst:
                    # Get the analysis content from state
                    analysis_content = ""
                    for analysis in self.state.analyses:
                        if analysis.get("analyst_id") == analyst_id:
                            analysis_content = analysis.get("content", "")
                            break
                    
                    # Extract key points from recommendation reasoning
                    reasoning = rec.get("reasoning", "")
                    key_points = [p.strip() for p in reasoning.split('.') if p.strip()][:5]
                    
                    analyses.append({
                        "role": analyst.role,
                        "analyst_id": analyst_id,
                        "content": analysis_content if analysis_content else reasoning,
                        "recommendation": rec.get("action", "hold"),
                        "confidence": rec.get("confidence", 5),
                        "key_points": key_points if key_points else ["Analysis provided"]
                    })
            
            if len(analyses) < 2:
                self.print_info("[!] Not enough analyses for peer review (need at least 2)", Fore.YELLOW)
                return
            
            # Step 1: Anonymize analyses
            self.print_info("\n[1/4] Anonymizing analyst opinions...", Fore.CYAN)
            anonymized = self.peer_review_system.anonymize_analyses(analyses)
            
            for anon in anonymized:
                self.print_info(f"  {anon.anonymous_id} -> [Hidden: {anon.original_role}]", Fore.WHITE)
            
            time.sleep(0.5)
            
            # Step 2: Conduct peer reviews
            self.print_info("\n[2/4] Conducting anonymous peer reviews...", Fore.CYAN)
            use_llm = self.llm_client is not None
            reviews = self.peer_review_system.conduct_peer_reviews(use_llm=use_llm)
            self.print_info(f"  Completed {len(reviews)} peer reviews", Fore.GREEN)
            
            # Show sample reviews
            if reviews:
                sample = reviews[0]
                self.print_info(f"\n  Sample Review ({sample.reviewer_id} -> {sample.reviewee_id}):", Fore.YELLOW)
                self.print_info(f"    Accuracy: {sample.scores.get('accuracy', 'N/A')}/10", Fore.WHITE)
                self.print_info(f"    Insight: {sample.scores.get('insight', 'N/A')}/10", Fore.WHITE)
                self.print_info(f"    Comment: {sample.overall_comment[:60]}...", Fore.WHITE)
            
            time.sleep(0.5)
            
            # Step 3: Calculate rankings
            self.print_info("\n[3/4] Calculating quality-weighted rankings...", Fore.CYAN)
            rankings = self.peer_review_system.calculate_rankings()
            
            self.print_info("\n  === PEER REVIEW RANKINGS ===", Fore.YELLOW)
            for r in rankings:
                rank_color = Fore.GREEN if r.rank == 1 else (Fore.CYAN if r.rank <= 3 else Fore.WHITE)
                self.print_info(f"  #{r.rank}. {r.anonymous_id} ({r.original_role})", rank_color)
                self.print_info(f"      Score: {r.average_score:.2f}/10 | Reviews: {r.total_reviews}", Fore.WHITE)
                
                # Show score breakdown
                breakdown_str = " | ".join([f"{k[:3].upper()}: {v:.1f}" for k, v in list(r.score_breakdown.items())[:3]])
                self.print_info(f"      [{breakdown_str}]", Fore.WHITE)
            
            time.sleep(0.5)
            
            # Step 4: Devil's Advocate Challenge
            self.print_info("\n[4/4] Devil's Advocate challenging consensus...", Fore.RED)
            challenge = self.peer_review_system.generate_devils_advocate_challenge(use_llm=use_llm)
            
            self.print_info(f"\n  Challenging: {challenge.consensus_view}", Fore.RED)
            self.print_info(f"  Counter-arguments:", Fore.YELLOW)
            for arg in challenge.counter_arguments[:3]:
                self.print_info(f"    - {arg[:70]}...", Fore.WHITE)
            
            self.print_info(f"  Risk Warnings:", Fore.YELLOW)
            for risk in challenge.risk_warnings[:2]:
                self.print_info(f"    ! {risk[:70]}...", Fore.RED)
            
            self.print_info(f"  Challenge Confidence: {challenge.confidence_in_challenge}/10", Fore.CYAN)
            
            # Get final synthesis
            synthesis = self.peer_review_system.get_final_synthesis(use_llm=False)
            
            # Store results in state
            self.state.final_decision["peer_review"] = {
                "rankings": [(r.anonymous_id, r.original_role, r.average_score) for r in rankings],
                "top_analyst": rankings[0].original_role if rankings else "N/A",
                "top_score": rankings[0].average_score if rankings else 0,
                "devils_advocate_confidence": challenge.confidence_in_challenge,
                "synthesis": synthesis
            }
            
            # Adjust final decision based on peer review quality
            if rankings and synthesis:
                quality_factor = rankings[0].average_score / 10  # 0-1
                
                # If top-ranked analyst disagrees with current consensus, flag it
                top_analyst_rec = self.peer_review_system.anonymous_analyses[rankings[0].anonymous_id].recommendation
                current_rec = self.state.final_decision.get("action", "hold")
                
                if top_analyst_rec != current_rec:
                    self.print_info(f"\n  [!] NOTE: Top-ranked analyst recommends {top_analyst_rec.upper()}", Fore.YELLOW)
                    self.print_info(f"      but current consensus is {current_rec.upper()}", Fore.YELLOW)
                    self.print_info(f"      Consider adjusting decision based on quality ranking", Fore.YELLOW)
            
            self.print_info("\n[OK] Anonymous Peer Review completed!", Fore.GREEN)
            
            # Initialize dynamic weight adjuster for future predictions
            try:
                self.weight_adjuster = DynamicWeightAdjuster()
                # Record this prediction for future validation
                for rec in recommendations:
                    self.weight_adjuster.record_prediction(
                        rec.get("role", "unknown"),
                        {
                            "recommendation": rec.get("action", "hold"),
                            "confidence": rec.get("confidence", 5),
                            "timestamp": datetime.now().isoformat()
                        }
                    )
            except Exception as e:
                pass  # Weight adjustment is optional
            
            time.sleep(1)
            
        except ImportError as e:
            self.print_info(f"[!] Anonymous Peer Review module not loaded: {e}", Fore.YELLOW)
        except Exception as e:
            self.print_info(f"[!] Anonymous Peer Review error: {e}", Fore.YELLOW)
            import traceback
            traceback.print_exc()
    
    def _red_blue_validation_phase(self):
        """red-blue teamvalidationphase"""
        self.print_section("[RedBlue] red-blue teamvalidation", Fore.RED)
        
        try:
            from advanced_features import RedBlueTeamValidator
            
            self.red_blue_validator = RedBlueTeamValidator(None)
            
            # Get final decision and context
            decision = self.state.final_decision
            context = {
                "prediction": self.state.prediction,
                "recommendations": self.state.get_all_recommendations()
            }
            
            # Run adversarial validation
            validation = self.red_blue_validator.validate_decision(decision, context)
            
            # showred teamchallenge
            self.print_info("\n[red team] decisionchallenge:", Fore.RED)
            for challenge in validation["red_team_challenge"]["challenges"]:
                severity_color = Fore.RED if challenge["severity"] == "high" else Fore.YELLOW
                self.print_info(f"  [{challenge['severity']}] {challenge['point']}", severity_color)
                self.print_info(f"      {challenge['detail'][:80]}...", Fore.WHITE)
            
            # showblue teamdefense
            self.print_info("\n[blue team] decisiondefense:", Fore.BLUE)
            for defense in validation["blue_team_defense"]["defenses"]:
                strength_color = Fore.GREEN if defense["strength"] == "strong" else Fore.CYAN
                self.print_info(f"  [{defense['strength']}] {defense['point']}", strength_color)
                self.print_info(f"      {defense['detail'][:80]}...", Fore.WHITE)
            
            # Show verdict
            verdict = validation["verdict"]
            verdict_color = Fore.GREEN if verdict["final_score"] >= 7 else (
                Fore.YELLOW if verdict["final_score"] >= 5 else Fore.RED
            )
            
            self.print_info(f"\n[Verdict] {verdict['verdict']}", verdict_color)
            self.print_info(f"  challenge: {verdict['challenge_score']:.1f}/10", Fore.WHITE)
            self.print_info(f"  defense: {verdict['defense_score']:.1f}/10", Fore.WHITE)
            self.print_info(f"  final: {verdict['final_score']:.1f}/10", Fore.WHITE)
            self.print_info(f"  recommendation: {verdict['recommendation']}", Fore.CYAN)
            
            # savingvalidationresults
            self.state.final_decision["red_blue_validation"] = validation
            
            self.print_info("\n[OK] red-blue teamvalidationcompleted!", Fore.GREEN)
            time.sleep(1)
            
        except ImportError:
            self.print_info("[!] Red-Blue validation module not loaded", Fore.YELLOW)
        except Exception as e:
            self.print_info(f"[!] Red-Blue validation error: {e}", Fore.YELLOW)
    
    def _risk_budget_phase(self):
        """riskbudgetpositioncalculatephase"""
        self.print_section("[Risk] riskbudgetpositioncalculate", Fore.YELLOW)
        
        try:
            from advanced_features import RiskBudgetManager
            
            # Assume total funds $100,000, maximum risk 5%
            total_capital = 100000
            max_risk = 5.0
            
            self.risk_manager = RiskBudgetManager(total_capital, max_risk)
            
            # getdecisiondata
            prediction = self.state.prediction
            decision = self.state.final_decision
            
            current_price = prediction.get("current_price", 67500)
            confidence = prediction.get("confidence", 0.5)
            
            # calculatestop-lossprice（false-5%stop-loss）
            stop_loss_price = current_price * 0.95
            
            # generatepositionreport
            position_report = self.risk_manager.generate_position_report(
                entry_price=current_price,
                stop_loss_price=stop_loss_price,
                confidence=confidence
            )
            
            print(f"{Fore.WHITE}{position_report}{Style.RESET_ALL}")
            
            # getcalculateresults
            position_calc = self.risk_manager.calculate_position_size(
                current_price, stop_loss_price, confidence
            )
            
            # updatedecision  positionrecommendation
            calculated_position = position_calc["position_percent"]
            original_position = decision.get("position", 50)
            
            if abs(calculated_position - original_position) > 10:
                self.print_info(f"\n[!] positionadjustrecommendation:", Fore.YELLOW)
                self.print_info(f"    Original Recommended Position: {original_position}%", Fore.WHITE)
                self.print_info(f"    riskbudgetposition: {calculated_position:.1f}%", Fore.CYAN)
                self.print_info(f"    Recommend using risk budget calculated position", Fore.GREEN)
            
            # savingriskcalculateresults
            self.state.final_decision["risk_budget_calculation"] = position_calc
            
            self.print_info("\n[OK] riskbudgetcalculatecompleted!", Fore.GREEN)
            time.sleep(1)
            
        except ImportError:
            self.print_info("[!] riskbudgetmodulenot yetloading", Fore.YELLOW)
        except Exception as e:
            self.print_info(f"[!] riskbudgetcalculateerror: {e}", Fore.YELLOW)
    
    def _debate_phase(self, context: dict):
        """debatephase"""
        self.print_section("[Debate] adversarial debatephase", Fore.RED)
        
        try:
            from enhanced_analysts import DebateManager
            
            # createdebatemanagement
            analyst_dict = {analyst.role: analyst for analyst in self.analysts.values()}
            self.debate_manager = DebateManager(analyst_dict, None)
            
            # Get initial analysis
            initial_analyses = [
                {"role": a["role"], "content": a["content"]}
                for a in self.state.get_all_analyses()
            ]
            
            # rundebate
            debate_results = self.debate_manager.run_debate(context, initial_analyses)
            
            # showdebateresults
            for result in debate_results:
                if result.get("type") == "consensus":
                    self.print_info(f"  {result.get('message', '')}", Fore.GREEN)
                else:
                    topic = result.get("topic", "investmentdirection")
                    self.print_info(f"\n  debatetheme: {topic}", Fore.YELLOW)
                    
                    for content in result.get("debate_content", []):
                        role = content.get("role", "unknown")
                        stance = "bullish" if content.get("stance") == "bullish" else "bearish"
                        argument = content.get("argument", "")[:200]
                        
                        color = Fore.GREEN if stance == "bullish" else Fore.RED
                        self.print_info(f"\n  [{role}] ({stance}):", color)
                        self.print_info(f"    {argument}...", Fore.WHITE)
                        
                        if content.get("rebuttal"):
                            self.print_info(f"    opposite: {content['rebuttal'][:100]}...", Fore.LIGHTBLACK_EX)
            
            self.print_info("\n[OK] debatephasecompleted!", Fore.GREEN)
            time.sleep(1)
            
        except ImportError:
            self.print_info("[!] debatemodulenot yetloading，skipdebatephase", Fore.YELLOW)
        except Exception as e:
            self.print_info(f"[!] debatephaseerror: {e}", Fore.YELLOW)
    
    def _analysis_phase(self):
        """decisionreviewphase"""
        self.print_section("[Analysis] decisionreviewanalysis", Fore.BLUE)
        
        try:
            from decision_analyzer import DecisionAnalyzer
            
            # exportdecisiondata
            decision_data = self.state.export_to_dict()
            
            # createanalysis
            analyzer = DecisionAnalyzer(decision_data)
            
            # generatereport
            report = analyzer.generate_report()
            print(report)
            
            self.print_info("\n[OK] reviewanalysiscompleted!", Fore.GREEN)
            
        except ImportError:
            self.print_info("[!] Analysis module not loaded, skip review phase", Fore.YELLOW)
        except Exception as e:
            self.print_info(f"[!] reviewanalysiserror: {e}", Fore.YELLOW)


# ==================== Main Program ====================

def print_banner():
    """Print welcome banner"""
    banner = f"""
{Fore.CYAN}+==============================================================================+
|                                                                              |
|   [Investment Committee] Intelligent Investment Decision System v3.0        |
|                                                                              |
|   Multi-Agent Collaborative Bitcoin Investment Analysis System              |
|   Multi-Agent Architecture for Investment Decision Analysis                 |
|                                                                              |
+==============================================================================+{Style.RESET_ALL}

{Fore.YELLOW}Core Features:{Style.RESET_ALL}
  * Analyst Personalization System - Each analyst has unique style and preferences
  * Adversarial Debate Phase - Analysts engage in viewpoint discussions
  * Decision Review Analysis - Automatically analyze decision quality and risk
  * Historical Decision Tracking - Track and validate historical decision accuracy

{Fore.GREEN}Advanced Features:{Style.RESET_ALL}
  * Multi-Timeframe Analysis - Short/Medium/Long-term comprehensive analysis
  * Scenario Analysis Engine - Bull/Bear/Volatile/Black Swan scenario simulation
  * Bayesian Signal Fusion - Intelligent weighted fusion of multiple signal sources
  * Red-Blue Team Validation - Decision quality dual validation mechanism
  * Risk Budget Management - Scientific position calculation based on risk budget
  * Confidence Calibration - Calibrate AI overconfidence
"""
    print(banner)


def print_game_intro():
    """Print system introduction"""
    intro = f"""
{Fore.CYAN}+==============================================================================+
|                            Analyst Team Introduction                         |
+==============================================================================+{Style.RESET_ALL}

{Fore.YELLOW}[Analyst Team]{Style.RESET_ALL}

  {Fore.GREEN}[Chart] Technical Analyst{Style.RESET_ALL} - "Chart Master"
     - Specializes in: Price trend analysis, technical indicator interpretation
     - Uses three-model ensemble prediction (Linear Regression + ARIMA + Prophet)
     - Direction prediction accuracy up to 82.1%

  {Fore.BLUE}[Industry] Industry Analyst{Style.RESET_ALL} - "Industry Observer"
     - Specializes in: Industry trends, policy impact analysis
     - Macro perspective, focus on long-term development

  {Fore.MAGENTA}[Finance] Financial Analyst{Style.RESET_ALL} - "Risk Controller"
     - Specializes in: Risk-return analysis, fund management
     - Conservative style, risk control priority

  {Fore.YELLOW}[Market] Market Expert{Style.RESET_ALL} - "Sentiment Catcher"
     - Specializes in: Market sentiment, fund flow analysis
     - Sharp insight, captures market hotspots

  {Fore.RED}[Risk] Risk Analyst{Style.RESET_ALL} - "Security Guardian"
     - Specializes in: Risk identification, risk quantification
     - Extremely cautious, guards against black swans

  {Fore.WHITE}[Manager] Investment Manager{Style.RESET_ALL} - "Decision Maker"
     - Specializes in: Comprehensive analysis, final decision
     - Weighs pros and cons, considers all parties

{Fore.CYAN}+==============================================================================+
|                              Decision Process                                |
+==============================================================================+{Style.RESET_ALL}

  1. [Data]     Data Preparation -> Run multi-model ensemble prediction
  2. [Analysis] Independent Analysis -> Each analyst provides independent analysis
  3. [Debate]   Adversarial Debate -> Analysts engage in viewpoint discussions
  4. [Discuss]  Discussion Phase -> Analysts provide supplementary opinions
  5. [Vote]     Voting Recommendation -> Each analyst provides final recommendation
  6. [Decision] Final Decision -> Investment manager makes comprehensive decision
  7. [Review]   Decision Review -> Analyze decision quality and risk

"""
    print(intro)


def confirm_start() -> bool:
    """Confirm whether to start"""
    while True:
        choice = input(f"\n{Fore.YELLOW}Start decision process? (yes/no): {Style.RESET_ALL}").strip().lower()
        
        if choice in ['yes', 'y', 'start', '']:
            return True
        elif choice in ['no', 'n', 'exit']:
            return False
        else:
            print(f"{Fore.RED}Please input yes or no{Style.RESET_ALL}")


def ask_continue() -> bool:
    """Ask whether to continue"""
    while True:
        choice = input(f"\n{Fore.YELLOW}Run decision process again? (yes/no): {Style.RESET_ALL}").strip().lower()
        
        if choice in ['yes', 'y']:
            return True
        elif choice in ['no', 'n', 'exit', '']:
            return False
        else:
            print(f"{Fore.RED}Please input yes or no{Style.RESET_ALL}")


def main(auto_mode: bool = False, enable_debate: bool = True, 
         enable_analysis: bool = True, enable_advanced: bool = True):
    """Main program entry"""
    # Set console encoding
    if sys.platform == 'win32':
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    
    print_banner()
    
    if not auto_mode:
        print_game_intro()
    
    try:
        if auto_mode:
            # Auto mode: run directly
            print(f"\n{Fore.GREEN}[Auto] Auto mode, starting decision process...{Style.RESET_ALL}\n")
            run_decision_cycle(enable_debate, enable_analysis, enable_advanced)
            print(f"\n{Fore.CYAN}Decision process completed! Thank you for using!{Style.RESET_ALL}\n")
        else:
            # Interactive mode
            while True:
                if not confirm_start():
                    print(f"\n{Fore.CYAN}Thank you for using Intelligent Investment Decision System! Goodbye!{Style.RESET_ALL}\n")
                    break
                
                print(f"\n{Fore.GREEN}Starting decision process...{Style.RESET_ALL}\n")
                run_decision_cycle(enable_debate, enable_analysis, enable_advanced)
                
                if not ask_continue():
                    print(f"\n{Fore.CYAN}Thank you for using Intelligent Investment Decision System! Goodbye!{Style.RESET_ALL}\n")
                    break
    
    except KeyboardInterrupt:
        print(f"\n\n{Fore.YELLOW}User interrupted{Style.RESET_ALL}")
        print(f"{Fore.CYAN}Thank you for using Intelligent Investment Decision System! Goodbye!{Style.RESET_ALL}\n")
        sys.exit(0)
    
    except Exception as e:
        print(f"\n{Fore.RED}[X] Program execution error: {e}{Style.RESET_ALL}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def run_decision_cycle(enable_debate: bool = True, enable_analysis: bool = True,
                       enable_advanced: bool = True):
    """Run one complete decision process"""
    # Create enhanced Investment Committee v3.0
    committee = EnhancedInvestmentCommittee(
        enable_debate=enable_debate,
        enable_analysis=enable_analysis,
        enable_advanced=enable_advanced
    )
    
    # Initialize committee
    committee.initialize_committee()
    
    # Prepare price data
    price_data = {
        "current_price": 67500,
        "Open": 67000,
        "High": 68500,
        "Low": 66800,
        "Volume": 1500000,
        "RSI_14": 58.5,
        "MACD": 350,
        "Signal_Line": 280,
        "SMA_5": 67200,
        "SMA_20": 65800,
        "BB_upper": 70000,
        "BB_middle": 67000,
        "BB_lower": 64000,
    }
    
    # Run decision process
    final_decision = committee.run_decision_process(price_data)
    
    # Show final results
    print(f"""
{Fore.GREEN}+==============================================================================+
|                           [OK] Decision Process Completed                    |
+==============================================================================+
|  Final Recommendation: {final_decision['action_cn']:<54}|
|  Recommended Position: {final_decision['position']}%{' ' * 52}|
|  Confidence Index: {final_decision['confidence']}/10{' ' * 55}|
+==============================================================================+{Style.RESET_ALL}
""")
    
    return final_decision


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Intelligent Investment Decision System v3.0 - manycaninvestmentanalysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
example:
  python investment_committee.py                    # Interactive mode (default)
  python investment_committee.py --auto             # automaticpattern（function）
  python investment_committee.py --auto --fast      # Fast mode (skip debate and advanced analysis)
  python investment_committee.py --no-debate        # disableddebatephase
  python investment_committee.py --no-analysis      # disabledreviewanalysis
  python investment_committee.py --no-advanced      # Disable advanced analysis features

Advanced Analysis Features (v3.0 new):
  * Multi-timeframe analysis - Short/Medium/Long-term 3D
  * Scenario analysis engine - Bull/Bear/Volatile/Black Swan scenarios
  * Bayesiansignal fusion - canweightedmanysignalsource
  * Red-blue team validation - Decision dual validation
  * Risk budget management - Scientific position calculation
  * Confidence calibration - Calibrate overconfidence
"""
    )
    parser.add_argument(
        '--auto', 
        action='store_true', 
        help='Auto mode, no interactive confirmation needed'
    )
    parser.add_argument(
        '--fast',
        action='store_true',
        help='Fast mode, skip debate and advanced analysis'
    )
    parser.add_argument(
        '--no-debate', 
        action='store_true', 
        help='disableddebatephase'
    )
    parser.add_argument(
        '--no-analysis', 
        action='store_true', 
        help='disabledreviewanalysis'
    )
    parser.add_argument(
        '--no-advanced',
        action='store_true',
        help='Disable advanced analysis features (multi-timeframe, scenario analysis, signal fusion, etc.)'
    )
    
    args = parser.parse_args()
    
    # fastpattern：disableddebate、analysisandadvancedfunction
    if args.fast:
        args.no_debate = True
        args.no_analysis = True
        args.no_advanced = True
    
    # Run main program
    main(
        auto_mode=args.auto,
        enable_debate=not args.no_debate,
        enable_analysis=not args.no_analysis,
        enable_advanced=not args.no_advanced
    )

