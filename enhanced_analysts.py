# -*- coding: utf-8 -*-
"""
Enhanced Analyst System - Personalized Analysts

Features:
1. Each analyst has unique style and strategy preferences
2. Analysts can engage in adversarial debates
3. Analysts have memory and learning capabilities
4. Analysts adjust strategies based on historical performance
"""

import random
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, List, Optional
from colorama import Fore, Style


# Analyst personality configuration
ANALYST_PERSONALITIES = {
    "technical_analyst": {
        "name": "Technical Analyst",
        "nickname": "Chart Master",
        "style": "Data Driven",
        "risk_preference": "neutral",  # conservative/neutral/aggressive
        "confidence_bias": 0,  # -2 to +2
        "focus_areas": ["Price Trends", "Technical Indicators", "Model Predictions"],
        "debate_style": "Logically Rigorous",
        "catchphrase": "Let the data speak",
        "strengths": ["Quantitative Analysis", "Trend Identification", "Signal Interpretation"],
        "weaknesses": ["May ignore fundamentals", "Slow to react to sudden events"]
    },
    "industry_analyst": {
        "name": "Industry Analyst",
        "nickname": "Industry Observer",
        "style": "Macro Perspective",
        "risk_preference": "neutral",
        "confidence_bias": 0,
        "focus_areas": ["Industry Trends", "Policy Impact", "Competitive Landscape"],
        "debate_style": "Comprehensive Objective",
        "catchphrase": "See the big picture",
        "strengths": ["Macro Analysis", "Policy Interpretation", "Trend Prediction"],
        "weaknesses": ["May ignore short-term volatility", "Information lag"]
    },
    "financial_analyst": {
        "name": "Financial Analyst",
        "nickname": "Risk Controller",
        "style": "Conservative Stable",
        "risk_preference": "conservative",
        "confidence_bias": -1,
        "focus_areas": ["Risk-Return", "Fund Management", "Stop-Loss/Take-Profit"],
        "debate_style": "Cautious Pragmatic",
        "catchphrase": "Risk control is the top priority",
        "strengths": ["Risk Quantification", "Fund Allocation", "Return Assessment"],
        "weaknesses": ["May be too conservative", "Miss opportunities"]
    },
    "market_expert": {
        "name": "Market Expert",
        "nickname": "Sentiment Catcher",
        "style": "Sharp Insight",
        "risk_preference": "aggressive",
        "confidence_bias": 1,
        "focus_areas": ["Market Sentiment", "Fund Flow", "Institutional Movements"],
        "debate_style": "Intuitive Sharp",
        "catchphrase": "The market is always right",
        "strengths": ["Sentiment Analysis", "Hotspot Capture", "Short-term Prediction"],
        "weaknesses": ["May be overly optimistic", "Ignore fundamentals"]
    },
    "risk_analyst": {
        "name": "Risk Analyst",
        "nickname": "Security Guardian",
        "style": "Extremely Cautious",
        "risk_preference": "conservative",
        "confidence_bias": -2,
        "focus_areas": ["Risk Identification", "Risk Quantification", "Risk Warning"],
        "debate_style": "Pessimistic Cautious",
        "catchphrase": "Better to miss than to make mistakes",
        "strengths": ["Risk Identification", "Stress Testing", "Extreme Scenario Analysis"],
        "weaknesses": ["May be overly pessimistic", "Affect team confidence"]
    },
    "investment_manager": {
        "name": "Investment Manager",
        "nickname": "Decision Maker",
        "style": "Comprehensive Balance",
        "risk_preference": "neutral",
        "confidence_bias": 0,
        "focus_areas": ["Comprehensive Analysis", "Weigh Pros and Cons", "Final Decision"],
        "debate_style": "Objective Neutral",
        "catchphrase": "Listen to all sides",
        "strengths": ["Comprehensive Judgment", "Team Coordination", "Decision Execution"],
        "weaknesses": ["May compromise too much", "Decision delay"]
    }
}


class DebateManager:
    """
    Debate Manager - Manages adversarial debates between analysts
    
    Design principles:
    - Analysts have different stances like players
    - Debates reveal different perspectives
    - Debate results influence final decisions
    """
    
    def __init__(self, analysts: dict, api_client):
        self.analysts = analysts
        self.client = api_client
        self.debate_rounds = []
    
    def run_debate(self, context: dict, initial_analyses: list) -> list:
        """
        Run debate phase
        
        Process:
        1. Identify divergences
        2. Organize adversarial debate
        3. Summarize debate results
        """
        debate_results = []
        
        # 1. Identify divergences
        divergences = self._identify_divergences(initial_analyses)
        
        if not divergences:
            return [{
                "type": "consensus",
                "message": "Analysts reached consensus, no debate needed"
            }]
        
        # 2. Conduct debate
        for divergence in divergences[:2]:  # Conduct at most 2 rounds of debate
            debate_round = self._conduct_debate_round(divergence, context)
            debate_results.append(debate_round)
            self.debate_rounds.append(debate_round)
        
        return debate_results
    
    def _identify_divergences(self, analyses: list) -> list:
        """Identify analyst divergences"""
        divergences = []
        
        # Group by stance
        bullish = []
        bearish = []
        neutral = []
        
        for analysis in analyses:
            role = analysis.get("role", "unknown")
            content = analysis.get("content", "")
            
            # Simple stance judgment
            if any(word in content.lower() for word in ["buy", "bullish", "rising", "opportunity", "positive"]):
                bullish.append(role)
            elif any(word in content.lower() for word in ["sell", "bearish", "decline", "risk", "negative"]):
                bearish.append(role)
            else:
                neutral.append(role)
        
        # If significant divergence exists
        if bullish and bearish:
            divergences.append({
                "topic": "Market Direction",
                "bullish_side": bullish,
                "bearish_side": bearish,
                "neutral_side": neutral
            })
        
        return divergences
    
    def _conduct_debate_round(self, divergence: dict, context: dict) -> dict:
        """Conduct one round of debate"""
        topic = divergence.get("topic", "Investment Decision")
        bullish_side = divergence.get("bullish_side", [])
        bearish_side = divergence.get("bearish_side", [])
        
        debate_content = []
        
        # Bullish side speaks
        if bullish_side:
            bullish_analyst = bullish_side[0]
            analyst = self.analysts.get(bullish_analyst)
            if analyst:
                bullish_argument = self._get_debate_argument(
                    analyst, "bullish", topic, context
                )
                debate_content.append({
                    "role": bullish_analyst,
                    "stance": "bullish",
                    "argument": bullish_argument
                })
        
        # Bearish side speaks
        if bearish_side:
            bearish_analyst = bearish_side[0]
            analyst = self.analysts.get(bearish_analyst)
            if analyst:
                bearish_argument = self._get_debate_argument(
                    analyst, "bearish", topic, context
                )
                debate_content.append({
                    "role": bearish_analyst,
                    "stance": "bearish",
                    "argument": bearish_argument
                })
        
        # Rebuttal phase
        if len(debate_content) >= 2:
            # Bullish side rebuttal
            if bullish_side:
                rebuttal = self._get_rebuttal(
                    self.analysts.get(bullish_side[0]),
                    debate_content[1]["argument"],
                    context
                )
                debate_content[0]["rebuttal"] = rebuttal
            
            # Bearish side rebuttal
            if bearish_side:
                rebuttal = self._get_rebuttal(
                    self.analysts.get(bearish_side[0]),
                    debate_content[0]["argument"],
                    context
                )
                debate_content[1]["rebuttal"] = rebuttal
        
        return {
            "topic": topic,
            "debate_content": debate_content,
            "timestamp": datetime.now().isoformat()
        }
    
    def _get_debate_argument(self, analyst, stance: str, topic: str, context: dict) -> str:
        """Get debate argument"""
        personality = ANALYST_PERSONALITIES.get(analyst.role, {})
        
        stance_display = "Bullish" if stance == "bullish" else "Bearish"
        
        prompt = f"""You are {personality.get('name', 'Analyst')}, nicknamed "{personality.get('nickname', '')}".
Your analysis style is {personality.get('style', 'objective')}, your catchphrase is "{personality.get('catchphrase', '')}".

You are currently participating in an Investment Committee debate. The topic is: {topic}

Your stance is: [{stance_display}]

[Market Data]
- Predicted Change: {context.get('prediction', {}).get('change_percent', 0):.2f}%
- Model Confidence: {context.get('prediction', {}).get('confidence', 0):.1%}

Please elaborate your {stance_display} reasoning using your professional knowledge and personal style.
Requirements:
1. Match your role characteristics and analysis style
2. Provide specific arguments to support your view
3. Keep it concise and powerful, not exceeding 200 characters
4. Demonstrate your professional expertise: {', '.join(personality.get('strengths', []))}
"""
        
        try:
            from investment_committee import InvestmentConfig
            response = analyst.client.chat(
                system_prompt=analyst.system_prompt,
                user_message=prompt,
                temperature=InvestmentConfig.TEMPERATURE
            )
            return response
        except Exception as e:
            return f"[Debate speech failed: {e}]"
    
    def _get_rebuttal(self, analyst, opponent_argument: str, context: dict) -> str:
        """Get rebuttal"""
        if analyst is None:
            return "[No rebuttal]"
        
        personality = ANALYST_PERSONALITIES.get(analyst.role, {})
        
        prompt = f"""You are {personality.get('name', 'Analyst')}, your debate style is {personality.get('debate_style', 'rational')}.

The opponent's viewpoint is:
{opponent_argument[:300]}

Please provide a brief rebuttal to the opponent's viewpoint (not exceeding 100 characters).
Note:
1. Maintain professionalism and courtesy
2. Use facts and logic to rebut
3. Demonstrate your analysis style
"""
        
        try:
            from investment_committee import InvestmentConfig
            response = analyst.client.chat(
                system_prompt=analyst.system_prompt,
                user_message=prompt,
                temperature=InvestmentConfig.TEMPERATURE
            )
            return response
        except Exception as e:
            return f"[Rebuttal failed: {e}]"
    
    def get_debate_summary(self) -> str:
        """Get debate summary"""
        if not self.debate_rounds:
            return "No debate conducted"
        
        summary_parts = []
        for i, debate in enumerate(self.debate_rounds, 1):
            summary_parts.append(f"\nRound {i} Debate - Topic: {debate['topic']}")
            for content in debate.get("debate_content", []):
                role = content.get("role", "Unknown")
                stance = "Bullish" if content.get("stance") == "bullish" else "Bearish"
                summary_parts.append(f"  [{role}] ({stance}): {content.get('argument', '')[:100]}...")
        
        return "\n".join(summary_parts)


class PersonalizedAnalystMixin:
    """
    Personalized Analyst Mixin Class
    Adds personalization features to analysts
    """
    
    def _get_personality(self) -> dict:
        """Get personality configuration"""
        return ANALYST_PERSONALITIES.get(self.role, {})
    
    def _apply_confidence_bias(self, base_confidence: int) -> int:
        """Apply confidence bias"""
        personality = self._get_personality()
        bias = personality.get("confidence_bias", 0)
        adjusted = base_confidence + bias
        return max(1, min(10, adjusted))
    
    def _apply_risk_preference(self, action: str) -> str:
        """Adjust action recommendation based on risk preference"""
        personality = self._get_personality()
        risk_pref = personality.get("risk_preference", "neutral")
        
        if risk_pref == "conservative":
            # Conservative: downgrade aggressive recommendations
            if action == "strong_buy":
                return "buy"
            elif action == "strong_sell":
                return "sell"
        elif risk_pref == "aggressive":
            # Aggressive: upgrade moderate recommendations
            if action == "buy":
                return random.choice(["buy", "strong_buy"])
            elif action == "sell":
                return random.choice(["sell", "strong_sell"])
        
        return action
    
    def get_personalized_intro(self) -> str:
        """Get personalized introduction"""
        personality = self._get_personality()
        return f"""
[{personality.get('name', self.role_name)}] - "{personality.get('nickname', '')}"
Style: {personality.get('style', 'objective')}
Specializes in: {', '.join(personality.get('strengths', []))}
Catchphrase: "{personality.get('catchphrase', '')}"
"""
    
    def enhance_analysis_prompt(self, base_prompt: str) -> str:
        """Enhance analysis prompt with personality elements"""
        personality = self._get_personality()
        
        personality_context = f"""
[Your Role Characteristics]
- Nickname: {personality.get('nickname', 'Professional Analyst')}
- Analysis Style: {personality.get('style', 'objective')}
- Risk Preference: {personality.get('risk_preference', 'neutral')}
- Core Strengths: {', '.join(personality.get('strengths', []))}
- Note Biases: {', '.join(personality.get('weaknesses', []))}
- Catchphrase: "{personality.get('catchphrase', '')}"

Please demonstrate your personal style in analysis, but maintain professional objectivity.
"""
        
        return base_prompt + personality_context


class ConfidenceCalculator:
    """
    Confidence Index Calculator
    Calculates final confidence based on multiple factors
    """
    
    @staticmethod
    def calculate_ensemble_confidence(
        model_confidence: float,
        analyst_confidences: list,
        consensus_level: str,
        market_volatility: float = 0.5
    ) -> float:
        """
        Calculate comprehensive confidence index
        
        Parameters:
            model_confidence: Model confidence (0-1)
            analyst_confidences: Each analyst's confidence (1-10)
            consensus_level: Consensus level
            market_volatility: Market volatility (0-1)
        
        Returns:
            Comprehensive confidence index (0-1)
        """
        # 1. Model confidence weight 40%
        model_weight = 0.4
        model_score = model_confidence * model_weight
        
        # 2. Analyst confidence weight 30%
        analyst_weight = 0.3
        if analyst_confidences:
            avg_analyst_confidence = sum(analyst_confidences) / len(analyst_confidences) / 10
            analyst_score = avg_analyst_confidence * analyst_weight
        else:
            analyst_score = 0.5 * analyst_weight
        
        # 3. Consensus weight 20%
        consensus_weight = 0.2
        consensus_scores = {
            "high_consensus": 1.0,
            "majority_consensus": 0.7,
            "significant_divergence": 0.4,
            "scattered": 0.2
        }
        consensus_score = consensus_scores.get(consensus_level, 0.5) * consensus_weight
        
        # 4. Market volatility penalty 10%
        volatility_weight = 0.1
        volatility_score = (1 - market_volatility) * volatility_weight
        
        # Comprehensive calculation
        total_confidence = model_score + analyst_score + consensus_score + volatility_score
        
        return min(1.0, max(0.0, total_confidence))
    
    @staticmethod
    def format_confidence(confidence: float) -> str:
        """Format confidence display"""
        if confidence >= 0.8:
            return f"{confidence:.1%} (High Confidence)"
        elif confidence >= 0.6:
            return f"{confidence:.1%} (Moderate Confidence)"
        elif confidence >= 0.4:
            return f"{confidence:.1%} (Relatively Cautious)"
        else:
            return f"{confidence:.1%} (Relatively Conservative)"


# Test
if __name__ == "__main__":
    print("=" * 60)
    print("Enhanced Analyst System Test")
    print("=" * 60)
    
    # Show all analysts
    for role, personality in ANALYST_PERSONALITIES.items():
        print(f"\n[{personality['name']}] - {personality['nickname']}")
        print(f"  Style: {personality['style']}")
        print(f"  Risk Preference: {personality['risk_preference']}")
        print(f"  Strengths: {', '.join(personality['strengths'])}")
        print(f"  Catchphrase: \"{personality['catchphrase']}\"")
    
    # Test confidence calculator
    print("\n" + "=" * 60)
    print("Confidence Index Calculation Test")
    print("=" * 60)
    
    confidence = ConfidenceCalculator.calculate_ensemble_confidence(
        model_confidence=0.75,
        analyst_confidences=[8, 7, 6, 7, 5],
        consensus_level="majority_consensus",
        market_volatility=0.3
    )
    print(f"\nComprehensive Confidence Index: {ConfidenceCalculator.format_confidence(confidence)}")
