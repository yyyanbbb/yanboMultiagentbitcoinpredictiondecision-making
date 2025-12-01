# -*- coding: utf-8 -*-
"""
Investment Decision Analyzer - Decision Review System

Features:
1. Analyze investment decision process
2. Evaluate analyst performance
3. Identify key decision points
4. Extract lessons learned
5. Generate review reports
"""

import json
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path


class DecisionAnalyzer:
    """
    Decision Analyzer - Investment Decision Review System
    
    Design principles:
    - Analyze overall decision process
    - Evaluate role performance
    - Identify key decision points
    - Extract lessons learned
    """
    
    def __init__(self, decision_data: dict):
        """Initialize analyzer"""
        self.decision_data = decision_data
        self.final_decision = decision_data.get("final_decision", {})
        self.prediction = decision_data.get("prediction", {})
        self.analyses = decision_data.get("analyses", [])
        self.recommendations = decision_data.get("recommendations", [])
        self.discussions = decision_data.get("discussions", [])
    
    def analyze_decision(self) -> dict:
        """Comprehensive analysis of decision process"""
        analysis = {
            "decision_summary": self._analyze_decision_summary(),
            "analyst_performance": self._analyze_analyst_performance(),
            "consensus_analysis": self._analyze_consensus(),
            "key_insights": self._extract_key_insights(),
            "lessons_learned": self._extract_lessons(),
            "risk_assessment": self._assess_decision_risk()
        }
        return analysis
    
    def _analyze_decision_summary(self) -> dict:
        """Analyze decision summary"""
        action = self.final_decision.get("action", "hold")
        position = self.final_decision.get("position", 0)
        confidence = self.final_decision.get("confidence", 5)
        
        # Calculate predicted change
        change_percent = self.prediction.get("change_percent", 0)
        
        # Judge decision aggressiveness
        if action in ["strong_buy", "strong_sell"]:
            aggressiveness = "aggressive"
        elif action in ["buy", "sell"]:
            aggressiveness = "moderate"
        else:
            aggressiveness = "conservative"
        
        # Judge confidence level
        if confidence >= 8:
            confidence_level = "high confidence"
        elif confidence >= 6:
            confidence_level = "moderate confidence"
        else:
            confidence_level = "relatively cautious"
        
        return {
            "action": action,
            "action_cn": self._action_to_chinese(action),
            "position": position,
            "confidence": confidence,
            "aggressiveness": aggressiveness,
            "confidence_level": confidence_level,
            "predicted_change": change_percent,
            "model_used": self.prediction.get("model_used", "unknown"),
            "model_confidence": self.prediction.get("confidence", 0)
        }
    
    def _analyze_analyst_performance(self) -> dict:
        """Analyze each analyst's performance"""
        performance = {}
        
        # Statistics recommendation distribution
        action_counts = {"strong_buy": 0, "buy": 0, "hold": 0, "sell": 0, "strong_sell": 0}
        
        for rec in self.recommendations:
            role = rec.get("role", "unknown")
            action = rec.get("action", "hold")
            confidence = rec.get("confidence", 5)
            
            action_counts[action] = action_counts.get(action, 0) + 1
            
            # Assess analyst performance
            final_action = self.final_decision.get("action", "hold")
            
            # Calculate alignment with final decision
            action_score = {
                "strong_buy": 2, "buy": 1, "hold": 0, "sell": -1, "strong_sell": -2
            }
            alignment = 1 - abs(action_score.get(action, 0) - action_score.get(final_action, 0)) / 4
            
            performance[role] = {
                "action": action,
                "action_cn": self._action_to_chinese(action),
                "confidence": confidence,
                "alignment_with_final": alignment,
                "influence": "high" if alignment > 0.7 else ("medium" if alignment > 0.4 else "low")
            }
        
        # Add overall statistics
        performance["_summary"] = {
            "action_distribution": action_counts,
            "majority_action": max(action_counts, key=action_counts.get),
            "consensus_strength": max(action_counts.values()) / len(self.recommendations) if self.recommendations else 0
        }
        
        return performance
    
    def _analyze_consensus(self) -> dict:
        """Analyze consensus level"""
        if not self.recommendations:
            return {"consensus_level": "no data", "details": []}
        
        # Calculate bullish/bearish/neutral distribution
        bullish = 0
        bearish = 0
        neutral = 0
        
        for rec in self.recommendations:
            action = rec.get("action", "hold")
            if action in ["strong_buy", "buy"]:
                bullish += 1
            elif action in ["strong_sell", "sell"]:
                bearish += 1
            else:
                neutral += 1
        
        total = len(self.recommendations)
        
        # Judge consensus level
        max_count = max(bullish, bearish, neutral)
        if max_count >= total * 0.8:
            consensus_level = "high consensus"
        elif max_count >= total * 0.6:
            consensus_level = "majority consensus"
        elif max_count >= total * 0.4:
            consensus_level = "significant divergence"
        else:
            consensus_level = "scattered"
        
        return {
            "consensus_level": consensus_level,
            "bullish_count": bullish,
            "bearish_count": bearish,
            "neutral_count": neutral,
            "bullish_ratio": bullish / total if total > 0 else 0,
            "bearish_ratio": bearish / total if total > 0 else 0,
            "neutral_ratio": neutral / total if total > 0 else 0,
            "dominant_view": "bullish" if bullish > bearish else ("bearish" if bearish > bullish else "neutral")
        }
    
    def _extract_key_insights(self) -> list:
        """Extract key insights"""
        insights = []
        
        # 1. Analyze prediction confidence
        model_confidence = self.prediction.get("confidence", 0)
        if model_confidence > 0.7:
            insights.append({
                "type": "model_confidence",
                "description": f"Model prediction confidence is high ({model_confidence:.1%}), prediction results are more reliable",
                "impact": "positive"
            })
        elif model_confidence < 0.5:
            insights.append({
                "type": "model_confidence",
                "description": f"Model prediction confidence is relatively low ({model_confidence:.1%}), recommend cautious reference",
                "impact": "warning"
            })
        
        # 2. Analyze consensus situation
        consensus = self._analyze_consensus()
        if consensus["consensus_level"] == "high consensus":
            insights.append({
                "type": "consensus",
                "description": f"Analysts reached high consensus ({consensus['dominant_view']})",
                "impact": "positive"
            })
        elif consensus["consensus_level"] == "significant divergence":
            insights.append({
                "type": "consensus",
                "description": "Analysts have significant divergence, market uncertainty is higher",
                "impact": "warning"
            })
        
        # 3. Analyze decision and prediction consistency
        change_percent = self.prediction.get("change_percent", 0)
        action = self.final_decision.get("action", "hold")
        
        is_consistent = (
            (change_percent > 5 and action in ["strong_buy", "buy"]) or
            (change_percent < -5 and action in ["strong_sell", "sell"]) or
            (abs(change_percent) <= 5 and action == "hold")
        )
        
        if is_consistent:
            insights.append({
                "type": "consistency",
                "description": "Final decision is consistent with model prediction direction",
                "impact": "positive"
            })
        else:
            insights.append({
                "type": "consistency",
                "description": "Final decision deviates from model prediction, may be based on human judgment adjustment",
                "impact": "neutral"
            })
        
        # 4. Analyze position recommendation
        position = self.final_decision.get("position", 0)
        if position > 70:
            insights.append({
                "type": "position",
                "description": f"Recommend heavy position ({position}%), indicates high confidence in decision",
                "impact": "info"
            })
        elif position < 30:
            insights.append({
                "type": "position",
                "description": f"Recommend light position ({position}%), indicates relatively cautious",
                "impact": "info"
            })
        
        return insights
    
    def _extract_lessons(self) -> dict:
        """Extract lessons learned"""
        lessons = {
            "technical": [],
            "strategy": [],
            "risk": [],
            "general": []
        }
        
        # Technical aspect
        model_used = self.prediction.get("model_used", "unknown")
        if model_used == "ensemble":
            lessons["technical"].append("Used multi-model ensemble prediction, improved prediction stability")
        elif model_used == "simulation":
            lessons["technical"].append("[!] Used simulated prediction, recommend checking model loading status")
        
        # Strategy aspect
        consensus = self._analyze_consensus()
        if consensus["consensus_level"] in ["significant divergence", "scattered"]:
            lessons["strategy"].append("When analysts diverge, final decision needs more caution")
            lessons["strategy"].append("Recommend lower position or batch operations to control risk")
        
        # Risk aspect
        confidence = self.final_decision.get("confidence", 5)
        position = self.final_decision.get("position", 0)
        
        if confidence < 6 and position > 50:
            lessons["risk"].append("[!] Confidence index relatively low but position is high, risk exposure may be too large")
        
        if confidence > 7 and position < 30:
            lessons["risk"].append("Confidence is high but position is conservative, may miss opportunities")
        
        # General experience
        lessons["general"].append("Regularly review decision process, continuously optimize analysis framework")
        lessons["general"].append("Track difference between model prediction and actual market movement")
        lessons["general"].append("Maintain diverse analysis, avoid groupthink")
        
        return lessons
    
    def _assess_decision_risk(self) -> dict:
        """Assess decision risk"""
        risk_score = 0
        risk_factors = []
        
        # 1. Model confidence risk
        model_confidence = self.prediction.get("confidence", 0)
        if model_confidence < 0.5:
            risk_score += 2
            risk_factors.append("Model prediction confidence relatively low")
        elif model_confidence < 0.7:
            risk_score += 1
            risk_factors.append("Model prediction confidence moderate")
        
        # 2. Consensus risk
        consensus = self._analyze_consensus()
        if consensus["consensus_level"] in ["significant divergence", "scattered"]:
            risk_score += 2
            risk_factors.append("Analysts have large divergence")
        
        # 3. Position risk
        position = self.final_decision.get("position", 0)
        if position > 70:
            risk_score += 1
            risk_factors.append("Position relatively heavy")
        
        # 4. Confidence and position mismatch risk
        confidence = self.final_decision.get("confidence", 5)
        if confidence < 6 and position > 50:
            risk_score += 2
            risk_factors.append("Confidence and position mismatch")
        
        # Calculate risk level
        if risk_score <= 1:
            risk_level = "low risk"
        elif risk_score <= 3:
            risk_level = "moderate risk"
        elif risk_score <= 5:
            risk_level = "higher risk"
        else:
            risk_level = "high risk"
        
        return {
            "risk_score": risk_score,
            "risk_level": risk_level,
            "risk_factors": risk_factors,
            "risk_mitigation": self._get_risk_mitigation(risk_factors)
        }
    
    def _get_risk_mitigation(self, risk_factors: list) -> list:
        """Get risk mitigation recommendations"""
        mitigations = []
        
        if "Model prediction confidence relatively low" in risk_factors:
            mitigations.append("Recommend waiting for more market signals to confirm")
        
        if "Analysts have large divergence" in risk_factors:
            mitigations.append("Recommend building positions in batches, control single investment")
        
        if "Position relatively heavy" in risk_factors:
            mitigations.append("Recommend setting strict stop-loss point")
        
        if "Confidence and position mismatch" in risk_factors:
            mitigations.append("Recommend lowering position or deepening analysis")
        
        if not mitigations:
            mitigations.append("Current risk is controllable, maintain normal operation")
        
        return mitigations
    
    def generate_report(self) -> str:
        """Generate readable review report"""
        analysis = self.analyze_decision()
        report_parts = []
        
        # Title
        report_parts.append("=" * 70)
        report_parts.append("         Investment Decision Review Report")
        report_parts.append("=" * 70)
        
        # Decision summary
        summary = analysis["decision_summary"]
        report_parts.append(f"\n[Decision Summary]")
        report_parts.append(f"  Investment Recommendation: {summary['action_cn']}")
        report_parts.append(f"  Recommended Position: {summary['position']}%")
        report_parts.append(f"  Confidence Index: {summary['confidence']}/10 ({summary['confidence_level']})")
        report_parts.append(f"  Decision Style: {summary['aggressiveness']}")
        report_parts.append(f"  Predicted Change: {summary['predicted_change']:.2f}%")
        report_parts.append(f"  Model Used: {summary['model_used']}")
        
        # Consensus analysis
        consensus = analysis["consensus_analysis"]
        report_parts.append(f"\n[Consensus Analysis]")
        report_parts.append(f"  Consensus Level: {consensus['consensus_level']}")
        report_parts.append(f"  Dominant View: {consensus['dominant_view']}")
        report_parts.append(f"  Bullish: {consensus['bullish_count']} ({consensus['bullish_ratio']:.0%})")
        report_parts.append(f"  Bearish: {consensus['bearish_count']} ({consensus['bearish_ratio']:.0%})")
        report_parts.append(f"  Neutral: {consensus['neutral_count']} ({consensus['neutral_ratio']:.0%})")
        
        # Key insights
        insights = analysis["key_insights"]
        if insights:
            report_parts.append(f"\n[Key Insights]")
            for insight in insights:
                icon = {"positive": "+", "warning": "!", "neutral": "-", "info": "*"}.get(insight["impact"], "*")
                report_parts.append(f"  [{icon}] {insight['description']}")
        
        # Risk assessment
        risk = analysis["risk_assessment"]
        report_parts.append(f"\n[Risk Assessment]")
        report_parts.append(f"  Risk Level: {risk['risk_level']} (Score: {risk['risk_score']})")
        if risk["risk_factors"]:
            report_parts.append(f"  Risk Factors:")
            for factor in risk["risk_factors"]:
                report_parts.append(f"    - {factor}")
        report_parts.append(f"  Mitigation Recommendations:")
        for mitigation in risk["risk_mitigation"]:
            report_parts.append(f"    - {mitigation}")
        
        # Lessons learned
        lessons = analysis["lessons_learned"]
        report_parts.append(f"\n[Lessons Learned]")
        if lessons["technical"]:
            report_parts.append(f"  Technical Aspect:")
            for lesson in lessons["technical"]:
                report_parts.append(f"    - {lesson}")
        if lessons["strategy"]:
            report_parts.append(f"  Strategy Aspect:")
            for lesson in lessons["strategy"]:
                report_parts.append(f"    - {lesson}")
        if lessons["risk"]:
            report_parts.append(f"  Risk Aspect:")
            for lesson in lessons["risk"]:
                report_parts.append(f"    - {lesson}")
        
        report_parts.append("\n" + "=" * 70)
        report_parts.append(f"  Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_parts.append("=" * 70)
        
        return "\n".join(report_parts)
    
    def save_to_memory(self, analyst_role: str) -> str:
        """Generate specific analyst experience memory"""
        analysis = self.analyze_decision()
        memory_parts = []
        
        # Get analyst performance
        performance = analysis["analyst_performance"].get(analyst_role, {})
        if performance:
            alignment = performance.get("alignment_with_final", 0)
            if alignment > 0.7:
                memory_parts.append(f"[+] This analysis was highly consistent with final decision")
            elif alignment < 0.4:
                memory_parts.append(f"[-] This analysis differed significantly from final decision, need to reflect on analysis angle")
        
        # Add consensus related experience
        consensus = analysis["consensus_analysis"]
        if consensus["consensus_level"] == "high consensus":
            memory_parts.append("Team reached high consensus, decision confidence is stronger")
        elif consensus["consensus_level"] == "significant divergence":
            memory_parts.append("When team diverges, need deeper analysis")
        
        # Add risk related experience
        risk = analysis["risk_assessment"]
        if risk["risk_level"] in ["higher risk", "high risk"]:
            memory_parts.append(f"Decision risk is higher, pay attention to risk control")
        
        # Add general experience
        lessons = analysis["lessons_learned"]
        memory_parts.extend(lessons.get("general", [])[:2])
        
        return " | ".join(memory_parts)
    
    def _action_to_chinese(self, action: str) -> str:
        """Convert action to display text"""
        mapping = {
            "strong_buy": "STRONG BUY",
            "buy": "BUY",
            "hold": "HOLD",
            "sell": "SELL",
            "strong_sell": "STRONG SELL"
        }
        return mapping.get(action, "HOLD")


class DecisionTracker:
    """
    Decision Tracker - Track historical decision accuracy
    
    Features:
    1. Record historical decisions
    2. Validate decision accuracy
    3. Calculate analyst accuracy
    4. Generate historical performance report
    """
    
    def __init__(self, history_dir: str = "investment_history"):
        self.history_dir = Path(history_dir)
        self.history_dir.mkdir(exist_ok=True)
        self.validation_file = self.history_dir / "decision_validations.json"
    
    def load_validations(self) -> list:
        """Load historical validation records"""
        if not self.validation_file.exists():
            return []
        
        try:
            with open(self.validation_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return []
    
    def save_validations(self, validations: list):
        """Save validation records"""
        with open(self.validation_file, 'w', encoding='utf-8') as f:
            json.dump(validations, f, ensure_ascii=False, indent=2)
    
    def validate_decision(self, decision_id: str, actual_change: float) -> dict:
        """
        Validate decision accuracy
        
        Parameters:
            decision_id: Decision ID
            actual_change: Actual price change percentage
        
        Returns:
            Validation results
        """
        # Load decision record
        decision_file = self.history_dir / f"decision_{decision_id}.json"
        if not decision_file.exists():
            return {"error": "Decision record not found"}
        
        with open(decision_file, 'r', encoding='utf-8') as f:
            decision_data = json.load(f)
        
        # Get prediction and decision
        predicted_change = decision_data.get("prediction", {}).get("change_percent", 0)
        action = decision_data.get("final_decision", {}).get("action", "hold")
        
        # Judge if direction is correct
        predicted_direction = "up" if predicted_change > 0 else ("down" if predicted_change < 0 else "neutral")
        actual_direction = "up" if actual_change > 0 else ("down" if actual_change < 0 else "neutral")
        direction_correct = predicted_direction == actual_direction
        
        # Judge if decision is correct
        action_direction = "up" if action in ["strong_buy", "buy"] else (
            "down" if action in ["strong_sell", "sell"] else "neutral"
        )
        decision_correct = action_direction == actual_direction
        
        # Calculate prediction error
        prediction_error = abs(predicted_change - actual_change)
        
        validation_result = {
            "decision_id": decision_id,
            "validation_time": datetime.now().isoformat(),
            "predicted_change": predicted_change,
            "actual_change": actual_change,
            "prediction_error": prediction_error,
            "direction_correct": direction_correct,
            "decision_correct": decision_correct,
            "action": action,
            "score": self._calculate_decision_score(
                predicted_change, actual_change, action, direction_correct, decision_correct
            )
        }
        
        # Save validation results
        validations = self.load_validations()
        validations.append(validation_result)
        self.save_validations(validations)
        
        return validation_result
    
    def _calculate_decision_score(
        self, predicted_change: float, actual_change: float,
        action: str, direction_correct: bool, decision_correct: bool
    ) -> float:
        """Calculate decision score (0-100)"""
        score = 50  # Base score
        
        # Direction correct bonus
        if direction_correct:
            score += 20
        
        # Decision correct bonus
        if decision_correct:
            score += 20
        
        # Prediction error penalty
        error = abs(predicted_change - actual_change)
        if error < 2:
            score += 10
        elif error < 5:
            score += 5
        elif error > 10:
            score -= 10
        
        return max(0, min(100, score))
    
    def get_historical_performance(self) -> dict:
        """Get historical performance statistics"""
        validations = self.load_validations()
        
        if not validations:
            return {"message": "No historical validation records"}
        
        total = len(validations)
        direction_correct = sum(1 for v in validations if v.get("direction_correct", False))
        decision_correct = sum(1 for v in validations if v.get("decision_correct", False))
        avg_score = sum(v.get("score", 50) for v in validations) / total
        avg_error = sum(v.get("prediction_error", 0) for v in validations) / total
        
        return {
            "total_decisions": total,
            "direction_accuracy": direction_correct / total,
            "decision_accuracy": decision_correct / total,
            "average_score": avg_score,
            "average_prediction_error": avg_error,
            "recent_validations": validations[-5:]
        }
    
    def generate_performance_report(self) -> str:
        """Generate historical performance report"""
        performance = self.get_historical_performance()
        
        if "message" in performance:
            return performance["message"]
        
        report = []
        report.append("=" * 60)
        report.append("         Historical Decision Performance Report")
        report.append("=" * 60)
        report.append(f"\nTotal Decisions: {performance['total_decisions']}")
        report.append(f"Direction Prediction Accuracy: {performance['direction_accuracy']:.1%}")
        report.append(f"Decision Accuracy: {performance['decision_accuracy']:.1%}")
        report.append(f"Average Decision Score: {performance['average_score']:.1f}/100")
        report.append(f"Average Prediction Error: {performance['average_prediction_error']:.2f}%")
        report.append("=" * 60)
        
        return "\n".join(report)


# Test
if __name__ == "__main__":
    # Simulate decision data
    test_decision_data = {
        "prediction": {
            "change_percent": 5.5,
            "confidence": 0.72,
            "model_used": "ensemble"
        },
        "final_decision": {
            "action": "buy",
            "position": 60,
            "confidence": 7
        },
        "recommendations": [
            {"role": "technical_analyst", "action": "buy", "confidence": 8},
            {"role": "industry_analyst", "action": "buy", "confidence": 7},
            {"role": "financial_analyst", "action": "hold", "confidence": 6},
            {"role": "market_expert", "action": "buy", "confidence": 7},
            {"role": "risk_analyst", "action": "hold", "confidence": 5}
        ],
        "analyses": [],
        "discussions": []
    }
    
    # Test analyzer
    analyzer = DecisionAnalyzer(test_decision_data)
    report = analyzer.generate_report()
    print(report)
