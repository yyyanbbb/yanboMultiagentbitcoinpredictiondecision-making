# -*- coding: utf-8 -*-
"""
Anonymous Peer Review System for Investment Decision Making

This module implements an innovative anonymous peer review mechanism where:
1. All analyst opinions are anonymized with labels (A/B/C/D/E)
2. Each analyst evaluates other analysts' opinions on "Accuracy" + "Insight"
3. A Devil's Advocate challenges the consensus view
4. The Chairman synthesizes all evaluations for final decision

This approach reduces:
- Groupthink and anchoring bias
- Authority bias (senior analysts having undue influence)
- Confirmation bias through forced critical evaluation

Inspired by:
- Academic peer review systems
- Delphi method for expert consensus
- Multi-agent debate mechanisms in LLM research
"""

import random
import string
import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np


class ReviewDimension(Enum):
    """Dimensions for peer review evaluation"""
    ACCURACY = "accuracy"           # How accurate is the analysis based on data?
    INSIGHT = "insight"             # How insightful and novel are the perspectives?
    LOGIC = "logic"                 # How logical and coherent is the reasoning?
    RISK_AWARENESS = "risk_awareness"  # How well are risks identified?
    ACTIONABILITY = "actionability"    # How actionable are the recommendations?


@dataclass
class AnonymousAnalysis:
    """Anonymized analysis container"""
    anonymous_id: str              # e.g., "Analyst A", "Analyst B"
    original_role: str             # Hidden: actual role
    original_analyst_id: int       # Hidden: actual analyst id
    content: str                   # The analysis content
    recommendation: str            # buy/sell/hold
    confidence: int                # 1-10
    key_points: List[str]          # Main arguments
    
    def get_display_content(self) -> str:
        """Get content for display to other reviewers (without role info)"""
        return f"""
========================================
{self.anonymous_id}
========================================
Analysis:
{self.content}

Recommendation: {self.recommendation.upper()}
Confidence: {self.confidence}/10

Key Arguments:
{chr(10).join(f"  - {point}" for point in self.key_points)}
========================================
"""


@dataclass  
class PeerReview:
    """A single peer review"""
    reviewer_id: str               # Anonymous ID of reviewer
    reviewee_id: str               # Anonymous ID of reviewee
    scores: Dict[str, int]         # Dimension -> Score (1-10)
    strengths: List[str]           # What's good about this analysis
    weaknesses: List[str]          # What's weak about this analysis
    critical_questions: List[str]  # Questions that challenge the analysis
    overall_comment: str           # Summary comment
    
    def get_average_score(self) -> float:
        """Calculate weighted average score"""
        weights = {
            "accuracy": 0.30,
            "insight": 0.25,
            "logic": 0.20,
            "risk_awareness": 0.15,
            "actionability": 0.10
        }
        total = 0
        weight_sum = 0
        for dim, score in self.scores.items():
            w = weights.get(dim, 0.1)
            total += score * w
            weight_sum += w
        return total / weight_sum if weight_sum > 0 else 5.0


@dataclass
class DevilsAdvocateChallenge:
    """Devil's Advocate challenge to consensus"""
    challenger_id: str
    consensus_view: str            # The majority view being challenged
    counter_arguments: List[str]   # Arguments against consensus
    alternative_scenarios: List[Dict]  # What if the consensus is wrong?
    risk_warnings: List[str]       # Risks that consensus may overlook
    confidence_in_challenge: int   # How confident is the devil's advocate? (1-10)
    
    def get_display(self) -> str:
        return f"""
╔══════════════════════════════════════════════════════════════════╗
║                    DEVIL'S ADVOCATE CHALLENGE                     ║
╠══════════════════════════════════════════════════════════════════╣
║ Challenging Consensus: {self.consensus_view[:40]}...
╠══════════════════════════════════════════════════════════════════╣
║ COUNTER-ARGUMENTS:
{chr(10).join(f"║   ⚡ {arg}" for arg in self.counter_arguments)}
╠══════════════════════════════════════════════════════════════════╣
║ ALTERNATIVE SCENARIOS:
{chr(10).join(f"║   🔄 {s.get('scenario', 'N/A')}: {s.get('impact', 'N/A')}" for s in self.alternative_scenarios)}
╠══════════════════════════════════════════════════════════════════╣
║ RISK WARNINGS:
{chr(10).join(f"║   ⚠️  {risk}" for risk in self.risk_warnings)}
╠══════════════════════════════════════════════════════════════════╣
║ Challenge Confidence: {self.confidence_in_challenge}/10
╚══════════════════════════════════════════════════════════════════╝
"""


@dataclass
class RankedAnalysis:
    """Analysis ranked by peer review scores"""
    anonymous_id: str
    original_role: str
    average_score: float
    rank: int
    total_reviews: int
    score_breakdown: Dict[str, float]  # Average score per dimension
    consensus_with_others: float       # How much this aligns with other analysts
    key_strengths: List[str]
    key_weaknesses: List[str]


class AnonymousPeerReviewSystem:
    """
    Main system for anonymous peer review of investment analyses
    
    Workflow:
    1. Collect all analyst analyses
    2. Anonymize with random labels (A, B, C, D, E, F)
    3. Each analyst reviews all other analyses
    4. Aggregate scores and rank analyses
    5. Devil's Advocate challenges the consensus
    6. Chairman synthesizes everything for final decision
    """
    
    def __init__(self, llm_client=None):
        """
        Initialize the peer review system
        
        Args:
            llm_client: Optional LLM client for generating reviews
        """
        self.llm_client = llm_client
        self.anonymous_analyses: Dict[str, AnonymousAnalysis] = {}
        self.peer_reviews: List[PeerReview] = []
        self.id_mapping: Dict[str, Tuple[str, int]] = {}  # anonymous_id -> (role, analyst_id)
        self.reverse_mapping: Dict[int, str] = {}  # analyst_id -> anonymous_id
        self.devils_advocate_challenge: Optional[DevilsAdvocateChallenge] = None
        self.final_ranking: List[RankedAnalysis] = []
        
    def anonymize_analyses(self, analyses: List[Dict]) -> List[AnonymousAnalysis]:
        """
        Anonymize all analyst analyses with random labels
        
        Args:
            analyses: List of analysis dicts with keys:
                - role: str
                - analyst_id: int
                - content: str
                - recommendation: str
                - confidence: int
                - key_points: List[str]
        
        Returns:
            List of AnonymousAnalysis objects
        """
        # Generate random labels
        labels = list("ABCDEFGHIJ")[:len(analyses)]
        random.shuffle(labels)
        
        anonymized = []
        for i, analysis in enumerate(analyses):
            anon_id = f"Analyst {labels[i]}"
            
            # Store mappings (hidden from review process)
            self.id_mapping[anon_id] = (analysis["role"], analysis["analyst_id"])
            self.reverse_mapping[analysis["analyst_id"]] = anon_id
            
            anon_analysis = AnonymousAnalysis(
                anonymous_id=anon_id,
                original_role=analysis["role"],
                original_analyst_id=analysis["analyst_id"],
                content=analysis["content"],
                recommendation=analysis.get("recommendation", "hold"),
                confidence=analysis.get("confidence", 5),
                key_points=analysis.get("key_points", [])
            )
            
            self.anonymous_analyses[anon_id] = anon_analysis
            anonymized.append(anon_analysis)
        
        return anonymized
    
    def generate_review_prompt(self, reviewer_analysis: AnonymousAnalysis, 
                                reviewee_analysis: AnonymousAnalysis) -> str:
        """Generate prompt for peer review"""
        return f"""You are {reviewer_analysis.anonymous_id} conducting an ANONYMOUS peer review.

You must evaluate {reviewee_analysis.anonymous_id}'s analysis OBJECTIVELY.
You do NOT know who this analyst is - evaluate purely on merit.

=== ANALYSIS TO REVIEW ===
{reviewee_analysis.get_display_content()}

=== YOUR TASK ===
Evaluate this analysis on the following dimensions (score 1-10):

1. ACCURACY (30% weight): How accurate is the analysis based on the data provided?
   - Are the interpretations of technical indicators correct?
   - Are the predictions reasonable given the evidence?

2. INSIGHT (25% weight): How insightful and novel are the perspectives?
   - Does the analysis offer unique viewpoints?
   - Are there creative connections or observations?

3. LOGIC (20% weight): How logical and coherent is the reasoning?
   - Is the argument flow clear?
   - Are conclusions well-supported by premises?

4. RISK_AWARENESS (15% weight): How well are risks identified?
   - Are potential downsides acknowledged?
   - Is there proper consideration of uncertainty?

5. ACTIONABILITY (10% weight): How actionable are the recommendations?
   - Can the advice be practically implemented?
   - Are entry/exit points clear?

Respond in this EXACT JSON format:
{{
    "scores": {{
        "accuracy": <1-10>,
        "insight": <1-10>,
        "logic": <1-10>,
        "risk_awareness": <1-10>,
        "actionability": <1-10>
    }},
    "strengths": ["<strength 1>", "<strength 2>"],
    "weaknesses": ["<weakness 1>", "<weakness 2>"],
    "critical_questions": ["<question that challenges the analysis>"],
    "overall_comment": "<2-3 sentence summary of your review>"
}}

BE OBJECTIVE AND CRITICAL. Do not give all high scores. Identify real weaknesses.
"""

    def generate_devils_advocate_prompt(self, consensus_view: str, 
                                         all_analyses: List[AnonymousAnalysis]) -> str:
        """Generate prompt for Devil's Advocate challenge"""
        analyses_summary = "\n\n".join([
            f"[{a.anonymous_id}] Recommends: {a.recommendation.upper()} "
            f"(Confidence: {a.confidence}/10)\nKey points: {'; '.join(a.key_points[:3])}"
            for a in all_analyses
        ])
        
        return f"""You are the DEVIL'S ADVOCATE in an investment committee.

Your SOLE PURPOSE is to CHALLENGE the consensus view and find potential flaws.
You MUST argue AGAINST the majority opinion, even if you personally agree with it.

=== CURRENT CONSENSUS ===
{consensus_view}

=== ALL ANALYST POSITIONS ===
{analyses_summary}

=== YOUR MISSION ===
1. Identify weaknesses in the consensus reasoning
2. Present counter-arguments that the majority may have overlooked
3. Describe alternative scenarios where the consensus is WRONG
4. Highlight risks that aren't being adequately considered

Respond in this EXACT JSON format:
{{
    "counter_arguments": [
        "<compelling argument against consensus>",
        "<another counter-argument>",
        "<third counter-argument>"
    ],
    "alternative_scenarios": [
        {{"scenario": "<what if X happens>", "probability": "<low/medium/high>", "impact": "<description of impact>"}},
        {{"scenario": "<what if Y happens>", "probability": "<low/medium/high>", "impact": "<description of impact>"}}
    ],
    "risk_warnings": [
        "<risk the consensus is ignoring>",
        "<another overlooked risk>"
    ],
    "confidence_in_challenge": <1-10 how confident are you that consensus could be wrong>
}}

BE AGGRESSIVE in your challenge. Your job is to stress-test the consensus.
"""

    def conduct_peer_reviews(self, use_llm: bool = True) -> List[PeerReview]:
        """
        Have each analyst review all other analyses
        
        Args:
            use_llm: If True, use LLM to generate reviews. If False, use simulation.
        
        Returns:
            List of all peer reviews
        """
        all_reviews = []
        analyses_list = list(self.anonymous_analyses.values())
        
        for reviewer in analyses_list:
            for reviewee in analyses_list:
                if reviewer.anonymous_id == reviewee.anonymous_id:
                    continue  # Don't review yourself
                
                if use_llm and self.llm_client:
                    review = self._generate_llm_review(reviewer, reviewee)
                else:
                    review = self._simulate_review(reviewer, reviewee)
                
                all_reviews.append(review)
        
        self.peer_reviews = all_reviews
        return all_reviews
    
    def _simulate_review(self, reviewer: AnonymousAnalysis, 
                         reviewee: AnonymousAnalysis) -> PeerReview:
        """Simulate a peer review (for testing or when LLM is unavailable)"""
        # Base scores with some randomness
        base_score = 6 + random.gauss(0, 1.5)
        
        # Adjust based on recommendation alignment
        if reviewer.recommendation == reviewee.recommendation:
            alignment_bonus = random.uniform(0.5, 1.5)
        else:
            alignment_bonus = random.uniform(-1.0, 0.5)
        
        scores = {}
        for dim in ReviewDimension:
            score = base_score + alignment_bonus + random.gauss(0, 0.8)
            scores[dim.value] = max(1, min(10, round(score)))
        
        strengths = [
            f"Clear presentation of {reviewee.recommendation} rationale",
            f"Confidence level ({reviewee.confidence}/10) appropriately justified"
        ]
        
        weaknesses = []
        if reviewee.confidence > 7:
            weaknesses.append("May be overconfident given market uncertainty")
        if len(reviewee.key_points) < 3:
            weaknesses.append("Could provide more supporting arguments")
        if not weaknesses:
            weaknesses.append("Limited consideration of alternative scenarios")
        
        critical_questions = [
            f"What would change your {reviewee.recommendation} recommendation?",
            "How would you respond if the market moves against your prediction?"
        ]
        
        return PeerReview(
            reviewer_id=reviewer.anonymous_id,
            reviewee_id=reviewee.anonymous_id,
            scores=scores,
            strengths=strengths,
            weaknesses=weaknesses,
            critical_questions=critical_questions,
            overall_comment=f"Analysis shows {['weak', 'moderate', 'solid', 'strong'][min(3, scores['accuracy']//3)]} "
                          f"reasoning with {['limited', 'adequate', 'good', 'excellent'][min(3, scores['insight']//3)]} insights."
        )
    
    def _generate_llm_review(self, reviewer: AnonymousAnalysis, 
                             reviewee: AnonymousAnalysis) -> PeerReview:
        """Generate peer review using LLM"""
        prompt = self.generate_review_prompt(reviewer, reviewee)
        
        try:
            response = self.llm_client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": "You are an objective peer reviewer in an investment committee."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=1000
            )
            
            content = response.choices[0].message.content
            # Parse JSON from response
            import re
            json_match = re.search(r'\{[\s\S]*\}', content)
            if json_match:
                data = json.loads(json_match.group())
                return PeerReview(
                    reviewer_id=reviewer.anonymous_id,
                    reviewee_id=reviewee.anonymous_id,
                    scores=data.get("scores", {}),
                    strengths=data.get("strengths", []),
                    weaknesses=data.get("weaknesses", []),
                    critical_questions=data.get("critical_questions", []),
                    overall_comment=data.get("overall_comment", "")
                )
        except Exception as e:
            print(f"[!] LLM review generation failed: {e}")
        
        # Fallback to simulation
        return self._simulate_review(reviewer, reviewee)
    
    def calculate_rankings(self) -> List[RankedAnalysis]:
        """
        Calculate final rankings based on all peer reviews
        
        Returns:
            List of RankedAnalysis sorted by score (highest first)
        """
        if not self.peer_reviews:
            return []
        
        # Aggregate scores per analyst
        analyst_scores: Dict[str, List[PeerReview]] = {}
        for review in self.peer_reviews:
            if review.reviewee_id not in analyst_scores:
                analyst_scores[review.reviewee_id] = []
            analyst_scores[review.reviewee_id].append(review)
        
        rankings = []
        for anon_id, reviews in analyst_scores.items():
            if anon_id not in self.anonymous_analyses:
                continue
            
            analysis = self.anonymous_analyses[anon_id]
            
            # Calculate average scores per dimension
            dimension_scores: Dict[str, List[float]] = {d.value: [] for d in ReviewDimension}
            for review in reviews:
                for dim, score in review.scores.items():
                    if dim in dimension_scores:
                        dimension_scores[dim].append(score)
            
            avg_dimension_scores = {
                dim: np.mean(scores) if scores else 5.0 
                for dim, scores in dimension_scores.items()
            }
            
            # Calculate overall average
            overall_avg = np.mean([r.get_average_score() for r in reviews])
            
            # Collect all strengths and weaknesses
            all_strengths = []
            all_weaknesses = []
            for review in reviews:
                all_strengths.extend(review.strengths)
                all_weaknesses.extend(review.weaknesses)
            
            # Calculate consensus alignment
            recommendations = [a.recommendation for a in self.anonymous_analyses.values()]
            consensus_count = recommendations.count(analysis.recommendation)
            consensus_alignment = consensus_count / len(recommendations)
            
            rankings.append(RankedAnalysis(
                anonymous_id=anon_id,
                original_role=analysis.original_role,
                average_score=overall_avg,
                rank=0,  # Will be set after sorting
                total_reviews=len(reviews),
                score_breakdown=avg_dimension_scores,
                consensus_with_others=consensus_alignment,
                key_strengths=list(set(all_strengths))[:5],
                key_weaknesses=list(set(all_weaknesses))[:5]
            ))
        
        # Sort and assign ranks
        rankings.sort(key=lambda x: x.average_score, reverse=True)
        for i, r in enumerate(rankings):
            r.rank = i + 1
        
        self.final_ranking = rankings
        return rankings
    
    def generate_devils_advocate_challenge(self, use_llm: bool = True) -> DevilsAdvocateChallenge:
        """
        Generate a Devil's Advocate challenge to the consensus
        
        Args:
            use_llm: Whether to use LLM for generating challenge
        
        Returns:
            DevilsAdvocateChallenge object
        """
        # Determine consensus
        recommendations = [a.recommendation for a in self.anonymous_analyses.values()]
        consensus_rec = max(set(recommendations), key=recommendations.count)
        consensus_count = recommendations.count(consensus_rec)
        total = len(recommendations)
        
        consensus_view = f"{consensus_rec.upper()} ({consensus_count}/{total} analysts agree)"
        
        if use_llm and self.llm_client:
            challenge = self._generate_llm_challenge(consensus_view)
        else:
            challenge = self._simulate_challenge(consensus_view, consensus_rec)
        
        self.devils_advocate_challenge = challenge
        return challenge
    
    def _simulate_challenge(self, consensus_view: str, consensus_rec: str) -> DevilsAdvocateChallenge:
        """Simulate Devil's Advocate challenge"""
        counter_args = {
            "buy": [
                "The market may be in a bull trap - recent gains could reverse sharply",
                "Institutional interest might be peaking, suggesting smart money is selling",
                "Technical indicators may lag behind fundamental deterioration"
            ],
            "sell": [
                "Oversold conditions often precede strong rebounds",
                "Negative sentiment extremes historically mark bottoms",
                "Selling pressure may be exhausted after recent declines"
            ],
            "hold": [
                "Waiting for clarity may mean missing significant moves",
                "Low volatility periods often precede major breakouts",
                "The opportunity cost of inaction could be substantial"
            ]
        }
        
        scenarios = [
            {
                "scenario": "Black swan event (exchange hack, regulatory ban)",
                "probability": "low",
                "impact": "Could cause 30-50% immediate drop regardless of analysis"
            },
            {
                "scenario": "Major institutional adoption announcement",
                "probability": "medium",
                "impact": "Could trigger 20-40% rally within days"
            },
            {
                "scenario": "Correlation breakdown with traditional markets",
                "probability": "medium",
                "impact": "Technical analysis may become temporarily unreliable"
            }
        ]
        
        risk_warnings = [
            "Model predictions have inherent uncertainty that may be underestimated",
            "Market regime changes can invalidate historical patterns",
            "Liquidity conditions can amplify moves beyond predictions"
        ]
        
        return DevilsAdvocateChallenge(
            challenger_id="Devil's Advocate",
            consensus_view=consensus_view,
            counter_arguments=counter_args.get(consensus_rec, counter_args["hold"]),
            alternative_scenarios=scenarios,
            risk_warnings=risk_warnings,
            confidence_in_challenge=random.randint(5, 8)
        )
    
    def _generate_llm_challenge(self, consensus_view: str) -> DevilsAdvocateChallenge:
        """Generate Devil's Advocate challenge using LLM"""
        prompt = self.generate_devils_advocate_prompt(
            consensus_view, 
            list(self.anonymous_analyses.values())
        )
        
        try:
            response = self.llm_client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": "You are a Devil's Advocate who must challenge the consensus view."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.8,
                max_tokens=1500
            )
            
            content = response.choices[0].message.content
            import re
            json_match = re.search(r'\{[\s\S]*\}', content)
            if json_match:
                data = json.loads(json_match.group())
                return DevilsAdvocateChallenge(
                    challenger_id="Devil's Advocate",
                    consensus_view=consensus_view,
                    counter_arguments=data.get("counter_arguments", []),
                    alternative_scenarios=data.get("alternative_scenarios", []),
                    risk_warnings=data.get("risk_warnings", []),
                    confidence_in_challenge=data.get("confidence_in_challenge", 5)
                )
        except Exception as e:
            print(f"[!] LLM challenge generation failed: {e}")
        
        # Fallback
        recommendations = [a.recommendation for a in self.anonymous_analyses.values()]
        consensus_rec = max(set(recommendations), key=recommendations.count)
        return self._simulate_challenge(consensus_view, consensus_rec)
    
    def generate_chairman_synthesis_prompt(self) -> str:
        """Generate prompt for Chairman's final synthesis"""
        # Rankings summary
        rankings_str = "\n".join([
            f"  #{r.rank}. {r.anonymous_id} (Score: {r.average_score:.1f}/10)\n"
            f"      Recommendation: {self.anonymous_analyses[r.anonymous_id].recommendation.upper()}\n"
            f"      Strengths: {', '.join(r.key_strengths[:2])}\n"
            f"      Weaknesses: {', '.join(r.key_weaknesses[:2])}"
            for r in self.final_ranking
        ])
        
        # Devil's advocate summary
        da_str = ""
        if self.devils_advocate_challenge:
            da = self.devils_advocate_challenge
            da_str = f"""
DEVIL'S ADVOCATE CHALLENGE:
  Challenging: {da.consensus_view}
  Counter-arguments:
{chr(10).join(f"    - {arg}" for arg in da.counter_arguments[:3])}
  Key risks:
{chr(10).join(f"    - {risk}" for risk in da.risk_warnings[:3])}
  Challenge confidence: {da.confidence_in_challenge}/10
"""
        
        return f"""You are the CHAIRMAN of the Investment Committee.

You must synthesize ALL inputs to make the FINAL investment decision.

=== PEER-REVIEWED ANALYST RANKINGS ===
{rankings_str}

{da_str}

=== YOUR TASK ===
As Chairman, consider:
1. The peer-reviewed scores reflect the quality of each analysis
2. Higher-ranked analyses should carry more weight
3. The Devil's Advocate challenge identifies potential blind spots
4. Consensus is important but not sufficient - quality matters

Provide your FINAL SYNTHESIS:
1. Which analyses were most valuable and why?
2. How does the Devil's Advocate challenge affect your decision?
3. What is your FINAL recommendation (STRONG_BUY/BUY/HOLD/SELL/STRONG_SELL)?
4. Recommended position size (0-100%)
5. Key factors driving your decision
6. Confidence level (1-10)

Be decisive but acknowledge uncertainty.
"""

    def get_final_synthesis(self, use_llm: bool = True) -> Dict:
        """
        Generate Chairman's final synthesis of all reviews and challenges
        
        Returns:
            Dict with final decision and reasoning
        """
        if not self.final_ranking:
            self.calculate_rankings()
        
        if not self.devils_advocate_challenge:
            self.generate_devils_advocate_challenge(use_llm=use_llm)
        
        if use_llm and self.llm_client:
            return self._generate_llm_synthesis()
        else:
            return self._simulate_synthesis()
    
    def _simulate_synthesis(self) -> Dict:
        """Simulate Chairman's synthesis"""
        if not self.final_ranking:
            return {"error": "No rankings available"}
        
        # Weight recommendations by rank
        weighted_recs = {"buy": 0, "sell": 0, "hold": 0}
        for r in self.final_ranking:
            rec = self.anonymous_analyses[r.anonymous_id].recommendation.lower()
            weight = (len(self.final_ranking) - r.rank + 1) * r.average_score
            if "buy" in rec:
                weighted_recs["buy"] += weight
            elif "sell" in rec:
                weighted_recs["sell"] += weight
            else:
                weighted_recs["hold"] += weight
        
        # Determine final recommendation
        final_rec = max(weighted_recs, key=weighted_recs.get)
        
        # Adjust confidence based on devil's advocate
        base_confidence = 7
        if self.devils_advocate_challenge:
            da_conf = self.devils_advocate_challenge.confidence_in_challenge
            if da_conf >= 7:
                base_confidence -= 1
            elif da_conf <= 3:
                base_confidence += 1
        
        # Calculate position
        top_score = self.final_ranking[0].average_score if self.final_ranking else 5
        position = min(80, max(20, int(top_score * 8)))
        
        return {
            "final_recommendation": final_rec,
            "confidence": base_confidence,
            "position_percent": position,
            "top_analyst": self.final_ranking[0].anonymous_id if self.final_ranking else "N/A",
            "top_analyst_role": self.final_ranking[0].original_role if self.final_ranking else "N/A",
            "reasoning": f"Decision based on peer-reviewed analysis quality. "
                        f"Top-ranked analyst ({self.final_ranking[0].anonymous_id}) "
                        f"scored {self.final_ranking[0].average_score:.1f}/10. "
                        f"Devil's Advocate challenge confidence: "
                        f"{self.devils_advocate_challenge.confidence_in_challenge if self.devils_advocate_challenge else 'N/A'}/10",
            "key_factors": [
                f"Quality-weighted consensus: {final_rec.upper()}",
                f"Top analyst score: {top_score:.1f}/10",
                f"Challenge to consensus: {self.devils_advocate_challenge.confidence_in_challenge if self.devils_advocate_challenge else 5}/10"
            ]
        }
    
    def _generate_llm_synthesis(self) -> Dict:
        """Generate synthesis using LLM"""
        prompt = self.generate_chairman_synthesis_prompt()
        
        try:
            response = self.llm_client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": "You are the Chairman of an Investment Committee making the final decision."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.5,
                max_tokens=2000
            )
            
            content = response.choices[0].message.content
            
            # Parse the response (this would need more sophisticated parsing in production)
            return {
                "final_recommendation": self._extract_recommendation(content),
                "full_synthesis": content,
                "confidence": self._extract_confidence(content),
                "reasoning": content
            }
        except Exception as e:
            print(f"[!] LLM synthesis failed: {e}")
            return self._simulate_synthesis()
    
    def _extract_recommendation(self, text: str) -> str:
        """Extract recommendation from text"""
        text_upper = text.upper()
        if "STRONG_BUY" in text_upper or "STRONG BUY" in text_upper:
            return "strong_buy"
        elif "STRONG_SELL" in text_upper or "STRONG SELL" in text_upper:
            return "strong_sell"
        elif "BUY" in text_upper:
            return "buy"
        elif "SELL" in text_upper:
            return "sell"
        return "hold"
    
    def _extract_confidence(self, text: str) -> int:
        """Extract confidence level from text"""
        import re
        matches = re.findall(r'confidence[:\s]*(\d+)', text.lower())
        if matches:
            return min(10, max(1, int(matches[0])))
        return 6
    
    def generate_full_report(self) -> str:
        """Generate complete peer review report"""
        report_lines = [
            "=" * 70,
            "          ANONYMOUS PEER REVIEW ANALYSIS REPORT",
            "=" * 70,
            "",
            f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Total Analysts: {len(self.anonymous_analyses)}",
            f"Total Reviews Conducted: {len(self.peer_reviews)}",
            "",
            "-" * 70,
            "                     ANALYST RANKINGS",
            "-" * 70,
        ]
        
        for r in self.final_ranking:
            report_lines.extend([
                f"",
                f"  #{r.rank}. {r.anonymous_id}",
                f"      Original Role: {r.original_role}",
                f"      Overall Score: {r.average_score:.2f}/10",
                f"      Reviews Received: {r.total_reviews}",
                f"      Consensus Alignment: {r.consensus_with_others:.0%}",
                f"      Score Breakdown:",
            ])
            for dim, score in r.score_breakdown.items():
                report_lines.append(f"        - {dim.replace('_', ' ').title()}: {score:.1f}")
            
            report_lines.append(f"      Key Strengths:")
            for s in r.key_strengths[:3]:
                report_lines.append(f"        + {s}")
            
            report_lines.append(f"      Key Weaknesses:")
            for w in r.key_weaknesses[:3]:
                report_lines.append(f"        - {w}")
        
        if self.devils_advocate_challenge:
            report_lines.extend([
                "",
                "-" * 70,
                "                  DEVIL'S ADVOCATE CHALLENGE",
                "-" * 70,
                self.devils_advocate_challenge.get_display(),
            ])
        
        synthesis = self.get_final_synthesis(use_llm=False)
        report_lines.extend([
            "",
            "-" * 70,
            "                    CHAIRMAN'S SYNTHESIS",
            "-" * 70,
            f"",
            f"  Final Recommendation: {synthesis.get('final_recommendation', 'N/A').upper()}",
            f"  Confidence Level: {synthesis.get('confidence', 'N/A')}/10",
            f"  Recommended Position: {synthesis.get('position_percent', 'N/A')}%",
            f"",
            f"  Key Factors:",
        ])
        for factor in synthesis.get("key_factors", []):
            report_lines.append(f"    - {factor}")
        
        report_lines.extend([
            "",
            "=" * 70,
            "                      END OF REPORT",
            "=" * 70,
        ])
        
        return "\n".join(report_lines)


class DynamicWeightAdjuster:
    """
    Adjusts analyst weights based on historical prediction accuracy
    
    This implements a learning mechanism where analysts who consistently
    provide accurate predictions gain more influence over time.
    """
    
    def __init__(self, history_file: str = "analyst_performance_history.json"):
        self.history_file = history_file
        self.performance_history: Dict[str, List[Dict]] = {}
        self.load_history()
    
    def load_history(self):
        """Load performance history from file"""
        try:
            with open(self.history_file, 'r') as f:
                self.performance_history = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            self.performance_history = {}
    
    def save_history(self):
        """Save performance history to file"""
        with open(self.history_file, 'w') as f:
            json.dump(self.performance_history, f, indent=2)
    
    def record_prediction(self, analyst_role: str, prediction: Dict):
        """Record an analyst's prediction for later validation"""
        if analyst_role not in self.performance_history:
            self.performance_history[analyst_role] = []
        
        self.performance_history[analyst_role].append({
            "timestamp": datetime.now().isoformat(),
            "prediction": prediction,
            "validated": False,
            "accuracy": None
        })
        self.save_history()
    
    def validate_prediction(self, analyst_role: str, prediction_index: int, 
                           actual_outcome: Dict) -> float:
        """
        Validate a previous prediction against actual outcome
        
        Returns:
            Accuracy score (0-1)
        """
        if analyst_role not in self.performance_history:
            return 0.5
        
        if prediction_index >= len(self.performance_history[analyst_role]):
            return 0.5
        
        pred = self.performance_history[analyst_role][prediction_index]
        
        # Calculate accuracy based on direction
        pred_direction = pred["prediction"].get("recommendation", "hold")
        actual_change = actual_outcome.get("actual_change_percent", 0)
        
        direction_correct = (
            (pred_direction in ["buy", "strong_buy"] and actual_change > 0) or
            (pred_direction in ["sell", "strong_sell"] and actual_change < 0) or
            (pred_direction == "hold" and abs(actual_change) < 2)
        )
        
        # More nuanced scoring
        if direction_correct:
            # Bonus for confidence alignment
            pred_conf = pred["prediction"].get("confidence", 5)
            if abs(actual_change) > 10:  # Big move
                accuracy = 0.8 + (pred_conf / 10) * 0.2  # Higher confidence rewarded
            else:
                accuracy = 0.6 + (pred_conf / 10) * 0.2
        else:
            accuracy = 0.3 - (pred["prediction"].get("confidence", 5) / 10) * 0.2
        
        # Update record
        pred["validated"] = True
        pred["accuracy"] = accuracy
        self.save_history()
        
        return accuracy
    
    def calculate_dynamic_weights(self) -> Dict[str, float]:
        """
        Calculate dynamic weights based on historical accuracy
        
        Returns:
            Dict of analyst_role -> weight (0-2, where 1 is baseline)
        """
        weights = {}
        
        for analyst_role, history in self.performance_history.items():
            validated = [h for h in history if h.get("validated", False)]
            
            if not validated:
                weights[analyst_role] = 1.0  # Baseline
                continue
            
            # Recent predictions weighted more heavily
            recent_accuracy = []
            for i, h in enumerate(validated[-10:]):  # Last 10 predictions
                recency_weight = (i + 1) / 10  # More recent = higher weight
                recent_accuracy.append(h.get("accuracy", 0.5) * recency_weight)
            
            avg_accuracy = np.mean(recent_accuracy) if recent_accuracy else 0.5
            
            # Convert accuracy to weight (0.5 -> 0.5, 1.0 -> 2.0)
            weights[analyst_role] = 0.5 + avg_accuracy * 1.5
        
        return weights
    
    def get_analyst_stats(self, analyst_role: str) -> Dict:
        """Get detailed statistics for an analyst"""
        if analyst_role not in self.performance_history:
            return {"error": "No history for this analyst"}
        
        history = self.performance_history[analyst_role]
        validated = [h for h in history if h.get("validated", False)]
        
        if not validated:
            return {
                "total_predictions": len(history),
                "validated_predictions": 0,
                "average_accuracy": "N/A",
                "current_weight": 1.0
            }
        
        accuracies = [h.get("accuracy", 0.5) for h in validated]
        
        return {
            "total_predictions": len(history),
            "validated_predictions": len(validated),
            "average_accuracy": np.mean(accuracies),
            "recent_accuracy": np.mean(accuracies[-5:]) if len(accuracies) >= 5 else np.mean(accuracies),
            "best_accuracy": max(accuracies),
            "worst_accuracy": min(accuracies),
            "current_weight": self.calculate_dynamic_weights().get(analyst_role, 1.0)
        }


# Test function
if __name__ == "__main__":
    print("Testing Anonymous Peer Review System...")
    
    # Create sample analyses
    sample_analyses = [
        {
            "role": "technical_analyst",
            "analyst_id": 1,
            "content": "Technical indicators suggest bullish momentum. RSI at 55, MACD showing positive divergence.",
            "recommendation": "buy",
            "confidence": 7,
            "key_points": ["RSI neutral-bullish", "MACD positive", "Price above 20-day MA"]
        },
        {
            "role": "industry_analyst", 
            "analyst_id": 2,
            "content": "Industry fundamentals remain strong. Institutional adoption continues.",
            "recommendation": "buy",
            "confidence": 6,
            "key_points": ["ETF inflows positive", "Regulatory clarity improving"]
        },
        {
            "role": "financial_analyst",
            "analyst_id": 3,
            "content": "Risk-reward ratio is balanced. Current valuations are fair.",
            "recommendation": "hold",
            "confidence": 5,
            "key_points": ["Valuation at historical average", "Risk metrics acceptable"]
        },
        {
            "role": "risk_analyst",
            "analyst_id": 4,
            "content": "Volatility remains elevated. Recommend caution.",
            "recommendation": "hold",
            "confidence": 4,
            "key_points": ["High volatility environment", "Geopolitical risks present"]
        },
        {
            "role": "market_expert",
            "analyst_id": 5,
            "content": "Market sentiment turning positive. Retail interest increasing.",
            "recommendation": "buy",
            "confidence": 7,
            "key_points": ["Sentiment shift detected", "Volume increasing"]
        }
    ]
    
    # Initialize system
    system = AnonymousPeerReviewSystem()
    
    # Anonymize
    print("\n1. Anonymizing analyses...")
    anonymized = system.anonymize_analyses(sample_analyses)
    for a in anonymized:
        print(f"   {a.anonymous_id} -> {a.original_role} (hidden)")
    
    # Conduct reviews
    print("\n2. Conducting peer reviews...")
    reviews = system.conduct_peer_reviews(use_llm=False)
    print(f"   Generated {len(reviews)} peer reviews")
    
    # Calculate rankings
    print("\n3. Calculating rankings...")
    rankings = system.calculate_rankings()
    for r in rankings:
        print(f"   #{r.rank}. {r.anonymous_id} ({r.original_role}): {r.average_score:.2f}/10")
    
    # Devil's Advocate
    print("\n4. Generating Devil's Advocate challenge...")
    challenge = system.generate_devils_advocate_challenge(use_llm=False)
    print(f"   Challenging: {challenge.consensus_view}")
    print(f"   Challenge confidence: {challenge.confidence_in_challenge}/10")
    
    # Full report
    print("\n5. Generating full report...")
    report = system.generate_full_report()
    print(report)

