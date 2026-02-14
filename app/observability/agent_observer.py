"""
Agent Observer - Core system for agentic self-observation and self-evaluation.
Tracks agent thought processes, quality metrics, and behavioral patterns.
"""
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
from enum import Enum
import hashlib

from app.models.agent_insight import AgentInsight
from app.models.trace import ExecutionTrace
from app.observability.tracer import Tracer
from app.repositories.agent_insight_repository import AgentInsightRepository
from app.utils.logger import get_logger

logger = get_logger(__name__)

class ObservationType(Enum):
    THOUGHT_PROCESS = "thought_process"
    DECISION_RATIONALE = "decision_rationale"
    SELF_EVALUATION = "self_evaluation"
    QUALITY_ASSESSMENT = "quality_assessment"
    CONFIDENCE_LEVEL = "confidence_level"
    ALTERNATIVES_CONSIDERED = "alternatives_considered"
    BEHAVIOR_PATTERN = "behavior_pattern"
    PERFORMANCE_METRIC = "performance_metric"


class AgentObserver:
    def __init__(self, tracer: Optional[Tracer] = None):
        self.tracer = tracer or Tracer()
        # Static thresholds – never modified after creation
        self._quality_thresholds = self._default_quality_thresholds()
        self._session_id = hashlib.md5(str(time.time()).encode()).hexdigest()[:8]
        self._insight_repository = AgentInsightRepository()
        
        logger.info(f"AgentObserver initialized with session: {self.session_id}")
    
    @property
    def session_id(self) -> str:
        return self._session_id
    
    def record_thought_process(
        self,
        agent_name: str,
        input_data: Any,
        thought_chain: List[str],
        final_thought: str,
        metadata: Optional[Dict] = None
    ) -> Dict[str, Any]:
        observation_id = f"thought_{hashlib.md5(str(time.time()).encode()).hexdigest()[:12]}"
        
        observation = {
            "observation_id": observation_id,
            "type": ObservationType.THOUGHT_PROCESS.value,
            "agent_name": agent_name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "thought_chain": thought_chain,
            "final_thought": final_thought,
            "input_fingerprint": self._create_fingerprint(input_data),
            "metadata": metadata or {},
            "session_id": self._session_id,
            "chain_length": len(thought_chain),
            "has_conclusion": bool(final_thought),
        }
        
        if self.tracer:
            self.tracer.record_agent_observation(
                "agent_thought_recorded",
                {
                    "observation_id": observation_id,
                    "agent_name": agent_name,
                    "chain_length": len(thought_chain),
                    "thought_chain": thought_chain,
                    "final_thought": final_thought,
                    "has_conclusion": bool(final_thought),
                }
            )
        
        logger.debug(
            f"Agent '{agent_name}' thought process recorded",
            extra={
                "observation_id": observation_id,
                "thought_chain_length": len(thought_chain),
                "has_conclusion": bool(final_thought),
            }
        )
        
        return observation
    
    def record_decision_rationale(
        self,
        decision_id: str,
        agent_name: str,
        options_considered: List[Dict[str, Any]],
        chosen_option: Dict[str, Any],
        rationale: str,
        confidence: float,
        tradeoffs: List[str],
        metadata: Optional[Dict] = None
    ) -> Dict[str, Any]:
        confidence = max(0.0, min(1.0, confidence))
        
        record = {
            "decision_id": decision_id,
            "type": ObservationType.DECISION_RATIONALE.value,
            "agent_name": agent_name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "options_count": len(options_considered),
            "options_considered": options_considered,
            "chosen_option": chosen_option,
            "rationale": rationale,
            "confidence_score": confidence,
            "confidence_level": self._classify_confidence(confidence),
            "tradeoffs": tradeoffs,
            "metadata": metadata or {},
            "session_id": self._session_id,
            "analysis_quality": self._evaluate_decision_quality(
                options_considered, rationale, confidence
            ),
        }
        
        if self.tracer:
            self.tracer.record_decision(
                "agent_decision_with_rationale",
                {
                    "decision_id": decision_id,
                    "confidence": confidence,
                    "options_considered": len(options_considered),
                    "analysis_quality": record["analysis_quality"],
                }
            )
        
        logger.info(
            f"Agent '{agent_name}' decision rationale recorded",
            extra={
                "decision_id": decision_id,
                "confidence": confidence,
                "options_considered": len(options_considered),
                "analysis_quality": record["analysis_quality"],
            }
        )
        
        return record
    
    def record_self_evaluation(
        self,
        agent_name: str,
        task_id: str,
        evaluation_criteria: Dict[str, float],
        self_scores: Dict[str, float],
        justification: str,
        improvements_suggested: List[str],
        metadata: Optional[Dict] = None
    ) -> Dict[str, Any]:
        weighted_score = 0.0
        total_weight = 0.0
        
        for criterion, weight in evaluation_criteria.items():
            if criterion in self_scores:
                score = self_scores[criterion]
                weighted_score += score * weight
                total_weight += weight
        
        overall_score = weighted_score / total_weight if total_weight > 0 else 0.0
        
        evaluation = {
            "evaluation_id": f"eval_{hashlib.md5(str(time.time()).encode()).hexdigest()[:12]}",
            "type": ObservationType.SELF_EVALUATION.value,
            "agent_name": agent_name,
            "task_id": task_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "evaluation_criteria": evaluation_criteria,
            "self_scores": self_scores,
            "overall_score": overall_score,
            "score_category": self._categorize_score(overall_score),
            "justification": justification,
            "improvements_suggested": improvements_suggested,
            "metadata": metadata or {},
            "session_id": self._session_id,
            "self_criticality": self._calculate_self_criticality(self_scores),
        }
        
        if self.tracer:
            self.tracer.record_agent_observation(
                "agent_self_evaluation",
                {
                    "evaluation_id": evaluation["evaluation_id"],
                    "overall_score": overall_score,
                    "score_category": evaluation["score_category"],
                    "self_criticality": evaluation["self_criticality"],
                }
            )
        
        logger.info(
            f"Agent '{agent_name}' self-evaluation recorded",
            extra={
                "task_id": task_id,
                "overall_score": overall_score,
                "score_category": evaluation["score_category"],
                "improvements_count": len(improvements_suggested),
            }
        )
        
        return evaluation

    def detect_behavior_pattern(
        self,
        agent_name: str,
        behavior_type: str,
        pattern_data: Dict[str, Any],
        frequency: Optional[int] = None,
        context: Optional[Dict] = None
    ) -> Dict[str, Any]:
        pattern_id = f"pattern_{hashlib.md5(str(time.time()).encode()).hexdigest()[:12]}"
        
        pattern_record = {
            "pattern_id": pattern_id,
            "type": ObservationType.BEHAVIOR_PATTERN.value,
            "agent_name": agent_name,
            "behavior_type": behavior_type,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "pattern_data": pattern_data,
            "frequency": frequency,
            "context": context or {},
            "session_id": self._session_id,
            "significance": self._assess_pattern_significance(pattern_data, behavior_type),
        }

        logger.info(
            f"Behavior pattern detected for agent '{agent_name}'",
            extra={
                "pattern_id": pattern_id,
                "behavior_type": behavior_type,
                "significance": pattern_record["significance"],
                "frequency": frequency,
            }
        )
        
        return pattern_record
    
    def get_agent_insights(self, agent_name: str, time_window_hours: int = 24) -> Dict[str, Any]:
        cutoff = datetime.now(timezone.utc) - timedelta(hours=time_window_hours)
        
        # Query database for persisted traces
        traces = ExecutionTrace.query.filter(
            ExecutionTrace.end_time >= cutoff
        ).all()
        
        # Also include current in-memory traces from tracer
        current_traces = []
        if self.tracer:
            current_traces = self.tracer.get_current_traces()
        
        observations = []
        quality_scores = []
        decision_count = 0
        decision_qualities = []
        
        # Process database traces
        for trace in traces:
            if trace.agent_context and trace.agent_context.get("agent_id") == agent_name:
                # Collect observations
                for obs in trace.agent_observations:
                    obs_copy = obs.copy()
                    obs_copy["timestamp"] = datetime.fromisoformat(obs_copy["timestamp"])
                    observations.append(obs_copy)
                
                if trace.quality_metrics and "composite_quality_score" in trace.quality_metrics:
                    score = trace.quality_metrics["composite_quality_score"]
                    if score is not None:
                        quality_scores.append(score)
                
                if trace.decisions:
                    for decision in trace.decisions:
                        decision_count += 1
                        if decision.get("quality") and decision["quality"].get("overall_score") is not None:
                            decision_qualities.append(decision["quality"]["overall_score"])
        
        # Process current in-memory traces
        for trace_dict in current_traces:
            if trace_dict.get("agent_context", {}).get("agent_id") == agent_name:
                # Collect observations
                for obs in trace_dict.get("agent_observations", []):
                    obs_copy = obs.copy()
                    obs_copy["timestamp"] = datetime.fromisoformat(obs_copy["timestamp"])
                    observations.append(obs_copy)
                
                if trace_dict.get("quality_metrics", {}).get("composite_quality_score"):
                    score = trace_dict["quality_metrics"]["composite_quality_score"]
                    if score is not None:
                        quality_scores.append(score)
                
                if trace_dict.get("decisions"):
                    for decision in trace_dict["decisions"]:
                        decision_count += 1
                        if decision.get("quality") and decision["quality"].get("overall_score") is not None:
                            decision_qualities.append(decision["quality"]["overall_score"])
        
        if not observations and decision_count == 0:
            return {"agent_name": agent_name, "no_data": True}
        
        avg_decision_quality = 0.0
        if decision_qualities:
            avg_decision_quality = sum(decision_qualities) / len(decision_qualities)
        
        # Other metrics from observations
        obs_decision_count = sum(1 for obs in observations 
                            if obs.get("type") == ObservationType.DECISION_RATIONALE.value)
        evaluation_count = sum(1 for obs in observations 
                            if obs.get("type") == ObservationType.SELF_EVALUATION.value)
        
        avg_quality = 0.0
        if quality_scores:
            avg_quality = sum(quality_scores) / len(quality_scores)
        
        common_behavior_patterns = self._analyze_behavior_patterns(agent_name)
        performance_trend = self._calculate_performance_trend(observations)
        confidence_distribution = self._calculate_confidence_distribution(observations)
        recommendations = self._generate_agent_recommendations(observations)
        
        insights = {
            "agent_name": agent_name,
            "observation_count": len(observations),
            "decision_count": decision_count,
            "average_decision_quality": avg_decision_quality,
            "self_evaluation_count": evaluation_count,
            "average_quality_score": avg_quality,
            "behavior_patterns": common_behavior_patterns,
            "performance_trend": performance_trend,
            "confidence_distribution": confidence_distribution,
            "recommendations": recommendations,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "time_window_hours": time_window_hours,
        }
        
        logger.info(
            f"Generated insights for agent '{agent_name}'",
            extra={
                "observation_count": len(observations),
                "decision_count": decision_count,
                "average_decision_quality": avg_decision_quality,
                "average_quality_score": avg_quality,
            }
        )
        
        return insights

    def _create_fingerprint(self, data: Any) -> str:
        data_str = str(data)
        if len(data_str) > 1000:
            data_str = data_str[:1000]
        return hashlib.md5(data_str.encode()).hexdigest()
    
    def _classify_confidence(self, confidence: float) -> str:
        if confidence >= 0.9:
            return "very_high"
        elif confidence >= 0.7:
            return "high"
        elif confidence >= 0.5:
            return "medium"
        elif confidence >= 0.3:
            return "low"
        else:
            return "very_low"
    
    def _evaluate_decision_quality(
        self,
        options_considered: List[Dict],
        rationale: str,
        confidence: float
    ) -> str:
        has_multiple_options = len(options_considered) > 1
        rationale_quality = "good" if len(rationale) > 50 else "poor"
        confidence_aligned = confidence > 0.5 if has_multiple_options else True
        
        if has_multiple_options and rationale_quality == "good" and confidence_aligned:
            return "high"
        elif has_multiple_options or rationale_quality == "good":
            return "medium"
        else:
            return "low"
    
    def _categorize_score(self, score: float) -> str:
        if score >= 0.9:
            return "excellent"
        elif score >= 0.8:
            return "good"
        elif score >= 0.6:
            return "acceptable"
        elif score >= 0.4:
            return "needs_improvement"
        else:
            return "poor"
    
    def _calculate_self_criticality(self, self_scores: Dict[str, float]) -> float:
        if not self_scores:
            return 0.5
        avg_score = sum(self_scores.values()) / len(self_scores)
        return 1.0 - avg_score

    def _assess_pattern_significance(
        self,
        pattern_data: Dict[str, Any],
        behavior_type: str
    ) -> str:
        pattern_size = len(str(pattern_data))
        
        if behavior_type in ["error_pattern", "failure_pattern"]:
            return "high"
        elif pattern_size > 1000:
            return "high"
        elif "repetition" in behavior_type.lower():
            return "medium"
        else:
            return "low"
    
    def _analyze_behavior_patterns(self, agent_name: str) -> List[Dict[str, Any]]:
        patterns = ExecutionTrace.query.filter(
            ExecutionTrace.agent_context['agent_id'].as_string() == agent_name,
            ExecutionTrace.agent_observations != None
        ).all()
        
        pattern_counts = {}
        for trace in patterns:
            for obs in trace.agent_observations:
                if obs.get("type") == ObservationType.BEHAVIOR_PATTERN.value:
                    pattern_type = obs.get("behavior_type", "unknown")
                    pattern_counts[pattern_type] = pattern_counts.get(pattern_type, 0) + 1
        
        return [
            {
                "type": pattern_type,
                "occurrence_count": count,
                "significance": "medium" if count >= 5 else "low",
            }
            for pattern_type, count in pattern_counts.items()
            if count >= 2
        ]

    def _calculate_performance_trend(self, observations: List[Dict]) -> str:
        if len(observations) < 2:
            return "insufficient_data"
        
        quality_scores = []
        for obs in observations:
            if "quality_score" in obs and obs["quality_score"] is not None:
                quality_scores.append(obs["quality_score"])
        
        if len(quality_scores) < 2:
            return "stable"
        
        recent_scores = quality_scores[-5:] if len(quality_scores) >= 5 else quality_scores
        older_scores = quality_scores[:5] if len(quality_scores) >= 10 else quality_scores[:len(quality_scores)//2]
        
        if not older_scores:
            return "stable"
        
        avg_recent = sum(recent_scores) / len(recent_scores)
        avg_older = sum(older_scores) / len(older_scores)
        
        if avg_recent > avg_older + 0.1:
            return "improving"
        elif avg_recent < avg_older - 0.1:
            return "declining"
        else:
            return "stable"
    
    def _calculate_confidence_distribution(self, observations: List[Dict]) -> Dict[str, float]:
        confidence_values = []
        
        for obs in observations:
            if "confidence_score" in obs and obs["confidence_score"] is not None:
                confidence_values.append(obs["confidence_score"])
        
        if not confidence_values:
            return {"average": 0.0, "min": 0.0, "max": 0.0}
        
        return {
            "average": sum(confidence_values) / len(confidence_values),
            "min": min(confidence_values),
            "max": max(confidence_values),
            "count": len(confidence_values),
        }
    
    def _generate_agent_recommendations(self, observations: List[Dict]) -> List[str]:
        recommendations = []
        
        quality_scores = [obs.get("quality_score") for obs in observations if "quality_score" in obs and obs["quality_score"] is not None]
        if quality_scores:
            avg_quality = sum(quality_scores) / len(quality_scores)
            if avg_quality < 0.7:
                recommendations.append("Focus on improving output quality")
        
        self_evaluations = [obs for obs in observations if obs.get("type") == ObservationType.SELF_EVALUATION.value]
        if self_evaluations:
            scores = [eval.get("overall_score") for eval in self_evaluations if eval.get("overall_score") is not None]
            if scores:
                avg_self_score = sum(scores) / len(scores)
                if avg_self_score > 0.9:
                    recommendations.append("Consider being more self-critical in evaluations")
                elif avg_self_score < 0.4:
                    recommendations.append("Consider being less harsh in self-evaluations")
        
        decisions = [obs for obs in observations if obs.get("type") == ObservationType.DECISION_RATIONALE.value]
        if decisions:
            single_option_decisions = sum(1 for d in decisions if d.get("options_count", 0) <= 1)
            if single_option_decisions / len(decisions) > 0.8:
                recommendations.append("Consider exploring more alternatives in decisions")
        
        return recommendations
    
    def _default_quality_thresholds(self) -> Dict[str, float]:
        return {
            "minimum_quality": 0.7,
            "good_quality": 0.8,
            "excellent_quality": 0.9,
        }

    def apply_learned_insights(self, agent_name: str) -> List[Dict]:
        insights = self.get_agent_insights(agent_name)
        improvements = []
        
        for insight in insights.get("recommendations", []):
            improvement = {
                "insight": insight,
                "applied_at": datetime.now(timezone.utc),
                "expected_impact": "quality_improvement"
            }
            improvements.append(improvement)
        
        return improvements

    def generate_insights_from_observations(
        self,
        agent_name: str,
        time_window_hours: int = 24
    ) -> List[Dict[str, Any]]:
        try:
            insights_summary = self.get_agent_insights(
                agent_name=agent_name,
                time_window_hours=time_window_hours
            )
            
            if insights_summary.get("no_data"):
                return []
            
            generated_insights = []
            
            # Insight 1: Quality trend
            performance_trend = insights_summary.get("performance_trend")
            avg_quality = insights_summary.get("average_quality_score", 0)
            
            if performance_trend == "declining" and avg_quality is not None and avg_quality < 0.7 and avg_quality > 0:
                insight = self._create_quality_insight(
                    agent_name=agent_name,
                    current_quality=avg_quality,
                    trend=performance_trend,
                    context=insights_summary
                )
                generated_insights.append(insight)
            
            # Insight 2: Decision patterns – use average_decision_quality instead of confidence_distribution
            decision_quality = insights_summary.get("average_decision_quality", 0)
            decision_count = insights_summary.get("decision_count", 0)
            
            if decision_quality is not None and decision_quality < 0.9 and decision_quality > 0 and decision_count > 0:
                insight = self._create_decision_insight(
                    agent_name=agent_name,
                    decision_quality=decision_quality,
                    context=insights_summary
                )
                generated_insights.append(insight)
            
            # Insight 3: Behavior patterns
            behavior_patterns = insights_summary.get("behavior_patterns", [])
            if behavior_patterns:
                for pattern in behavior_patterns:
                    if pattern.get("significance") == "high":
                        insight = self._create_behavior_insight(
                            agent_name=agent_name,
                            pattern=pattern,
                            context=insights_summary
                        )
                        generated_insights.append(insight)
            
            # Insight 4: Self-evaluation patterns
            recommendations = insights_summary.get("recommendations", [])
            if recommendations:
                insight = self._create_recommendation_insight(
                    agent_name=agent_name,
                    recommendations=recommendations,
                    context=insights_summary
                )
                generated_insights.append(insight)
            
            for insight_data in generated_insights:
                self._save_insight(insight_data)
            
            return generated_insights
            
        except Exception as e:
            logger.error(f"Failed to generate insights: {str(e)}")
            return []

    def _create_quality_insight(
        self,
        agent_name: str,
        current_quality: float,
        trend: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        insight_id = f"quality_{hashlib.md5(str(time.time()).encode()).hexdigest()[:12]}"
        gap = self._quality_thresholds["minimum_quality"] - (current_quality if current_quality is not None else 0)
        confidence = min(0.95, max(0.5, 0.5 + gap))
        return {
            "insight_id": insight_id,
            "agent_name": agent_name,
            "insight_type": "quality_improvement",
            "insight_data": {
                "current_quality": current_quality,
                "trend": trend,
                "threshold": self._quality_thresholds["minimum_quality"],
                "gap": gap,
                "context_summary": {
                    "observation_count": context.get("observation_count", 0),
                    "decision_count": context.get("decision_count", 0),
                },
            },
            "confidence_score": confidence,
            "recommended_action": "Implement additional quality checks and review error patterns",
            "source": "quality_analysis",
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

    def _create_decision_insight(
        self,
        agent_name: str,
        decision_quality: float,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        insight_id = f"decision_{hashlib.md5(str(time.time()).encode()).hexdigest()[:12]}"
        target_quality = 0.9
        gap = target_quality - (decision_quality if decision_quality is not None else 0)
        confidence = min(0.95, max(0.5, 0.5 + gap))
        return {
            "insight_id": insight_id,
            "agent_name": agent_name,
            "insight_type": "decision_improvement",
            "insight_data": {
                "current_decision_quality": decision_quality,
                "target_quality": target_quality,
                "gap": gap,
                "confidence_distribution": context.get("confidence_distribution", {}),
                "context_summary": {
                    "decision_count": context.get("decision_count", 0),
                },
            },
            "confidence_score": confidence,
            "recommended_action": "Review decision rationale and consider more alternatives",
            "source": "decision_analysis",
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

    def _create_behavior_insight(
        self,
        agent_name: str,
        pattern: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        insight_id = f"behavior_{hashlib.md5(str(time.time()).encode()).hexdigest()[:12]}"
        return {
            "insight_id": insight_id,
            "agent_name": agent_name,
            "insight_type": "behavior_insight",
            "insight_data": {
                "pattern_type": pattern.get("type"),
                "occurrence_count": pattern.get("occurrence_count"),
                "significance": pattern.get("significance"),
            },
            "confidence_score": 0.8,
            "recommended_action": "Monitor this behavior pattern",
            "source": "behavior_analysis",
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

    def _create_recommendation_insight(
        self,
        agent_name: str,
        recommendations: List[str],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        insight_id = f"recommendation_{hashlib.md5(str(time.time()).encode()).hexdigest()[:12]}"
        confidence = max(0.5, 1.0 - (len(recommendations) * 0.1))
        return {
            "insight_id": insight_id,
            "agent_name": agent_name,
            "insight_type": "optimization_suggestion",
            "insight_data": {
                "recommendations": recommendations,
                "recommendation_count": len(recommendations),
                "priority_recommendations": recommendations[:2] if len(recommendations) >= 2 else recommendations,
                "context_summary": {
                    "performance_trend": context.get("performance_trend"),
                    "average_quality": context.get("average_quality_score", 0),
                },
            },
            "confidence_score": confidence,
            "recommended_action": "Implement priority recommendations",
            "source": "recommendation_analysis",
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

    def _save_insight(self, insight_data: Dict[str, Any]) -> None:
        try:
            insight = AgentInsight(
                agent_name=insight_data["agent_name"],
                insight_type=insight_data["insight_type"],
                insight_data=insight_data["insight_data"],
                confidence_score=insight_data.get("confidence_score"),
                impact_prediction="high",
            )
            insight.impact_prediction = insight.calculate_impact_prediction()
            self._insight_repository.save(insight)
            logger.debug(
                f"Insight saved for agent '{insight.agent_name}'",
                extra={
                    "insight_id": insight.id,
                    "insight_type": insight.insight_type,
                    "impact_prediction": insight.impact_prediction,
                }
            )
        except Exception as e:
            logger.error(
                f"Failed to save insight: {str(e)}",
                extra={
                    "insight_data": insight_data.get("insight_id", "unknown"),
                    "error_type": type(e).__name__,
                }
            )
