"""
Agent Observer - Core system for agentic self-observation and self-evaluation.
Tracks agent thought processes, quality metrics, and behavioral patterns.
"""
import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from enum import Enum
import hashlib

from app.observability.tracer import Tracer
from app.utils.logger import get_logger

logger = get_logger(__name__)

class ObservationType(Enum):
    """Types of agent observations."""
    THOUGHT_PROCESS = "thought_process"
    DECISION_RATIONALE = "decision_rationale"
    SELF_EVALUATION = "self_evaluation"
    QUALITY_ASSESSMENT = "quality_assessment"
    CONFIDENCE_LEVEL = "confidence_level"
    ALTERNATIVES_CONSIDERED = "alternatives_considered"
    BEHAVIOR_PATTERN = "behavior_pattern"
    PERFORMANCE_METRIC = "performance_metric"


class AgentObserver:
    """
    Main observer that tracks agent behavior, thoughts, and self-evaluations.
    Provides structured self-observation capabilities.
    """
    
    def __init__(self, tracer: Optional[Tracer] = None):
        self.tracer = tracer or Tracer()
        self._behavior_patterns: Dict[str, List[Dict]] = {}
        self._quality_thresholds = self._default_quality_thresholds()
        self._observation_buffer: List[Dict] = []
        self._buffer_size = 100
        self._session_id = hashlib.md5(str(time.time()).encode()).hexdigest()[:8]
        
        logger.info(f"AgentObserver initialized with session: {self.session_id}")
    
    @property
    def session_id(self) -> str:
        """Get current observation session ID."""
        return self._session_id
    
    def record_thought_process(
        self,
        agent_name: str,
        input_data: Any,
        thought_chain: List[str],
        final_thought: str,
        metadata: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Record the complete thought process of an agent.
        
        Args:
            agent_name: Name/identifier of the agent
            input_data: What the agent was considering
            thought_chain: Sequential thoughts leading to decision
            final_thought: The conclusive thought
            metadata: Additional context
            
        Returns:
            Observation record
        """
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
        
        # Store in buffer
        self._store_observation(observation)
        
        # Record in trace if available
        if self.tracer:
            self.tracer.record_event(
                "agent_thought_recorded",
                {
                    "observation_id": observation_id,
                    "agent_name": agent_name,
                    "chain_length": len(thought_chain),
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
        confidence: float,  # 0.0 to 1.0
        tradeoffs: List[str],
        metadata: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Record detailed rationale for a decision, including alternatives.
        
        Args:
            decision_id: Unique identifier for the decision
            agent_name: Name of the decision-making agent
            options_considered: List of options that were evaluated
            chosen_option: The selected option
            rationale: Why this option was chosen
            confidence: Confidence score (0.0-1.0)
            tradeoffs: List of tradeoffs considered
            metadata: Additional context
            
        Returns:
            Decision rationale record
        """
        # Validate confidence score
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
        
        # Store in buffer
        self._store_observation(record)
        
        # Record in trace
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
        evaluation_criteria: Dict[str, float],  # criterion -> weight (0-1)
        self_scores: Dict[str, float],  # criterion -> self-assigned score (0-1)
        justification: str,
        improvements_suggested: List[str],
        metadata: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Record agent's self-evaluation of its performance.
        
        Args:
            agent_name: Name of the evaluating agent
            task_id: ID of the task being evaluated
            evaluation_criteria: Criteria and their weights
            self_scores: Agent's self-assessment scores
            justification: Justification for scores
            improvements_suggested: Suggested improvements
            metadata: Additional context
            
        Returns:
            Self-evaluation record
        """
        # Calculate weighted score
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
        
        # Store in buffer
        self._store_observation(evaluation)
        
        # Record in trace
        if self.tracer:
            self.tracer.record_event(
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
    
    def assess_quality(
        self,
        output: Any,
        expected_criteria: Dict[str, Any],
        agent_name: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Assess the quality of an agent's output against expected criteria.
        
        Args:
            output: The output to assess
            expected_criteria: Dictionary of criteria to check
            agent_name: Name of the agent producing the output
            context: Context of the assessment
            
        Returns:
            Quality assessment record
        """
        assessment_id = f"qa_{hashlib.md5(str(time.time()).encode()).hexdigest()[:12]}"
        
        # Perform quality checks
        quality_checks = self._perform_quality_checks(output, expected_criteria)
        
        # Calculate overall quality score
        passed_checks = sum(1 for check in quality_checks.values() if check["passed"])
        total_checks = len(quality_checks)
        quality_score = passed_checks / total_checks if total_checks > 0 else 0.0
        
        assessment = {
            "assessment_id": assessment_id,
            "type": ObservationType.QUALITY_ASSESSMENT.value,
            "agent_name": agent_name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "output_fingerprint": self._create_fingerprint(output),
            "quality_checks": quality_checks,
            "quality_score": quality_score,
            "passed_checks": passed_checks,
            "total_checks": total_checks,
            "meets_threshold": quality_score >= self._quality_thresholds["minimum_quality"],
            "threshold": self._quality_thresholds["minimum_quality"],
            "context": context,
            "session_id": self._session_id,
            "recommendations": self._generate_quality_recommendations(quality_checks),
        }
        
        # Store in buffer
        self._store_observation(assessment)
        
        # Record in trace
        if self.tracer:
            self.tracer.record_event(
                "quality_assessment",
                {
                    "assessment_id": assessment_id,
                    "quality_score": quality_score,
                    "passed_checks": passed_checks,
                    "total_checks": total_checks,
                    "meets_threshold": assessment["meets_threshold"],
                }
            )
        
        # Log quality result
        log_level = logging.INFO if assessment["meets_threshold"] else logging.WARNING
        logger.log(
            log_level,
            f"Quality assessment for agent '{agent_name}'",
            extra={
                "assessment_id": assessment_id,
                "quality_score": quality_score,
                "passed_checks": passed_checks,
                "total_checks": total_checks,
                "meets_threshold": assessment["meets_threshold"],
            }
        )
        
        return assessment
    
    def detect_behavior_pattern(
        self,
        agent_name: str,
        behavior_type: str,
        pattern_data: Dict[str, Any],
        frequency: Optional[int] = None,
        context: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Detect and record patterns in agent behavior.
        
        Args:
            agent_name: Name of the agent
            behavior_type: Type of behavior pattern
            pattern_data: Details of the pattern
            frequency: How often this pattern occurs
            context: Context of the pattern
            
        Returns:
            Behavior pattern record
        """
        pattern_id = f"pattern_{hashlib.md5(str(time.time()).encode()).hexdigest()[:12]}"
        
        # Store pattern in memory for tracking
        if agent_name not in self._behavior_patterns:
            self._behavior_patterns[agent_name] = []
        
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
        
        self._behavior_patterns[agent_name].append(pattern_record)
        
        # Keep only recent patterns
        if len(self._behavior_patterns[agent_name]) > 100:
            self._behavior_patterns[agent_name] = self._behavior_patterns[agent_name][-100:]
        
        # Store in observation buffer
        self._store_observation(pattern_record)
        
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
    
    def get_agent_insights(
        self,
        agent_name: str,
        time_window_hours: int = 24
    ) -> Dict[str, Any]:
        """
        Generate insights about an agent's behavior and performance.
        
        Args:
            agent_name: Name of the agent
            time_window_hours: Time window for analysis
            
        Returns:
            Agent insights dictionary
        """
        # Filter observations for this agent within time window
        relevant_observations = [
            obs for obs in self._observation_buffer
            if obs.get("agent_name") == agent_name
        ]
        
        if not relevant_observations:
            return {"agent_name": agent_name, "no_data": True}
        
        # Calculate various metrics
        decision_count = sum(1 for obs in relevant_observations 
                           if obs.get("type") == ObservationType.DECISION_RATIONALE.value)
        evaluation_count = sum(1 for obs in relevant_observations 
                             if obs.get("type") == ObservationType.SELF_EVALUATION.value)
        quality_assessments = [obs for obs in relevant_observations 
                             if obs.get("type") == ObservationType.QUALITY_ASSESSMENT.value]
        
        # Calculate average quality score
        avg_quality = 0.0
        if quality_assessments:
            avg_quality = sum(obs.get("quality_score", 0) for obs in quality_assessments) / len(quality_assessments)
        
        # Detect common patterns
        common_behavior_patterns = self._analyze_behavior_patterns(agent_name)
        
        insights = {
            "agent_name": agent_name,
            "observation_count": len(relevant_observations),
            "decision_count": decision_count,
            "self_evaluation_count": evaluation_count,
            "average_quality_score": avg_quality,
            "behavior_patterns": common_behavior_patterns,
            "performance_trend": self._calculate_performance_trend(relevant_observations),
            "confidence_distribution": self._calculate_confidence_distribution(relevant_observations),
            "recommendations": self._generate_agent_recommendations(relevant_observations),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "time_window_hours": time_window_hours,
        }
        
        logger.info(
            f"Generated insights for agent '{agent_name}'",
            extra={
                "observation_count": len(relevant_observations),
                "average_quality_score": avg_quality,
                "decision_count": decision_count,
            }
        )
        
        return insights
    
    def _store_observation(self, observation: Dict[str, Any]) -> None:
        """Store observation in buffer."""
        self._observation_buffer.append(observation)
        
        # Maintain buffer size
        if len(self._observation_buffer) > self._buffer_size:
            self._observation_buffer = self._observation_buffer[-self._buffer_size:]
    
    def _create_fingerprint(self, data: Any) -> str:
        """Create a fingerprint for data (privacy-safe)."""
        data_str = str(data)
        if len(data_str) > 1000:
            data_str = data_str[:1000]
        return hashlib.md5(data_str.encode()).hexdigest()
    
    def _classify_confidence(self, confidence: float) -> str:
        """Classify confidence level."""
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
        """Evaluate the quality of a decision analysis."""
        # Check if multiple options were considered
        has_multiple_options = len(options_considered) > 1
        
        # Check rationale quality
        rationale_quality = "good" if len(rationale) > 50 else "poor"
        
        # Check confidence alignment
        confidence_aligned = confidence > 0.5 if has_multiple_options else True
        
        if has_multiple_options and rationale_quality == "good" and confidence_aligned:
            return "high"
        elif has_multiple_options or rationale_quality == "good":
            return "medium"
        else:
            return "low"
    
    def _categorize_score(self, score: float) -> str:
        """Categorize a score."""
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
        """
        Calculate how self-critical the agent is.
        Lower scores indicate higher self-criticality.
        """
        if not self_scores:
            return 0.5
        avg_score = sum(self_scores.values()) / len(self_scores)
        # Invert so lower scores = more critical
        return 1.0 - avg_score
    
    def _perform_quality_checks(
        self,
        output: Any,
        expected_criteria: Dict[str, Any]
    ) -> Dict[str, Dict[str, Any]]:
        """Perform various quality checks on output."""
        checks = {}
        
        # Check 1: Output is not empty
        checks["not_empty"] = {
            "passed": bool(output),
            "actual": "empty" if not output else "has_content",
            "expected": "has_content",
        }
        
        # Check 2: Output is appropriate type
        expected_type = expected_criteria.get("type", "any")
        if expected_type != "any":
            checks["correct_type"] = {
                "passed": isinstance(output, self._get_type_from_string(expected_type)),
                "actual": type(output).__name__,
                "expected": expected_type,
            }
        
        # Check 3: Output length (if applicable)
        if isinstance(output, (str, list, dict)):
            output_length = len(output)
            min_length = expected_criteria.get("min_length", 0)
            max_length = expected_criteria.get("max_length", float('inf'))
            
            checks["length"] = {
                "passed": min_length <= output_length <= max_length,
                "actual": output_length,
                "expected": f"{min_length}-{max_length}",
            }
        
        # Check 4: Contains required keys/fields
        required_keys = expected_criteria.get("required_keys", [])
        if required_keys and isinstance(output, dict):
            missing_keys = [key for key in required_keys if key not in output]
            checks["required_keys"] = {
                "passed": len(missing_keys) == 0,
                "actual": f"missing: {missing_keys}" if missing_keys else "all_present",
                "expected": f"all of {required_keys}",
            }
        
        # Check 5: Structured format (for JSON/dict)
        if isinstance(output, (dict, list)):
            checks["structured"] = {
                "passed": True,  # Already structured
                "actual": "structured",
                "expected": "structured",
            }
        
        return checks
    
    def _get_type_from_string(self, type_str: str) -> type:
        """Convert type string to actual type."""
        type_map = {
            "str": str,
            "string": str,
            "int": int,
            "integer": int,
            "float": float,
            "bool": bool,
            "boolean": bool,
            "list": list,
            "dict": dict,
            "tuple": tuple,
        }
        return type_map.get(type_str.lower(), object)
    
    def _generate_quality_recommendations(self, quality_checks: Dict) -> List[str]:
        """Generate recommendations based on quality check failures."""
        recommendations = []
        
        for check_name, check_result in quality_checks.items():
            if not check_result["passed"]:
                if check_name == "not_empty":
                    recommendations.append("Output should not be empty")
                elif check_name == "correct_type":
                    recommendations.append(f"Output should be of type {check_result['expected']}")
                elif check_name == "length":
                    recommendations.append(f"Output length should be between {check_result['expected']}")
                elif check_name == "required_keys":
                    recommendations.append("Output is missing required fields")
        
        return recommendations
    
    def _assess_pattern_significance(
        self,
        pattern_data: Dict[str, Any],
        behavior_type: str
    ) -> str:
        """Assess the significance of a behavior pattern."""
        # Simple heuristic-based significance assessment
        pattern_size = len(str(pattern_data))
        
        if behavior_type in ["error_pattern", "failure_pattern"]:
            return "high"
        elif pattern_size > 1000:  # Large/complex patterns
            return "high"
        elif "repetition" in behavior_type.lower():
            return "medium"
        else:
            return "low"
    
    def _analyze_behavior_patterns(self, agent_name: str) -> List[Dict[str, Any]]:
        """Analyze behavior patterns for an agent."""
        if agent_name not in self._behavior_patterns:
            return []
        
        patterns = self._behavior_patterns[agent_name]
        
        # Group patterns by type
        pattern_counts = {}
        for pattern in patterns:
            pattern_type = pattern["behavior_type"]
            pattern_counts[pattern_type] = pattern_counts.get(pattern_type, 0) + 1
        
        # Find most common patterns
        common_patterns = []
        for pattern_type, count in pattern_counts.items():
            if count >= 2:  # Only include patterns that occurred at least twice
                common_patterns.append({
                    "type": pattern_type,
                    "occurrence_count": count,
                    "significance": "medium" if count >= 5 else "low",
                })
        
        return common_patterns
    
    def _calculate_performance_trend(self, observations: List[Dict]) -> str:
        """Calculate performance trend from observations."""
        if len(observations) < 2:
            return "insufficient_data"
        
        # Get quality scores over time
        quality_scores = []
        for obs in observations:
            if "quality_score" in obs:
                quality_scores.append(obs["quality_score"])
        
        if len(quality_scores) < 2:
            return "stable"
        
        # Simple trend calculation
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
        """Calculate distribution of confidence levels."""
        confidence_values = []
        
        for obs in observations:
            if "confidence_score" in obs:
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
        """Generate recommendations for agent improvement."""
        recommendations = []
        
        # Analyze quality scores
        quality_scores = [obs.get("quality_score", 0) for obs in observations if "quality_score" in obs]
        if quality_scores:
            avg_quality = sum(quality_scores) / len(quality_scores)
            if avg_quality < 0.7:
                recommendations.append("Focus on improving output quality")
        
        # Analyze self-criticality
        self_evaluations = [obs for obs in observations if obs.get("type") == ObservationType.SELF_EVALUATION.value]
        if self_evaluations:
            avg_self_score = sum(eval.get("overall_score", 0.5) for eval in self_evaluations) / len(self_evaluations)
            if avg_self_score > 0.9:
                recommendations.append("Consider being more self-critical in evaluations")
            elif avg_self_score < 0.4:
                recommendations.append("Consider being less harsh in self-evaluations")
        
        # Check for decision diversity
        decisions = [obs for obs in observations if obs.get("type") == ObservationType.DECISION_RATIONALE.value]
        if decisions:
            single_option_decisions = sum(1 for d in decisions if d.get("options_count", 0) <= 1)
            if single_option_decisions / len(decisions) > 0.8:
                recommendations.append("Consider exploring more alternatives in decisions")
        
        return recommendations
    
    def _default_quality_thresholds(self) -> Dict[str, float]:
        """Default quality thresholds."""
        return {
            "minimum_quality": 0.7,
            "good_quality": 0.8,
            "excellent_quality": 0.9,
        }
