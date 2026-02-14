"""
Metrics Calculator - Calculates agentic metrics from traces and observations.
"""
import statistics
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
from collections import defaultdict

from app.models.trace import ExecutionTrace
from app.repositories.trace_repository import TraceRepository
from app.utils.logger import get_logger

logger = get_logger(__name__)


class MetricsCalculator:
    def __init__(self, trace_repository: Optional[TraceRepository] = None):
        self.trace_repo = trace_repository or TraceRepository()
    
    def calculate_agent_performance(
        self,
        agent_name: Optional[str] = None,
        time_window_hours: int = 24
    ) -> Dict[str, Any]:
        traces = self._get_traces_for_period(time_window_hours)
        
        if not traces:
            return {
                "agent_name": agent_name,
                "total_traces": 0,
                "no_data": True,
                "calculated_at": datetime.now(timezone.utc).isoformat(),
            }
        
        if agent_name:
            traces = [
                t for t in traces 
                if t.agent_context and t.agent_context.get("agent_id") == agent_name
            ]
        
        total_traces = len(traces)
        successful_traces = [t for t in traces if t.success]
        failed_traces = [t for t in traces if not t.success]
        
        success_rate = len(successful_traces) / total_traces if total_traces > 0 else 0.0
        
        quality_scores = []
        for trace in traces:
            if trace.quality_metrics and "composite_quality_score" in trace.quality_metrics:
                score = trace.quality_metrics["composite_quality_score"]
                if score is not None:
                    quality_scores.append(score)
        
        avg_quality = statistics.mean(quality_scores) if quality_scores else 0.0
        
        durations = [t.duration_ms for t in traces if t.duration_ms is not None]
        avg_duration = statistics.mean(durations) if durations else 0.0
        
        total_decisions = 0
        decision_qualities = []
        for trace in traces:
            if trace.decisions:
                total_decisions += len(trace.decisions)
                for decision in trace.decisions:
                    if isinstance(decision, dict) and "quality" in decision:
                        quality = decision["quality"]
                        if isinstance(quality, dict) and quality.get("overall_score") is not None:
                            decision_qualities.append(quality["overall_score"])

        avg_decisions_per_trace = total_decisions / len(traces) if traces else 0.0
        avg_decision_quality = statistics.mean(decision_qualities) if decision_qualities else 0.0
        
        observation_counts = []
        for trace in traces:
            if trace.agent_observations:
                observation_counts.append(len(trace.agent_observations))
        
        avg_observations_per_trace = statistics.mean(observation_counts) if observation_counts else 0.0
        
        error_types = defaultdict(int)
        for trace in failed_traces:
            if trace.error and isinstance(trace.error, dict):
                error_type = trace.error.get("type", "unknown")
                error_types[error_type] += 1
        
        current_time = datetime.now(timezone.utc)
        recent_window_start = current_time - timedelta(hours=3)
        previous_window_start = current_time - timedelta(hours=6)
        
        recent_traces = [t for t in traces if t.end_time and t.end_time >= recent_window_start]
        previous_traces = [
            t for t in traces 
            if t.end_time and previous_window_start <= t.end_time < recent_window_start
        ]
        
        trend = "stable"
        if recent_traces and previous_traces:
            recent_success_rate = len([t for t in recent_traces if t.success]) / len(recent_traces)
            previous_success_rate = len([t for t in previous_traces if t.success]) / len(previous_traces)
            
            if recent_success_rate > previous_success_rate + 0.1:
                trend = "improving"
            elif recent_success_rate < previous_success_rate - 0.1:
                trend = "declining"
        
        result = {
            "agent_name": agent_name or "all_agents",
            "total_traces": total_traces,
            "time_window_hours": time_window_hours,
            "success_rate": success_rate,
            "average_quality_score": avg_quality,
            "average_duration_ms": avg_duration,
            "average_decisions_per_trace": avg_decisions_per_trace,
            "average_decision_quality": avg_decision_quality,
            "average_observations_per_trace": avg_observations_per_trace,
            "failed_traces": len(failed_traces),
            "error_types": dict(error_types),
            "performance_trend": trend,
            "calculated_at": datetime.now(timezone.utc).isoformat(),
            "quality_distribution": self._calculate_quality_distribution(traces),
            "recommendations": self._generate_performance_recommendations(
                success_rate, avg_quality, avg_decision_quality
            ),
        }
        
        logger.info(
            f"Agent performance metrics calculated",
            extra={
                "agent_name": agent_name or "all_agents",
                "success_rate": success_rate,
                "average_quality_score": avg_quality,
                "total_traces": total_traces,
            }
        )
        
        return result
    
    def calculate_quality_metrics(
        self,
        time_window_hours: int = 24,
        group_by: str = "operation"
    ) -> Dict[str, Any]:
        traces: List[ExecutionTrace] = self._get_traces_for_period(time_window_hours)
        
        if not traces:
            return {
                "group_by": group_by,
                "time_window_hours": time_window_hours,
                "no_data": True,
                "calculated_at": datetime.now(timezone.utc).isoformat(),
            }
        
        grouped_data = defaultdict(list)
        
        for trace in traces:
            if group_by == "operation":
                key = trace.operation
            elif group_by == "agent":
                key = trace.agent_context.get("agent_id", "unknown") if trace.agent_context else "unknown"
            elif group_by == "hour":
                if trace.end_time:
                    key = trace.end_time.strftime("%Y-%m-%d %H:00")
                else:
                    key = "unknown"
            else:
                key = "unknown"
            
            grouped_data[key].append(trace)
        
        result_groups = {}
        
        for group_key, group_traces in grouped_data.items():
            quality_scores = []
            for trace in group_traces:
                if trace.quality_metrics and "composite_quality_score" in trace.quality_metrics:
                    score = trace.quality_metrics["composite_quality_score"]
                    if score is not None:
                        quality_scores.append(score)
            
            if quality_scores:
                avg_quality = statistics.mean(quality_scores)
                min_quality = min(quality_scores)
                max_quality = max(quality_scores)
                median_quality = statistics.median(quality_scores)
                
                quality_distribution = {
                    "excellent": len([s for s in quality_scores if s >= 0.9]),
                    "good": len([s for s in quality_scores if 0.8 <= s < 0.9]),
                    "acceptable": len([s for s in quality_scores if 0.6 <= s < 0.8]),
                    "needs_improvement": len([s for s in quality_scores if 0.4 <= s < 0.6]),
                    "poor": len([s for s in quality_scores if s < 0.4]),
                }
            else:
                avg_quality = min_quality = max_quality = median_quality = 0.0
                quality_distribution = {}
            
            successful = len([t for t in group_traces if t.success])
            success_rate = successful / len(group_traces) if group_traces else 0.0
            
            result_groups[group_key] = {
                "trace_count": len(group_traces),
                "average_quality": avg_quality,
                "min_quality": min_quality,
                "max_quality": max_quality,
                "median_quality": median_quality,
                "success_rate": success_rate,
                "quality_distribution": quality_distribution,
            }
        
        sorted_groups = dict(sorted(
            result_groups.items(),
            key=lambda x: x[1]["trace_count"],
            reverse=True
        ))
        
        result = {
            "group_by": group_by,
            "time_window_hours": time_window_hours,
            "total_traces": len(traces),
            "groups": sorted_groups,
            "overall_metrics": self._calculate_overall_quality_metrics(traces),
            "calculated_at": datetime.now(timezone.utc).isoformat(),
        }
        
        return result
    
    def calculate_decision_analytics(
        self,
        time_window_hours: int = 24
    ) -> Dict[str, Any]:
        traces = self._get_traces_for_period(time_window_hours)
        
        if not traces:
            return {
                "time_window_hours": time_window_hours,
                "no_data": True,
                "calculated_at": datetime.now(timezone.utc).isoformat(),
            }
        
        all_decisions = []
        decision_types = defaultdict(list)
        
        for trace in traces:
            if trace.decisions:
                for decision in trace.decisions:
                    if isinstance(decision, dict):
                        decision["trace_operation"] = trace.operation
                        decision["trace_success"] = trace.success
                        decision["agent"] = trace.agent_context.get("agent_id") if trace.agent_context else None
                        
                        all_decisions.append(decision)
                        
                        decision_type = decision.get("type", "unknown")
                        decision_types[decision_type].append(decision)
        
        total_decisions = len(all_decisions)
        
        quality_scores = []
        for decision in all_decisions:
            if "quality" in decision and isinstance(decision["quality"], dict):
                score = decision["quality"].get("overall_score")
                if score is not None:
                    quality_scores.append(score)
        
        avg_decision_quality = statistics.mean(quality_scores) if quality_scores else 0.0
        
        type_analysis = {}
        for decision_type, type_decisions in decision_types.items():
            type_quality_scores = []
            for decision in type_decisions:
                if "quality" in decision and isinstance(decision["quality"], dict):
                    score = decision["quality"].get("overall_score")
                    if score is not None:
                        type_quality_scores.append(score)
            
            avg_type_quality = statistics.mean(type_quality_scores) if type_quality_scores else 0.0
            
            type_analysis[decision_type] = {
                "count": len(type_decisions),
                "average_quality": avg_type_quality,
                "percentage": len(type_decisions) / total_decisions if total_decisions > 0 else 0.0,
            }
        
        context_lengths = []
        for decision in all_decisions:
            context = decision.get("context", {})
            context_lengths.append(len(str(context)))
        
        avg_context_length = statistics.mean(context_lengths) if context_lengths else 0.0
        
        successful_decisions = []
        failed_decisions = []
        
        for decision in all_decisions:
            if decision.get("trace_success", False):
                successful_decisions.append(decision)
            else:
                failed_decisions.append(decision)
        
        successful_qualities = []
        for decision in successful_decisions:
            if "quality" in decision and isinstance(decision["quality"], dict):
                score = decision["quality"].get("overall_score")
                if score is not None:
                    successful_qualities.append(score)
        
        failed_qualities = []
        for decision in failed_decisions:
            if "quality" in decision and isinstance(decision["quality"], dict):
                score = decision["quality"].get("overall_score")
                if score is not None:
                    failed_qualities.append(score)
        
        avg_successful_quality = statistics.mean(successful_qualities) if successful_qualities else 0.0
        avg_failed_quality = statistics.mean(failed_qualities) if failed_qualities else 0.0
        
        result = {
            "time_window_hours": time_window_hours,
            "total_decisions": total_decisions,
            "average_decisions_per_trace": total_decisions / len(traces) if traces else 0.0,
            "average_decision_quality": avg_decision_quality,
            "average_context_length": avg_context_length,
            "decision_type_analysis": type_analysis,
            "quality_distribution": {
                "excellent": len([s for s in quality_scores if s >= 0.9]),
                "good": len([s for s in quality_scores if 0.8 <= s < 0.9]),
                "acceptable": len([s for s in quality_scores if 0.6 <= s < 0.8]),
                "needs_improvement": len([s for s in quality_scores if 0.4 <= s < 0.6]),
                "poor": len([s for s in quality_scores if s < 0.4]),
            },
            "success_correlation": {
                "successful_trace_decisions_avg_quality": avg_successful_quality,
                "failed_trace_decisions_avg_quality": avg_failed_quality,
                "quality_difference": avg_successful_quality - avg_failed_quality,
            },
            "top_decision_types": dict(sorted(
                type_analysis.items(),
                key=lambda x: x[1]["count"],
                reverse=True
            )[:10]),
            "calculated_at": datetime.now(timezone.utc).isoformat(),
        }
        
        logger.info(
            "Decision analytics calculated",
            extra={
                "total_decisions": total_decisions,
                "average_decision_quality": avg_decision_quality,
                "decision_types_count": len(type_analysis),
            }
        )
        
        return result
    
    def calculate_agent_behavior_patterns(
        self,
        agent_name: Optional[str] = None,
        time_window_hours: int = 24
    ) -> Dict[str, Any]:
        traces = self._get_traces_for_period(time_window_hours)
        
        if not traces:
            return {
                "agent_name": agent_name,
                "no_data": True,
                "calculated_at": datetime.now(timezone.utc).isoformat(),
            }
        
        if agent_name:
            traces = [
                t for t in traces 
                if t.agent_context and t.agent_context.get("agent_id") == agent_name
            ]
        
        operation_sequences = defaultdict(list)
        error_patterns = defaultdict(int)
        timing_patterns = defaultdict(list)
        
        for trace in traces:
            agent_id = trace.agent_context.get("agent_id") if trace.agent_context else "unknown"
            
            operation_sequences[agent_id].append({
                "operation": trace.operation,
                "success": trace.success,
                "duration": trace.duration_ms,
                "timestamp": trace.end_time.isoformat() if trace.end_time else None,
            })
            
            if not trace.success and trace.error:
                error_type = trace.error.get("type", "unknown")
                error_patterns[error_type] += 1
            
            if trace.end_time:
                hour = trace.end_time.hour
                timing_patterns[hour].append(trace.duration_ms or 0)
        
        sequence_analysis = {}
        for agent_id, sequences in operation_sequences.items():
            if len(sequences) >= 2:
                operation_pairs = defaultdict(int)
                for i in range(len(sequences) - 1):
                    pair = f"{sequences[i]['operation']} -> {sequences[i+1]['operation']}"
                    operation_pairs[pair] += 1
                
                common_sequences = sorted(
                    operation_pairs.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:5]
                
                sequence_analysis[agent_id] = {
                    "total_operations": len(sequences),
                    "success_rate": len([s for s in sequences if s["success"]]) / len(sequences) if sequences else 0,
                    "common_sequences": dict(common_sequences),
                    "average_duration": statistics.mean([s["duration"] for s in sequences if s["duration"]]) 
                    if any(s["duration"] for s in sequences) else 0,
                }
        
        timing_analysis = {}
        for hour, durations in timing_patterns.items():
            if durations:
                timing_analysis[hour] = {
                    "count": len(durations),
                    "avg_duration": statistics.mean(durations),
                    "min_duration": min(durations),
                    "max_duration": max(durations),
                }
        
        behavioral_consistency = {}
        for agent_id, sequences in operation_sequences.items():
            if len(sequences) >= 3:
                operations = [s["operation"] for s in sequences]
                unique_operations = set(operations)
                consistency_score = len(unique_operations) / len(operations) if operations else 0
                
                behavioral_consistency[agent_id] = {
                    "consistency_score": consistency_score,
                    "unique_operations": len(unique_operations),
                    "total_operations": len(operations),
                    "consistency_level": "high" if consistency_score < 0.3 else 
                                        "medium" if consistency_score < 0.6 else "low",
                }
        
        result = {
            "agent_name": agent_name or "all_agents",
            "time_window_hours": time_window_hours,
            "total_traces_analyzed": len(traces),
            "operation_sequences": sequence_analysis,
            "error_patterns": dict(error_patterns),
            "timing_patterns": timing_analysis,
            "behavioral_consistency": behavioral_consistency,
            "calculated_at": datetime.now(timezone.utc).isoformat(),
            "insights": self._generate_behavior_insights(
                sequence_analysis, error_patterns, behavioral_consistency
            ),
        }
        
        return result
    
    def _get_traces_for_period(self, hours: int) -> List[ExecutionTrace]:
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        return ExecutionTrace.query.filter(
            ExecutionTrace.end_time >= cutoff
        ).all()

    def _calculate_overall_quality_metrics(self, traces: List[ExecutionTrace]) -> Dict[str, Any]:
        if not traces:
            return {}
        
        quality_scores = []
        for trace in traces:
            if trace.quality_metrics and "composite_quality_score" in trace.quality_metrics:
                score = trace.quality_metrics["composite_quality_score"]
                if score is not None:
                    quality_scores.append(score)
        
        if not quality_scores:
            return {}
        
        return {
            "average": statistics.mean(quality_scores),
            "median": statistics.median(quality_scores),
            "min": min(quality_scores),
            "max": max(quality_scores),
            "std_dev": statistics.stdev(quality_scores) if len(quality_scores) > 1 else 0.0,
        }
    
    def _calculate_quality_distribution(self, traces: List[ExecutionTrace]) -> Dict[str, int]:
        distribution = {
            "excellent": 0,
            "good": 0,
            "acceptable": 0,
            "needs_improvement": 0,
            "poor": 0,
        }
        
        for trace in traces:
            if trace.quality_metrics and "composite_quality_score" in trace.quality_metrics:
                score = trace.quality_metrics["composite_quality_score"]
                if score is not None:
                    if score >= 0.9:
                        distribution["excellent"] += 1
                    elif score >= 0.8:
                        distribution["good"] += 1
                    elif score >= 0.6:
                        distribution["acceptable"] += 1
                    elif score >= 0.4:
                        distribution["needs_improvement"] += 1
                    else:
                        distribution["poor"] += 1
        
        return distribution
    
    def _generate_performance_recommendations(
        self,
        success_rate: float,
        avg_quality: float,
        avg_decision_quality: float
    ) -> List[str]:
        recommendations = []
        
        if success_rate is not None and success_rate < 0.7:
            recommendations.append("Focus on improving success rate through better error handling")
        
        if avg_quality is not None and avg_quality < 0.7:
            recommendations.append("Implement more rigorous quality checks for outputs")
        
        if avg_decision_quality is not None and avg_decision_quality < 0.6:
            recommendations.append("Improve decision-making process with more alternatives and rationale")
        
        if success_rate is not None and avg_quality is not None and success_rate > 0.9 and avg_quality > 0.85:
            recommendations.append("Consider optimizing for efficiency and speed")
        
        return recommendations
    
    def _generate_behavior_insights(
        self,
        sequence_analysis: Dict,
        error_patterns: Dict,
        behavioral_consistency: Dict
    ) -> List[str]:
        insights = []
        
        if error_patterns:
            most_common_error = max(error_patterns.items(), key=lambda x: x[1])[0]
            insights.append(f"Most common error: {most_common_error}")
        
        for agent_id, consistency in behavioral_consistency.items():
            if consistency["consistency_level"] == "low":
                insights.append(f"Agent {agent_id} shows low behavioral consistency")
            elif consistency["consistency_level"] == "high":
                insights.append(f"Agent {agent_id} shows high behavioral consistency (predictable)")
        
        if len(error_patterns) == 1 and list(error_patterns.values())[0] > 5:
            insights.append("Single error type dominating failures - consider targeted fix")
        
        return insights
