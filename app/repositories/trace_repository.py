from collections import defaultdict
from datetime import datetime, timedelta, timezone
import statistics
from typing import Any, Dict, List, Optional

from app.extensions import db
from app.models.trace import ExecutionTrace


class TraceRepository:
    """Repository for ExecutionTrace model operations."""

    def save(self, trace: ExecutionTrace) -> ExecutionTrace:
        """Save trace to database."""
        db.session.add(trace)
        db.session.commit()
        return trace

    def get_by_id(self, trace_id: int) -> Optional[ExecutionTrace]:
        """Get trace by database ID."""
        return ExecutionTrace.query.get(trace_id)

    def get_by_trace_id(self, trace_uuid: str) -> Optional[ExecutionTrace]:
        """Get trace by UUID trace ID."""
        return ExecutionTrace.query.filter_by(trace_id=trace_uuid).first()

    def get_traces_for_task(self, task_id: int) -> List[ExecutionTrace]:
        """Get all traces for a task."""
        return (
            ExecutionTrace.query.filter_by(task_id=task_id)
            .order_by(ExecutionTrace.start_time)
            .all()
        )

    def get_trace_tree(self, root_trace_id: str) -> Dict[str, Any]:
        """Get a trace and all its child traces as a tree."""
        root_trace = self.get_by_trace_id(root_trace_id)
        if not root_trace:
            return {}

        # Get all traces with this root as ancestor
        all_traces = ExecutionTrace.query.filter(
            ExecutionTrace.trace_id.like(f"{root_trace_id}%")
        ).all()

        # Build tree structure
        trace_dict = {trace.trace_id: trace for trace in all_traces}
        tree = self._build_tree(root_trace_id, trace_dict)

        return tree

    def _build_tree(
        self, current_id: str, trace_dict: Dict[str, ExecutionTrace]
    ) -> Dict[str, Any]:
        """Recursively build trace tree."""
        trace = trace_dict.get(current_id)
        if not trace:
            return {}

        node = trace.to_dict()
        node["children"] = []

        # Find children
        for trace_id, child_trace in trace_dict.items():
            if child_trace.parent_trace_id == current_id:
                child_node = self._build_tree(trace_id, trace_dict)
                if child_node:
                    node["children"].append(child_node)

        return node

    def find_by_operation(
        self, operation: str, limit: int = 100
    ) -> List[ExecutionTrace]:
        """Find traces by operation type."""
        return (
            ExecutionTrace.query.filter_by(operation=operation)
            .order_by(ExecutionTrace.start_time.desc())
            .limit(limit)
            .all()
        )

    def find_failed_traces(
        self, hours: int = 24, limit: int = 100
    ) -> List[ExecutionTrace]:
        """Find failed traces within time period."""
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        return (
            ExecutionTrace.query.filter(
                not ExecutionTrace.success and ExecutionTrace.created_at >= cutoff
            )
            .order_by(ExecutionTrace.created_at.desc())
            .limit(limit)
            .all()
        )

    def find_slow_traces(
        self, threshold_ms: int = 1000, hours: int = 24, limit: int = 100
    ) -> List[ExecutionTrace]:
        """Find slow traces exceeding threshold."""
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        return (
            ExecutionTrace.query.filter(
                ExecutionTrace.duration_ms >= threshold_ms,
                ExecutionTrace.created_at >= cutoff,
            )
            .order_by(ExecutionTrace.duration_ms.desc())
            .limit(limit)
            .all()
        )

    def get_statistics(self, hours: int = 24) -> Dict[str, Any]:
        """Get trace statistics for a time period."""
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)

        traces = ExecutionTrace.query.filter(ExecutionTrace.created_at >= cutoff).all()

        if not traces:
            return {
                "total": 0,
                "success_rate": 0,
                "by_operation": {},
                "avg_duration": 0,
                "error_count": 0,
            }

        by_operation = {}
        total_duration = 0
        successful_traces = 0
        error_count = 0

        for trace in traces:
            # Count by operation
            by_operation[trace.operation] = by_operation.get(trace.operation, 0) + 1

            # Calculate average duration
            if trace.duration_ms:
                total_duration += trace.duration_ms

            # Count successes
            if trace.success:
                successful_traces += 1

            # Count errors
            if trace.error:
                error_count += 1

        avg_duration = total_duration / len(traces) if traces else 0
        success_rate = successful_traces / len(traces) if traces else 0

        return {
            "total": len(traces),
            "success_rate": success_rate,
            "by_operation": by_operation,
            "avg_duration": avg_duration,
            "error_count": error_count,
        }

    def cleanup_old_traces(self, days: int = 30) -> int:
        """Clean up traces older than specified days."""
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        deleted_count = ExecutionTrace.query.filter(
            ExecutionTrace.created_at < cutoff
        ).delete()
        db.session.commit()
        return deleted_count


    def get_agent_metrics(self, time_window_hours: int = 24) -> Dict[str, Any]:
        """Get metrics aggregated by agent."""
        cutoff = datetime.now(timezone.utc) - timedelta(hours=time_window_hours)
        
        # Query for agent metrics
        traces = ExecutionTrace.query.filter(
            ExecutionTrace.created_at >= cutoff,
            ExecutionTrace.agent_context.isnot(None)
        ).all()
        
        if not traces:
            return {"agents": {}, "total": 0}
        
        # Group by agent
        agent_metrics = defaultdict(lambda: {
            "total_traces": 0,
            "successful_traces": 0,
            "failed_traces": 0,
            "total_duration": 0,
            "operations": defaultdict(int),
            "quality_scores": [],
        })
        
        for trace in traces:
            agent_id = trace.agent_context.get("agent_id", "unknown") if trace.agent_context else "unknown"
            metrics = agent_metrics[agent_id]
            
            metrics["total_traces"] += 1
            if trace.success:
                metrics["successful_traces"] += 1
            else:
                metrics["failed_traces"] += 1
            
            if trace.duration_ms:
                metrics["total_duration"] += trace.duration_ms
            
            metrics["operations"][trace.operation] += 1
            
            if trace.quality_metrics and "quality_score" in trace.quality_metrics:
                metrics["quality_scores"].append(trace.quality_metrics["quality_score"])
        
        # Calculate derived metrics
        result = {}
        for agent_id, metrics in agent_metrics.items():
            total_traces = metrics["total_traces"]
            
            success_rate = metrics["successful_traces"] / total_traces if total_traces > 0 else 0
            avg_duration = metrics["total_duration"] / total_traces if total_traces > 0 else 0
            
            # Quality metrics
            quality_scores = metrics["quality_scores"]
            avg_quality = statistics.mean(quality_scores) if quality_scores else 0
            quality_distribution = {
                "excellent": len([s for s in quality_scores if s >= 0.9]),
                "good": len([s for s in quality_scores if 0.8 <= s < 0.9]),
                "acceptable": len([s for s in quality_scores if 0.6 <= s < 0.8]),
                "needs_improvement": len([s for s in quality_scores if 0.4 <= s < 0.6]),
                "poor": len([s for s in quality_scores if s < 0.4]),
            }
            
            # Most common operations
            common_operations = sorted(
                metrics["operations"].items(),
                key=lambda x: x[1],
                reverse=True
            )[:5]
            
            result[agent_id] = {
                "total_traces": total_traces,
                "success_rate": success_rate,
                "average_duration_ms": avg_duration,
                "average_quality_score": avg_quality,
                "quality_distribution": quality_distribution,
                "most_common_operations": dict(common_operations),
            }
        
        return {
            "agents": dict(result),
            "total_agents": len(result),
            "total_traces": len(traces),
            "time_window_hours": time_window_hours,
        }
    
    def get_quality_trends(self, days: int = 7) -> Dict[str, Any]:
        """Get quality trends over time."""
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        
        # Group by day
        daily_quality = defaultdict(list)
        
        traces = ExecutionTrace.query.filter(
            ExecutionTrace.created_at >= cutoff,
            ExecutionTrace.quality_metrics.isnot(None)
        ).all()
        
        for trace in traces:
            if trace.quality_metrics and "quality_score" in trace.quality_metrics:
                day = trace.created_at.strftime("%Y-%m-%d")
                daily_quality[day].append(trace.quality_metrics["quality_score"])
        
        # Calculate daily averages
        trends = {}
        for day, scores in daily_quality.items():
            if scores:
                trends[day] = {
                    "average_quality": statistics.mean(scores),
                    "min_quality": min(scores),
                    "max_quality": max(scores),
                    "sample_count": len(scores),
                }
        
        # Sort by date
        sorted_trends = dict(sorted(trends.items()))
        
        return {
            "trends": sorted_trends,
            "days_analyzed": days,
            "total_samples": sum(len(scores) for scores in daily_quality.values()),
        }
    
    def get_decision_analytics(self, time_window_hours: int = 24) -> Dict[str, Any]:
        """Get analytics about decisions in traces."""
        cutoff = datetime.now(timezone.utc) - timedelta(hours=time_window_hours)
        
        traces = ExecutionTrace.query.filter(
            ExecutionTrace.created_at >= cutoff,
            ExecutionTrace.decisions.isnot(None)
        ).all()
        
        if not traces:
            return {"no_decisions": True}
        
        # Extract all decisions
        all_decisions = []
        for trace in traces:
            if trace.decisions:
                for decision in trace.decisions:
                    if isinstance(decision, dict):
                        decision_record = {
                            "type": decision.get("type", "unknown"),
                            "timestamp": decision.get("timestamp"),
                            "trace_success": trace.success,
                            "trace_operation": trace.operation,
                            "agent": trace.agent_context.get("agent_id") if trace.agent_context else None,
                        }
                        
                        # Add quality if available
                        if "quality" in decision and isinstance(decision["quality"], dict):
                            decision_record["quality_score"] = decision["quality"].get("overall_score")
                        
                        all_decisions.append(decision_record)
        
        # Group by decision type
        decisions_by_type = defaultdict(list)
        for decision in all_decisions:
            decisions_by_type[decision["type"]].append(decision)
        
        # Calculate type statistics
        type_stats = {}
        for decision_type, type_decisions in decisions_by_type.items():
            quality_scores = [d.get("quality_score") for d in type_decisions if d.get("quality_score") is not None]
            success_rates = [1 if d["trace_success"] else 0 for d in type_decisions]
            
            type_stats[decision_type] = {
                "count": len(type_decisions),
                "average_quality": statistics.mean(quality_scores) if quality_scores else None,
                "success_rate": statistics.mean(success_rates) if success_rates else 0,
                "percentage": len(type_decisions) / len(all_decisions),
            }
        
        # Overall statistics
        overall_quality_scores = [d.get("quality_score") for d in all_decisions if d.get("quality_score") is not None]
        
        return {
            "total_decisions": len(all_decisions),
            "unique_decision_types": len(type_stats),
            "average_quality": statistics.mean(overall_quality_scores) if overall_quality_scores else None,
            "decision_types": dict(sorted(
                type_stats.items(),
                key=lambda x: x[1]["count"],
                reverse=True
            )),
            "time_window_hours": time_window_hours,
        }
