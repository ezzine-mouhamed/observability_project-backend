"""
Observability API endpoints for dashboard.
"""
from datetime import datetime, timezone

from flask import Blueprint, jsonify, request

from app.observability.agent_observer import AgentObserver
from app.observability.metrics_calculator import MetricsCalculator
from app.observability.tracer import Tracer
from app.repositories.trace_repository import TraceRepository
from app.repositories.task_repository import TaskRepository
from app.utils.logger import get_logger

logger = get_logger(__name__)

observability_bp = Blueprint('observability', __name__, url_prefix='/api/observability')

# Initialize components
trace_repo = TraceRepository()
task_repo = TaskRepository()
tracer = Tracer()
agent_observer = AgentObserver(tracer)
metrics_calculator = MetricsCalculator(trace_repo)


@observability_bp.route('/agent-metrics', methods=['GET'])
def get_agent_metrics():
    """Get metrics for all agents or specific agent."""
    agent_name = request.args.get('agent')
    time_window = int(request.args.get('time_window', 24))
    
    try:
        metrics = metrics_calculator.calculate_agent_performance(
            agent_name=agent_name,
            time_window_hours=time_window
        )
        return jsonify(metrics)
    except Exception as e:
        logger.error(f"Failed to get agent metrics: {str(e)}")
        return jsonify({"error": str(e)}), 500


@observability_bp.route('/quality-metrics', methods=['GET'])
def get_quality_metrics():
    """Get quality metrics grouped by operation, agent, or time."""
    group_by = request.args.get('group_by', 'operation')
    time_window = int(request.args.get('time_window', 24))
    
    try:
        metrics = metrics_calculator.calculate_quality_metrics(
            time_window_hours=time_window,
            group_by=group_by
        )
        return jsonify(metrics)
    except Exception as e:
        logger.error(f"Failed to get quality metrics: {str(e)}")
        return jsonify({"error": str(e)}), 500


@observability_bp.route('/decision-analytics', methods=['GET'])
def get_decision_analytics():
    """Get analytics about decisions."""
    time_window = int(request.args.get('time_window', 24))
    
    try:
        analytics = metrics_calculator.calculate_decision_analytics(
            time_window_hours=time_window
        )
        return jsonify(analytics)
    except Exception as e:
        logger.error(f"Failed to get decision analytics: {str(e)}")
        return jsonify({"error": str(e)}), 500


@observability_bp.route('/agent-insights/<agent_name>', methods=['GET'])
def get_agent_insights(agent_name: str):
    """Get insights about a specific agent."""
    time_window = int(request.args.get('time_window', 24))
    
    try:
        insights = agent_observer.get_agent_insights(
            agent_name=agent_name,
            time_window_hours=time_window
        )
        return jsonify(insights)
    except Exception as e:
        logger.error(f"Failed to get agent insights: {str(e)}")
        return jsonify({"error": str(e)}), 500


@observability_bp.route('/behavior-patterns', methods=['GET'])
def get_behavior_patterns():
    """Get behavior patterns for agents."""
    agent_name = request.args.get('agent')
    time_window = int(request.args.get('time_window', 24))
    
    try:
        patterns = metrics_calculator.calculate_agent_behavior_patterns(
            agent_name=agent_name,
            time_window_hours=time_window
        )
        return jsonify(patterns)
    except Exception as e:
        logger.error(f"Failed to get behavior patterns: {str(e)}")
        return jsonify({"error": str(e)}), 500


@observability_bp.route('/performance-trends', methods=['GET'])
def get_performance_trends():
    """Get performance trends over time."""
    days = int(request.args.get('days', 7))
    
    try:
        trends = trace_repo.get_quality_trends(days=days)
        return jsonify(trends)
    except Exception as e:
        logger.error(f"Failed to get performance trends: {str(e)}")
        return jsonify({"error": str(e)}), 500


@observability_bp.route('/summary', methods=['GET'])
def get_observability_summary():
    """Get a comprehensive observability summary."""
    time_window = int(request.args.get('time_window', 24))
    
    try:
        # Get multiple metrics in parallel
        agent_metrics = metrics_calculator.calculate_agent_performance(
            time_window_hours=time_window
        )
        quality_metrics = metrics_calculator.calculate_quality_metrics(
            time_window_hours=time_window,
            group_by='operation'
        )
        decision_analytics = metrics_calculator.calculate_decision_analytics(
            time_window_hours=time_window
        )
        
        summary = {
            "summary": {
                "time_window_hours": time_window,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "agent_count": agent_metrics.get("total_agents", 0),
                "total_traces": agent_metrics.get("total_traces", 0),
                "overall_success_rate": agent_metrics.get("success_rate", 0),
                "overall_quality": quality_metrics.get("overall_metrics", {}).get("average", 0),
            },
            "agent_performance": agent_metrics,
            "quality_overview": quality_metrics,
            "decision_insights": decision_analytics,
            "key_insights": _generate_key_insights(agent_metrics, quality_metrics, decision_analytics),
        }
        
        return jsonify(summary)
    except Exception as e:
        logger.error(f"Failed to get observability summary: {str(e)}")
        return jsonify({"error": str(e)}), 500


@observability_bp.route('/agent/<agent_name>/recommendations', methods=['GET'])
def get_agent_recommendations(agent_name: str):
    """Get recommendations for improving agent performance."""
    time_window = int(request.args.get('time_window', 24))
    
    try:
        # Get agent insights
        insights = agent_observer.get_agent_insights(
            agent_name=agent_name,
            time_window_hours=time_window
        )
        
        # Get agent performance
        performance = metrics_calculator.calculate_agent_performance(
            agent_name=agent_name,
            time_window_hours=time_window
        )
        
        # Generate recommendations
        recommendations = _generate_agent_recommendations(insights, performance)
        
        return jsonify({
            "agent_name": agent_name,
            "recommendations": recommendations,
            "insights_summary": insights.get("performance_trend", "stable"),
            "quality_score": performance.get("average_quality_score", 0),
            "time_window_hours": time_window,
        })
    except Exception as e:
        logger.error(f"Failed to get agent recommendations: {str(e)}")
        return jsonify({"error": str(e)}), 500


def _generate_key_insights(agent_metrics, quality_metrics, decision_analytics):
    """Generate key insights from metrics."""
    insights = []
    
    # Agent performance insights
    success_rate = agent_metrics.get("success_rate", 0)
    if success_rate is not None:
        if success_rate < 0.7:
            insights.append(f"Low overall success rate ({success_rate:.1%}) - consider improving error handling")
        elif success_rate > 0.9:
            insights.append(f"High overall success rate ({success_rate:.1%}) - system is performing well")
    
    # Quality insights
    overall_quality = quality_metrics.get("overall_metrics", {}).get("average", 0)
    if overall_quality is not None and overall_quality < 0.7:
        insights.append(f"Quality needs improvement (score: {overall_quality:.2f})")
    
    # Decision insights
    decision_quality = decision_analytics.get("average_decision_quality", 0)
    if decision_quality is not None and decision_quality < 0.6:
        insights.append(f"Decision quality could be improved (score: {decision_quality:.2f})")
    
    # Add trend-based insights
    if "performance_trend" in agent_metrics:
        trend = agent_metrics["performance_trend"]
        if trend == "improving":
            insights.append("Performance trend is improving")
        elif trend == "declining":
            insights.append("Performance trend is declining - investigate recent changes")
    
    return insights


def _generate_agent_recommendations(insights, performance):
    """Generate specific recommendations for an agent."""
    recommendations = []
    
    # Quality-based recommendations
    quality_score = performance.get("average_quality_score", 0)
    if quality_score is not None and quality_score < 0.7:
        recommendations.append({
            "type": "quality",
            "priority": "high",
            "action": "Implement additional quality checks",
            "reason": f"Quality score ({quality_score:.2f}) below threshold",
        })
    
    # Success rate recommendations
    success_rate = performance.get("success_rate", 0)
    if success_rate is not None and success_rate < 0.7:
        recommendations.append({
            "type": "reliability",
            "priority": "high",
            "action": "Review error patterns and improve error handling",
            "reason": f"Success rate ({success_rate:.1%}) needs improvement",
        })
    
    # Decision quality recommendations
    decision_quality = performance.get("average_decision_quality", 0)
    if decision_quality is not None and decision_quality < 0.6:
        recommendations.append({
            "type": "decision_making",
            "priority": "medium",
            "action": "Record more decision alternatives and rationale",
            "reason": f"Decision quality ({decision_quality:.2f}) could be improved",
        })
    
    # Efficiency recommendations
    avg_duration = performance.get("average_duration_ms", 0)
    if avg_duration is not None and avg_duration > 5000:  # 5 seconds
        recommendations.append({
            "type": "efficiency",
            "priority": "medium",
            "action": "Optimize slow operations",
            "reason": f"Average duration ({avg_duration:.0f}ms) is high",
        })
    
    # Self-evaluation recommendations
    if insights.get("performance_trend") == "declining":
        recommendations.append({
            "type": "monitoring",
            "priority": "high",
            "action": "Investigate recent performance decline",
            "reason": "Performance trend is declining",
        })
    
    return recommendations
