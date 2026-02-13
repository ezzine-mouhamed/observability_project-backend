"""Unit tests for observability API endpoints."""
import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timezone

from app.api.routes.observability import observability_bp, _generate_key_insights, _generate_agent_recommendations


class TestObservabilityAPI:
    """Unit tests for observability API endpoints."""
    
    def test_generate_key_insights_low_success_rate(self):
        """Test generating key insights with low success rate."""
        agent_metrics = {"success_rate": 0.65}  # Low success rate
        quality_metrics = {"overall_metrics": {"average": 0.8}}
        decision_analytics = {"average_decision_quality": 0.7}
        
        insights = _generate_key_insights(agent_metrics, quality_metrics, decision_analytics)
        
        assert any("Low overall success rate" in insight for insight in insights)
        assert any("65%" in insight for insight in insights)
    
    def test_generate_key_insights_high_success_rate(self):
        """Test generating key insights with high success rate."""
        agent_metrics = {"success_rate": 0.95}  # High success rate
        quality_metrics = {"overall_metrics": {"average": 0.8}}
        decision_analytics = {"average_decision_quality": 0.7}
        
        insights = _generate_key_insights(agent_metrics, quality_metrics, decision_analytics)
        
        assert any("High overall success rate" in insight for insight in insights)
        assert any("95%" in insight for insight in insights)
    
    def test_generate_key_insights_low_quality(self):
        """Test generating key insights with low quality."""
        agent_metrics = {"success_rate": 0.8}
        quality_metrics = {"overall_metrics": {"average": 0.65}}  # Low quality
        decision_analytics = {"average_decision_quality": 0.7}
        
        insights = _generate_key_insights(agent_metrics, quality_metrics, decision_analytics)
        
        assert any("Quality needs improvement" in insight for insight in insights)
        assert any("0.65" in insight for insight in insights)
    
    def test_generate_key_insights_low_decision_quality(self):
        """Test generating key insights with low decision quality."""
        agent_metrics = {"success_rate": 0.8}
        quality_metrics = {"overall_metrics": {"average": 0.8}}
        decision_analytics = {"average_decision_quality": 0.55}  # Low decision quality
        
        insights = _generate_key_insights(agent_metrics, quality_metrics, decision_analytics)
        
        assert any("Decision quality could be improved" in insight for insight in insights)
        assert any("0.55" in insight for insight in insights)
    
    def test_generate_key_insights_with_trend(self):
        """Test generating key insights with performance trend."""
        agent_metrics = {
            "success_rate": 0.8,
            "performance_trend": "improving"
        }
        quality_metrics = {"overall_metrics": {"average": 0.8}}
        decision_analytics = {"average_decision_quality": 0.7}
        
        insights = _generate_key_insights(agent_metrics, quality_metrics, decision_analytics)
        
        assert any("Performance trend is improving" in insight for insight in insights)
    
    def test_generate_key_insights_declining_trend(self):
        """Test generating key insights with declining trend."""
        agent_metrics = {
            "success_rate": 0.8,
            "performance_trend": "declining"
        }
        quality_metrics = {"overall_metrics": {"average": 0.8}}
        decision_analytics = {"average_decision_quality": 0.7}
        
        insights = _generate_key_insights(agent_metrics, quality_metrics, decision_analytics)
        
        assert any("Performance trend is declining" in insight for insight in insights)
    
    def test_generate_agent_recommendations_low_quality(self):
        """Test generating agent recommendations with low quality."""
        insights = {}
        performance = {
            "average_quality_score": 0.65,  # Low quality
            "success_rate": 0.8,
            "average_decision_quality": 0.7,
            "average_duration_ms": 1000
        }
        
        recommendations = _generate_agent_recommendations(insights, performance)
        
        quality_recs = [r for r in recommendations if r["type"] == "quality"]
        assert len(quality_recs) == 1
        assert quality_recs[0]["priority"] == "high"
        assert "Quality score (0.65)" in quality_recs[0]["reason"]
    
    def test_generate_agent_recommendations_low_success_rate(self):
        """Test generating agent recommendations with low success rate."""
        insights = {}
        performance = {
            "average_quality_score": 0.8,
            "success_rate": 0.65,  # Low success rate
            "average_decision_quality": 0.7,
            "average_duration_ms": 1000
        }
        
        recommendations = _generate_agent_recommendations(insights, performance)
        
        reliability_recs = [r for r in recommendations if r["type"] == "reliability"]
        assert len(reliability_recs) == 1
        assert reliability_recs[0]["priority"] == "high"
        assert "Success rate (65" in reliability_recs[0]["reason"]
    
    def test_generate_agent_recommendations_low_decision_quality(self):
        """Test generating agent recommendations with low decision quality."""
        insights = {}
        performance = {
            "average_quality_score": 0.8,
            "success_rate": 0.8,
            "average_decision_quality": 0.55,  # Low decision quality
            "average_duration_ms": 1000
        }
        
        recommendations = _generate_agent_recommendations(insights, performance)
        
        decision_recs = [r for r in recommendations if r["type"] == "decision_making"]
        assert len(decision_recs) == 1
        assert decision_recs[0]["priority"] == "medium"
        assert "Decision quality (0.55)" in decision_recs[0]["reason"]
    
    def test_generate_agent_recommendations_high_duration(self):
        """Test generating agent recommendations with high duration."""
        insights = {}
        performance = {
            "average_quality_score": 0.8,
            "success_rate": 0.8,
            "average_decision_quality": 0.7,
            "average_duration_ms": 6000  # High duration (6 seconds)
        }
        
        recommendations = _generate_agent_recommendations(insights, performance)
        
        efficiency_recs = [r for r in recommendations if r["type"] == "efficiency"]
        assert len(efficiency_recs) == 1
        assert efficiency_recs[0]["priority"] == "medium"
        assert "6000ms" in efficiency_recs[0]["reason"]
    
    def test_generate_agent_recommendations_declining_trend(self):
        """Test generating agent recommendations with declining trend."""
        insights = {"performance_trend": "declining"}
        performance = {
            "average_quality_score": 0.8,
            "success_rate": 0.8,
            "average_decision_quality": 0.7,
            "average_duration_ms": 1000
        }
        
        recommendations = _generate_agent_recommendations(insights, performance)
        
        monitoring_recs = [r for r in recommendations if r["type"] == "monitoring"]
        assert len(monitoring_recs) == 1
        assert monitoring_recs[0]["priority"] == "high"
        assert "Performance trend is declining" in monitoring_recs[0]["reason"]


class TestObservabilityAPIEndpoints:
    """Integration tests for observability API endpoints."""
    
    @patch('app.api.routes.observability.metrics_calculator')
    def test_get_agent_metrics_success(self, mock_metrics_calculator, client):
        """Test getting agent metrics successfully."""
        mock_metrics = {
            "total_agents": 3,
            "success_rate": 0.85,
            "agents": {"agent1": {"success_rate": 0.9}}
        }
        mock_metrics_calculator.calculate_agent_performance.return_value = mock_metrics
        
        response = client.get('/api/observability/agent-metrics?agent=agent1&time_window=48')
        
        assert response.status_code == 200
        assert response.json == mock_metrics
        mock_metrics_calculator.calculate_agent_performance.assert_called_once_with(
            agent_name="agent1",
            time_window_hours=48
        )
    
    @patch('app.api.routes.observability.metrics_calculator')
    def test_get_agent_metrics_default_params(self, mock_metrics_calculator, client):
        """Test getting agent metrics with default parameters."""
        mock_metrics = {"total_agents": 0}
        mock_metrics_calculator.calculate_agent_performance.return_value = mock_metrics
        
        response = client.get('/api/observability/agent-metrics')
        
        assert response.status_code == 200
        mock_metrics_calculator.calculate_agent_performance.assert_called_once_with(
            agent_name=None,
            time_window_hours=24
        )
    
    @patch('app.api.routes.observability.metrics_calculator')
    def test_get_agent_metrics_error(self, mock_metrics_calculator, client):
        """Test getting agent metrics with error."""
        mock_metrics_calculator.calculate_agent_performance.side_effect = Exception("DB error")
        
        response = client.get('/api/observability/agent-metrics')
        
        assert response.status_code == 500
        assert "error" in response.json
        assert "DB error" in response.json["error"]
    
    @patch('app.api.routes.observability.metrics_calculator')
    def test_get_quality_metrics_success(self, mock_metrics_calculator, client):
        """Test getting quality metrics successfully."""
        mock_metrics = {
            "overall_metrics": {"average": 0.85},
            "by_operation": {"llm_call": 0.9}
        }
        mock_metrics_calculator.calculate_quality_metrics.return_value = mock_metrics
        
        response = client.get('/api/observability/quality-metrics?group_by=agent&time_window=72')
        
        assert response.status_code == 200
        assert response.json == mock_metrics
        mock_metrics_calculator.calculate_quality_metrics.assert_called_once_with(
            time_window_hours=72,
            group_by="agent"
        )
    
    @patch('app.api.routes.observability.metrics_calculator')
    def test_get_decision_analytics_success(self, mock_metrics_calculator, client):
        """Test getting decision analytics successfully."""
        mock_analytics = {
            "total_decisions": 100,
            "average_quality": 0.8
        }
        mock_metrics_calculator.calculate_decision_analytics.return_value = mock_analytics
        
        response = client.get('/api/observability/decision-analytics?time_window=12')
        
        assert response.status_code == 200
        assert response.json == mock_analytics
        mock_metrics_calculator.calculate_decision_analytics.assert_called_once_with(
            time_window_hours=12
        )
    
    @patch('app.api.routes.observability.agent_observer')
    def test_get_agent_insights_success(self, mock_agent_observer, client):
        """Test getting agent insights successfully."""
        mock_insights = {
            "performance_trend": "improving",
            "key_patterns": ["efficient_decision_making"]
        }
        mock_agent_observer.get_agent_insights.return_value = mock_insights
        
        response = client.get('/api/observability/agent-insights/test_agent?time_window=36')
        
        assert response.status_code == 200
        assert response.json == mock_insights
        mock_agent_observer.get_agent_insights.assert_called_once_with(
            agent_name="test_agent",
            time_window_hours=36
        )
    
    @patch('app.api.routes.observability.metrics_calculator')
    def test_get_behavior_patterns_success(self, mock_metrics_calculator, client):
        """Test getting behavior patterns successfully."""
        mock_patterns = {
            "patterns": [{"type": "inefficient_thinking", "frequency": 5}]
        }
        mock_metrics_calculator.calculate_agent_behavior_patterns.return_value = mock_patterns
        
        response = client.get('/api/observability/behavior-patterns?agent=agent1&time_window=24')
        
        assert response.status_code == 200
        assert response.json == mock_patterns
        mock_metrics_calculator.calculate_agent_behavior_patterns.assert_called_once_with(
            agent_name="agent1",
            time_window_hours=24
        )
    
    @patch('app.api.routes.observability.trace_repo')
    def test_get_performance_trends_success(self, mock_trace_repo, client):
        """Test getting performance trends successfully."""
        mock_trends = {
            "trends": {"2024-01-01": {"average_quality": 0.8}},
            "days_analyzed": 7
        }
        mock_trace_repo.get_quality_trends.return_value = mock_trends
        
        response = client.get('/api/observability/performance-trends?days=14')
        
        assert response.status_code == 200
        assert response.json == mock_trends
        mock_trace_repo.get_quality_trends.assert_called_once_with(days=14)
    
    @patch('app.api.routes.observability.metrics_calculator')
    @patch('app.api.routes.observability.datetime')
    def test_get_observability_summary_success(self, mock_datetime, mock_metrics_calculator, client):
        """Test getting observability summary successfully."""
        # Mock datetime
        fixed_time = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        mock_datetime.now.return_value = fixed_time
        
        # Mock metrics
        agent_metrics = {
            "total_agents": 3,
            "total_traces": 100,
            "success_rate": 0.85,
            "agents": {"agent1": {"success_rate": 0.9}}
        }
        
        quality_metrics = {
            "overall_metrics": {"average": 0.8},
            "by_operation": {"llm_call": 0.85}
        }
        
        decision_analytics = {
            "total_decisions": 50,
            "average_decision_quality": 0.75
        }
        
        mock_metrics_calculator.calculate_agent_performance.return_value = agent_metrics
        mock_metrics_calculator.calculate_quality_metrics.return_value = quality_metrics
        mock_metrics_calculator.calculate_decision_analytics.return_value = decision_analytics
        
        response = client.get('/api/observability/summary?time_window=48')
        
        assert response.status_code == 200
        data = response.json
        
        # Check summary structure
        assert "summary" in data
        assert "agent_performance" in data
        assert "quality_overview" in data
        assert "decision_insights" in data
        assert "key_insights" in data
        
        # Check summary data
        summary = data["summary"]
        assert summary["time_window_hours"] == 48
        assert summary["agent_count"] == 3
        assert summary["total_traces"] == 100
        assert summary["overall_success_rate"] == 0.85
        assert summary["overall_quality"] == 0.8
        
        # Check metrics were called correctly
        mock_metrics_calculator.calculate_agent_performance.assert_called_once_with(
            time_window_hours=48
        )
        mock_metrics_calculator.calculate_quality_metrics.assert_called_once_with(
            time_window_hours=48,
            group_by='operation'
        )
        mock_metrics_calculator.calculate_decision_analytics.assert_called_once_with(
            time_window_hours=48
        )
    
    @patch('app.api.routes.observability.agent_observer')
    @patch('app.api.routes.observability.metrics_calculator')
    def test_get_agent_recommendations_success(self, mock_metrics_calculator, mock_agent_observer, client):
        """Test getting agent recommendations successfully."""
        # Mock insights and performance
        mock_insights = {
            "performance_trend": "stable",
            "key_patterns": []
        }
        
        mock_performance = {
            "average_quality_score": 0.75,
            "success_rate": 0.8,
            "average_decision_quality": 0.7,
            "average_duration_ms": 3000
        }
        
        mock_agent_observer.get_agent_insights.return_value = mock_insights
        mock_metrics_calculator.calculate_agent_performance.return_value = mock_performance
        
        response = client.get('/api/observability/agent/test_agent/recommendations?time_window=24')
        
        assert response.status_code == 200
        data = response.json
        
        assert data["agent_name"] == "test_agent"
        assert "recommendations" in data
        assert data["insights_summary"] == "stable"
        assert data["quality_score"] == 0.75
        assert data["time_window_hours"] == 24
        
        # Should have at least one recommendation for high duration
        assert len(data["recommendations"]) >= 1
        
        mock_agent_observer.get_agent_insights.assert_called_once_with(
            agent_name="test_agent",
            time_window_hours=24
        )
        mock_metrics_calculator.calculate_agent_performance.assert_called_once_with(
            agent_name="test_agent",
            time_window_hours=24
        )


class TestObservabilityAPIEdgeCases:
    """Edge case tests for observability API."""
    
    @patch('app.api.routes.observability.metrics_calculator')
    def test_get_agent_metrics_invalid_time_window(self, mock_metrics_calculator, client):
        """Test getting agent metrics with invalid time window parameter."""
        mock_metrics = {"total_agents": 0}
        mock_metrics_calculator.calculate_agent_performance.return_value = mock_metrics
        
        # Invalid time_window should be converted to int or cause error
        response = client.get('/api/observability/agent-metrics?time_window=invalid')
        
        # Depending on implementation, this might return 500 or default to 24
        # The int() conversion in the route will raise ValueError for 'invalid'
        assert response.status_code == 500  # Should get error
    
    def test_generate_key_insights_empty_metrics(self):
        """Test generating key insights with empty metrics."""
        agent_metrics = {}
        quality_metrics = {"overall_metrics": {}}
        decision_analytics = {}
        
        insights = _generate_key_insights(agent_metrics, quality_metrics, decision_analytics)
        
        # Should handle missing keys gracefully
        assert isinstance(insights, list)
    
    def test_generate_agent_recommendations_empty_data(self):
        """Test generating agent recommendations with empty data."""
        insights = {}
        performance = {}
        
        recommendations = _generate_agent_recommendations(insights, performance)
        
        # Should return empty list or handle missing data
        assert isinstance(recommendations, list)
    
    @patch('app.api.routes.observability.metrics_calculator')
    def test_get_observability_summary_partial_failure(self, mock_metrics_calculator, client):
        """Test getting observability summary when one metric calculation fails."""
        # Mock some successes and one failure
        agent_metrics = {"total_agents": 3}
        quality_metrics = {"overall_metrics": {"average": 0.8}}
        
        mock_metrics_calculator.calculate_agent_performance.return_value = agent_metrics
        mock_metrics_calculator.calculate_quality_metrics.return_value = quality_metrics
        mock_metrics_calculator.calculate_decision_analytics.side_effect = Exception("DB error")
        
        response = client.get('/api/observability/summary')
        
        # Should return 500 error if any component fails
        assert response.status_code == 500
        assert "error" in response.json
