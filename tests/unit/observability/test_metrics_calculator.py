"""
Unit tests for MetricsCalculator class.
"""
import pytest
from datetime import datetime, timedelta, timezone
from unittest.mock import Mock, patch, MagicMock
from collections import defaultdict

from app.observability.metrics_calculator import MetricsCalculator


class TestMetricsCalculator:
    """Unit tests for MetricsCalculator."""
    
    def test_initialization(self):
        """Test MetricsCalculator initializes correctly."""
        # Test with default repository
        calculator = MetricsCalculator()
        assert calculator.trace_repo is not None
        assert calculator._cache == {}
        assert calculator._cache_ttl == timedelta(minutes=5)
        
        # Test with custom repository
        mock_repo = Mock()
        calculator = MetricsCalculator(trace_repository=mock_repo)
        assert calculator.trace_repo == mock_repo

    @patch('app.observability.metrics_calculator.ExecutionTrace')
    def test_calculate_agent_performance_no_traces(self, mock_trace_model):
        """Test performance calculation with no traces."""
        # Setup
        calculator = MetricsCalculator()
        
        # Mock the SQLAlchemy query chain properly
        mock_query = Mock()
        mock_filter = Mock()
        
        # Create a mock column descriptor for end_time
        mock_column = Mock()
        # Make the >= comparison work
        mock_column.__ge__ = Mock(return_value=Mock())
        
        # Set up the mock model
        mock_trace_model.end_time = mock_column
        mock_trace_model.query = mock_query
        mock_query.filter.return_value = mock_filter
        mock_filter.all.return_value = []
        
        # Test
        result = calculator.calculate_agent_performance()
        
        # Verify
        assert result["agent_name"] is None
        assert result["total_traces"] == 0
        assert result["no_data"] == True
        assert "calculated_at" in result

    def test_calculate_agent_performance_with_traces(self):
        """Test performance calculation with sample traces."""
        # Setup
        calculator = MetricsCalculator()
        
        # Create mock traces - use Mock() without spec to avoid SQLAlchemy issues
        mock_traces = []
        for i in range(5):
            trace = Mock()
            trace.success = i < 4  # 4 successful, 1 failed
            trace.duration_ms = 1000 if i % 2 == 0 else 2000
            trace.quality_metrics = {"composite_quality_score": 0.8 + (i * 0.05)}
            trace.decisions = [
                {"type": "test_decision", "quality": {"overall_score": 0.7}},
                {"type": "test_decision", "quality": {"overall_score": 0.8}}
            ] if i % 2 == 0 else []
            trace.agent_observations = ["obs1", "obs2"] if i % 2 == 0 else []
            trace.agent_context = {"agent_id": f"agent_{i % 2}"}
            trace.end_time = datetime.now(timezone.utc) - timedelta(hours=1)
            trace.error = {"type": "test_error"} if i == 4 else None
            mock_traces.append(trace)
        
        with patch.object(calculator, '_get_traces_for_period') as mock_get_traces:
            mock_get_traces.return_value = mock_traces
            
            # Test
            result = calculator.calculate_agent_performance()
            
            # Verify
            assert result["agent_name"] == "all_agents"
            assert result["total_traces"] == 5
            assert result["success_rate"] == 0.8  # 4 out of 5
            assert result["average_quality_score"] == pytest.approx(0.9, rel=0.1)
            assert result["average_duration_ms"] == 1400  # (1000*3 + 2000*2)/5
            assert result["failed_traces"] == 1
            assert "error_types" in result
            assert "performance_trend" in result
            assert "quality_distribution" in result
            assert "recommendations" in result
    
    def test_calculate_agent_performance_specific_agent(self):
        """Test performance calculation for a specific agent."""
        # Setup
        calculator = MetricsCalculator()
        
        # Create mock traces
        mock_traces = []
        for i in range(3):
            trace = Mock()
            trace.success = True
            trace.duration_ms = 1000
            trace.quality_metrics = {"composite_quality_score": 0.9}
            trace.decisions = []
            trace.agent_observations = []
            trace.agent_context = {"agent_id": "test_agent"}
            trace.end_time = datetime.now(timezone.utc) - timedelta(hours=1)
            trace.error = None
            mock_traces.append(trace)
        
        with patch.object(calculator, '_get_traces_for_period') as mock_get_traces:
            mock_get_traces.return_value = mock_traces
            
            # Test
            result = calculator.calculate_agent_performance(agent_name="test_agent")
            
            # Verify
            assert result["agent_name"] == "test_agent"
            assert result["total_traces"] == 3
            assert result["success_rate"] == 1.0
    
    def test_calculate_quality_metrics_no_traces(self):
        """Test quality metrics calculation with no traces."""
        # Setup
        calculator = MetricsCalculator()
        
        with patch.object(calculator, '_get_traces_for_period') as mock_get_traces:
            mock_get_traces.return_value = []
            
            # Test
            result = calculator.calculate_quality_metrics()
            
            # Verify
            assert result["group_by"] == "operation"
            assert result["no_data"] == True
            assert "calculated_at" in result
    
    def test_calculate_quality_metrics_group_by_operation(self):
        """Test quality metrics grouped by operation."""
        # Setup
        calculator = MetricsCalculator()
        
        # Create mock traces
        mock_traces = []
        operations = ["llm_call", "data_transform", "llm_call", "validation"]
        
        for i, operation in enumerate(operations):
            trace = Mock()
            trace.operation = operation
            trace.success = i % 2 == 0
            trace.quality_metrics = {"composite_quality_score": 0.7 + (i * 0.1)}
            trace.agent_context = {"agent_id": f"agent_{i % 2}"}
            trace.end_time = datetime.now(timezone.utc) - timedelta(hours=1)
            mock_traces.append(trace)
        
        with patch.object(calculator, '_get_traces_for_period') as mock_get_traces:
            mock_get_traces.return_value = mock_traces
            
            # Test
            result = calculator.calculate_quality_metrics(group_by="operation")
            
            # Verify
            assert result["group_by"] == "operation"
            assert result["total_traces"] == 4
            assert "groups" in result
            assert "llm_call" in result["groups"]
            assert "data_transform" in result["groups"]
            assert "validation" in result["groups"]
            assert "overall_metrics" in result
    
    def test_calculate_quality_metrics_group_by_agent(self):
        """Test quality metrics grouped by agent."""
        # Setup
        calculator = MetricsCalculator()
        
        # Create mock traces
        mock_traces = []
        agents = ["agent_0", "agent_1", "agent_0", "agent_1"]
        
        for i, agent in enumerate(agents):
            trace = Mock()
            trace.operation = "test_operation"
            trace.success = True
            trace.quality_metrics = {"composite_quality_score": 0.8}
            trace.agent_context = {"agent_id": agent}
            trace.end_time = datetime.now(timezone.utc) - timedelta(hours=1)
            mock_traces.append(trace)
        
        with patch.object(calculator, '_get_traces_for_period') as mock_get_traces:
            mock_get_traces.return_value = mock_traces
            
            # Test
            result = calculator.calculate_quality_metrics(group_by="agent")
            
            # Verify
            assert result["group_by"] == "agent"
            assert "agent_0" in result["groups"]
            assert "agent_1" in result["groups"]
            assert result["groups"]["agent_0"]["trace_count"] == 2
            assert result["groups"]["agent_1"]["trace_count"] == 2
    
    def test_calculate_quality_metrics_group_by_hour(self):
        """Test quality metrics grouped by hour."""
        # Setup
        calculator = MetricsCalculator()
        
        # Create mock traces with different hours
        mock_traces = []
        current_time = datetime.now(timezone.utc)
        
        for i in range(3):
            trace = Mock()
            trace.operation = "test_operation"
            trace.success = True
            trace.quality_metrics = {"composite_quality_score": 0.8}
            trace.agent_context = {"agent_id": "test_agent"}
            trace.end_time = current_time - timedelta(hours=i)
            mock_traces.append(trace)
        
        with patch.object(calculator, '_get_traces_for_period') as mock_get_traces:
            mock_get_traces.return_value = mock_traces
            
            # Test
            result = calculator.calculate_quality_metrics(group_by="hour")
            
            # Verify
            assert result["group_by"] == "hour"
            assert len(result["groups"]) >= 1
    
    def test_calculate_decision_analytics_no_traces(self):
        """Test decision analytics with no traces."""
        # Setup
        calculator = MetricsCalculator()
        
        with patch.object(calculator, '_get_traces_for_period') as mock_get_traces:
            mock_get_traces.return_value = []
            
            # Test
            result = calculator.calculate_decision_analytics()
            
            # Verify
            assert result["no_data"] == True
            assert "calculated_at" in result
    
    def test_calculate_decision_analytics_with_decisions(self):
        """Test decision analytics with sample decisions."""
        # Setup
        calculator = MetricsCalculator()
        
        # Create mock traces with decisions
        mock_traces = []
        
        for i in range(3):
            trace = Mock()
            trace.operation = f"operation_{i}"
            trace.success = i < 2  # First 2 successful, last one failed
            trace.agent_context = {"agent_id": f"agent_{i}"}
            trace.decisions = [
                {
                    "type": "choice_decision",
                    "quality": {"overall_score": 0.8},
                    "context": {"options": ["A", "B", "C"]}
                },
                {
                    "type": "validation_decision",
                    "quality": {"overall_score": 0.9},
                    "context": {"rule": "length_check"}
                }
            ]
            mock_traces.append(trace)
        
        with patch.object(calculator, '_get_traces_for_period') as mock_get_traces:
            mock_get_traces.return_value = mock_traces
            
            # Test
            result = calculator.calculate_decision_analytics()
            
            # Verify
            assert result["total_decisions"] == 6  # 3 traces * 2 decisions each
            assert "average_decision_quality" in result
            assert result["average_decision_quality"] == pytest.approx(0.85, rel=0.1)
            assert "decision_type_analysis" in result
            assert "choice_decision" in result["decision_type_analysis"]
            assert "validation_decision" in result["decision_type_analysis"]
            assert "quality_distribution" in result
            assert "success_correlation" in result
            assert "top_decision_types" in result
    
    def test_calculate_agent_behavior_patterns_no_traces(self):
        """Test behavior patterns with no traces."""
        # Setup
        calculator = MetricsCalculator()
        
        with patch.object(calculator, '_get_traces_for_period') as mock_get_traces:
            mock_get_traces.return_value = []
            
            # Test
            result = calculator.calculate_agent_behavior_patterns()
            
            # Verify
            assert result["no_data"] == True
            assert "calculated_at" in result
    
    def test_calculate_agent_behavior_patterns_with_traces(self):
        """Test behavior patterns with sample traces."""
        # Setup
        calculator = MetricsCalculator()
        
        # Create mock traces for behavior analysis
        mock_traces = []
        current_time = datetime.now(timezone.utc)
        
        for i in range(5):
            trace = Mock()
            trace.operation = f"op_{(i % 3) + 1}"  # ops: op1, op2, op3, op1, op2
            trace.success = i < 4  # All but last successful
            trace.duration_ms = 1000 + (i * 500)
            trace.agent_context = {"agent_id": "test_agent"}
            trace.end_time = current_time - timedelta(hours=i)
            trace.error = {"type": "timeout"} if i == 4 else None
            mock_traces.append(trace)
        
        with patch.object(calculator, '_get_traces_for_period') as mock_get_traces:
            mock_get_traces.return_value = mock_traces
            
            # Test
            result = calculator.calculate_agent_behavior_patterns(agent_name="test_agent")
            
            # Verify
            assert result["agent_name"] == "test_agent"
            assert result["total_traces_analyzed"] == 5
            assert "operation_sequences" in result
            assert "error_patterns" in result
            assert "timing_patterns" in result
            assert "behavioral_consistency" in result
            assert "insights" in result

    @patch('app.observability.metrics_calculator.ExecutionTrace')
    def test_get_traces_for_period(self, mock_trace_model):
        """Test getting traces for a time period."""
        # Setup
        calculator = MetricsCalculator()
        
        # Mock the SQLAlchemy query chain properly
        mock_traces = [Mock() for _ in range(3)]
        mock_query = Mock()
        mock_filter = Mock()
        
        # Create a mock column descriptor for end_time
        mock_column = Mock()
        mock_column.__ge__ = Mock(return_value=Mock())
        
        # Set up the mock model
        mock_trace_model.end_time = mock_column
        mock_trace_model.query = mock_query
        mock_query.filter.return_value = mock_filter
        mock_filter.all.return_value = mock_traces
        
        # Test
        traces = calculator._get_traces_for_period(hours=24)
        
        # Verify
        assert len(traces) == 3
        mock_query.filter.assert_called_once()

    def test_calculate_overall_quality_metrics(self):
        """Test overall quality metrics calculation."""
        # Setup
        calculator = MetricsCalculator()
        
        # Create mock traces with quality scores
        mock_traces = []
        for i in range(5):
            trace = Mock()
            trace.quality_metrics = {"composite_quality_score": 0.6 + (i * 0.1)}
            mock_traces.append(trace)
        
        # Test
        result = calculator._calculate_overall_quality_metrics(mock_traces)
        
        # Verify
        assert "average" in result
        assert "median" in result
        assert "min" in result
        assert "max" in result
        assert "std_dev" in result
        assert result["min"] == 0.6
        assert result["max"] == 1.0
        assert result["average"] == pytest.approx(0.8, rel=0.1)
    
    def test_calculate_overall_quality_metrics_no_scores(self):
        """Test overall quality metrics with no quality scores."""
        # Setup
        calculator = MetricsCalculator()
        
        # Create mock traces without quality scores
        mock_traces = []
        for _ in range(3):
            trace = Mock()
            trace.quality_metrics = None
            mock_traces.append(trace)
        
        # Test
        result = calculator._calculate_overall_quality_metrics(mock_traces)
        
        # Verify
        assert result == {}
    
    def test_calculate_quality_distribution(self):
        """Test quality distribution calculation."""
        # Setup
        calculator = MetricsCalculator()
        
        # Create mock traces with different quality scores
        mock_traces = []
        scores = [0.95, 0.85, 0.75, 0.55, 0.35, 0.25]  # excellent, good, acceptable, needs_improvement, poor, poor
        
        for score in scores:
            trace = Mock()
            trace.quality_metrics = {"composite_quality_score": score}
            mock_traces.append(trace)
        
        # Test
        result = calculator._calculate_quality_distribution(mock_traces)
        
        # Verify
        assert result["excellent"] == 1  # 0.95
        assert result["good"] == 1  # 0.85
        assert result["acceptable"] == 1  # 0.75
        assert result["needs_improvement"] == 1  # 0.55
        assert result["poor"] == 2  # 0.35, 0.25
    
    def test_generate_performance_recommendations(self):
        """Test performance recommendations generation."""
        # Setup
        calculator = MetricsCalculator()
        
        # Test case 1: Low success rate
        recommendations = calculator._generate_performance_recommendations(
            success_rate=0.6,  # < 0.7
            avg_quality=0.8,
            avg_decision_quality=0.7
        )
        assert "improving success rate" in recommendations[0]
        
        # Test case 2: Low quality
        recommendations = calculator._generate_performance_recommendations(
            success_rate=0.8,
            avg_quality=0.65,  # < 0.7
            avg_decision_quality=0.7
        )
        assert "rigorous quality checks" in recommendations[0]
        
        # Test case 3: Low decision quality
        recommendations = calculator._generate_performance_recommendations(
            success_rate=0.8,
            avg_quality=0.8,
            avg_decision_quality=0.55  # < 0.6
        )
        assert "decision-making process" in recommendations[0]
        
        # Test case 4: High performance
        recommendations = calculator._generate_performance_recommendations(
            success_rate=0.95,  # > 0.9
            avg_quality=0.9,  # > 0.85
            avg_decision_quality=0.8
        )
        assert "optimizing for efficiency" in recommendations[0]
    
    def test_generate_behavior_insights(self):
        """Test behavior insights generation."""
        # Setup
        calculator = MetricsCalculator()
        
        # Create sample data
        sequence_analysis = {
            "agent1": {
                "total_operations": 10,
                "success_rate": 0.8,
                "common_sequences": {"op1 -> op2": 5},
                "average_duration": 1000
            }
        }
        
        error_patterns = {"timeout": 5, "validation_error": 2}
        
        behavioral_consistency = {
            "agent1": {
                "consistency_score": 0.25,
                "unique_operations": 3,
                "total_operations": 10,
                "consistency_level": "high"
            }
        }
        
        # Test
        insights = calculator._generate_behavior_insights(
            sequence_analysis, error_patterns, behavioral_consistency
        )
        
        # Verify insights are generated
        assert len(insights) > 0
        assert any("Most common error" in insight for insight in insights)
        assert any("high behavioral consistency" in insight for insight in insights)
    
    def test_cache_operations(self):
        """Test cache get and set operations."""
        # Setup
        calculator = MetricsCalculator()
        
        # Test setting cache
        test_key = "test_key"
        test_value = {"data": "value"}
        
        calculator._cache_result(test_key, test_value)
        
        # Test getting cached value (should be there)
        cached = calculator._get_cached(test_key)
        assert cached == test_value
        
        # Test getting non-existent key
        non_existent = calculator._get_cached("non_existent")
        assert non_existent is None
    
    def test_cache_expiration(self):
        """Test cache expiration."""
        # Setup
        calculator = MetricsCalculator()
        
        # Set a cache entry with old timestamp
        test_key = "test_key"
        old_time = datetime.now(timezone.utc) - timedelta(minutes=10)  # Older than TTL
        calculator._cache[test_key] = (old_time, {"data": "old"})
        
        # Should not return expired cache
        cached = calculator._get_cached(test_key)
        assert cached is None
        assert test_key not in calculator._cache  # Should be cleaned up
    
    @patch('app.observability.metrics_calculator.logger')
    def test_logging_in_performance_calculation(self, mock_logger):
        """Test that performance calculation logs appropriately."""
        # Setup
        calculator = MetricsCalculator()
        
        # Create mock traces
        mock_traces = []
        for i in range(2):
            trace = Mock()
            trace.success = True
            trace.duration_ms = 1000
            trace.quality_metrics = {"composite_quality_score": 0.8}
            trace.decisions = []
            trace.agent_observations = []
            trace.agent_context = {"agent_id": "test_agent"}
            trace.end_time = datetime.now(timezone.utc) - timedelta(hours=1)
            mock_traces.append(trace)
        
        with patch.object(calculator, '_get_traces_for_period') as mock_get_traces:
            mock_get_traces.return_value = mock_traces
            
            # Test
            calculator.calculate_agent_performance()
            
            # Verify logging was called
            mock_logger.info.assert_called_once()


# Additional test for edge cases
class TestMetricsCalculatorEdgeCases:
    """Edge case tests for MetricsCalculator."""
    
    def test_empty_quality_metrics(self):
        """Test handling of traces with empty quality metrics."""
        # Setup
        calculator = MetricsCalculator()
        
        # Create mock traces without composite_quality_score
        mock_traces = []
        for i in range(3):
            trace = Mock()
            trace.success = True
            trace.duration_ms = 1000
            trace.quality_metrics = {}  # Empty or missing composite_quality_score
            trace.agent_context = {"agent_id": "test_agent"}
            trace.end_time = datetime.now(timezone.utc) - timedelta(hours=1)
            trace.error = None
            trace.decisions = []  # Ensure this is a list, not a Mock
            trace.agent_observations = []  # Ensure this is a list
            mock_traces.append(trace)
        
        with patch.object(calculator, '_get_traces_for_period') as mock_get_traces:
            mock_get_traces.return_value = mock_traces
            
            # Test
            result = calculator.calculate_agent_performance()
            
            # Verify
            assert result["average_quality_score"] == 0.0

    def test_malformed_decision_data(self):
        """Test handling of malformed decision data."""
        # Setup
        calculator = MetricsCalculator()
        
        # Create mock traces with malformed decisions
        mock_traces = []
        for i in range(2):
            trace = Mock()
            trace.success = True
            trace.duration_ms = 1000
            trace.quality_metrics = {"composite_quality_score": 0.8}
            # Set decisions as actual list, not Mock
            trace.decisions = [
                "not_a_dict",  # Malformed - will be filtered out (not a dict)
                {"type": "valid", "quality": "not_a_dict"},  # Malformed quality (dict, but quality is string)
                {"type": "valid", "quality": {"overall_score": 0.8}}  # Valid
            ]
            trace.agent_context = {"agent_id": "test_agent"}
            trace.end_time = datetime.now(timezone.utc) - timedelta(hours=1)
            trace.agent_observations = []
            mock_traces.append(trace)
        
        with patch.object(calculator, '_get_traces_for_period') as mock_get_traces:
            mock_get_traces.return_value = mock_traces
            
            # Test - should not crash
            result = calculator.calculate_decision_analytics()
            
            # Verify
            # The code filters out non-dict decisions with: if isinstance(decision, dict)
            # So: "not_a_dict" (string) is filtered out
            # Each trace has 2 dict decisions, 2 traces = 4 total decisions
            assert result["total_decisions"] == 4
            # For quality calculation: only decisions with dict quality and overall_score are counted
            # Only the last decision in each trace has valid quality structure
            # So 2 valid decisions with score 0.8 each = average 0.8
            assert result["average_decision_quality"] == 0.8

    def test_calculate_agent_performance_cache(self):
        """Test that performance calculation uses cache."""
        # Setup
        calculator = MetricsCalculator()
        
        # Put something in cache
        cache_key = "agent_performance_None_24"
        cached_value = {
            "agent_name": "all_agents",
            "total_traces": 100,
            "cached": True
        }
        calculator._cache_result(cache_key, cached_value)
        
        # Test - should return cached value
        result = calculator.calculate_agent_performance()
        
        # Verify
        assert result["cached"] == True
        assert result["total_traces"] == 100
