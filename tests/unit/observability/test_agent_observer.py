import pytest
from datetime import datetime, timezone
from unittest.mock import Mock, patch, MagicMock

from app.observability.agent_observer import AgentObserver, ObservationType


class TestAgentObserver:
    """Unit tests for AgentObserver."""
    
    def test_initialization(self, agent_observer):
        """Test AgentObserver initializes correctly."""
        assert agent_observer.tracer is not None
        assert agent_observer.session_id is not None
        assert len(agent_observer.session_id) == 8
        assert agent_observer._buffer_size == 100
        
    def test_record_thought_process(self, agent_observer):
        """Test recording thought process."""
        # Mock the tracer
        mock_tracer = Mock()
        agent_observer.tracer = mock_tracer
        
        # Record a thought process
        result = agent_observer.record_thought_process(
            agent_name="test_agent",
            input_data="test input",
            thought_chain=["step1", "step2", "step3"],
            final_thought="decision made",
            metadata={"test": "metadata"}
        )
        
        # Verify result structure
        assert result["agent_name"] == "test_agent"
        assert result["type"] == ObservationType.THOUGHT_PROCESS.value
        assert result["thought_chain"] == ["step1", "step2", "step3"]
        assert result["final_thought"] == "decision made"
        assert result["chain_length"] == 3
        assert result["has_conclusion"] == True
        
        # Verify tracer was called
        mock_tracer.record_agent_observation.assert_called_once()
        
    def test_record_decision_rationale(self, agent_observer):
        """Test recording decision rationale."""
        mock_tracer = Mock()
        agent_observer.tracer = mock_tracer
        
        options = [
            {"name": "option1", "score": 0.8},
            {"name": "option2", "score": 0.6},
            {"name": "option3", "score": 0.4}
        ]
        
        result = agent_observer.record_decision_rationale(
            decision_id="test_decision_123",
            agent_name="test_agent",
            options_considered=options,
            chosen_option={"name": "option1", "score": 0.8},
            rationale="Option 1 has the highest score",
            confidence=0.8,
            tradeoffs=["speed vs accuracy", "cost vs quality"]
        )
        
        assert result["decision_id"] == "test_decision_123"
        assert result["options_count"] == 3
        assert result["confidence_score"] == 0.8
        assert result["confidence_level"] == "high"
        assert len(result["tradeoffs"]) == 2
        assert result["analysis_quality"] in ["high", "medium", "low"]
        
        mock_tracer.record_decision.assert_called_once()

    def test_record_self_evaluation(self, agent_observer):
        """Test recording self-evaluation."""
        # Get the mock tracer from the fixture
        mock_tracer = agent_observer.tracer
        
        # Reset call count to ensure we're starting fresh
        mock_tracer.record_agent_observation .reset_mock()
        
        result = agent_observer.record_self_evaluation(
            agent_name="test_agent",
            task_id="task_123",
            evaluation_criteria={
                "accuracy": 0.4,
                "speed": 0.3,
                "clarity": 0.3
            },
            self_scores={
                "accuracy": 0.8,
                "speed": 0.7,
                "clarity": 0.9
            },
            justification="Performed well overall",
            improvements_suggested=["Improve speed", "Add more examples"]
        )
        
        assert result["agent_name"] == "test_agent"
        assert result["task_id"] == "task_123"
        assert 0 <= result["overall_score"] <= 1
        assert result["score_category"] in ["excellent", "good", "acceptable", "needs_improvement", "poor"]
        assert len(result["improvements_suggested"]) == 2
        assert 0 <= result["self_criticality"] <= 1
        
        # Check that the tracer's record_agent_observation  was called
        # Use assert_called() instead of assert_called_once() for better debugging
        mock_tracer.record_agent_observation .assert_called()
        
        # Get the actual call to see what was called
        call_args = mock_tracer.record_agent_observation .call_args
        assert call_args is not None
        assert call_args[0][0] == "agent_self_evaluation"

    def test_get_agent_insights(self, agent_observer):
        """Test generating agent insights."""
        # First, add some observations to the buffer
        for i in range(5):
            agent_observer.record_thought_process(
                agent_name="test_agent",
                input_data=f"test input {i}",
                thought_chain=[f"thought_{i}"],
                final_thought=f"final_{i}"
            )
        
        insights = agent_observer.get_agent_insights(
            agent_name="test_agent",
            time_window_hours=24
        )
        
        assert insights["agent_name"] == "test_agent"
        assert insights["observation_count"] == 5
        assert "performance_trend" in insights
        assert "confidence_distribution" in insights
        assert "recommendations" in insights
        
    def test_classify_confidence(self, agent_observer):
        """Test confidence classification."""
        assert agent_observer._classify_confidence(0.95) == "very_high"
        assert agent_observer._classify_confidence(0.75) == "high"
        assert agent_observer._classify_confidence(0.55) == "medium"
        assert agent_observer._classify_confidence(0.35) == "low"
        assert agent_observer._classify_confidence(0.15) == "very_low"
        
    def test_categorize_score(self, agent_observer):
        """Test score categorization."""
        assert agent_observer._categorize_score(0.95) == "excellent"
        assert agent_observer._categorize_score(0.85) == "good"
        assert agent_observer._categorize_score(0.65) == "acceptable"
        assert agent_observer._categorize_score(0.45) == "needs_improvement"
        assert agent_observer._categorize_score(0.25) == "poor"
        
    @patch('app.repositories.agent_insight_repository.AgentInsightRepository')
    def test_generate_insights_from_observations(self, mock_repo_class, agent_observer):
        """Test insight generation."""
        # Mock the repository
        mock_repo = Mock()
        mock_repo_class.return_value = mock_repo
        agent_observer._insight_repository = mock_repo
        
        # Mock get_agent_insights to return test data
        agent_observer.get_agent_insights = Mock(return_value={
            "agent_name": "test_agent",
            "performance_trend": "declining",
            "average_quality_score": 0.6,
            "confidence_distribution": {"average": 0.5},
            "behavior_patterns": [
                {"type": "error_pattern", "significance": "high", "occurrence_count": 3}
            ],
            "recommendations": ["Improve quality", "Add more tests"],
            "observation_count": 10,
            "decision_count": 5
        })
        
        insights = agent_observer.generate_insights_from_observations(
            agent_name="test_agent",
            time_window_hours=24
        )
        
        assert isinstance(insights, list)
        # Should generate at least one insight because quality is low
        assert len(insights) > 0
        
        # Verify insights have proper structure
        for insight in insights:
            assert "insight_id" in insight
            assert "agent_name" in insight
            assert "insight_type" in insight
            assert "insight_data" in insight
            assert "confidence_score" in insight
            
        # Verify repository was called
        assert mock_repo.save.call_count >= len(insights)
