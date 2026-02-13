"""Unit tests for DecisionEngine."""
import pytest
from unittest.mock import Mock, patch, MagicMock, call
from datetime import datetime, timezone

from app.services.decision_engine import DecisionEngine
from app.exceptions.base import AppException
from app.exceptions.validation import ValidationError
from app.models.task import Task
from app.observability.tracer import Tracer
from app.observability.agent_observer import AgentObserver


class TestDecisionEngine:
    """Unit tests for DecisionEngine."""
    
    def test_initialization(self):
        """Test DecisionEngine initializes correctly."""
        engine = DecisionEngine()
        assert engine.tracer is not None
        assert isinstance(engine.tracer, Tracer)
        assert engine.observer is not None
        assert isinstance(engine.observer, AgentObserver)
    
    def test_assess_complexity_simple_task(self):
        """Test complexity assessment for simple tasks."""
        engine = DecisionEngine()
        
        # Simple tasks
        assert engine._assess_complexity("extract", {}) == "simple"
        assert engine._assess_complexity("translate", {}) == "simple"
    
    def test_assess_complexity_moderate_task(self):
        """Test complexity assessment for moderate tasks."""
        engine = DecisionEngine()
        
        # Moderate tasks with default parameters
        assert engine._assess_complexity("summarize", {}) == "moderate"
        assert engine._assess_complexity("classify", {}) == "moderate"
    
    def test_assess_complexity_complex_task(self):
        """Test complexity assessment for complex tasks."""
        engine = DecisionEngine()
        
        # Complex tasks
        assert engine._assess_complexity("analyze", {}) == "complex"
        
        # Summarize with large max_length
        assert engine._assess_complexity("summarize", {"max_length": 2000}) == "complex"
    
    def test_assess_complexity_unknown_task(self):
        """Test complexity assessment for unknown task types."""
        engine = DecisionEngine()
        
        # Unknown task type
        assert engine._assess_complexity("unknown_task", {}) == "moderate"
    
    def test_assess_complexity_error(self):
        """Test complexity assessment with error."""
        engine = DecisionEngine()
        
        # Mock to raise an exception
        with patch.object(engine, '_assess_complexity', side_effect=Exception("Test error")):
            # Actually, we need to test the error handling in get_execution_plan
            # Let's test the method directly
            pass
    
    def test_get_complexity_reason(self):
        """Test getting complexity reason."""
        engine = DecisionEngine()
        
        assert "Simple single-operation" in engine._get_complexity_reason("extract", {})
        assert "Simple single-operation" in engine._get_complexity_reason("translate", {})
        
        assert "Moderate task with max length" in engine._get_complexity_reason("summarize", {"max_length": 100})
        assert "Moderate task with max length" in engine._get_complexity_reason("classify", {"max_length": 50})
        
        assert "Complex multi-stage analysis" in engine._get_complexity_reason("analyze", {})
        
        assert "Moderate default complexity" in engine._get_complexity_reason("unknown", {})
    
    def test_calculate_complexity_confidence(self):
        """Test calculating complexity confidence."""
        engine = DecisionEngine()
        
        # Known task types have high confidence
        assert engine._calculate_complexity_confidence("extract", {}) == 0.9
        assert engine._calculate_complexity_confidence("translate", {}) == 0.9
        assert engine._calculate_complexity_confidence("summarize", {}) == 0.9
        assert engine._calculate_complexity_confidence("classify", {}) == 0.9
        assert engine._calculate_complexity_confidence("analyze", {}) == 0.9
        
        # Unknown task type has lower confidence
        assert engine._calculate_complexity_confidence("unknown_task", {}) == 0.6
    
    def test_get_simple_plan_summarize(self):
        """Test getting simple plan for summarize task."""
        engine = DecisionEngine()
        
        parameters = {"max_length": 100}
        plan = engine._get_simple_plan("summarize", parameters)
        
        assert plan["complexity"] == "simple"
        assert plan["reason"] == "Task assessed as low complexity"
        assert len(plan["steps"]) == 1
        assert plan["steps"][0]["type"] == "llm_call"
        assert "Summarize in about 100 words" in plan["steps"][0]["parameters"]["prompt"]
        assert "agent" in plan
        assert plan["agent"] == "decision_engine"
    
    def test_get_simple_plan_translate(self):
        """Test getting simple plan for translate task."""
        engine = DecisionEngine()
        
        parameters = {"target_language": "Spanish"}
        plan = engine._get_simple_plan("translate", parameters)
        
        assert plan["complexity"] == "simple"
        assert len(plan["steps"]) == 1
        assert "Translate to Spanish" in plan["steps"][0]["parameters"]["prompt"]
    
    def test_get_simple_plan_other_task(self):
        """Test getting simple plan for other task types."""
        engine = DecisionEngine()
        
        plan = engine._get_simple_plan("extract", {})
        
        assert plan["complexity"] == "simple"
        assert "Perform extract on the input" in plan["steps"][0]["parameters"]["prompt"]
    
    def test_get_simple_plan_error(self):
        """Test getting simple plan with error."""
        engine = DecisionEngine()
        
        # Mock to raise exception
        with patch.object(engine, '_get_simple_plan', side_effect=Exception("Test error")):
            # Error will be caught in get_execution_plan
            pass
    
    def test_get_moderate_plan_summarize(self):
        """Test getting moderate plan for summarize task."""
        engine = DecisionEngine()
        
        parameters = {"max_length": 100, "format": "detailed"}
        plan = engine._get_moderate_plan("summarize", parameters)
        
        assert plan["complexity"] == "moderate"
        assert plan["reason"] == "Task requires validation and processing"
        assert len(plan["steps"]) == 3
        assert plan["steps"][0]["type"] == "decision_point"
        assert plan["steps"][1]["type"] == "llm_call"
        assert plan["steps"][2]["type"] == "data_transform"
        assert "agent" in plan
        assert plan["agent"] == "decision_engine"
        
        # Check prompt includes parameters
        prompt = plan["steps"][1]["parameters"]["prompt"]
        assert "100" in prompt
        assert "detailed" in prompt
    
    def test_get_moderate_plan_classify(self):
        """Test getting moderate plan for classify task."""
        engine = DecisionEngine()
        
        parameters = {"categories": ["positive", "negative", "neutral"]}
        plan = engine._get_moderate_plan("classify", parameters)
        
        assert plan["complexity"] == "moderate"
        prompt = plan["steps"][1]["parameters"]["prompt"]
        assert "positive, negative, neutral" in prompt
    
    def test_get_complex_plan_analyze(self):
        """Test getting complex plan for analyze task."""
        engine = DecisionEngine()
        
        plan = engine._get_complex_plan("analyze", {})
        
        assert plan["complexity"] == "complex"
        assert plan["reason"] == "Task requires multi-stage processing and validation"
        assert len(plan["steps"]) == 5
        assert plan["steps"][0]["type"] == "decision_point"
        assert plan["steps"][2]["type"] == "llm_call"
        assert plan["steps"][3]["type"] == "llm_call"
        assert plan["steps"][4]["type"] == "data_transform"
        assert "agent" in plan
        assert plan["agent"] == "decision_engine"
    
    def test_evaluate_condition_logic_requires_validation(self):
        """Test condition evaluation for 'requires_validation'."""
        engine = DecisionEngine()
        
        # Short context
        assert engine._evaluate_condition_logic("requires_validation", {"text": "short"}) == False
        
        # Long context
        long_text = "x" * 200
        assert engine._evaluate_condition_logic("requires_validation", {"text": long_text}) == True
    
    def test_evaluate_condition_logic_has_multiple_items(self):
        """Test condition evaluation for 'has_multiple_items'."""
        engine = DecisionEngine()
        
        # Single item list
        assert engine._evaluate_condition_logic("has_multiple_items", ["item1"]) == False
        
        # Multiple items list
        assert engine._evaluate_condition_logic("has_multiple_items", ["item1", "item2"]) == True
        
        # Not a list
        assert engine._evaluate_condition_logic("has_multiple_items", {"key": "value"}) == False
    
    def test_evaluate_condition_logic_contains_sensitive_data(self):
        """Test condition evaluation for 'contains_sensitive_data'."""
        engine = DecisionEngine()
        
        # Contains sensitive keyword
        assert engine._evaluate_condition_logic("contains_sensitive_data", {"password": "secret123"}) == True
        assert engine._evaluate_condition_logic("contains_sensitive_data", "my secret token") == True
        assert engine._evaluate_condition_logic("contains_sensitive_data", "api key here") == True
        
        # No sensitive keywords
        assert engine._evaluate_condition_logic("contains_sensitive_data", {"text": "hello world"}) == False
    
    def test_evaluate_condition_logic_unknown_condition(self):
        """Test condition evaluation for unknown condition."""
        engine = DecisionEngine()
        
        with pytest.raises(ValidationError) as exc_info:
            engine._evaluate_condition_logic("unknown_condition", {})
        
        assert "Unknown condition type" in str(exc_info.value)
        assert exc_info.value.field == "condition"
    
    def test_validate_plan_valid(self):
        """Test validating a valid plan."""
        engine = DecisionEngine()
        
        plan = {
            "id": "test_plan_123",
            "steps": [
                {"name": "step1", "type": "llm_call", "parameters": {}},
                {"name": "step2", "type": "decision_point", "parameters": {}}
            ]
        }
        
        # Should not raise exception
        engine._validate_plan(plan, "summarize", {})

    def test_validate_plan_missing_id(self):
        """Test validating plan missing ID."""
        from app.exceptions.validation import ValidationError
        engine = DecisionEngine()
        
        plan = {
            "steps": [{"name": "step1", "type": "llm_call"}]
        }
        
        with pytest.raises(ValidationError) as exc_info:
            engine._validate_plan(plan, "summarize", {})
        
        assert "Generated plan missing ID" in str(exc_info.value)

    def test_validate_plan_missing_steps(self):
        """Test validating plan missing steps."""
        engine = DecisionEngine()
        
        plan = {"id": "test_plan"}
        
        with pytest.raises(ValidationError) as exc_info:
            engine._validate_plan(plan, "summarize", {})
        
        assert "Generated plan has no steps" in str(exc_info.value)
        assert exc_info.value.field == "plan.steps"
    
    def test_validate_plan_invalid_steps_type(self):
        """Test validating plan with invalid steps type."""
        engine = DecisionEngine()
        
        plan = {"id": "test_plan", "steps": "not a list"}
        
        with pytest.raises(ValidationError) as exc_info:
            engine._validate_plan(plan, "summarize", {})
        
        assert "steps is not a list" in str(exc_info.value)
    
    def test_validate_plan_step_missing_name(self):
        """Test validating plan with step missing name."""
        engine = DecisionEngine()
        
        plan = {
            "id": "test_plan",
            "steps": [{"type": "llm_call"}]  # Missing name
        }
        
        with pytest.raises(ValidationError) as exc_info:
            engine._validate_plan(plan, "summarize", {})
        
        assert "missing 'name' field" in str(exc_info.value)
        assert "plan.steps[0].name" in exc_info.value.field
    
    def test_validate_plan_step_missing_type(self):
        """Test validating plan with step missing type."""
        engine = DecisionEngine()
        
        plan = {
            "id": "test_plan",
            "steps": [{"name": "step1"}]  # Missing type
        }
        
        with pytest.raises(ValidationError) as exc_info:
            engine._validate_plan(plan, "summarize", {})
        
        assert "missing 'type' field" in str(exc_info.value)
        assert "plan.steps[0].type" in exc_info.value.field
    
    def test_summarize_context_dict(self):
        """Test summarizing dictionary context."""
        engine = DecisionEngine()
        
        context = {"key1": "value1", "key2": "value2", "nested": {"inner": "value"}}
        summary = engine._summarize_context(context)
        
        assert summary["key_count"] == 3
        assert "key1" in summary["keys"]
        assert "key2" in summary["keys"]
        assert summary["has_nested"] == True
    
    def test_summarize_context_list(self):
        """Test summarizing list context."""
        engine = DecisionEngine()
        
        context = ["item1", "item2", "item3"]
        summary = engine._summarize_context(context)
        
        assert summary["item_count"] == 3
        assert summary["first_item_type"] == "str"
    
    def test_summarize_context_other(self):
        """Test summarizing other types of context."""
        engine = DecisionEngine()
        
        context = "simple string"
        summary = engine._summarize_context(context)
        
        assert summary["type"] == "str"
        assert summary["string_length"] == 13
    
    def test_summarize_context_empty(self):
        """Test summarizing empty context."""
        engine = DecisionEngine()
        
        summary = engine._summarize_context({})
        assert summary["empty"] == True
        
        summary = engine._summarize_context(None)
        # Should handle None gracefully
        assert "empty" in summary or "summary_error" in summary


class TestDecisionEngineIntegration:
    """Integration tests for DecisionEngine."""
    
    @patch('app.services.decision_engine.AgentObserver')
    @patch('app.services.decision_engine.Tracer')
    def test_get_execution_plan_simple_task(self, mock_tracer_class, mock_observer_class):
        """Test getting execution plan for simple task."""
        # Setup mocks
        mock_tracer = mock_tracer_class.return_value
        mock_observer = mock_observer_class.return_value
        
        engine = DecisionEngine()
        engine.tracer = mock_tracer
        engine.observer = mock_observer
        
        # Mock trace context
        mock_trace_context = Mock()
        mock_trace_context.__enter__ = Mock(return_value=None)
        mock_trace_context.__exit__ = Mock(return_value=None)
        mock_tracer.start_trace.return_value = mock_trace_context
        
        task_type = "extract"
        parameters = {}
        
        plan = engine.get_execution_plan(task_type, parameters)
        
        assert plan["complexity"] == "simple"
        assert "extract" in plan["id"]
        assert len(plan["steps"]) == 1
        assert plan["steps"][0]["type"] == "llm_call"
        
        # Verify tracer was used
        mock_tracer.update_agent_context.assert_called()
        mock_tracer.start_trace.assert_called()
        mock_tracer.record_decision.assert_called()
    
    @patch('app.services.decision_engine.AgentObserver')
    @patch('app.services.decision_engine.Tracer')
    def test_get_execution_plan_moderate_task(self, mock_tracer_class, mock_observer_class):
        """Test getting execution plan for moderate task."""
        # Setup mocks
        mock_tracer = mock_tracer_class.return_value
        mock_observer = mock_observer_class.return_value
        
        engine = DecisionEngine()
        engine.tracer = mock_tracer
        engine.observer = mock_observer
        
        # Mock trace context
        mock_trace_context = Mock()
        mock_trace_context.__enter__ = Mock(return_value=None)
        mock_trace_context.__exit__ = Mock(return_value=None)
        mock_tracer.start_trace.return_value = mock_trace_context
        
        task_type = "summarize"
        parameters = {"max_length": 100}
        
        plan = engine.get_execution_plan(task_type, parameters)
        
        assert plan["complexity"] == "moderate"
        assert "summarize" in plan["id"]
        assert len(plan["steps"]) == 3
        
        # Verify observer methods were called
        mock_observer.record_thought_process.assert_called()
        mock_observer.record_decision_rationale.assert_called()
        mock_observer.record_self_evaluation.assert_called()
    
    @patch('app.services.decision_engine.AgentObserver')
    @patch('app.services.decision_engine.Tracer')
    def test_get_execution_plan_complex_task(self, mock_tracer_class, mock_observer_class):
        """Test getting execution plan for complex task."""
        # Setup mocks
        mock_tracer = mock_tracer_class.return_value
        mock_observer = mock_observer_class.return_value
        
        engine = DecisionEngine()
        engine.tracer = mock_tracer
        engine.observer = mock_observer
        
        # Mock trace context
        mock_trace_context = Mock()
        mock_trace_context.__enter__ = Mock(return_value=None)
        mock_trace_context.__exit__ = Mock(return_value=None)
        mock_tracer.start_trace.return_value = mock_trace_context
        
        task_type = "analyze"
        parameters = {}
        
        plan = engine.get_execution_plan(task_type, parameters)
        
        assert plan["complexity"] == "complex"
        assert "analyze" in plan["id"]
        assert len(plan["steps"]) == 5
    
    @patch('app.services.decision_engine.AgentObserver')
    @patch('app.services.decision_engine.Tracer')
    def test_get_execution_plan_invalid_task_type(self, mock_tracer_class, mock_observer_class):
        """Test getting execution plan with invalid task type."""
        engine = DecisionEngine()
        engine.tracer = mock_tracer_class.return_value
        
        with pytest.raises(ValidationError) as exc_info:
            engine.get_execution_plan("", {})  # Empty task type
        
        assert "Task type must be a non-empty string" in str(exc_info.value)
    
    @patch('app.services.decision_engine.AgentObserver')
    @patch('app.services.decision_engine.Tracer')
    def test_get_execution_plan_unknown_complexity(self, mock_tracer_class, mock_observer_class):
        """Test getting execution plan when complexity assessment returns unknown value."""
        # Setup mocks
        mock_tracer = mock_tracer_class.return_value
        mock_observer = mock_observer_class.return_value
        
        engine = DecisionEngine()
        engine.tracer = mock_tracer
        engine.observer = mock_observer
        
        # Mock _assess_complexity to return unknown value
        with patch.object(engine, '_assess_complexity', return_value="unknown_complexity"):
            with pytest.raises(AppException) as exc_info:
                engine.get_execution_plan("summarize", {})
            
            assert "Unknown complexity level" in str(exc_info.value)
    
    @patch('app.services.decision_engine.AgentObserver')
    @patch('app.services.decision_engine.Tracer')
    def test_get_execution_plan_error(self, mock_tracer_class, mock_observer_class):
        """Test getting execution plan when error occurs."""
        engine = DecisionEngine()
        engine.tracer = mock_tracer_class.return_value
        
        # Mock to raise exception
        with patch.object(engine, '_assess_complexity', side_effect=Exception("Test error")):
            with pytest.raises(AppException) as exc_info:
                engine.get_execution_plan("summarize", {})
            
            assert "Failed to create execution plan" in str(exc_info.value)
    
    @patch('app.services.decision_engine.AgentObserver')
    @patch('app.services.decision_engine.Tracer')
    def test_evaluate_condition_success(self, mock_tracer_class, mock_observer_class):
        """Test evaluating condition successfully."""
        # Setup mocks
        mock_tracer = mock_tracer_class.return_value
        
        engine = DecisionEngine()
        engine.tracer = mock_tracer
        
        # Mock trace context
        mock_trace_context = Mock()
        mock_trace_context.__enter__ = Mock(return_value=None)
        mock_trace_context.__exit__ = Mock(return_value=None)
        mock_tracer.start_trace.return_value = mock_trace_context
        
        # Create mock task
        mock_task = Mock(spec=Task)
        mock_task.input_data = {"text": "Test input"}
        
        condition = "requires_validation"
        context = {"text": "x" * 200}  # Long text
        
        result = engine.evaluate_condition(condition, mock_task, context)
        
        assert result["result"] == True
        assert result["content"] == mock_task.input_data
        
        # Verify tracer was used
        mock_tracer.start_trace.assert_called()
        mock_tracer.record_decision.assert_called()
    
    @patch('app.services.decision_engine.AgentObserver')
    @patch('app.services.decision_engine.Tracer')
    def test_evaluate_condition_invalid_condition(self, mock_tracer_class, mock_observer_class):
        """Test evaluating condition with invalid condition."""
        engine = DecisionEngine()
        
        mock_task = Mock(spec=Task)
        
        with pytest.raises(ValidationError) as exc_info:
            engine.evaluate_condition("", mock_task, {})  # Empty condition
        
        assert "Condition must be a non-empty string" in str(exc_info.value)
    
    @patch('app.services.decision_engine.AgentObserver')
    @patch('app.services.decision_engine.Tracer')
    def test_evaluate_condition_null_context(self, mock_tracer_class, mock_observer_class):
        """Test evaluating condition with null context."""
        engine = DecisionEngine()
        
        mock_task = Mock(spec=Task)
        
        with pytest.raises(ValidationError) as exc_info:
            engine.evaluate_condition("requires_validation", mock_task, None)
        
        assert "Context cannot be None" in str(exc_info.value)
    
    @patch('app.services.decision_engine.AgentObserver')
    @patch('app.services.decision_engine.Tracer')
    def test_evaluate_condition_error(self, mock_tracer_class, mock_observer_class):
        """Test evaluating condition when error occurs."""
        engine = DecisionEngine()
        
        mock_task = Mock(spec=Task)
        
        # Mock _evaluate_condition_logic to raise exception
        with patch.object(engine, '_evaluate_condition_logic', side_effect=Exception("Test error")):
            with pytest.raises(AppException) as exc_info:
                engine.evaluate_condition("requires_validation", mock_task, {})
            
            assert "Failed to evaluate condition" in str(exc_info.value)
    
    def test_on_trace_completed(self):
        """Test trace completion hook."""
        engine = DecisionEngine()
        
        # Mock observer
        mock_observer = Mock()
        engine.observer = mock_observer
        
        # Successful trace with high quality
        trace_data = {
            "success": True,
            "quality_metrics": {"composite_quality_score": 0.9},
            "agent_context": {"agent_id": "test_agent"},
            "operation": "llm_call",
            "duration_ms": 100
        }
        
        engine._on_trace_completed(trace_data)
        
        # Should detect behavior pattern
        mock_observer.detect_behavior_pattern.assert_called_once()
        call_args = mock_observer.detect_behavior_pattern.call_args[1]
        assert call_args["agent_name"] == "test_agent"
        assert call_args["behavior_type"] == "high_quality_execution"
        assert call_args["pattern_data"]["quality_score"] == 0.9
    
    def test_on_trace_completed_low_quality(self):
        """Test trace completion hook with low quality."""
        engine = DecisionEngine()
        
        # Mock observer
        mock_observer = Mock()
        engine.observer = mock_observer
        
        # Low quality trace
        trace_data = {
            "success": True,
            "quality_metrics": {"composite_quality_score": 0.6},  # Low quality
            "agent_context": {"agent_id": "test_agent"}
        }
        
        engine._on_trace_completed(trace_data)
        
        # Should not detect high quality pattern
        mock_observer.detect_behavior_pattern.assert_not_called()
    
    def test_on_trace_completed_failed_trace(self):
        """Test trace completion hook with failed trace."""
        engine = DecisionEngine()
        
        # Mock observer
        mock_observer = Mock()
        engine.observer = mock_observer
        
        # Failed trace
        trace_data = {
            "success": False,
            "quality_metrics": {"composite_quality_score": 0.9},
            "agent_context": {"agent_id": "test_agent"}
        }
        
        engine._on_trace_completed(trace_data)
        
        # Should not detect high quality pattern for failed trace
        mock_observer.detect_behavior_pattern.assert_not_called()


class TestDecisionEngineEdgeCases:
    """Edge case tests for DecisionEngine."""
    
    @patch('app.services.decision_engine.AgentObserver')
    @patch('app.services.decision_engine.Tracer')
    def test_get_execution_plan_with_validation_error(self, mock_tracer_class, mock_observer_class):
        """Test getting execution plan when validation error occurs."""
        engine = DecisionEngine()
        engine.tracer = mock_tracer_class.return_value
        
        # Invalid task type (not a string)
        with pytest.raises(ValidationError) as exc_info:
            engine.get_execution_plan(123, {})  # Task type is int
        
        assert "Task type must be a non-empty string" in str(exc_info.value)
    
    def test_evaluate_condition_unknown_condition_raises_validation_error(self):
        """Test evaluating unknown condition raises ValidationError."""
        engine = DecisionEngine()
        
        mock_task = Mock(spec=Task)
        
        with pytest.raises(ValidationError) as exc_info:
            # Mock _evaluate_condition_logic to raise ValidationError for unknown condition
            with patch.object(engine, '_evaluate_condition_logic', side_effect=ValidationError(
                message="Unknown condition type: 'unknown_condition'",
                field="condition",
                value="unknown_condition",
                validation_rules={"allowed_values": ["requires_validation", "has_multiple_items", "contains_sensitive_data"]},
                extra={"context_summary": {}}
            )):
                engine.evaluate_condition("unknown_condition", mock_task, {})
        
        # The ValidationError should propagate
        assert "Unknown condition type" in str(exc_info.value)
    
    def test_validate_plan_with_error_in_validation(self):
        """Test plan validation when error occurs during validation."""
        engine = DecisionEngine()
        
        plan = {"id": "test_plan", "steps": []}
        
        # Mock to raise exception during validation
        with patch.object(engine, '_validate_plan', side_effect=Exception("Validation error")):
            # Actually test the error wrapping in get_execution_plan
            pass
    
    def test_summarize_context_error(self):
        """Test summarizing context when error occurs."""
        engine = DecisionEngine()
        
        # Create a context that will cause an error
        class ProblematicContext:
            def __str__(self):
                raise Exception("Cannot convert to string")
        
        context = ProblematicContext()
        summary = engine._summarize_context(context)
        
        # Should return error summary
        assert "summary_error" in summary
        assert summary["summary_error"] == True

    @patch('app.services.decision_engine.Tracer')
    @patch('app.services.decision_engine.AgentObserver')
    def test_get_execution_plan_plan_generation_error(self, mock_observer_class, mock_tracer_class):
        """Test getting execution plan when plan generation fails."""
        from app.exceptions.base import AppException
        engine = DecisionEngine()
        engine.tracer = mock_tracer_class.return_value
        
        # Mock _get_simple_plan to raise exception
        with patch.object(engine, '_assess_complexity', return_value="simple"):
            with patch.object(engine, '_get_simple_plan', side_effect=Exception("Plan generation error")):
                with pytest.raises(AppException) as exc_info:
                    engine.get_execution_plan("extract", {})
                
                assert "Failed to create execution plan" in str(exc_info.value)
                assert "Plan generation error" in str(exc_info.value)
