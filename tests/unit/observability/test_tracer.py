"""
Unit tests for Tracer class - core of agentic observability.
"""
import time
import pytest
import threading
import uuid
from datetime import datetime, timezone
from unittest.mock import Mock, patch, MagicMock
from contextlib import contextmanager

from app.observability.tracer import Tracer, TraceContext
from app.exceptions.base import AppException


class TestTraceContext:
    """Unit tests for TraceContext."""
    
    def test_initialization(self):
        """Test TraceContext initializes correctly."""
        context = TraceContext()
        assert context.stack == []
        assert context.current is None
        assert context.agent_context == {}
    
    def test_push_pop(self):
        """Test pushing and popping trace contexts."""
        context = TraceContext()
        
        # Push a context
        trace_ctx = {"trace_id": "test_123", "operation": "test"}
        context.push(trace_ctx)
        
        assert len(context.stack) == 1
        assert context.current == trace_ctx
        
        # Push another
        trace_ctx2 = {"trace_id": "test_456", "operation": "test2"}
        context.push(trace_ctx2)
        
        assert len(context.stack) == 2
        assert context.current == trace_ctx2
        
        # Pop
        popped = context.pop()
        assert popped == trace_ctx2
        assert len(context.stack) == 1
        assert context.current == trace_ctx
    
    def test_agent_context(self):
        """Test agent context management."""
        context = TraceContext()
        
        # Set agent context
        agent_ctx = {"agent_id": "test_agent", "goal": "test_goal"}
        context.agent_context = agent_ctx
        
        assert context.agent_context == agent_ctx
        
        # Update agent context - takes dict
        context.update_agent_context({"step": 1, "agent_type": "assistant"})
        assert context.agent_context["step"] == 1
        assert context.agent_context["agent_type"] == "assistant"
        assert context.agent_context["agent_id"] == "test_agent"

    def test_clear(self):
        """Test clearing the context stack."""
        context = TraceContext()
        context.push({"trace_id": "test1"})
        context.push({"trace_id": "test2"})
        
        assert len(context.stack) == 2
        context.clear()
        assert context.stack == []
        assert context.current is None


class TestTracer:
    """Unit tests for Tracer class."""
    
    def test_initialization(self):
        """Test Tracer initializes correctly."""
        tracer = Tracer()
        assert tracer.trace_repo is not None
        assert isinstance(tracer.context, TraceContext)
        assert tracer._quality_hooks == []
        assert tracer._agent_observation_hooks == []
    
    def test_start_trace_context_manager(self):
        """Test trace context manager basic functionality."""
        tracer = Tracer()
        
        # Mock repository to avoid database operations
        tracer.trace_repo = Mock()
        
        with tracer.start_trace(operation="test_operation", context={"param": "value"}) as trace_ctx:
            # Verify trace context is created
            assert "trace_id" in trace_ctx
            assert trace_ctx["operation"] == "test_operation"
            assert trace_ctx["context"] == {"param": "value"}
            assert trace_ctx["success"] == True
            assert "agent_context" in trace_ctx
            
            # Verify context stack is updated
            assert tracer.context.current == trace_ctx
        
        # Verify trace is popped after context manager
        assert tracer.context.current is None

    def test_start_trace_with_agent_context(self):
        """Test trace with agent context."""
        tracer = Tracer()
        tracer.trace_repo = Mock()
        
        # Set agent context BEFORE starting trace
        tracer.update_agent_context(
            agent_id="test_agent",
            agent_type="assistant",
            goal="complete_task",
            step=1
        )
        
        with tracer.start_trace(operation="agent_operation") as trace_ctx:
            # Verify agent context is included
            assert trace_ctx["agent_context"]["agent_id"] == "test_agent"
            assert trace_ctx["agent_context"]["agent_type"] == "assistant"
            assert trace_ctx["agent_context"]["goal"] == "complete_task"
            assert trace_ctx["agent_context"]["step"] == 1

    def test_start_trace_nested(self):
        """Test nested traces."""
        tracer = Tracer()
        tracer.trace_repo = Mock()
        
        with tracer.start_trace(operation="parent_operation") as parent_trace:
            parent_id = parent_trace["trace_id"]
            
            with tracer.start_trace(operation="child_operation") as child_trace:
                # Verify child references parent
                assert child_trace["parent_trace_id"] == parent_id
                assert tracer.context.current == child_trace
            
            # After child completes, parent should be current
            assert tracer.context.current == parent_trace
    
    def test_start_trace_with_exception(self):
        """Test trace handles exceptions properly."""
        tracer = Tracer()
        tracer.trace_repo = Mock()
        
        # Should raise AppException (wrapped from ValueError)
        with pytest.raises(AppException) as exc_info:
            with tracer.start_trace(operation="failing_operation") as trace_ctx:
                # Verify trace is active
                assert tracer.context.current == trace_ctx
                raise ValueError("Test exception")
        
        # Verify trace was marked as failed
        assert trace_ctx["success"] == False
        assert "error" in trace_ctx
        assert trace_ctx["error"]["type"] == "ValueError"
        # Verify AppException was raised
        assert "Test exception" in str(exc_info.value)

    def test_start_trace_with_app_exception(self):
        """Test trace handles AppException specially."""
        tracer = Tracer()
        tracer.trace_repo = Mock()
        
        with pytest.raises(AppException):
            with tracer.start_trace(operation="app_exception_operation") as trace_ctx:
                raise AppException(message="Test app exception", extra={"code": 500})
        
        # Verify error was recorded
        assert trace_ctx["success"] == False
        assert "error" in trace_ctx
        assert trace_ctx["error"]["type"] == "AppException"
    
    def test_record_decision(self):
        """Test recording decisions in trace."""
        tracer = Tracer()
        tracer.trace_repo = Mock()
        
        with tracer.start_trace(operation="decision_test"):
            # Record a decision
            decision_context = {
                "options": ["A", "B", "C"],
                "criteria": ["accuracy", "speed"],
                "chosen": "A",
                "reason": "Highest accuracy"
            }
            tracer.record_decision("model_selection", decision_context)
            
            current = tracer.context.current
            assert "decisions" in current
            assert len(current["decisions"]) == 1
            
            decision = current["decisions"][0]
            assert decision["type"] == "model_selection"
            assert decision["context"] == decision_context
            assert "quality" in decision
            assert "overall_score" in decision["quality"]
            assert "agent_context" in decision
            assert "timestamp" in decision
    
    def test_record_agent_observation(self):
        """Test recording agent observations."""
        tracer = Tracer()
        tracer.trace_repo = Mock()
        
        with tracer.start_trace(operation="observation_test"):
            # Record agent observation
            observation_data = {
                "thought_process": "Considering options A and B",
                "confidence": 0.8,
                "reasoning": "Option A has better historical performance"
            }
            tracer.record_agent_observation("thought_process", observation_data)
            
            current = tracer.context.current
            assert "agent_observations" in current
            assert len(current["agent_observations"]) == 1
            
            observation = current["agent_observations"][0]
            assert observation["type"] == "thought_process"
            assert observation["data"] == observation_data
            assert "agent_context" in observation
            assert "timestamp" in observation
    
    def test_record_agent_observation_with_hooks(self):
        """Test agent observation hooks."""
        tracer = Tracer()
        tracer.trace_repo = Mock()
        
        # Add a hook
        hook_called = []
        def test_hook(obs_type, data):
            hook_called.append((obs_type, data))
        
        tracer.add_agent_observation_hook(test_hook)
        
        with tracer.start_trace(operation="hook_test"):
            observation_data = {"test": "data"}
            tracer.record_agent_observation("test_type", observation_data)
            
            # Verify hook was called
            assert len(hook_called) == 1
            assert hook_called[0][0] == "test_type"
            assert hook_called[0][1] == observation_data
    
    def test_record_event(self):
        """Test recording events in trace."""
        tracer = Tracer()
        tracer.trace_repo = Mock()
        
        with tracer.start_trace(operation="event_test"):
            event_data = {"status": "started", "progress": 0.5}
            tracer.record_event("progress_update", event_data)
            
            current = tracer.context.current
            assert "events" in current
            assert len(current["events"]) == 1
            
            event = current["events"][0]
            assert event["type"] == "progress_update"
            assert event["data"] == event_data
            assert "agent_context" in event
            assert "timestamp" in event
    
    def test_record_quality_metric(self):
        """Test recording quality metrics."""
        tracer = Tracer()
        tracer.trace_repo = Mock()
        
        with tracer.start_trace(operation="quality_test"):
            tracer.record_quality_metric("accuracy", 0.95, {"source": "validation"})
            tracer.record_quality_metric("latency_ms", 150, {"percentile": "p95"})
            
            current = tracer.context.current
            assert "quality_metrics" in current
            
            metrics = current["quality_metrics"]
            assert "accuracy" in metrics
            assert "latency_ms" in metrics
            
            assert metrics["accuracy"]["value"] == 0.95
            assert metrics["accuracy"]["metadata"] == {"source": "validation"}
            assert "timestamp" in metrics["accuracy"]
    
    def test_record_error(self):
        """Test recording errors in trace."""
        tracer = Tracer()
        tracer.trace_repo = Mock()
        
        with tracer.start_trace(operation="error_test"):
            error_context = {"input": "test_input", "attempt": 3}
            tracer.record_error(
                error_type="ValidationError",
                error_message="Input validation failed",
                context=error_context
            )
            
            current = tracer.context.current
            assert current["success"] == False
            assert "error" in current
            
            error = current["error"]
            assert error["type"] == "ValidationError"
            assert error["message"] == "Input validation failed"
            assert error["context"] == error_context
            assert "timestamp" in error
            
            # Should also record quality metric
            assert "quality_metrics" in current
            assert "error_occurred" in current["quality_metrics"]
    
    def test_update_agent_context(self):
        """Test updating agent context."""
        tracer = Tracer()
        
        # Initial update
        tracer.update_agent_context(
            agent_id="agent_1",
            agent_type="classifier",
            goal="categorize_items"
        )
        
        assert tracer.context.agent_context["agent_id"] == "agent_1"
        assert tracer.context.agent_context["agent_type"] == "classifier"
        assert tracer.context.agent_context["goal"] == "categorize_items"
        
        # Partial update
        tracer.update_agent_context(step=5, confidence=0.9)
        
        assert tracer.context.agent_context["step"] == 5
        assert tracer.context.agent_context["confidence"] == 0.9
        # Previous values preserved
        assert tracer.context.agent_context["agent_id"] == "agent_1"
    
    def test_increment_agent_step(self):
        """Test incrementing agent step counter."""
        tracer = Tracer()
        
        # Set initial step
        tracer.update_agent_context(step=0)
        
        # Increment
        new_step = tracer.increment_agent_step()
        assert new_step == 1
        assert tracer.context.agent_context["step"] == 1
        
        # Increment again
        new_step = tracer.increment_agent_step()
        assert new_step == 2
        assert tracer.context.agent_context["step"] == 2
    
    def test_add_quality_hook(self):
        """Test adding quality hooks."""
        tracer = Tracer()
        tracer.trace_repo = Mock()
        
        # Add hooks
        hook1_called = False
        hook2_called = False
        
        def hook1(trace):
            nonlocal hook1_called
            hook1_called = True
            
        def hook2(trace):
            nonlocal hook2_called
            hook2_called = True
        
        tracer.add_quality_hook(hook1)
        tracer.add_quality_hook(hook2)
        
        assert len(tracer._quality_hooks) == 2
        
        # Start and complete a trace to trigger hooks
        with tracer.start_trace(operation="hook_trigger"):
            pass
        
        # Verify hooks were called
        assert hook1_called == True
        assert hook2_called == True
    
    def test_get_current_traces(self):
        """Test getting current trace stack."""
        tracer = Tracer()
        tracer.trace_repo = Mock()
        
        # Start multiple traces
        with tracer.start_trace(operation="trace1") as trace1:
            with tracer.start_trace(operation="trace2") as trace2:
                # Get current traces
                traces = tracer.get_current_traces()
                
                assert len(traces) == 2
                assert traces[0] == trace1
                assert traces[1] == trace2
                
                # Verify it's a copy
                traces.append({"test": "modification"})
                assert len(tracer.context.stack) == 2  # Original unchanged
    
    def test_get_current_agent_context(self):
        """Test getting current agent context."""
        tracer = Tracer()
        
        # Set agent context
        original_context = {
            "agent_id": "test_agent",
            "agent_type": "tester",
            "goal": "test_goal",
            "step": 5
        }
        tracer.context.agent_context = original_context.copy()
        
        # Get context
        retrieved = tracer.get_current_agent_context()
        
        assert retrieved == original_context
        # Verify it's a copy
        retrieved["step"] = 10
        assert tracer.context.agent_context["step"] == 5  # Original unchanged
    
    def test_should_persist_trace(self):
        """Test trace persistence decision logic."""
        tracer = Tracer()
        
        # Important operations should persist
        assert tracer._should_persist_trace("task_execution") == True
        assert tracer._should_persist_trace("llm_call") == True
        assert tracer._should_persist_trace("agent_decision") == True
        assert tracer._should_persist_trace("self_evaluation") == True
        
        # Other operations should not persist
        assert tracer._should_persist_trace("regular_operation") == False
        assert tracer._should_persist_trace("test") == False
    
    @patch('app.observability.tracer.ExecutionTrace')
    def test_persist_trace(self, mock_trace_model):
        """Test persisting trace to database."""
        tracer = Tracer()
        
        # Mock the save method
        mock_save = Mock()
        tracer.trace_repo.save = mock_save
        
        trace_data = {
            "trace_id": "test_persist_123",
            "parent_trace_id": "parent_123",
            "operation": "task_execution",
            "context": {"test": "data"},
            "decisions": [{"type": "test_decision"}],
            "events": [{"type": "test_event"}],
            "agent_observations": [{"type": "thought"}],
            "quality_metrics": {"score": 0.9},
            "error": None,
            "start_time": datetime.now(timezone.utc),
            "end_time": datetime.now(timezone.utc),
            "duration_ms": 1500,
            "success": True,
            "agent_context": {"agent_id": "test_agent"},
            "task_id": 123
        }

        # Test persistence
        tracer._persist_trace(trace_data)
        
        # Verify save was called with an ExecutionTrace instance
        mock_save.assert_called_once()
        
        # Get the actual ExecutionTrace instance that was created
        call_args = mock_save.call_args[0]
        trace_instance = call_args[0]
        
        # Verify it's an instance (not checking attributes since it's a mock)
        assert trace_instance is not None
    
    @patch('app.observability.tracer.ExecutionTrace')
    def test_persist_trace_with_error(self, mock_trace_model):
        """Test trace persistence with database error."""
        tracer = Tracer()
        
        # Mock save to raise exception
        def mock_save_raise(trace):
            raise Exception("Database connection failed")
        
        tracer.trace_repo.save = mock_save_raise
        
        trace_data = {
            "trace_id": "test_error_123",
            "operation": "task_execution",
            "start_time": datetime.now(timezone.utc),
            "end_time": datetime.now(timezone.utc),
            "duration_ms": 1000,
            "success": True,
        }
        
        # Should raise AppException
        with pytest.raises(AppException) as exc_info:
            tracer._persist_trace(trace_data)
        
        assert "Failed to persist trace" in str(exc_info.value)
        assert exc_info.value.extra["trace_id"] == "test_error_123"
    
    def test_verify_trace_hierarchy(self):
        """Test trace hierarchy verification."""
        tracer = Tracer()
        
        # Mock repository to return None (trace not found)
        tracer.trace_repo.get_by_trace_id = Mock(return_value=None)
        
        # Test with non-existent trace
        result = tracer.verify_trace_hierarchy("nonexistent_trace")
        assert "error" in result
        assert result["error"] == "Trace not found"

    def test_calculate_trace_quality(self):
        """Test trace quality calculation."""
        tracer = Tracer()
        
        # Test case 1: Successful, fast trace
        trace_data = {
            "success": True,
            "duration_ms": 1000,
            "decisions": [
                {"quality": {"overall_score": 0.8}},
                {"quality": {"overall_score": 0.9}},
            ]
        }
        
        tracer._calculate_trace_quality(trace_data)
        
        assert "quality_metrics" in trace_data
        metrics = trace_data["quality_metrics"]
        
        assert "success_factor" in metrics
        assert "efficiency_factor" in metrics
        assert "decision_quality_factor" in metrics
        assert "composite_quality_score" in metrics
        
        assert metrics["success_factor"] == 1.0
        assert metrics["efficiency_factor"] == 1.0  # Fast execution
        assert metrics["decision_quality_factor"] == pytest.approx(0.85, rel=1e-10)  # Average of 0.8 and 0.9
        assert 0.8 < metrics["composite_quality_score"] < 1.0

    def test_assess_decision_quality(self):
        """Test decision quality assessment."""
        tracer = Tracer()
        
        # Test with context
        context = {"options": ["A", "B"], "criteria": "accuracy", "reasoning": "detailed"}
        quality = tracer._assess_decision_quality("model.selection", context)
        
        assert "has_context" in quality
        assert "context_size" in quality
        assert "decision_type_specificity" in quality
        assert "overall_score" in quality
        assert "factor_weights" in quality
        
        assert quality["has_context"] == True
        assert quality["context_size"] > 0
        assert quality["decision_type_specificity"] == 0.5  # Contains 'selection' -> generic
        assert 0.5 <= quality["overall_score"] <= 1.0

    def test_assess_specificity(self):
        """Test decision type specificity assessment."""
        tracer = Tracer()
        
        # Generic types (contains 'decision', 'choice', or 'selection')
        assert tracer._assess_specificity("decision") == 0.5
        assert tracer._assess_specificity("choice") == 0.5
        assert tracer._assess_specificity("model_selection") == 0.5  # Contains 'selection'
        assert tracer._assess_specificity("model.selection") == 0.5  # Contains 'selection'
        assert tracer._assess_specificity("model.selection.specific") == 0.5  # Contains 'selection'
        
        # Specific types (contains '.' or '_' but NOT generic words)
        assert tracer._assess_specificity("classification.routing") == 0.9  # Has '.' and no generic words
        assert tracer._assess_specificity("agent_decision_routing") == 0.5  # Contains 'decision' (generic)
        assert tracer._assess_specificity("agent.routing") == 0.9  # Has '.' and no generic words
        
        # Moderately specific (neither generic nor has '.'/'_')
        assert tracer._assess_specificity("classification") == 0.7
        assert tracer._assess_specificity("validation") == 0.7

    def test_sanitize_for_logging(self):
        """Test data sanitization for logging."""
        tracer = Tracer()
        
        # Test with sensitive data
        data = {
            "username": "test_user",
            "password": "secret123",
            "api_key": "sk-1234567890",
            "token": "bearer_abc",
            "normal_field": "normal_value",
            "nested": {
                "secret_key": "another_secret",
                "public_data": "ok_to_log"
            },
            "list_data": [
                "item1",
                "secret_password",
                {"key": "value"}
            ],
            "long_string": "a" * 250
        }
        
        sanitized = tracer._sanitize_for_logging(data)
        
        # Verify sensitive data is redacted
        assert sanitized["password"] == "***REDACTED***"
        assert sanitized["api_key"] == "***REDACTED***"
        assert sanitized["token"] == "***REDACTED***"
        
        # Verify normal data preserved
        assert sanitized["username"] == "test_user"
        assert sanitized["normal_field"] == "normal_value"
        
        # Verify nested sanitization
        assert sanitized["nested"]["secret_key"] == "***REDACTED***"
        assert sanitized["nested"]["public_data"] == "ok_to_log"
        
        # Verify list sanitization
        assert sanitized["list_data"][0] == "item1"
        assert sanitized["list_data"][1] == "***REDACTED***"
        
        # Verify long string truncation
        assert len(sanitized["long_string"]) <= 203  # 200 + "..."
        assert sanitized["long_string"].endswith("...")


class TestTracerEdgeCases:
    """Edge case tests for Tracer."""
    
    def test_start_trace_without_agent_context_initialization(self):
        """Test trace starts even without explicit agent context initialization."""
        tracer = Tracer()
        tracer.trace_repo = Mock()
        
        # Don't set agent context
        with tracer.start_trace(operation="test") as trace_ctx:
            # Should have default agent context structure
            assert "agent_context" in trace_ctx
            agent_ctx = trace_ctx["agent_context"]
            assert agent_ctx["agent_id"] is None
            assert agent_ctx["agent_type"] is None
            assert agent_ctx["goal"] is None
            assert agent_ctx["step"] == 0
            assert agent_ctx["total_steps"] is None
    
    def test_record_outside_trace(self):
        """Test recording methods when no trace is active."""
        tracer = Tracer()
        
        # These should not crash when no trace is active
        tracer.record_decision("test", {})
        tracer.record_agent_observation("thought", {})
        tracer.record_event("update", {})
        tracer.record_quality_metric("test", 1.0)
        tracer.record_error("TestError", "test message")
        
        # No assertion needed - just verifying no exceptions
    
    def test_concurrent_tracing(self):
        """Test tracer in concurrent scenarios."""
        import threading
        
        tracer = Tracer()
        tracer.trace_repo = Mock()
        results = []
        
        def trace_worker(worker_id):
            with tracer.start_trace(operation=f"worker_{worker_id}") as trace_ctx:
                trace_ctx["worker_id"] = worker_id
                tracer.record_event("work_started", {"worker": worker_id})
                time.sleep(0.01)  # Simulate work
                tracer.record_event("work_completed", {"worker": worker_id})
                results.append(trace_ctx["trace_id"])
        
        # Start multiple workers
        threads = []
        for i in range(3):
            t = threading.Thread(target=trace_worker, args=(i,))
            threads.append(t)
            t.start()
        
        # Wait for completion
        for t in threads:
            t.join()
        
        # Verify all workers completed
        assert len(results) == 3
        # Each should have unique trace ID
        assert len(set(results)) == 3
    
    def test_trace_with_complex_decisions(self):
        """Test trace with complex decision structures."""
        tracer = Tracer()
        tracer.trace_repo = Mock()
        
        with tracer.start_trace(operation="complex_decision_test"):
            # Record decision with nested context
            complex_context = {
                "options": [
                    {"name": "option_a", "score": 0.8, "reasoning": "Fast"},
                    {"name": "option_b", "score": 0.9, "reasoning": "Accurate"},
                    {"name": "option_c", "score": 0.7, "reasoning": "Cheap"}
                ],
                "selection_criteria": {
                    "weights": {"speed": 0.3, "accuracy": 0.5, "cost": 0.2},
                    "thresholds": {"min_score": 0.6}
                },
                "chosen_option": "option_b",
                "analysis": {
                    "tradeoffs": ["speed vs accuracy"],
                    "confidence": 0.85,
                    "risk_assessment": "low"
                }
            }
            
            tracer.record_decision("multi_criteria_decision", complex_context)
            
            current = tracer.context.current
            decision = current["decisions"][0]
            
            # Verify complex context is preserved
            assert decision["context"] == complex_context
            # Verify quality assessment
            assert "quality" in decision
            assert decision["quality"]["has_context"] == True
            assert decision["quality"]["context_size"] > 0
    
    def test_quality_hook_exception_handling(self):
        """Test that quality hook exceptions don't break trace completion."""
        tracer = Tracer()
        tracer.trace_repo = Mock()
        
        # Add a hook that raises exception
        hook_called = False
        def failing_hook(trace):
            nonlocal hook_called
            hook_called = True
            raise Exception("Hook failed!")
        
        tracer.add_quality_hook(failing_hook)
        
        # Trace should still complete despite hook failure
        with tracer.start_trace(operation="hook_exception_test") as trace_ctx:
            pass
        
        # Verify hook was called
        assert hook_called == True
        # Verify trace was still completed
        assert "end_time" in trace_ctx
        assert "duration_ms" in trace_ctx


# Test for threading behavior
class TestTracerThreadSafety:
    """Tests for tracer thread safety."""
    
    def test_thread_local_context(self):
        """Verify that trace context is thread-local."""
        tracer = Tracer()
        tracer.trace_repo = Mock()
        
        main_thread_trace_id = None
        worker_thread_trace_id = None
        
        def worker():
            nonlocal worker_thread_trace_id
            with tracer.start_trace(operation="worker_operation") as trace_ctx:
                worker_thread_trace_id = trace_ctx["trace_id"]
                # Worker should see its own trace, not main thread's
                assert tracer.context.current["trace_id"] == worker_thread_trace_id
        
        # Main thread starts trace
        with tracer.start_trace(operation="main_operation") as trace_ctx:
            main_thread_trace_id = trace_ctx["trace_id"]
            
            # Start worker thread
            t = threading.Thread(target=worker)
            t.start()
            t.join()
        
        # Verify traces are different
        assert main_thread_trace_id is not None
        assert worker_thread_trace_id is not None
        assert main_thread_trace_id != worker_thread_trace_id
