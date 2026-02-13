from datetime import datetime, timezone
from app.models.task import Task
from app.models.trace import ExecutionTrace
from app.models.agent_insight import AgentInsight


class TestDatabaseModels:
    """Integration tests for database models."""
    
    def test_task_creation(self, db_session):
        """Test creating and retrieving a task."""
        task = Task(
            task_type="summarize",
            input_data={"text": "Test text"},
            parameters={"max_length": 100},
            status="pending",
        )
        
        db_session.add(task)
        db_session.commit()
        
        # Retrieve the task
        retrieved = Task.query.get(task.id)
        assert retrieved is not None
        assert retrieved.task_type == "summarize"
        assert retrieved.status == "pending"
        assert isinstance(retrieved.created_at, datetime)
        
    def test_task_completion(self, db_session):
        """Test completing a task with quality score."""
        task = Task(
            task_type="summarize",
            input_data={"text": "Test text"},
            parameters={"max_length": 100},
            status="pending",
        )
        db_session.add(task)
        db_session.commit()
        
        # Start execution
        task.start_execution()
        assert task.status == "running"
        assert task.started_at is not None
        
        # Complete execution
        task.complete_execution(
            success=True,
            output={"summary": "Test summary"},
            error=None
        )
        
        # Add validation
        task.record_validation(
            check_name="length_check",
            passed=True,
            details={"max_length": 100, "actual": 50}
        )
        
        # Calculate quality
        quality_score = task.calculate_quality_score(success=True)
        
        assert task.status == "completed"
        assert task.completed_at is not None
        assert task.execution_time_ms > 0
        assert quality_score is not None
        assert 0 <= quality_score <= 1
        assert task.validation_results["passed"] == 1
        assert task.validation_results["total"] == 1
        
        # Test to_dict method
        task_dict = task.to_dict()
        assert task_dict["id"] == task.id
        assert task_dict["status"] == "completed"
        assert task_dict["quality_score"] == quality_score
        assert "trace_count" in task_dict
        
    def test_trace_creation(self, db_session, sample_task):
        """Test creating and retrieving a trace."""
        trace = ExecutionTrace(
            trace_id="test_trace_456",
            operation="llm_call",
            task_id=sample_task.id,
            context={"model": "test-model", "temperature": 0.5},
            quality_metrics={"composite_quality_score": 0.85, "success_factor": 1.0},
            agent_context={"agent_id": "test_agent", "goal": "summarize text"},
            start_time=datetime.now(timezone.utc),
            end_time=datetime.now(timezone.utc),
            duration_ms=1234,
            success=True,
        )
        
        db_session.add(trace)
        db_session.commit()
        
        # Retrieve trace
        retrieved = ExecutionTrace.query.filter_by(trace_id="test_trace_456").first()
        assert retrieved is not None
        assert retrieved.operation == "llm_call"
        assert retrieved.task_id == sample_task.id
        assert retrieved.success == True
        assert retrieved.duration_ms == 1234
        assert retrieved.quality_metrics["composite_quality_score"] == 0.85
        
        # Test to_dict method
        trace_dict = retrieved.to_dict()
        assert trace_dict["trace_id"] == "test_trace_456"
        assert trace_dict["operation"] == "llm_call"
        assert trace_dict["success"] == True
        
    def test_agent_insight_creation(self, db_session, sample_task):
        """Test creating and retrieving agent insights."""
        insight = AgentInsight(
            agent_name="test_agent",
            insight_type="quality_improvement",
            insight_data={
                "current_quality": 0.6,
                "target_quality": 0.8,
                "gap": 0.2,
                "recommendations": ["Add validation", "Improve error handling"]
            },
            source_task_id=sample_task.id,
            confidence_score=0.8,
            impact_prediction="high"
        )
        
        db_session.add(insight)
        db_session.commit()
        
        # Retrieve insight
        retrieved = AgentInsight.query.filter_by(agent_name="test_agent").first()
        assert retrieved is not None
        assert retrieved.insight_type == "quality_improvement"
        assert retrieved.confidence_score == 0.8
        assert retrieved.impact_prediction == "high"
        assert not retrieved.is_applied()  # Should not be applied yet
        
        # Test marking as applied
        retrieved.mark_applied("implemented_successfully")
        db_session.commit()
        
        assert retrieved.applied_at is not None
        assert retrieved.applied_result == "implemented_successfully"
        assert retrieved.is_applied() == True
        
        # Test to_dict method
        insight_dict = retrieved.to_dict()
        assert insight_dict["agent_name"] == "test_agent"
        assert insight_dict["insight_type"] == "quality_improvement"
        assert insight_dict["is_applied"] == True
        
    def test_task_trace_relationship(self, db_session, sample_task, sample_trace):
        """Test the relationship between tasks and traces."""
        # Add another trace to the same task
        trace2 = ExecutionTrace(
            trace_id="test_trace_456",
            operation="data_transform",
            task_id=sample_task.id,
            context={"transform": "summarize"},
            start_time=datetime.now(timezone.utc),
            end_time=datetime.now(timezone.utc),
            duration_ms=500,
            success=True,
        )
        db_session.add(trace2)
        db_session.commit()
        
        # Retrieve task with traces
        task_with_traces = Task.query.get(sample_task.id)
        assert task_with_traces is not None
        assert len(task_with_traces.traces) == 2
        
        # Verify trace details
        trace_operations = [trace.operation for trace in task_with_traces.traces]
        assert "task_execution" in trace_operations
        assert "data_transform" in trace_operations
        
    def test_agent_insight_statistics(self, db_session):
        """Test agent insight statistics."""
        # Create multiple insights for different agents
        insights_data = [
            ("agent1", "quality_improvement", 0.7, False),
            ("agent1", "efficiency_gain", 0.8, True),
            ("agent2", "behavior_pattern", 0.6, False),
            ("agent2", "quality_improvement", 0.9, False),
            ("agent2", "decision_improvement", 0.5, True),
        ]
        
        for agent_name, insight_type, confidence, applied in insights_data:
            insight = AgentInsight(
                agent_name=agent_name,
                insight_type=insight_type,
                insight_data={"test": "data"},
                confidence_score=confidence,
                impact_prediction="medium"
            )
            if applied:
                insight.mark_applied("test")
            db_session.add(insight)
        
        db_session.commit()
        
        # Query statistics
        from app.repositories.agent_insight_repository import AgentInsightRepository
        repo = AgentInsightRepository()
        
        # Test find_by_agent
        agent1_insights = repo.find_by_agent("agent1")
        assert len(agent1_insights) == 2
        
        # Test find_by_type
        quality_insights = repo.find_by_type("quality_improvement")
        assert len(quality_insights) == 2
        
        # Test find_unapplied
        unapplied = repo.find_unapplied()
        assert len(unapplied) == 3  # 5 total, 2 applied
        
        # Test statistics
        stats = repo.get_statistics()
        assert stats["total"] == 5
        assert stats["applied_count"] == 2
        assert stats["applied_rate"] == 0.4
        assert "quality_improvement" in stats["by_type"]
        assert stats["by_type"]["quality_improvement"] == 2
