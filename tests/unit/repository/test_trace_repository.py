"""
Unit tests for TraceRepository.
"""
import uuid
import pytest
from datetime import datetime, timedelta, timezone
from unittest.mock import Mock, patch, MagicMock
from collections import defaultdict

from app.repositories.trace_repository import TraceRepository
from app.models.trace import ExecutionTrace


class TestTraceRepository:
    """Unit tests for TraceRepository."""
    
    def test_initialization(self):
        """Test TraceRepository initializes correctly."""
        repo = TraceRepository()
        assert repo is not None
    
    def test_save(self, db_session):
        """Test saving a trace to database."""
        repo = TraceRepository()
        
        # Create a trace
        trace = ExecutionTrace(
            trace_id="test_save_123",
            operation="test_operation",
            context={"test": "data"},
            start_time=datetime.now(timezone.utc),
            success=True,
        )
        
        # Save it
        saved_trace = repo.save(trace)
        
        # Verify
        assert saved_trace.id is not None
        assert saved_trace.trace_id == "test_save_123"
        assert saved_trace.operation == "test_operation"
    
    def test_get_by_id(self, db_session):
        """Test getting trace by database ID."""
        repo = TraceRepository()
        
        # Create and save a trace
        trace = ExecutionTrace(
            trace_id="test_get_by_id_123",
            operation="test_operation",
            start_time=datetime.now(timezone.utc),
            success=True,
        )
        db_session.add(trace)
        db_session.commit()
        
        # Retrieve by ID
        retrieved = repo.get_by_id(trace.id)
        
        # Verify
        assert retrieved is not None
        assert retrieved.id == trace.id
        assert retrieved.trace_id == "test_get_by_id_123"
    
    def test_get_by_id_not_found(self, db_session):
        """Test getting non-existent trace by ID."""
        repo = TraceRepository()
        
        # Try to get non-existent ID
        result = repo.get_by_id(999999)
        
        # Should return None
        assert result is None
    
    def test_get_by_trace_id(self, db_session):
        """Test getting trace by UUID trace ID."""
        repo = TraceRepository()
        
        # Create and save a trace
        trace = ExecutionTrace(
            trace_id="test_uuid_123",
            operation="test_operation",
            start_time=datetime.now(timezone.utc),
            success=True,
        )
        db_session.add(trace)
        db_session.commit()
        
        # Retrieve by trace_id
        retrieved = repo.get_by_trace_id("test_uuid_123")
        
        # Verify
        assert retrieved is not None
        assert retrieved.trace_id == "test_uuid_123"
        assert retrieved.operation == "test_operation"
    
    def test_get_by_trace_id_not_found(self, db_session):
        """Test getting non-existent trace by UUID."""
        repo = TraceRepository()
        
        # Try to get non-existent trace_id
        result = repo.get_by_trace_id("non_existent_uuid")
        
        # Should return None
        assert result is None
    
    def test_get_traces_for_task(self, db_session):
        """Test getting all traces for a task."""
        repo = TraceRepository()
        
        # Create traces for task 1
        for i in range(3):
            trace = ExecutionTrace(
                trace_id=f"task1_trace_{i}",
                operation=f"operation_{i}",
                task_id=1,
                start_time=datetime.now(timezone.utc) - timedelta(minutes=i),
                success=True,
            )
            db_session.add(trace)
        
        # Create traces for task 2
        for i in range(2):
            trace = ExecutionTrace(
                trace_id=f"task2_trace_{i}",
                operation=f"operation_{i}",
                task_id=2,
                start_time=datetime.now(timezone.utc) - timedelta(minutes=i),
                success=True,
            )
            db_session.add(trace)
        
        db_session.commit()
        
        # Get traces for task 1
        task1_traces = repo.get_traces_for_task(1)
        
        # Verify
        assert len(task1_traces) == 3
        assert all(trace.task_id == 1 for trace in task1_traces)
        # Should be ordered by start_time
        assert task1_traces[0].start_time <= task1_traces[1].start_time
        
        # Get traces for task 2
        task2_traces = repo.get_traces_for_task(2)
        assert len(task2_traces) == 2
    
    def test_get_traces_for_task_empty(self, db_session):
        """Test getting traces for non-existent task."""
        repo = TraceRepository()
        
        # Get traces for non-existent task
        traces = repo.get_traces_for_task(999)
        
        # Should return empty list
        assert traces == []
    
    def test_find_by_operation(self, db_session):
        """Test finding traces by operation type."""
        repo = TraceRepository()
        
        # Create traces with different operations
        operations = ["llm_call", "data_transform", "llm_call", "validation"]
        
        for i, operation in enumerate(operations):
            trace = ExecutionTrace(
                trace_id=f"trace_{i}",
                operation=operation,
                start_time=datetime.now(timezone.utc) - timedelta(minutes=i),
                success=True,
            )
            db_session.add(trace)
        
        db_session.commit()
        
        # Find llm_call operations
        llm_traces = repo.find_by_operation("llm_call")
        
        # Verify
        assert len(llm_traces) == 2
        assert all(trace.operation == "llm_call" for trace in llm_traces)
        # Should be ordered by start_time descending (most recent first)
        assert llm_traces[0].start_time >= llm_traces[1].start_time
    
    def test_find_by_operation_with_limit(self, db_session):
        """Test finding traces by operation with limit."""
        repo = TraceRepository()
        
        # Create many llm_call traces with UNIQUE trace IDs
        for i in range(10):
            trace = ExecutionTrace(
                trace_id=f"trace_{uuid.uuid4()}",  # Use UUID for uniqueness
                operation="llm_call",
                start_time=datetime.now(timezone.utc) - timedelta(minutes=i),
                success=True,
            )
            db_session.add(trace)
        
        db_session.commit()
        
        # Find with limit 3
        traces = repo.find_by_operation("llm_call", limit=3)
        
        # Verify limit works
        assert len(traces) == 3

    def test_find_by_operation_none_found(self, db_session):
        """Test finding traces by non-existent operation."""
        repo = TraceRepository()
        
        # Find non-existent operation
        traces = repo.find_by_operation("non_existent_operation")
        
        # Should return empty list
        assert traces == []
    
    @patch('app.repositories.trace_repository.datetime')
    def test_get_statistics(self, mock_datetime, db_session):
        """Test getting trace statistics."""
        repo = TraceRepository()
        
        # Clean existing traces first
        db_session.query(ExecutionTrace).delete()
        db_session.commit()
        
        # Mock current time
        now = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        mock_datetime.now.return_value = now
        
        # Create test traces (within 24 hours)
        cutoff = now - timedelta(hours=24)
        
        traces_data = [
            # operation, success, duration_ms, error, created_at offset
            ("llm_call", True, 1000, None, 1),  # 1 hour ago
            ("llm_call", False, 2000, {"type": "error"}, 2),  # 2 hours ago
            ("data_transform", True, 1500, None, 3),  # 3 hours ago
            ("validation", True, 500, None, 23),  # 23 hours ago (still within 24h)
        ]
        
        for i, (operation, success, duration_ms, error, hours_ago) in enumerate(traces_data):
            trace = ExecutionTrace(
                trace_id=f"stat_trace_{uuid.uuid4()}",  # Use UUID
                operation=operation,
                start_time=now - timedelta(hours=hours_ago),
                end_time=now - timedelta(hours=hours_ago) + timedelta(milliseconds=duration_ms),
                duration_ms=duration_ms,
                success=success,
                error=error,
                created_at=now - timedelta(hours=hours_ago),
            )
            db_session.add(trace)
        
        # Create one trace older than 24 hours (should be excluded)
        old_trace = ExecutionTrace(
            trace_id=f"old_trace_{uuid.uuid4()}",  # Use UUID
            operation="old_operation",
            start_time=now - timedelta(hours=25),
            success=True,
            created_at=now - timedelta(hours=25),
        )
        db_session.add(old_trace)
        
        db_session.commit()
        
        # Get statistics
        stats = repo.get_statistics(hours=24)
        
        # Verify
        assert stats["total"] == 4  # Only traces within 24 hours
        assert stats["success_rate"] == 0.75  # 3 out of 4 successful
        assert stats["avg_duration"] == 1250  # (1000+2000+1500+500)/4
        assert stats["error_count"] == 1  # One trace with error
        
        # Verify operation counts
        assert stats["by_operation"]["llm_call"] == 2
        assert stats["by_operation"]["data_transform"] == 1
        assert stats["by_operation"]["validation"] == 1
        assert "old_operation" not in stats["by_operation"]

    @patch('app.repositories.trace_repository.datetime')
    def test_get_statistics_no_traces(self, mock_datetime, db_session):
        """Test getting statistics with no traces."""
        repo = TraceRepository()
        
        # Clean existing traces first
        db_session.query(ExecutionTrace).delete()
        db_session.commit()
        
        # Mock current time
        now = datetime.now(timezone.utc)
        mock_datetime.now.return_value = now
        
        # Get statistics (no traces in DB)
        stats = repo.get_statistics(hours=24)
        
        # Verify empty statistics
        assert stats["total"] == 0
        assert stats["success_rate"] == 0
        assert stats["avg_duration"] == 0
        assert stats["error_count"] == 0
        assert stats["by_operation"] == {}

    @patch('app.repositories.trace_repository.datetime')
    def test_cleanup_old_traces(self, mock_datetime, db_session):
        """Test cleaning up old traces."""
        repo = TraceRepository()
        
        # Mock current time
        now = datetime(2024, 1, 10, 12, 0, 0, tzinfo=timezone.utc)
        mock_datetime.now.return_value = now
        
        # Create traces with different ages
        traces_data = [
            ("recent_trace", 5),   # 5 days old (keep)
            ("old_trace_1", 35),   # 35 days old (delete)
            ("old_trace_2", 40),   # 40 days old (delete)
            ("borderline_trace", 30),  # Exactly 30 days old (keep - >= cutoff)
        ]
        
        for trace_id, days_ago in traces_data:
            trace = ExecutionTrace(
                trace_id=trace_id,
                operation="test",
                start_time=now - timedelta(days=days_ago),
                created_at=now - timedelta(days=days_ago),
                success=True,
            )
            db_session.add(trace)
        
        db_session.commit()
        
        # Clean up traces older than 30 days
        deleted_count = repo.cleanup_old_traces(days=30)
        
        # Verify
        assert deleted_count == 2  # old_trace_1 and old_trace_2
        
        # Check remaining traces
        remaining = ExecutionTrace.query.all()
        remaining_ids = {t.trace_id for t in remaining}
        
        assert "recent_trace" in remaining_ids
        assert "borderline_trace" in remaining_ids  # Exactly 30 days should be kept
        assert "old_trace_1" not in remaining_ids
        assert "old_trace_2" not in remaining_ids
    
    @patch('app.repositories.trace_repository.datetime')
    def test_cleanup_old_traces_no_old_traces(self, mock_datetime, db_session):
        """Test cleanup when no old traces exist."""
        repo = TraceRepository()
        
        # Clean existing traces first
        db_session.query(ExecutionTrace).delete()
        db_session.commit()
        
        # Mock current time
        now = datetime.now(timezone.utc)
        mock_datetime.now.return_value = now
        
        # Create only recent traces with UNIQUE trace ID
        trace = ExecutionTrace(
            trace_id=f"recent_trace_{uuid.uuid4()}",  # Use UUID
            operation="test",
            start_time=now - timedelta(days=5),
            created_at=now - timedelta(days=5),
            success=True,
        )
        db_session.add(trace)
        db_session.commit()
        
        # Clean up traces older than 30 days
        deleted_count = repo.cleanup_old_traces(days=30)
        
        # Verify
        assert deleted_count == 0
        
        # Trace should still exist
        remaining = ExecutionTrace.query.all()
        assert len(remaining) == 1

    @patch('app.repositories.trace_repository.datetime')
    @patch('app.repositories.trace_repository.statistics')
    def test_get_agent_metrics(self, mock_statistics, mock_datetime, db_session):
        """Test getting agent metrics."""
        repo = TraceRepository()
        
        # Clean any existing traces first
        db_session.query(ExecutionTrace).delete()
        db_session.commit()
        
        # Mock current time
        now = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        mock_datetime.now.return_value = now
        
        # Mock statistics.mean
        mock_statistics.mean.side_effect = lambda x: sum(x) / len(x) if x else 0
        
        # Create test traces with agent contexts
        traces_data = [
            # agent_id, operation, success, duration_ms, quality_score, hours_ago
            ("agent_1", "llm_call", True, 1000, 0.9, 1),
            ("agent_1", "llm_call", False, 2000, 0.7, 2),
            ("agent_1", "data_transform", True, 1500, 0.8, 3),
            ("agent_2", "validation", True, 500, 0.95, 1),
            ("agent_2", "validation", True, 600, 0.85, 2),
            (None, "unknown_op", True, 300, None, 1),  # No agent context
        ]
        
        for i, (agent_id, operation, success, duration_ms, quality_score, hours_ago) in enumerate(traces_data):
            trace = ExecutionTrace(
                trace_id=f"agent_trace_{i}_{uuid.uuid4()}",  # Unique ID
                operation=operation,
                agent_context={"agent_id": agent_id} if agent_id else None,
                quality_metrics={"quality_score": quality_score} if quality_score is not None else {},
                start_time=now - timedelta(hours=hours_ago),
                end_time=now - timedelta(hours=hours_ago) + timedelta(milliseconds=duration_ms),
                duration_ms=duration_ms,
                success=success,
                created_at=now - timedelta(hours=hours_ago),
            )
            db_session.add(trace)
        
        db_session.commit()
        
        # Get agent metrics
        metrics = repo.get_agent_metrics(time_window_hours=24)
        
        # Verify structure
        assert "agents" in metrics
        assert "total_agents" in metrics
        assert "total_traces" in metrics
        assert "time_window_hours" in metrics
        
        # Verify agent counts
        assert metrics["total_agents"] == 3  # agent_1, agent_2, unknown
        assert metrics["total_traces"] == 6
        
        # Verify agent_1 metrics
        agent1 = metrics["agents"]["agent_1"]
        assert agent1["total_traces"] == 3
        assert agent1["success_rate"] == pytest.approx(2/3, rel=0.01)  # 2 out of 3 successful
        assert agent1["average_duration_ms"] == pytest.approx(1500, rel=0.01)  # (1000+2000+1500)/3
        assert agent1["average_quality_score"] == pytest.approx(0.8, rel=0.01)  # (0.9+0.7+0.8)/3
        
        # Verify quality distribution for agent_1
        quality_dist = agent1["quality_distribution"]
        assert quality_dist["excellent"] == 1  # 0.9
        assert quality_dist["good"] == 1  # 0.8
        assert quality_dist["acceptable"] == 1  # 0.7
        
        # Verify most common operations for agent_1
        assert agent1["most_common_operations"]["llm_call"] == 2
        assert agent1["most_common_operations"]["data_transform"] == 1

    @patch('app.repositories.trace_repository.datetime')
    def test_get_agent_metrics_no_traces(self, mock_datetime, db_session):
        """Test getting agent metrics with no traces."""
        repo = TraceRepository()
        
        # Clean any existing traces first
        db_session.query(ExecutionTrace).delete()
        db_session.commit()
        
        # Mock current time
        now = datetime.now(timezone.utc)
        mock_datetime.now.return_value = now
        
        # Get agent metrics (no traces in DB)
        metrics = repo.get_agent_metrics(time_window_hours=24)
        
        # Verify empty metrics
        assert metrics["agents"] == {}
        assert metrics["total_agents"] == 0
        assert metrics["total_traces"] == 0
        assert metrics["time_window_hours"] == 24

    @patch('app.repositories.trace_repository.datetime')
    @patch('app.repositories.trace_repository.statistics')
    def test_get_quality_trends(self, mock_statistics, mock_datetime, db_session):
        """Test getting quality trends over time."""
        repo = TraceRepository()
        
        # Clean any existing traces first
        db_session.query(ExecutionTrace).delete()
        db_session.commit()
        
        # Mock current time
        now = datetime(2024, 1, 7, 12, 0, 0, tzinfo=timezone.utc)
        mock_datetime.now.return_value = now
        
        # Mock statistics functions
        mock_statistics.mean.side_effect = lambda x: sum(x) / len(x) if x else 0
        mock_statistics.min = min
        mock_statistics.max = max
        
        # Create test traces with quality scores over 7 days
        # Format: (date_str, quality_score)
        quality_data = [
            ("2024-01-01", 0.9),
            ("2024-01-01", 0.8),
            ("2024-01-02", 0.7),
            ("2024-01-03", 0.95),
            ("2024-01-03", 0.85),
            ("2024-01-05", 0.75),
            ("2024-01-06", 0.9),
            ("2024-01-06", 0.8),
            ("2024-01-06", 0.85),
        ]
        
        for i, (date_str, quality_score) in enumerate(quality_data):
            # Parse date
            date = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            
            trace = ExecutionTrace(
                trace_id=f"quality_trace_{i}_{uuid.uuid4()}",  # Unique ID
                operation="test",
                quality_metrics={"quality_score": quality_score},
                start_time=date,
                created_at=date,
                success=True,
            )
            db_session.add(trace)
        
        # Add one trace without quality metrics (should be excluded)
        no_quality_trace = ExecutionTrace(
            trace_id=f"no_quality_trace_{uuid.uuid4()}",  # Unique ID
            operation="test",
            quality_metrics={},  # No quality_score
            start_time=now - timedelta(days=1),
            created_at=now - timedelta(days=1),
            success=True,
        )
        db_session.add(no_quality_trace)
        
        db_session.commit()
        
        # Get quality trends for 7 days
        trends = repo.get_quality_trends(days=7)
        
        # Verify structure
        assert "trends" in trends
        assert "days_analyzed" in trends
        assert "total_samples" in trends
        
        # Verify total samples (excludes trace without quality_score)
        assert trends["total_samples"] == 9
        
        # Verify trends are sorted by date
        trend_dates = list(trends["trends"].keys())
        assert trend_dates == sorted(trend_dates)
        
        # Verify specific day stats
        jan_6_stats = trends["trends"]["2024-01-06"]
        assert jan_6_stats["sample_count"] == 3
        assert jan_6_stats["average_quality"] == pytest.approx(0.85, rel=0.01)  # (0.9+0.8+0.85)/3
        assert jan_6_stats["min_quality"] == 0.8
        assert jan_6_stats["max_quality"] == 0.9

    @patch('app.repositories.trace_repository.datetime')
    def test_get_quality_trends_no_data(self, mock_datetime, db_session):
        """Test getting quality trends with no quality data."""
        repo = TraceRepository()
        
        # Clean any existing traces first
        db_session.query(ExecutionTrace).delete()
        db_session.commit()
        
        # Mock current time
        now = datetime.now(timezone.utc)
        mock_datetime.now.return_value = now
        
        # Create trace without quality metrics
        trace = ExecutionTrace(
            trace_id=f"no_quality_trace_{uuid.uuid4()}",  # Unique ID
            operation="test",
            quality_metrics={},  # Empty
            start_time=now - timedelta(days=1),
            created_at=now - timedelta(days=1),
            success=True,
        )
        db_session.add(trace)
        db_session.commit()
        
        # Get quality trends
        trends = repo.get_quality_trends(days=7)
        
        # Should return empty trends
        assert trends["trends"] == {}
        assert trends["total_samples"] == 0
        assert trends["days_analyzed"] == 7

    @patch('app.repositories.trace_repository.datetime')
    @patch('app.repositories.trace_repository.statistics')
    def test_get_decision_analytics(self, mock_statistics, mock_datetime, db_session):
        """Test getting decision analytics."""
        repo = TraceRepository()
        
        # Mock current time
        now = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        mock_datetime.now.return_value = now
        
        # Mock statistics.mean
        mock_statistics.mean.side_effect = lambda x: sum(x) / len(x) if x else 0
        
        # Create test traces with decisions
        traces_data = [
            # trace success, decisions list, hours_ago
            (True, [
                {"type": "model_selection", "quality": {"overall_score": 0.8}},
                {"type": "parameter_tuning", "quality": {"overall_score": 0.9}},
            ], 1),
            (False, [
                {"type": "model_selection", "quality": {"overall_score": 0.6}},
                {"type": "validation", "timestamp": "2024-01-01T10:00:00Z"},
            ], 2),
            (True, [
                {"type": "model_selection", "quality": {"overall_score": 0.85}},
            ], 3),
            (True, [], 4),  # Empty decisions list
            (True, None, 5),  # None decisions
        ]
        
        for i, (success, decisions, hours_ago) in enumerate(traces_data):
            trace = ExecutionTrace(
                trace_id=f"decision_trace_{i}",
                operation=f"operation_{i}",
                agent_context={"agent_id": f"agent_{i % 2}"},
                decisions=decisions,
                start_time=now - timedelta(hours=hours_ago),
                success=success,
                created_at=now - timedelta(hours=hours_ago),
            )
            db_session.add(trace)
        
        db_session.commit()
        
        # Get decision analytics
        analytics = repo.get_decision_analytics(time_window_hours=24)
        
        # Verify structure
        assert "total_decisions" in analytics
        assert "unique_decision_types" in analytics
        assert "average_quality" in analytics
        assert "decision_types" in analytics
        assert "time_window_hours" in analytics
        
        # Verify counts
        # Trace 0: 2 decisions, Trace 1: 2 decisions, Trace 2: 1 decision = 5 total
        assert analytics["total_decisions"] == 5
        
        # Verify unique types
        assert analytics["unique_decision_types"] == 3  # model_selection, parameter_tuning, validation
        
        # Verify average quality (only decisions with quality_score)
        # model_selection: 0.8, 0.6, 0.85 = average 0.75
        # parameter_tuning: 0.9
        # validation: no quality_score
        # Overall: (0.8 + 0.9 + 0.6 + 0.85) / 4 = 0.7875
        assert analytics["average_quality"] == pytest.approx(0.7875, rel=0.01)
        
        # Verify decision type statistics
        type_stats = analytics["decision_types"]
        
        # model_selection should have highest count
        assert "model_selection" in type_stats
        assert type_stats["model_selection"]["count"] == 3
        assert type_stats["model_selection"]["average_quality"] == pytest.approx(0.75, rel=0.01)
        assert type_stats["model_selection"]["success_rate"] == pytest.approx(2/3, rel=0.01)  # 2 successful out of 3
        
        # parameter_tuning
        assert "parameter_tuning" in type_stats
        assert type_stats["parameter_tuning"]["count"] == 1
        assert type_stats["parameter_tuning"]["average_quality"] == 0.9
        assert type_stats["parameter_tuning"]["success_rate"] == 1.0  # Only one, successful
        
        # validation (no quality score)
        assert "validation" in type_stats
        assert type_stats["validation"]["count"] == 1
        assert type_stats["validation"]["average_quality"] is None  # No quality score
        assert type_stats["validation"]["success_rate"] == 0.0  # Failed trace
    
    @patch('app.repositories.trace_repository.datetime')
    def test_get_decision_analytics_no_decisions(self, mock_datetime, db_session):
        """Test getting decision analytics with no decisions."""
        repo = TraceRepository()
        
        # Clean any existing traces first
        db_session.query(ExecutionTrace).delete()
        db_session.commit()
        
        # Mock current time
        now = datetime.now(timezone.utc)
        mock_datetime.now.return_value = now
        
        # Create trace without decisions
        trace = ExecutionTrace(
            trace_id=f"no_decisions_trace_{uuid.uuid4()}",  # Unique ID
            operation="test",
            decisions=[],  # Empty list
            start_time=now - timedelta(hours=1),
            created_at=now - timedelta(hours=1),
            success=True,
        )
        db_session.add(trace)
        db_session.commit()
        
        # Get decision analytics
        analytics = repo.get_decision_analytics(time_window_hours=24)
        
        # Should indicate no decisions
        assert analytics["no_decisions"] == True

    @patch('app.repositories.trace_repository.datetime')
    def test_get_decision_analytics_empty_db(self, mock_datetime, db_session):
        """Test getting decision analytics with empty database."""
        repo = TraceRepository()
        
        # Clean any existing traces first
        db_session.query(ExecutionTrace).delete()
        db_session.commit()
        
        # Mock current time
        now = datetime.now(timezone.utc)
        mock_datetime.now.return_value = now
        
        # Get decision analytics (no traces in DB)
        analytics = repo.get_decision_analytics(time_window_hours=24)
        
        # Should indicate no decisions
        assert analytics["no_decisions"] == True

# Edge case tests
class TestTraceRepositoryEdgeCases:
    """Edge case tests for TraceRepository."""
    
    def test_save_with_existing_id(self, db_session):
        """Test saving trace that already exists (update)."""
        repo = TraceRepository()
        
        # Create and save initial trace
        trace = ExecutionTrace(
            trace_id="update_test_123",
            operation="initial_operation",
            start_time=datetime.now(timezone.utc),
            success=True,
        )
        db_session.add(trace)
        db_session.commit()
        
        # Modify and save again
        trace.operation = "updated_operation"
        trace.success = False
        
        saved = repo.save(trace)
        
        # Verify update
        assert saved.operation == "updated_operation"
        assert saved.success == False
        assert saved.id == trace.id  # Same ID
    
    def test_get_statistics_traces_without_duration(self, db_session):
        """Test statistics with traces that have no duration."""
        repo = TraceRepository()
        
        # Clean any existing traces first
        db_session.query(ExecutionTrace).delete()
        db_session.commit()
        
        # Create traces without duration_ms
        for i in range(3):
            trace = ExecutionTrace(
                trace_id=f"no_duration_{i}_{uuid.uuid4()}",  # Unique ID
                operation="test",
                start_time=datetime.now(timezone.utc),
                success=True,
                duration_ms=None,  # No duration
            )
            db_session.add(trace)
        
        db_session.commit()
        
        # Get statistics
        stats = repo.get_statistics(hours=24)
        
        # Should handle None duration gracefully
        assert stats["avg_duration"] == 0
        assert stats["total"] == 3

    def test_agent_metrics_with_complex_agent_context(self, db_session):
        """Test agent metrics with complex agent context structures."""
        repo = TraceRepository()
        
        # Create traces with various agent context structures
        traces_data = [
            # agent_context structure
            ({"agent_id": "simple_agent", "extra": "data"}, True),
            ({"agent_id": "nested_agent", "metadata": {"level": 1}}, True),
            ({"not_agent_id": "wrong_key"}, False),  # No agent_id key
            ({}, False),  # Empty dict
            (None, False),  # None
        ]
        
        for i, (agent_context, has_agent_id) in enumerate(traces_data):
            trace = ExecutionTrace(
                trace_id=f"complex_agent_{i}",
                operation="test",
                agent_context=agent_context,
                start_time=datetime.now(timezone.utc),
                success=True,
            )
            db_session.add(trace)
        
        db_session.commit()
        
        # Get agent metrics
        metrics = repo.get_agent_metrics(time_window_hours=24)
        
        # Verify
        assert metrics["total_agents"] == 3  # simple_agent, nested_agent, unknown
        assert "simple_agent" in metrics["agents"]
        assert "nested_agent" in metrics["agents"]
        assert "unknown" in metrics["agents"]
    
    def test_quality_trends_with_malformed_quality_metrics(self, db_session):
        """Test quality trends with malformed quality metrics."""
        repo = TraceRepository()
        
        # Create traces with various quality metric structures
        quality_metrics_data = [
            {"quality_score": 0.8},  # Valid
            {"score": 0.7},  # Wrong key
            {},  # Empty
            None,  # None
            "not_a_dict",  # Wrong type
        ]
        
        for i, quality_metrics in enumerate(quality_metrics_data):
            trace = ExecutionTrace(
                trace_id=f"malformed_quality_{i}",
                operation="test",
                quality_metrics=quality_metrics,
                start_time=datetime.now(timezone.utc),
                success=True,
            )
            db_session.add(trace)
        
        db_session.commit()
        
        # Get quality trends
        trends = repo.get_quality_trends(days=7)
        
        # Should only count the valid quality_score
        assert trends["total_samples"] == 1  # Only first trace has quality_score


# Performance/scale tests
class TestTraceRepositoryPerformance:
    """Performance and scale tests for TraceRepository."""
    
    def test_find_by_operation_large_dataset(self, db_session):
        """Test finding operations with large dataset."""
        repo = TraceRepository()
        
        # Create many traces
        for i in range(1000):
            operation = "llm_call" if i % 3 == 0 else "data_transform"
            trace = ExecutionTrace(
                trace_id=f"perf_trace_{i}",
                operation=operation,
                start_time=datetime.now(timezone.utc) - timedelta(minutes=i),
                success=True,
            )
            db_session.add(trace)
        
        db_session.commit()
        
        # Find with limit (should be fast even with large dataset)
        traces = repo.find_by_operation("llm_call", limit=10)
        
        # Verify
        assert len(traces) == 10
        assert all(trace.operation == "llm_call" for trace in traces)
