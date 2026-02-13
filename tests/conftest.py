import pytest
import tempfile
import os
from datetime import datetime, timezone
from unittest.mock import Mock, patch, MagicMock

from app import create_app
from app.extensions import db
from app.models.task import Task
from app.models.trace import ExecutionTrace
from app.models.agent_insight import AgentInsight
from app.observability.agent_observer import AgentObserver
from app.observability.tracer import Tracer

@pytest.fixture(scope='session')
def app():
    """Create and configure a Flask app for testing."""
    # Create a temporary database file
    db_fd, db_path = tempfile.mkstemp()
    
    app = create_app()
    app.config.update({
        'TESTING': True,
        'SQLALCHEMY_DATABASE_URI': f'sqlite:///{db_path}',
        'SQLALCHEMY_TRACK_MODIFICATIONS': False,
    })
    
    with app.app_context():
        db.create_all()
        yield app
    
    # Cleanup
    os.close(db_fd)
    os.unlink(db_path)

@pytest.fixture
def client(app):
    """Test client for the app."""
    return app.test_client()

@pytest.fixture
def runner(app):
    """CLI test runner."""
    return app.test_cli_runner()

@pytest.fixture
def db_session(app):
    """Database session for testing."""
    with app.app_context():
        yield db.session
        db.session.remove()

@pytest.fixture
def sample_task(db_session):
    """Create a sample task for testing."""
    task = Task(
        task_type="summarize",
        input_data={"text": "Test text for observability testing"},
        parameters={"max_length": 100},
        status="completed",
        quality_score=0.85,
        execution_time_ms=1234,
    )
    db_session.add(task)
    db_session.commit()
    return task

@pytest.fixture
def sample_trace(db_session, sample_task):
    """Create a sample trace for testing."""
    trace = ExecutionTrace(
        trace_id="test_trace_123",
        operation="task_execution",
        task_id=sample_task.id,
        context={"test": "data"},
        decisions=[{"type": "test_decision", "timestamp": datetime.now(timezone.utc).isoformat()}],
        events=[{"type": "test_event", "timestamp": datetime.now(timezone.utc).isoformat()}],
        quality_metrics={"composite_quality_score": 0.9, "success_factor": 1.0},
        agent_context={"agent_id": "test_agent", "agent_type": "task_processor"},
        start_time=datetime.now(timezone.utc),
        end_time=datetime.now(timezone.utc),
        duration_ms=1500,
        success=True,
    )
    db_session.add(trace)
    db_session.commit()
    return trace

@pytest.fixture
def agent_observer():
    """Create an AgentObserver instance for testing."""
    mock_tracer = Mock(spec=Tracer)
    observer = AgentObserver(tracer=mock_tracer)
    return observer

@pytest.fixture
def mock_llm():
    """Mock LLM client for testing."""
    with patch('app.services.llm_client.LLMClient') as mock:
        mock_instance = mock.return_value
        mock_instance.process.return_value = {
            "content": "Mocked LLM response",
            "tokens_used": 100,
            "latency_ms": 500,
        }
        yield mock_instance

@pytest.fixture
def mock_decision_engine():
    """Mock DecisionEngine for testing."""
    with patch('app.services.decision_engine.DecisionEngine') as mock:
        mock_instance = mock.return_value
        mock_instance.get_execution_plan.return_value = {
            "id": "test_plan_123",
            "complexity": "simple",
            "steps": [
                {"name": "test_step", "type": "llm_call", "parameters": {"prompt": "Test"}}
            ]
        }
        mock_instance.evaluate_condition.return_value = {
            "result": True,
            "content": {"test": "data"}
        }
        yield mock_instance
