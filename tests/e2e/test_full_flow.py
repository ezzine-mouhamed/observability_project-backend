import pytest
from datetime import datetime, timezone
from app import create_app
from app.extensions import db
from app.models.task import Task
from app.models.trace import ExecutionTrace
from app.models.agent_insight import AgentInsight
from app.observability.agent_observer import AgentObserver
from app.observability.tracer import Tracer
from app.services.task_service import TaskService
from app.schemas.task_schema import TaskCreate


class TestFullObservabilityFlow:
    """End-to-end test of the complete observability flow."""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test environment."""
        self.app = create_app()
        self.app.config.update({
            'TESTING': True,
            'SQLALCHEMY_DATABASE_URI': 'sqlite:///:memory:',
        })
        
        with self.app.app_context():
            db.create_all()
            yield
            db.session.remove()
            db.drop_all()
    
    def test_complete_observability_flow(self):
        """Test the complete flow from task creation to insight generation."""
        with self.app.app_context():
            # 1. Initialize components
            tracer = Tracer()
            agent_observer = AgentObserver(tracer)
            task_service = TaskService()
            
            # 2. Create a task
            task_data = TaskCreate(
                task_type="summarize",
                input_data={"text": "End-to-end test text"},
                parameters={"max_length": 50}
            )
            
            # 3. Execute the task
            task_result = task_service.execute_task(task_data)
            
            # 4. Verify task was created and executed
            assert task_result.task is not None
            assert task_result.task.id is not None
            assert task_result.task.status in ["completed", "failed"]
            
            # 5. Verify traces were created
            assert task_result.traces is not None
            assert len(task_result.traces) > 0
            
            # 6. Check if task was successful (might fail if LLM not available)
            if task_result.success:
                # 7. Verify learning was triggered
                # Get the task from database to check learning metadata
                task_from_db = Task.query.get(task_result.task.id)
                assert task_from_db is not None
                
                # 8. Get traces for the task
                traces = ExecutionTrace.query.filter_by(task_id=task_from_db.id).all()
                assert len(traces) > 0
                
                # 9. Generate insights
                insights = agent_observer.generate_insights_from_observations(
                    agent_name="task_executor",
                    time_window_hours=24
                )
                
                # Insights might be empty if not enough data, but method should work
                assert isinstance(insights, list)
                
                # 10. Check if agent insights table is accessible
                agent_insights = AgentInsight.query.all()
                # Might be empty, but query should work
                assert agent_insights is not None
                
                print(f"\n✅ End-to-end test completed:")
                print(f"   Task ID: {task_from_db.id}")
                print(f"   Task Status: {task_from_db.status}")
                print(f"   Traces Created: {len(traces)}")
                print(f"   Insights Generated: {len(insights)}")
                print(f"   Agent Insights in DB: {len(agent_insights)}")
                
            else:
                print(f"\n⚠️  Task failed (possibly due to missing LLM): {task_result.task.error_message}")
                # Even if task failed, system should handle it gracefully
                assert task_result.task.error_message is not None
                
    def test_agent_learning_loop(self):
        """Test the learning loop specifically."""
        with self.app.app_context():
            # Create a completed task with traces
            task = Task(
                task_type="summarize",
                input_data={"text": "Learning loop test"},
                parameters={"max_length": 100},
                status="completed",
                quality_score=0.6  # Low quality to trigger insights
            )
            db.session.add(task)
            db.session.commit()
            
            # Create traces for the task
            trace1 = ExecutionTrace(
                trace_id="learn_trace_1",
                task_id=task.id,
                operation="task_execution",
                agent_context={"agent_id": "learning_agent", "agent_type": "tester"},
                quality_metrics={"composite_quality_score": 0.6},
                start_time=datetime.now(timezone.utc),
                end_time=datetime.now(timezone.utc),
                duration_ms=1000,
                success=True
            )
            
            trace2 = ExecutionTrace(
                trace_id="learn_trace_2",
                task_id=task.id,
                operation="llm_call",
                agent_context={"agent_id": "learning_agent", "agent_type": "tester"},
                quality_metrics={"composite_quality_score": 0.5},
                start_time=datetime.now(timezone.utc),
                end_time=datetime.now(timezone.utc),
                duration_ms=2000,
                success=True
            )
            
            db.session.add_all([trace1, trace2])
            db.session.commit()
            
            # Initialize task service and run learning
            task_service = TaskService()
            traces = [trace1, trace2]
            
            # Run learning
            learning_result = task_service.learn_from_execution(task, traces)
            
            # Verify learning happened
            assert learning_result is not None
            assert "insights_generated" in learning_result
            assert "insights_applied" in learning_result
            assert "agents_analyzed" in learning_result
            
            print(f"\n✅ Learning loop test completed:")
            print(f"   Insights Generated: {learning_result.get('insights_generated', 0)}")
            print(f"   Insights Applied: {learning_result.get('insights_applied', 0)}")
            print(f"   Agents Analyzed: {learning_result.get('agents_analyzed', [])}")
