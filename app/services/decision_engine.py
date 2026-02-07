from datetime import datetime, timezone
from typing import Any, Dict

from pydantic_core import ValidationError
from app.exceptions.base import AppException
from app.observability.agent_observer import AgentObserver, ObservationType
from app.observability.tracer import Tracer
from app.utils import logger

class DecisionEngine:
    def __init__(self):
        self.tracer = Tracer()
        self.observer = AgentObserver(self.tracer)  # NEW: Agent observer
        
    def get_execution_plan(self, task_type: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Determine the best execution plan for a given task."""
        start_time = datetime.now(timezone.utc)
        
        # Set agent context
        self.tracer.update_agent_context(
            agent_id="decision_engine",
            agent_type="planner",
            goal=f"Create execution plan for {task_type}",
            total_steps=3,  # complexity assessment, plan selection, validation
        )
        
        with self.tracer.start_trace(
            "execution_planning", 
            {"task_type": task_type, "parameters": parameters}
        ):
            try:
                # NEW: Record thought process
                self.observer.record_thought_process(
                    agent_name="decision_engine",
                    input_data={"task_type": task_type, "parameters": parameters},
                    thought_chain=[
                        f"Starting execution plan for {task_type}",
                        f"Parameters: {len(parameters)} parameters provided",
                        "Assessing task complexity...",
                    ],
                    final_thought="Beginning complexity assessment phase",
                    metadata={"phase": "initialization"}
                )
                
                # Validate inputs
                if not task_type or not isinstance(task_type, str):
                    raise ValidationError(
                        message="Task type must be a non-empty string",
                        field="task_type",
                        value=task_type,
                        validation_rules={"type": "string", "required": True},
                        log=False,
                    )
                
                # Increment agent step
                self.tracer.increment_agent_step()
                
                # Analyze task complexity
                complexity = self._assess_complexity(task_type, parameters)
                
                # NEW: Record decision rationale
                complexity_options = [
                    {"name": "simple", "criteria": "basic operations, minimal processing"},
                    {"name": "moderate", "criteria": "requires validation, multi-step"},
                    {"name": "complex", "criteria": "multi-stage, requires analysis"},
                ]
                
                self.observer.record_decision_rationale(
                    decision_id=f"complexity_{task_type}_{start_time.timestamp()}",
                    agent_name="decision_engine",
                    options_considered=complexity_options,
                    chosen_option={"name": complexity, "reason": self._get_complexity_reason(task_type, parameters)},
                    rationale=f"Selected {complexity} based on task type '{task_type}' and parameter analysis",
                    confidence=self._calculate_complexity_confidence(task_type, parameters),
                    tradeoffs=[
                        f"Simple might under-process, complex might over-process",
                        f"Balance between accuracy and efficiency",
                    ],
                    metadata={"task_type": task_type, "parameter_count": len(parameters)}
                )
                
                # Select appropriate plan
                if complexity == "simple":
                    plan = self._get_simple_plan(task_type, parameters)
                elif complexity == "moderate":
                    plan = self._get_moderate_plan(task_type, parameters)
                elif complexity == "complex":
                    plan = self._get_complex_plan(task_type, parameters)
                else:
                    raise AppException(
                        message=f"Unknown complexity level returned: {complexity}",
                        extra={
                            "task_type": task_type,
                            "parameters": parameters,
                            "complexity": complexity,
                        }
                    )
                
                # NEW: Record plan selection rationale
                self.tracer.increment_agent_step()
                
                plan_options = [
                    {"name": "simple_plan", "steps": 1, "description": "Single LLM call"},
                    {"name": "moderate_plan", "steps": 2, "description": "Validation + processing"},
                    {"name": "complex_plan", "steps": 3, "description": "Multi-stage analysis"},
                ]
                
                self.observer.record_decision_rationale(
                    decision_id=f"plan_selection_{plan['id']}",
                    agent_name="decision_engine",
                    options_considered=plan_options,
                    chosen_option={
                        "name": f"{complexity}_plan",
                        "id": plan["id"],
                        "steps": len(plan.get("steps", [])),
                        "reason": plan.get("reason", ""),
                    },
                    rationale=f"Selected {complexity} plan with {len(plan.get('steps', []))} steps based on complexity assessment",
                    confidence=0.8,  # High confidence in plan selection
                    tradeoffs=[
                        f"More steps increase reliability but reduce speed",
                        f"Fewer steps are faster but may miss edge cases",
                    ]
                )
                
                # Validate the plan
                self._validate_plan(plan, task_type, parameters)
                
                # NEW: Record final self-evaluation
                self.tracer.increment_agent_step()
                
                self.observer.record_self_evaluation(
                    agent_name="decision_engine",
                    task_id=plan["id"],
                    evaluation_criteria={
                        "completeness": 0.3,
                        "appropriateness": 0.4,
                        "efficiency": 0.3,
                    },
                    self_scores={
                        "completeness": 0.9,
                        "appropriateness": 0.8,
                        "efficiency": 0.7,
                    },
                    justification=f"Plan covers all required aspects, appropriate for {complexity} task, could be more efficient",
                    improvements_suggested=[
                        "Cache similar plan selections",
                        "Optimize step ordering based on historical data",
                    ],
                    metadata={"complexity": complexity, "step_count": len(plan.get("steps", []))}
                )
                
                # Record decision
                decision_time = (datetime.now(timezone.utc) - start_time).total_seconds()
                self.tracer.record_decision("execution_plan_selected", {
                    "task_type": task_type,
                    "complexity": complexity,
                    "plan_id": plan["id"],
                    "plan_steps": len(plan.get("steps", [])),
                    "decision_time": decision_time,
                    "quality_score": 0.8,  # Self-assessed quality
                })
                
                logger.info("Execution plan selected", extra={
                    "task_type": task_type,
                    "complexity": complexity,
                    "plan_id": plan["id"],
                    "plan_steps": len(plan.get("steps", [])),
                    "decision_confidence": 0.8,
                })
                
                return plan
                
            except (AppException, ValidationError):
                raise
                
            except Exception as e:
                raise AppException(
                    message=f"Failed to create execution plan: {str(e)}",
                    extra={
                        "task_type": task_type,
                        "parameters": parameters,
                        "original_exception_type": type(e).__name__,
                        "original_exception": str(e),
                    }
                ) from e
    
    def _get_complexity_reason(self, task_type: str, parameters: Dict[str, Any]) -> str:
        """Generate a reason for complexity assessment."""
        if task_type in ["extract", "translate"]:
            return "Simple single-operation task"
        elif task_type in ["summarize", "classify"]:
            max_length = parameters.get("max_length", 0)
            return f"Moderate task with max length {max_length}"
        elif task_type == "analyze":
            return "Complex multi-stage analysis task"
        else:
            return "Moderate default complexity"
    
    def _calculate_complexity_confidence(self, task_type: str, parameters: Dict[str, Any]) -> float:
        """Calculate confidence in complexity assessment."""
        # Simple heuristic
        if task_type in ["extract", "translate", "summarize", "classify", "analyze"]:
            return 0.9  # High confidence for known types
        else:
            return 0.6  # Lower confidence for unknown types
