from datetime import datetime, timezone, timedelta
from typing import Any, Dict, Optional

from app.exceptions.validation import ValidationError
from app.exceptions.base import AppException
from app.models.task import Task
from app.models.trace import ExecutionTrace
from app.observability.agent_observer import AgentObserver
from app.observability.tracer import Tracer
from app.services.llm_client import LLMClient
from app.utils.logger import get_logger

logger = get_logger(__name__)

class DecisionEngine:
    def __init__(self, 
            tracer: Optional[Tracer] = None,
            agent_observer: Optional[AgentObserver] = None,
            llm_client: Optional[LLMClient] = None):
        self.tracer = tracer or Tracer()
        self.tracer.add_quality_hook(self._on_trace_completed)
        self.observer = agent_observer or AgentObserver(self.tracer)
        self.llm_client = llm_client or LLMClient(tracer=self.tracer)
        
    def get_execution_plan(self, task_type: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        start_time = datetime.now(timezone.utc)
        original_context = self.tracer.get_current_agent_context()
        
        self.tracer.update_agent_context(
            agent_id="decision_engine",
            agent_type="planner",
            goal=f"Create execution plan for {task_type}",
            total_steps=3,
        )
        
        with self.tracer.start_trace(
            "execution_planning", 
            {"task_type": task_type, "parameters": parameters}
        ):
            try:
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
                
                if not task_type or not isinstance(task_type, str):
                    raise ValidationError(
                        message="Task type must be a non-empty string",
                        field="task_type",
                        value=task_type,
                        validation_rules={"type": "string", "required": True},
                        log=False,
                    )
                
                self.tracer.increment_agent_step()
                
                complexity = self._assess_complexity(task_type, parameters)
                
                complexity_options = [
                    {"name": "simple", "criteria": "basic operations, minimal processing"},
                    {"name": "moderate", "criteria": "requires validation, multi-step"},
                    {"name": "complex", "criteria": "multi-stage, requires analysis"},
                ]
                
                confidence = self._calculate_complexity_confidence(task_type, parameters)
                
                self.observer.record_decision_rationale(
                    decision_id=f"complexity_{task_type}_{start_time.timestamp()}",
                    agent_name="decision_engine",
                    options_considered=complexity_options,
                    chosen_option={"name": complexity, "reason": self._get_complexity_reason(task_type, parameters)},
                    rationale=f"Selected {complexity} based on task type '{task_type}' and parameter analysis",
                    confidence=confidence,
                    tradeoffs=[
                        "Simple might under-process, complex might over-process",
                        "Balance between accuracy and efficiency",
                    ],
                    metadata={"task_type": task_type, "parameter_count": len(parameters)}
                )
                
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
                plan["agent"] = "decision_engine"
                
                self.tracer.increment_agent_step()
                
                plan_options = [
                    {"name": "simple_plan", "steps": 1, "description": "Single LLM call"},
                    {"name": "moderate_plan", "steps": 2, "description": "Validation + processing"},
                    {"name": "complex_plan", "steps": 3, "description": "Multi-stage analysis"},
                ]
                
                # Confidence based on historical success for this complexity
                plan_confidence = self._get_plan_confidence(complexity)
                
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
                    confidence=plan_confidence,
                    tradeoffs=[
                        "More steps increase reliability but reduce speed",
                        "Fewer steps are faster but may miss edge cases",
                    ]
                )
                
                self._validate_plan(plan, task_type, parameters)
                
                decision_time = (datetime.now(timezone.utc) - start_time).total_seconds()
                self.tracer.record_decision("execution_plan_selected", {
                    "task_type": task_type,
                    "complexity": complexity,
                    "plan_id": plan["id"],
                    "plan_steps": len(plan.get("steps", [])),
                    "decision_time": decision_time,
                })
                
                logger.info("Execution plan selected", extra={
                    "task_type": task_type,
                    "complexity": complexity,
                    "plan_id": plan["id"],
                    "plan_steps": len(plan.get("steps", [])),
                    "decision_confidence": plan_confidence,
                })
                
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
            finally:
                self.tracer.context.agent_context = original_context
                
            return plan

    def _get_plan_confidence(self, complexity: str) -> float:
        """Compute confidence based on recent success rate for this complexity."""
        cutoff = datetime.now(timezone.utc) - timedelta(days=7)
        # Find all task_execution traces that had a decision with this complexity
        traces = ExecutionTrace.query.filter(
            ExecutionTrace.operation == 'task_execution',
            ExecutionTrace.end_time >= cutoff
        ).all()
        relevant = []
        for t in traces:
            for dec in t.decisions:
                if dec.get("type") == "execution_plan_selected" and dec.get("context", {}).get("complexity") == complexity:
                    relevant.append(t)
                    break
        if not relevant:
            return 0.7  # fallback
        success = sum(1 for t in relevant if t.success)
        return success / len(relevant)

    def _get_complexity_reason(self, task_type: str, parameters: Dict[str, Any]) -> str:
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
        if task_type in ["extract", "translate", "summarize", "classify", "analyze"]:
            return 0.9
        else:
            return 0.6

    def evaluate_condition(self, condition: str, task: Task, context: Dict[str, Any]) -> bool:
        start_time = datetime.now(timezone.utc)
        
        with self.tracer.start_trace(
            "condition_evaluation", 
            {
                "condition": condition, 
                "context_size": len(str(context)),
                "context_keys": list(context.keys()) if isinstance(context, dict) else None,
            }
        ):
            try:
                if not condition or not isinstance(condition, str):
                    raise ValidationError(
                        message="Condition must be a non-empty string",
                        field="condition",
                        value=condition,
                        validation_rules={"type": "string", "required": True},
                        log=False,
                    )
                
                if context is None:
                    raise ValidationError(
                        message="Context cannot be None",
                        field="context",
                        value=None,
                        validation_rules={"required": True, "not_null": True},
                        log=False,
                    )
                
                result = self._evaluate_condition_logic(condition, context)
                
                evaluation_time = (datetime.now(timezone.utc) - start_time).total_seconds()
                self.tracer.record_decision("condition_evaluated", {
                    "condition": condition,
                    "result": result,
                    "evaluation_time": evaluation_time,
                    "context_summary": self._summarize_context(context),
                })
                
                logger.debug("Condition evaluated", extra={
                    "condition": condition,
                    "result": result,
                    "evaluation_time": evaluation_time,
                })
                
                return {
                    "result": result,
                    "content": task.input_data,
                }
                
            except (AppException, ValidationError):
                raise
            except Exception as e:
                raise AppException(
                    message=f"Failed to evaluate condition: {str(e)}",
                    extra={
                        "condition": condition,
                        "context_type": type(context).__name__,
                        "context_size": len(str(context)),
                        "original_exception_type": type(e).__name__,
                        "original_exception": str(e),
                    }
                ) from e
    
    def _assess_complexity(self, task_type: str, parameters: Dict[str, Any]) -> str:
        try:
            prompt = f"Given task type '{task_type}' and parameters {parameters}, what is the complexity? Respond with only one word: simple, moderate, or complex."
            response = self.llm_client.process(prompt, {})
            logger.debug("Complexity assessment response", extra={"response": response})
            complexity = response.get("content", "moderate").strip().lower()
            return complexity if complexity in ["simple", "moderate", "complex"] else "moderate"
        except Exception as e:
            raise AppException(
                message=f"Failed to assess task complexity: {str(e)}",
                extra={
                    "task_type": task_type,
                    "parameters": parameters,
                    "original_exception_type": type(e).__name__,
                }
            ) from e

    def _get_simple_plan(self, task_type: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        try:
            if task_type == "summarize":
                max_length = parameters.get('max_length', 100)
                prompt = f"Summarize in about {max_length} words:"
            elif task_type == "translate":
                target_lang = parameters.get('target_language', 'English')
                prompt = f"Translate to {target_lang}:"
            else:
                prompt = f"Perform {task_type} on the input"
                
            return {
                "id": f"simple_{task_type}_{datetime.now(timezone.utc).timestamp()}",
                "complexity": "simple",
                "agent": "decision_engine",
                "reason": "Task assessed as low complexity",
                "steps": [
                    {
                        "name": f"{task_type}_execution",
                        "type": "llm_call",
                        "parameters": {
                            "prompt": prompt,
                            "llm_params": {"temperature": 0.1, "max_tokens": 500},
                        },
                    }
                ],
            }
        except Exception as e:
            raise AppException(
                message=f"Failed to generate simple plan: {str(e)}",
                extra={
                    "task_type": task_type,
                    "parameters": parameters,
                    "original_exception_type": type(e).__name__,
                    "plan_type": "simple",
                }
            ) from e

    def _get_moderate_plan(self, task_type: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        try:
            if task_type == "summarize":
                max_length = parameters.get('max_length', 100)
                format_type = parameters.get('format', 'concise')
                prompt = f"Please provide a {format_type} summary of the text in about {max_length} words or less."
            elif task_type == "classify":
                categories = parameters.get('categories', [])
                if categories:
                    prompt = f"Categorize the input into one of these categories: {', '.join(categories)}"
                else:
                    prompt = "Classify the input into appropriate categories"
            else:
                prompt = f"Process and {task_type} the input"
            
            return {
                "id": f"moderate_{task_type}_{datetime.now(timezone.utc).timestamp()}",
                "complexity": "moderate",
                "agent": "decision_engine",
                "reason": "Task requires validation and processing",
                "steps": [
                    {
                        "name": "input_validation",
                        "type": "decision_point",
                        "parameters": {"condition": "requires_validation"},
                    },
                    {
                        "name": f"{task_type}_processing",
                        "type": "llm_call",
                        "parameters": {
                            "prompt": prompt,
                            "llm_params": {"temperature": 0.3, "max_tokens": 1000},
                        },
                    },
                    {
                        "name": "result_formatting",
                        "type": "data_transform",
                        "parameters": {"transform": "summarize"},
                    },
                ],
            }
        except Exception as e:
            raise AppException(
                message=f"Failed to generate moderate plan: {str(e)}",
                extra={
                    "task_type": task_type,
                    "parameters": parameters,
                    "original_exception_type": type(e).__name__,
                    "plan_type": "moderate",
                }
            ) from e

    def _get_complex_plan(self, task_type: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        try:
            if task_type == "analyze":
                prompt1 = "Perform initial analysis of the input, identifying key themes and structure"
                prompt2 = "Provide detailed analysis with critical insights and recommendations"
            else:
                prompt1 = "Perform initial analysis of the input"
                prompt2 = f"Perform detailed {task_type} with critical analysis"
            
            return {
                "id": f"complex_{task_type}_{datetime.now(timezone.utc).timestamp()}",
                "complexity": "complex",
                "agent": "decision_engine",
                "reason": "Task requires multi-stage processing and validation",
                "steps": [
                    {
                        "name": "sanity_check",
                        "type": "decision_point",
                        "parameters": {
                            "condition": "has_multiple_items",
                            "continue_on_failure": True,
                        },
                    },
                    {
                        "name": "security_check",
                        "type": "decision_point",
                        "parameters": {"condition": "contains_sensitive_data"},
                    },
                    {
                        "name": "initial_analysis",
                        "type": "llm_call",
                        "parameters": {
                            "prompt": prompt1,
                            "llm_params": {"temperature": 0.2, "max_tokens": 500},
                        },
                    },
                    {
                        "name": "detailed_processing",
                        "type": "llm_call",
                        "parameters": {
                            "prompt": prompt2,
                            "llm_params": {"temperature": 0.5, "max_tokens": 2000},
                        },
                    },
                    {
                        "name": "result_synthesis",
                        "type": "data_transform",
                        "parameters": {"transform": "summarize"},
                    },
                ],
            }
        except Exception as e:
            raise AppException(
                message=f"Failed to generate complex plan: {str(e)}",
                extra={
                    "task_type": task_type,
                    "parameters": parameters,
                    "original_exception_type": type(e).__name__,
                    "plan_type": "complex",
                }
            ) from e

    def _evaluate_condition_logic(self, condition: str, context: Dict[str, Any]) -> bool:
        try:
            if not condition or not isinstance(condition, str):
                raise ValidationError(
                    message="Condition must be a non-empty string",
                    field="condition",
                    value=condition
                )
            result = False
            
            if condition == "requires_validation":
                result = len(str(context)) > 100
            elif condition == "has_multiple_items":
                result = isinstance(context, list) and len(context) > 1
            elif condition == "contains_sensitive_data":
                sensitive_keywords = ["password", "secret", "key", "token"]
                context_str = str(context).lower()
                result = any(keyword in context_str for keyword in sensitive_keywords)
            else:
                raise ValidationError(
                    message=f"Unknown condition type: '{condition}'",
                    field="condition",
                    value=condition,
                    validation_rules={
                        "allowed_values": [
                            "requires_validation", 
                            "has_multiple_items", 
                            "contains_sensitive_data"
                        ]
                    },
                    extra={
                        "context_summary": self._summarize_context(context),
                    }
                )
            
            return result
            
        except ValidationError:
            raise
        except Exception as e:
            raise AppException(
                message=f"Condition evaluation logic failed: {str(e)}",
                extra={
                    "condition": condition,
                    "context_type": type(context).__name__,
                    "original_exception_type": type(e).__name__,
                }
            ) from e

    def _validate_plan(self, plan: Dict[str, Any], task_type: str, parameters: Dict[str, Any]) -> None:
        try:
            if not plan.get("id"):
                raise ValidationError(
                    message="Generated plan missing ID",
                    field="plan.id",
                    value=None,
                    validation_rules={"required": True},
                    extra={
                        "plan": plan,
                        "task_type": task_type,
                        "parameters": parameters,
                    }
                )
            
            if not plan.get("steps") or not isinstance(plan["steps"], list):
                raise ValidationError(
                    message="Generated plan has no steps or steps is not a list",
                    field="plan.steps",
                    value=plan.get("steps"),
                    validation_rules={"type": "list", "required": True},
                    extra={
                        "plan_id": plan.get("id"),
                        "task_type": task_type,
                        "steps_type": type(plan.get("steps")).__name__,
                    }
                )
            
            for i, step in enumerate(plan["steps"]):
                if not step.get("name"):
                    raise ValidationError(
                        message=f"Step {i} missing 'name' field",
                        field=f"plan.steps[{i}].name",
                        value=None,
                        validation_rules={"required": True},
                        extra={
                            "plan_id": plan.get("id"),
                            "step_index": i,
                            "step": step,
                        }
                    )
                
                if not step.get("type"):
                    raise ValidationError(
                        message=f"Step {i} missing 'type' field",
                        field=f"plan.steps[{i}].type",
                        value=None,
                        validation_rules={"required": True},
                        extra={
                            "plan_id": plan.get("id"),
                            "step_index": i,
                            "step": step,
                        }
                    )
            
            logger.debug("Plan validation passed", extra={
                "plan_id": plan.get("id"),
                "step_count": len(plan["steps"]),
                "task_type": task_type,
            })
            
        except ValidationError:
            raise
        except Exception as e:
            raise AppException(
                message=f"Plan validation failed: {str(e)}",
                extra={
                    "plan_id": plan.get("id"),
                    "task_type": task_type,
                    "original_exception_type": type(e).__name__,
                }
            ) from e
    
    def _summarize_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        try:
            if not context:
                return {"empty": True}
            
            if isinstance(context, dict):
                return {
                    "key_count": len(context),
                    "keys": list(context.keys())[:5],
                    "has_nested": any(isinstance(v, (dict, list)) for v in context.values()),
                }
            elif isinstance(context, list):
                return {
                    "item_count": len(context),
                    "first_item_type": type(context[0]).__name__ if context else None,
                }
            else:
                return {
                    "type": type(context).__name__,
                    "string_length": len(str(context)),
                }
        except Exception:
            return {"summary_error": True}

    def _on_trace_completed(self, trace_data):
        if trace_data.get("success") and trace_data.get("quality_metrics", {}).get("composite_quality_score", 0) > 0.8:
            self.observer.detect_behavior_pattern(
                agent_name=trace_data.get("agent_context", {}).get("agent_id", "unknown"),
                behavior_type="high_quality_execution",
                pattern_data={
                    "operation": trace_data.get("operation"),
                    "quality_score": trace_data["quality_metrics"]["composite_quality_score"],
                    "duration": trace_data.get("duration_ms", 0)
                }
            )
