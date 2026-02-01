from datetime import datetime, timezone
from typing import Any, Dict

from app.exceptions.base import AppException
from app.exceptions.validation import ValidationError
from app.observability.tracer import Tracer
from app.utils.logger import get_logger

logger = get_logger(__name__)


class DecisionEngine:
    def __init__(self):
        self.tracer = Tracer()
    
    def get_execution_plan(
        self, task_type: str, parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Determine the best execution plan for a given task."""
        start_time = datetime.now(timezone.utc)
        
        with self.tracer.start_trace(
            "execution_planning", 
            {"task_type": task_type, "parameters": parameters}
        ):
            try:
                # Validate inputs using ValidationError
                if not task_type or not isinstance(task_type, str):
                    raise ValidationError(
                        message="Task type must be a non-empty string",
                        field="task_type",
                        value=task_type,
                        validation_rules={"type": "string", "required": True},
                        log=False,
                    )
                
                # Analyze task complexity
                complexity = self._assess_complexity(task_type, parameters)
                
                # Select appropriate plan
                if complexity == "simple":
                    plan = self._get_simple_plan(task_type, parameters)
                elif complexity == "moderate":
                    plan = self._get_moderate_plan(task_type, parameters)
                elif complexity == "complex":
                    plan = self._get_complex_plan(task_type, parameters)
                else:
                    # This should never happen, but handle it
                    raise AppException(
                        message=f"Unknown complexity level returned: {complexity}",
                        extra={
                            "task_type": task_type,
                            "parameters": parameters,
                            "complexity": complexity,
                        }
                    )
                
                # Validate the plan
                self._validate_plan(plan, task_type, parameters)
                
                # Record decision
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
                })
                
                return plan
                
            except (AppException, ValidationError):
                # Re-raise our custom exceptions (they already log themselves)
                raise
                
            except Exception as e:
                # Wrap other exceptions in AppException
                raise AppException(
                    message=f"Failed to create execution plan: {str(e)}",
                    extra={
                        "task_type": task_type,
                        "parameters": parameters,
                        "original_exception_type": type(e).__name__,
                        "original_exception": str(e),
                    }
                ) from e
    
    def evaluate_condition(self, condition: str, context: Dict[str, Any]) -> bool:
        """Evaluate a condition in the given context."""
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
                # Validate inputs using ValidationError
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
                
                # Evaluate condition
                result = self._evaluate_condition_logic(condition, context)
                
                # Record decision
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
                
                return result
                
            except (AppException, ValidationError):
                # Re-raise our custom exceptions
                raise
                
            except Exception as e:
                # Wrap other exceptions in AppException
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
        """Assess the complexity of a task."""
        try:
            # Simple complexity assessment based on task type and parameters
            if task_type in ["extract", "translate"]:
                return "simple"
            elif task_type in ["summarize", "classify"]:
                max_length = parameters.get("max_length", 0)
                if max_length > 1000:
                    return "complex"
                return "moderate"
            elif task_type == "analyze":
                return "complex"
            else:
                return "moderate"
                
        except Exception as e:
            raise AppException(
                message=f"Failed to assess task complexity: {str(e)}",
                extra={
                    "task_type": task_type,
                    "parameters": parameters,
                    "original_exception_type": type(e).__name__,
                }
            ) from e
    
    def _get_simple_plan(
        self, task_type: str, parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Get simple execution plan."""
        try:
            return {
                "id": f"simple_{task_type}_{datetime.now(timezone.utc).timestamp()}",
                "complexity": "simple",
                "reason": "Task assessed as low complexity",
                "steps": [
                    {
                        "name": f"{task_type}_execution",
                        "type": "llm_call",
                        "parameters": {
                            "prompt": f"Perform {task_type} on the input",
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
    
    def _get_moderate_plan(
        self, task_type: str, parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Get moderate execution plan."""
        try:
            return {
                "id": f"moderate_{task_type}_{datetime.now(timezone.utc).timestamp()}",
                "complexity": "moderate",
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
                            "prompt": f"Analyze and {task_type} the input",
                            "llm_params": {"temperature": 0.3, "max_tokens": 1000},
                        },
                    },
                    {
                        "name": "result_formatting",
                        "type": "data_transform",
                        "parameters": {"transform": "extract_key_points"},
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
    
    def _get_complex_plan(
        self, task_type: str, parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Get complex execution plan."""
        try:
            return {
                "id": f"complex_{task_type}_{datetime.now(timezone.utc).timestamp()}",
                "complexity": "complex",
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
                            "prompt": "Perform initial analysis of the input",
                            "llm_params": {"temperature": 0.2, "max_tokens": 500},
                        },
                    },
                    {
                        "name": "detailed_processing",
                        "type": "llm_call",
                        "parameters": {
                            "prompt": f"Perform detailed {task_type} with critical analysis",
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
        """Internal logic for condition evaluation."""
        try:
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
                # For unknown conditions, use ValidationError
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
            raise  # Re-raise without wrapping
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
        """Validate the generated execution plan."""
        try:
            # Basic validation
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
            
            # Validate each step
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
        """Create a safe summary of context for logging."""
        try:
            if not context:
                return {"empty": True}
            
            if isinstance(context, dict):
                return {
                    "key_count": len(context),
                    "keys": list(context.keys())[:5],  # First 5 keys only
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