from flask import Blueprint, jsonify, request
from pydantic import ValidationError

from app.schemas.task_schema import TaskCreate, TaskResponse
from app.services.task_service import TaskService
from app.utils.logger import get_logger

bp = Blueprint("api", __name__, url_prefix="/api/v1")
logger = get_logger(__name__)


@bp.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "healthy", "service": "agent-observability"})


@bp.route("/tasks", methods=["POST"])
def create_task():
    try:
        if not request.is_json:
            return jsonify({"error": "Request must be JSON"}), 400
        
        request_data = request.get_json()
        if request_data is None:
            return jsonify({"error": "Invalid JSON"}), 400
        
        # Validate request data
        task_data = TaskCreate(**request_data)
        
    except ValidationError as e:
        errors = []
        for error in e.errors():
            errors.append({
                "field": ".".join(str(loc) for loc in error.get("loc", [])),
                "message": error.get("msg"),
                "type": error.get("type"),
            })
        return jsonify({"error": "Validation failed", "details": errors}), 400
        
    except Exception as e:
        logger.error("Unexpected error during request validation", extra={"error": str(e)})
        return jsonify({"error": "Invalid request format"}), 400
    
    try:
        task_service = TaskService()
        task_result = task_service.execute_task(task_data)
        
        # Use model_dump() instead of dict() for Pydantic v2
        response = TaskResponse.model_validate(task_result.task)
        return jsonify(response.model_dump()), 201
        
    except Exception as e:
        logger.error(
            "Task execution failed",
            extra={"error": str(e), "task_type": task_data.task_type},
        )
        return jsonify({"error": "Internal server error"}), 500


@bp.route("/tasks/<int:task_id>", methods=["GET"])
def get_task(task_id):
    try:
        task_service = TaskService()
        task = task_service.get_task_by_id(task_id)
        
        if not task:
            return jsonify({"error": "Task not found"}), 404
        
        response = TaskResponse.model_validate(task)
        return jsonify(response.model_dump()), 200
        
    except Exception as e:
        logger.error(
            "Failed to retrieve task", 
            extra={"task_id": task_id, "error": str(e)}
        )
        return jsonify({"error": "Internal server error"}), 500


@bp.route("/tasks/<int:task_id>/traces", methods=["GET"])
def get_task_traces(task_id):
    try:
        task_service = TaskService()
        traces = task_service.get_traces_for_task(task_id)
        
        if traces is None:
            return jsonify({"error": "Task not found"}), 404
        
        return jsonify([trace.to_dict() for trace in traces]), 200
        
    except Exception as e:
        logger.error(
            "Failed to retrieve traces", 
            extra={"task_id": task_id, "error": str(e)}
        )
        return jsonify({"error": "Internal server error"}), 500


@bp.route("/metrics/summary", methods=["GET"])
def get_metrics_summary():
    try:
        task_service = TaskService()
        summary = task_service.get_metrics_summary()
        return jsonify(summary), 200
        
    except Exception as e:
        logger.error("Failed to retrieve metrics", extra={"error": str(e)})
        return jsonify({"error": "Internal server error"}), 500
