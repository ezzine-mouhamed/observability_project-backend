from flask import Blueprint, jsonify, request
from pydantic import ValidationError as PydanticValidationError

from app.schemas.task_schema import TaskCreate, TaskResponse
from app.services.task_service import TaskService
from app.utils.logger import get_logger

task_bp = Blueprint("api", __name__, url_prefix="/api/v1")
logger = get_logger(__name__)


@task_bp.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "healthy", "service": "agent-observability"})


@task_bp.route("/tasks", methods=["POST"])
def create_task():
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    request_data = request.get_json()
    if request_data is None:
        return jsonify({"error": "Invalid JSON"}), 400

    task_data = TaskCreate(**request_data)

    task_service = TaskService()
    task_result = task_service.execute_task(task_data)

    response = TaskResponse.model_validate(task_result.task)
    return jsonify(response.model_dump()), 201


@task_bp.route("/tasks/<int:task_id>", methods=["GET"])
def get_task(task_id):
    task_service = TaskService()
    task = task_service.get_task_by_id(task_id)

    if not task:
        return jsonify({"error": "Task not found"}), 404

    response = TaskResponse.model_validate(task)
    return jsonify(response.model_dump()), 200


@task_bp.route("/tasks/<int:task_id>/traces", methods=["GET"])
def get_task_traces(task_id):
    task_service = TaskService()
    traces = task_service.get_traces_for_task(task_id)

    if traces is None:
        return jsonify({"error": "Task not found"}), 404

    return jsonify([trace.to_dict() for trace in traces]), 200
