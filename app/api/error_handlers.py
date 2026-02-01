from flask import jsonify, current_app
from pydantic import ValidationError as PydanticValidationError

from app.exceptions.base import AppException
from app.exceptions.validation import ValidationError
from app.exceptions.rest import RestAPIError
from app.utils.logger import get_logger

logger = get_logger(__name__)


def register_error_handlers(app):
    """Register centralized error handlers for the Flask app."""
    
    @app.errorhandler(PydanticValidationError)
    def handle_pydantic_validation_error(error: PydanticValidationError):
        """Handle Pydantic validation errors."""
        errors = []
        for e in error.errors():
            errors.append({
                "field": ".".join(str(loc) for loc in e.get("loc", [])),
                "message": e.get("msg"),
                "type": e.get("type"),
            })
        
        logger.warning("Pydantic validation failed", extra={"validation_errors": errors})
        
        return jsonify({
            "error": "Validation failed",
            "message": "Invalid request data",
            "validation_errors": errors,
            "error_code": "VALIDATION_ERROR"
        }), 400
    
    @app.errorhandler(ValidationError)
    def handle_validation_error(error: ValidationError):
        """Handle our custom validation errors."""
        # ValidationError already logs itself, so we don't need additional logging
        
        response_data = {
            "error": "Validation failed",
            "message": error.message,
            "error_code": "VALIDATION_ERROR"
        }
        
        # Add validation details if available
        if error.extra and "validation_details" in error.extra:
            response_data["validation_errors"] = error.extra["validation_details"]
        
        return jsonify(response_data), 400
    
    @app.errorhandler(AppException)
    def handle_app_exception(error: AppException):
        """Handle all AppException-based errors."""
        # AppException already logs itself
        
        # For RestAPIError, use its status_code
        if isinstance(error, RestAPIError):
            status_code = error.status_code or 500
            response = error.response or {
                "error": "API Error",
                "message": error.message,
                "error_code": "API_ERROR"
            }
            
            # If response is a string, wrap it
            if isinstance(response, str):
                response = {"message": response}
                
            return jsonify(response), status_code
        
        # For other AppException subclasses
        return jsonify({
            "error": "Application Error",
            "message": "An internal error occurred",
            "error_code": "INTERNAL_ERROR"
        }), 500
    
    @app.errorhandler(Exception)
    def handle_generic_exception(error: Exception):
        """Catch-all for any unhandled exceptions."""
        logger.error(
            "Unhandled exception in request",
            extra={
                "exception_type": type(error).__name__,
                "exception_message": str(error),
            },
            exc_info=True
        )
        
        # In production, don't expose internal error details
        if current_app.config.get("DEBUG", False):
            message = str(error)
        else:
            message = "Internal server error"
        
        return jsonify({
            "error": "Internal Server Error",
            "message": message,
            "error_code": "INTERNAL_SERVER_ERROR"
        }), 500