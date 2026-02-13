from flask import Flask
from flask_cors import CORS

from app.config import Config
from app.extensions import db, migrate
from app.utils.logger import setup_logging
from app.api.error_handlers import register_error_handlers
from app.api.routes.tasks import task_bp
from app.api.routes.observability import observability_bp


def create_app(config_class=Config):
    """Create and configure the Flask application."""
    app = Flask(__name__)
    app.config.from_object(config_class)
    
    # Initialize extensions
    db.init_app(app)
    migrate.init_app(app, db)
    
    # Setup logging
    setup_logging()
    
    CORS(app, resources={
        r"/api/*": {
            "origins": [
                "http://localhost:3000"
            ],
            "allow_headers": ["Content-Type", "Authorization"],
            "expose_headers": ["Content-Type"],
            "methods": ["GET", "POST", "PUT", "DELETE"],
            "supports_credentials": True
        }
    })
    
    # Register blueprints
    app.register_blueprint(task_bp)
    
    # Register observability routes
    app.register_blueprint(observability_bp)
    
    register_error_handlers(app)

    return app
