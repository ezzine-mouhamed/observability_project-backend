from flask import Flask

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
    
    # Register blueprints
    app.register_blueprint(task_bp)
    
    # Register observability routes
    app.register_blueprint(observability_bp)
    
    register_error_handlers(app)

    return app