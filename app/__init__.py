from flask import Flask

from app.config import Config
from app.extensions import db, migrate
from app.utils.logger import setup_logging
from app.api.error_handlers import register_error_handlers


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
    from app.api.routes import bp as api_bp
    app.register_blueprint(api_bp)
    
    register_error_handlers(app)
    
    # Create database tables (in development)
    if app.config.get("DEBUG", False):
        with app.app_context():
            db.create_all()
    
    return app
