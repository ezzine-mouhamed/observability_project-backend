import logging
import os
import sys

from app import create_app
from app.config import DevelopmentConfig, ProductionConfig
from app.exceptions.base import AppException
from app.exceptions.keys import RestErrorKey
from app.exceptions.rest import RestAPIError
from app.utils.logger import get_logger

logger = get_logger(__name__)

# Determine environment
env = os.getenv("FLASK_ENV", "development")
config = DevelopmentConfig if env == "development" else ProductionConfig

app = create_app(config)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = app.config.get("DEBUG", False)

    logger.info(f"Starting Agent Observability Service on port {port}")
    logger.info(f"Environment: {env}")
    logger.info(f"Debug mode: {debug}")
    logger.info(f"Database: {app.config.get('SQLALCHEMY_DATABASE_URI')}")

    app.run(host="0.0.0.0", port=port, debug=debug)
