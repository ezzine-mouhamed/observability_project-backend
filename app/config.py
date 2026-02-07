import os


class Config:
    SECRET_KEY = os.environ.get("SECRET_KEY") or "dev-secret-key-change-in-production"
    SQLALCHEMY_DATABASE_URI = (
        os.environ.get("DATABASE_URL") or "sqlite:///observability.db"
    )
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    TRACING_ENABLED = True
    METRICS_ENABLED = True
    LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")

    DEFAULT_TASK_TIMEOUT = 60
    MAX_RETRY_ATTEMPTS = 3
    LLM_TIMEOUT_SECONDS = 10

    DEBUG = os.environ.get("FLASK_DEBUG", "False").lower() == "true"

    OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_DEFAULT_MODEL = os.environ.get("OLLAMA_DEFAULT_MODEL", "deepseek-coder")
    OLLAMA_REQUEST_TIMEOUT = 60


class DevelopmentConfig(Config):
    DEBUG = True


class TestingConfig(Config):
    TESTING = True
    SQLALCHEMY_DATABASE_URI = "sqlite:///:memory:"
    TRACING_ENABLED = False


class ProductionConfig(Config):
    DEBUG = False
    LOG_LEVEL = "WARNING"


config_by_name = {
    "development": DevelopmentConfig,
    "testing": TestingConfig,
    "production": ProductionConfig,
    "default": DevelopmentConfig,
}
