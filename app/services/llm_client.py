import time
from typing import Any, Dict

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from app.config import Config
from app.exceptions.base import AppException  # Import AppException
from app.observability.tracer import Tracer
from app.utils.logger import get_logger

logger = get_logger(__name__)


class LLMClient:
    def __init__(self):
        self.tracer = Tracer()
        self.config = Config()
        self.session = self._create_retry_session()
        self.request_timeout = self.config.OLLAMA_REQUEST_TIMEOUT

    def _create_retry_session(self) -> requests.Session:
        """Create a session with retry strategy for resilience"""
        session = requests.Session()

        retry_strategy = Retry(
            total=3,  # Maximum number of retries
            backoff_factor=1,  # Exponential backoff: 1, 2, 4 seconds
            status_forcelist=[429, 500, 502, 503, 504],  # Retry on these status codes
            allowed_methods=["GET", "POST"],  # Only retry safe methods
            raise_on_status=False,  # Don't raise exception, just retry
        )

        adapter = HTTPAdapter(
            max_retries=retry_strategy,
            pool_connections=10,  # Connection pool size
            pool_maxsize=20,
        )

        session.mount("http://", adapter)
        session.mount("https://", adapter)

        return session

    def process(self, prompt: str, input_data: Any, **kwargs) -> Dict[str, Any]:
        """Process LLM request with comprehensive observability"""
        start_time = time.time()

        with self.tracer.start_trace(
            "llm_call",
            {
                "prompt_type": prompt,
                "model": kwargs.get("model"),
                "input_size": len(str(input_data)),
            },
        ):
            try:
                # Validate prompt before sending
                validation = self.validate_prompt(prompt)
                if not validation["is_valid"]:
                    raise AppException(
                        message=f"Prompt validation failed: {validation.get('rejection_reason')}",
                        extra={
                            "validation_result": validation,
                            "prompt_preview": prompt[:100],
                        }
                    )

                # Make the LLM call
                result = self._ollama_call(prompt, input_data, **kwargs)

                latency_ms = int((time.time() - start_time) * 1000)

                # Record success event
                self.tracer.record_event(
                    "llm_call_completed",
                    {
                        "prompt_hash": hash(prompt) % 10000,  # Hash for privacy
                        "response_length": len(str(result.get("content", ""))),
                        "processing_time_ms": latency_ms,
                        "tokens_used": result.get("tokens_used", 0),
                        "success": True,
                    },
                )

                # Log for observability
                logger.info(
                    "LLM call completed",
                    extra={
                        "model": kwargs.get("model", self.config.OLLAMA_DEFAULT_MODEL),
                        "latency_ms": latency_ms,
                        "tokens_used": result.get("tokens_used", 0),
                        "success": True,
                    },
                )

                return result

            except requests.exceptions.Timeout:
                error_msg = f"LLM request timed out after {self.request_timeout} seconds"
                self.tracer.record_error(
                    "llm_timeout",
                    error_msg,
                    {
                        "prompt_preview": prompt[:100],
                        "timeout_seconds": self.request_timeout,
                    },
                )
                raise AppException(
                    message=error_msg,
                    extra={
                        "error_type": "timeout",
                        "prompt_preview": prompt[:100],
                        "timeout_seconds": self.request_timeout,
                    }
                )

            except requests.exceptions.ConnectionError:
                error_msg = f"Cannot connect to Ollama service at {self.config.OLLAMA_BASE_URL}"
                self.tracer.record_error(
                    "llm_connection_failed",
                    error_msg,
                    {"ollama_url": self.config.OLLAMA_BASE_URL},
                )
                raise AppException(
                    message=error_msg,
                    extra={
                        "error_type": "connection_error",
                        "ollama_url": self.config.OLLAMA_BASE_URL,
                    }
                )

            except AppException:
                # Re-raise AppException without wrapping
                raise

            except Exception as e:
                error_msg = f"LLM call failed: {str(e)}"
                self.tracer.record_error(
                    "llm_call_failed",
                    error_msg,
                    {
                        "prompt_preview": prompt[:100],
                        "error_type": type(e).__name__,
                        "model": kwargs.get("model"),
                    },
                )
                logger.error(
                    "LLM call failed",
                    extra={
                        "error": str(e),
                        "model": kwargs.get("model"),
                        "processing_time_ms": int((time.time() - start_time) * 1000),
                    },
                )
                raise AppException(
                    message=error_msg,
                    extra={
                        "error_type": type(e).__name__,
                        "prompt_preview": prompt[:100],
                        "model": kwargs.get("model"),
                        "original_exception": str(e),
                    }
                )

    def _ollama_call(self, prompt: str, input_data: Any, **kwargs) -> Dict[str, Any]:
        """Make actual call to Ollama API with proper error handling"""
        model = kwargs.get("model", self.config.OLLAMA_DEFAULT_MODEL)

        # Construct the prompt with context
        full_prompt = self._construct_prompt(prompt, input_data)

        # Prepare request payload
        payload = {
            "model": model,
            "prompt": full_prompt,
            "stream": False,
            "options": {
                "temperature": float(kwargs.get("temperature", 0.7)),
                "top_p": float(kwargs.get("top_p", 0.9)),
                "num_predict": int(kwargs.get("max_tokens", 512)),
                "top_k": int(kwargs.get("top_k", 40)),
                "repeat_penalty": float(kwargs.get("repeat_penalty", 1.1)),
            },
        }

        # Make the request
        try:
            response = self.session.post(
                f"{self.config.OLLAMA_BASE_URL}/api/generate",
                json=payload,
                timeout=(
                    self.config.LLM_TIMEOUT_SECONDS,
                    self.config.OLLAMA_REQUEST_TIMEOUT,
                ),
            )

            response.raise_for_status()  # Raise exception for bad status codes

            result = response.json()

            return {
                "content": result.get("response", ""),
                "model": model,
                "tokens_used": result.get("eval_count", 0)
                + result.get("prompt_eval_count", 0),
                "latency_ms": int(
                    result.get("total_duration", 0) / 1000000
                ),  # Convert nanoseconds to ms
                "raw_response": result,
            }

        except requests.exceptions.HTTPError as e:
            if response.status_code == 404:
                raise AppException(
                    message=f"Model '{model}' not found in Ollama",
                    extra={
                        "status_code": response.status_code,
                        "model": model,
                        "response_text": response.text[:500] if response.text else None,
                    }
                )
            elif response.status_code == 400:
                raise AppException(
                    message=f"Bad request to Ollama API",
                    extra={
                        "status_code": response.status_code,
                        "response_text": response.text[:500] if response.text else None,
                        "model": model,
                    }
                )
            else:
                raise AppException(
                    message=f"Ollama API error (HTTP {response.status_code})",
                    extra={
                        "status_code": response.status_code,
                        "response_text": response.text[:500] if response.text else None,
                        "model": model,
                    }
                )
        except Exception as e:
            raise AppException(
                message=f"Failed to call Ollama API: {str(e)}",
                extra={
                    "error_type": type(e).__name__,
                    "model": model,
                    "prompt_length": len(full_prompt),
                }
            )

    def _construct_prompt(self, prompt: str, input_data: Any) -> str:
        """Construct the full prompt with context and instructions"""
        if isinstance(input_data, dict):
            input_str = str(input_data)
        elif isinstance(input_data, str):
            input_str = input_data
        else:
            input_str = str(input_data)

        return f"""You are an AI assistant executing a backend task with observability.

Context:
- Task requires full observability and traceability
- All decisions and reasoning must be recorded
- Provide structured, actionable responses

Task input:
{input_str}

Instructions:
{prompt}

Important: Include reasoning in your response for observability purposes.
""".strip()

    def validate_prompt(self, prompt: str) -> Dict[str, Any]:
        """Validate prompt for safety and quality"""
        validation_result = {
            "is_valid": True,
            "warnings": [],
            "rejection_reason": None,
            "prompt_length": len(prompt),
        }

        # Basic validation
        if not prompt or len(prompt.strip()) == 0:
            validation_result["is_valid"] = False
            validation_result["rejection_reason"] = "Empty prompt"

        # Length validation
        if len(prompt) > 10000:
            validation_result["warnings"].append(
                "Prompt exceeds recommended length (10,000 chars)"
            )

        if len(prompt) < 10:
            validation_result["warnings"].append(
                "Prompt is very short, may be insufficient"
            )

        # Safety checks
        blocked_terms = [
            "harmful",
            "dangerous",
            "illegal",
            "malicious",
            "bypass",
            "exploit",
            "vulnerability",
            "hack",
        ]

        prompt_lower = prompt.lower()
        found_terms = [term for term in blocked_terms if term in prompt_lower]

        if found_terms:
            validation_result["warnings"].append(
                f"Prompt contains concerning terms: {found_terms}"
            )

        # Record validation for observability
        self.tracer.record_event("prompt_validated", validation_result)

        return validation_result

    def get_available_models(self) -> list:
        """Get list of available models from Ollama"""
        try:
            response = self.session.get(
                f"{self.config.OLLAMA_BASE_URL}/api/tags", timeout=5
            )
            response.raise_for_status()
            data = response.json()
            return [model["name"] for model in data.get("models", [])]
        except Exception as e:
            raise AppException(
                message="Failed to get available models from Ollama",
                extra={
                    "error_type": type(e).__name__,
                    "error_details": str(e),
                    "ollama_url": self.config.OLLAMA_BASE_URL,
                }
            )