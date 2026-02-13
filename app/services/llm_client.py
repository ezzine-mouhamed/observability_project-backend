import time
from typing import Any, Dict, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from app.config import Config
from app.exceptions.base import AppException
from app.observability.tracer import Tracer
from app.utils.logger import get_logger

logger = get_logger(__name__)


class LLMClient:
    def __init__(self, tracer: Optional[Tracer] = None):
        self.tracer = tracer or Tracer()
        self.config = Config()
        self.session = self._create_retry_session()
        self.request_timeout = self.config.OLLAMA_REQUEST_TIMEOUT

    def _create_retry_session(self) -> requests.Session:
        session = requests.Session()

        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "POST"],
            raise_on_status=False,
        )

        adapter = HTTPAdapter(
            max_retries=retry_strategy,
            pool_connections=10,
            pool_maxsize=20,
        )

        session.mount("http://", adapter)
        session.mount("https://", adapter)

        return session

    def process(self, prompt: str, input_data: Any, **kwargs) -> Dict[str, Any]:
        start_time = time.time()

        model = kwargs.get("model", self.config.OLLAMA_DEFAULT_MODEL)
        
        trace_context = {
            "prompt_type": prompt,
            "model": model,
            "input_size": len(str(input_data)),
        }
        
        current_trace = self.tracer.context.current
        if current_trace and "task_id" in current_trace.get("context", {}):
            trace_context["task_id"] = current_trace["context"]["task_id"]

        with self.tracer.start_trace(
            "llm_call",
            trace_context,
        ):
            try:
                validation = self.validate_prompt(prompt)
                if not validation["is_valid"]:
                    raise AppException(
                        message=f"Prompt validation failed: {validation.get('rejection_reason')}",
                        extra={
                            "validation_result": validation,
                            "prompt_preview": prompt[:100],
                        }
                    )

                kwargs["model"] = model
                result = self._ollama_call(prompt, input_data, **kwargs)

                latency_ms = int((time.time() - start_time) * 1000)

                self.tracer.record_event(
                    "llm_call_completed",
                    {
                        "prompt_hash": hash(prompt) % 10000,
                        "response_length": len(str(result.get("content", ""))),
                        "processing_time_ms": latency_ms,
                        "tokens_used": result.get("tokens_used", 0),
                        "model": model,
                        "success": True,
                    },
                )

                logger.info("LLM call completed", extra={
                    "model": model,
                    "latency_ms": latency_ms,
                    "tokens_used": result.get("tokens_used", 0),
                    "success": True,
                })

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
                raise

            except Exception as e:
                error_msg = f"LLM call failed: {str(e)}"
                self.tracer.record_error(
                    "llm_call_failed",
                    error_msg,
                    {
                        "prompt_preview": prompt[:100],
                        "error_type": type(e).__name__,
                        "model": model,
                    },
                )
                logger.error(
                    "LLM call failed",
                    extra={
                        "error": str(e),
                        "model": model,
                        "processing_time_ms": int((time.time() - start_time) * 1000),
                    },
                )
                raise AppException(
                    message=error_msg,
                    extra={
                        "error_type": type(e).__name__,
                        "prompt_preview": prompt[:100],
                        "model": model,
                        "original_exception": str(e),
                    }
                )

    def _ollama_call(self, prompt: str, input_data: Any, **kwargs) -> Dict[str, Any]:
        model = kwargs.get("model", self.config.OLLAMA_DEFAULT_MODEL)

        full_prompt = self._construct_prompt(prompt, input_data)

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

        try:
            response = self.session.post(
                f"{self.config.OLLAMA_BASE_URL}/api/generate",
                json=payload,
                timeout=(
                    self.config.LLM_TIMEOUT_SECONDS,
                    self.config.OLLAMA_REQUEST_TIMEOUT,
                ),
            )

            response.raise_for_status()

            result = response.json()

            return {
                "content": result.get("response", ""),
                "model": model,
                "tokens_used": result.get("eval_count", 0)
                + result.get("prompt_eval_count", 0),
                "latency_ms": int(
                    result.get("total_duration", 0) / 1000000
                ),
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
- Only respond with the final answer. Do not include any explanations or reasoning in the response.

Task input:
{input_str}

Instructions:
{prompt}

Important: Include reasoning in your response for observability purposes.
""".strip()

    def validate_prompt(self, prompt: str) -> Dict[str, Any]:
        validation_result = {
            "is_valid": True,
            "warnings": [],
            "rejection_reason": None,
            "prompt_length": len(prompt),
        }

        if not prompt or len(prompt.strip()) == 0:
            validation_result["is_valid"] = False
            validation_result["rejection_reason"] = "Empty prompt"

        if len(prompt) > 10000:
            validation_result["warnings"].append(
                "Prompt exceeds recommended length (10,000 chars)"
            )

        if len(prompt) < 10:
            validation_result["warnings"].append(
                "Prompt is very short, may be insufficient"
            )

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

        self.tracer.record_event("prompt_validated", validation_result)

        return validation_result

    def get_available_models(self) -> list:
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
