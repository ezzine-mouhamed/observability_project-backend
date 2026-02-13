import pytest
from unittest.mock import Mock, patch
from requests.exceptions import Timeout, ConnectionError, HTTPError

from app.exceptions.base import AppException
from app.services.llm_client import LLMClient


class TestLLMClient:

    def setup_method(self):
        """Setup for each test"""
        with patch('app.services.llm_client.Config') as mock_config_class, \
             patch('app.services.llm_client.Tracer') as mock_tracer_class:
            
            # Mock config values
            self.mock_config = Mock()
            self.mock_config.OLLAMA_BASE_URL = "http://localhost:11434"
            self.mock_config.OLLAMA_DEFAULT_MODEL = "llama2"
            self.mock_config.OLLAMA_REQUEST_TIMEOUT = 30
            self.mock_config.LLM_TIMEOUT_SECONDS = 30  # Add this config
            
            mock_config_class.return_value = self.mock_config
            
            # Mock tracer
            self.mock_tracer = Mock()
            mock_tracer_class.return_value = self.mock_tracer
            
            # Create the client
            self.client = LLMClient()

    def test_init_creates_session_with_retry(self):
        """Test that __init__ creates a session with retry strategy"""
        assert hasattr(self.client, 'session')
        assert hasattr(self.client, 'tracer')
        assert hasattr(self.client, 'config')
        assert self.client.request_timeout == 30
        
        # Check that session has adapters mounted
        assert 'http://' in self.client.session.adapters
        assert 'https://' in self.client.session.adapters
    
    def test_create_retry_session(self):
        """Test retry session creation"""
        session = self.client._create_retry_session()
        assert session is not None
        
        # Check retry configuration
        adapter = session.get_adapter('http://')
        assert adapter.max_retries.total == 3
        assert adapter.max_retries.backoff_factor == 1
        
    def test_construct_prompt_with_dict_input(self):
        """Test prompt construction with dictionary input"""
        prompt = "Analyze this data"
        input_data = {"key": "value", "number": 42}
        
        result = self.client._construct_prompt(prompt, input_data)
        
        assert "Task input:" in result
        assert "{'key': 'value', 'number': 42}" in result or '"key": "value"' in result
        assert "Instructions:" in result
        assert "Analyze this data" in result
        assert "reasoning" in result.lower()
    
    def test_construct_prompt_with_string_input(self):
        """Test prompt construction with string input"""
        prompt = "Summarize this text"
        input_data = "This is some sample text to summarize."
        
        result = self.client._construct_prompt(prompt, input_data)
        
        assert "Task input:" in result
        assert "This is some sample text to summarize." in result
        assert "Summarize this text" in result
    
    def test_construct_prompt_with_other_input(self):
        """Test prompt construction with other input types"""
        prompt = "Process this"
        input_data = 123.45
        
        result = self.client._construct_prompt(prompt, input_data)
        
        assert "123.45" in result
    
    def test_validate_prompt_valid(self):
        """Test prompt validation with valid prompt"""
        prompt = "This is a valid prompt for testing."
        
        result = self.client.validate_prompt(prompt)
        
        assert result["is_valid"] == True
        assert result["rejection_reason"] is None
        assert result["prompt_length"] == len(prompt)
        self.mock_tracer.record_event.assert_called_with("prompt_validated", result)
    
    def test_validate_prompt_empty(self):
        """Test prompt validation with empty prompt"""
        prompt = ""
        
        result = self.client.validate_prompt(prompt)
        
        assert result["is_valid"] == False
        assert result["rejection_reason"] == "Empty prompt"
    
    def test_validate_prompt_whitespace_only(self):
        """Test prompt validation with whitespace-only prompt"""
        prompt = "   \n\t   "
        
        result = self.client.validate_prompt(prompt)
        
        assert result["is_valid"] == False
        assert result["rejection_reason"] == "Empty prompt"
    
    def test_validate_prompt_too_long(self):
        """Test prompt validation with very long prompt"""
        prompt = "x" * 15000
        
        result = self.client.validate_prompt(prompt)
        
        assert result["is_valid"] == True  # Should still be valid but with warning
        assert len(result["warnings"]) == 1
        assert "exceeds recommended length" in result["warnings"][0]
    
    def test_validate_prompt_too_short(self):
        """Test prompt validation with very short prompt"""
        prompt = "Hi"
        
        result = self.client.validate_prompt(prompt)
        
        assert result["is_valid"] == True  # Should still be valid but with warning
        assert len(result["warnings"]) == 1
        assert "very short" in result["warnings"][0]
    
    def test_validate_prompt_with_blocked_terms(self):
        """Test prompt validation with concerning terms"""
        prompt = "This is a test with harmful content and exploit techniques"
        
        result = self.client.validate_prompt(prompt)
        
        assert result["is_valid"] == True  # Only warnings for concerning terms
        assert len(result["warnings"]) == 1
        assert "concerning terms" in result["warnings"][0]
        assert "harmful" in result["warnings"][0]
        assert "exploit" in result["warnings"][0]
    
    @patch('app.services.llm_client.LLMClient._ollama_call')
    def test_process_success(self, mock_ollama_call):
        """Test successful process method execution"""
        # Setup
        prompt = "Test prompt"
        input_data = {"test": "data"}
        expected_result = {
            "content": "Test response",
            "model": "llama2",
            "tokens_used": 100,
            "latency_ms": 500,
            "raw_response": {"response": "Test response"}
        }
        mock_ollama_call.return_value = expected_result
        
        # Create a proper mock for the context manager
        mock_trace_context = Mock()
        
        # Mock start_trace to return a context manager
        mock_context_manager = Mock()
        mock_context_manager.__enter__ = Mock(return_value=mock_trace_context)
        mock_context_manager.__exit__ = Mock(return_value=None)
        self.mock_tracer.start_trace.return_value = mock_context_manager
        
        # Execute
        result = self.client.process(prompt, input_data, model="llama2")
        
        # Verify
        assert result == expected_result
        mock_ollama_call.assert_called_once_with(prompt, input_data, model="llama2")
        
        # Verify tracing
        self.mock_tracer.start_trace.assert_called_once_with(
            "llm_call",
            {
                "prompt_type": prompt,
                "model": "llama2",
                "input_size": len(str(input_data))
            }
        )
        
        # Verify success event was recorded
        self.mock_tracer.record_event.assert_called_with(
            "llm_call_completed",
            {
                "prompt_hash": hash(prompt) % 10000,
                "response_length": len(expected_result["content"]),
                "processing_time_ms": pytest.approx(0, abs=100),  # Approximate
                "tokens_used": expected_result["tokens_used"],
                "success": True
            }
        )

    @patch('app.services.llm_client.LLMClient.validate_prompt')
    def test_process_invalid_prompt(self, mock_validate_prompt):
        """Test process with invalid prompt"""
        # Setup
        prompt = "Invalid prompt"
        input_data = {"test": "data"}
        validation_result = {
            "is_valid": False,
            "rejection_reason": "Test rejection reason",
            "warnings": []
        }
        mock_validate_prompt.return_value = validation_result
        
        # Create a proper mock for the context manager
        mock_trace_context = Mock()
        
        # Mock start_trace to return a context manager
        mock_context_manager = Mock()
        mock_context_manager.__enter__ = Mock(return_value=mock_trace_context)
        mock_context_manager.__exit__ = Mock(return_value=None)
        self.mock_tracer.start_trace.return_value = mock_context_manager
        
        # Execute and verify exception
        with pytest.raises(AppException) as exc_info:
            self.client.process(prompt, input_data)
        
        assert "Prompt validation failed" in str(exc_info.value)
        assert "Test rejection reason" in str(exc_info.value)
        
        # Verify validation was called
        mock_validate_prompt.assert_called_once_with(prompt)
        
        # Verify error was recorded
        self.mock_tracer.record_error.assert_not_called()  # AppException shouldn't trigger record_error

    @patch('app.services.llm_client.LLMClient._ollama_call')
    def test_process_timeout(self, mock_ollama_call):
        """Test process with timeout exception"""
        # Setup
        prompt = "Test prompt"
        input_data = {"test": "data"}
        mock_ollama_call.side_effect = Timeout("Request timed out")
        
        # Create a proper mock for the context manager
        mock_trace_context = Mock()
        
        # Mock start_trace to return a context manager
        mock_context_manager = Mock()
        mock_context_manager.__enter__ = Mock(return_value=mock_trace_context)
        mock_context_manager.__exit__ = Mock(return_value=None)
        self.mock_tracer.start_trace.return_value = mock_context_manager
        
        # Execute and verify
        with pytest.raises(AppException) as exc_info:
            self.client.process(prompt, input_data)
        
        assert "timed out after 30 seconds" in str(exc_info.value)
        
        # Verify error was recorded
        self.mock_tracer.record_error.assert_called_once()
        call_args = self.mock_tracer.record_error.call_args
        assert call_args[0][0] == "llm_timeout"
        assert "timed out after 30 seconds" in call_args[0][1]

    @patch('app.services.llm_client.LLMClient._ollama_call')
    def test_process_connection_error(self, mock_ollama_call):
        """Test process with connection error"""
        # Setup
        prompt = "Test prompt"
        input_data = {"test": "data"}
        mock_ollama_call.side_effect = ConnectionError("Connection failed")
        
        # Create a proper mock for the context manager
        mock_trace_context = Mock()
        
        # Mock start_trace to return a context manager
        mock_context_manager = Mock()
        mock_context_manager.__enter__ = Mock(return_value=mock_trace_context)
        mock_context_manager.__exit__ = Mock(return_value=None)
        self.mock_tracer.start_trace.return_value = mock_context_manager
        
        # Execute and verify
        with pytest.raises(AppException) as exc_info:
            self.client.process(prompt, input_data)
        
        assert "Cannot connect to Ollama service" in str(exc_info.value)
        assert "http://localhost:11434" in str(exc_info.value)
        
        # Verify error was recorded
        self.mock_tracer.record_error.assert_called_once()
        call_args = self.mock_tracer.record_error.call_args
        assert call_args[0][0] == "llm_connection_failed"

    @patch('app.services.llm_client.LLMClient._ollama_call')
    def test_process_generic_exception(self, mock_ollama_call):
        """Test process with generic exception"""
        # Setup
        prompt = "Test prompt"
        input_data = {"test": "data"}
        mock_ollama_call.side_effect = ValueError("Some random error")
        
        # Create a proper mock for the context manager
        mock_trace_context = Mock()
        
        # Mock start_trace to return a context manager
        mock_context_manager = Mock()
        mock_context_manager.__enter__ = Mock(return_value=mock_trace_context)
        mock_context_manager.__exit__ = Mock(return_value=None)
        self.mock_tracer.start_trace.return_value = mock_context_manager
        
        # Execute and verify
        with pytest.raises(AppException) as exc_info:
            self.client.process(prompt, input_data)
        
        assert "LLM call failed" in str(exc_info.value)
        # Check for the actual error message, not the exception type
        assert "Some random error" in str(exc_info.value)
        
        # Verify error was recorded
        self.mock_tracer.record_error.assert_called_once()
        call_args = self.mock_tracer.record_error.call_args
        assert call_args[0][0] == "llm_call_failed"

    @patch('app.services.llm_client.requests.Session.post')
    def test_ollama_call_success(self, mock_post):
        """Test successful Ollama API call"""
        # Setup mock response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "response": "Test response from LLM",
            "eval_count": 50,
            "prompt_eval_count": 10,
            "total_duration": 500000000,  # 500ms in nanoseconds
            "model": "llama2"
        }
        mock_post.return_value = mock_response
        
        # Execute
        prompt = "Test prompt"
        input_data = {"test": "data"}
        result = self.client._ollama_call(prompt, input_data, temperature=0.5, max_tokens=256)
        
        # Verify
        assert result["content"] == "Test response from LLM"
        assert result["model"] == "llama2"
        assert result["tokens_used"] == 60  # 50 + 10
        assert result["latency_ms"] == 500
        
        # Verify request payload
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        assert call_args[0][0] == "http://localhost:11434/api/generate"
        
        payload = call_args[1]['json']
        assert payload["model"] == "llama2"
        assert "Test prompt" in payload["prompt"]
        assert payload["options"]["temperature"] == 0.5
        assert payload["options"]["num_predict"] == 256
        assert payload["stream"] == False

    @patch('app.services.llm_client.requests.Session.post')
    def test_ollama_call_model_not_found(self, mock_post):
        """Test Ollama call with model not found error"""
        # Setup mock response with 404
        mock_response = Mock()
        mock_response.status_code = 404
        mock_response.text = "model 'nonexistent' not found"
        mock_response.raise_for_status.side_effect = HTTPError("404 Error")
        mock_post.return_value = mock_response
        
        # Execute and verify
        prompt = "Test prompt"
        input_data = {"test": "data"}
        
        with pytest.raises(AppException) as exc_info:
            self.client._ollama_call(prompt, input_data, model="nonexistent")
        
        assert "Model 'nonexistent' not found in Ollama" in str(exc_info.value)
        assert exc_info.value.extra["status_code"] == 404
        assert exc_info.value.extra["model"] == "nonexistent"

    @patch('app.services.llm_client.requests.Session.post')
    def test_ollama_call_bad_request(self, mock_post):
        """Test Ollama call with bad request error"""
        # Setup mock response with 400
        mock_response = Mock()
        mock_response.status_code = 400
        mock_response.text = "Invalid request parameters"
        mock_response.raise_for_status.side_effect = HTTPError("400 Error")
        mock_post.return_value = mock_response
        
        # Execute and verify
        prompt = "Test prompt"
        input_data = {"test": "data"}
        
        with pytest.raises(AppException) as exc_info:
            self.client._ollama_call(prompt, input_data)
        
        assert "Bad request to Ollama API" in str(exc_info.value)
        assert exc_info.value.extra["status_code"] == 400

    @patch('app.services.llm_client.requests.Session.post')
    def test_ollama_call_http_error(self, mock_post):
        """Test Ollama call with other HTTP error"""
        # Setup mock response with 500
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.text = "Internal server error"
        mock_response.raise_for_status.side_effect = HTTPError("500 Error")
        mock_post.return_value = mock_response
        
        # Execute and verify
        prompt = "Test prompt"
        input_data = {"test": "data"}
        
        with pytest.raises(AppException) as exc_info:
            self.client._ollama_call(prompt, input_data)
        
        assert "Ollama API error (HTTP 500)" in str(exc_info.value)
        assert exc_info.value.extra["status_code"] == 500

    @patch('app.services.llm_client.requests.Session.post')
    def test_ollama_call_generic_exception(self, mock_post):
        """Test Ollama call with generic exception"""
        # Setup mock to raise exception
        mock_post.side_effect = Exception("Network failure")
        
        # Execute and verify
        prompt = "Test prompt"
        input_data = {"test": "data"}
        
        with pytest.raises(AppException) as exc_info:
            self.client._ollama_call(prompt, input_data)
        
        assert "Failed to call Ollama API" in str(exc_info.value)
        assert "Network failure" in str(exc_info.value)

    @patch('app.services.llm_client.requests.Session.get')
    def test_get_available_models_success(self, mock_get):
        """Test successful retrieval of available models"""
        # Setup mock response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "models": [
                {"name": "llama2"},
                {"name": "mistral"},
                {"name": "codellama"}
            ]
        }
        mock_get.return_value = mock_response
        
        # Execute
        models = self.client.get_available_models()
        
        # Verify
        assert models == ["llama2", "mistral", "codellama"]
        mock_get.assert_called_once_with(
            "http://localhost:11434/api/tags",
            timeout=5
        )

    @patch('app.services.llm_client.requests.Session.get')
    def test_get_available_models_failure(self, mock_get):
        """Test failure to retrieve available models"""
        # Setup mock to raise exception
        mock_get.side_effect = Exception("Connection refused")
        
        # Execute and verify
        with pytest.raises(AppException) as exc_info:
            self.client.get_available_models()
        
        assert "Failed to get available models from Ollama" in str(exc_info.value)
        assert "Connection refused" in str(exc_info.value.extra["error_details"])
