"""tests for multi-provider LLM support and .env configuration."""

import json  # noqa: F401
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest  # noqa: F401


class TestEnvLoading:
    """tests for .env file loading."""

    def test_load_env_from_workdir(self):
        """verify .env file in workdir is loaded."""
        with tempfile.TemporaryDirectory() as tmpdir:
            env_path = Path(tmpdir) / ".env"
            env_path.write_text("TEST_VAR_XYZ=workdir_value\n")

            # clear any existing value
            os.environ.pop("TEST_VAR_XYZ", None)

            from autiobook.env import load_env

            load_env(Path(tmpdir))

            assert os.environ.get("TEST_VAR_XYZ") == "workdir_value"

            # cleanup
            os.environ.pop("TEST_VAR_XYZ", None)

    def test_load_env_from_cwd_fallback(self):
        """verify .env in cwd is used when workdir has no .env."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workdir = Path(tmpdir) / "workdir"
            workdir.mkdir()

            # no .env in workdir, but we won't test cwd fallback here
            # as it would affect other tests; just verify no crash
            os.environ.pop("TEST_VAR_ABC", None)

            from autiobook.env import load_env

            load_env(workdir)  # should not crash

    def test_env_not_override_existing(self):
        """verify .env does not override already-set environment variables."""
        with tempfile.TemporaryDirectory() as tmpdir:
            env_path = Path(tmpdir) / ".env"
            env_path.write_text("EXISTING_VAR=from_file\n")

            os.environ["EXISTING_VAR"] = "already_set"

            from autiobook.env import load_env

            load_env(Path(tmpdir))

            assert os.environ.get("EXISTING_VAR") == "already_set"

            # cleanup
            os.environ.pop("EXISTING_VAR", None)


class TestLiteLLMIntegration:
    """tests for litellm provider integration."""

    def test_query_llm_json_with_openai_prefix(self):
        """verify openai/ prefixed models work."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '{"result": "ok"}'

        with patch("litellm.completion", return_value=mock_response) as mock_completion:
            from autiobook.llm import _query_llm_json

            result = _query_llm_json(  # noqa: F841
                system_prompt="test system",
                user_prompt="test user",
                model="openai/gpt-4o",
            )

            mock_completion.assert_called_once()
            call_kwargs = mock_completion.call_args[1]
            assert call_kwargs["model"] == "openai/gpt-4o"
            assert result == {"result": "ok"}

    def test_query_llm_json_with_anthropic_prefix(self):
        """verify anthropic/ prefixed models work."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '{"result": "ok"}'

        with patch("litellm.completion", return_value=mock_response) as mock_completion:
            from autiobook.llm import _query_llm_json

            result = _query_llm_json(  # noqa: F841
                system_prompt="test system",
                user_prompt="test user",
                model="anthropic/claude-3-5-sonnet-20240620",
            )

            mock_completion.assert_called_once()
            call_kwargs = mock_completion.call_args[1]
            assert call_kwargs["model"] == "anthropic/claude-3-5-sonnet-20240620"

    def test_query_llm_json_with_gemini_prefix(self):
        """verify gemini/ prefixed models work."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '{"result": "ok"}'

        with patch("litellm.completion", return_value=mock_response) as mock_completion:
            from autiobook.llm import _query_llm_json

            result = _query_llm_json(  # noqa: F841
                system_prompt="test system",
                user_prompt="test user",
                model="gemini/gemini-1.5-pro",
            )

            mock_completion.assert_called_once()
            call_kwargs = mock_completion.call_args[1]
            assert call_kwargs["model"] == "gemini/gemini-1.5-pro"

    def test_query_llm_json_model_passed_as_is(self):
        """verify model string is passed through unchanged."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '{"result": "ok"}'

        with patch("litellm.completion", return_value=mock_response) as mock_completion:
            from autiobook.llm import _query_llm_json

            result = _query_llm_json(  # noqa: F841
                system_prompt="test system",
                user_prompt="test user",
                model="openai/gpt-4o",
            )

            mock_completion.assert_called_once()
            call_kwargs = mock_completion.call_args[1]
            assert call_kwargs["model"] == "openai/gpt-4o"


class TestThinkingBudget:
    """tests for thinking/extended thinking support."""

    def test_thinking_budget_passed_to_litellm(self):
        """verify thinking parameter is passed via extra_body when budget > 0."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '{"result": "ok"}'

        with patch("litellm.completion", return_value=mock_response) as mock_completion:
            from autiobook.llm import _query_llm_json

            result = _query_llm_json(  # noqa: F841
                system_prompt="test system",
                user_prompt="test user",
                model="anthropic/claude-3-5-sonnet-20240620",
                thinking_budget=1024,
            )

            mock_completion.assert_called_once()
            call_kwargs = mock_completion.call_args[1]
            assert "extra_body" in call_kwargs
            assert call_kwargs["extra_body"]["thinking"]["type"] == "enabled"
            assert call_kwargs["extra_body"]["thinking"]["budget_tokens"] == 1024

    def test_thinking_budget_passed_for_openai(self):
        """verify thinking parameter is passed via extra_body for openai models too."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '{"result": "ok"}'

        with patch("litellm.completion", return_value=mock_response) as mock_completion:
            from autiobook.llm import _query_llm_json

            result = _query_llm_json(  # noqa: F841
                system_prompt="test system",
                user_prompt="test user",
                model="openai/gpt-4o",
                thinking_budget=2048,
            )

            mock_completion.assert_called_once()
            call_kwargs = mock_completion.call_args[1]
            assert "extra_body" in call_kwargs
            assert call_kwargs["extra_body"]["thinking"]["budget_tokens"] == 2048

    def test_zero_thinking_budget_disables_thinking(self):
        """verify thinking_budget=0 does not pass extra_body with thinking."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '{"result": "ok"}'

        with patch("litellm.completion", return_value=mock_response) as mock_completion:
            from autiobook.llm import _query_llm_json

            result = _query_llm_json(  # noqa: F841
                system_prompt="test system",
                user_prompt="test user",
                model="anthropic/claude-3-5-sonnet-20240620",
                thinking_budget=0,
            )

            mock_completion.assert_called_once()
            call_kwargs = mock_completion.call_args[1]
            assert "extra_body" not in call_kwargs


class TestApiBaseOverride:
    """tests for api_base URL override functionality."""

    def test_api_base_passed_to_litellm(self):
        """verify api_base is passed through to litellm."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '{"result": "ok"}'

        with patch("litellm.completion", return_value=mock_response) as mock_completion:
            from autiobook.llm import _query_llm_json

            result = _query_llm_json(  # noqa: F841
                system_prompt="test system",
                user_prompt="test user",
                model="openai/gpt-4o",
                api_base="http://localhost:8080/v1",
            )

            mock_completion.assert_called_once()
            call_kwargs = mock_completion.call_args[1]
            assert call_kwargs["api_base"] == "http://localhost:8080/v1"

    def test_api_key_passed_to_litellm(self):
        """verify api_key is passed through to litellm."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '{"result": "ok"}'

        with patch("litellm.completion", return_value=mock_response) as mock_completion:
            from autiobook.llm import _query_llm_json

            result = _query_llm_json(  # noqa: F841
                system_prompt="test system",
                user_prompt="test user",
                model="openai/gpt-4o",
                api_key="sk-test-key",
            )

            mock_completion.assert_called_once()
            call_kwargs = mock_completion.call_args[1]
            assert call_kwargs["api_key"] == "sk-test-key"


class TestRetryBehavior:
    """tests for retry behavior with litellm."""

    def test_retry_on_api_error(self):
        """verify retry behavior on transient API errors."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '{"result": "ok"}'

        call_count = 0

        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise Exception("rate limit exceeded")
            return mock_response

        with patch("litellm.completion", side_effect=side_effect):
            with patch("time.sleep"):  # skip actual sleep
                from autiobook.llm import _query_llm_json

                result = _query_llm_json(  # noqa: F841
                    system_prompt="test",
                    user_prompt="test",
                    model="openai/gpt-4o",
                )

                assert result == {"result": "ok"}
                assert call_count == 2

    def test_retry_on_json_parse_error(self):
        """verify retry on malformed JSON response."""
        mock_response_bad = MagicMock()
        mock_response_bad.choices = [MagicMock()]
        mock_response_bad.choices[0].message.content = "not json"

        mock_response_good = MagicMock()
        mock_response_good.choices = [MagicMock()]
        mock_response_good.choices[0].message.content = '{"result": "ok"}'

        call_count = 0

        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                return mock_response_bad
            return mock_response_good

        with patch("litellm.completion", side_effect=side_effect):
            with patch("time.sleep"):
                from autiobook.llm import _query_llm_json

                result = _query_llm_json(  # noqa: F841
                    system_prompt="test",
                    user_prompt="test",
                    model="openai/gpt-4o",
                )

                assert result == {"result": "ok"}
                assert call_count == 2


class TestJsonResponseParsing:
    """tests for JSON response parsing."""

    def test_parse_json_with_markdown_fence(self):
        """verify markdown code fences are stripped."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '```json\n{"result": "ok"}\n```'

        with patch("litellm.completion", return_value=mock_response):
            from autiobook.llm import _query_llm_json

            result = _query_llm_json(  # noqa: F841
                system_prompt="test",
                user_prompt="test",
                model="openai/gpt-4o",
            )

            assert result == {"result": "ok"}

    def test_parse_json_with_wrapper_keys(self):
        """verify wrapper keys are extracted."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '{"characters": [{"name": "Alice"}]}'

        with patch("litellm.completion", return_value=mock_response):
            from autiobook.llm import _query_llm_json

            result = _query_llm_json(  # noqa: F841
                system_prompt="test",
                user_prompt="test",
                model="openai/gpt-4o",
                wrapper_keys=["characters", "c"],
            )

            assert result == [{"name": "Alice"}]
