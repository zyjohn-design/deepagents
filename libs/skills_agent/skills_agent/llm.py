"""
LLM factory and provider management.

Supports:
  - Cloud providers: openai, anthropic, google, deepseek
  - Local OpenAI-compatible servers: ollama, vllm, lmstudio, llamacpp, xinference
  - Generic local: "local:model-name" with custom base_url
  - API key resolution: settings → env → placeholder for local
  - Retry wrapper with exponential backoff
  - Callback wrapper for skill executor integration

Usage — cloud:
    settings.llm.model = "openai:gpt-4o"
    llm = create_llm(settings.llm)

Usage — local (Ollama):
    settings.llm.model = "ollama:qwen2.5:72b"
    llm = create_llm(settings.llm)
    # → ChatOpenAI(base_url="http://localhost:11434/v1", model="qwen2.5:72b")

Usage — local (vLLM):
    settings.llm.model = "vllm:Qwen/Qwen2.5-72B-Instruct"
    settings.llm.base_url = "http://gpu-server:8000/v1"
    llm = create_llm(settings.llm)

Usage — local (generic):
    settings.llm.model = "local:my-finetuned-model"
    settings.llm.base_url = "http://10.0.0.5:8080/v1"
    llm = create_llm(settings.llm)
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from .config import LLMSettings

from .exceptions import LLMCallError, LLMConfigError

logger = logging.getLogger(__name__)


# ======================================================================
# Provider Registry
# ======================================================================

# Cloud providers: provider → (package, class, env_key)
_CLOUD_PROVIDERS: dict[str, tuple[str, str, str]] = {
    "openai":    ("langchain_openai",       "ChatOpenAI",                "OPENAI_API_KEY"),
    "anthropic": ("langchain_anthropic",    "ChatAnthropic",             "ANTHROPIC_API_KEY"),
    "google":    ("langchain_google_genai", "ChatGoogleGenerativeAI",    "GOOGLE_API_KEY"),
    "deepseek":  ("langchain_openai",       "ChatOpenAI",                "DEEPSEEK_API_KEY"),
}

# Local providers: all use ChatOpenAI with custom base_url
# provider → (default_base_url, default_api_key_placeholder)
_LOCAL_PROVIDERS: dict[str, tuple[str, str]] = {
    "local":      ("http://localhost:8000/v1",  "not-needed"),
    "ollama":     ("http://localhost:11434/v1",  "ollama"),
    "lmstudio":   ("http://localhost:1234/v1",   "lm-studio"),
    "vllm":       ("http://localhost:8000/v1",   "not-needed"),
    "llamacpp":   ("http://localhost:8080/v1",   "not-needed"),
    "xinference": ("http://localhost:9997/v1",   "not-needed"),
}

# Cloud providers that need a non-standard base_url
_CLOUD_BASE_URLS: dict[str, str] = {
    "deepseek": "https://api.deepseek.com/v1",
}

# Union for quick lookup
_ALL_PROVIDERS = set(_CLOUD_PROVIDERS) | set(_LOCAL_PROVIDERS)


# ======================================================================
# Factory
# ======================================================================

def create_llm(llm_settings: LLMSettings | None = None, **overrides: Any) -> Any:
    """Create a LangChain chat model from settings.

    For cloud providers:
      1. Try `init_chat_model` (LangChain universal)
      2. Fallback to direct class instantiation

    For local providers (ollama, vllm, lmstudio, etc.):
      → Always uses `ChatOpenAI` with the appropriate base_url.
      → API key defaults to a placeholder (local servers don't check it).

    Args:
        llm_settings: LLMSettings to use. If None, uses defaults.
        **overrides: Additional kwargs passed to the model constructor.

    Returns:
        A LangChain BaseChatModel instance.

    Raises:
        LLMConfigError: If provider unknown or package missing.
    """
    if llm_settings is None:
        from .config import LLMSettings
        llm_settings = LLMSettings()

    provider = llm_settings.provider
    model_name = llm_settings.model_name

    # ---- Local OpenAI-compatible ----
    if provider in _LOCAL_PROVIDERS:
        return _create_local_llm(llm_settings, **overrides)

    # ---- Cloud providers ----
    return _create_cloud_llm(llm_settings, **overrides)


def _create_local_llm(llm_settings: LLMSettings, **overrides: Any) -> Any:
    """Create ChatOpenAI pointing at a local server.

    Works with: Ollama, vLLM, LM Studio, llama.cpp server, Xinference,
    or any server that exposes an OpenAI-compatible /v1/chat/completions.
    """
    provider = llm_settings.provider
    model_name = llm_settings.model_name
    default_url, default_key = _LOCAL_PROVIDERS[provider]

    # Resolve base_url: user override > provider default
    base_url = llm_settings.base_url or default_url

    # Resolve API key: user setting > env > placeholder
    api_key = _resolve_api_key(llm_settings) or default_key

    kwargs: dict[str, Any] = {
        "model": model_name,
        "base_url": base_url,
        "api_key": api_key,
        "temperature": llm_settings.temperature,
        "max_tokens": llm_settings.max_tokens,
        "timeout": llm_settings.timeout,
    }
    kwargs.update(overrides)

    try:
        from langchain_openai import ChatOpenAI
    except ImportError:
        raise LLMConfigError(
            "Package 'langchain_openai' required for local models. "
            "Install with: pip install langchain-openai"
        )

    logger.info(
        "Creating local LLM: provider=%s model=%s base_url=%s",
        provider, model_name, base_url,
    )
    return ChatOpenAI(**kwargs)


def _create_cloud_llm(llm_settings: LLMSettings, **overrides: Any) -> Any:
    """Create a cloud-hosted LLM via init_chat_model or direct instantiation."""
    provider = llm_settings.provider
    model_name = llm_settings.model_name
    model_str = llm_settings.model

    kwargs: dict[str, Any] = {
        "temperature": llm_settings.temperature,
        "max_tokens": llm_settings.max_tokens,
    }

    api_key = _resolve_api_key(llm_settings)
    if api_key:
        kwargs["api_key"] = api_key

    base_url = llm_settings.base_url or _CLOUD_BASE_URLS.get(provider, "")
    if base_url:
        kwargs["base_url"] = base_url

    kwargs.update(overrides)

    # Strategy 1: init_chat_model
    try:
        from langchain.chat_models import init_chat_model
        logger.info("Creating cloud LLM via init_chat_model: %s", model_str)
        return init_chat_model(model_str, **kwargs)
    except ImportError:
        logger.debug("init_chat_model not available, trying direct instantiation")
    except Exception as e:
        logger.warning("init_chat_model failed for '%s': %s", model_str, e)

    # Strategy 2: Direct class
    if provider not in _CLOUD_PROVIDERS:
        raise LLMConfigError(
            f"Unknown provider '{provider}'. "
            f"Supported cloud: {list(_CLOUD_PROVIDERS.keys())}. "
            f"Supported local: {list(_LOCAL_PROVIDERS.keys())}."
        )

    pkg_name, class_name, _ = _CLOUD_PROVIDERS[provider]

    try:
        import importlib
        mod = importlib.import_module(pkg_name)
        cls = getattr(mod, class_name)
    except ImportError:
        raise LLMConfigError(
            f"Package '{pkg_name}' not installed. "
            f"Install with: pip install {pkg_name}"
        )
    except AttributeError:
        raise LLMConfigError(f"Class '{class_name}' not found in '{pkg_name}'.")

    if provider in ("openai", "deepseek"):
        kwargs["model"] = model_name
    elif provider == "anthropic":
        kwargs["model_name"] = model_name
    elif provider == "google":
        kwargs["model"] = model_name

    logger.info("Creating cloud LLM via %s.%s: model=%s", pkg_name, class_name, model_name)
    return cls(**kwargs)


# ======================================================================
# API Key Resolution
# ======================================================================

def _resolve_api_key(llm_settings: LLMSettings) -> str:
    """Resolve API key: settings → environment → empty.

    For local providers, returns the placeholder if nothing else is set.
    """
    import os

    # Explicit setting
    if llm_settings.api_key:
        return llm_settings.api_key

    provider = llm_settings.provider

    # Cloud env keys
    if provider in _CLOUD_PROVIDERS:
        env_key = _CLOUD_PROVIDERS[provider][2]
        val = os.environ.get(env_key, "")
        if val:
            logger.debug("API key loaded from env: %s", env_key)
            return val

    # Local: try generic env key
    if provider in _LOCAL_PROVIDERS:
        # Check provider-specific env first, then generic
        for env_name in (
            f"{provider.upper()}_API_KEY",    # e.g. OLLAMA_API_KEY
            "LOCAL_LLM_API_KEY",               # generic fallback
        ):
            val = os.environ.get(env_name, "")
            if val:
                logger.debug("Local API key from env: %s", env_name)
                return val

    return ""


# ======================================================================
# Retry Wrapper
# ======================================================================

def invoke_with_retry(
    llm: Any,
    messages: list,
    max_retries: int = 3,
    retry_delay: float = 1.0,
) -> Any:
    """Invoke LLM with exponential backoff retry.

    Args:
        llm: LangChain chat model.
        messages: List of message objects.
        max_retries: Maximum retry attempts.
        retry_delay: Initial delay in seconds (doubles each retry).

    Returns:
        The LLM response.

    Raises:
        LLMCallError: If all retries exhausted.
    """
    last_error = None
    delay = retry_delay

    for attempt in range(max_retries + 1):
        try:
            return llm.invoke(messages)
        except Exception as e:
            last_error = e
            if attempt < max_retries:
                logger.warning(
                    "LLM call failed (attempt %d/%d): %s. Retrying in %.1fs...",
                    attempt + 1, max_retries + 1, e, delay,
                )
                time.sleep(delay)
                delay *= 2
            else:
                logger.error("LLM call failed after %d attempts: %s", max_retries + 1, e)

    raise LLMCallError(reason=str(last_error), retries=max_retries)


# ======================================================================
# Executor Callback Factory
# ======================================================================

def create_llm_callback(
    llm: Any,
    max_retries: int = 3,
    retry_delay: float = 1.0,
) -> Callable[[str, str], str]:
    """Create a `(system_prompt, user_message) -> str` callback for SkillExecutor."""
    def callback(system: str, user: str) -> str:
        from langchain_core.messages import HumanMessage, SystemMessage
        messages = [SystemMessage(content=system), HumanMessage(content=user)]
        response = invoke_with_retry(llm, messages, max_retries, retry_delay)
        return response.content
    return callback


# ======================================================================
# Utility
# ======================================================================

def list_providers() -> dict[str, list[str]]:
    """List all registered providers grouped by type."""
    return {
        "cloud": list(_CLOUD_PROVIDERS.keys()),
        "local": list(_LOCAL_PROVIDERS.keys()),
    }


def check_provider(provider: str) -> dict[str, Any]:
    """Check if a provider is available.

    Returns:
        {"type": "cloud"|"local", "installed": bool, "has_key": bool,
         "package": str, "env_key": str, "base_url": str}
    """
    import os
    import importlib

    if provider in _CLOUD_PROVIDERS:
        pkg, _, env_key = _CLOUD_PROVIDERS[provider]
        installed = True
        try:
            importlib.import_module(pkg)
        except ImportError:
            installed = False
        return {
            "type": "cloud",
            "installed": installed,
            "has_key": bool(os.environ.get(env_key, "")),
            "package": pkg,
            "env_key": env_key,
            "base_url": _CLOUD_BASE_URLS.get(provider, ""),
        }

    if provider in _LOCAL_PROVIDERS:
        default_url, default_key = _LOCAL_PROVIDERS[provider]
        installed = True
        try:
            importlib.import_module("langchain_openai")
        except ImportError:
            installed = False
        return {
            "type": "local",
            "installed": installed,
            "has_key": True,  # local doesn't need real key
            "package": "langchain_openai",
            "env_key": f"{provider.upper()}_API_KEY",
            "base_url": default_url,
        }

    return {"type": "unknown", "installed": False, "has_key": False,
            "package": "?", "env_key": "?", "base_url": ""}
