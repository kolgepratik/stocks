"""
LLM Resilience & Multi-Provider Model Handler
============================================
Supports Anthropic (Claude Haiku 5.5, Sonnet, Opus) and Google Gemini (Flash, Pro).
By default, routes all calls to Claude Haiku 5.5 with no fallback unless explicitly enabled.
"""
from __future__ import annotations

from typing import Any, Type, TypeVar
from loguru import logger
from langchain_core.messages import BaseMessage
from pydantic import BaseModel

from config.settings import get_settings

T = TypeVar("T", bound=BaseModel)


def build_llm(
    model_name: str | None = None,
    provider: str | None = None,
    temperature: float | None = None,
    request_timeout: int = 45,
):
    """
    Constructs an LLM client for Anthropic Claude or Google Gemini based on settings or parameters.
    """
    settings = get_settings()
    active_provider = (provider or settings.llm_provider).lower()

    # Detect provider from model name if prefix is obvious
    if model_name:
        if "claude" in model_name.lower():
            active_provider = "anthropic"
        elif "gemini" in model_name.lower():
            active_provider = "gemini"

    if active_provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        m_name = model_name or settings.anthropic_model
        kwargs: dict[str, Any] = {
            "model": m_name,
            "api_key": settings.anthropic_api_key,
            "default_request_timeout": float(request_timeout),
        }
        # Note: Haiku 5.5 and reasoning models disallow temperature overrides
        if temperature is not None and "haiku-5" not in m_name.lower():
            kwargs["temperature"] = temperature

        return ChatAnthropic(**kwargs)

    elif active_provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        m_name = model_name or settings.gemini_model
        return ChatGoogleGenerativeAI(
            model=m_name,
            google_api_key=settings.gemini_api_key,
            temperature=temperature if temperature is not None else 0.2,
            request_timeout=request_timeout,
        )
    else:
        raise ValueError(f"Unsupported LLM provider: {active_provider}")


def invoke_structured_with_fallback(
    messages: list[BaseMessage],
    schema: Type[T],
    primary_model: str | None = None,
    candidate_models: list[str] | None = None,
    temperature: float | None = None,
    request_timeout: int = 45,
) -> T:
    """
    Execute structured output generation using the default model (Claude Haiku 5.5).
    If fallback is disabled (settings.enable_llm_fallback == False), it invokes the primary model directly.
    """
    settings = get_settings()

    if settings.llm_provider == "anthropic":
        primary = primary_model or settings.anthropic_model
    else:
        primary = primary_model or settings.gemini_model

    # If fallback is explicitly disabled, call primary model directly with no cascade
    if not settings.enable_llm_fallback:
        logger.info(
            "🧠 Invoking LLM: {} (provider={}, fallback=DISABLED, timeout={}s)",
            primary,
            settings.llm_provider,
            request_timeout,
        )
        llm = build_llm(model_name=primary, temperature=temperature, request_timeout=request_timeout)
        structured_llm = llm.with_structured_output(schema)
        result = structured_llm.invoke(messages)
        if result is None:
            raise RuntimeError(f"Model '{primary}' returned empty output")
        logger.info("✅ LLM structured output succeeded using model: {}", primary)
        return result

    # Multi-tier cascade (only if fallback is enabled)
    if candidate_models is None:
        if "claude" in primary.lower():
            cascade = [primary, "gemini-3.5-flash", "gemini-3.5-flash-lite"]
        elif "3.8" in primary:
            cascade = [primary, "gemini-3.5-flash", "gemini-3.5-flash-lite"]
        else:
            cascade = [primary, "gemini-3.8-flash", "gemini-3.5-flash-lite"]
    else:
        cascade = [primary] + [m for m in candidate_models if m != primary]

    last_error = None
    for model_name in cascade:
        try:
            logger.info("🧠 Invoking LLM: {} (timeout={}s)", model_name, request_timeout)
            llm = build_llm(model_name=model_name, temperature=temperature, request_timeout=request_timeout)
            structured_llm = llm.with_structured_output(schema)
            result = structured_llm.invoke(messages)
            if result is not None:
                logger.info("✅ LLM structured output succeeded using model: {}", model_name)
                return result
        except Exception as exc:
            last_error = exc
            err_msg = str(exc)
            logger.warning(
                "⚠️ Model '{}' encountered error: {}. Cascading to next fallback model...",
                model_name,
                err_msg[:180],
            )

    raise last_error if last_error else RuntimeError("All cascade models failed")
