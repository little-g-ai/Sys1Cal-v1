from __future__ import annotations

from typing import Any

from jev_prob_bench.adapters.generic_http import GenericHTTPAdapter
from jev_prob_bench.adapters.jev import JevAdapter, JevNoulAdapter, JevScoreAdapter
from jev_prob_bench.adapters.mock import MockAdapter
from jev_prob_bench.adapters.semif import SemIfAdapter


def build_adapter(config: dict[str, Any]):
    adapter = config.get("adapter", "mock")
    name = config.get("name", adapter)
    if adapter == "mock":
        return MockAdapter(name=name, strategy=config.get("strategy", name))
    if adapter == "jev":
        return JevAdapter(
            name=name,
            endpoint=config.get("endpoint") or _env_value(config.get("endpoint_env")),
            api_key_env=config.get("api_key_env", "JEV_API_KEY"),
            model=config.get("model", "jev-latest"),
        )
    if adapter == "jev_noul":
        return JevNoulAdapter(
            name=name,
            endpoint=config.get("endpoint") or _env_value(config.get("endpoint_env")),
            api_key_env=config.get("api_key_env", "JEV_API_KEY"),
            model=config.get("model", "jev-latest"),
        )
    if adapter == "jev_score":
        return JevScoreAdapter(
            name=name,
            endpoint=config.get("endpoint") or _env_value(config.get("endpoint_env")),
            api_key_env=config.get("api_key_env", "JEV_API_KEY"),
            model=config.get("model", "jev-latest"),
        )
    if adapter == "generic_http":
        return GenericHTTPAdapter(name=name, endpoint=config.get("endpoint"), api_key_env=config.get("api_key_env"))
    if adapter == "semif":
        return SemIfAdapter(
            name=name,
            command=config.get("command"),
            mode=config.get("mode", "direct"),
            model=config.get("model", "Qwen/Qwen3.5-4B"),
            revision=config.get("revision", "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"),
            timeout=config.get("timeout"),
            extra_args=config.get("extra_args", []),
            keep_files=config.get("keep_files", False),
            env=config.get("env"),
        )
    raise ValueError(f"unknown adapter: {adapter}")


def _env_value(name: str | None) -> str | None:
    if not name:
        return None
    import os

    return os.environ.get(name)

