from __future__ import annotations

import json
import os
import urllib.request
from typing import Any

from jev_prob_bench.adapters.base import ModelAdapter


class GenericHTTPAdapter(ModelAdapter):
    def __init__(
        self,
        name: str = "generic_http",
        endpoint: str | None = None,
        api_key_env: str | None = None,
        timeout: float = 60.0,
    ) -> None:
        super().__init__(name)
        self.endpoint = endpoint or os.environ.get("GENERIC_PROB_BENCH_URL", "")
        self.api_key_env = api_key_env
        self.timeout = timeout

    def predict_distribution(
        self,
        state: dict[str, Any],
        prompt: str,
        outcomes: list[str],
        item: dict[str, Any] | None = None,
    ) -> dict[str, float]:
        if not self.endpoint:
            raise RuntimeError("generic HTTP adapter requires an endpoint")
        body = json.dumps({"state": state, "prompt": prompt, "outcomes": outcomes}).encode()
        headers = {"Content-Type": "application/json"}
        if self.api_key_env and os.environ.get(self.api_key_env):
            headers["Authorization"] = f"Bearer {os.environ[self.api_key_env]}"
        request = urllib.request.Request(self.endpoint, data=body, headers=headers, method="POST")
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            payload = json.loads(response.read().decode())
        self.last_metadata = {"raw_response": payload, "model_version": payload.get("model_version")}
        return payload.get("distribution") or payload.get("predicted_distribution") or payload

