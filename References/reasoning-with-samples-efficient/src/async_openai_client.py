"""
Async OpenAI-compatible API client using aiohttp.

Used for parallel proposal generation in ParallelMCMCSampling.
"""

import asyncio
import aiohttp
from typing import Dict, List, Optional, Any
from dataclasses import dataclass


@dataclass
class ChatCompletionResponse:
    """Parsed response from chat completion API."""
    text: str
    tokens: List[str]
    log_probs: List[float]
    prompt_tokens: int
    completion_tokens: int
    finish_reason: str


class AsyncOpenAIClient:
    """
    Async OpenAI-compatible API client using aiohttp.

    Supports parallel batch requests with rate limiting via semaphore.
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.x.ai/v1",
        model: str = "grok-4-1-fast-non-reasoning",
        timeout: float = 60.0,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self._session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self) -> "AsyncOpenAIClient":
        """Context manager entry - create session."""
        self._session = aiohttp.ClientSession(
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            timeout=aiohttp.ClientTimeout(total=self.timeout),
        )
        return self

    async def __aexit__(self, *args) -> None:
        """Context manager exit - close session."""
        if self._session:
            await self._session.close()
            self._session = None

    async def _ensure_session(self) -> aiohttp.ClientSession:
        """Lazily create aiohttp session if not in context manager."""
        if self._session is None:
            self._session = aiohttp.ClientSession(
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                timeout=aiohttp.ClientTimeout(total=self.timeout),
            )
        return self._session

    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 1.0,
        max_tokens: int = 512,
        logprobs: bool = True,
        top_logprobs: int = 5,
    ) -> ChatCompletionResponse:
        """
        Make async chat completion request with retry logic.

        Returns parsed ChatCompletionResponse.
        """
        session = await self._ensure_session()

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "logprobs": logprobs,
            "top_logprobs": top_logprobs,
        }

        last_error: Optional[Exception] = None

        for attempt in range(self.max_retries):
            try:
                async with session.post(
                    f"{self.base_url}/chat/completions",
                    json=payload,
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        return self._parse_response(data)
                    elif response.status == 429:  # Rate limited
                        await asyncio.sleep(self.retry_delay * (2 ** attempt))
                        continue
                    else:
                        error_text = await response.text()
                        raise aiohttp.ClientResponseError(
                            response.request_info,
                            response.history,
                            status=response.status,
                            message=f"API error: {error_text}",
                        )
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay * (2 ** attempt))
                continue

        raise last_error or RuntimeError("Max retries exceeded")

    def _parse_response(self, data: Dict[str, Any]) -> ChatCompletionResponse:
        """Parse API response into ChatCompletionResponse."""
        choice = data["choices"][0]
        message = choice["message"]
        usage = data["usage"]

        # Extract logprobs if available
        tokens: List[str] = []
        log_probs: List[float] = []

        if choice.get("logprobs") and choice["logprobs"].get("content"):
            for token_info in choice["logprobs"]["content"]:
                tokens.append(token_info["token"])
                log_probs.append(token_info["logprob"])

        return ChatCompletionResponse(
            text=message.get("content", ""),
            tokens=tokens,
            log_probs=log_probs,
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            finish_reason=choice.get("finish_reason", ""),
        )

    async def chat_completion_batch(
        self,
        requests: List[Dict[str, Any]],
        max_concurrent: int = 10,
    ) -> List[ChatCompletionResponse]:
        """
        Execute multiple chat completions in parallel with rate limiting.

        Args:
            requests: List of kwargs dicts for chat_completion
            max_concurrent: Maximum concurrent requests (semaphore limit)

        Returns:
            List of ChatCompletionResponse in same order as requests.
            Failed requests raise exceptions.
        """
        semaphore = asyncio.Semaphore(max_concurrent)

        async def bounded_request(req: Dict[str, Any]) -> ChatCompletionResponse:
            async with semaphore:
                return await self.chat_completion(**req)

        tasks = [bounded_request(req) for req in requests]
        results = await asyncio.gather(*tasks, return_exceptions=False)
        return results
