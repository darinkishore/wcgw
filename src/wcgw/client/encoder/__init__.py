import logging
import threading
from typing import Callable, Protocol, TypeVar, cast

import tokenizers  # type: ignore[import-untyped]

logger = logging.getLogger("wcgw")

T = TypeVar("T")


class EncoderDecoder(Protocol[T]):
    def encoder(self, text: str) -> list[T]: ...

    def decoder(self, tokens: list[T]) -> str: ...


class LazyEncoder:
    def __init__(self) -> None:
        self._tokenizer: tokenizers.Tokenizer | None = None
        self._init_lock = threading.Lock()
        self._init_error: Exception | None = None
        self._init_thread = threading.Thread(target=self._initialize, daemon=True)
        self._init_thread.start()

    def _initialize(self) -> None:
        with self._init_lock:
            if self._tokenizer is None:
                try:
                    logger.info("Initializing tokenizer from Xenova/claude-tokenizer")
                    # Set SSL cert bundle for this thread if needed
                    import os
                    import certifi
                    if not os.environ.get("REQUESTS_CA_BUNDLE"):
                        os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()
                    if not os.environ.get("SSL_CERT_FILE"):
                        os.environ["SSL_CERT_FILE"] = certifi.where()
                    
                    self._tokenizer = tokenizers.Tokenizer.from_pretrained(
                        "Xenova/claude-tokenizer"
                    )
                    logger.info("Tokenizer initialized successfully")
                except Exception as e:
                    self._init_error = e
                    logger.error(f"Failed to initialize tokenizer: {e}")

    def _ensure_initialized(self) -> None:
        if self._tokenizer is None:
            with self._init_lock:
                if self._tokenizer is None:
                    self._init_thread.join()

    def encoder(self, text: str) -> list[int]:
        self._ensure_initialized()
        if self._tokenizer is None:
            error_msg = "Couldn't initialize tokenizer"
            if self._init_error:
                error_msg += f": {self._init_error}"
            raise RuntimeError(error_msg)
        return cast(list[int], self._tokenizer.encode(text).ids)

    def decoder(self, tokens: list[int]) -> str:
        self._ensure_initialized()
        if self._tokenizer is None:
            error_msg = "Couldn't initialize tokenizer"
            if self._init_error:
                error_msg += f": {self._init_error}"
            raise RuntimeError(error_msg)
        return cast(str, self._tokenizer.decode(tokens))


def get_default_encoder() -> EncoderDecoder[int]:
    return LazyEncoder()
