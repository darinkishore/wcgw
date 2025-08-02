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
        # Don't initialize in background - do it on first use

    def _ensure_initialized(self) -> None:
        if self._tokenizer is None:
            with self._init_lock:
                if self._tokenizer is None:
                    try:
                        logger.info("Initializing tokenizer from Xenova/claude-tokenizer")
                        
                        # Method 1: Try direct file load if we can find it
                        import os
                        from pathlib import Path
                        hf_home = os.environ.get("HF_HOME", os.path.join(os.path.expanduser("~"), ".cache", "huggingface"))
                        tokenizer_paths = list(Path(hf_home).rglob("**/tokenizer.json"))
                        
                        for tf in tokenizer_paths:
                            if "Xenova" in str(tf) and "claude" in str(tf):
                                logger.info(f"Found tokenizer file at: {tf}")
                                try:
                                    self._tokenizer = tokenizers.Tokenizer.from_file(str(tf))
                                    logger.info("Tokenizer loaded from direct file path")
                                    return
                                except Exception as e:
                                    logger.warning(f"Direct file load failed: {e}")
                        
                        # Method 2: Try with local files first
                        try:
                            self._tokenizer = tokenizers.Tokenizer.from_pretrained(
                                "Xenova/claude-tokenizer", 
                                local_files_only=True
                            )
                            logger.info("Tokenizer loaded from cache")
                        except Exception as e:
                            logger.info(f"Local cache load failed: {e}")
                            # Method 3: Fall back to downloading
                            self._tokenizer = tokenizers.Tokenizer.from_pretrained(
                                "Xenova/claude-tokenizer"
                            )
                            logger.info("Tokenizer downloaded and initialized")
                    except Exception as e:
                        self._init_error = e
                        logger.error(f"Failed to initialize tokenizer: {e}")
                        # As a last resort, use character fallback
                        logger.warning("Using fallback character tokenizer")
                        self._tokenizer = self._create_fallback_tokenizer()

    def _create_fallback_tokenizer(self) -> None:
        """Use simple character splitting as fallback - not a real tokenizer"""
        # This is a dummy that just returns None to signal we should use char splitting
        return None

    def encoder(self, text: str) -> list[int]:
        self._ensure_initialized()
        if self._tokenizer is None:
            # Fallback: simple character encoding
            logger.warning("Using fallback character encoding")
            return [ord(c) for c in text]
        return cast(list[int], self._tokenizer.encode(text).ids)

    def decoder(self, tokens: list[int]) -> str:
        self._ensure_initialized()
        if self._tokenizer is None:
            # Fallback: simple character decoding
            logger.warning("Using fallback character decoding")
            return ''.join(chr(t) if t < 0x110000 else '?' for t in tokens)
        return cast(str, self._tokenizer.decode(tokens))


def get_default_encoder() -> EncoderDecoder[int]:
    return LazyEncoder()
