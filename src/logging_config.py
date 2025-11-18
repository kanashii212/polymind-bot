import logging
import os
import re
from typing import Optional


class RedactFilter(logging.Filter):
    """Filter to redact sensitive values from log messages.

    - Removes common tokens, init_data, and other auth artifacts from logged strings.
    - Applied to all StreamHandlers to avoid leaking production secrets.
    """

    TOKEN_REPLACEMENTS = [
        (re.compile(r"(?i)(bot[-_]?token)\s*[:=]\s*\S+"), r"\1=<REDACTED>"),
        (re.compile(r"(?i)(authorization)\s*[:=]\s*\S+"), r"\1=<REDACTED>"),
        (re.compile(r"tma\s+\S+"), "tma <REDACTED>"),
        (re.compile(r"(?i)(init_data|initData)\s*[:=]\s*\S+"), r"\1=<REDACTED>"),
    ]

    SENSITIVE_ENV = [
        "TELEGRAM_BOT_TOKEN",
        "GEMINI_API_KEY",
        "OPENROUTER_API_KEY",
        "OPENAI_API_KEY",
        "MONGODB_URI",
        "MONGO_URI",
        "DB_NAME",
        "API_KEY",
        "SECRET_KEY",
    ]

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            if isinstance(record.msg, str):
                new_msg = record.msg
                for pattern, repl in self.TOKEN_REPLACEMENTS:
                    new_msg = pattern.sub(repl, new_msg)
                for env in self.SENSITIVE_ENV:
                    try:
                        val = os.getenv(env)
                        if val and isinstance(val, str) and val.strip():
                            new_msg = new_msg.replace(val, "<REDACTED>")
                    except Exception:
                        pass
                # Also inspect the final formatted message
                try:
                    formatted = record.getMessage()
                    for pattern, repl in self.TOKEN_REPLACEMENTS:
                        formatted = pattern.sub(repl, formatted)
                    for env in self.SENSITIVE_ENV:
                        val = os.getenv(env)
                        if val and isinstance(val, str) and val.strip():
                            formatted = formatted.replace(val, "<REDACTED>")
                    # Replace the record msg with the sanitized full string
                    record.msg = formatted
                    # Clear args to avoid re-formatting unredacted values
                    record.args = ()
                except Exception:
                    pass
                record.msg = new_msg
        except Exception:
            # Don't interrupt logging if redaction fails
            pass
        return True


def setup_logging(level: Optional[int] = None) -> None:
    """Configure global logging for the application.

    - Default level is WARNING in production and INFO in development.
    - Adds RedactFilter to all handlers to avoid leaking secrets.
    - Optionally disables debug logs from external libraries.
    """

    dev_mode = os.getenv("DEV_MODE", "false").lower() == "true"
    requested_level = level or (
        logging.DEBUG if dev_mode else os.getenv("LOG_LEVEL", "WARNING")
    )

    # Normalize the level
    if isinstance(requested_level, str):
        requested_level = getattr(logging, requested_level.upper(), logging.WARNING)

    # Clear existing handlers and reconfigure
    for h in list(logging.root.handlers):
        logging.root.removeHandler(h)

    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=requested_level,
        handlers=[logging.StreamHandler()],
    )

    # Apply redact filter to all current handlers
    redact = RedactFilter()
    for handler in logging.root.handlers:
        handler.addFilter(redact)

    # Reduce verbosity of noisy libraries in production
    if not dev_mode:
        logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("asyncio").setLevel(logging.WARNING)
