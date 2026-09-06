"""Load the local model into memory before anyone asks the first question.

Run in the background at container start. Ollama loads lazily, so without this
the first question pays the full model-load cost -- and that first question is
the one an evaluator is watching.
"""

from __future__ import annotations

from src.talk_to_data.llm_client import OllamaClient, get_llm_client
from src.utils.logger import get_logger

logger = get_logger(__name__)


def main() -> None:
    """Warm the model if the local provider is the one in use."""
    client = get_llm_client()
    if not isinstance(client, OllamaClient):
        logger.info("Provider is not the local runtime; nothing to warm up")
        return

    available, message = client.is_available()
    if not available:
        # Normal on first start while the model downloads.
        logger.info("Skipping warm-up: %s", message)
        return
    client.warm_up()


if __name__ == "__main__":
    main()
