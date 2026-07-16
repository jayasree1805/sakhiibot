import os
import re
import time
import logging
from google import genai

logger = logging.getLogger(__name__)

MAX_RETRIES = 3
RETRY_BASE_DELAY = 2


def _get_client():
    api_key = os.getenv("gemini_token")
    if not api_key:
        raise ValueError("GEMINI_TOKEN environment variable not set")
    return genai.Client(api_key=api_key)


_client = _get_client()


def _extract_retry_delay(error_str: str) -> float:
    """
    Reads the retryDelay value from the API error message.
    Falls back to exponential backoff if not found.
    """
    match = re.search(r"retryDelay.*?(\d+)s", error_str)
    if match:
        return float(match.group(1)) + 2  # add 2s buffer
    return None


def query_gem(prompt: str) -> str:
    """
    Single entry point for all Gemini API calls.
    Uses retry delay from API response instead of fixed backoff.
    """
    for attempt in range(MAX_RETRIES):
        try:
            response = _client.models.generate_content(
                model="gemini-flash-latest",
                contents=prompt
            )
            return response.text if response.text else ""

        except Exception as e:
            error_str = str(e)

            if "429" in error_str or "500" in error_str or "503" in error_str:
                suggested = _extract_retry_delay(error_str)
                delay = suggested if suggested else RETRY_BASE_DELAY * (2 ** attempt)

                logger.warning(f"API call failed (attempt {attempt+1}/{MAX_RETRIES}): rate limited")
                logger.info(f"Retrying in {delay:.0f}s...")
                time.sleep(delay)
            else:
                raise

    logger.error("All retries exhausted.")
    raise Exception("Gemini API unavailable after retries.")