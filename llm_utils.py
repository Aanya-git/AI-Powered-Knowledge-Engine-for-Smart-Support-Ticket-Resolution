# llm_utils.py
import time
from groq import Groq
from groq import BadRequestError
from config import GROQ_API_KEY
from typing import List, Dict

if not GROQ_API_KEY:
    raise ValueError("Missing GROQ_API_KEY in your .env. Add GROQ_API_KEY and restart.")

_client = Groq(api_key=GROQ_API_KEY)

def groq_chat_completion(messages: List[Dict], model: str = "llama-3.3-8b-instant",
                        max_tokens: int = 512, temperature: float = 0.2,
                        retry: int = 3, backoff_base: float = 1.2) -> str:
    """
    Wrapper for Groq chat completion with retries and exponential backoff.
    messages: list of {"role": "...", "content": "..."}
    """
    attempt = 0
    while True:
        try:
            completion = _client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return completion.choices[0].message.content.strip()
        except Exception as e:
            attempt += 1
            # If it's a known Groq error with a message, include it
            err_msg = str(e)
            if attempt > retry:
                raise RuntimeError(f"LLM failure after {attempt} attempts: {err_msg}")
            # exponential backoff with jitter
            sleep_time = (backoff_base ** attempt) + (0.1 * attempt)
            time.sleep(sleep_time)
