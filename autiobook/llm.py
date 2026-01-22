"""llm integration for script and cast generation."""

import json
import os
import time
from dataclasses import dataclass
from typing import Callable, List, Optional, TypeVar

from openai import OpenAI

from .config import DEFAULT_LLM_MODEL, LLM_MAX_RETRIES, LLM_RETRY_DELAY

T = TypeVar("T")


def retry_with_backoff(
    fn: Callable[[], T],
    max_retries: int = LLM_MAX_RETRIES,
    initial_delay: float = LLM_RETRY_DELAY,
) -> T:
    """retry a function with exponential backoff on API or JSON errors."""
    delay = initial_delay
    last_error = None

    for attempt in range(max_retries + 1):
        try:
            return fn()
        except json.JSONDecodeError as e:
            last_error = e
            if attempt < max_retries:
                print(f"  json parse error, retrying in {delay:.1f}s...")
                time.sleep(delay)
                delay *= 2
        except Exception as e:
            # catch API errors (connection, rate limit, etc.)
            last_error = e
            if attempt < max_retries:
                print(f"  api error: {e}, retrying in {delay:.1f}s...")
                time.sleep(delay)
                delay *= 2

    raise last_error


@dataclass
class Character:
    name: str
    description: str  # visual/vocal description for VoiceDesign
    audition_line: str  # short text to generate the reference voice
    aliases: list[str] | None = None  # alternate names for the same character
    appearances: int = 0  # number of speaking appearances in scripts


@dataclass
class ScriptSegment:
    speaker: str
    text: str
    instruction: str  # e.g., "laughing", "whispering", "angry"


def get_client(api_base: Optional[str] = None, api_key: Optional[str] = None) -> OpenAI:
    """create openai client with optional overrides."""
    return OpenAI(
        base_url=api_base or os.getenv("OPENAI_BASE_URL"),
        api_key=api_key or os.getenv("OPENAI_API_KEY"),
    )


def generate_cast(
    text_sample: str,
    api_base: Optional[str] = None,
    api_key: Optional[str] = None,
    model: str = DEFAULT_LLM_MODEL,
    existing_cast_summary: Optional[str] = None,
) -> List[Character]:
    """analyze text to identify characters and generate voice descriptions."""
    client = get_client(api_base, api_key)

    context_str = ""
    if existing_cast_summary:
        context_str = f"\nExisting Cast (for reference):\n{existing_cast_summary}\n"

    prompt = f"""
You are an expert audiobook director. Analyze the following text sample from a book.
Identify the main characters appearing in this text. For each character, provide:
1. Name (use the character's primary/full name)
2. A detailed vocal description (timbre, pitch, speed, gender, age) suitable for TTS.
3. A short "audition line" (1-2 sentences) that captures their typical style/personality.
4. Aliases: other names this character is referred to by (nicknames, titles, first/last name only, etc.)
{context_str}
Instructions:
- If a character is new, create a full profile.
- If a character exists in the "Existing Cast" list, check if this reveals NEW traits or aliases.
- If there is new info, UPDATE their description (merge new details with old ones).
- If existing profile is accurate, you can still output it to confirm presence.
- ALWAYS include a 'Narrator' character.
- IMPORTANT: Detect when different names refer to the SAME person (e.g., "Dr. Smith" and "John" are the same person).
  Use the most complete name as "name" and list variations as "aliases".

Output strictly valid JSON in this format:
[
    {{"name": "Narrator", "description": "...", "audition_line": "...", "aliases": []}},
    {{"name": "Dr. John Smith", "description": "...", "audition_line": "...", "aliases": ["John", "Dr. Smith", "the doctor"]}}
]
"""

    def _call_api() -> List[Character]:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": text_sample},
            ],
            response_format={"type": "json_object"},
        )

        content = response.choices[0].message.content
        data = _parse_json_response(content)

        # handle potential wrapper keys like {"characters": [...]}
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, list):
                    data = value
                    break

        # normalize aliases to list or None
        characters = []
        for c in data:
            aliases = c.get("aliases")
            if aliases and isinstance(aliases, list) and len(aliases) > 0:
                c["aliases"] = aliases
            else:
                c["aliases"] = None
            characters.append(Character(**c))
        return characters

    return retry_with_backoff(_call_api)


def _parse_json_response(content: str) -> dict | list:
    """parse JSON from LLM response, handling markdown code blocks."""
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
            return json.loads(content)
        raise


def split_text_smart(text: str, max_words: int = 1500) -> List[str]:
    """split text into chunks at paragraph boundaries."""
    paragraphs = text.split("\n\n")
    chunks = []
    current_chunk = []
    current_count = 0

    for p in paragraphs:
        word_count = len(p.split())
        if current_count + word_count > max_words and current_chunk:
            chunks.append("\n\n".join(current_chunk))
            current_chunk = []
            current_count = 0

        current_chunk.append(p)
        current_count += word_count

    if current_chunk:
        chunks.append("\n\n".join(current_chunk))

    return chunks


def process_script_chunk(
    text_chunk: str,
    cast: List[Character],
    api_base: Optional[str] = None,
    api_key: Optional[str] = None,
    model: str = DEFAULT_LLM_MODEL,
) -> List[ScriptSegment]:
    """convert a text chunk into dramatized script segments."""
    client = get_client(api_base, api_key)

    cast_info = []
    for c in cast:
        if c.aliases:
            cast_info.append(f"{c.name} (also known as: {', '.join(c.aliases)})")
        else:
            cast_info.append(c.name)
    cast_str = "\n- ".join(cast_info)

    prompt = f"""
You are an expert audiobook scriptwriter. Convert the following text into a script for TTS.

Available Characters:
- {cast_str}

Rules:
1. Break the text into segments based on who is speaking. Always separate non-dialogue text from spoken text.
2. Use "Narrator" as the speaker for all NON-DIALOGUE text, including attribution such as ("John said", "she whispered"). NEVER associate the Narrator with quoted dialogue.
3. Use the appropriate Character as the speaker for ALL quotes and dialogue. The Character should voice ONLY the words inside quotation marks,
4. Provide a SHORT 'instruction' for the voice actor for EVERY segment (e.g., "neutral", "whispering", "laughing", "shouting", "sadly", "excitedly").
5. Keep the text EXACTLY as written in the book. Do not paraphrase or omit.

Example input: "Hello there," John said with a smile. "How are you?"
Example output (strictly valid JSON):
```
{{
    "segments": [
        {{"s": "John", "t": "Hello there.", "i": "friendly greeting"}}
        {{"s": "Narrator", "t": "John said with a smile.", "i": "narrative"}},
        {{"s": "John", "t": "How are you?", "i": "warm, curious"}}
    ]
}}
```
"""

    def _call_api() -> List[ScriptSegment]:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": text_chunk},
            ],
            response_format={"type": "json_object"},
        )

        content = response.choices[0].message.content
        data = _parse_json_response(content)
        segments = []
        for s in data.get("segments", []):
            segments.append(ScriptSegment(
                speaker=s["s"],
                text=s["t"],
                instruction=s["i"],
            ))
        return segments

    return retry_with_backoff(_call_api)
