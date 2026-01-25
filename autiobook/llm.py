"""llm integration for script and cast generation."""

import json
import os
import time
from dataclasses import dataclass
from typing import Callable, List, Optional, TypeVar, cast

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
    last_error: Exception = RuntimeError("unknown error in retry_with_backoff")

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
        context_str = f"\nExisting Cast:\n{existing_cast_summary}\n"

    prompt = f"""
Identify book characters. Output NEW or UPDATED profiles.
Rules:
1. Keys: n (name), al (aliases list), d (vocal description), a (audition line).
   - 'd' includes: timbre, pitch, speed, gender, age.
2. Deduplicate: Use full name for 'n', variations in 'al'.
3. Omit existing characters unless updating.

Existing Cast:
{context_str}

JSON: {{"c":[{{"n":"Narrator","d":"...","a":"...","al":[]}}]}}
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
        if not content:
            raise RuntimeError("llm returned empty content")
        data = _parse_json_response(content)

        # handle potential wrapper keys
        if isinstance(data, dict):
            if "c" in data:
                data = data["c"]
            elif "characters" in data:
                data = data["characters"]
            else:
                for key, value in data.items():
                    if isinstance(value, list):
                        data = value
                        break

        characters = []
        for c in data:
            if not isinstance(c, dict):
                continue
            characters.append(
                Character(
                    name=str(c.get("n", c.get("name", ""))),
                    description=str(c.get("d", c.get("description", ""))),
                    audition_line=str(c.get("a", c.get("audition_line", ""))),
                    aliases=c.get("al", c.get("aliases")) or None,
                )
            )
        return characters

    return retry_with_backoff(_call_api)


def _parse_json_response(content: str) -> dict | list:
    """parse JSON from LLM response, handling markdown code blocks."""
    try:
        return cast(dict | list, json.loads(content))
    except json.JSONDecodeError:
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
            return cast(dict | list, json.loads(content))
        raise


def split_text_smart(text: str, max_words: int = 1500) -> List[str]:
    """split text into chunks at paragraph boundaries."""
    paragraphs = text.split("\n\n")
    chunks = []
    current_chunk: List[str] = []
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
    cast_str = _format_cast_list(cast)

    prompt = f"""
Convert text to JSON script.
Cast: {cast_str}
Rules:
1. Segments: Separate quotes from narration.
2. Narrator: All unquoted text.
3. Characters: Only words inside quotes. Use "Extra Female/Male" if unknown.
4. Keys: s (speaker), t (text), i (mood tag).
5. EXACT text only.

Example: "Hi," John said. ->
{{"seg":[{{"s":"John","t":"Hi.","i":"warm"}},{{"s":"Narrator","t":"John said.","i":"narrative"}}]}}
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
        if not content:
            raise RuntimeError("llm returned empty content")
        data = _parse_json_response(content)
        segments = []
        seg_list = []
        if isinstance(data, dict):
            seg_list = data.get("seg", data.get("segments", []))
        elif isinstance(data, list):
            seg_list = data

        for s in seg_list:
            segments.append(
                ScriptSegment(
                    speaker=s["s"],
                    text=s["t"],
                    instruction=s["i"],
                )
            )
        return segments

    return retry_with_backoff(_call_api)


def _format_cast_list(cast: List[Character]) -> str:
    """format cast list for LLM prompts."""
    cast_info = []
    for c in cast:
        if c.aliases:
            cast_info.append(f"{c.name} (also known as: {', '.join(c.aliases)})")
        else:
            cast_info.append(c.name)
    return "\n- ".join(cast_info)


def fix_missing_segment(
    missing_text: str,
    context_before: str,
    context_after: str,
    cast: List[Character],
    api_base: Optional[str] = None,
    api_key: Optional[str] = None,
    model: str = DEFAULT_LLM_MODEL,
) -> List[ScriptSegment]:
    """convert a missing text fragment into script segments using surrounding context."""
    client = get_client(api_base, api_key)
    cast_str = _format_cast_list(cast)

    prompt = f"""
Convert ONLY "MISSING TEXT" to JSON segments.
Cast: {cast_str}
Rules:
1. Use CONTEXT for speaker/tone; do NOT output it.
2. Narrator: All unquoted text.
3. Characters: Words inside quotes.
4. Keys: s (speaker), t (text), i (mood tag).
5. EXACT text only.

Output JSON: {{"seg":[{{"s":"...","t":"...","i":"..."}}]}}
"""

    user_content = f"""
CONTEXT BEFORE:
{context_before}

--- MISSING TEXT (convert this) ---
{missing_text}
--- END MISSING TEXT ---

CONTEXT AFTER:
{context_after}
"""

    def _call_api() -> List[ScriptSegment]:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": user_content},
            ],
            response_format={"type": "json_object"},
        )

        content = response.choices[0].message.content
        if not content:
            raise RuntimeError("llm returned empty content")
        data = _parse_json_response(content)
        segments = []
        seg_list = []
        if isinstance(data, dict):
            seg_list = data.get("seg", data.get("segments", []))
        elif isinstance(data, list):
            seg_list = data

        for s in seg_list:
            segments.append(
                ScriptSegment(
                    speaker=s["s"],
                    text=s["t"],
                    instruction=s["i"],
                )
            )
        return segments

    return retry_with_backoff(_call_api)
