"""llm integration for script and cast generation."""

import difflib
import json
import re
import time
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, TypeVar, cast

import litellm

from .config import (
    DEFAULT_LLM_MODEL,
    DEFAULT_THINKING_BUDGET,
    LLM_MAX_RETRIES,
    LLM_RETRY_DELAY,
)

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
                print(f"  json parse error: {e}, retrying in {delay:.1f}s...")
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
    gender: str | None = None  # 'm' or 'f', used for fallback voice mapping


@dataclass
class ScriptSegment:
    speaker: str
    text: str
    instruction: str  # e.g., "laughing", "whispering", "angry"


def _strip_thinking_tokens(content: str) -> str:
    """remove thinking/reasoning blocks from LLM response."""
    # handle <think>...</think> blocks (DeepSeek, Qwen, etc.)
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)
    # handle <reasoning>...</reasoning> blocks
    content = re.sub(r"<reasoning>.*?</reasoning>", "", content, flags=re.DOTALL)
    return content.strip()


def _parse_json_response(content: str) -> dict | list:
    """parse JSON from LLM response, handling markdown code blocks and thinking tokens."""
    content = _strip_thinking_tokens(content)
    if content.startswith("```json"):
        content = content[7:]
    elif content.startswith("```"):
        content = content[3:]
    if content.endswith("```"):
        content = content[:-3]

    content = content.strip()
    try:
        return cast(dict | list, json.loads(content))
    except json.JSONDecodeError:
        # handle trailing garbage (extra data) from some LLMs
        obj, _ = json.JSONDecoder().raw_decode(content)
        return cast(dict | list, obj)


def _query_llm_json(
    system_prompt: str,
    user_prompt: str,
    model: str,
    api_base: Optional[str] = None,
    api_key: Optional[str] = None,
    wrapper_keys: List[str] | None = None,
    thinking_budget: int = DEFAULT_THINKING_BUDGET,
) -> dict | list:
    """query LLM via litellm and return JSON data.

    model must include provider prefix (e.g., openai/gpt-4o, anthropic/claude-3-5-sonnet).
    see https://docs.litellm.ai/docs/providers for supported providers.
    """
    # disable thinking for Qwen3 models when thinking_budget is 0
    if thinking_budget <= 0:
        user_prompt = "/no_think\n" + user_prompt

    kwargs: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "response_format": {"type": "json_object"},
    }

    if api_base:
        kwargs["api_base"] = api_base
    if api_key:
        kwargs["api_key"] = api_key
    if thinking_budget > 0:
        # pass via extra_body to bypass litellm's param validation for custom endpoints
        kwargs["extra_body"] = {
            "thinking": {"type": "enabled", "budget_tokens": thinking_budget}
        }

    def _call():
        res = litellm.completion(**kwargs)
        # litellm types are incomplete; access response data via Any
        choices: Any = res.choices
        content: str | None = choices[0].message.content
        if not content:
            raise RuntimeError("llm returned empty content")

        data = _parse_json_response(content)

        # unwrap nested structures when wrapper_keys specified
        if wrapper_keys:
            # handle {"seg": [...]} or {"segments": [...]}
            if isinstance(data, dict):
                for key in wrapper_keys:
                    if key in data:
                        return data[key]
            # handle [{"seg": [...]}] - single-element list wrapping
            if isinstance(data, list) and len(data) == 1 and isinstance(data[0], dict):
                for key in wrapper_keys:
                    if key in data[0]:
                        return data[0][key]
        return data

    return cast(dict | list, retry_with_backoff(_call))


def scan_chapter_characters(
    text: str,
    api_base: Optional[str] = None,
    api_key: Optional[str] = None,
    model: str = DEFAULT_LLM_MODEL,
    thinking_budget: int = DEFAULT_THINKING_BUDGET,
) -> list[dict]:
    """scan a chapter and return all speaking characters with counts and gender."""
    prompt = """List every character who SPEAKS (has dialogue) in this text.
Output ONLY valid JSON, no markdown.

For each character:
- n: name exactly as it appears in the text
- c: number of times they speak (dialogue lines count)
- g: gender ("m" or "f")
- al: list of alternate names/titles used for the same character in the text

Include "Narrator" for the narrative voice.

Output format (REQUIRED):
{"chars":[{"n":"name","c":5,"g":"m","al":["alias1"]}]}
"""

    data = _query_llm_json(
        prompt,
        text,
        model,
        api_base,
        api_key,
        wrapper_keys=["chars", "characters"],
        thinking_budget=thinking_budget,
    )

    results = []
    for item in cast(list, data):
        if not isinstance(item, dict):
            continue
        results.append({
            "name": str(item.get("n", item.get("name", ""))),
            "count": int(item.get("c", item.get("count", 1))),
            "gender": str(item.get("g", item.get("gender", "m"))),
            "aliases": item.get("al", item.get("aliases", [])) or [],
        })
    return results


def _cluster_by_name(entries: list[dict]) -> list[dict]:
    """programmatic dedup pass: cluster entries by surname/substring similarity."""
    merged: list[dict] = []
    used: set[str] = set()

    for entry in entries:
        key = entry["name"].lower()
        if key in used:
            continue

        cluster = [entry]
        used.add(key)

        for other in entries:
            other_key = other["name"].lower()
            if other_key in used:
                continue
            if _names_match(entry["name"], other["name"]):
                cluster.append(other)
                used.add(other_key)

        canonical = min(cluster, key=lambda x: len(x["name"]))
        total_count = sum(c["count"] for c in cluster)
        all_aliases: set[str] = set()
        for c in cluster:
            all_aliases.add(c["name"])
            all_aliases.update(c.get("aliases", []))
        all_aliases.discard(canonical["name"])

        merged.append({
            "name": canonical["name"],
            "count": total_count,
            "gender": canonical["gender"],
            "aliases": sorted(all_aliases) if all_aliases else [],
        })

    return merged


def _names_match(a: str, b: str) -> bool:
    """check if two character names refer to the same person."""
    a_low, b_low = a.lower(), b.lower()
    if a_low == b_low:
        return True
    a_surname = _last_word(a_low)
    b_surname = _last_word(b_low)
    if len(a_surname) >= 3 and len(b_surname) >= 3:
        if a_surname == b_surname:
            return True
        if _fuzzy_ratio(a_surname, b_surname) >= 0.75:
            return True
    if len(a_low) >= 3 and len(b_low) >= 3:
        if a_low in b_low or b_low in a_low:
            return True
    return False


def merge_scanned_characters(
    all_scans: list[list[dict]],
    api_base: Optional[str] = None,
    api_key: Optional[str] = None,
    model: str = DEFAULT_LLM_MODEL,
    thinking_budget: int = DEFAULT_THINKING_BUDGET,
) -> list[dict]:
    """merge character lists: LLM semantic merge, then programmatic dedup."""
    # flatten all scans
    raw: dict[str, dict] = {}
    for scan in all_scans:
        for ch in scan:
            name = ch["name"]
            key = name.lower().strip()
            if not key:
                continue
            if key in raw:
                raw[key]["count"] += ch["count"]
                raw[key]["aliases"] = list(set(raw[key]["aliases"]) | set(ch.get("aliases", [])))
            else:
                raw[key] = {
                    "name": name,
                    "count": ch["count"],
                    "gender": ch.get("gender", "m"),
                    "aliases": list(ch.get("aliases", [])),
                }

    char_list = sorted(raw.values(), key=lambda x: -x["count"])
    summary = json.dumps(char_list, ensure_ascii=False, indent=2)

    # pass 1: LLM merge (catches semantic duplicates like nicknames)
    prompt = """Merge and deduplicate this character list. Output ONLY valid JSON.

Rules:
1. Characters that are the same person under different names/titles/spellings must be merged into ONE entry.
2. Pick the shortest common name as "n". Put all variants in "al".
3. Sum up "c" (counts) for merged characters.
4. Keep "g" (gender).
5. Remove entries with c=0.
6. Each real person must appear exactly once.

Output format (REQUIRED):
{"chars":[{"n":"canonical name","c":total_count,"g":"m","al":["alias1","alias2"]}]}
"""

    data = _query_llm_json(
        prompt,
        summary,
        model,
        api_base,
        api_key,
        wrapper_keys=["chars", "characters"],
        thinking_budget=thinking_budget,
    )

    llm_merged = []
    for item in cast(list, data):
        if not isinstance(item, dict):
            continue
        llm_merged.append({
            "name": str(item.get("n", item.get("name", ""))),
            "count": int(item.get("c", item.get("count", 1))),
            "gender": str(item.get("g", item.get("gender", "m"))),
            "aliases": item.get("al", item.get("aliases", [])) or [],
        })

    # pass 2: programmatic dedup (catches what LLM missed)
    result = _cluster_by_name(llm_merged)

    # filter zero-count and ensure Narrator
    result = [m for m in result if m["count"] > 0]
    for m in result:
        if m["name"].lower() in ("narrator", "\u043d\u0430\u0440\u0440\u0430\u0442\u043e\u0440", "\u0440\u0430\u0441\u0441\u043a\u0430\u0437\u0447\u0438\u043a"):
            if m["name"] != "Narrator":
                m["aliases"].append(m["name"])
                m["name"] = "Narrator"
            break

    return sorted(result, key=lambda x: -x["count"])


def generate_cast_from_scan(
    scanned_characters: list[dict],
    text_sample: str,
    api_base: Optional[str] = None,
    api_key: Optional[str] = None,
    model: str = DEFAULT_LLM_MODEL,
    thinking_budget: int = DEFAULT_THINKING_BUDGET,
) -> List[Character]:
    """generate voice descriptions for pre-scanned characters."""
    char_summary = json.dumps(scanned_characters, ensure_ascii=False, indent=2)

    prompt = f"""Generate voice descriptions for these book characters. Output ONLY valid JSON.

Characters to describe:
{char_summary}

Rules:
1. For each character: n (EXACT same name), d (vocal description), a (audition line), al (EXACT same aliases), g (EXACT same gender).
2. Vocal description 'd': timbre, pitch, speed, gender, age.
3. Audition line 'a': write in the SAME LANGUAGE as the text sample below. Use a short quote or typical line.
4. Keep the EXACT name from the input, do NOT rename.

Output format (REQUIRED):
{{"c":[{{"n":"name","d":"voice description","a":"audition line","al":["alias"],"g":"m"}}]}}
"""

    data = _query_llm_json(
        prompt,
        text_sample,
        model,
        api_base,
        api_key,
        wrapper_keys=["c", "characters"],
        thinking_budget=thinking_budget,
    )

    return [
        Character(
            name=str(c.get("n", c.get("name", ""))),
            description=str(c.get("d", c.get("description", ""))),
            audition_line=str(c.get("a", c.get("audition_line", ""))),
            aliases=c.get("al", c.get("aliases")),
            gender=c.get("g", c.get("gender")),
        )
        for c in cast(list, data)
        if isinstance(c, dict)
    ]


def generate_cast(
    text_sample: str,
    api_base: Optional[str] = None,
    api_key: Optional[str] = None,
    model: str = DEFAULT_LLM_MODEL,
    existing_cast_summary: Optional[str] = None,
    thinking_budget: int = DEFAULT_THINKING_BUDGET,
) -> List[Character]:
    """analyze text to identify characters and generate voice descriptions."""
    context_str = (
        f"\nExisting Cast:\n{existing_cast_summary}\n" if existing_cast_summary else ""
    )

    prompt = f"""Identify book characters. Output ONLY valid JSON, no markdown.

Rules:
1. For each character: n (name), d (vocal description), a (audition line), al (aliases), g (gender: "m" or "f").
2. Vocal description 'd': timbre, pitch, speed, gender, age.
3. Use the character name AS IT APPEARS in the source text for 'n'. Put alternate spellings/translations in 'al'.
4. Audition line 'a': write in the SAME LANGUAGE as the source text. Use a quote from the text.
5. If a character matches an existing one, use the EXACT existing 'n' value and add new aliases.
6. Update existing characters if their audition line 'a' is not in the source text language.
{context_str}
Output format (REQUIRED):
{{"c":[{{"n":"NameFromText","d":"voice description","a":"line from text","al":["AlternateName"],"g":"m"}}]}}
"""

    data = _query_llm_json(
        prompt,
        text_sample,
        model,
        api_base,
        api_key,
        wrapper_keys=["c", "characters"],
        thinking_budget=thinking_budget,
    )

    return [
        Character(
            name=str(c.get("n", c.get("name", ""))),
            description=str(c.get("d", c.get("description", ""))),
            audition_line=str(c.get("a", c.get("audition_line", ""))),
            aliases=c.get("al", c.get("aliases")),
            gender=c.get("g", c.get("gender")),
        )
        for c in cast(list, data)
        if isinstance(c, dict)
    ]


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


# cyrillic → latin transliteration for cross-script fuzzy matching
_CYR_TO_LAT = str.maketrans(
    "абвгдеёжзийклмнопрстуфхцчшщъыьэюя",
    "abvgdeežziiklmnoprstufhccššʺyʹèûâ",
)


def _to_latin(s: str) -> str:
    """transliterate cyrillic to latin for cross-script comparison."""
    return s.lower().translate(_CYR_TO_LAT)


def _build_speaker_map(characters_list: List[Character]) -> dict[str, str]:
    """build a case-insensitive map from all known names/aliases to canonical name."""
    speaker_map: dict[str, str] = {}
    for c in characters_list:
        speaker_map[c.name.lower()] = c.name
        if c.aliases:
            for alias in c.aliases:
                speaker_map[alias.lower()] = c.name
    return speaker_map


def _fuzzy_ratio(a: str, b: str) -> float:
    """similarity ratio that handles cross-script (cyrillic/latin) comparison."""
    ratio = difflib.SequenceMatcher(None, a, b).ratio()
    if ratio < 0.5:
        ratio = max(ratio, difflib.SequenceMatcher(None, _to_latin(a), _to_latin(b)).ratio())
    return ratio


def _last_word(name: str) -> str:
    """extract the last word (surname) from a name."""
    parts = name.split()
    return parts[-1] if parts else name


def _normalize_speaker(name: str, speaker_map: dict[str, str]) -> str:
    """resolve a speaker name to its canonical form using fuzzy matching."""
    key = name.lower()
    if key in speaker_map:
        return speaker_map[key]
    # partial match: check if the last word (surname) matches
    surname = _last_word(key)
    if len(surname) >= 3:
        for map_key, canonical in speaker_map.items():
            map_surname = _last_word(map_key)
            if surname == map_surname or _fuzzy_ratio(surname, map_surname) >= 0.75:
                return canonical
    # full fuzzy match (cross-script aware)
    best_canonical, best_ratio = None, 0.0
    for map_key, canonical in speaker_map.items():
        ratio = _fuzzy_ratio(key, map_key)
        if ratio > best_ratio:
            best_ratio = ratio
            best_canonical = canonical
    if best_ratio >= 0.75 and best_canonical:
        return best_canonical
    return name


def process_script_chunk(
    text_chunk: str,
    characters_list: List[Character],
    api_base: Optional[str] = None,
    api_key: Optional[str] = None,
    model: str = DEFAULT_LLM_MODEL,
    thinking_budget: int = DEFAULT_THINKING_BUDGET,
) -> List[ScriptSegment]:
    """convert a text chunk into dramatized script segments."""
    cast_str = _format_cast_list(characters_list)

    prompt = f"""Convert text to JSON script. Output ONLY valid JSON, no markdown.

Cast: {cast_str}

Rules:
1. Separate quotes from narration into segments.
2. Narrator speaks all unquoted text.
3. Characters speak only words inside quotes. Use "Extra Female/Male" if unknown.
4. EXACT text only - do not add or remove words.
5. For "s" field, copy-paste the EXACT character name from the Cast list above. NEVER translate, transliterate, or modify the name in any way.

Output format (REQUIRED - use these exact keys):
{{"seg":[{{"s":"exact name from Cast","t":"exact text","i":"mood"}}]}}
"""

    data = _query_llm_json(
        prompt,
        text_chunk,
        model,
        api_base,
        api_key,
        wrapper_keys=["seg", "segments"],
        thinking_budget=thinking_budget,
    )

    speaker_map = _build_speaker_map(characters_list)
    return _parse_script_segments(data, speaker_map)


def _format_cast_list(characters_list: List[Character]) -> str:
    """format cast list for LLM prompts."""
    cast_info = []
    for c in characters_list:
        if c.aliases:
            cast_info.append(f"{c.name} (also known as: {', '.join(c.aliases)})")
        else:
            cast_info.append(c.name)
    return "\n- ".join(cast_info)


def _parse_script_segments(
    data: list | dict, speaker_map: dict[str, str] | None = None
) -> List[ScriptSegment]:
    """parse LLM response into ScriptSegment list with robust error handling."""
    if not isinstance(data, list):
        raise ValueError(f"expected list of segments, got {type(data).__name__}")

    results = []
    for i, s in enumerate(data):
        if not isinstance(s, dict):
            continue
        if "s" not in s or "t" not in s:
            # skip malformed segments (e.g. {"i": "narrative"} with no speaker/text)
            continue
        text = s["t"]
        if not text or not text.strip():
            continue
        speaker = s["s"]
        if speaker_map:
            speaker = _normalize_speaker(speaker, speaker_map)
        results.append(
            ScriptSegment(
                speaker=speaker,
                text=text,
                instruction=s.get("i", "narrative"),
            )
        )
    return results


def fix_missing_segment(
    missing_text: str,
    context_before: str,
    context_after: str,
    characters_list: List[Character],
    api_base: Optional[str] = None,
    api_key: Optional[str] = None,
    model: str = DEFAULT_LLM_MODEL,
    thinking_budget: int = DEFAULT_THINKING_BUDGET,
) -> List[ScriptSegment]:
    """convert a missing text fragment into script segments using surrounding context."""
    cast_str = _format_cast_list(characters_list)

    prompt = f"""Convert ONLY the "MISSING TEXT" to JSON. Output ONLY valid JSON, no markdown.

Cast: {cast_str}

Rules:
1. Use CONTEXT for speaker/tone but do NOT include context in output.
2. Narrator speaks all unquoted text.
3. Characters speak only words inside quotes.
4. EXACT text only - do not add or remove words.
5. For "s" field, copy-paste the EXACT character name from the Cast list above. NEVER translate, transliterate, or modify the name.

Output format (REQUIRED - use these exact keys):
{{"seg":[{{"s":"exact name from Cast","t":"exact text","i":"mood"}}]}}
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

    data = _query_llm_json(
        prompt,
        user_content,
        model,
        api_base,
        api_key,
        wrapper_keys=["seg", "segments"],
        thinking_budget=thinking_budget,
    )

    speaker_map = _build_speaker_map(characters_list)
    return _parse_script_segments(data, speaker_map)
