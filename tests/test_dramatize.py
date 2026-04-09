"""tests for dramatize module."""

import json

import pytest

from autiobook.dramatize import (
    _attempt_merge,
    _extract_context,
    _find_text_in_source,
    _normalize_text,
    _remove_hallucinations,
    _strip_boundary_quotes,
    _tokenize_with_positions,
    load_cast,
    load_script,
    save_cast,
    save_script,
    validate_script,
)
from autiobook.llm import Character, ScriptSegment


@pytest.fixture
def workdir(tmp_path):
    """create a workdir with extract/ and cast/ directories."""
    (tmp_path / "extract").mkdir()
    (tmp_path / "cast").mkdir()
    (tmp_path / "script").mkdir()
    return tmp_path


class TestCastSerialization:
    """tests for save_cast / load_cast."""

    def test_round_trip(self, workdir):
        cast = [
            Character(name="Alice", description="warm voice", audition_line="Hello."),
            Character(
                name="Bob", description="deep voice", audition_line="Hi.", aliases=["Robert"]
            ),
        ]
        save_cast(workdir, cast)
        loaded = load_cast(workdir)

        assert len(loaded) == 2
        assert loaded[0].name == "Alice"
        assert loaded[0].description == "warm voice"
        assert loaded[1].name == "Bob"
        assert loaded[1].aliases == ["Robert"]

    def test_load_missing_returns_defaults(self, workdir):
        # remove cast file so it falls back to defaults
        cast_file = workdir / "cast" / "characters.json"
        if cast_file.exists():
            cast_file.unlink()
        loaded = load_cast(workdir)
        assert len(loaded) == 3  # DEFAULT_CAST has 3 entries
        assert loaded[0].name == "Narrator"

    def test_load_legacy_list_format(self, workdir):
        legacy = [
            {"name": "Eve", "description": "soft", "audition_line": "Test."},
        ]
        path = workdir / "cast" / "characters.json"
        with open(path, "w") as f:
            json.dump(legacy, f)
        loaded = load_cast(workdir)
        assert len(loaded) == 1
        assert loaded[0].name == "Eve"


class TestScriptSerialization:
    """tests for save_script / load_script."""

    def test_round_trip(self, tmp_path):
        path = tmp_path / "script.json"
        segments = [
            ScriptSegment(speaker="Narrator", text="Once upon a time.", instruction="calm"),
            ScriptSegment(speaker="Alice", text="Hello!", instruction="cheerful"),
        ]
        save_script(path, segments)
        loaded = load_script(path)

        assert len(loaded) == 2
        assert loaded[0].speaker == "Narrator"
        assert loaded[0].text == "Once upon a time."
        assert loaded[1].speaker == "Alice"
        assert loaded[1].instruction == "cheerful"

    def test_load_missing_returns_empty(self, tmp_path):
        path = tmp_path / "nonexistent.json"
        assert load_script(path) == []


class TestTextHelpers:
    """tests for text normalization and tokenization."""

    def test_normalize_text(self):
        assert _normalize_text("  hello   world  \n\n  foo  ") == "hello world foo"

    def test_strip_boundary_quotes(self):
        assert _strip_boundary_quotes('"Hello"') == "Hello"
        assert _strip_boundary_quotes("'test'") == "test"
        assert _strip_boundary_quotes("  plain  ") == "plain"

    def test_tokenize_with_positions(self):
        tokens = _tokenize_with_positions("Hello, world!")
        assert len(tokens) == 2
        assert tokens[0][0] == "hello"
        assert tokens[1][0] == "world"
        # positions should be correct
        assert "Hello, world!"[tokens[0][1] : tokens[0][2]] == "Hello"
        assert "Hello, world!"[tokens[1][1] : tokens[1][2]] == "world"

    def test_tokenize_empty(self):
        assert _tokenize_with_positions("") == []
        assert _tokenize_with_positions("...") == []


class TestFindTextInSource:
    """tests for fuzzy text matching."""

    def test_exact_match(self):
        result = _find_text_in_source("hello world", "prefix hello world suffix")
        assert result is not None
        start, end = result
        assert "prefix hello world suffix"[start:end] == "hello world"

    def test_match_with_offset(self):
        haystack = "aaa bbb ccc ddd eee"
        result = _find_text_in_source("ccc ddd", haystack, start_pos=4)
        assert result is not None

    def test_no_match(self):
        result = _find_text_in_source("xyz completely different", "hello world foo bar")
        assert result is None

    def test_empty_needle(self):
        assert _find_text_in_source("", "hello") is None


class TestValidateScript:
    """tests for script validation."""

    def test_perfect_match(self, tmp_path):
        txt = tmp_path / "chapter.txt"
        txt.write_text("Hello world. This is a test.")
        script = tmp_path / "chapter.json"
        save_script(
            script,
            [ScriptSegment(speaker="N", text="Hello world. This is a test.", instruction="")],
        )
        result = validate_script(txt, script)
        assert len(result.missing) == 0
        assert len(result.hallucinated) == 0

    def test_missing_text_detected(self, tmp_path):
        txt = tmp_path / "chapter.txt"
        txt.write_text("First sentence. Second sentence. Third sentence.")
        script = tmp_path / "chapter.json"
        save_script(
            script,
            [
                ScriptSegment(speaker="N", text="First sentence.", instruction=""),
                ScriptSegment(speaker="N", text="Third sentence.", instruction=""),
            ],
        )
        result = validate_script(txt, script)
        assert len(result.missing) > 0
        # the missing text should contain "Second sentence"
        missing_text = " ".join(m[0] for m in result.missing)
        assert "Second" in missing_text

    def test_hallucinated_text_detected(self, tmp_path):
        txt = tmp_path / "chapter.txt"
        txt.write_text("The real text of the chapter.")
        script = tmp_path / "chapter.json"
        save_script(
            script,
            [
                ScriptSegment(speaker="N", text="The real text of the chapter.", instruction=""),
                ScriptSegment(
                    speaker="N",
                    text="This text was completely invented by the language model.",
                    instruction="",
                ),
            ],
        )
        result = validate_script(txt, script)
        assert len(result.hallucinated) > 0
        assert 1 in result.hallucinated

    def test_no_script_reports_missing(self, tmp_path):
        txt = tmp_path / "chapter.txt"
        txt.write_text("Some text.")
        script = tmp_path / "chapter.json"
        # no script file
        result = validate_script(txt, script)
        assert len(result.missing) > 0


class TestAttemptMerge:
    """tests for segment merging."""

    def test_merge_same_speaker(self):
        segments = [
            ScriptSegment(speaker="N", text="Hello.", instruction="calm"),
            ScriptSegment(speaker="N", text="World.", instruction="calm"),
        ]
        merged = _attempt_merge(segments, 0)
        assert merged is True
        assert len(segments) == 1
        assert segments[0].text == "Hello. World."

    def test_no_merge_different_speakers(self):
        segments = [
            ScriptSegment(speaker="N", text="Hello.", instruction=""),
            ScriptSegment(speaker="Alice", text="Hi!", instruction=""),
        ]
        merged = _attempt_merge(segments, 0)
        assert merged is False
        assert len(segments) == 2

    def test_merge_out_of_bounds(self):
        segments = [ScriptSegment(speaker="N", text="Only one.", instruction="")]
        assert _attempt_merge(segments, 0) is False
        assert _attempt_merge(segments, -1) is False
        assert _attempt_merge(segments, 5) is False


class TestRemoveHallucinations:
    """tests for hallucination removal."""

    def test_removes_correct_indices(self):
        segments = [
            ScriptSegment(speaker="N", text="Real.", instruction=""),
            ScriptSegment(speaker="N", text="Fake.", instruction=""),
            ScriptSegment(speaker="N", text="Also real.", instruction=""),
        ]
        removed = _remove_hallucinations(segments, [1])
        assert removed == 1
        assert len(segments) == 2
        assert segments[0].text == "Real."
        assert segments[1].text == "Also real."

    def test_removes_multiple_reverse_order(self):
        segments = [
            ScriptSegment(speaker="N", text="A.", instruction=""),
            ScriptSegment(speaker="N", text="B.", instruction=""),
            ScriptSegment(speaker="N", text="C.", instruction=""),
            ScriptSegment(speaker="N", text="D.", instruction=""),
        ]
        removed = _remove_hallucinations(segments, [1, 3])
        assert removed == 2
        assert len(segments) == 2
        assert segments[0].text == "A."
        assert segments[1].text == "C."


class TestExtractContext:
    """tests for context extraction around missing fragments."""

    def test_char_based_context(self):
        text = "AAA BBB CCC DDD EEE FFF GGG"
        before, after = _extract_context(text, "DDD", context_chars=8)
        assert "CCC" in before or "BBB" in before
        assert "EEE" in after or "FFF" in after

    def test_paragraph_based_context(self):
        text = "Para one.\n\nPara two.\n\nPara three.\n\nPara four."
        before, after = _extract_context(text, "Para two.", context_paragraphs=1)
        assert "Para one" in before
        assert "Para three" in after

    def test_fragment_not_found(self):
        before, after = _extract_context("hello world", "nonexistent")
        assert before == ""
        assert after == ""
