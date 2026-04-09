"""tests for fb2 parsing."""

import tempfile
from pathlib import Path

import pytest

from autiobook.fb2 import Book, Chapter, parse_fb2


def create_test_fb2(content: str) -> Path:
    """create a temporary fb2 file with given content."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".fb2", delete=False, encoding="utf-8") as f:
        f.write(content)
        return Path(f.name)


def test_parse_simple_fb2():
    """test parsing a simple fb2 file."""
    fb2_content = """<?xml version="1.0" encoding="utf-8"?>
<FictionBook xmlns="http://www.gribuser.ru/xml/fictionbook/2.0">
  <description>
    <title-info>
      <book-title>Test Book</book-title>
      <author>
        <first-name>John</first-name>
        <last-name>Doe</last-name>
      </author>
      <lang>en</lang>
    </title-info>
  </description>
  <body>
    <section>
      <title><p>Chapter One</p></title>
      <p>This is the first paragraph of chapter one.</p>
      <p>This is the second paragraph of chapter one.</p>
    </section>
    <section>
      <title><p>Chapter Two</p></title>
      <p>This is the first paragraph of chapter two.</p>
      <p>This is the second paragraph of chapter two.</p>
    </section>
  </body>
</FictionBook>"""

    fb2_path = create_test_fb2(fb2_content)
    try:
        book, cover = parse_fb2(fb2_path)

        assert book.title == "Test Book"
        assert book.author == "John Doe"
        assert book.language == "en"
        assert len(book.chapters) == 2

        assert book.chapters[0].index == 1
        assert book.chapters[0].title == "Chapter One"
        assert "first paragraph of chapter one" in book.chapters[0].text
        assert "second paragraph of chapter one" in book.chapters[0].text

        assert book.chapters[1].index == 2
        assert book.chapters[1].title == "Chapter Two"
        assert "first paragraph of chapter two" in book.chapters[1].text

        assert cover is None  # no cover in this test
    finally:
        fb2_path.unlink()


def test_parse_fb2_with_poem():
    """test parsing fb2 with poem/stanza content."""
    fb2_content = """<?xml version="1.0" encoding="utf-8"?>
<FictionBook xmlns="http://www.gribuser.ru/xml/fictionbook/2.0">
  <description>
    <title-info>
      <book-title>Poetry Book</book-title>
      <author>
        <first-name>Jane</first-name>
        <last-name>Smith</last-name>
      </author>
      <lang>en</lang>
    </title-info>
  </description>
  <body>
    <section>
      <title><p>Poem One</p></title>
      <p>Introduction to the poem.</p>
      <stanza>
        <v>First line of verse</v>
        <v>Second line of verse</v>
        <v>Third line of verse</v>
      </stanza>
      <p>Conclusion of the poem.</p>
    </section>
  </body>
</FictionBook>"""

    fb2_path = create_test_fb2(fb2_content)
    try:
        book, _ = parse_fb2(fb2_path)

        assert len(book.chapters) == 1
        assert "Introduction to the poem" in book.chapters[0].text
        assert "First line of verse" in book.chapters[0].text
        assert "Second line of verse" in book.chapters[0].text
        assert "Conclusion of the poem" in book.chapters[0].text
    finally:
        fb2_path.unlink()


def test_parse_fb2_no_title():
    """test parsing fb2 without explicit chapter titles."""
    fb2_content = """<?xml version="1.0" encoding="utf-8"?>
<FictionBook xmlns="http://www.gribuser.ru/xml/fictionbook/2.0">
  <description>
    <title-info>
      <book-title>Untitled Chapters</book-title>
      <author>
        <first-name>Test</first-name>
        <last-name>Author</last-name>
      </author>
      <lang>en</lang>
    </title-info>
  </description>
  <body>
    <section>
      <p>This section has no title element.</p>
      <p>It should get a default chapter name.</p>
    </section>
  </body>
</FictionBook>"""

    fb2_path = create_test_fb2(fb2_content)
    try:
        book, _ = parse_fb2(fb2_path)

        assert len(book.chapters) == 1
        assert book.chapters[0].title == "Chapter 1"
        assert "This section has no title element" in book.chapters[0].text
    finally:
        fb2_path.unlink()


def test_chapter_word_count():
    """test chapter word count calculation."""
    chapter = Chapter(index=1, title="Test", text="This is a test with seven words.")
    assert chapter.word_count == 7


def test_chapter_filename_base():
    """test chapter filename generation."""
    chapter = Chapter(index=1, title="Chapter One: The Beginning", text="test")
    assert chapter.filename_base == "01_Chapter_One__The_Beginning"

    # test with unsafe characters
    chapter2 = Chapter(index=2, title='Test/Chapter\\With:Bad*Chars?', text="test")
    filename = chapter2.filename_base
    assert "/" not in filename
    assert "\\" not in filename
    assert ":" not in filename
    assert "*" not in filename
    assert "?" not in filename


def test_book_to_metadata():
    """test book metadata serialization."""
    chapters = [
        Chapter(index=1, title="Chapter One", text="text one"),
        Chapter(index=2, title="Chapter Two", text="text two"),
    ]
    book = Book(title="Test Book", author="Test Author", language="en", chapters=chapters)

    metadata = book.to_metadata()

    assert metadata["title"] == "Test Book"
    assert metadata["author"] == "Test Author"
    assert metadata["language"] == "en"
    assert len(metadata["chapters"]) == 2
    assert metadata["chapters"][0]["index"] == 1
    assert metadata["chapters"][0]["title"] == "Chapter One"
    assert "text" not in metadata["chapters"][0]  # text should not be in metadata


def test_parse_fb2_min_words():
    """test that chapters with too few words are skipped."""
    fb2_content = """<?xml version="1.0" encoding="utf-8"?>
<FictionBook xmlns="http://www.gribuser.ru/xml/fictionbook/2.0">
  <description>
    <title-info>
      <book-title>Test</book-title>
      <author>
        <first-name>Test</first-name>
        <last-name>Author</last-name>
      </author>
      <lang>en</lang>
    </title-info>
  </description>
  <body>
    <section>
      <title><p>Too Short</p></title>
      <p>Short.</p>
    </section>
    <section>
      <title><p>Long Enough</p></title>
      <p>This chapter has enough words to meet the minimum word count requirement for chapters. It should be included in the parsed output while the previous short chapter should be skipped.</p>
    </section>
  </body>
</FictionBook>"""

    fb2_path = create_test_fb2(fb2_content)
    try:
        book, _ = parse_fb2(fb2_path)

        # only the second chapter should be included
        assert len(book.chapters) == 1
        assert book.chapters[0].title == "Long Enough"
    finally:
        fb2_path.unlink()
