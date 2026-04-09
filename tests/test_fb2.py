"""tests for fb2 format support."""

import base64
import zipfile
from pathlib import Path

import pytest

from autiobook.fb2 import Book, Chapter, parse_fb2
from autiobook.utils import detect_book_format, ensure_book_extracted, parse_book

# minimal fb2 XML template
FB2_TEMPLATE = """\
<?xml version="1.0" encoding="utf-8"?>
<FictionBook xmlns="http://www.gribuser.ru/xml/fictionbook/2.0"
             xmlns:l="http://www.w3.org/1999/xlink">
<description>
  <title-info>
    <book-title>{title}</book-title>
    <author><first-name>{first}</first-name><last-name>{last}</last-name></author>
    <lang>{lang}</lang>
    {coverpage}
  </title-info>
</description>
<body>
{sections}
</body>
{binaries}
</FictionBook>
"""

SECTION_TEMPLATE = """\
<section>
  <title><p>{title}</p></title>
  {paragraphs}
</section>
"""


def _make_section(title: str, paragraphs: list[str]) -> str:
    ps = "\n  ".join(f"<p>{p}</p>" for p in paragraphs)
    return SECTION_TEMPLATE.format(title=title, paragraphs=ps)


def _make_fb2(
    title: str = "Test Book",
    first: str = "Test",
    last: str = "Author",
    lang: str = "en",
    sections: list[tuple[str, list[str]]] | None = None,
    cover_data: bytes | None = None,
) -> str:
    if sections is None:
        sections = [
            ("Chapter One", ["This is the first chapter. " * 20]),
            ("Chapter Two", ["This is the second chapter. " * 20]),
        ]
    secs = "\n".join(_make_section(t, ps) for t, ps in sections)

    coverpage = ""
    binaries = ""
    if cover_data:
        coverpage = '<coverpage><image l:href="#cover.jpg"/></coverpage>'
        b64 = base64.b64encode(cover_data).decode()
        binaries = f'<binary id="cover.jpg" content-type="image/jpeg">{b64}</binary>'

    return FB2_TEMPLATE.format(
        title=title, first=first, last=last, lang=lang,
        sections=secs, coverpage=coverpage, binaries=binaries,
    )


@pytest.fixture
def fb2_file(tmp_path):
    """create a minimal fb2 file for testing."""
    content = _make_fb2()
    path = tmp_path / "test.fb2"
    path.write_text(content, encoding="utf-8")
    return path


@pytest.fixture
def fb2_zip_file(tmp_path):
    """create a zipped fb2 file for testing."""
    content = _make_fb2(title="Zipped Book")
    zip_path = tmp_path / "test.fb2.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("book.fb2", content)
    return zip_path


class TestDetectBookFormat:
    """tests for book format detection."""

    def test_epub_detected(self):
        assert detect_book_format(Path("book.epub")) == "epub"

    def test_fb2_detected(self):
        assert detect_book_format(Path("book.fb2")) == "fb2"

    def test_fb2_zip_detected(self):
        assert detect_book_format(Path("book.fb2.zip")) == "fb2"

    def test_unsupported_format_raises(self):
        with pytest.raises(ValueError, match="Unsupported"):
            detect_book_format(Path("book.pdf"))

    def test_case_insensitive(self):
        assert detect_book_format(Path("book.FB2")) == "fb2"
        assert detect_book_format(Path("book.EPUB")) == "epub"


class TestParseFb2:
    """tests for fb2 parsing."""

    def test_basic_parse(self, fb2_file):
        book, cover = parse_fb2(fb2_file)
        assert book.title == "Test Book"
        assert book.author == "Test Author"
        assert book.language == "en"
        assert len(book.chapters) == 2
        assert cover is None

    def test_chapter_titles(self, fb2_file):
        book, _ = parse_fb2(fb2_file)
        assert book.chapters[0].title == "Chapter One"
        assert book.chapters[1].title == "Chapter Two"

    def test_chapter_text_extracted(self, fb2_file):
        book, _ = parse_fb2(fb2_file)
        assert "first chapter" in book.chapters[0].text
        assert "second chapter" in book.chapters[1].text

    def test_chapter_indices(self, fb2_file):
        book, _ = parse_fb2(fb2_file)
        assert book.chapters[0].index == 1
        assert book.chapters[1].index == 2

    def test_short_sections_skipped(self, tmp_path):
        content = _make_fb2(sections=[
            ("Short", ["Too short."]),
            ("Long Enough", ["This chapter has enough words to pass the minimum. " * 20]),
        ])
        path = tmp_path / "short.fb2"
        path.write_text(content, encoding="utf-8")
        book, _ = parse_fb2(path)
        assert len(book.chapters) == 1
        assert book.chapters[0].title == "Long Enough"

    def test_zip_parsing(self, fb2_zip_file):
        book, _ = parse_fb2(fb2_zip_file)
        assert book.title == "Zipped Book"
        assert len(book.chapters) == 2

    def test_cover_extraction(self, tmp_path):
        fake_cover = b"\xff\xd8\xff\xe0fake-jpeg-data"
        content = _make_fb2(cover_data=fake_cover)
        path = tmp_path / "cover.fb2"
        path.write_text(content, encoding="utf-8")
        _, cover = parse_fb2(path)
        assert cover == fake_cover

    def test_missing_metadata_defaults(self, tmp_path):
        # fb2 with no description
        xml = """\
<?xml version="1.0" encoding="utf-8"?>
<FictionBook xmlns="http://www.gribuser.ru/xml/fictionbook/2.0">
<body>
<section><title><p>Ch1</p></title>
<p>{text}</p>
</section>
</body>
</FictionBook>""".format(text="Word " * 60)
        path = tmp_path / "noinfo.fb2"
        path.write_text(xml, encoding="utf-8")
        book, _ = parse_fb2(path)
        assert book.title == "noinfo"  # falls back to filename stem
        assert book.author == "Unknown"

    def test_poem_extraction(self, tmp_path):
        xml = (
            '<?xml version="1.0" encoding="utf-8"?>'
            '<FictionBook xmlns="http://www.gribuser.ru/xml/fictionbook/2.0">'
            "<description><title-info><book-title>Poems</book-title>"
            "<author><first-name>A</first-name><last-name>B</last-name></author>"
            "<lang>en</lang></title-info></description>"
            "<body>"
            "<section><title><p>Poem Chapter</p></title>"
            "<p>Introduction paragraph with enough words to pass the minimum "
            "threshold for chapter extraction in the parser module.</p>"
            "<poem><stanza>"
            "<v>Roses are red</v>"
            "<v>Violets are blue</v>"
            "</stanza></poem>"
            "<p>More text to ensure we pass the word count minimum for this "
            "chapter section and get it included in output.</p>"
            "<p>And yet another paragraph of filler text to make absolutely "
            "sure we exceed the fifty word minimum threshold.</p>"
            "</section>"
            "</body>"
            "</FictionBook>"
        )
        path = tmp_path / "poem.fb2"
        path.write_text(xml, encoding="utf-8")
        book, _ = parse_fb2(path)
        assert len(book.chapters) == 1
        assert "Roses are red" in book.chapters[0].text


class TestBookModel:
    """tests for Book and Chapter dataclasses."""

    def test_chapter_word_count(self):
        ch = Chapter(index=1, title="Test", text="one two three four five")
        assert ch.word_count == 5

    def test_chapter_filename_base(self):
        ch = Chapter(index=3, title="My Chapter Title", text="text")
        assert ch.filename_base == "03_My_Chapter_Title"

    def test_book_to_metadata(self):
        book = Book(
            title="T", author="A", language="en",
            chapters=[Chapter(index=1, title="Ch1", text="text")],
        )
        meta = book.to_metadata()
        assert meta["title"] == "T"
        assert meta["author"] == "A"
        assert len(meta["chapters"]) == 1
        assert meta["chapters"][0]["index"] == 1


class TestParseBookDispatch:
    """tests for the parse_book dispatcher in utils.py."""

    def test_dispatches_to_fb2(self, fb2_file):
        book, _ = parse_book(fb2_file)
        assert book.title == "Test Book"

    def test_dispatches_to_epub(self):
        epub_path = Path("testdata/isaac-asimov_short-science-fiction_advanced.epub")
        if not epub_path.exists():
            pytest.skip("test epub not available")
        book, _ = parse_book(epub_path)
        assert book.title  # just verify it parses


class TestEnsureBookExtracted:
    """tests for fb2 extraction to workdir."""

    def test_extract_creates_files(self, fb2_file, tmp_path):
        workdir = tmp_path / "workdir"
        ensure_book_extracted(fb2_file, workdir)

        extract_dir = workdir / "extract"
        assert (extract_dir / "metadata.json").exists()
        txt_files = list(extract_dir.glob("*.txt"))
        assert len(txt_files) == 2

    def test_extract_idempotent(self, fb2_file, tmp_path):
        workdir = tmp_path / "workdir"
        ensure_book_extracted(fb2_file, workdir)
        # second call should not fail
        ensure_book_extracted(fb2_file, workdir)

        txt_files = list((workdir / "extract").glob("*.txt"))
        assert len(txt_files) == 2

    def test_extract_with_cover(self, tmp_path):
        fake_cover = b"\xff\xd8\xff\xe0fake-jpeg"
        content = _make_fb2(cover_data=fake_cover)
        fb2_path = tmp_path / "cover.fb2"
        fb2_path.write_text(content, encoding="utf-8")

        workdir = tmp_path / "workdir"
        ensure_book_extracted(fb2_path, workdir)
        assert (workdir / "extract" / "cover.jpg").exists()

    def test_metadata_content(self, fb2_file, tmp_path):
        import json

        workdir = tmp_path / "workdir"
        ensure_book_extracted(fb2_file, workdir)

        with open(workdir / "extract" / "metadata.json") as f:
            meta = json.load(f)

        assert meta["title"] == "Test Book"
        assert meta["author"] == "Test Author"
        assert len(meta["chapters"]) == 2
