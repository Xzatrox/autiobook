"""fb2 (FictionBook 2) parsing and chapter extraction."""

import hashlib
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree as ET

from .config import (
    COVER_FILE,
    METADATA_FILE,
    MIN_CHAPTER_WORDS,
    TXT_EXT,
)

# FB2 namespace
FB2_NS = "http://www.gribuser.ru/xml/fictionbook/2.0"
XLINK_NS = "http://www.w3.org/1999/xlink"


@dataclass
class Chapter:
    """single chapter extracted from an fb2 file."""

    index: int
    title: str
    text: str

    @property
    def word_count(self) -> int:
        return len(self.text.split())

    @property
    def filename_base(self) -> str:
        """sanitized filename without extension."""
        from .config import UNSAFE_FILENAME_CHARS

        safe_title = UNSAFE_FILENAME_CHARS.sub("_", self.title)
        safe_title = safe_title.strip().replace(" ", "_")[:50]
        return f"{self.index:02d}_{safe_title}"


@dataclass
class Book:
    """parsed fb2 book with metadata and chapters."""

    title: str
    author: str
    language: str
    chapters: list[Chapter]

    def to_metadata(self) -> dict:
        """return metadata dict for serialization (without chapter text)."""
        return {
            "title": self.title,
            "author": self.author,
            "language": self.language,
            "chapters": [
                {"index": c.index, "title": c.title, "filename_base": c.filename_base}
                for c in self.chapters
            ],
        }


def _ns(tag: str) -> str:
    """add fb2 namespace to tag."""
    return f"{{{FB2_NS}}}{tag}"


def _xlink_ns(attr: str) -> str:
    """add xlink namespace to attribute."""
    return f"{{{XLINK_NS}}}{attr}"


def extract_text_from_element(elem: ET.Element) -> str:
    """recursively extract text from an element and its children."""
    paragraphs = []

    # handle paragraphs
    for p in elem.findall(f".//{_ns('p')}"):
        text = "".join(p.itertext()).strip()
        if text:
            paragraphs.append(text)

    # handle poems (stanzas and verses)
    for stanza in elem.findall(f".//{_ns('stanza')}"):
        verses = []
        for v in stanza.findall(_ns("v")):
            text = "".join(v.itertext()).strip()
            if text:
                verses.append(text)
        if verses:
            paragraphs.append(" ".join(verses))

    # handle epigraphs
    for epigraph in elem.findall(f".//{_ns('epigraph')}"):
        for p in epigraph.findall(_ns("p")):
            text = "".join(p.itertext()).strip()
            if text:
                paragraphs.append(text)

    return "\n\n".join(paragraphs)


def extract_title_from_section(section: ET.Element) -> str | None:
    """extract title from a section element."""
    title_elem = section.find(_ns("title"))
    if title_elem is not None:
        # title can contain multiple paragraphs
        title_parts = []
        for p in title_elem.findall(_ns("p")):
            text = "".join(p.itertext()).strip()
            if text:
                title_parts.append(text)
        if title_parts:
            return " ".join(title_parts)
    return None


def extract_cover_image(root: ET.Element, fb2_path: Path) -> bytes | None:
    """extract cover image from fb2 file."""
    # find coverpage reference
    description = root.find(_ns("description"))
    if description is None:
        return None

    title_info = description.find(_ns("title-info"))
    if title_info is None:
        return None

    coverpage = title_info.find(_ns("coverpage"))
    if coverpage is None:
        return None

    # get image reference
    image_elem = coverpage.find(_ns("image"))
    if image_elem is None:
        return None

    href = image_elem.get(_xlink_ns("href"))
    if not href:
        return None

    # remove leading '#' from href
    image_id = href.lstrip("#")

    # find binary data
    for binary in root.findall(_ns("binary")):
        if binary.get("id") == image_id:
            content_type = binary.get("content-type", "")
            data = binary.text
            if data:
                import base64

                return base64.b64decode(data.strip())

    return None


def parse_fb2(path: Path) -> tuple[Book, bytes | None]:
    """parse an fb2 file and extract all chapters with metadata and cover."""
    # handle both .fb2 and .fb2.zip files
    if path.suffix.lower() == ".zip" or path.name.lower().endswith(".fb2.zip"):
        with zipfile.ZipFile(path, "r") as zf:
            # find first .fb2 file in archive
            fb2_files = [name for name in zf.namelist() if name.lower().endswith(".fb2")]
            if not fb2_files:
                raise ValueError(f"No .fb2 file found in archive: {path}")
            with zf.open(fb2_files[0]) as f:
                tree = ET.parse(f)
    else:
        tree = ET.parse(path)

    root = tree.getroot()

    # extract metadata
    description = root.find(_ns("description"))
    title = "Unknown"
    author = "Unknown"
    language = "en"

    if description is not None:
        title_info = description.find(_ns("title-info"))
        if title_info is not None:
            # extract title
            book_title = title_info.find(_ns("book-title"))
            if book_title is not None and book_title.text:
                title = book_title.text.strip()

            # extract author
            author_elem = title_info.find(_ns("author"))
            if author_elem is not None:
                first_name = author_elem.find(_ns("first-name"))
                last_name = author_elem.find(_ns("last-name"))
                author_parts = []
                if first_name is not None and first_name.text:
                    author_parts.append(first_name.text.strip())
                if last_name is not None and last_name.text:
                    author_parts.append(last_name.text.strip())
                if author_parts:
                    author = " ".join(author_parts)

            # extract language
            lang_elem = title_info.find(_ns("lang"))
            if lang_elem is not None and lang_elem.text:
                language = lang_elem.text.strip()

    # if no title in metadata, use filename
    if title == "Unknown":
        title = path.stem

    book = Book(title=title, author=author, language=language, chapters=[])

    # extract chapters from body sections
    body = root.find(_ns("body"))
    if body is not None:
        sections = body.findall(_ns("section"))

        for section in sections:
            text = extract_text_from_element(section)
            if len(text.split()) < MIN_CHAPTER_WORDS:
                continue

            idx = len(book.chapters) + 1
            section_title = extract_title_from_section(section) or f"Chapter {idx}"
            book.chapters.append(Chapter(index=idx, title=section_title, text=text))

    # extract cover image
    cover_data = extract_cover_image(root, path)

    return book, cover_data


def ensure_extracted(fb2_path: Path, workdir: Path, force: bool = False) -> None:
    """ensure fb2 is extracted to workdir, skipping if already fresh."""
    from .resume import get_command_dir

    extract_dir = get_command_dir(workdir, "extract")
    state_path = extract_dir / "state.json"

    with open(fb2_path, "rb") as f:
        fb2_hash = hashlib.sha256(f.read()).hexdigest()

    if not force and state_path.exists():
        try:
            with open(state_path) as f:
                if json.load(f).get("fb2_hash") == fb2_hash:
                    if any(extract_dir.glob(f"*{TXT_EXT}")):
                        return
        except Exception:
            pass

    book, cover_data = parse_fb2(fb2_path)
    save_extracted(book, workdir, cover_data)

    with open(state_path, "w") as f:
        json.dump({"fb2_hash": fb2_hash}, f, indent=2)


def save_extracted(book: Book, workdir: Path, cover_data: bytes | None = None) -> None:
    """save extracted chapters and cover to workdir."""
    from .resume import get_command_dir

    extract_dir = get_command_dir(workdir, "extract")

    # save metadata
    metadata_path = extract_dir / METADATA_FILE
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(book.to_metadata(), f, indent=2)

    # save cover image
    if cover_data:
        cover_path = extract_dir / COVER_FILE
        with open(cover_path, "wb") as f:
            f.write(cover_data)
        print(f"  saved cover image ({len(cover_data)} bytes)")

    # save chapter text files
    for chapter in book.chapters:
        txt_path = extract_dir / f"{chapter.filename_base}{TXT_EXT}"
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(chapter.text)
