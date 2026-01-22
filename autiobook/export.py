"""mp3 export with id3 metadata."""

from dataclasses import dataclass
from pathlib import Path

from mutagen.id3 import APIC, ID3
from mutagen.mp3 import MP3
from pydub import AudioSegment

from .config import COVER_FILE, DEFAULT_BITRATE, MP3_EXT, WAV_EXT
from .epub import load_metadata


@dataclass
class MP3Metadata:
    """id3 tag metadata for mp3 file."""

    title: str
    album: str
    artist: str
    track_number: int
    total_tracks: int


def wav_to_mp3(
    wav_path: Path,
    mp3_path: Path,
    metadata: MP3Metadata,
    bitrate: str = DEFAULT_BITRATE,
    cover_path: Path | None = None,
) -> None:
    """convert wav to mp3 with id3 metadata tags and cover art."""
    audio = AudioSegment.from_wav(str(wav_path))

    tags = {
        "title": metadata.title,
        "album": metadata.album,
        "artist": metadata.artist,
        "track": f"{metadata.track_number}/{metadata.total_tracks}",
    }

    audio.export(str(mp3_path), format="mp3", bitrate=bitrate, tags=tags)

    # add cover art if available
    if cover_path and cover_path.exists():
        mp3 = MP3(str(mp3_path), ID3=ID3)
        if mp3.tags is None:
            mp3.add_tags()

        cover_data = cover_path.read_bytes()
        mp3.tags.add(
            APIC(
                encoding=3,  # utf-8
                mime="image/jpeg",
                type=3,  # front cover
                desc="Cover",
                data=cover_data,
            )
        )
        mp3.save()


def export_audiobook(
    workdir: Path,
    output_dir: Path,
    bitrate: str = DEFAULT_BITRATE,
) -> list[Path]:
    """export all chapters as mp3 files with cover art."""
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = load_metadata(workdir)
    book_title = metadata["title"]
    author = metadata["author"]
    chapters = metadata["chapters"]
    total_tracks = len(chapters)

    # check for cover image
    cover_path = workdir / COVER_FILE
    if not cover_path.exists():
        cover_path = None

    exported = []

    for chapter_info in chapters:
        filename_base = chapter_info["filename_base"]
        wav_path = workdir / f"{filename_base}{WAV_EXT}"
        mp3_path = output_dir / f"{filename_base}{MP3_EXT}"

        # skip if wav doesn't exist (not synthesized yet)
        if not wav_path.exists():
            continue

        # skip if already exported (idempotent)
        if mp3_path.exists():
            print(f"skipping {wav_path.name} (already exported)")
            exported.append(mp3_path)
            continue

        print(f"exporting {wav_path.name}...")

        mp3_meta = MP3Metadata(
            title=chapter_info["title"],
            album=book_title,
            artist=author,
            track_number=chapter_info["index"],
            total_tracks=total_tracks,
        )

        wav_to_mp3(wav_path, mp3_path, mp3_meta, bitrate, cover_path)
        print(f"  -> {mp3_path.name}")
        exported.append(mp3_path)

    return exported
