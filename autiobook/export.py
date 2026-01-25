"""mp3 export with id3 metadata."""

from dataclasses import dataclass
from pathlib import Path

from mutagen.id3 import APIC, ID3  # type: ignore
from mutagen.mp3 import MP3  # type: ignore
from pydub import AudioSegment  # type: ignore

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

        if mp3.tags is not None:
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
    force: bool = False,
) -> tuple[int, int]:
    """export all chapters as mp3 files with cover art."""
    from .resume import ResumeManager, compute_hash, get_command_dir, list_chapters

    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = load_metadata(workdir)
    book_title = metadata["title"]
    author = metadata["author"]
    chapters = metadata["chapters"]
    total_tracks = len(chapters)

    # try performance dir first (dramatized), then synthesize dir
    perform_dir = get_command_dir(workdir, "perform")
    synth_dir = get_command_dir(workdir, "synthesize")

    source_dir = perform_dir
    chapter_paths = list_chapters(
        metadata, source_dir, output_dir, source_ext=WAV_EXT, target_ext=MP3_EXT
    )

    if not chapter_paths:
        source_dir = synth_dir
        chapter_paths = list_chapters(
            metadata, source_dir, output_dir, source_ext=WAV_EXT, target_ext=MP3_EXT
        )

    if not chapter_paths:
        print(f"export: no wav files found in {perform_dir} or {synth_dir}")
        return 0, 0

    # check for cover image in extract dir
    extract_dir = get_command_dir(workdir, "extract")
    cover_path_val = extract_dir / COVER_FILE
    final_cover_path: Path | None = cover_path_val if cover_path_val.exists() else None

    resume = ResumeManager.for_command(workdir, "export", force=force)
    newly_exported_count = 0
    skipped_count = 0

    # build index for metadata lookup
    chapter_info_map = {c["index"]: c for c in chapters}

    for idx, wav_path, mp3_path in chapter_paths:
        chapter_info = chapter_info_map.get(idx)
        if not chapter_info:
            continue

        # Compute hash for resumability
        # include wav size, mtime, and metadata
        export_data = {
            "wav_size": wav_path.stat().st_size,
            "wav_mtime": wav_path.stat().st_mtime,
            "title": chapter_info["title"],
            "album": book_title,
            "artist": author,
            "track": idx,
            "bitrate": bitrate,
            "cover": str(final_cover_path) if final_cover_path else None,
        }
        chapter_hash = compute_hash(export_data)

        # skip if already exported (idempotent)
        if (
            not force
            and mp3_path.exists()
            and resume.is_fresh(str(mp3_path), chapter_hash)
        ):
            skipped_count += 1
            continue

        print(f"exporting {wav_path.name}...")

        mp3_meta = MP3Metadata(
            title=chapter_info["title"],
            album=book_title,
            artist=author,
            track_number=chapter_info["index"],
            total_tracks=total_tracks,
        )

        wav_to_mp3(wav_path, mp3_path, mp3_meta, bitrate, final_cover_path)
        resume.update(str(mp3_path), chapter_hash)
        resume.save()
        print(f"  -> {mp3_path.name}")
        newly_exported_count += 1

    if newly_exported_count == 0 and skipped_count == 0:
        print("export: no chapters found.")
    elif newly_exported_count == 0 and skipped_count > 0:
        print(f"export: all {skipped_count} chapters up to date.")

    return newly_exported_count, skipped_count
