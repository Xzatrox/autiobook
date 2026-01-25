"""cli entry point for autiobook."""

import argparse
import sys
from pathlib import Path

from .config import (
    BASE_MODEL,
    DEFAULT_BITRATE,
    DEFAULT_MODEL,
    VOICE_DESIGN_MODEL,
)
from .dramatize import (
    cmd_audition,
    cmd_cast,
    cmd_fix,
    cmd_perform,
    cmd_script,
    cmd_validate,
    dramatize_book,
)
from .epub import parse_epub, save_extracted
from .export import export_audiobook
from .tts import synthesize_chapters
from .utils import add_common_args, get_chapters, get_pipeline_paths, get_tts_config


def cmd_download(args):
    """download tts model weights."""
    from huggingface_hub import snapshot_download

    models = []
    if args.all:
        models = [DEFAULT_MODEL, VOICE_DESIGN_MODEL, BASE_MODEL]
    else:
        models = [args.model]

    for model in models:
        print(f"download: downloading model {model}...")
        path = snapshot_download(repo_id=model)
        print(f"download: model downloaded to {path}")


def cmd_chapters(args):
    """list chapters in an epub file."""
    epub_path = Path(args.epub)
    book, cover_data = parse_epub(epub_path)

    print(f"chapters: title: {book.title}")
    print(f"chapters: author: {book.author}")
    print(f"chapters: language: {book.language}")
    print(f"chapters: count: {len(book.chapters)}")
    print(f"chapters: cover: {'yes' if cover_data else 'no'}")
    print()

    for chapter in book.chapters:
        print(f"  {chapter.index:2d}. {chapter.title} ({chapter.word_count} words)")


def cmd_extract(args):
    """extract chapter text from epub to workdir."""
    epub_path = Path(args.epub)
    workdir = Path(args.output)

    print(f"extract: parsing {epub_path.name}...")
    book, cover_data = parse_epub(epub_path)

    print(f"extract: extracting {len(book.chapters)} chapters to {workdir}/extract/...")
    save_extracted(book, workdir, cover_data)

    print("extract: done")


def cmd_dramatize(args):
    """generate script and cast using LLM."""
    epub_path, workdir = get_pipeline_paths(args)
    audiobook_dir = workdir / "audiobook"
    chapters = get_chapters(args)

    # extract
    print(f"extract: parsing {epub_path.name}...")
    book, cover_data = parse_epub(epub_path)
    print(f"extract: extracting {len(book.chapters)} chapters to {workdir}/extract/...")
    save_extracted(book, workdir, cover_data)

    # dramatize
    tts_config = get_tts_config(args)
    print(f"dramatize: dramatizing chapters in {workdir}...")
    dramatize_book(
        workdir,
        api_base=args.api_base,
        api_key=args.api_key,
        model=args.model,
        chapters=chapters,
        tts_config=tts_config,
        pooled=args.pooled,
        verbose=args.verbose,
        force=args.force,
    )

    # export
    print(f"export: exporting chapters to {audiobook_dir}/...")
    new, skipped = export_audiobook(
        workdir, audiobook_dir, args.bitrate, force=args.force
    )

    msg = f"dramatize: done - {new} chapter(s) exported"
    if skipped > 0:
        msg += f" ({skipped} skipped)"
    print(msg)


def cmd_synthesize(args):
    """convert text files to wav audio."""
    workdir = Path(args.workdir)
    chapters = get_chapters(args)
    config = get_tts_config(args)

    print(f"synthesize: synthesizing chapters in {workdir}/synthesize/...")
    synthesize_chapters(
        workdir, config, chapters, args.instruct, args.pooled, force=args.force
    )

    print("synthesize: done")


def cmd_export(args):
    """convert wav files to mp3 with metadata."""
    workdir = Path(args.workdir)
    output_dir = Path(args.output) if args.output else workdir / "audiobook"

    print(f"export: exporting chapters to {output_dir}/...")
    new, skipped = export_audiobook(workdir, output_dir, args.bitrate, force=args.force)

    msg = f"export: {new} chapter(s) exported"
    if skipped > 0:
        msg += f" ({skipped} skipped)"
    print(msg)


def cmd_clean(args):
    """remove intermediate files (segment caches)."""
    import shutil

    from .resume import get_command_dir

    workdir = Path(args.workdir)
    clean_dirs = [
        get_command_dir(workdir, "perform") / "segments",
        get_command_dir(workdir, "synthesize") / "segments",
    ]

    to_remove = [d for d in clean_dirs if d.exists() and d.is_dir()]

    if not to_remove:
        print("clean: no segment caches found")
        return

    for d in to_remove:
        if args.dry_run:
            print(f"clean: would remove {d}")
        else:
            shutil.rmtree(d)
            print(f"clean: removed {d}")


def cmd_convert(args):
    """run full conversion pipeline."""
    epub_path, workdir = get_pipeline_paths(args)
    audiobook_dir = workdir / "audiobook"
    chapters = get_chapters(args)

    # extract
    print(f"extract: parsing {epub_path.name}...")
    book, cover_data = parse_epub(epub_path)
    print(f"extract: extracting {len(book.chapters)} chapters to {workdir}/extract/...")
    save_extracted(book, workdir, cover_data)

    # synthesize
    config = get_tts_config(args)
    print(f"synthesize: synthesizing chapters in {workdir}/synthesize/...")
    synthesize_chapters(
        workdir, config, chapters, args.instruct, args.pooled, force=args.force
    )

    # export
    print(f"export: exporting chapters to {audiobook_dir}/...")
    new, skipped = export_audiobook(
        workdir, audiobook_dir, args.bitrate, force=args.force
    )

    msg = f"convert: done - {new} chapter(s) exported"
    if skipped > 0:
        msg += f" ({skipped} skipped)"
    print(msg)


def main():
    parser = argparse.ArgumentParser(
        prog="autiobook",
        description="convert epub files to audiobooks using qwen3-tts",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # download command
    p_download = subparsers.add_parser("download", help="download tts model weights")
    p_download.add_argument("-m", "--model", default=DEFAULT_MODEL, help="model name")
    p_download.add_argument(
        "--all", action="store_true", help="download all models (custom, design, base)"
    )
    add_common_args(p_download, group="runtime")
    p_download.set_defaults(func=cmd_download)

    # chapters command
    p_chapters = subparsers.add_parser("chapters", help="list chapters in an epub file")
    p_chapters.add_argument("epub", help="path to epub file")
    add_common_args(p_chapters, group="runtime")
    p_chapters.set_defaults(func=cmd_chapters)

    # extract command
    p_extract = subparsers.add_parser("extract", help="extract chapter text from epub")
    p_extract.add_argument("epub", help="path to epub file")
    p_extract.add_argument("-o", "--output", required=True, help="output workdir")
    add_common_args(p_extract, group="runtime")
    p_extract.set_defaults(func=cmd_extract)

    # dramatize command
    p_dramatize = subparsers.add_parser(
        "dramatize", help="run full dramatization pipeline"
    )
    add_common_args(p_dramatize, group="paths")
    add_common_args(p_dramatize, group="scripting")
    add_common_args(p_dramatize, group="chapter_selection")
    add_common_args(p_dramatize, group="tts_engine")
    add_common_args(p_dramatize, group="cast")
    add_common_args(p_dramatize, group="export")
    add_common_args(p_dramatize, group="runtime")
    p_dramatize.set_defaults(func=cmd_dramatize)

    # cast command
    p_cast = subparsers.add_parser("cast", help="generate cast list from book text")
    p_cast.add_argument("workdir", help="path to workdir")
    add_common_args(p_cast, group="scripting")
    add_common_args(p_cast, group="chapter_selection")
    add_common_args(p_cast, group="cast")
    add_common_args(p_cast, group="runtime")
    p_cast.set_defaults(func=cmd_cast)

    # audition command
    p_audition = subparsers.add_parser(
        "audition", help="generate character voice samples"
    )
    p_audition.add_argument("workdir", help="path to workdir")
    add_common_args(p_audition, group="cast")
    add_common_args(p_audition, group="runtime")
    p_audition.set_defaults(func=cmd_audition)

    # script command
    p_script = subparsers.add_parser("script", help="dramatize chapters into scripts")
    p_script.add_argument("workdir", help="path to workdir")
    add_common_args(p_script, group="scripting")
    add_common_args(p_script, group="chapter_selection")
    add_common_args(p_script, group="runtime")
    p_script.set_defaults(func=cmd_script)

    # perform command
    p_perform = subparsers.add_parser(
        "perform", help="synthesize audio from dramatized scripts"
    )
    p_perform.add_argument("workdir", help="path to workdir")
    add_common_args(p_perform, group="chapter_selection")
    add_common_args(p_perform, group="tts_engine")
    add_common_args(p_perform, group="cast")
    add_common_args(p_perform, group="runtime")
    p_perform.set_defaults(func=cmd_perform)

    # validate command
    p_validate = subparsers.add_parser(
        "validate", help="verify scripts match source text"
    )
    p_validate.add_argument("workdir", help="path to workdir")
    p_validate.add_argument(
        "--missing", action="store_true", help="check for missing text"
    )
    p_validate.add_argument(
        "--hallucinated", action="store_true", help="check for hallucinated segments"
    )
    add_common_args(p_validate, group="chapter_selection")
    add_common_args(p_validate, group="runtime")
    p_validate.set_defaults(func=cmd_validate)

    # fix command
    p_fix = subparsers.add_parser(
        "fix", help="fix script issues (fill missing, remove hallucinated)"
    )
    p_fix.add_argument("workdir", help="path to workdir")
    p_fix.add_argument(
        "--missing", action="store_true", help="fill missing text segments"
    )
    p_fix.add_argument(
        "--hallucinated", action="store_true", help="remove hallucinated segments"
    )
    p_fix.add_argument(
        "--context-chars",
        type=int,
        metavar="N",
        help="characters of context before/after missing text (default: 500)",
    )
    p_fix.add_argument(
        "--context-paragraphs",
        type=int,
        metavar="N",
        help="paragraphs of context before/after missing text (alternative to --context-chars)",
    )
    add_common_args(p_fix, group="scripting")
    add_common_args(p_fix, group="chapter_selection")
    add_common_args(p_fix, group="runtime")
    p_fix.set_defaults(func=cmd_fix)

    # synthesize command
    p_synth = subparsers.add_parser(
        "synthesize", help="convert text files to wav audio"
    )
    p_synth.add_argument("workdir", help="path to workdir")
    add_common_args(p_synth, group="chapter_selection")
    add_common_args(p_synth, group="delivery")
    add_common_args(p_synth, group="tts_engine")
    add_common_args(p_synth, group="runtime")
    p_synth.set_defaults(func=cmd_synthesize)

    # export command
    p_export = subparsers.add_parser("export", help="convert wav files to mp3")
    p_export.add_argument("workdir", help="path to workdir")
    p_export.add_argument(
        "-o",
        "--output",
        help="output directory for mp3 files (default: <workdir>/audiobook/)",
    )
    p_export.add_argument(
        "-b", "--bitrate", default=DEFAULT_BITRATE, help="mp3 bitrate"
    )
    add_common_args(p_export, group="runtime")
    p_export.set_defaults(func=cmd_export)

    # clean command
    p_clean = subparsers.add_parser("clean", help="remove intermediate chunk files")
    p_clean.add_argument("workdir", help="path to workdir")
    p_clean.add_argument(
        "-n", "--dry-run", action="store_true", help="show what would be removed"
    )
    add_common_args(p_clean, group="runtime")
    p_clean.set_defaults(func=cmd_clean)

    # convert command (full pipeline)
    p_convert = subparsers.add_parser("convert", help="run full conversion pipeline")
    add_common_args(p_convert, group="paths")
    add_common_args(p_convert, group="export")
    add_common_args(p_convert, group="chapter_selection")
    add_common_args(p_convert, group="delivery")
    add_common_args(p_convert, group="runtime")
    add_common_args(p_convert, group="tts_engine")
    p_convert.set_defaults(func=cmd_convert)

    args = parser.parse_args()
    try:
        args.func(args)
    except KeyboardInterrupt:
        print("\nAborted!")
        sys.exit(1)


if __name__ == "__main__":
    main()
