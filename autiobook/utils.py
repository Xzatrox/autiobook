"""utility functions."""

import argparse
from pathlib import Path

from .config import (
    BASE_MODEL,
    DEFAULT_LLM_MODEL,
    DEFAULT_MODEL,
    DEFAULT_SPEAKER,
    DEFAULT_THINKING_BUDGET,
    MAX_CHUNK_SIZE,
)


def detect_book_format(path: Path) -> str:
    """detect book format from file extension."""
    suffix = path.suffix.lower()
    name_lower = path.name.lower()

    if suffix == ".epub":
        return "epub"
    elif suffix == ".fb2" or name_lower.endswith(".fb2.zip"):
        return "fb2"
    else:
        raise ValueError(f"Unsupported book format: {path.suffix}")


def parse_book(path: Path):
    """parse a book file (epub or fb2) and return (book, cover_data)."""
    book_format = detect_book_format(path)

    if book_format == "epub":
        from .epub import parse_epub
        return parse_epub(path)
    elif book_format == "fb2":
        from .fb2 import parse_fb2
        return parse_fb2(path)
    else:
        raise ValueError(f"Unsupported book format: {book_format}")


def ensure_book_extracted(path: Path, workdir: Path, force: bool = False) -> None:
    """extract book (epub or fb2) to workdir."""
    book_format = detect_book_format(path)

    if book_format == "epub":
        from .epub import ensure_extracted as ensure_extracted_epub
        ensure_extracted_epub(path, workdir, force)
    elif book_format == "fb2":
        from .fb2 import ensure_extracted as ensure_extracted_fb2
        ensure_extracted_fb2(path, workdir, force)
    else:
        raise ValueError(f"Unsupported book format: {book_format}")


def parse_chapter_range(spec: str) -> list[int]:
    """parse chapter range spec like '1-5' or '1,3,5' into list of ints."""
    chapters: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            start, end = part.split("-", 1)
            chapters.extend(range(int(start), int(end) + 1))
        else:
            chapters.append(int(part))
    return chapters


def add_common_args(parser: argparse.ArgumentParser, group: str = "all"):
    """add common arguments to parser."""
    g = parser.add_argument_group(group) if group != "all" else parser

    if group in ["all", "chapter_selection"]:
        g.add_argument("-c", "--chapters", help="chapter range (e.g., 1-5, 1,3,5)")

    if group in ["all", "tts_engine"]:
        g.add_argument("--batch-size", type=int, default=64, help="batch size for tts generation")
        g.add_argument(
            "--chunk-size",
            type=int,
            default=MAX_CHUNK_SIZE,
            help="max chars per chunk (smaller = faster)",
        )
        g.add_argument(
            "--no-compile",
            action="store_true",
            help="disable torch.compile optimization",
        )
        g.add_argument("--no-warmup", action="store_true", help="skip model warmup")
        g.add_argument(
            "--pooled",
            action="store_true",
            help="pool chunks across chapters for better batch utilization",
        )
        g.add_argument(
            "--greedy",
            action="store_true",
            help="use greedy decoding (faster, less varied)",
        )
        g.add_argument(
            "--temperature",
            type=float,
            default=0.9,
            help="sampling temperature (lower = faster)",
        )

    if group in ["all", "delivery"]:
        g.add_argument(
            "-s",
            "--speaker",
            default=DEFAULT_SPEAKER,
            help="base tts speaker",
        )
        g.add_argument(
            "--voice",
            help="cloned voice (name from audition/)",
        )
        g.add_argument("-i", "--instruct", help="instruction for tts (string or file path)")

    if group in ["all", "scripting"]:
        g.add_argument("--api-base", help="llm api base url")
        g.add_argument("--api-key", help="llm api key")
        g.add_argument("--model", default=DEFAULT_LLM_MODEL, help="llm model name")
        g.add_argument(
            "--thinking-budget",
            type=int,
            default=DEFAULT_THINKING_BUDGET,
            help="tokens for extended thinking (0 = disabled)",
        )

    if group in ["all", "llm_server"]:
        g.add_argument(
            "--llm-server-model",
            help="path to GGUF model to auto-start llama-server",
        )
        g.add_argument(
            "--llm-server-hf-model",
            help="HuggingFace model id to auto-start transformers server (uses ROCm torch)",
        )
        g.add_argument(
            "--llm-server-draft-model",
            help="path to draft GGUF model for speculative decoding",
        )
        g.add_argument(
            "--llm-server-port",
            type=int,
            default=8080,
            help="port for llama-server (default: 8080)",
        )
        g.add_argument(
            "--llm-server-ctx-size",
            type=int,
            default=8192,
            help="context size for llama-server (default: 8192)",
        )

    if group in ["all", "runtime"]:
        g.add_argument("-v", "--verbose", action="store_true", help="enable verbose logging")
        g.add_argument(
            "-f",
            "--force",
            action="store_true",
            help="ignore resume state and force processing",
        )

    if group in ["all", "paths"]:
        g.add_argument("book", help="path to book file (epub or fb2)")
        g.add_argument(
            "-o",
            "--output",
            help="workdir for intermediate files (default: <book>_output/)",
        )

    if group in ["all", "export"]:
        from .config import DEFAULT_BITRATE

        g.add_argument("-b", "--bitrate", default=DEFAULT_BITRATE, help="mp3 bitrate")
        g.add_argument("--m4b", action="store_true", help="export as m4b audiobook with metadata")


def get_pipeline_paths(args) -> tuple[Path, Path]:
    """get book_path and workdir from args, inferring if needed."""
    book_path = Path(args.book)
    if args.output:
        workdir = Path(args.output)
    else:
        # infer workdir: /path/to/book.epub -> /path/to/book_output/
        workdir = book_path.parent / (book_path.stem + "_output")

    return book_path, workdir


def get_chapters(args) -> list[int] | None:
    """extract chapter list from args."""
    if args.chapters:
        return parse_chapter_range(args.chapters)
    return None


def get_tts_config(args):
    """extract tts config from args."""
    from .tts import TTSConfig

    model_name = DEFAULT_MODEL
    # use base model for voice cloning if voice is specified
    if hasattr(args, "voice") and args.voice:
        model_name = BASE_MODEL

    config = TTSConfig(
        model_name=model_name,
        batch_size=args.batch_size,
        chunk_size=args.chunk_size,
        compile_model=not args.no_compile,
        warmup=not args.no_warmup,
        do_sample=not args.greedy,
        temperature=args.temperature,
    )

    if hasattr(args, "speaker"):
        config.speaker = args.speaker

    if hasattr(args, "voice") and args.voice:
        setattr(config, "voice", args.voice)

    return config
