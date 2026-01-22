"dramatization workflow logic."

import json
from pathlib import Path
from typing import List

import numpy as np
import soundfile as sf
from tqdm import tqdm

from .audio import concatenate_audio
from .config import (
    BASE_MODEL,
    CAST_FILE,
    PARAGRAPH_PAUSE_MS,
    SAMPLE_RATE,
    SCRIPT_EXT,
    TXT_EXT,
    VOICE_DESIGN_MODEL,
    WAV_EXT,
)
from .llm import Character, ScriptSegment, generate_cast, process_script_chunk, split_text_smart
from .tts import (
    TTSConfig,
    TTSEngine,
    chunk_text,
    finalize_chunks,
    get_chunk_dir,
    load_chunk_progress,
    save_chunk_audio,
    save_chunk_progress,
)
from .utils import iter_pending_chapters


def save_cast(workdir: Path, cast: List[Character], analyzed_chapters: List[int] = None) -> None:
    """save cast to json file."""
    path = workdir / CAST_FILE

    characters = []
    for c in cast:
        char_data = {
            "name": c.name,
            "description": c.description,
            "audition_line": c.audition_line,
        }
        if c.aliases:
            char_data["aliases"] = c.aliases
        if c.appearances > 0:
            char_data["appearances"] = c.appearances
        characters.append(char_data)

    data = {
        "version": 3,
        "analyzed_chapters": analyzed_chapters or [],
        "characters": characters,
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_cast(workdir: Path) -> tuple[List[Character], List[int]]:
    """load cast from json file. returns (characters, analyzed_chapters)."""
    path = workdir / CAST_FILE
    if not path.exists():
        return [], []

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    # handle legacy list format
    if isinstance(data, list):
        chars = []
        for c in data:
            c.setdefault("aliases", None)
            c.setdefault("appearances", 0)
            chars.append(Character(**c))
        return chars, []

    # handle dict format (v1, v2, v3)
    chars = []
    for c in data.get("characters", []):
        c.setdefault("aliases", None)
        c.setdefault("appearances", 0)
        chars.append(Character(**c))
    analyzed = data.get("analyzed_chapters", [])
    return chars, analyzed


def save_script(
    chapter_file: Path,
    segments: List[ScriptSegment],
    completed_chunks: int | None = None,
    total_chunks: int | None = None,
) -> None:
    """save dramatized script for a chapter with progress tracking."""
    script_path = chapter_file.with_suffix(SCRIPT_EXT)
    data = {
        "version": 1,
        "completed_chunks": completed_chunks,
        "total_chunks": total_chunks,
        "segments": [
            {
                "speaker": s.speaker,
                "text": s.text,
                "instruction": s.instruction,
            }
            for s in segments
        ],
    }
    with open(script_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_script(chapter_file: Path) -> tuple[List[ScriptSegment], int, int]:
    """load dramatized script for a chapter.

    returns (segments, completed_chunks, total_chunks).
    """
    script_path = chapter_file.with_suffix(SCRIPT_EXT)
    if not script_path.exists():
        return [], 0, 0

    with open(script_path, encoding="utf-8") as f:
        data = json.load(f)

    # handle legacy format (list or dict without version)
    if isinstance(data, list):
        segments = [ScriptSegment(**s) for s in data]
        return segments, len(segments), len(segments)

    segments = [ScriptSegment(**s) for s in data.get("segments", [])]
    completed = data.get("completed_chunks", len(segments))
    total = data.get("total_chunks", len(segments))

    return segments, completed, total


def count_appearances(workdir: Path, chapters: list[int] | None = None) -> dict[str, int]:
    """count speaking appearances for each character across scripts."""
    from collections import Counter

    counts = Counter()
    script_files = sorted(workdir.glob(f"*{SCRIPT_EXT}"))

    for script_path in script_files:
        # filter by chapter if specified
        if chapters:
            try:
                chapter_num = int(script_path.stem.split("_")[0])
                if chapter_num not in chapters:
                    continue
            except ValueError:
                continue

        segments, _, _ = load_script(script_path.with_suffix(TXT_EXT))
        for seg in segments:
            counts[seg.speaker] += 1

    return dict(counts)


def _merge_character_into_cast(
    c: Character,
    cast_map: dict[str, Character],
    alias_map: dict[str, str],
) -> str:
    """merge a character into the cast, returns 'added', 'updated', or 'merged'."""
    key = c.name.lower()

    # check if this name is an alias of an existing character
    if key in alias_map:
        canonical_key = alias_map[key]
        existing = cast_map[canonical_key]
        new_aliases = set(existing.aliases or [])
        if c.aliases:
            new_aliases.update(c.aliases)
        new_aliases.add(c.name)
        new_aliases.discard(existing.name)
        existing.aliases = sorted(new_aliases) if new_aliases else None
        return "merged"

    # check if any of this character's aliases match an existing character
    matched_key = None
    if c.aliases:
        for alias in c.aliases:
            alias_lower = alias.lower()
            if alias_lower in cast_map:
                matched_key = alias_lower
                break
            if alias_lower in alias_map:
                matched_key = alias_map[alias_lower]
                break

    if matched_key:
        existing = cast_map[matched_key]
        new_aliases = set(existing.aliases or [])
        if c.aliases:
            new_aliases.update(c.aliases)
        new_aliases.add(c.name)
        new_aliases.discard(existing.name)
        existing.aliases = sorted(new_aliases) if new_aliases else None
        if len(c.description) > len(existing.description):
            existing.description = c.description
        return "merged"

    if key in cast_map:
        old = cast_map[key]
        changed = False
        if c.description != old.description:
            old.description = c.description
            changed = True
        new_aliases = set(old.aliases or [])
        if c.aliases:
            new_aliases.update(c.aliases)
        if new_aliases != set(old.aliases or []):
            old.aliases = sorted(new_aliases) if new_aliases else None
            changed = True
        return "updated" if changed else "unchanged"

    # new character
    cast_map[key] = c
    if c.aliases:
        for alias in c.aliases:
            alias_map[alias.lower()] = key
    return "added"


def run_cast_generation(
    workdir: Path,
    api_base: str | None = None,
    api_key: str | None = None,
    model: str | None = None,
    chapters: list[int] | None = None,
) -> List[Character]:
    """analyze book and generate cast list."""
    existing_cast, analyzed_chapters = load_cast(workdir)

    txt_files = sorted(workdir.glob(f"*{TXT_EXT}"))
    if not txt_files:
        print("no extracted text files found!")
        return existing_cast

    # map chapter numbers to files
    chapter_map = {}
    for txt_path in txt_files:
        try:
            num = int(txt_path.stem.split("_")[0])
            chapter_map[num] = txt_path
        except ValueError:
            continue

    all_chapter_nums = sorted(chapter_map.keys())

    # determine which chapters to process
    if chapters:
        chapters_to_process = [c for c in chapters if c in chapter_map]
    else:
        chapters_to_process = [num for num in all_chapter_nums if num not in analyzed_chapters]

    if not chapters_to_process:
        if not chapters:
            print(f"all chapters ({len(all_chapter_nums)}) have been analyzed for cast.")
            return existing_cast
        else:
            print("no matching chapters found to process.")
            return existing_cast

    print(f"analyzing {len(chapters_to_process)} chapters for cast...")

    # build lookup maps from existing cast
    cast_map = {c.name.lower(): c for c in existing_cast}
    alias_map = {}
    for c in existing_cast:
        if c.aliases:
            for alias in c.aliases:
                alias_map[alias.lower()] = c.name.lower()

    # track stats across all batches
    added_count = 0
    updated_count = 0
    merged_count = 0
    total_processed = 0

    # process in batches to avoid overwhelming the LLM
    batch_size = 3
    num_batches = (len(chapters_to_process) + batch_size - 1) // batch_size

    for batch_start in range(0, len(chapters_to_process), batch_size):
        batch_chapters = chapters_to_process[batch_start : batch_start + batch_size]
        batch_num = batch_start // batch_size + 1

        # collect samples for this batch
        full_sample = ""
        for num in batch_chapters:
            txt_path = chapter_map[num]
            text = txt_path.read_text(encoding="utf-8")
            full_sample += f"\n--- Chapter {txt_path.stem} ---\n"
            full_sample += text[:2000]

        # format current cast for context (include aliases)
        current_cast = list(cast_map.values())
        summary = ""
        if current_cast:
            lines = []
            for c in current_cast:
                line = f"- {c.name}: {c.description}"
                if c.aliases:
                    line += f" (also known as: {', '.join(c.aliases)})"
                lines.append(line)
            summary = "\n".join(lines)

        print(
            f"  batch {batch_num}/{num_batches}: chapters {batch_chapters} ({len(current_cast)} characters known)..."
        )

        batch_cast = generate_cast(
            full_sample, api_base, api_key, model or "gpt-4o", existing_cast_summary=summary
        )
        total_processed += len(batch_cast)

        # merge batch results into cast
        for c in batch_cast:
            result = _merge_character_into_cast(c, cast_map, alias_map)
            if result == "added":
                added_count += 1
            elif result == "updated":
                updated_count += 1
            elif result == "merged":
                merged_count += 1

        # update analyzed chapters and save after each batch
        analyzed_chapters = sorted(list(set(analyzed_chapters + batch_chapters)))
        final_cast = list(cast_map.values())

        # ensure Narrator is at top if present
        narrator = next((c for c in final_cast if c.name.lower() == "narrator"), None)
        if narrator:
            final_cast.remove(narrator)
            final_cast.insert(0, narrator)

        save_cast(workdir, final_cast, analyzed_chapters)

    print(
        f"processed {total_processed} character mentions from {len(chapters_to_process)} chapters."
    )
    print(f"stats: {added_count} added, {updated_count} updated, {merged_count} merged (aliases).")
    print(
        f"final cast: {len(final_cast)} characters. analyzed: {len(analyzed_chapters)}/{len(all_chapter_nums)} chapters."
    )

    return final_cast


def run_auditions(
    workdir: Path,
    cast: List[Character] | None = None,
    min_appearances: int = 0,
) -> None:
    """generate voice samples for cast."""
    from .config import (
        EXTRA_FEMALE,
        EXTRA_FEMALE_DESC,
        EXTRA_FEMALE_LINE,
        EXTRA_MALE,
        EXTRA_MALE_DESC,
        EXTRA_MALE_LINE,
    )

    if cast is None:
        cast, _ = load_cast(workdir)

    if not cast:
        print("no cast found. run 'cast' command first.")
        return

    voices_dir = workdir / "voices"
    voices_dir.mkdir(exist_ok=True)

    # determine which characters need dedicated voices vs generic extras
    main_cast = []
    minor_count = 0
    for char in cast:
        # narrator and characters meeting threshold get dedicated voices
        if char.name.lower() == "narrator" or char.appearances >= min_appearances:
            main_cast.append(char)
        else:
            minor_count += 1

    # add generic extras if we have minor characters
    need_extras = minor_count > 0 and min_appearances > 0
    if need_extras:
        # check if extras already exist
        extra_female_path = voices_dir / f"{EXTRA_FEMALE}{WAV_EXT}"
        extra_male_path = voices_dir / f"{EXTRA_MALE}{WAV_EXT}"
        if not extra_female_path.exists():
            main_cast.append(
                Character(EXTRA_FEMALE, EXTRA_FEMALE_DESC, EXTRA_FEMALE_LINE, appearances=0)
            )
        if not extra_male_path.exists():
            main_cast.append(
                Character(EXTRA_MALE, EXTRA_MALE_DESC, EXTRA_MALE_LINE, appearances=0)
            )

    engine = TTSEngine(TTSConfig(model_name=VOICE_DESIGN_MODEL))

    if min_appearances > 0:
        print(
            f"generating auditions for {len(main_cast)} characters "
            f"({minor_count} minor characters will use generic voices)..."
        )
    else:
        print(f"generating auditions for {len(main_cast)} characters...")

    for char in tqdm(main_cast, desc="casting voices"):
        wav_path = voices_dir / f"{char.name}{WAV_EXT}"
        if wav_path.exists():
            continue

        try:
            audio, sr = engine.design_voice(text=char.audition_line, instruct=char.description)
            sf.write(str(wav_path), audio, sr)
        except Exception as e:
            print(f"failed to generate voice for {char.name}: {e}")


def run_script_generation(
    workdir: Path,
    api_base: str | None = None,
    api_key: str | None = None,
    model: str | None = None,
    chapters: list[int] | None = None,
) -> None:
    """generate dramatized scripts for chapters incrementally."""
    cast, _ = load_cast(workdir)
    if not cast:
        print("no cast found. run 'cast' command first.")
        return

    # collect chapters to process
    txt_files = sorted(workdir.glob(f"*{TXT_EXT}"))
    pending_chapters = []
    for txt_path in txt_files:
        try:
            chapter_num = int(txt_path.stem.split("_")[0])
        except ValueError:
            continue
        if chapters and chapter_num not in chapters:
            continue
        pending_chapters.append((chapter_num, txt_path))

    if not pending_chapters:
        print("no chapters to process.")
        return

    # count completed vs pending
    completed_count = 0
    pending_count = 0
    for _, txt_path in pending_chapters:
        _, completed, total = load_script(txt_path)
        text = txt_path.read_text(encoding="utf-8")
        actual_total = len(split_text_smart(text))
        if completed >= actual_total and completed > 0:
            completed_count += 1
        else:
            pending_count += 1

    print(
        f"script generation: {pending_count} chapters to process, {completed_count} already complete"
    )

    total_segments = 0
    chapters_processed = 0

    for chapter_idx, (chapter_num, txt_path) in enumerate(pending_chapters):
        text = txt_path.read_text(encoding="utf-8")
        chunks = split_text_smart(text)
        total_chunks = len(chunks)

        # load existing progress
        existing_segments, completed_chunks, saved_total = load_script(txt_path)

        # if source changed (different chunk count), restart
        if saved_total and saved_total != total_chunks:
            print(
                f"  [{chapter_idx + 1}/{len(pending_chapters)}] restarting {txt_path.name} (source changed)..."
            )
            existing_segments = []
            completed_chunks = 0

        # check if already complete
        if completed_chunks >= total_chunks and existing_segments:
            continue

        if completed_chunks > 0:
            status = f"resuming at chunk {completed_chunks + 1}"
        else:
            status = "starting"

        print(
            f"  [{chapter_idx + 1}/{len(pending_chapters)}] {txt_path.name}: {status} ({total_chunks} chunks)"
        )

        segments = list(existing_segments)

        for i in tqdm(
            range(completed_chunks, total_chunks),
            desc=f"    chapter {chapter_num}",
            unit="chunk",
            initial=completed_chunks,
            total=total_chunks,
        ):
            chunk = chunks[i]
            try:
                chunk_segments = process_script_chunk(
                    chunk, cast, api_base, api_key, model or "gpt-4o"
                )
                segments.extend(chunk_segments)
                completed_chunks = i + 1
                save_script(txt_path, segments, completed_chunks, total_chunks)

            except Exception as e:
                print(f"\n    chunk {i + 1} FAILED: {e}")
                return

        total_segments += len(segments)
        chapters_processed += 1
        print(f"    -> {len(segments)} segments")

    print(f"done: {chapters_processed} chapters, {total_segments} total segments")

    # update appearance counts in cast
    appearances = count_appearances(workdir, chapters)
    cast, analyzed = load_cast(workdir)
    for c in cast:
        # sum appearances for character and all aliases
        total = appearances.get(c.name, 0)
        if c.aliases:
            for alias in c.aliases:
                total += appearances.get(alias, 0)
        c.appearances = total
    save_cast(workdir, cast, analyzed)
    print(f"updated appearance counts for {len(cast)} characters")


def run_performance(
    workdir: Path,
    chapters: list[int] | None = None,
    config: TTSConfig | None = None,
    pooled: bool = False,
    min_appearances: int = 0,
) -> None:
    """synthesize audio from scripts with segment-level resume."""
    from .config import EXTRA_FEMALE, EXTRA_MALE

    cast, _ = load_cast(workdir)
    if not cast:
        print("no cast found. run 'cast' command first.")
        return

    # build cast map including aliases
    # minor characters (below min_appearances) get mapped to generic Extra voices
    cast_map = {}
    for c in cast:
        # narrator always gets own voice
        if c.name.lower() == "narrator" or c.appearances >= min_appearances:
            cast_map[c.name] = c
            if c.aliases:
                for alias in c.aliases:
                    cast_map[alias] = c
        elif min_appearances > 0:
            # map minor character to generic extra (alternate male/female)
            # use description to guess gender, default to male
            desc_lower = c.description.lower()
            is_female = any(w in desc_lower for w in ["female", "woman", "girl", "she", "her"])
            extra_name = EXTRA_FEMALE if is_female else EXTRA_MALE
            extra_char = Character(extra_name, "", "", appearances=0)
            cast_map[c.name] = extra_char
            if c.aliases:
                for alias in c.aliases:
                    cast_map[alias] = extra_char

    voices_dir = workdir / "voices"

    if not voices_dir.exists():
        print("no voices found. run 'audition' command first.")
        return

    # use provided config but override model to BASE_MODEL for voice cloning
    if config is None:
        config = TTSConfig(model_name=BASE_MODEL)
    else:
        config = TTSConfig(
            model_name=BASE_MODEL,
            batch_size=config.batch_size,
            chunk_size=config.chunk_size,
            compile_model=config.compile_model,
            warmup=config.warmup,
            do_sample=config.do_sample,
            temperature=config.temperature,
        )
    engine = TTSEngine(config)

    for txt_path, wav_path in iter_pending_chapters(workdir, chapters, skip_message="audio exists"):
        segments, _, _ = load_script(txt_path)
        if not segments:
            print(f"skipping {txt_path.name} (no script found)")
            continue

        _perform_chapter_resumable(engine, txt_path, wav_path, segments, voices_dir, cast_map)


def _perform_chapter_resumable(
    engine: TTSEngine,
    txt_path: Path,
    wav_path: Path,
    segments: List[ScriptSegment],
    voices_dir: Path,
    cast_map: dict[str, Character],
) -> None:
    """synthesize dramatized chapter with segment-level progress tracking."""
    import shutil

    total_segments = len(segments)
    chunk_dir = get_chunk_dir(wav_path)

    # check for existing progress
    completed, saved_total = load_chunk_progress(chunk_dir)

    # restart if source changed
    if saved_total and saved_total != total_segments:
        print(f"restarting {txt_path.name} (script changed)...")
        shutil.rmtree(chunk_dir, ignore_errors=True)
        completed = 0

    # check if already complete
    if completed >= total_segments:
        print(f"finalizing {txt_path.name}...")
        finalize_chunks(chunk_dir, wav_path, total_segments)
        print(f"  -> {wav_path.name}")
        return

    chunk_dir.mkdir(parents=True, exist_ok=True)

    if completed > 0:
        print(f"resuming {txt_path.name} at segment {completed + 1}/{total_segments}...")
    else:
        print(f"synthesizing {txt_path.name} with cast ({total_segments} segments)...")

    for seg_idx in tqdm(
        range(completed, total_segments),
        desc="performing",
        unit="seg",
        initial=completed,
        total=total_segments,
        leave=False,
    ):
        segment = segments[seg_idx]
        character = cast_map.get(segment.speaker) or cast_map.get("Narrator")

        if not character:
            # save silence for missing character
            save_chunk_audio(chunk_dir, seg_idx, np.zeros(1, dtype=np.float32), SAMPLE_RATE)
            save_chunk_progress(chunk_dir, seg_idx + 1, total_segments)
            continue

        ref_audio_path = voices_dir / f"{character.name}{WAV_EXT}"
        if not ref_audio_path.exists():
            save_chunk_audio(chunk_dir, seg_idx, np.zeros(1, dtype=np.float32), SAMPLE_RATE)
            save_chunk_progress(chunk_dir, seg_idx + 1, total_segments)
            continue

        chunks = [c for c in chunk_text(segment.text, engine.config.chunk_size) if c.strip()]
        if not chunks:
            save_chunk_audio(chunk_dir, seg_idx, np.zeros(1, dtype=np.float32), SAMPLE_RATE)
            save_chunk_progress(chunk_dir, seg_idx + 1, total_segments)
            continue

        try:
            # synthesize all chunks for this segment
            segment_audio = []
            for i in range(0, len(chunks), engine.config.batch_size):
                batch = chunks[i : i + engine.config.batch_size]
                if len(batch) == 1:
                    audio, _ = engine.clone_voice(
                        text=batch[0], ref_audio=ref_audio_path, ref_text=character.audition_line
                    )
                    segment_audio.append(audio)
                else:
                    audios, _ = engine.clone_voice(
                        text=batch, ref_audio=ref_audio_path, ref_text=character.audition_line
                    )
                    segment_audio.extend(audios)

            combined = concatenate_audio(segment_audio, SAMPLE_RATE, PARAGRAPH_PAUSE_MS)
            save_chunk_audio(chunk_dir, seg_idx, combined, SAMPLE_RATE)
        except Exception as e:
            print(f"failed segment {seg_idx}: {e}")
            save_chunk_audio(chunk_dir, seg_idx, np.zeros(1, dtype=np.float32), SAMPLE_RATE)

        save_chunk_progress(chunk_dir, seg_idx + 1, total_segments)

    finalize_chunks(chunk_dir, wav_path, total_segments)
    print(f"  -> {wav_path.name}")


def synthesize_dramatized_chapter(
    engine: TTSEngine,
    segments: List[ScriptSegment],
    voices_dir: Path,
    cast_map: dict[str, Character],
) -> tuple[np.ndarray, int]:
    """synthesize a single chapter script using cloned voices with batching."""
    audio_segments = []
    sample_rate = SAMPLE_RATE

    for segment in tqdm(segments, desc="synthesizing segments", leave=False):
        character = cast_map.get(segment.speaker)
        if not character:
            character = cast_map.get("Narrator")

        if not character:
            print(f"warning: unknown speaker {segment.speaker} and no Narrator found")
            continue

        ref_audio_path = voices_dir / f"{character.name}{WAV_EXT}"
        if not ref_audio_path.exists():
            print(f"warning: missing voice for {character.name}")
            continue

        chunks = [c for c in chunk_text(segment.text, engine.config.chunk_size) if c.strip()]
        if not chunks:
            continue

        # batch all chunks for this segment (same voice reference)
        try:
            for i in range(0, len(chunks), engine.config.batch_size):
                batch = chunks[i : i + engine.config.batch_size]
                if len(batch) == 1:
                    audio, sample_rate = engine.clone_voice(
                        text=batch[0], ref_audio=ref_audio_path, ref_text=character.audition_line
                    )
                    audio_segments.append(audio)
                else:
                    audios, sample_rate = engine.clone_voice(
                        text=batch, ref_audio=ref_audio_path, ref_text=character.audition_line
                    )
                    audio_segments.extend(audios)
        except Exception as e:
            print(f"failed to synthesize segment: {e}")

    return concatenate_audio(audio_segments, sample_rate, PARAGRAPH_PAUSE_MS), sample_rate


# CLI Command Wrappers


def cmd_cast(args):
    chapters = None
    if args.chapters:
        from .utils import parse_chapter_range

        chapters = parse_chapter_range(args.chapters)

    run_cast_generation(
        Path(args.workdir),
        api_base=args.api_base,
        api_key=args.api_key,
        model=args.model,
        chapters=chapters,
    )


def cmd_audition(args):
    min_appearances = getattr(args, "min_appearances", 0)
    run_auditions(Path(args.workdir), min_appearances=min_appearances)


def cmd_script(args):
    chapters = None
    if args.chapters:
        from .utils import parse_chapter_range

        chapters = parse_chapter_range(args.chapters)

    run_script_generation(
        Path(args.workdir),
        api_base=args.api_base,
        api_key=args.api_key,
        model=args.model,
        chapters=chapters,
    )


def cmd_perform(args):
    chapters = None
    if args.chapters:
        from .utils import parse_chapter_range

        chapters = parse_chapter_range(args.chapters)

    config = TTSConfig(
        batch_size=args.batch_size,
        chunk_size=args.chunk_size,
        compile_model=not args.no_compile,
        warmup=not args.no_warmup,
        do_sample=not args.greedy,
        temperature=args.temperature,
    )

    min_appearances = getattr(args, "min_appearances", 0)
    run_performance(Path(args.workdir), chapters, config, args.pooled, min_appearances)


def dramatize_book(
    workdir: Path,
    api_base: str | None = None,
    api_key: str | None = None,
    model: str | None = None,
    chapters: list[int] | None = None,
    tts_config: TTSConfig | None = None,
    pooled: bool = False,
    min_appearances: int = 0,
) -> None:
    """run full dramatization pipeline."""
    cast = run_cast_generation(workdir, api_base, api_key, model, chapters)
    # generate scripts first to count appearances before auditions
    run_script_generation(workdir, api_base, api_key, model, chapters)
    # now run auditions with appearance counts available
    run_auditions(workdir, min_appearances=min_appearances)
    run_performance(workdir, chapters, tts_config, pooled, min_appearances)
