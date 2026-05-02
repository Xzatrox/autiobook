"""shared pooling logic for batch processing."""

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import soundfile as sf  # type: ignore
from tqdm import tqdm  # type: ignore

from .audio import (
    check_segment_exists,
    concatenate_audio,
    load_segment,
    save_segment,
)
from .config import SAMPLE_RATE
from .resume import ResumeManager, compute_hash


@dataclass
class AudioTask:
    """represents a single unit of audio generation work."""

    text: str
    segment_hash: str
    segments_dir: Path
    voice_ref_audio: Optional[Path] = None
    voice_ref_text: Optional[str] = None
    voice_clone_prompt: Optional[Any] = None  # cached VoiceClonePromptItem
    instruct: str = ""
    chapter_idx: int = 0  # for chapter-ordered scheduling
    metadata: Optional[dict[str, Any]] = None  # arbitrary metadata for callbacks


@dataclass
class ChapterState:
    """tracks synthesis and assembly state for a chapter."""

    wav_path: Path
    hashes: list[str]
    segments_dir: Path
    manifest_hash: str
    pending_hashes: set[str] = field(default_factory=set)
    assembled: bool = False


def _synthesize_batch(
    engine: Any,
    batch: list[AudioTask],
    pbar: tqdm | None = None,
) -> None:
    """synthesize a batch of tasks with the same voice reference."""
    if not batch:
        return

    texts = [t.text for t in batch]
    ref_audio = batch[0].voice_ref_audio
    ref_text = batch[0].voice_ref_text
    instruct = batch[0].instruct
    has_prompt = batch[0].voice_clone_prompt is not None

    if pbar and len(texts) > 1:
        avg_len = sum(len(t) for t in texts) // len(texts)
        max_len = max(len(t) for t in texts)
        pbar.write(f"  batch: {len(texts)} segs, prompt={'yes' if has_prompt else 'no'}, avg={avg_len}c, max={max_len}c")

    import time as _time
    _batch_t0 = _time.time()

    # pad batch to avoid torch.compile recompilation/hangs on last batch
    original_size = len(texts)
    padded = False

    if (
        hasattr(engine, "config")
        and getattr(engine.config, "compile_model", False)
        and len(texts) < engine.config.batch_size
    ):
        pad_size = engine.config.batch_size - len(texts)
        # pad with minimal text to fill batch without consuming much memory
        texts.extend(["."] * pad_size)
        padded = True

    try:
        if ref_audio:
            prompt = batch[0].voice_clone_prompt
            if prompt is not None:
                wavs, _ = engine.clone_voice(texts, voice_clone_prompt=prompt)
            else:
                wavs, _ = engine.clone_voice(texts, ref_audio, ref_text)
        else:
            wavs, _ = engine.synthesize(texts, instruct)

        if isinstance(wavs, np.ndarray):
            wavs = [wavs]

        if pbar and len(texts) > 1:
            _batch_elapsed = _time.time() - _batch_t0
            pbar.write(f"  batch done: {_batch_elapsed:.1f}s for {len(batch)} segs = {_batch_elapsed/len(batch):.1f}s/seg")

        if padded:
            wavs = wavs[:original_size]

        for j, audio in enumerate(wavs):
            if j < len(batch):
                save_segment(
                    batch[j].segments_dir,
                    batch[j].segment_hash,
                    audio,
                    SAMPLE_RATE,
                )
            if pbar:
                pbar.update(1)

    except RuntimeError as e:
        if "probability tensor" in str(e).lower() or "inf" in str(e).lower() or "nan" in str(e).lower():
            # sampling produced invalid logits — retry with greedy decoding
            if pbar:
                pbar.write(f"  warning: sampling failed, retrying batch with greedy decoding")
            try:
                old_sample = engine.config.do_sample
                engine.config.do_sample = False
                if ref_audio:
                    prompt = batch[0].voice_clone_prompt
                    if prompt is not None:
                        wavs, _ = engine.clone_voice(texts, voice_clone_prompt=prompt)
                    else:
                        wavs, _ = engine.clone_voice(texts, ref_audio, ref_text)
                else:
                    wavs, _ = engine.synthesize(texts, instruct)
                engine.config.do_sample = old_sample

                if isinstance(wavs, np.ndarray):
                    wavs = [wavs]
                if padded:
                    wavs = wavs[:original_size]
                for j, audio in enumerate(wavs):
                    if j < len(batch):
                        save_segment(batch[j].segments_dir, batch[j].segment_hash, audio, SAMPLE_RATE)
                    if pbar:
                        pbar.update(1)
                return
            except Exception as retry_e:
                print(f"greedy retry also failed: {retry_e}")
                raise retry_e
        elif "out of memory" in str(e).lower():
            print("warning: OOM detected, clearing cache and retrying...")
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                torch.mps.empty_cache()
            try:
                if ref_audio:
                    prompt = batch[0].voice_clone_prompt
                    if prompt is not None:
                        wavs, _ = engine.clone_voice(texts, voice_clone_prompt=prompt)
                    else:
                        wavs, _ = engine.clone_voice(texts, ref_audio, ref_text)
                else:
                    wavs, _ = engine.synthesize(texts, instruct)

                if isinstance(wavs, np.ndarray):
                    wavs = [wavs]

                if padded:
                    wavs = wavs[:original_size]

                for j, audio in enumerate(wavs):
                    if j < len(batch):
                        save_segment(
                            batch[j].segments_dir,
                            batch[j].segment_hash,
                            audio,
                            SAMPLE_RATE,
                        )
                    if pbar:
                        pbar.update(1)
                return
            except Exception as retry_e:
                print(f"retry failed: {retry_e}")
                raise retry_e
        else:
            print(f"batch generation failed: {e}")
            raise e

    except Exception as e:
        print(f"batch generation failed: {e}")
        raise e


def _assemble_chapter(
    state: ChapterState,
    resume: ResumeManager | None = None,
) -> bool:
    """assemble a chapter from its segments. returns True on success."""
    print(f"assembling {state.wav_path.name}...")
    try:
        audio = concatenate_audio([load_segment(state.segments_dir, h) for h in state.hashes])
        sf.write(str(state.wav_path), audio, SAMPLE_RATE)
        if resume:
            resume.update(str(state.wav_path), state.manifest_hash)
            resume.save()
        print(f"  -> {state.wav_path.name}")
        state.assembled = True
        return True
    except Exception as e:
        print(f"failed to assemble {state.wav_path.name}: {e}")
        return False


def _try_assemble_ready_chapters(
    chapters: list[ChapterState],
    generated_hashes: set[str],
    resume: ResumeManager | None = None,
) -> None:
    """assemble chapters that have all segments ready, in order."""
    for ch in chapters:
        if ch.assembled:
            continue
        # update pending hashes based on what's been generated
        ch.pending_hashes -= generated_hashes
        # can only assemble if all segments are ready
        if ch.pending_hashes:
            # stop at first incomplete chapter to maintain order
            break
        _assemble_chapter(ch, resume)


def _get_voice_key(task: AudioTask) -> tuple[str | None, str | None]:
    """get the voice grouping key for a task."""
    return (
        str(task.voice_ref_audio) if task.voice_ref_audio else None,
        task.voice_ref_text,
    )


def _select_batch_for_chapter(
    chapter_idx: int,
    tasks_by_voice: dict[tuple, list[AudioTask]],
    batch_size: int,
    needed_hashes: set[str],
) -> list[AudioTask]:
    """select a batch of tasks prioritizing the given chapter.

    for voice cloning, we batch by voice but prefer tasks needed by the current chapter.
    this increases the chance of completing earlier chapters sooner.
    """
    # find voice groups that have tasks for this chapter
    for voice_key, tasks in tasks_by_voice.items():
        # get tasks needed by this chapter first
        batch = []
        for t in tasks:
            if t.segment_hash in needed_hashes:
                batch.append(t)
                if len(batch) == batch_size:
                    break

        if batch:
            # fill remaining slots with tasks from same voice, any chapter
            if len(batch) < batch_size:
                for t in tasks:
                    if t not in batch:
                        batch.append(t)
                        if len(batch) == batch_size:
                            break

            # remove selected tasks from the pool
            for t in batch:
                tasks.remove(t)
            # clean up empty voice groups
            if not tasks:
                del tasks_by_voice[voice_key]
            return batch
    return []


def process_audio_pipeline(
    engine: Any,
    chapter_data: list[tuple[Path, list[AudioTask]]],
    resume: ResumeManager | None = None,
    desc: str = "generating audio",
    force: bool = False,
) -> None:
    """unified pipeline for batch processing audio segments and assembling chapters.

    processes chapters in order, assembling each as soon as all its segments are ready.
    for voice cloning, batches by voice but prioritizes earlier chapters.
    """
    if not chapter_data:
        return

    # build chapter states and collect all pending tasks
    chapters: list[ChapterState] = []
    all_pending_tasks: list[AudioTask] = []
    generated_hashes: set[str] = set()  # track globally generated segments

    for ch_idx, (wav_path, tasks) in enumerate(chapter_data):
        if not tasks:
            continue

        seg_dir = tasks[0].segments_dir
        hashes = [t.segment_hash for t in tasks]
        m_hash = compute_hash(hashes)

        # determine which segments need generation
        pending_hashes = set()
        for t in tasks:
            t.chapter_idx = ch_idx
            if force or not check_segment_exists(t.segments_dir, t.segment_hash):
                pending_hashes.add(t.segment_hash)
                # avoid duplicate synthesis across chapters
                if t.segment_hash not in generated_hashes:
                    all_pending_tasks.append(t)
                    generated_hashes.add(t.segment_hash)

        needs_assembly = (
            force
            or not wav_path.exists()
            or (resume and not resume.is_fresh(str(wav_path), m_hash))
        )

        chapters.append(
            ChapterState(
                wav_path=wav_path,
                hashes=hashes,
                segments_dir=seg_dir,
                manifest_hash=m_hash,
                pending_hashes=pending_hashes,
                assembled=not needs_assembly,
            )
        )

    # reset for tracking during synthesis
    generated_hashes = set()

    if not all_pending_tasks:
        # all segments cached, just assemble
        _try_assemble_ready_chapters(chapters, set(), resume)
        return

    # group tasks by voice for efficient batching
    tasks_by_voice: dict[tuple, list[AudioTask]] = defaultdict(list)
    for task in all_pending_tasks:
        key = _get_voice_key(task)
        tasks_by_voice[key].append(task)

    # sort tasks within each voice group by text length (similar lengths batch together)
    for tasks in tasks_by_voice.values():
        tasks.sort(key=lambda t: len(t.text))

    # re-bucket by voice + length tier so batches have uniform text lengths
    LENGTH_TIERS = [50, 100, 200, 10000]  # char thresholds
    rebucketed: dict[tuple, list[AudioTask]] = defaultdict(list)
    for voice_key, tasks in tasks_by_voice.items():
        for t in tasks:
            tlen = len(t.text)
            tier = next(th for th in LENGTH_TIERS if tlen <= th)
            rebucketed[(voice_key, tier)].append(t)
    tasks_by_voice.clear()
    tasks_by_voice.update(rebucketed)

    total_tasks = len(all_pending_tasks)
    with tqdm(total=total_tasks, desc=desc, unit="seg") as pbar:
        # process chapter by chapter, but batch by voice
        for ch_idx, ch in enumerate(chapters):
            if ch.assembled:
                continue

            # synthesize batches until this chapter is ready
            while ch.pending_hashes - generated_hashes:
                needed = ch.pending_hashes - generated_hashes
                batch = _select_batch_for_chapter(
                    ch_idx, tasks_by_voice, engine.config.batch_size, needed
                )
                if not batch:
                    # no more tasks for any voice, shouldn't happen
                    break

                _synthesize_batch(engine, batch, pbar)
                for t in batch:
                    generated_hashes.add(t.segment_hash)

                # try to assemble any chapters that are now ready
                _try_assemble_ready_chapters(chapters, generated_hashes, resume)

        # process any remaining tasks (for later chapters)
        while any(tasks_by_voice.values()):
            # pick first non-empty voice group
            for voice_key, tasks in list(tasks_by_voice.items()):
                if tasks:
                    batch = tasks[: engine.config.batch_size]
                    for t in batch:
                        tasks.remove(t)
                    if not tasks:
                        del tasks_by_voice[voice_key]
                    _synthesize_batch(engine, batch, pbar)
                    for t in batch:
                        generated_hashes.add(t.segment_hash)
                    _try_assemble_ready_chapters(chapters, generated_hashes, resume)
                    break

    # final assembly pass for any remaining chapters
    _try_assemble_ready_chapters(chapters, generated_hashes, resume)
