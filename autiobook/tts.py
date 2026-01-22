"""tts engine wrapper for qwen3-tts with rocm optimizations."""

import json
import os
import re
import shutil
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import transformers
from tqdm import tqdm

from .audio import concatenate_audio
from .config import (
    CHUNK_PROGRESS_FILE,
    CHUNKS_DIR,
    DEFAULT_MODEL,
    DEFAULT_SPEAKER,
    MAX_CHUNK_SIZE,
    PARAGRAPH_PAUSE_MS,
    SAMPLE_RATE,
    WAV_EXT,
)
from .utils import iter_pending_chapters

transformers.logging.set_verbosity_error()

SENTENCE_ENDINGS = re.compile(r"(?<=[.!?])\s+")

# warmup text for model compilation
WARMUP_TEXT = "Hello, this is a warmup."


def get_chunk_dir(wav_path: Path) -> Path:
    """get chunk directory for a chapter wav file."""
    return wav_path.parent / CHUNKS_DIR / wav_path.stem


def save_chunk_progress(chunk_dir: Path, completed: int, total: int) -> None:
    """save synthesis progress to json file."""
    progress_path = chunk_dir / CHUNK_PROGRESS_FILE
    with open(progress_path, "w") as f:
        json.dump({"completed": completed, "total": total}, f)


def load_chunk_progress(chunk_dir: Path) -> tuple[int, int]:
    """load synthesis progress. returns (completed, total)."""
    progress_path = chunk_dir / CHUNK_PROGRESS_FILE
    if not progress_path.exists():
        return 0, 0
    with open(progress_path) as f:
        data = json.load(f)
    return data.get("completed", 0), data.get("total", 0)


def save_chunk_audio(chunk_dir: Path, chunk_idx: int, audio: np.ndarray, sample_rate: int) -> None:
    """save a single audio chunk to disk."""
    chunk_path = chunk_dir / f"chunk_{chunk_idx:04d}{WAV_EXT}"
    sf.write(str(chunk_path), audio, sample_rate)


def load_chunk_audio(chunk_dir: Path, chunk_idx: int) -> np.ndarray:
    """load a single audio chunk from disk."""
    chunk_path = chunk_dir / f"chunk_{chunk_idx:04d}{WAV_EXT}"
    audio, _ = sf.read(str(chunk_path))
    return audio.astype(np.float32)


def finalize_chunks(chunk_dir: Path, wav_path: Path, total_chunks: int) -> None:
    """concatenate all chunks into final wav. chunks are kept for resume capability."""
    audio_chunks = [load_chunk_audio(chunk_dir, i) for i in range(total_chunks)]
    combined = concatenate_audio(audio_chunks, SAMPLE_RATE, PARAGRAPH_PAUSE_MS)
    sf.write(str(wav_path), combined, SAMPLE_RATE)


def get_default_device() -> str:
    """get default device (cuda/rocm if available, else cpu)."""
    has_cuda = torch.cuda.is_available()
    has_rocm = hasattr(torch.version, "hip") and torch.version.hip is not None

    if has_cuda or has_rocm:
        device_count = torch.cuda.device_count()
        if device_count > 0:
            device_type = "cuda"
        else:
            print("WARNING: CUDA/ROCm libraries detected but no devices found. Fallback to CPU.")
            device_type = "cpu"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device_type = "mps"
    else:
        device_type = "cpu"

    print(f"autodetected device type: {device_type}")
    return device_type


def is_rocm() -> bool:
    """check if running on rocm."""
    return hasattr(torch.version, "hip") and torch.version.hip is not None


def setup_rocm_env():
    """set environment variables for optimal rocm performance."""
    if not is_rocm():
        return
    # enable experimental aotriton kernels
    os.environ.setdefault("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1")
    # enable flash attention on amd
    os.environ.setdefault("FLASH_ATTENTION_TRITON_AMD_ENABLE", "TRUE")
    # pre-allocate MIOpen workspace to avoid fallback to slower kernels
    # 256MB should cover most GEMM operations
    os.environ.setdefault("MIOPEN_WORKSPACE_MAX", "256000000")
    # use immediate mode for faster kernel selection
    os.environ.setdefault("MIOPEN_FIND_MODE", "FAST")
    # cache compiled kernels to speed up subsequent runs
    os.environ.setdefault("MIOPEN_USER_DB_PATH", os.path.expanduser("~/.cache/miopen"))


@dataclass
class TTSConfig:
    """configuration for tts generation."""

    model_name: str = DEFAULT_MODEL
    speaker: str = DEFAULT_SPEAKER
    language: str = "English"
    device: str = field(default_factory=get_default_device)
    batch_size: int = 64  # batch 64 shows 7x throughput vs batch 1
    chunk_size: int = MAX_CHUNK_SIZE  # 500 chars balances coherence and speed
    compile_model: bool = True  # use torch.compile for faster inference
    warmup: bool = True  # warmup model on first load
    # generation parameters - can tune for speed vs quality
    do_sample: bool = True  # False = greedy (faster), True = sampling (better quality)
    temperature: float = 0.9  # lower = faster/more deterministic
    max_new_tokens: int = 2048  # limit output length per chunk


class TTSEngine:
    """wrapper for qwen3-tts model with rocm optimizations."""

    def __init__(self, config: TTSConfig | None = None):
        setup_rocm_env()
        self.config = config or TTSConfig()
        self._model = None
        self._loaded_model_name = None  # track loaded model name
        self._compiled = False
        self._sdpa_backends = None  # store backends, create context each time

    def _load_model(self):
        """lazy load and optimize the tts model."""
        if self._model is not None and self._loaded_model_name == self.config.model_name:
            return

        # unload existing model if different
        if self._model is not None:
            print(f"unloading model {self._loaded_model_name}...")
            del self._model
            torch.cuda.empty_cache()
            self._model = None
            self._loaded_model_name = None

        from qwen_tts import Qwen3TTSModel

        # determine dtype and attention implementation
        if "cuda" in self.config.device:
            dtype = torch.bfloat16
            attn_impl = "sdpa" if is_rocm() else "flash_attention_2"
        elif "mps" in self.config.device:
            dtype = torch.float32
            attn_impl = "sdpa"
        else:
            dtype = torch.float32
            attn_impl = "sdpa"

        print(
            f"loading model {self.config.model_name} on {self.config.device} "
            f"with {dtype} and {attn_impl}..."
        )
        self._model = Qwen3TTSModel.from_pretrained(
            self.config.model_name,
            device_map=self.config.device,
            dtype=dtype,
            attn_implementation=attn_impl,
        )
        self._loaded_model_name = self.config.model_name

        # setup attention backends for rocm (context created fresh each call)
        if "cuda" in self.config.device and is_rocm():
            try:
                from torch.nn.attention import SDPBackend

                self._sdpa_backends = [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]
            except ImportError:
                self._sdpa_backends = None
        else:
            self._sdpa_backends = None

        # apply torch.compile for faster inference
        if self.config.compile_model and "cuda" in self.config.device:
            self._compile_model()

        # warmup the model
        if self.config.warmup:
            self._warmup()

    def _get_attn_ctx(self):
        """create fresh attention context for each use."""
        if self._sdpa_backends is not None:
            from torch.nn.attention import sdpa_kernel

            return sdpa_kernel(self._sdpa_backends)
        return nullcontext()

    def _compile_model(self):
        """apply torch.compile to model components."""
        if self._compiled:
            return

        print("compiling model for faster inference...")
        try:
            # compile the main talker model with reduce-overhead mode
            # this optimizes for inference with minimal CPU overhead
            self._model.model.talker = torch.compile(
                self._model.model.talker,
                mode="reduce-overhead",
                fullgraph=False,  # allow graph breaks for compatibility
            )
            self._compiled = True
            print("model compilation complete")
        except Exception as e:
            print(f"warning: torch.compile failed ({e}), using eager mode")
            self._compiled = False

    def _warmup(self):
        """warmup model with a short synthesis to trigger compilation."""
        print("warming up model...")
        with self._get_attn_ctx():
            with torch.inference_mode():
                try:
                    # simple warmup depending on model type
                    if "VoiceDesign" in self.config.model_name:
                        self._model.generate_voice_design(
                            text=WARMUP_TEXT,
                            language=self.config.language,
                            instruct="neutral voice",
                        )
                    elif "Base" in self.config.model_name:
                        # base model needs ref audio, skip complex warmup or use dummy
                        pass
                    else:
                        self._model.generate_custom_voice(
                            text=WARMUP_TEXT,
                            language=self.config.language,
                            speaker=self.config.speaker,
                        )
                    print("warmup complete")
                except Exception as e:
                    print(f"warning: warmup failed ({e})")

    def synthesize(
        self, text: str | list[str], instruct: str = ""
    ) -> tuple[np.ndarray | list[np.ndarray], int]:
        """synthesize speech from text using current model."""
        self._load_model()

        with self._get_attn_ctx():
            with torch.inference_mode():
                wavs, sr = self._model.generate_custom_voice(
                    text=text,
                    language=self.config.language,
                    speaker=self.config.speaker,
                    instruct=instruct,
                    non_streaming_mode=True,
                    do_sample=self.config.do_sample,
                    temperature=self.config.temperature,
                    max_new_tokens=self.config.max_new_tokens,
                )

        if isinstance(text, str):
            return wavs[0], sr
        return wavs, sr

    def design_voice(self, text: str, instruct: str) -> tuple[np.ndarray, int]:
        """generate a voice design sample."""
        self._load_model()

        with self._get_attn_ctx():
            with torch.inference_mode():
                wavs, sr = self._model.generate_voice_design(
                    text=text,
                    language=self.config.language,
                    instruct=instruct,
                    non_streaming_mode=True,
                    do_sample=self.config.do_sample,
                    temperature=self.config.temperature,
                )
        return wavs[0], sr

    def clone_voice(
        self,
        text: str | list[str],
        ref_audio: np.ndarray | tuple | str,
        ref_text: str,
    ) -> tuple[np.ndarray | list[np.ndarray], int]:
        """clone voice from reference audio."""
        self._load_model()

        # ensure ref_audio is a tuple (audio, sr) as required by qwen_tts
        if isinstance(ref_audio, (str, Path)):
            audio_data, audio_sr = sf.read(str(ref_audio))
            ref_audio = (audio_data, audio_sr)
        elif isinstance(ref_audio, np.ndarray):
            # assume default sample rate if raw array passed
            ref_audio = (ref_audio, SAMPLE_RATE)

        with self._get_attn_ctx():
            with torch.inference_mode():
                wavs, sr = self._model.generate_voice_clone(
                    text=text,
                    language=self.config.language,
                    ref_audio=ref_audio,
                    ref_text=ref_text,
                    non_streaming_mode=True,
                    do_sample=self.config.do_sample,
                    temperature=self.config.temperature,
                )

        if isinstance(text, str):
            return wavs[0], sr
        return wavs, sr

    def synthesize_long(self, text: str, instruct: str = "") -> tuple[np.ndarray, int]:
        """synthesize long text by chunking at sentence boundaries."""
        chunks = chunk_text(text, self.config.chunk_size)
        chunks = [c for c in chunks if c.strip()]

        if not chunks:
            return np.array([], dtype=np.float32), SAMPLE_RATE

        audio_chunks = []
        sample_rate = SAMPLE_RATE

        # process in batches
        for i in tqdm(
            range(0, len(chunks), self.config.batch_size),
            desc="synthesizing chunks",
            unit="batch",
            leave=False,
        ):
            batch_texts = chunks[i : i + self.config.batch_size]

            # synthesize batch
            batch_audio, sample_rate = self.synthesize(batch_texts, instruct)
            audio_chunks.extend(batch_audio)

        return concatenate_audio(audio_chunks, sample_rate, PARAGRAPH_PAUSE_MS), sample_rate


def chunk_text(text: str, max_size: int = MAX_CHUNK_SIZE) -> list[str]:
    """split text into chunks at sentence boundaries."""
    sentences = SENTENCE_ENDINGS.split(text)

    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        sentence_len = len(sentence)

        if current_length + sentence_len > max_size and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_length = 0

        current_chunk.append(sentence)
        current_length += sentence_len + 1

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def synthesize_chapters(
    workdir: Path,
    config: TTSConfig | None = None,
    chapters: list[int] | None = None,
    instruct: str = "",
    pooled: bool = False,
) -> None:
    """synthesize audio for chapters in workdir.

    if pooled=True, combines chunks from multiple chapters into larger batches
    for better GPU utilization on high-VRAM systems.
    """
    engine = TTSEngine(config)
    pending = list(iter_pending_chapters(workdir, chapters, skip_message="already synthesized"))

    if not pending:
        return

    if pooled and len(pending) > 1:
        _synthesize_pooled(engine, pending, instruct)
    else:
        for txt_path, wav_path in pending:
            _synthesize_chapter_resumable(engine, txt_path, wav_path, instruct)


def _synthesize_chapter_resumable(
    engine: TTSEngine,
    txt_path: Path,
    wav_path: Path,
    instruct: str = "",
) -> None:
    """synthesize a chapter with chunk-level progress tracking for resume."""
    text = txt_path.read_text()
    chunks = chunk_text(text, engine.config.chunk_size)
    chunks = [c for c in chunks if c.strip()]

    if not chunks:
        sf.write(str(wav_path), np.array([], dtype=np.float32), SAMPLE_RATE)
        return

    total_chunks = len(chunks)
    chunk_dir = get_chunk_dir(wav_path)

    # check for existing progress
    completed, saved_total = load_chunk_progress(chunk_dir)

    # restart if source changed
    if saved_total and saved_total != total_chunks:
        print(f"restarting {txt_path.name} (source changed)...")
        shutil.rmtree(chunk_dir, ignore_errors=True)
        completed = 0

    # check if already complete (just needs finalization)
    if completed >= total_chunks:
        print(f"finalizing {txt_path.name}...")
        finalize_chunks(chunk_dir, wav_path, total_chunks)
        print(f"  -> {wav_path.name}")
        return

    chunk_dir.mkdir(parents=True, exist_ok=True)

    if completed > 0:
        print(f"resuming {txt_path.name} at chunk {completed + 1}/{total_chunks}...")
    else:
        print(f"synthesizing {txt_path.name} ({total_chunks} chunks)...")

    # process remaining chunks in batches
    for i in tqdm(
        range(completed, total_chunks, engine.config.batch_size),
        desc="synthesizing",
        unit="batch",
        initial=completed // engine.config.batch_size,
        total=(total_chunks + engine.config.batch_size - 1) // engine.config.batch_size,
        leave=False,
    ):
        batch_end = min(i + engine.config.batch_size, total_chunks)
        batch_texts = chunks[i:batch_end]

        batch_audio, _ = engine.synthesize(batch_texts, instruct)

        # save each chunk immediately
        for j, audio in enumerate(batch_audio):
            save_chunk_audio(chunk_dir, i + j, audio, SAMPLE_RATE)

        # update progress after each batch
        save_chunk_progress(chunk_dir, batch_end, total_chunks)

    # finalize: concatenate all chunks into final wav
    finalize_chunks(chunk_dir, wav_path, total_chunks)
    print(f"  -> {wav_path.name}")


def _synthesize_pooled(
    engine: TTSEngine,
    pending: list[tuple[Path, Path]],
    instruct: str = "",
) -> None:
    """synthesize multiple chapters with pooled batching for high-VRAM systems."""
    from collections import defaultdict

    # collect all chunks with their source chapter
    all_chunks = []  # (chapter_idx, chunk_idx, text)
    chapter_chunk_counts = []

    for chapter_idx, (txt_path, _) in enumerate(pending):
        text = txt_path.read_text()
        chunks = chunk_text(text, engine.config.chunk_size)
        chunks = [c for c in chunks if c.strip()]
        chapter_chunk_counts.append(len(chunks))
        for chunk_idx, chunk in enumerate(chunks):
            all_chunks.append((chapter_idx, chunk_idx, chunk))

    if not all_chunks:
        return

    print(f"pooled synthesis: {len(pending)} chapters, {len(all_chunks)} total chunks")

    # synthesize all chunks in large batches
    audio_results = {}  # (chapter_idx, chunk_idx) -> audio
    chunks_completed = defaultdict(int)
    sample_rate = SAMPLE_RATE

    for i in tqdm(
        range(0, len(all_chunks), engine.config.batch_size),
        desc="synthesizing pooled",
        unit="batch",
    ):
        batch = all_chunks[i : i + engine.config.batch_size]
        batch_texts = [c[2] for c in batch]

        batch_audio, sample_rate = engine.synthesize(batch_texts, instruct)

        # store results and update progress
        touched_chapters = set()
        for j, (chapter_idx, chunk_idx, _) in enumerate(batch):
            audio_results[(chapter_idx, chunk_idx)] = batch_audio[j]
            chunks_completed[chapter_idx] += 1
            touched_chapters.add(chapter_idx)

        # check for completed chapters and write them immediately
        for chapter_idx in touched_chapters:
            if chunks_completed[chapter_idx] == chapter_chunk_counts[chapter_idx]:
                chunk_count = chapter_chunk_counts[chapter_idx]
                audio_list = [audio_results.pop((chapter_idx, i)) for i in range(chunk_count)]
                combined = concatenate_audio(audio_list, sample_rate, PARAGRAPH_PAUSE_MS)
                _, wav_path = pending[chapter_idx]
                sf.write(str(wav_path), combined, sample_rate)
                print(f"  -> {wav_path.name} ({chunk_count} chunks)")
