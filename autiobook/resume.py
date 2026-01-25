"""resumability utilities."""

import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, TypeVar, cast

from tqdm import tqdm  # type: ignore

from .config import STATE_FILE


def get_command_dir(workdir: Path, command: str) -> Path:
    """get and create directory for a specific command."""
    path = workdir / command
    path.mkdir(parents=True, exist_ok=True)
    return path


def list_chapters(
    metadata: Dict[str, Any],
    source_dir: Path,
    target_dir: Path,
    chapters_filter: list[int] | None = None,
    source_ext: str = ".txt",
    target_ext: str = ".wav",
) -> List[tuple[int, Path, Path]]:
    """list (index, source, target) for chapters based on metadata."""
    results = []
    for c in metadata.get("chapters", []):
        idx = c["index"]
        if chapters_filter and idx not in chapters_filter:
            continue

        base = c.get("filename_base")
        if not base:
            continue

        source_path = source_dir / (base + source_ext)
        target_path = target_dir / (base + target_ext)

        if source_path.exists():
            results.append((idx, source_path, target_path))

    return results


def compute_hash(obj: Any) -> str:
    """compute stable sha256 hash of a json-serializable object."""
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def load_state(path: Path) -> Dict[str, Any]:
    """load json state file, returning empty dict if missing/invalid."""
    if not path.exists():
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            return cast(Dict[str, Any], json.load(f))
    except Exception:
        return {}


def save_state(path: Path, data: Dict[str, Any]):
    """save json state file."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


class ResumeManager:
    """helper for managing granular resume state (e.g. for chunks/segments)."""

    def __init__(self, state_path: Path, force: bool = False):
        self.state_path = state_path
        self.state = load_state(state_path)
        self.dirty = False
        self.force = force

    @classmethod
    def for_command(
        cls, workdir: Path, command: str, force: bool = False
    ) -> "ResumeManager":
        """create a resume manager for a specific command."""
        state_path = get_command_dir(workdir, command) / STATE_FILE
        return cls(state_path, force=force)

    def is_fresh(self, key: str, current_hash: str) -> bool:
        """check if key exists and matches current hash."""
        if self.force:
            return False
        return self.state.get(str(key)) == current_hash

    def update(self, key: str, current_hash: str):
        """update state for key."""
        self.state[str(key)] = current_hash
        self.dirty = True

    def save(self):
        """save state to disk if modified."""
        if self.dirty:
            save_state(self.state_path, self.state)
            self.dirty = False


T = TypeVar("T")
R = TypeVar("R")


def batch_process_with_resume(
    items: List[T],
    resume_manager: ResumeManager,
    process_fn: Callable[[List[T]], List[R]],
    save_fn: Callable[[T, R], None],
    get_key_fn: Callable[[T], str],
    get_hash_fn: Callable[[T], str],
    batch_size: int = 1,
    desc: str = "processing",
    force: bool = False,
) -> None:
    """generic resume-aware batch processing loop.

    args:
        items: list of work items.
        resume_manager: ResumeManager instance.
        process_fn: function taking list of items and returning list of results.
        save_fn: function taking (item, result) and saving/updating state (excluding resume).
        get_key_fn: function extracting unique key from item.
        get_hash_fn: function extracting hash from item.
        batch_size: size of batches for process_fn.
        desc: description for progress bar.
        force: whether to force processing even if up to date.
    """
    # 1. filter items
    work_items = []
    for item in items:
        key = get_key_fn(item)
        current_hash = get_hash_fn(item)
        if force or not resume_manager.is_fresh(key, current_hash):
            work_items.append(item)

    if not work_items:
        return

    # 2. process batches
    for i in tqdm(
        range(0, len(work_items), batch_size), desc=desc, unit="batch", leave=False
    ):
        batch = work_items[i : i + batch_size]
        try:
            results = process_fn(batch)
            if len(results) != len(batch):
                raise RuntimeError(
                    f"process_fn returned {len(results)} results for {len(batch)} items"
                )

            for item, result in zip(batch, results):
                save_fn(item, result)
                resume_manager.update(get_key_fn(item), get_hash_fn(item))

            resume_manager.save()
        except Exception as e:
            print(f"batch failed: {e}")
            raise e
