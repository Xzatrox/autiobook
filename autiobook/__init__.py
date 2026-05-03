"""autiobook - convert epub files to audiobooks using qwen3-tts."""

import os as _os
import shutil as _shutil

# expose all GPUs: device 0 = RX 7600M XT (TTS), device 1 = 780M iGPU (LLM)
# do not filter with HIP_VISIBLE_DEVICES so both are accessible
_os.environ.pop("HIP_VISIBLE_DEVICES", None)
_os.environ.pop("ROCR_VISIBLE_DEVICES", None)

# add winget-installed tool paths if not already on PATH
_WINGET_TOOLS = [
    r"C:\Users\Trox\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.1-full_build\bin",
    r"C:\Users\Trox\AppData\Local\Microsoft\WinGet\Packages\ChrisBagwell.SoX_Microsoft.Winget.Source_8wekyb3d8bbwe\sox-14.4.2",
]
for _p in _WINGET_TOOLS:
    if _os.path.isdir(_p) and _p not in _os.environ.get("PATH", ""):
        _os.environ["PATH"] = _p + _os.pathsep + _os.environ.get("PATH", "")

__version__ = "0.1.0"
