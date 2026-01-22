"""shared constants and configuration."""

import re

# epub parsing
CONTENT_TAGS = ["p", "div", "h1", "h2", "h3", "h4", "h5", "h6", "li", "td", "th"]
SKIP_TAGS = ["script", "style", "meta", "head", "link", "noscript", "nav", "header", "footer"]
MIN_CHAPTER_WORDS = 50

# tts settings
DEFAULT_MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
VOICE_DESIGN_MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
BASE_MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
DEFAULT_SPEAKER = "Ryan"
MAX_CHUNK_SIZE = 500  # balance between coherence and decode speed
SAMPLE_RATE = 24000

# llm settings
DEFAULT_LLM_MODEL = "gpt-4o"
LLM_MAX_RETRIES = 3
LLM_RETRY_DELAY = 1.0  # initial delay in seconds, doubles on each retry

# audio processing
PARAGRAPH_PAUSE_MS = 500
CHAPTER_PAUSE_MS = 1000

# mp3 export
DEFAULT_BITRATE = "192k"
UNSAFE_FILENAME_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')

# file extensions
TXT_EXT = ".txt"
WAV_EXT = ".wav"
MP3_EXT = ".mp3"
METADATA_FILE = "metadata.json"
CAST_FILE = "cast.json"
SCRIPT_EXT = ".json"
COVER_FILE = "cover.jpg"
CHUNKS_DIR = "chunks"
CHUNK_PROGRESS_FILE = "progress.json"

# generic extra voices for minor characters
EXTRA_FEMALE = "Extra Female"
EXTRA_MALE = "Extra Male"
EXTRA_FEMALE_DESC = "generic female voice, neutral tone, middle-aged, clear enunciation"
EXTRA_MALE_DESC = "generic male voice, neutral tone, middle-aged, clear enunciation"
EXTRA_FEMALE_LINE = "I have a small part in this story."
EXTRA_MALE_LINE = "I have a small part in this story."
