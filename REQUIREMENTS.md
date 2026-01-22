# product requirements

## overview

autiobook converts epub files to audiobooks using qwen3-tts.

## functional requirements

- parse epub files and extract chapter content
- convert text to speech using qwen3-tts
- output one mp3 file per chapter
- include id3 metadata (title, album, artist, track number)
- compatible with "voice" audiobook player for android

## non-functional requirements

- support gpu acceleration for tts inference
- handle long chapters by chunking text at sentence boundaries
- sanitize filenames for cross-platform compatibility

## output format

```
Book_Title/
├── 01_Chapter_Title.mp3
├── 02_Chapter_Title.mp3
└── ...
```

## supported voices

qwen3-tts preset voices: Vivian, Ryan, Sunny, Aria, Bella, Nova, Echo, Finn, Atlas
