# autiobook

convert epub and fb2 files to audiobooks using qwen3-tts.

## requirements

- python 3.12
- ffmpeg
- sox
- uv (python package manager)
- gpu recommended (cuda, rocm, or apple mps)

## installation

```bash
# cuda gpu (default)
make build-cuda

# apple silicon (mps)
make build-mps

# amd rocm gpu - linux (gfx1151 / rdna4)
make build-rocm

# amd rocm gpu - windows (gfx110x / rdna3+, e.g. rx 7600, 7800, 7900, 9070)
make build-rocm-win
# or without make:
uv venv && uv sync --extra rocm-gfx110x

# cpu only
make build-cpu
```

### windows amd gpu notes

the `rocm-gfx110x` extra uses pre-built pytorch wheels from
[TheRock](https://github.com/scottt/rocm-TheRock/releases/tag/v6.5.0rc-pytorch-gfx110x)
with bundled ROCm 6.5. no separate HIP SDK or ZLUDA required.

if you have both a discrete and integrated AMD GPU, set:
```
setx HIP_VISIBLE_DEVICES 0
setx ROCR_VISIBLE_DEVICES 0
```
to target the discrete GPU (device 0).

## usage

### enter the venv

```bash
# linux/mac
source .venv/bin/activate

# windows
.venv\Scripts\activate

autiobook --help
```

### list chapters

```bash
autiobook chapters book.epub
autiobook chapters book.fb2
```

### full conversion (idempotent)

```bash
autiobook convert book.epub -o workdir/
autiobook convert book.fb2 -o workdir/
```

runs all phases, skipping already-completed steps.

### extract

extract chapter text from book file (epub or fb2) to workdir.

```
autiobook extract book.epub -o workdir/
autiobook extract book.fb2 -o workdir/
```

creates:
- `workdir/extract/metadata.json` - book metadata
- `workdir/extract/NN_Title.txt` - chapter text files
- `workdir/extract/state.json` - resumability state

### synthesize

convert text files to wav audio.

```
autiobook synthesize workdir/ -s Ryan
```

creates:
- `workdir/synthesize/NN_Title.wav` - audio files
- `workdir/synthesize/state.json` - resumability state

### export

convert wav files to mp3 with metadata.

```
autiobook export workdir/
```

creates:
- `workdir/export/NN_Title.mp3` - mp3 files with id3 tags
- `workdir/export/state.json` - resumability state


### dramatized conversion (llm)

generate a full cast performance using an openai-compatible llm and voice cloning.
book language is detected automatically from metadata — voices and audition lines
are generated in the correct language.

#### using a cloud llm (recommended)

free tier available on [NVIDIA NIM](https://build.nvidia.com):

```bash
autiobook dramatize book.epub -o workdir/ \
  --api-key nvapi-... \
  --api-base https://integrate.api.nvidia.com/v1 \
  --model openai/mistralai/mistral-nemotron \
  -c 1 -v
```

#### using a local llm (llama.cpp)

```bash
autiobook dramatize book.epub -o workdir/ \
  --llm-server-model models/Qwen3-8B-Q4_K_M.gguf \
  --api-key local \
  -v
```

download model:
```bash
hf download Qwen/Qwen3-8B-GGUF Qwen3-8B-Q4_K_M.gguf --local-dir models
```

#### using a local hf model (transformers)

runs on your GPU via ROCm/CUDA torch:

```bash
autiobook dramatize book.epub -o workdir/ \
  --llm-server-hf-model Qwen/Qwen3-4B \
  --api-key local \
  -v
```

#### step by step

```bash
# 1. extract text
autiobook extract book.epub -o workdir/

# 2. generate cast list (using llm)
autiobook cast workdir/ --api-key nvapi-... --api-base https://integrate.api.nvidia.com/v1 --model openai/mistralai/mistral-nemotron

# 3. generate voice auditions (review/edit characters.json first if needed)
autiobook audition workdir/

# 4. create dramatized script (using llm)
autiobook script workdir/ --api-key nvapi-...

# 5. validate script against source (optional)
autiobook validate workdir/

# 6. fix any issues found (optional)
autiobook fix workdir/ --api-key nvapi-...

# 7. perform the script (voice cloning)
autiobook perform workdir/

# 8. export to mp3
autiobook export workdir/
```

or run the full dramatization pipeline in one go:

```bash
autiobook dramatize book.epub -o workdir/ --api-key nvapi-... --api-base https://integrate.api.nvidia.com/v1 --model openai/mistralai/mistral-nemotron
```

### script validation and repair

after generating scripts, you can validate that all source text is covered
and detect any hallucinated content:

```bash
# check for both missing text and hallucinated segments
autiobook validate workdir/

# check only for missing text
autiobook validate workdir/ --missing

# check only for hallucinated segments
autiobook validate workdir/ --hallucinated
```

to fix issues found during validation:

```bash
# fill missing text and remove hallucinated segments
autiobook fix workdir/ --api-key sk-...

# only fill missing text (uses LLM with surrounding context)
autiobook fix workdir/ --missing --api-key sk-...

# only remove hallucinated segments (no LLM needed)
autiobook fix workdir/ --hallucinated

# control context amount for LLM (characters or paragraphs)
autiobook fix workdir/ --missing --context-chars 1000 --api-key sk-...
autiobook fix workdir/ --missing --context-paragraphs 3 --api-key sk-...
```

### options

- `-o, --output DIR` - output directory
- `-s, --speaker NAME` - tts voice (default: Ryan)
- `-c, --chapters RANGE` - chapter selection (e.g., 1-5, 3,7,10)
- `-v, --verbose` - verbose output
- `--api-key KEY` - llm api key
- `--api-base URL` - llm api base url (for openai-compatible endpoints)
- `--model NAME` - llm model name (with provider prefix, e.g. `openai/mistralai/mistral-nemotron`)
- `--llm-server-model PATH` - path to GGUF model to auto-start llama-server
- `--llm-server-hf-model ID` - huggingface model id to auto-start transformers server (uses local gpu)

### available voices (convert/synthesize)

Vivian, Ryan, Sunny, Aria, Bella, Nova, Echo, Finn, Atlas

## language support

book language is detected automatically from epub/fb2 metadata.
supported languages for dramatized conversion:
Russian, English, Chinese, Japanese, Korean, German, French, Portuguese, Spanish, Italian.

default cast voices (Narrator, Extra Female, Extra Male) use language-appropriate
descriptions and audition lines automatically.

## output

creates one mp3 file per chapter in `workdir/export/`:

```
workdir/export/
├── 01_Introduction.mp3
├── 02_Chapter_One.mp3
└── ...
```

compatible with the [Voice](https://github.com/PaulWoitaschek/Voice) audiobook player for android.

## workdir structure

Intermediate files are organized into subdirectories by command:

```
workdir/
├── extract/               # extracted text and metadata
│   ├── metadata.json
│   ├── cover.jpg
│   ├── NN_Title.txt
│   └── state.json
├── cast/                  # character list and analysis state
│   ├── characters.json
│   └── state.json
├── audition/              # character voice samples
│   ├── Character.wav
│   └── state.json
├── script/                # dramatized scripts (speaker segments)
│   ├── NN_Title.json
│   └── state.json
├── perform/               # dramatized audio performance
│   ├── NN_Title.wav
│   ├── segments/          # segment cache
│   └── state.json
├── synthesize/            # standard mono-voice audio
│   ├── NN_Title.wav
│   ├── segments/          # segment cache
│   └── state.json
└── export/                # final mp3 output
    ├── NN_Title.mp3
    └── state.json
```

Each command is fully resumable based on content hashes stored in `state.json`.
