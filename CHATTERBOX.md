# Chatterbox backend (optional)

This repo ships a standalone Gradio demo for [Chatterbox](https://github.com/resemble-ai/chatterbox),
a TTS model with zero-shot voice cloning and a multilingual variant that
includes Portuguese. It is a third backend alongside Qwen3-TTS and Kokoro.

Chatterbox uses **fixed model checkpoints**, not a fixed voice list. With no
reference audio it uses the checkpoint's built-in voice; with a short reference
clip it clones that voice. The Qwen3-TTS and Kokoro demos are unchanged.

> Chatterbox is retired research upstream (August 2026). The code remains
> available under the MIT license, but it is no longer maintained or supported
> by Resemble AI.

## Why a separate environment?

`chatterbox-tts` 0.1.7 pins `torch==2.6.0` and `transformers==5.2.0`. The Qwen
stack in this repo uses `transformers==4.57.3` and the existing `.venv` has
`torch==2.2.2`. Mixing them breaks Qwen/Kokoro.

On **Intel macOS**, PyTorch stopped publishing x86_64 wheels after 2.2.2, so
`torch==2.6.0` cannot be installed from PyPI at all. The tested workaround is a
separate `.venv-chatterbox` with torch 2.2.2, transformers 4.46.3, and
`chatterbox-tts` installed with `--no-deps`. This is not an upstream-supported
combination, but it loads and generates Portuguese audio on this machine.

Do **not** install `chatterbox-tts` into `.venv`.

## Install (tested on Intel macOS)

```bash
uv venv .venv-chatterbox --python 3.12
uv pip install --python .venv-chatterbox/bin/python -r requirements-chatterbox.txt
uv pip install --python .venv-chatterbox/bin/python --no-deps chatterbox-tts==0.1.7
```

The requirements file pins `setuptools==75.8.0` because the Perth watermarker
imports `pkg_resources`; newer setuptools releases removed it.

On Apple Silicon, Linux, or Windows, use the official package in a separate
environment instead:

```bash
uv venv .venv-chatterbox --python 3.12
uv pip install --python .venv-chatterbox/bin/python chatterbox-tts==0.1.7
uv pip install --python .venv-chatterbox/bin/python gradio soundfile
```

## First run

The first model load downloads about 3.0 GB from
`ResembleAI/chatterbox` into `~/.cache/huggingface`. The multilingual tokenizer
also initializes `spacy-pkuseg`, which downloads a ~33 MB Chinese segmentation
model to `~/.pkuseg`.

The pkuseg downloader has a hard-coded 5-second read timeout. `chatterbox_demo.py`
patches it to 60 seconds before loading the model. If it still fails, pre-download
it manually and rerun:

```bash
mkdir -p ~/.pkuseg
curl -L --retry 5 -o ~/.pkuseg/spacy_ontonotes.zip \
  https://github.com/explosion/spacy-pkuseg/releases/download/v0.0.26/spacy_ontonotes.zip
unzip -o ~/.pkuseg/spacy_ontonotes.zip -d ~/.pkuseg/spacy_ontonotes
```

The pkuseg model is only needed for Chinese text, but the multilingual tokenizer
initializes it for every language. The English-only `--model english` checkpoint
uses a different set of weights (~3.2 GB) and is downloaded separately the first
time you select it.

## Run

```bash
.venv-chatterbox/bin/python chatterbox_demo.py --device cpu --threads 4
# Open http://127.0.0.1:7861
```

Portuguese is the default. The demo uses port 7861 so it can run alongside the
Kokoro demo (7860) and the Qwen demo (8000).

Useful flags:

```bash
.venv-chatterbox/bin/python chatterbox_demo.py \
  --model multilingual \
  --language pt \
  --device cpu --threads 4 \
  --max-chars 300 \
  --ip 127.0.0.1 --port 7861
```

| Flag | Default | Description |
| --- | --- | --- |
| `--model` | `multilingual` | `multilingual` (23 languages, including Portuguese) or `english` (English-only checkpoint). |
| `--language` | `pt` | Default language for the multilingual model. The UI has the full list. |
| `--device` | `auto` | `auto`, `cpu`, `cuda`, `cuda:0`, or `mps`. On this Intel Mac, use `cpu`. |
| `--threads` | `0` | PyTorch CPU threads. `4` is a good start on a 4-core CPU. |
| `--max-chars` | `300` | Text is split at sentence boundaries into chunks of at most this many characters. |
| `--warmup` | off | Run one short generation at startup (adds ~10-30 s on CPU). |
| `--ip`, `--port` | `127.0.0.1`, `7861` | Server bind address. |
| `--share` | off | Create a public Gradio link. |
| `--ssl-certfile`, `--ssl-keyfile` | unset | Optional TLS for a Gradio HTTPS server. |

The UI also exposes exaggeration, CFG weight, temperature, repetition penalty,
min-p, top-p, and chunk size under **Advanced**. The defaults match the
multilingual model's library defaults.

## Languages and voices

The multilingual model supports: Arabic (`ar`), Chinese (`zh`), Danish (`da`),
Dutch (`nl`), English (`en`), Finnish (`fi`), French (`fr`), German (`de`),
Greek (`el`), Hebrew (`he`), Hindi (`hi`), Italian (`it`), Japanese (`ja`),
Korean (`ko`), Malay (`ms`), Norwegian (`no`), Polish (`pl`), Portuguese (`pt`),
Russian (`ru`), Spanish (`es`), Swahili (`sw`), Swedish (`sv`), and Turkish
(`tr`).

There is no fixed voice list. Leave the reference audio empty to use the
built-in `conds.pt` voice, or upload a clean 6-20 second single-speaker clip to
clone a voice. The reference is prepared once per request and reused for every
text chunk.

Chatterbox exposes a single `pt` language code; it does not distinguish
Brazilian and European Portuguese. The built-in voice and the reference clip
determine the accent. For Brazilian Portuguese, use a Brazilian reference clip;
Kokoro's Portuguese voices are Brazilian by default.

The PyPI 0.1.7 release uses the multilingual **V2** checkpoint
(`t3_mtl23ls_v2.safetensors`). The newer V3 checkpoint is only in the upstream
Git repository and is not wired into this demo.

## Performance

Measured on the Intel i7-8559U used for this repo (4 threads, CPU, fp32):

- Model load: ~15-30 s after the first download.
- `"Olá, tudo bem?"`: 27.5 s for 2.24 s of audio (~12.3x real time).
- `"Olá."`: 33.9 s for 2.88 s of audio (~11.8x real time).
- English-only model: ~57 s first load, 24.5 s for 1.88 s of audio.
- Peak RSS: ~6.1 GB; peak memory footprint: ~4.6 GB.

This is usable for short phrases, not for long-form narration on CPU. Close
other memory-heavy applications before running. A CUDA GPU or Apple Silicon
with MPS will be much faster; `--device auto` selects one when available.

Long text is split into sentence-sized chunks and concatenated with 150 ms of
silence. Each chunk is generated serially, so the real-time factor is roughly
constant per second of audio.

## Notes and limitations

- Chatterbox output is watermarked by Perth. There is no toggle in this demo.
- The English-only model (`--model english`) uses a different checkpoint from the
  multilingual model. It still supports zero-shot voice cloning.
- The English Turbo/Nano checkpoints are not included. They are English-only and
  have a different API; the multilingual model is the better fit for Portuguese.
- The demo uses `gradio==6.17.3` in the Intel requirements file. The upstream
  package pins a different Gradio version, but the demo was tested with 6.17.3.
- `--model multilingual` loads the V2 checkpoint from the PyPI 0.1.7 package.
  Do not expect upstream support for the torch 2.2.2/transformers 4.46.3
  combination on Intel macOS.
- On Apple Silicon, `--device auto` may select MPS. If generation fails on an
  unsupported MPS op, rerun with `--device cpu`.

## Python API

```python
from chatterbox_demo import ChatterboxEngine

engine = ChatterboxEngine(model_name="multilingual", device="cpu", threads=4)
wav, sr = engine.synthesize("Olá! Este é um teste.", language="pt")
# wav is a mono float32 numpy array, sr is 24000
```

To clone a voice, pass `reference_path="speaker.wav"` to `synthesize`.
