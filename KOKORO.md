# Kokoro-82M backend (optional)

This repo ships a standalone Gradio demo for [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M),
an 82M-parameter TTS model that is much lighter and faster on CPU than the
0.6B/1.7B Qwen3-TTS checkpoints.

Kokoro uses **fixed voices**. It does not support zero-shot voice cloning or
voice-design prompts. The Qwen3-TTS demo and fine-tuning code remain unchanged.

## Install

Kokoro requires Python 3.10–3.13. In the existing `.venv`:

```bash
uv pip install --python .venv/bin/python -r requirements-kokoro.txt
```

Or create a small standalone environment that does not need the Qwen stack:

```bash
uv venv .venv-kokoro --python 3.12
uv pip install --python .venv-kokoro/bin/python -r requirements-kokoro.txt
```

The first run downloads the model (~330 MB) and the selected voice from Hugging
Face into `~/.cache/huggingface`.

## Run

```bash
.venv/bin/python kokoro_demo.py
# Portuguese is the default; open http://127.0.0.1:7860
```

Useful flags:

```bash
.venv/bin/python kokoro_demo.py \
  --lang p --voice pf_dora \
  --device cpu --threads 4 \
  --ip 127.0.0.1 --port 7860
```

`--lang` accepts `p` (Brazilian Portuguese), `a`/`b` (US/UK English), `e`
(Spanish), `f` (French), `i` (Italian), and `h` (Hindi). The UI has a language
selector and a voice selector that updates with the language. Portuguese G2P
uses the espeak-ng build bundled by `espeakng-loader`; no Homebrew install is
needed.

On Apple Silicon, `--device auto` still selects CPU. `--device mps` is available
but some Kokoro ops may need `PYTORCH_ENABLE_MPS_FALLBACK=1`.

## Voices

| Language | Code | Voices |
| --- | --- | --- |
| Portuguese (BR) | `p` | `pf_dora`, `pm_alex`, `pm_santa` |
| English (US) | `a` | `af_heart`, `af_bella`, `af_nicole`, `af_aoede`, `af_kore`, `af_sarah`, `af_alloy`, `af_nova`, `af_sky`, `af_river`, `am_fenrir`, `am_michael`, `am_puck`, `am_echo`, `am_eric`, `am_liam`, `am_onyx`, `am_adam`, `am_santa` |
| English (UK) | `b` | `bf_emma`, `bf_isabella`, `bf_alice`, `bf_lily`, `bm_george`, `bm_fable`, `bm_lewis`, `bm_daniel` |
| Spanish | `e` | `ef_dora`, `em_alex`, `em_santa` |
| French | `f` | `ff_siwis` |
| Italian | `i` | `if_sara`, `im_nicola` |
| Hindi | `h` | `hf_alpha`, `hf_beta`, `hm_omega`, `hm_psi` |

## Performance

On the Intel i7-8559U used for this repo (4 threads, CPU, fp32):

- Model load + warm-up: ~20 s once.
- After warm-up: ~0.45x real-time factor (about 1.5 s to generate 3.5 s of audio).
- The first request for English also downloads spaCy's `en_core_web_sm` once.

Startup performs a warm-up inference so the first UI request does not pay the
lazy-initialization cost. Use `--threads 4` on 4-core CPUs; more threads did not
help in testing.

## Python API

```python
from kokoro_demo import KokoroEngine

engine = KokoroEngine(device="cpu", threads=4)
wav, sr = engine.synthesize(
    "Olá! Este é um teste do Kokoro.",
    lang="p",
    voice="pf_dora",
    speed=1.0,
)
```

`wav` is a mono `float32` NumPy array at 24 kHz.
