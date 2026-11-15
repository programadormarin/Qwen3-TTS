# AGENTS.md

Qwen3-TTS: Python package for Qwen3-TTS inference (CustomVoice / VoiceDesign / Base voice clone), a Gradio demo, optional Kokoro-82M and Chatterbox demos, and standalone fine-tuning scripts.

## Environment and commands

- Python 3.12 is the tested version (see `README.md`); `pyproject.toml` allows >=3.9.
- This checkout has an editable install in `.venv`: prefer `.venv/bin/python` and `.venv/bin/qwen-tts-demo`. Otherwise install with `pip install -e .`.
- `transformers==4.57.3` is an exact pin. The modeling code relies on Transformers internals (custom AutoModel classes, `fix_mistral_regex=True`); do not upgrade it without rechecking model loading.
- Launch the demo with a repo id or local checkpoint:
  `qwen-tts-demo Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice --device cuda:0 --port 8000`
- There is no pytest suite, lint config, or test CI. `examples/test_*.py` are manual GPU smoke scripts that download weights and write WAVs; run one explicitly, e.g. `python examples/test_model_12hz_custom_voice.py`.
- Quick import check: `python -c "from qwen_tts import Qwen3TTSModel, Qwen3TTSTokenizer"`.
- Optional Kokoro backend: `uv pip install --python .venv/bin/python -r requirements-kokoro.txt`, then `.venv/bin/python kokoro_demo.py` (Portuguese default, port 7860). Kokoro needs Python 3.10-3.13.
- Optional Chatterbox backend: use a separate `.venv-chatterbox`, never `.venv`:
  `uv venv .venv-chatterbox --python 3.12`,
  `uv pip install --python .venv-chatterbox/bin/python -r requirements-chatterbox.txt`,
  `uv pip install --python .venv-chatterbox/bin/python --no-deps chatterbox-tts==0.1.7`,
  then `.venv-chatterbox/bin/python chatterbox_demo.py --device cpu --threads 4` (Portuguese default, port 7861).

## Layout

- `qwen_tts/core/` — real Hugging Face Transformers implementation (`Qwen3TTSConfig`, `Qwen3TTSForConditionalGeneration`, processor, 12 Hz and 25 Hz tokenizers). Not a thin wrapper.
- `qwen_tts/inference/` — public wrappers `Qwen3TTSModel` and `Qwen3TTSTokenizer`.
- `qwen_tts/cli/demo.py` — Gradio app exposed as `qwen-tts-demo`.
- `kokoro_demo.py` — standalone Gradio demo for Kokoro-82M; `KOKORO.md` and `requirements-kokoro.txt` document/install it. It does not import `qwen_tts`.
- `chatterbox_demo.py` — standalone Gradio demo for Chatterbox (multilingual and English models); `CHATTERBOX.md` and `requirements-chatterbox.txt` document/install it. It does not import `qwen_tts`.
- `finetuning/` and `examples/` — runnable from the repo only; both are pruned from the built package.

## Behavior and gotchas

- `Qwen3TTSModel.from_pretrained` requires `speech_tokenizer/` and `generation_config.json` inside the checkpoint directory/repo; it auto-downloads them. A copied/fine-tuned checkpoint missing either file fails to load.
- Generation APIs return `(list[np.ndarray], sample_rate)`. Sampling defaults come from the checkpoint's `generation_config.json`, then hardcoded fallbacks (`max_new_tokens=2048`, etc.). `non_streaming_mode` only controls whether text is fed in chunks to simulate streaming input; there is no true streaming-generation API.
- Two tokenizer generations coexist: 25 Hz V1 is legacy, 12 Hz V2 is what released models and fine-tuning use. `Qwen3TTSTokenizer` dispatches on checkpoint config.
- FlashAttention-2 is the default in the examples, CLI, and fine-tuning, but it is CUDA-only and requires fp16/bf16. On CPU/Mac pass `--no-flash-attn` to the demo; `examples/` and `finetuning/sft_12hz.py` hardcode it, so they need edits.
- CPU inference must use `--dtype float32`. On x86 CPUs without AVX512-BF16 (including this repo's Intel Mac), bf16/fp16 matmul and conv are roughly 50-300x slower than fp32; the demo's bfloat16 default is only appropriate for CUDA.
- The checkpoint's `generation_config.json` sets `max_new_tokens: 8192` (~11 minutes of audio at 12 Hz), and the wrapper uses it unless overridden. Always pass `--max-new-tokens` (e.g. 256-600) for demo/CPU use, or requests appear to hang.
- Use `--concurrency 1` for the demo on CPU. The default is 16, and parallel generations thrash the shared model and memory.
- The demo's Base (voice clone) mode uses the browser mic and generally needs HTTPS via `--ssl-certfile` and `--ssl-keyfile`.
- Loading any model downloads multi-GB weights from Hugging Face. For offline runs, pass a local path or `local_files_only=True`. Avoid running examples casually.
- `finetuning/sft_12hz.py` saves checkpoints with `shutil.copytree(--init_model_path, ...)`, so pass a local model directory, not a HuggingFace repo id (even though the flag's default is one). It only supports single-speaker 12Hz Base models.
- Fine-tuning is two steps: `python prepare_data.py ...` (adds `audio_codes`) then `python sft_12hz.py ...`. Run both from `finetuning/` because `sft_12hz.py` does `from dataset import TTSDataset`, which depends on the working directory.
- Fine-tuned checkpoints are rewritten as `custom_voice` models and register the speaker under `--speaker_name`; use `generate_custom_voice(speaker=<name>)` at inference, not `generate_voice_clone`.
- Keep `ref_audio` identical across training samples; the finetuning README notes this improves speaker consistency.
- Kokoro uses fixed voices (`pf_dora`, `af_heart`, ...), not zero-shot cloning or voice design. Keep it out of the `qwen_tts` package and the `qwen-tts-demo` entrypoint; `kokoro_demo.py` is the integration point.
- Kokoro's first run downloads ~330 MB; the demo warms up at startup. English also downloads spaCy's `en_core_web_sm` once. On 4-core CPUs `--threads 4` is optimal; after warm-up expect ~0.45x real time for Portuguese.
- Install Kokoro from `requirements-kokoro.txt` (or a separate venv), not as a `pyproject.toml` extra. Re-resolving the full project pulls numpy 2.x, which breaks the numpy 1.26 / torch 2.2.2 / Qwen pin.
- Chatterbox 0.1.7 pins `torch==2.6.0` / `transformers==5.2.0`; PyTorch has no macOS x86_64 wheel after 2.2.2. The tested Intel workaround is `.venv-chatterbox` with torch 2.2.2, transformers 4.46.3, and `chatterbox-tts --no-deps`. Do not install it into `.venv`; do not expect upstream support for this combination.
- Chatterbox's Perth watermarker imports `pkg_resources`; `requirements-chatterbox.txt` pins `setuptools==75.8.0` because newer setuptools releases removed it.
- The multilingual Chatterbox tokenizer initializes `spacy-pkuseg` even for Portuguese, downloading a ~33 MB model to `~/.pkuseg`. Its built-in downloader has a 5-second read timeout; `chatterbox_demo.py` patches it to 60 seconds. If it fails, pre-download `spacy_ontonotes.zip` as shown in `CHATTERBOX.md`.
- Chatterbox's first multilingual load downloads about 3.0 GB. On this Intel CPU, generation is roughly 12x real time (`~28 s` for `~2.2 s` of audio) and peak RSS is about 6.1 GB. Use short phrases; long text is chunked automatically.
- Chatterbox has no fixed voice list: with no reference it uses the checkpoint's `conds.pt` built-in voice; with a reference clip it clones that voice. The PyPI 0.1.7 release uses the multilingual V2 checkpoint; V3 is only in upstream Git.
- `chatterbox_demo.py` uses port 7861, `--model english` for the English-only checkpoint, and `--model multilingual` (default) for the 23-language model. `requirements-chatterbox.txt` targets Intel macOS; other platforms should install the official package in a separate venv.

## Conventions

- New files use the Apache-2.0 header. Some `core/` files retain upstream Hugging Face/Qwen or Meta headers (e.g. `tokenizer_25hz/vq/core_vq.py`); preserve existing headers. Comments and user-facing strings are English.
- Add public API to `qwen_tts/inference/` and export it from `qwen_tts/__init__.py` rather than requiring callers to use `qwen_tts/core/` directly.
