# coding=utf-8
# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Standalone Gradio demo for Chatterbox TTS.

Chatterbox is a separate TTS backend from Qwen3-TTS and Kokoro. It supports
zero-shot voice cloning from a short reference clip and a multilingual model
that includes Portuguese. It needs its own virtual environment because the
upstream package pins torch/transformers versions that conflict with the Qwen
stack; see CHATTERBOX.md.
"""

import argparse
import inspect
import re
import threading
import time
from typing import Callable, List, Optional, Sequence, Tuple

import gradio as gr
import numpy as np
import torch


def _patch_pkuseg_download_timeout() -> None:
    """Give spacy-pkuseg's model download more than its hard-coded 5 seconds.

    The multilingual tokenizer initializes pkuseg even for non-Chinese text.
    Its downloader uses ``requests.get(..., timeout=5)``, which is too short on
    a cold GitHub release connection. The patch only affects the optional
    download; if spacy-pkuseg is unavailable the import is simply skipped.
    """
    try:
        import spacy_pkuseg.download as pkuseg_download
    except ImportError:
        return

    original = pkuseg_download.urlopen
    if getattr(original, "_chatterbox_timeout_patch", False):
        return

    def urlopen(url, *args, **kwargs):
        timeout = kwargs.get("timeout")
        if isinstance(timeout, (int, float)):
            kwargs["timeout"] = max(60, timeout)
        else:
            kwargs["timeout"] = 60
        return original(url, *args, **kwargs)

    urlopen._chatterbox_timeout_patch = True
    pkuseg_download.urlopen = urlopen


_patch_pkuseg_download_timeout()

try:
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS, SUPPORTED_LANGUAGES
    from chatterbox.tts import ChatterboxTTS
except ImportError as exc:  # pragma: no cover - depends on the environment
    raise SystemExit(
        "Chatterbox is not installed in this Python environment. "
        "Create and use .venv-chatterbox as described in CHATTERBOX.md; "
        "do not install chatterbox-tts into the Qwen/Kokoro .venv."
    ) from exc

try:
    import perth
except ImportError:
    perth = None
if perth is not None and getattr(perth, "PerthImplicitWatermarker", None) is None:
    raise SystemExit(
        "Chatterbox's Perth watermarker failed to load, usually because "
        "setuptools removed pkg_resources. Install setuptools==75.8.0 in the "
        "Chatterbox environment (requirements-chatterbox.txt pins it) and retry."
    )


DEFAULT_MODEL = "multilingual"
DEFAULT_LANGUAGE = "pt"
DEFAULT_MAX_CHARS = 300
DEFAULT_TEXT = "Olá! Este é um teste do Chatterbox em português."
DEFAULT_ENGLISH_TEXT = "Hello! This is a Chatterbox test."

LANGUAGE_CHOICES = [
    (f"{name} ({code})", code)
    for code, name in sorted(SUPPORTED_LANGUAGES.items(), key=lambda item: item[1])
]

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?…。！？])\s+")


def split_text(text: str, max_chars: int = DEFAULT_MAX_CHARS) -> List[str]:
    """Split text at sentence boundaries into chunks of at most max_chars."""
    text = " ".join((text or "").split())
    if not text:
        return []
    max_chars = max(1, int(max_chars))
    if len(text) <= max_chars:
        return [text]

    chunks: List[str] = []
    current = ""
    for piece in _SENTENCE_SPLIT.split(text):
        piece = piece.strip()
        if not piece:
            continue
        if len(piece) > max_chars:
            if current:
                chunks.append(current)
                current = ""
            for word in piece.split():
                candidate = f"{current} {word}".strip()
                if len(candidate) > max_chars:
                    if current:
                        chunks.append(current)
                    current = word
                else:
                    current = candidate
            continue
        candidate = f"{current} {piece}".strip()
        if len(candidate) > max_chars:
            if current:
                chunks.append(current)
            current = piece
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks


def _resolve_device(device: str) -> str:
    if device == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but not available.")
    if device == "mps" and not torch.backends.mps.is_available():
        raise SystemExit("MPS requested but not available.")
    return device


class ChatterboxEngine:
    """Loads one Chatterbox model and serializes generation requests."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str = "auto",
        threads: int = 0,
    ):
        if threads > 0:
            torch.set_num_threads(threads)
        if model_name not in {"multilingual", "english"}:
            raise ValueError(f"Unsupported Chatterbox model: {model_name!r}")

        self.model_name = model_name
        self.device = _resolve_device(device)
        if model_name == "multilingual":
            self.model = ChatterboxMultilingualTTS.from_pretrained(device=self.device)
        else:
            self.model = ChatterboxTTS.from_pretrained(device=self.device)

        self.sr = int(self.model.sr)
        self._lock = threading.Lock()
        # prepare_conditionals() replaces model.conds in place. Keep the
        # checkpoint's built-in voice so clearing the reference restores it.
        self._default_conds = getattr(self.model, "conds", None)
        self._generate_params = set(inspect.signature(self.model.generate).parameters)
        self._prepare_params = set(
            inspect.signature(self.model.prepare_conditionals).parameters
        )

    def _prepare_reference(self, reference_path: Optional[str], exaggeration: float) -> None:
        if not reference_path:
            if self._default_conds is not None:
                self.model.conds = self._default_conds
            return
        if isinstance(reference_path, (tuple, list)):
            reference_path = reference_path[0]
        if hasattr(reference_path, "path"):
            reference_path = reference_path.path
        kwargs = {}
        if "exaggeration" in self._prepare_params:
            kwargs["exaggeration"] = float(exaggeration)
        self.model.prepare_conditionals(str(reference_path), **kwargs)

    def _generate_chunk(self, text: str, language: str, kwargs: dict):
        if self.model_name == "multilingual":
            return self.model.generate(text, language_id=language, **kwargs)
        return self.model.generate(text, **kwargs)

    def synthesize(
        self,
        text: str,
        language: str = DEFAULT_LANGUAGE,
        reference_path: Optional[str] = None,
        exaggeration: float = 0.5,
        cfg_weight: float = 0.5,
        temperature: float = 0.8,
        repetition_penalty: Optional[float] = None,
        min_p: float = 0.05,
        top_p: float = 1.0,
        max_chars: int = DEFAULT_MAX_CHARS,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> Tuple[np.ndarray, int]:
        text = (text or "").strip()
        if not text:
            raise ValueError("Text is required.")
        if self.model_name == "multilingual":
            if language not in SUPPORTED_LANGUAGES:
                raise ValueError(f"Unsupported language code: {language}")
        else:
            language = "en"

        chunks = split_text(text, max_chars)
        if not chunks:
            raise ValueError("No speakable text was found.")

        kwargs = {
            "exaggeration": float(exaggeration),
            "cfg_weight": float(cfg_weight),
            "temperature": float(temperature),
            "min_p": float(min_p),
            "top_p": float(top_p),
        }
        if repetition_penalty is not None:
            kwargs["repetition_penalty"] = float(repetition_penalty)
        kwargs = {key: value for key, value in kwargs.items() if key in self._generate_params}

        audios: List[np.ndarray] = []
        with self._lock, torch.inference_mode():
            self._prepare_reference(reference_path, exaggeration)
            for index, chunk in enumerate(chunks):
                if progress_callback is not None:
                    progress_callback(index, len(chunks))
                wav = self._generate_chunk(chunk, language, kwargs)
                audio = wav.detach().cpu().numpy().astype(np.float32).squeeze()
                if audio.ndim != 1:
                    audio = audio.reshape(-1)
                audios.append(audio)
            if progress_callback is not None:
                progress_callback(len(chunks), len(chunks))

        if not audios:
            raise ValueError("Chatterbox produced no audio for this text.")

        if len(audios) == 1:
            wav = audios[0]
        else:
            gap = np.zeros(int(0.15 * self.sr), dtype=np.float32)
            parts: List[np.ndarray] = []
            for index, audio in enumerate(audios):
                if index:
                    parts.append(gap)
                parts.append(audio)
            wav = np.concatenate(parts)

        return np.ascontiguousarray(np.clip(wav, -1.0, 1.0), dtype=np.float32), self.sr


def build_demo(
    engine: ChatterboxEngine,
    default_language: str = DEFAULT_LANGUAGE,
    default_max_chars: int = DEFAULT_MAX_CHARS,
) -> gr.Blocks:
    multilingual = engine.model_name == "multilingual"
    if not multilingual:
        default_language = "en"
    language_choices = LANGUAGE_CHOICES if multilingual else [("English (en)", "en")]
    default_text = (
        DEFAULT_TEXT
        if multilingual and default_language == "pt"
        else DEFAULT_ENGLISH_TEXT
    )
    default_rep = 2.0 if multilingual else 1.2

    with gr.Blocks(title="Chatterbox TTS") as demo:
        gr.Markdown(
            """
# Chatterbox TTS
Zero-shot voice cloning with an optional multilingual model. Leave the reference
audio empty to use the built-in voice, or upload a clean 6-20 second clip to
clone a voice. Portuguese is available as `pt` in the multilingual model.
CPU generation is slow; see `CHATTERBOX.md` for the tested timings.
"""
        )
        with gr.Row():
            with gr.Column(scale=2):
                text_in = gr.Textbox(
                    label="Text",
                    lines=5,
                    value=default_text,
                    placeholder="Enter text to synthesize.",
                )
                language_in = gr.Dropdown(
                    label="Language",
                    choices=language_choices,
                    value=default_language,
                    interactive=multilingual,
                )
                reference_in = gr.Audio(
                    label="Reference voice (optional)",
                    sources=["upload"],
                    type="filepath",
                )
                gr.Markdown(
                    "Upload a clean, single-speaker clip (6-20 s is a good range). "
                    "With no reference, Chatterbox uses its built-in voice."
                )
                with gr.Accordion("Advanced", open=False):
                    exaggeration_in = gr.Slider(
                        label="Exaggeration",
                        minimum=0.0,
                        maximum=1.5,
                        value=0.5,
                        step=0.05,
                    )
                    cfg_in = gr.Slider(
                        label="CFG weight",
                        minimum=0.0,
                        maximum=1.0,
                        value=0.5,
                        step=0.05,
                    )
                    temperature_in = gr.Slider(
                        label="Temperature",
                        minimum=0.1,
                        maximum=1.5,
                        value=0.8,
                        step=0.05,
                    )
                    repetition_in = gr.Slider(
                        label="Repetition penalty",
                        minimum=0.5,
                        maximum=3.0,
                        value=default_rep,
                        step=0.1,
                    )
                    min_p_in = gr.Slider(
                        label="Min-p",
                        minimum=0.0,
                        maximum=0.5,
                        value=0.05,
                        step=0.01,
                    )
                    top_p_in = gr.Slider(
                        label="Top-p",
                        minimum=0.1,
                        maximum=1.0,
                        value=1.0,
                        step=0.05,
                    )
                    max_chars_in = gr.Slider(
                        label="Max characters per chunk",
                        minimum=80,
                        maximum=1000,
                        value=default_max_chars,
                        step=20,
                    )
                btn = gr.Button("Generate audio", variant="primary")
            with gr.Column(scale=3):
                audio_out = gr.Audio(label="Generated audio", type="numpy")
                status = gr.Textbox(label="Status", lines=4, interactive=False)

        gr.ClearButton(
            [text_in, reference_in, audio_out, status],
            value="Clear",
        )

        examples = (
            [
                ["Olá! Este é um teste do Chatterbox em português.", "pt"],
                ["Hello! This is a Chatterbox test.", "en"],
                ["Hola, esta es una prueba de Chatterbox.", "es"],
            ]
            if multilingual
            else [["Hello! This is a Chatterbox test.", "en"]]
        )
        gr.Examples(examples=examples, inputs=[text_in, language_in], label="Examples")

        def run(
            text: str,
            language: str,
            reference_path: Optional[str],
            exaggeration: float,
            cfg_weight: float,
            temperature: float,
            repetition_penalty: float,
            min_p: float,
            top_p: float,
            max_chars: float,
            progress=gr.Progress(),
        ):
            chunks = split_text(text, int(max_chars))

            def on_progress(index: int, total: int) -> None:
                if index >= total:
                    progress(1.0, desc="Finalizing audio")
                else:
                    progress(index / total, desc=f"Generating chunk {index + 1}/{total}")

            try:
                t0 = time.perf_counter()
                wav, sr = engine.synthesize(
                    text,
                    language=language,
                    reference_path=reference_path,
                    exaggeration=exaggeration,
                    cfg_weight=cfg_weight,
                    temperature=temperature,
                    repetition_penalty=repetition_penalty,
                    min_p=min_p,
                    top_p=top_p,
                    max_chars=int(max_chars),
                    progress_callback=on_progress,
                )
                elapsed = time.perf_counter() - t0
                duration = len(wav) / sr
                rtf = elapsed / duration if duration else 0.0
                voice = "reference voice" if reference_path else "built-in voice"
                return (sr, wav), (
                    f"OK - {duration:.2f}s of audio in {elapsed:.1f}s "
                    f"({rtf:.1f}x real time), {len(chunks)} chunk(s), {voice}, "
                    f"language={language}."
                )
            except Exception as exc:
                return None, f"{type(exc).__name__}: {exc}"

        inputs = [
            text_in,
            language_in,
            reference_in,
            exaggeration_in,
            cfg_in,
            temperature_in,
            repetition_in,
            min_p_in,
            top_p_in,
            max_chars_in,
        ]
        btn.click(run, inputs=inputs, outputs=[audio_out, status])
        text_in.submit(run, inputs=inputs, outputs=[audio_out, status])

        gr.Markdown(
            "Model: `ResembleAI/chatterbox`. Output is watermarked by Perth. "
            "Chatterbox is MIT-licensed; see `CHATTERBOX.md` for details."
        )

    return demo


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="chatterbox-demo",
        description="Gradio demo for Chatterbox TTS (multilingual and English models).",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        choices=["multilingual", "english"],
        help="Chatterbox model family (default: multilingual).",
    )
    parser.add_argument(
        "--language",
        default=DEFAULT_LANGUAGE,
        choices=sorted(SUPPORTED_LANGUAGES),
        help="Default language for the multilingual model.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda", "cuda:0", "mps"],
        help="Inference device (default: auto; CUDA/MPS if available, otherwise CPU).",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=0,
        help="PyTorch CPU threads (0 = library default; 4 is a good start on 4 cores).",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=DEFAULT_MAX_CHARS,
        help="Maximum characters per generation chunk.",
    )
    parser.add_argument("--ip", default="127.0.0.1", help="Server bind IP.")
    parser.add_argument("--port", type=int, default=7861, help="Server port.")
    parser.add_argument("--share", action="store_true", help="Create a public Gradio link.")
    parser.add_argument(
        "--warmup",
        action="store_true",
        help="Run one short generation at startup (adds ~10-30 s on CPU).",
    )
    parser.add_argument("--ssl-certfile", default=None, help="Optional TLS certificate file.")
    parser.add_argument("--ssl-keyfile", default=None, help="Optional TLS key file.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.max_chars < 1:
        raise SystemExit("--max-chars must be positive.")

    language = args.language if args.model == "multilingual" else "en"
    print(f"Loading Chatterbox ({args.model}) ...", flush=True)
    if args.model == "multilingual":
        print(
            "The first run downloads ~3 GB from Hugging Face and a small pkuseg "
            "model; see CHATTERBOX.md if the pkuseg download times out.",
            flush=True,
        )
    t0 = time.perf_counter()
    engine = ChatterboxEngine(
        model_name=args.model,
        device=args.device,
        threads=args.threads,
    )
    print(
        f"Loaded in {time.perf_counter() - t0:.1f}s on {engine.device}; "
        f"sample rate {engine.sr} Hz.",
        flush=True,
    )

    if args.warmup:
        warmup_text = (
            "Olá, este é um aquecimento do Chatterbox."
            if language == "pt"
            else "Hello, this is a Chatterbox warm-up."
        )
        print("Running warm-up ...", flush=True)
        t1 = time.perf_counter()
        engine.synthesize(warmup_text, language=language, max_chars=200)
        print(f"Warm-up finished in {time.perf_counter() - t1:.1f}s.", flush=True)

    demo = build_demo(engine, language, args.max_chars)
    print(f"Open http://{args.ip}:{args.port}", flush=True)
    demo.queue(default_concurrency_limit=1).launch(
        server_name=args.ip,
        server_port=args.port,
        share=args.share,
        show_error=True,
        ssl_certfile=args.ssl_certfile,
        ssl_keyfile=args.ssl_keyfile,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
