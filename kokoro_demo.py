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
"""Standalone Gradio demo for Kokoro-82M.

Kokoro is a small (82M parameter), CPU-friendly TTS model with fixed voices.
It is an alternative to the Qwen3-TTS demo in this repository: it does not
support voice cloning or voice-design prompts.
"""

import argparse
import threading
import time
from typing import Dict, List, Tuple

import gradio as gr
import numpy as np
import torch
from kokoro import KModel, KPipeline

MODEL_ID = "hexgrad/Kokoro-82M"
SAMPLE_RATE = 24000

# Kokoro language codes: https://huggingface.co/hexgrad/Kokoro-82M/blob/main/VOICES.md
LANGUAGE_CHOICES = [
    ("Português (Brasil)", "p"),
    ("English (US)", "a"),
    ("English (UK)", "b"),
    ("Español", "e"),
    ("Français", "f"),
    ("Italiano", "i"),
    ("हिन्दी", "h"),
]

VOICE_CHOICES: Dict[str, List[Tuple[str, str]]] = {
    "p": [
        ("Dora (female)", "pf_dora"),
        ("Alex (male)", "pm_alex"),
        ("Santa (male)", "pm_santa"),
    ],
    "a": [
        ("Heart (female)", "af_heart"),
        ("Bella (female)", "af_bella"),
        ("Nicole (female)", "af_nicole"),
        ("Aoede (female)", "af_aoede"),
        ("Kore (female)", "af_kore"),
        ("Sarah (female)", "af_sarah"),
        ("Alloy (female)", "af_alloy"),
        ("Nova (female)", "af_nova"),
        ("Sky (female)", "af_sky"),
        ("River (female)", "af_river"),
        ("Fenrir (male)", "am_fenrir"),
        ("Michael (male)", "am_michael"),
        ("Puck (male)", "am_puck"),
        ("Echo (male)", "am_echo"),
        ("Eric (male)", "am_eric"),
        ("Liam (male)", "am_liam"),
        ("Onyx (male)", "am_onyx"),
        ("Adam (male)", "am_adam"),
        ("Santa (male)", "am_santa"),
    ],
    "b": [
        ("Emma (female)", "bf_emma"),
        ("Isabella (female)", "bf_isabella"),
        ("Alice (female)", "bf_alice"),
        ("Lily (female)", "bf_lily"),
        ("George (male)", "bm_george"),
        ("Fable (male)", "bm_fable"),
        ("Lewis (male)", "bm_lewis"),
        ("Daniel (male)", "bm_daniel"),
    ],
    "e": [
        ("Dora (female)", "ef_dora"),
        ("Alex (male)", "em_alex"),
        ("Santa (male)", "em_santa"),
    ],
    "f": [("Siwis (female)", "ff_siwis")],
    "i": [("Sara (female)", "if_sara"), ("Nicola (male)", "im_nicola")],
    "h": [
        ("Alpha (female)", "hf_alpha"),
        ("Beta (female)", "hf_beta"),
        ("Omega (male)", "hm_omega"),
        ("Psi (male)", "hm_psi"),
    ],
}

VOICE_CODES = {code for choices in VOICE_CHOICES.values() for _, code in choices}

WARMUP_TEXTS = {
    "p": "Olá, este é um aquecimento do modelo.",
    "a": "Hello, this is a model warm-up.",
    "b": "Hello, this is a model warm-up.",
    "e": "Hola, esta es una prueba del modelo.",
    "f": "Bonjour, ceci est un test du modèle.",
    "i": "Ciao, questo è un test del modello.",
    "h": "नमस्ते, यह मॉडल का परीक्षण है।",
}


def _resolve_device(device: str) -> str:
    if device == "auto":
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but not available.")
    if device == "mps" and not torch.backends.mps.is_available():
        raise SystemExit("MPS requested but not available.")
    return device


class KokoroEngine:
    """Reuses one KModel across language-specific KPipelines."""

    def __init__(self, device: str = "auto", threads: int = 0):
        if threads > 0:
            torch.set_num_threads(threads)
        self.device = _resolve_device(device)
        self.model = KModel(repo_id=MODEL_ID).to(self.device).eval()
        self._pipelines: Dict[str, KPipeline] = {}
        self._lock = threading.Lock()

    def pipeline(self, lang: str) -> KPipeline:
        if lang not in self._pipelines:
            self._pipelines[lang] = KPipeline(
                lang_code=lang,
                repo_id=MODEL_ID,
                model=self.model,
            )
        return self._pipelines[lang]

    def synthesize(
        self,
        text: str,
        lang: str,
        voice: str,
        speed: float = 1.0,
    ) -> Tuple[np.ndarray, int]:
        text = (text or "").strip()
        if not text:
            raise ValueError("Text is required.")
        if lang not in VOICE_CHOICES:
            raise ValueError(f"Unsupported language code: {lang}")
        if voice not in {code for _, code in VOICE_CHOICES[lang]}:
            raise ValueError(
                f"Voice {voice!r} does not belong to language {lang!r}. "
                f"Available: {[code for _, code in VOICE_CHOICES[lang]]}"
            )

        pipe = self.pipeline(lang)
        chunks: List[np.ndarray] = []
        with self._lock, torch.inference_mode():
            for result in pipe(text, voice=voice, speed=float(speed)):
                audio = result.audio
                if audio is None:
                    continue
                chunks.append(audio.detach().cpu().numpy().astype(np.float32))

        if not chunks:
            raise ValueError("Kokoro produced no audio for this text.")

        if len(chunks) == 1:
            wav = chunks[0]
        else:
            # Kokoro synthesizes chunks independently; add a short pause so the
            # joins do not click or sound rushed.
            gap = np.zeros(int(0.12 * SAMPLE_RATE), dtype=np.float32)
            parts: List[np.ndarray] = []
            for index, chunk in enumerate(chunks):
                if index:
                    parts.append(gap)
                parts.append(chunk)
            wav = np.concatenate(parts)
        return np.ascontiguousarray(wav, dtype=np.float32), SAMPLE_RATE


def build_demo(
    engine: KokoroEngine,
    default_lang: str,
    default_voice: str,
    default_speed: float = 1.0,
) -> gr.Blocks:
    default_voices = VOICE_CHOICES[default_lang]

    with gr.Blocks(title="Kokoro-82M TTS") as demo:
        gr.Markdown(
            """
# Kokoro-82M TTS
Lightweight (82M parameters) and fast on CPU. Kokoro uses **fixed voices**; it
does not clone voices or follow voice-design prompts. The first run downloads
~330 MB from Hugging Face.
"""
        )
        with gr.Row():
            with gr.Column(scale=2):
                text_in = gr.Textbox(
                    label="Text",
                    lines=5,
                    value="Olá! Este é um teste do Kokoro, um modelo de voz leve e rápido.",
                    placeholder="Enter text to synthesize.",
                )
                with gr.Row():
                    lang_in = gr.Dropdown(
                        label="Language",
                        choices=LANGUAGE_CHOICES,
                        value=default_lang,
                        interactive=True,
                    )
                    voice_in = gr.Dropdown(
                        label="Voice",
                        choices=default_voices,
                        value=default_voice,
                        interactive=True,
                    )
                speed_in = gr.Slider(
                    label="Speed",
                    minimum=0.5,
                    maximum=2.0,
                    value=default_speed,
                    step=0.05,
                )
                btn = gr.Button("Generate audio", variant="primary")
            with gr.Column(scale=3):
                audio_out = gr.Audio(label="Generated audio", type="numpy")
                status = gr.Textbox(label="Status", lines=3, interactive=False)

        def on_lang_change(lang: str):
            choices = VOICE_CHOICES.get(lang, [])
            return gr.update(choices=choices, value=choices[0][1] if choices else None)

        def run(text: str, lang: str, voice: str, speed: float):
            try:
                t0 = time.perf_counter()
                wav, sr = engine.synthesize(text, lang, voice, speed)
                elapsed = time.perf_counter() - t0
                duration = len(wav) / sr
                rtf = elapsed / duration if duration else 0.0
                return (sr, wav), (
                    f"OK - {duration:.1f}s of audio in {elapsed:.2f}s "
                    f"({rtf:.2f}x real time)."
                )
            except Exception as exc:
                return None, f"{type(exc).__name__}: {exc}"

        lang_in.change(on_lang_change, inputs=lang_in, outputs=voice_in)
        btn.click(run, inputs=[text_in, lang_in, voice_in, speed_in], outputs=[audio_out, status])
        text_in.submit(run, inputs=[text_in, lang_in, voice_in, speed_in], outputs=[audio_out, status])

    return demo


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kokoro-demo",
        description="Gradio demo for Kokoro-82M (CPU-friendly alternative to Qwen3-TTS).",
    )
    parser.add_argument(
        "--lang",
        default="p",
        choices=[code for _, code in LANGUAGE_CHOICES],
        help="Default language code (p = Brazilian Portuguese).",
    )
    parser.add_argument(
        "--voice",
        default="pf_dora",
        choices=sorted(VOICE_CODES),
        help="Default Kokoro voice.",
    )
    parser.add_argument("--speed", type=float, default=1.0, help="Default speech speed.")
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda", "cuda:0", "mps"],
        help="Inference device (default: auto; CUDA if available, otherwise CPU).",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=0,
        help="PyTorch CPU threads (0 = library default).",
    )
    parser.add_argument("--ip", default="127.0.0.1", help="Server bind IP.")
    parser.add_argument("--port", type=int, default=7860, help="Server port.")
    parser.add_argument("--share", action="store_true", help="Create a public Gradio link.")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.voice not in {code for _, code in VOICE_CHOICES[args.lang]}:
        raise SystemExit(
            f"Voice {args.voice!r} does not belong to language {args.lang!r}. "
            f"Choose from: {[code for _, code in VOICE_CHOICES[args.lang]]}"
        )
    if not 0.5 <= args.speed <= 2.0:
        raise SystemExit("--speed must be between 0.5 and 2.0.")

    print(f"Loading {MODEL_ID} on {args.device} ...")
    engine = KokoroEngine(device=args.device, threads=args.threads)
    print(f"Ready on {engine.device}. Warming up {args.lang}/{args.voice} ...")
    engine.pipeline(args.lang).load_voice(args.voice)
    # The first inference pays lazy-init and G2P setup costs; do it at startup so
    # the first UI request feels responsive.
    engine.synthesize(
        WARMUP_TEXTS.get(args.lang, "Hello, this is a model warm-up."),
        args.lang,
        args.voice,
        1.0,
    )

    demo = build_demo(engine, args.lang, args.voice, args.speed)
    print(f"Open http://{args.ip}:{args.port}")
    demo.queue(default_concurrency_limit=1).launch(
        server_name=args.ip,
        server_port=args.port,
        share=args.share,
        show_error=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
