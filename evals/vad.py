"""Offline VAD measurement harness: compare fastrtc's chunked ReplyOnPause against
InterruptSafeReplyOnPause's streaming 32 ms Silero VAD (#64).

Measures reply-start latency on clean utterances (including short one-word commands), false
mid-sentence turn splits across 300-800 ms pauses, false triggers on brief noise/clap bursts,
and CPU cost per second of audio. Runs offline (no llama-server, browser, or mic required).
"""
import argparse
import json
import statistics
import subprocess
import sys
import tempfile
import time
import wave
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from fastrtc import AlgoOptions, ReplyOnPause
from fastrtc.pause_detection.silero import SileroVADModel, SileroVadOptions

from vaice import interruptible

HERE = Path(__file__).parent
SAMPLE_RATE = 48000
FRAME_MS = 20
PHASES_MS = (0, 100, 200, 300, 400, 500)
PAUSE_GAPS_MS = (300, 500, 600, 700, 800)
CLEAN_PHRASES = (
    ("time", "What time is it?", 0),
    ("battery", "Check my battery level.", 0),
    ("safari", "Open Safari please.", 0),
    ("disk", "How much disk space do I have left?", 0),
    ("yes", "Yes.", 0),
    ("stop", "Stop.", 0),
    ("cancel", "Cancel.", 0),
    ("hello", "Hello.", 1),
)
SPLIT_PHRASES = (
    ("research-1", "Can you research some", "coffee machines for me", 0),
    ("research-2", "Research some coffee machine online", "for me please", 0),
    ("weather", "Open Safari and check", "the weather in Boston", 0),
    ("battery-split", "What is the battery level", "on this laptop right now", 0),
    ("memory-split", "Remember that I park", "on level three", 1),
    ("apps-split", "Quit Calculator", "and open Notes", 0),
)


def _trim_silence(pcm: np.ndarray, floor: int = 200) -> np.ndarray:
    active = np.flatnonzero(np.abs(pcm) > floor)
    return pcm[active[0] : active[-1] + 1] if len(active) else pcm


def _read_mono_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as wav:
        if wav.getsampwidth() != 2 or wav.getframerate() != SAMPLE_RATE:
            sys.exit(f"evals: {path} must be 16-bit PCM at {SAMPLE_RATE} Hz")
        raw = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2")
        channels = wav.getnchannels()
    if channels > 1:
        raw = raw.reshape(-1, channels).mean(axis=1).astype("<i2")
    return _trim_silence(raw)


def _synth_say(text: str, voice: str | None) -> np.ndarray:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        path = Path(tmp.name)
    try:
        cmd = ["say", "-o", str(path), f"--data-format=LEI16@{SAMPLE_RATE}"]
        if voice:
            cmd += ["-v", voice]
        cmd.append(text)
        subprocess.run(cmd, check=True, capture_output=True, stdin=subprocess.DEVNULL, timeout=30)
        return _read_mono_wav(path)
    except FileNotFoundError:
        sys.exit(f"evals: macOS `say` not found and no WAV provided for {text!r}; pass --recordings DIR")
    finally:
        path.unlink(missing_ok=True)


def _load_clip(clip_id: str, text: str, voice: str | None, recordings: Path | None) -> np.ndarray:
    if recordings is not None:
        wav_path = recordings / f"{clip_id}.wav"
        if wav_path.is_file():
            return _read_mono_wav(wav_path)
    return _synth_say(text, voice)


def _silence(ms: int) -> np.ndarray:
    return np.zeros(SAMPLE_RATE * ms // 1000, dtype=np.int16)


def _run_handler(
    clip: np.ndarray,
    *,
    handler_cls: type[ReplyOnPause],
    vad_model: SileroVADModel,
    algo: AlgoOptions,
    opts: SileroVadOptions,
) -> tuple[bool, list[float]]:
    handler = handler_cls(
        lambda _audio: iter(()),
        algo_options=algo,
        model_options=opts,
        model=vad_model,
        input_sample_rate=SAMPLE_RATE,
    )
    handler.send_message_sync = lambda _msg: None
    handler._clear_queue = lambda: None
    step = SAMPLE_RATE * FRAME_MS // 1000
    started = False
    pauses: list[float] = []
    for i in range(0, len(clip), step):
        frame = clip[i : i + step]
        if len(frame) == 0:
            continue
        handler.receive((SAMPLE_RATE, frame.reshape(1, -1)))
        if handler.state.started_talking:
            started = True
        if handler.event.is_set():
            pauses.append((i + len(frame)) / SAMPLE_RATE)
            handler.reset()
    return started, pauses


def _evaluate_setting(
    label: str,
    *,
    handler_cls: type[ReplyOnPause],
    vad_model: SileroVADModel,
    algo: AlgoOptions,
    opts: SileroVadOptions,
    clean_clips: Sequence[np.ndarray],
    split_clips: Sequence[tuple[np.ndarray, np.ndarray]],
    noise_100: np.ndarray,
    clap_180: np.ndarray,
) -> dict:
    # Measure per-second CPU cost on a 10 s clip (utterance + trailing silence).
    ten_s = np.concatenate([clean_clips[0], np.zeros(max(0, SAMPLE_RATE * 10 - len(clean_clips[0])), dtype=np.int16)])
    t0 = time.monotonic()
    _run_handler(ten_s, handler_cls=handler_cls, vad_model=vad_model, algo=algo, opts=opts)
    cpu_ms_per_s = (time.monotonic() - t0) * 1000.0 / (len(ten_s) / SAMPLE_RATE)

    latencies_ms: list[float] = []
    misses = 0
    for utt in clean_clips:
        for lead_ms in PHASES_MS:
            lead = _silence(lead_ms)
            clip = np.concatenate([lead, utt, _silence(2500)])
            eos = (len(lead) + len(utt)) / SAMPLE_RATE
            _, pauses = _run_handler(clip, handler_cls=handler_cls, vad_model=vad_model, algo=algo, opts=opts)
            if not pauses:
                misses += 1
            else:
                latencies_ms.append((pauses[-1] - eos) * 1000.0)

    false_stops: dict[str, dict[str, int]] = {}
    for gap_ms in PAUSE_GAPS_MS:
        bad = 0
        total = 0
        for part_a, part_b in split_clips:
            gap = _silence(gap_ms)
            for lead_ms in PHASES_MS:
                total += 1
                lead = _silence(lead_ms)
                clip = np.concatenate([lead, part_a, gap, part_b, _silence(2500)])
                eos = (len(lead) + len(part_a) + len(gap) + len(part_b)) / SAMPLE_RATE
                _, pauses = _run_handler(clip, handler_cls=handler_cls, vad_model=vad_model, algo=algo, opts=opts)
                if len(pauses) != 1 or pauses[0] < eos:
                    bad += 1
        false_stops[str(gap_ms)] = {"bad": bad, "total": total}

    def false_positives(burst: np.ndarray) -> int:
        hits = 0
        for lead_ms in PHASES_MS:
            clip = np.concatenate([_silence(lead_ms), burst, _silence(1500)])
            started, pauses = _run_handler(clip, handler_cls=handler_cls, vad_model=vad_model, algo=algo, opts=opts)
            if started or pauses:
                hits += 1
        return hits

    lat_sorted = sorted(latencies_ms)
    p90 = lat_sorted[int(0.9 * (len(lat_sorted) - 1))] if lat_sorted else None
    med = statistics.median(lat_sorted) if lat_sorted else None
    return {
        "label": label,
        "mode": "streaming" if handler_cls is interruptible.InterruptSafeReplyOnPause else "chunked",
        "min_silence_ms": opts.min_silence_duration_ms,
        "audio_chunk_s": algo.audio_chunk_duration,
        "speech_threshold": algo.speech_threshold,
        "cpu_ms_per_s": round(cpu_ms_per_s, 1),
        "clean_latency_ms": {"median": round(med) if med is not None else None, "p90": round(p90) if p90 else None},
        "clean_misses": misses,
        "clean_total": len(clean_clips) * len(PHASES_MS),
        "false_stops": false_stops,
        "noise_100ms_fp": false_positives(noise_100),
        "clap_180ms_fp": false_positives(clap_180),
        "burst_trials": len(PHASES_MS),
    }


def format_table(rows: Sequence[dict]) -> str:
    lines = [
        "| Config | Lat med/p90 | Miss | FS 300 | FS 500 | FS 600 | FS 700 | FS 800 | Noise/Clap FP | CPU ms/s |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lat = r["clean_latency_ms"]
        med = "-" if lat["median"] is None else str(lat["median"])
        p90 = "-" if lat["p90"] is None else str(lat["p90"])
        fs = r["false_stops"]
        fs_cells = [f"{fs[str(g)]['bad']}/{fs[str(g)]['total']}" for g in PAUSE_GAPS_MS]
        fp = f"{r['noise_100ms_fp']}/{r['burst_trials']}, {r['clap_180ms_fp']}/{r['burst_trials']}"
        lines.append(
            f"| {r['label']} | {med} / {p90} | {r['clean_misses']}/{r['clean_total']} | "
            + " | ".join(fs_cells)
            + f" | {fp} | {r['cpu_ms_per_s']} |"
        )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark ReplyOnPause vs streaming Silero VAD (#64)")
    parser.add_argument("--recordings", type=Path, help="Optional directory with pre-recorded 48 kHz mono WAVs")
    parser.add_argument("--voices", default="Samantha,Daniel", help="Comma-separated macOS `say` voices")
    parser.add_argument("--out", type=Path, help="JSON output path (default: evals/results/vad-<UTC>.json)")
    args = parser.parse_args(argv)

    voices = [v.strip() for v in args.voices.split(",") if v.strip()] or [""]

    def voice_for(idx: int) -> str | None:
        chosen = voices[idx % len(voices)]
        return chosen or None

    interruptible.use_cached_vad_model()
    vad_model = SileroVADModel()

    clean_clips = [_load_clip(cid, text, voice_for(v_idx), args.recordings) for cid, text, v_idx in CLEAN_PHRASES]
    split_clips = [
        (
            _load_clip(f"{cid}-a", part_a, voice_for(v_idx), args.recordings),
            _load_clip(f"{cid}-b", part_b, voice_for(v_idx), args.recordings),
        )
        for cid, part_a, part_b, v_idx in SPLIT_PHRASES
    ]

    rng = np.random.default_rng(42)
    noise_100 = rng.normal(0, 3000, SAMPLE_RATE * 100 // 1000).astype(np.int16)
    t = np.linspace(0, 1, SAMPLE_RATE * 180 // 1000)
    clap_180 = (rng.normal(0, 8000, len(t)) * np.exp(-8 * t)).astype(np.int16)

    # Warm up both handlers (librosa/numba and ONNX) so row 1's CPU timing reflects steady state.
    for warmup_cls, warmup_algo, warmup_opts in (
        (ReplyOnPause, AlgoOptions(), SileroVadOptions(0.6, 250, float("inf"), 150, 1024, 30)),
        (interruptible.InterruptSafeReplyOnPause, AlgoOptions(), interruptible.DEFAULT_VAD_OPTIONS),
    ):
        _run_handler(
            np.concatenate([clean_clips[0], _silence(1000)]),
            handler_cls=warmup_cls,
            vad_model=vad_model,
            algo=warmup_algo,
            opts=warmup_opts,
        )

    sweep = [
        ("chunked: previous defaults (0.6s, sil=150ms)", ReplyOnPause, AlgoOptions(0.6, 0.2, 0.1),
         SileroVadOptions(0.6, 250, float("inf"), 150, 1024, 30)),
        ("chunked: sil=75ms (0.6s chunk)", ReplyOnPause, AlgoOptions(0.6, 0.2, 0.1),
         SileroVadOptions(0.6, 250, float("inf"), 75, 1024, 30)),
        ("chunked: sil=100ms (0.6s chunk)", ReplyOnPause, AlgoOptions(0.6, 0.2, 0.1),
         SileroVadOptions(0.6, 250, float("inf"), 100, 1024, 30)),
        ("chunked: sil=600ms (0.6s chunk)", ReplyOnPause, AlgoOptions(0.6, 0.2, 0.1),
         SileroVadOptions(0.6, 250, float("inf"), 600, 1024, 30)),
        ("chunked: chunk=1.0s, speech_th=0.05, sil=150, min_sp=100", ReplyOnPause, AlgoOptions(1.0, 0.2, 0.05),
         SileroVadOptions(0.6, 100, float("inf"), 150, 1024, 30)),
        *[
            (
                f"streaming: VAICE_VAD_MIN_SILENCE_MS={sil_ms}",
                interruptible.InterruptSafeReplyOnPause,
                AlgoOptions(),
                SileroVadOptions(0.6, 250, float("inf"), sil_ms, 1024, 30),
            )
            for sil_ms in (500, 600, 700, 750, 800, 850, 1000)
        ],
    ]

    results = [
        _evaluate_setting(
            label,
            handler_cls=handler_cls,
            vad_model=vad_model,
            algo=algo,
            opts=opts,
            clean_clips=clean_clips,
            split_clips=split_clips,
            noise_100=noise_100,
            clap_180=clap_180,
        )
        for label, handler_cls, algo, opts in sweep
    ]

    table = format_table(results)
    print(table)

    out_path = args.out
    if out_path is None:
        stamp = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
        out_path = HERE / "results" / f"vad-{stamp}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"generated_at": datetime.now(tz=UTC).isoformat(), "results": results}, indent=2))
    print(f"\nSaved raw results to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
