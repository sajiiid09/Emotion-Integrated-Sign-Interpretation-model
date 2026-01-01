"""Real-time BdSL recognition demo with Brain HUD integration."""

from __future__ import annotations

import argparse
import dataclasses
import time
from pathlib import Path
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np
import torch

from brain import BrainExecutor, BrainOutput, load_config
from brain.prompt_builder import infer_mode
from brain.rules import resolve_emotion
from brain.intent import Intent
from brain.constants import MODE_TUTOR
from brain.service import is_affirmative_bn, is_negative_bn
from demo.hud_renderer import HUDRenderer
from models.constants import FACE_POINTS, HAND_POINTS, POSE_POINTS
from models.fusion import FusionModel
from preprocess.normalize import NormalizationConfig, normalize_sample
from train.vocab import build_vocab_from_manifest


GRAMMAR_IDX_TO_TAG = ["neutral", "question", "negation", "happy", "sad"]
DEFAULT_STABLE_FRAMES = 10
DEFAULT_MIN_CONF = 0.60
MAX_SENTENCE_WORDS = 12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run real-time BdSL demo with AI tutor overlay.")
    parser.add_argument("checkpoint", type=Path, help="Path to trained fusion model weights.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--buffer", type=int, default=48, help="Sliding window length for model input.")
    parser.add_argument("--manifest", type=Path, default=Path("data/manifest.csv"), help="Manifest CSV to recover vocabulary labels.")
    parser.add_argument("--font-path", type=Path, default=Path("demo/kalpurush.ttf"), help="Path to Bangla font (kalpurush/SolaimanLipi).")
    parser.add_argument("--stable-frames", type=int, default=DEFAULT_STABLE_FRAMES, help="Frames required before accepting a word.")
    parser.add_argument("--min-conf", type=float, default=DEFAULT_MIN_CONF, help="Confidence threshold for stable word selection.")
    parser.add_argument("--use-gemini", action="store_true", help="Force Gemini usage regardless of env config.")
    parser.add_argument("--no-gemini", action="store_true", help="Force stub mode regardless of env config.")
    parser.add_argument("--show-prompt", action="store_true", help="Show prompt preview in HUD debug strip.")
    return parser.parse_args()


def load_labels(manifest_path: Path) -> list[str]:
    if not manifest_path.exists() or manifest_path.stat().st_size == 0:
        return []
    try:
        vocab = build_vocab_from_manifest(manifest_path)
    except Exception:
        return []
    return vocab.idx_to_label


def _landmark_array(landmarks, size: int) -> np.ndarray:
    if landmarks is None:
        return np.zeros((size, 3), dtype=np.float32)
    return np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark], dtype=np.float32)


def _init_buffers(size: int) -> dict[str, dict[str, np.ndarray | int]]:
    return {
        "hand_left": {"data": np.zeros((size, HAND_POINTS, 3), dtype=np.float32), "write_idx": 0, "filled": 0},
        "hand_right": {"data": np.zeros((size, HAND_POINTS, 3), dtype=np.float32), "write_idx": 0, "filled": 0},
        "face": {"data": np.zeros((size, FACE_POINTS, 3), dtype=np.float32), "write_idx": 0, "filled": 0},
        "pose": {"data": np.zeros((size, POSE_POINTS, 3), dtype=np.float32), "write_idx": 0, "filled": 0},
    }


def _append_sample(buffers: dict[str, dict[str, np.ndarray | int]], sample: dict[str, np.ndarray]) -> None:
    for key, buffer in buffers.items():
        buffer["data"][buffer["write_idx"]] = sample[key]
    first = next(iter(buffers.values()))
    first["write_idx"] = (first["write_idx"] + 1) % first["data"].shape[0]
    first["filled"] = min(first["filled"] + 1, first["data"].shape[0])
    for buffer in buffers.values():
        buffer["write_idx"] = first["write_idx"]
        buffer["filled"] = first["filled"]


def _is_full(buffers: dict[str, dict[str, np.ndarray | int]]) -> bool:
    meta = next(iter(buffers.values()))
    return meta["filled"] == meta["data"].shape[0]


def _stack_window(buffers: dict[str, dict[str, np.ndarray | int]]) -> dict[str, np.ndarray]:
    stacked = {}
    sample_meta = next(iter(buffers.values()))
    size = sample_meta["data"].shape[0]
    write_idx = sample_meta["write_idx"]
    for key, buffer in buffers.items():
        if buffer["filled"] < size:
            stacked[key] = buffer["data"][: buffer["filled"]]
            continue
        if write_idx == 0:
            stacked[key] = buffer["data"]
        else:
            stacked[key] = np.concatenate((buffer["data"][write_idx:], buffer["data"][:write_idx]), axis=0)
    return stacked


def _format_word(sign_idx: int, labels: list[str]) -> str:
    if sign_idx < 0:
        return "..."
    if labels and 0 <= sign_idx < len(labels):
        return labels[sign_idx]
    return f"#{sign_idx}"


def _extract_prompt_preview(output: BrainOutput | None, limit: int = 120) -> str | None:
    if not output:
        return None
    prompt_debug = output.debug.get("prompt") if output.debug else None
    if not isinstance(prompt_debug, dict):
        return None
    text_preview: Optional[str] = prompt_debug.get("as_text") or prompt_debug.get("as_text_preview")
    if not text_preview:
        return None
    return text_preview[:limit]


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    model = FusionModel().to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    cfg = load_config()
    if args.use_gemini:
        cfg = dataclasses.replace(cfg, use_gemini=True)
    if args.no_gemini:
        cfg = dataclasses.replace(cfg, use_gemini=False)

    executor = BrainExecutor(cfg)
    executor.start()

    holistic = mp.solutions.holistic.Holistic()
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    if not ret:
        print("Failed to read from camera.")
        executor.stop()
        cap.release()
        holistic.close()
        return

    renderer = HUDRenderer(frame.shape, font_path=args.font_path)
    buffers = _init_buffers(args.buffer)
    config = NormalizationConfig(sequence_length=args.buffer)
    ema_sign = None
    ema_grammar = None
    alpha = 0.6
    idx_to_label = load_labels(args.manifest)
    stable_count = 0
    last_word_frame: Optional[str] = None
    last_stable_word: Optional[str] = None
    sentence_buffer: list[str] = []
    last_submitted_signature = ""
    use_gemini_override = cfg.use_gemini
    show_prompt = args.show_prompt
    presentation_mode = False
    show_help_until: float = 0.0
    last_phrase_ts = time.time()  # Phase 2: Phrase boundary timer
    last_submitted_keywords: list[str] = []  # Phase 2: Track keywords for trigger policy

    try:
        while True:
            start = time.time()
            ret, frame = cap.read()
            if not ret:
                break
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = holistic.process(image_rgb)
            sample = {
                "hand_left": _landmark_array(result.left_hand_landmarks, HAND_POINTS),
                "hand_right": _landmark_array(result.right_hand_landmarks, HAND_POINTS),
                "face": _landmark_array(result.face_landmarks, FACE_POINTS),
                "pose": _landmark_array(result.pose_landmarks, POSE_POINTS),
            }
            _append_sample(buffers, sample)

            sign_pred = -1
            grammar_pred = -1
            sign_conf = 0.0
            grammar_tag = "neutral"
            if _is_full(buffers):
                ordered = _stack_window(buffers)
                normalized = normalize_sample(ordered, config)
                tensor_sample = {k: torch.from_numpy(v).unsqueeze(0).to(device).float() for k, v in normalized.items()}
                with torch.no_grad():
                    sign_logits, grammar_logits = model(tensor_sample)
                sign_prob = torch.softmax(sign_logits, dim=1)
                grammar_prob = torch.softmax(grammar_logits, dim=1)
                ema_sign = sign_prob if ema_sign is None else alpha * sign_prob + (1 - alpha) * ema_sign
                ema_grammar = grammar_prob if ema_grammar is None else alpha * grammar_prob + (1 - alpha) * ema_grammar
                sign_pred = int(torch.argmax(ema_sign))
                grammar_pred = int(torch.argmax(ema_grammar))
                sign_conf = float(ema_sign[0, sign_pred]) if ema_sign is not None else 0.0
                grammar_tag = GRAMMAR_IDX_TO_TAG[grammar_pred] if 0 <= grammar_pred < len(GRAMMAR_IDX_TO_TAG) else "neutral"

            current_word = _format_word(sign_pred, idx_to_label)
            if current_word == last_word_frame and sign_conf >= args.min_conf:
                stable_count += 1
            else:
                stable_count = 0
            last_word_frame = current_word

            new_word_added = False
            if stable_count >= args.stable_frames and current_word not in (None, "..."):
                if current_word != last_stable_word:
                    sentence_buffer.append(current_word)
                    last_stable_word = current_word
                    stable_count = 0
                    new_word_added = True
                    if len(sentence_buffer) > MAX_SENTENCE_WORDS:
                        sentence_buffer = sentence_buffer[-MAX_SENTENCE_WORDS:]

            display_sentence = " ".join(sentence_buffer)
            tag_for_submission = grammar_tag or "neutral"
            signature = f"{display_sentence}|{tag_for_submission}"
            
            # Phase 2: Smart trigger policy
            should_submit = False
            if new_word_added and (sentence_buffer or tag_for_submission != "neutral"):
                # Parse intent to check mode and emotion
                intent = Intent(
                    keywords=sentence_buffer,
                    raw_keywords=sentence_buffer,
                    detected_emotion=tag_for_submission,
                    meta=None,
                    flags={},
                    notes=[],
                )
                resolved = resolve_emotion(intent)
                built_prompt_intent = Intent(
                    keywords=sentence_buffer,
                    raw_keywords=sentence_buffer,
                    detected_emotion=resolved.resolved_emotion,
                    meta=None,
                    flags={},
                    notes=[],
                )
                resolved_for_mode = resolve_emotion(built_prompt_intent)
                inferred_mode = infer_mode(resolved_for_mode)
                
                # Trigger if tutor mode or question
                if inferred_mode == MODE_TUTOR or resolved.resolved_emotion == "question":
                    should_submit = True
                # Or if phrase boundary exceeded
                elif (time.time() - last_phrase_ts) > (cfg.phrase_pause_ms / 1000.0):
                    should_submit = True
                    last_phrase_ts = time.time()
                
                # Check for yes/no as continuation
                if should_submit and (is_affirmative_bn(sentence_buffer) or is_negative_bn(sentence_buffer)):
                    should_submit = True  # Always submit yes/no
            
            if should_submit and signature != last_submitted_signature:
                executor.submit_tokens(sentence_buffer + [tag_for_submission])
                last_submitted_signature = signature
                last_phrase_ts = time.time()
                last_submitted_keywords = list(sentence_buffer)

            snapshot = executor.poll_latest()
            last_output: Optional[BrainOutput] = snapshot.last_output
            tutor_text = last_output.response_bn if last_output else "ভাবছি..."
            resolved_tag = last_output.resolved_emotion if last_output else tag_for_submission
            latency_ms = last_output.latency_ms if last_output else None
            if snapshot.status == "thinking" and (not last_output or last_output.response_bn == ""):
                tutor_text = "ভাবছি..."

            prompt_preview = _extract_prompt_preview(last_output) if show_prompt else None

            fps_val = 1.0 / max((time.time() - start), 1e-6)
            overlay = renderer.render(
                frame,
                status=snapshot.status,
                predicted_word=current_word,
                confidence=sign_conf,
                resolved_tag=resolved_tag,
                display_sentence=display_sentence if grammar_tag != "question" else f"{display_sentence}?",
                tutor_text=tutor_text,
                fps=fps_val,
                latency_ms=latency_ms,
                prompt_preview=prompt_preview,
                gemini_on=use_gemini_override,
                api_key_present=bool(cfg.api_key),
                presentation_mode=presentation_mode,
                last_word=last_stable_word,
                show_help=time.time() < show_help_until,
            )

            cv2.imshow("BdSL Demo", overlay)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("c"):
                sentence_buffer.clear()
                last_stable_word = None
                stable_count = 0
                last_submitted_signature = ""
            if key == ord("g"):
                use_gemini_override = not use_gemini_override
                cfg = dataclasses.replace(cfg, use_gemini=use_gemini_override)
                executor.stop()
                executor = BrainExecutor(cfg)
                executor.start()
                if sentence_buffer or tag_for_submission != "neutral":
                    executor.submit_tokens(sentence_buffer + [tag_for_submission])
            if key == ord("p"):
                show_prompt = not show_prompt
            if key == ord("m"):
                presentation_mode = not presentation_mode
            if key == ord("h"):
                show_help_until = time.time() + 3

    finally:
        executor.stop()
        cap.release()
        holistic.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
