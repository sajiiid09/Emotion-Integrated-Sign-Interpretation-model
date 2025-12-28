"""HUD renderer for the realtime demo overlay."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


@dataclass(frozen=True)
class HUDTheme:
    panel_bg: tuple[int, int, int, int] = (12, 12, 12, 200)
    card_bg: tuple[int, int, int, int] = (22, 22, 22, 220)
    text_primary: tuple[int, int, int] = (235, 235, 235)
    text_muted: tuple[int, int, int] = (180, 180, 180)
    accent: tuple[int, int, int] = (104, 187, 227)
    danger: tuple[int, int, int] = (240, 84, 84)
    warning: tuple[int, int, int] = (244, 198, 86)
    success: tuple[int, int, int] = (96, 189, 104)
    neutral: tuple[int, int, int] = (124, 126, 132)


def _load_font(font_path: Path | None, size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    if font_path and font_path.exists():
        try:
            return ImageFont.truetype(str(font_path), size=size)
        except OSError:
            pass
    return ImageFont.load_default()


def _draw_chip(draw: ImageDraw.ImageDraw, xy: tuple[int, int, int, int], text: str, font, fill):
    draw.rounded_rectangle(xy, radius=10, fill=fill)
    text_w = draw.textlength(text, font=font)
    tx = xy[0] + (xy[2] - xy[0] - text_w) / 2
    ty = xy[1] + (xy[3] - xy[1] - font.size) / 2
    draw.text((tx, ty), text, font=font, fill=(0, 0, 0))


def _wrap_text(draw: ImageDraw.ImageDraw, text: str, font, max_width: int, max_lines: int = 4) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current: list[str] = []
    for word in words:
        test_line = " ".join(current + [word]).strip()
        if test_line and draw.textlength(test_line, font=font) <= max_width:
            current.append(word)
        else:
            if current:
                lines.append(" ".join(current))
            current = [word]
        if len(lines) >= max_lines:
            break
    if current and len(lines) < max_lines:
        lines.append(" ".join(current))
    if len(lines) > max_lines:
        lines = lines[:max_lines]
    if len(lines) == max_lines and len(words) > len(" ".join(lines).split()):
        lines[-1] = lines[-1].rstrip() + "..."
    return lines


def _status_color(theme: HUDTheme, status: str) -> tuple[int, int, int]:
    if status == "ready":
        return theme.success
    if status == "thinking":
        return theme.accent
    if status == "listening":
        return theme.warning
    if status == "error":
        return theme.danger
    return theme.neutral


class HUDRenderer:
    """Render a polished overlay on top of the webcam feed."""

    def __init__(
        self,
        frame_shape: Sequence[int],
        *,
        font_path: Path | None = None,
        theme: HUDTheme | None = None,
    ) -> None:
        height, width = frame_shape[:2]
        self.panel_width = int(width * 0.36)
        self.margin = 16
        self.theme = theme or HUDTheme()
        self.font_title = _load_font(font_path, 28)
        self.font_label = _load_font(font_path, 20)
        self.font_body = _load_font(font_path, 18)
        self.font_small = _load_font(font_path, 16)
        self.font_path = font_path
        self.font_missing = font_path is not None and not font_path.exists()

    def render(
        self,
        frame: np.ndarray,
        *,
        status: str,
        predicted_word: str,
        confidence: float,
        resolved_tag: str,
        display_sentence: str,
        tutor_text: str,
        fps: float,
        latency_ms: float | None,
        prompt_preview: str | None = None,
    ) -> np.ndarray:
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert("RGBA")
        draw = ImageDraw.Draw(img)
        w, h = img.size
        panel_x = w - self.panel_width - self.margin
        panel_y = self.margin
        panel_h = h - 2 * self.margin
        draw.rounded_rectangle(
            (panel_x, panel_y, panel_x + self.panel_width, panel_y + panel_h),
            radius=18,
            fill=self.theme.panel_bg,
        )

        cursor_y = panel_y + 18
        padding = 16
        draw.text((panel_x + padding, cursor_y), "AI Tutor", font=self.font_title, fill=self.theme.text_primary)

        chip_w, chip_h = 120, 34
        chip_x = panel_x + self.panel_width - padding - chip_w
        chip_y = cursor_y - 6
        _draw_chip(draw, (chip_x, chip_y, chip_x + chip_w, chip_y + chip_h), status.title(), self.font_small, _status_color(self.theme, status))

        cursor_y += chip_h + 12
        sections = [
            ("Detected", self._render_detected, {
                "predicted_word": predicted_word,
                "confidence": confidence,
                "resolved_tag": resolved_tag,
            }),
            ("Sentence", self._render_sentence, {"text": display_sentence}),
            ("Tutor reply", self._render_reply, {
                "text": tutor_text,
                "status": status,
                "latency_ms": latency_ms,
                "prompt_preview": prompt_preview,
            }),
        ]

        for title, renderer, payload in sections:
            cursor_y = self._render_card(draw, panel_x + padding, cursor_y, self.panel_width - 2 * padding, title, renderer, payload)

        footer_text = f"FPS: {fps:.1f}"
        if latency_ms is not None:
            footer_text += f" | Latency: {latency_ms:.0f} ms"
        draw.text((panel_x + padding, panel_y + panel_h - padding - self.font_small.size), footer_text, font=self.font_small, fill=self.theme.text_muted)

        if self.font_missing:
            warning = f"Font not found: {self.font_path.name if self.font_path else ''}. Using default."
            draw.text((self.margin, h - self.margin - self.font_small.size), warning, font=self.font_small, fill=(255, 200, 120))

        return cv2.cvtColor(np.array(img), cv2.COLOR_RGBA2BGR)

    def _render_card(
        self,
        draw: ImageDraw.ImageDraw,
        x: int,
        y: int,
        width: int,
        title: str,
        renderer,
        payload: dict,
    ) -> int:
        card_pad = 12
        max_height = 280
        inner_x1 = x + card_pad
        inner_y1 = y + card_pad
        inner_x2 = x + width - card_pad
        card_height = renderer(draw, inner_x1, inner_y1, inner_x2, payload, measure_only=True)
        outer_height = card_height + 2 * card_pad
        draw.rounded_rectangle((x, y, x + width, y + outer_height), radius=14, fill=self.theme.card_bg)
        draw.text((x + card_pad, y + card_pad), title, font=self.font_label, fill=self.theme.text_primary)
        renderer(draw, inner_x1, inner_y1 + self.font_label.size + 6, inner_x2, payload, measure_only=False)
        return y + min(outer_height + 12, max_height + card_pad)

    def _render_detected(self, draw: ImageDraw.ImageDraw, x1: int, y1: int, x2: int, payload: dict, measure_only: bool = False) -> int:
        height = self.font_label.size + 6 + self.font_body.size * 2 + 12
        if measure_only:
            return height
        word = payload.get("predicted_word", "...")
        confidence = payload.get("confidence", 0.0)
        tag = payload.get("resolved_tag", "neutral")
        draw.text((x1, y1 + self.font_label.size + 6), f"শব্দ: {word}", font=self.font_body, fill=self.theme.text_primary)
        draw.text((x1, y1 + self.font_label.size + 6 + self.font_body.size + 6), f"বিশ্বাস: {confidence*100:.1f}%", font=self.font_body, fill=self.theme.text_muted)
        chip_y = y1 + self.font_label.size + 6
        chip_h = 30
        chip_w = 110
        chip_x = x2 - chip_w
        _draw_chip(draw, (chip_x, chip_y, chip_x + chip_w, chip_y + chip_h), tag.title(), self.font_small, _status_color(self.theme, tag))
        return height

    def _render_sentence(
        self,
        draw: ImageDraw.ImageDraw,
        x1: int,
        y1: int,
        x2: int,
        payload: dict,
        measure_only: bool = False,
    ) -> int:
        text = payload.get("text", "") or "(কোনো শব্দ পাওয়া যায়নি)"
        lines = _wrap_text(draw, text, self.font_body, max_width=int(x2 - x1), max_lines=3)
        height = self.font_label.size + 6 + len(lines) * (self.font_body.size + 4) + 6
        if measure_only:
            return height
        for idx, line in enumerate(lines):
            draw.text((x1, y1 + idx * (self.font_body.size + 4)), line, font=self.font_body, fill=self.theme.text_primary)
        return height

    def _render_reply(
        self,
        draw: ImageDraw.ImageDraw,
        x1: int,
        y1: int,
        x2: int,
        payload: dict,
        measure_only: bool = False,
    ) -> int:
        text = payload.get("text") or "ভাবছি..."
        status = payload.get("status", "idle")
        prompt_preview = payload.get("prompt_preview")
        lines = _wrap_text(draw, text, self.font_body, max_width=int(x2 - x1), max_lines=4)
        offset = len(lines) * (self.font_body.size + 4)
        if prompt_preview:
            offset += self.font_small.size + 4
        total_height = self.font_label.size + 6 + offset + self.font_small.size + 6
        if measure_only:
            return total_height
        for idx, line in enumerate(lines):
            draw.text((x1, y1 + idx * (self.font_body.size + 4)), line, font=self.font_body, fill=self.theme.text_primary)
        offset_draw = len(lines) * (self.font_body.size + 4)
        if prompt_preview:
            draw.text((x1, y1 + offset_draw + 4), prompt_preview, font=self.font_small, fill=self.theme.text_muted)
            offset_draw += self.font_small.size + 4
        if status == "thinking":
            draw.text((x2 - 80, y1 + offset_draw), "ভাবছি...", font=self.font_small, fill=self.theme.text_muted)
        latency_ms = payload.get("latency_ms")
        if latency_ms:
            draw.text((x2 - 110, y1 + offset_draw + self.font_small.size + 4), f"{latency_ms:.0f} ms", font=self.font_small, fill=self.theme.text_muted)
        return total_height
