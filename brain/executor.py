"""Async executor for non-blocking Brain responses (Phase 6)."""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from typing import Optional

from .config import BrainConfig, load_config
from .constants import ALLOWED_TAGS, DEFAULT_COOLDOWN_MS, DEFAULT_DEBOUNCE_MS
from .service import clean_token, respond
from .types import BrainInput, BrainOutput, BrainStatus, ExecutorSnapshot


@dataclass
class _WorkItem:
    request_id: int
    submitted_ts: float
    brain_input: BrainInput
    source_id: Optional[str]


class BrainExecutor:
    """Background executor that ensures Gemini calls do not block realtime loops."""

    def __init__(self, cfg: BrainConfig | None = None) -> None:
        self._cfg = cfg or load_config()
        self._queue: "queue.Queue[_WorkItem]" = queue.Queue(maxsize=max(self._cfg.queue_maxsize, 1))
        self._stop_event = threading.Event()
        self._worker: threading.Thread | None = None
        self._lock = threading.Lock()
        self._latest_request_id = 0
        self._last_update_ts = time.time()
        self._pending: _WorkItem | None = None
        self._pending_timer: threading.Thread | None = None
        self._last_publish_ts: float | None = None
        self._last_submit_ts: float | None = None
        self._last_error: str | None = None
        self._snapshot = ExecutorSnapshot(
            status="idle",
            last_output=None,
            last_update_ts=self._last_update_ts,
            request_id=0,
            in_flight=False,
            last_error=None,
            debug={},
        )

    # Public API ---------------------------------------------------------
    def start(self) -> None:
        if self._worker and self._worker.is_alive():
            return
        self._stop_event.clear()
        self._worker = threading.Thread(target=self._run, name="BrainExecutor", daemon=True)
        self._worker.start()

    def stop(self, timeout_s: float = 2.0) -> None:
        self._stop_event.set()
        if self._worker:
            self._worker.join(timeout_s)

    def submit_tokens(self, tokens: list[str], *, source_id: str | None = None) -> None:
        if not tokens:
            brain_input = BrainInput(keywords=[], emotion="neutral")
            self.submit_input(brain_input, source_id=source_id)
            return

        last_cleaned = clean_token(tokens[-1])
        if last_cleaned in ALLOWED_TAGS:
            brain_input = BrainInput(keywords=tokens[:-1], emotion=last_cleaned)  # type: ignore[arg-type]
        else:
            brain_input = BrainInput(keywords=tokens, emotion="neutral")
        self.submit_input(brain_input, source_id=source_id)

    def submit_input(self, brain_input: BrainInput, *, source_id: str | None = None) -> None:
        now = time.perf_counter()
        with self._lock:
            self._latest_request_id += 1
            request_id = self._latest_request_id
            prev_submit = self._last_submit_ts
            self._last_submit_ts = now
            item = _WorkItem(request_id, now, brain_input, source_id)
            self._update_snapshot(status="listening", in_flight=False, request_id=request_id)

            debounce_s = max(self._cfg.debounce_ms, DEFAULT_DEBOUNCE_MS) / 1000.0
            if self._should_debounce(now, debounce_s, prev_submit):
                self._pending = item
                if not self._pending_timer or not self._pending_timer.is_alive():
                    self._pending_timer = threading.Thread(
                        target=self._flush_pending, args=(debounce_s,), daemon=True
                    )
                    self._pending_timer.start()
                return

        self._enqueue_latest(item)

    def poll_latest(self) -> ExecutorSnapshot:
        with self._lock:
            return self._snapshot

    def get_status(self) -> ExecutorSnapshot:
        return self.poll_latest()

    # Internal helpers ---------------------------------------------------
    def _flush_pending(self, delay_s: float) -> None:
        time.sleep(delay_s)
        with self._lock:
            item = self._pending
            self._pending = None
        if item:
            self._enqueue_latest(item)

    def _should_debounce(self, now: float, debounce_s: float, prev_submit: float | None) -> bool:
        if prev_submit is None:
            return False
        return (now - prev_submit) < debounce_s

    def _enqueue_latest(self, item: _WorkItem) -> None:
        # Clear queue to enforce latest-wins.
        try:
            while True:
                self._queue.get_nowait()
        except queue.Empty:
            pass
        try:
            self._queue.put_nowait(item)
        except queue.Full:
            # Drop silently; snapshot debug will note queue pressure.
            pass

    def _update_snapshot(
        self,
        *,
        status: BrainStatus | None = None,
        last_output: BrainOutput | None = None,
        in_flight: Optional[bool] = None,
        last_error: Optional[str] = None,
        debug_updates: Optional[dict[str, object]] = None,
        request_id: Optional[int] = None,
    ) -> None:
        debug = dict(self._snapshot.debug)
        if debug_updates:
            debug.update(debug_updates)
        self._last_update_ts = time.time()
        self._snapshot = ExecutorSnapshot(
            status=status or self._snapshot.status,
            last_output=last_output if last_output is not None else self._snapshot.last_output,
            last_update_ts=self._last_update_ts,
            request_id=request_id if request_id is not None else self._snapshot.request_id,
            in_flight=in_flight if in_flight is not None else self._snapshot.in_flight,
            last_error=last_error if last_error is not None else self._snapshot.last_error,
            debug=debug,
        )

    def _run(self) -> None:
        debounce_s = max(self._cfg.debounce_ms, DEFAULT_DEBOUNCE_MS) / 1000.0
        cooldown_s = max(self._cfg.cooldown_ms, DEFAULT_COOLDOWN_MS) / 1000.0
        while not self._stop_event.is_set():
            try:
                item = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if item.request_id < self._latest_request_id:
                continue

            # Cooldown gating for Gemini path
            if self._cfg.use_gemini and self._last_publish_ts is not None:
                while (
                    (time.perf_counter() - self._last_publish_ts) < cooldown_s
                    and item.brain_input.emotion != "question"
                    and not self._stop_event.is_set()
                ):
                    if item.request_id < self._latest_request_id:
                        break
                    time.sleep(0.05)

            self._update_snapshot(
                status="thinking",
                in_flight=True,
                request_id=item.request_id,
                debug_updates={
                    "queue_size": self._queue.qsize(),
                    "last_submit_ts": item.submitted_ts,
                    "last_request_keywords": item.brain_input.keywords,
                },
            )

            output = respond(item.brain_input, cfg=self._cfg)

            publish_time = time.perf_counter()
            if item.request_id != self._latest_request_id:
                # Drop stale result.
                continue

            self._last_publish_ts = publish_time
            self._last_error = output.error
            status: BrainStatus = output.status
            self._update_snapshot(
                status=status,
                last_output=output,
                in_flight=False,
                last_error=output.error,
                debug_updates={
                    "queue_size": self._queue.qsize(),
                    "last_publish_ts": publish_time,
                    "debounce_ms": debounce_s * 1000,
                    "cooldown_ms": cooldown_s * 1000,
                },
                request_id=item.request_id,
            )

        self._update_snapshot(status="idle", in_flight=False)

