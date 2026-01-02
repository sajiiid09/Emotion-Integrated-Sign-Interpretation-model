"""Async executor for non-blocking Brain responses (Phase 6)."""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from typing import Optional

from .cache import make_cache_key
from .config import BrainConfig, load_config
from .constants import ALLOWED_TAGS, DEFAULT_COOLDOWN_MS, DEFAULT_DEBOUNCE_MS, MODE_TUTOR
from .logging_utils import append_event, build_log_event
from .prompt_builder import build_prompt
from .lang.pipeline import run_language_pipeline
from .rules import resolve_emotion
from .service import clean_token, is_affirmative_bn, is_negative_bn, respond
from .types import BrainInput, BrainOutput, BrainStatus, ExecutorSnapshot


@dataclass
class _WorkItem:
    request_id: int
    submitted_ts: float
    brain_input: BrainInput
    source_id: Optional[str]


def should_trigger_gemini(
    resolved_emotion: str,
    mode: str,
    now_ts: float,
    last_submit_ts: float | None,
    cfg: BrainConfig,
) -> bool:
    """Determine if Gemini call should be triggered based on Phase 2 policy.
    
    Returns True if:
    1) Tutor mode enabled
    2) Question emotion
    3) Phrase boundary exceeded with mode change
    Otherwise returns False.
    """
    # Tutor always triggers
    if mode == MODE_TUTOR:
        return True
    
    # Questions always trigger
    if resolved_emotion == "question":
        return True
    
    # Phrase boundary with tutor mode would have triggered
    if last_submit_ts is not None:
        phrase_boundary_s = cfg.phrase_pause_ms / 1000.0
        if (now_ts - last_submit_ts) > phrase_boundary_s:
            return True
    
    return False


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
        self._dropped_results = 0
        self._cache_hits = 0
        
        # Phase 2: Request minimization state
        self._last_published_gloss: str | None = None
        self._last_tutor_topics: list[str] = []
        self._last_subject: str | None = None
        
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
            with self._lock:
                self._dropped_results += 1
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

            # Phase 2: Check for local continuation (yes/no handling)
            is_affirmative = is_affirmative_bn(item.brain_input.keywords)
            is_negative = is_negative_bn(item.brain_input.keywords)
            
            if is_affirmative and self._last_tutor_topics:
                # Local continuation without Gemini call
                continuation_response = (
                    f"চলুন আরও শিখি। আপনি কোনটা আগে দেখতে চান: {', '.join(self._last_tutor_topics)}?"
                )
                # Create a synthetic output for continuation
                from .types import BrainOutput
                output = BrainOutput(
                    response_bn=continuation_response,
                    resolved_emotion="neutral",
                    status="ready",
                    error=None,
                    latency_ms=0,
                    debug={
                        "continuation_local": True,
                        "last_topics": self._last_tutor_topics,
                    },
                )
                publish_time = time.perf_counter()
                self._last_publish_ts = publish_time
                self._update_snapshot(
                    status="ready",
                    last_output=output,
                    in_flight=False,
                    debug_updates={
                        "continuation_local": True,
                        "topics_count": len(self._last_tutor_topics),
                    },
                )
                continue
            
            if is_negative:
                # Local rejection without Gemini call
                rejection_response = "ঠিক আছে। আপনি কোন বিষয় শিখতে চান? যেমন বিজ্ঞান, গণিত, ইতিহাস, কম্পিউটার।"
                from .types import BrainOutput
                output = BrainOutput(
                    response_bn=rejection_response,
                    resolved_emotion="neutral",
                    status="ready",
                    error=None,
                    latency_ms=0,
                    debug={"continuation_rejected": True},
                )
                publish_time = time.perf_counter()
                self._last_publish_ts = publish_time
                self._last_tutor_topics = []  # Clear topics
                self._update_snapshot(
                    status="ready",
                    last_output=output,
                    in_flight=False,
                    debug_updates={"continuation_rejected": True},
                )
                continue

            # Phase 2: Smart trigger policy
            intent = BrainInput(keywords=item.brain_input.keywords, emotion=item.brain_input.emotion)
            resolved = resolve_emotion(intent)
            shaped = run_language_pipeline(resolved.keywords, resolved.resolved_emotion)
            built_prompt = build_prompt(resolved, shaped=shaped, cfg=self._cfg)

            will_trigger = should_trigger_gemini(
                resolved.resolved_emotion,
                built_prompt.mode,
                item.submitted_ts,
                self._last_submit_ts,
                self._cfg,
            )

            if shaped.intent_type in {"clarify", "interaction"}:
                will_trigger = False
            
            # Check cache hit before potentially triggering Gemini
            cache_key = None
            if self._cfg.use_gemini and self._cfg.cache_enabled:
                cache_key = make_cache_key(shaped.proto_bn, built_prompt.mode, resolved.resolved_emotion)
            
            # If gloss unchanged and recent, skip call (Phase 2 optimization)
            if (
                self._last_published_gloss == shaped.proto_bn
                and self._last_publish_ts is not None
                and (time.perf_counter() - self._last_publish_ts) < (self._cfg.cooldown_ms / 1000.0)
            ):
                will_trigger = False

            # Cooldown gating for Gemini path
            if self._cfg.use_gemini and self._last_publish_ts is not None:
                while (
                    (time.perf_counter() - self._last_publish_ts) < cooldown_s
                    and item.brain_input.emotion != "question"
                    and not self._stop_event.is_set()
                    and will_trigger
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
                    "will_trigger_gemini": will_trigger,
                },
            )

            output = respond(item.brain_input, cfg=self._cfg)

            publish_time = time.perf_counter()
            if item.request_id != self._latest_request_id:
                # Drop stale result.
                with self._lock:
                    self._dropped_results += 1
                continue

            # Extract topics for Phase 2 continuation
            if output.debug and output.debug.get("gemini"):
                gemini_meta = output.debug.get("gemini", {})
                if isinstance(gemini_meta, dict):
                    next_topics = gemini_meta.get("next_topics", [])
                    if next_topics:
                        self._last_tutor_topics = next_topics
                        self._last_subject = item.brain_input.keywords[0] if item.brain_input.keywords else None

            if output.status == "ready":
                self._last_publish_ts = publish_time
                if output.debug:
                    shaped_debug = output.debug.get("shaped", {}) if isinstance(output.debug, dict) else {}
                    if isinstance(shaped_debug, dict):
                        self._last_published_gloss = shaped_debug.get("proto_bn", "")

            self._last_error = output.error
            status: BrainStatus = output.status
            cache_hit = bool(output.debug.get("cache_hit")) if output.debug else False
            if cache_hit:
                self._cache_hits += 1
            
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
                    "dropped_results": self._dropped_results,
                    "cache_hits": self._cache_hits,
                    "last_tutor_topics_count": len(self._last_tutor_topics),
                },
                request_id=item.request_id,
            )

            if self._cfg.log_enabled:
                prompt_hash = None
                prompt_debug = output.debug.get("prompt") if output.debug else None
                if isinstance(prompt_debug, dict):
                    prompt_hash = prompt_debug.get("hash") or prompt_debug.get("metadata", {}).get("hash")
                event = build_log_event(
                    cfg=self._cfg,
                    request_id=item.request_id,
                    keywords=item.brain_input.keywords,
                    detected_emotion=item.brain_input.emotion,
                    resolved_emotion=output.resolved_emotion,
                    status=status,
                    latency_ms=output.latency_ms,
                    used_gemini=self._cfg.use_gemini,
                    cache_hit=cache_hit,
                    error=output.error,
                    prompt_hash=prompt_hash,
                    response_preview=output.response_bn[:120],
                )
                append_event(self._cfg.log_path, event, max_bytes=self._cfg.log_max_bytes)

        self._update_snapshot(status="idle", in_flight=False)

