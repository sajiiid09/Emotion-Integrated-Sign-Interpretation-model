# Brain Phase 6: Async Executor for Realtime

## Overview
Phase 6 introduces `BrainExecutor`, a background worker that offloads the full Brain
pipeline (intent parsing → resolution → prompt building → Gemini/stub response) so
the realtime webcam loop never blocks. The executor provides lightweight snapshots
with status, last output, and debug telemetry for HUD updates.

## Core behaviors
- **Latest-wins:** every submit increments a request_id; stale results are dropped
  if a newer request arrives before publish.
- **Debounce:** rapid submissions within `BRAIN_DEBOUNCE_MS` are coalesced; only
the most recent request is enqueued.
- **Cooldown:** after publishing a Gemini-backed response, the worker waits
  `BRAIN_COOLDOWN_MS` before starting another Gemini call (questions bypass the
  cooldown).
- **Stub + Gemini modes:** executor works in stub-only mode (Gemini disabled) and
  in Gemini mode using the Phase 5 client wrapper.
- **Status transitions:** idle → listening (on submit) → thinking (worker
  processing) → ready/error (on publish). Snapshots include queue size and last
  keywords for HUD hints.

## Configuration
New optional env vars (safe defaults applied):
- `BRAIN_DEBOUNCE_MS` (default 350)
- `BRAIN_COOLDOWN_MS` (default 1200)
- `BRAIN_QUEUE_MAXSIZE` (default 2)

Existing Gemini settings still apply (`BRAIN_USE_GEMINI`, `GEMINI_API_KEY`/`GOOGLE_API_KEY`,
`BRAIN_TIMEOUT_S`, etc.).

## CLI realtime simulation
Use the CLI to validate executor behavior without the demo loop:

```bash
python -m brain.cli --simulate-realtime \
  --sequence "আমি ভাত খাওয়া neutral|আমি ভাত খাওয়া neutral|আমি ভাত খাওয়া question" \
  --no-gemini
```

With Gemini enabled (requires API key):
```bash
export GEMINI_API_KEY=...  # or GOOGLE_API_KEY
python -m brain.cli --simulate-realtime \
  --sequence "মহাবিশ্ব question|মহাবিশ্ব question|গণিত question" \
  --use-gemini
```

## Integration notes
- `BrainExecutor.poll_latest()`/`get_status()` return `ExecutorSnapshot` for HUDs.
- `submit_tokens` mirrors the Phase 2 token semantics; emotion tags on the end of
  the token list are honored.
- Phase 7 will wire this executor into the demo UI; no demo changes are made here.
