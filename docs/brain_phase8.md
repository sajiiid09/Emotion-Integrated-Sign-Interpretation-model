# Brain Phase 8: Quality, Logging, and Demo Polish

## What changed
- Unified Bangla response post-processing to strip markdown, enforce 2–3 sentence, short replies, and cap word count.
- Added optional response caching, structured JSONL logging, and prompt hashing for reproducibility.
- Tightened prompt builder instructions and Gemini client cleanup for stable Bangla output.
- Enhanced realtime demo HUD with Gemini/API indicators, presentation mode, help overlay, and Bangla sentence/prompt visibility.
- Added a lightweight smoke test suite to verify imports and core flows.

## Runtime controls
- **Env vars:** `BRAIN_USE_GEMINI`, `GEMINI_API_KEY/GOOGLE_API_KEY`, `BRAIN_CACHE_ENABLED`, `BRAIN_CACHE_TTL_S`, `BRAIN_LOG_ENABLED`, `BRAIN_LOG_PATH`, `BRAIN_LOG_MAX_BYTES`, `BRAIN_DEBOUNCE_MS`, `BRAIN_COOLDOWN_MS`.
- **Demo keys:** `g` toggle Gemini, `c` clear buffer, `m` presentation mode, `h` help overlay, `p` prompt preview, `q/esc` quit.

## Logging and cache
- Logs live in `logs/brain_events.jsonl` (rotates by size). Each event captures request_id, intent, resolved emotion, latency, cache hit, prompt hash, and preview.
- Cache prevents repeated Gemini calls for the same prompt signature with a small TTL.

## Smoke tests
Run `pytest -q` to execute the minimal sanity checks in `tests/test_smoke.py`.

## Demo tips
- Place a Bangla font (e.g., `demo/kalpurush.ttf`) for crisp rendering; HUD shows a warning if missing.
- Stub mode works offline; Gemini mode requires an API key. Presentation mode hides debug counters for clean screens.
