# Brain Phase 5: Gemini Client Wrapper + Fallback + Retries

Phase 5 introduces a defensive Gemini client so future phases can call the LLM safely without coupling to UI/demo code.

## What was added
- `brain/gemini_client.py` wraps the Google GenAI SDK with retries, timeouts, fallback responses, and optional streaming.
- `BrainConfig` now loads Gemini-related options (API key, retries, temperature, streaming, etc.).
- `brain/service.respond` can call Gemini when enabled, otherwise it preserves the deterministic stub response.
- CLI flags allow forcing/stubbing Gemini use and showing prompts or streaming output.

## Required and optional environment variables
- **API key (required for live calls):** `GEMINI_API_KEY` or `GOOGLE_API_KEY`
- **Optional toggles and tuning:**
  - `BRAIN_USE_GEMINI` (bool)
  - `BRAIN_MODEL_NAME`
  - `BRAIN_TIMEOUT_S`
  - `BRAIN_TEMPERATURE`
  - `BRAIN_MAX_OUTPUT_TOKENS`
  - `BRAIN_RETRIES`
  - `BRAIN_STREAMING`

## CLI examples
- Stub mode (no Gemini):
  ```bash
  python -m brain.cli --tokens "মহাবিশ্ব question" --no-gemini
  ```
- Gemini generation with prompt preview:
  ```bash
  export GEMINI_API_KEY="your-key"
  python -m brain.cli --tokens "মহাবিশ্ব question" --use-gemini --show-prompt
  ```
- Streaming (best effort):
  ```bash
  python -m brain.cli --tokens "গণিত question" --use-gemini --stream
  ```

## Notes
- If the SDK or API key is missing, the client returns the Bangla fallback string and surfaces the error in debug without crashing callers.
- Responses are constrained to short Bangla answers (~40 words) with empathetic tone aligned to the resolved emotion.
