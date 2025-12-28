# Brain Module Phase 3 — Contradiction Resolution & Rule Engine

Phase 3 introduces a deterministic rule engine that resolves contradictions between face emotion tags and keyword-level emotion cues. The goal is to produce a stable `ResolvedIntent` that later phases (prompt builder, Gemini) can consume without changing interfaces.

## What changed
- Added `ResolvedIntent` to capture post-rule resolution intent state.
- Added `brain.rules.resolve_emotion` with traceable rule firing for debugging and telemetry.
- Stub responses now use `resolved_emotion` instead of the raw detected tag.
- CLI `--show-debug` now surfaces both the parsed intent and resolved intent alongside rule traces.

## Current rules (deterministic)
1. **Negation overrides happy cues**: If the face tag is `negation` and keywords include a happy cue (e.g., `খুশি`), the emotion remains `negation` and the rule is traced.
2. **Negation cancels negative keyword emotions**: If the face tag is `negation` and keywords include sorrow/anger/bad/sick cues, the resolved emotion becomes `neutral` with `negated_state` flagged.
3. **Question priority**: If the face tag is `question`, it stays `question` even when keyword emotion cues exist.
4. **Keyword-only emotion inference (neutral tag only)**: When the face tag is `neutral`, `খুশি` infers `happy`, while `দুঃখ`, `খারাপ`, or `অসুস্থ` infer `sad`.

## Normalization recap
The Phase 2 normalization still applies: punctuation trimming, unknown-token removal, tag stripping from keywords, jitter deduplication, and truncation to the maximum keyword count.

## Examples
- `python -m brain.cli --tokens "দুঃখ negation" --show-debug`
  - Resolved emotion: `neutral`
  - Rule trace includes `negation_cancels_negative_state`
- `python -m brain.cli --tokens "খুশি neutral" --show-debug`
  - Resolved emotion: `happy`
  - Rule trace includes `keyword_emotion_inference`
- `python -m brain.cli --tokens "খুশি question" --show-debug`
  - Resolved emotion: `question`
  - Rule trace includes `question_priority` (if a change occurs)
- `python -m brain.cli --tokens "neutral" --show-debug`
  - Resolved emotion: `neutral`, default Bangla response returned.

## Still not implemented
- Gemini API calls and prompt construction
- Demo integration or UI wiring
- Async processing
- Automated tests (will follow in later phases)
