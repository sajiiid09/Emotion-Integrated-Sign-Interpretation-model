# Brain Module – Phase 2

Phase 2 introduces input normalization and intent parsing on top of the Phase 1
skeleton. The goal is to make the Brain module resilient to noisy upstream
signals while keeping deterministic, testable behavior.

## What is new
- Intent representation via `brain/intent.py` with flags and notes for debug
- Robust normalization in `brain/service.py`: punctuation trimming, unknown token
  removal, duplicate collapsing within a jitter window, and keyword truncation
- Emotion tag stripping when tags leak into keyword lists
- Expanded debug payloads (`intent`, `normalization`, `token_stats`)
- CLI flag `--show-debug` for quick inspection

## Normalization behaviors
- Unknown tokens (e.g., `???`, `unknown`, `null`, `none`) are dropped
- Punctuation on token edges is trimmed using `PUNCT_STRIP_CHARS`
- Consecutive jitter duplicates are collapsed using `DEDUPE_WINDOW`
- Emotion tags found inside keywords are removed and noted
- Keyword lists are truncated to `MAX_KEYWORDS`, keeping the most recent tokens

## Examples
- Input tokens: `"আমি আমি ভাত ??? neutral"`
  - Unknown token dropped, duplicate collapsed, intent emotion detected from
    last token; debug shows `had_unknowns=True`, `had_duplicates=True`.
- Input tokens: `"আমি neutral ভাত question"`
  - `neutral` stripped from keywords, `question` used as emotion.
- Input tokens: `"আমি, ভাত! খাওয়া? neutral"`
  - Punctuation stripped to `"আমি", "ভাত", "খাওয়া"` then parsed.

## Still intentionally missing
- Gemini or any external API calls
- Prompt building, contradiction rules, or async orchestration
- Integration with `demo/`, `train/`, or other packages
- Automated tests (planned after later phases)

## Running the CLI
Examples:

```bash
python -m brain.cli --keywords "আমি আমি ভাত ???" --emotion neutral --show-debug
python -m brain.cli --tokens "আমি, ভাত! খাওয়া? neutral" --show-debug
```

Optional environment variables remain the same as Phase 1 (see
`docs/brain_phase1.md`).
