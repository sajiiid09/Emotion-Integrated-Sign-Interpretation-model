# Brain Module – Phase 1

Phase 1 establishes the foundational structure for the Brain system. It adds a
self-contained Python package (`brain/`) with stable data contracts, safe
configuration handling, and a deterministic stub responder for manual testing.

## What is included
- Stable data contracts in `brain/types.py` using `@dataclass(frozen=True)`
- Module-wide constants in `brain/constants.py`
- Safe environment-based configuration loading via `brain/config.py`
- Core stub responder and helpers in `brain/service.py`
- Minimal CLI for manual verification in `brain/cli.py`
- Public exports in `brain/__init__.py`

## Not implemented yet
- Gemini or any external API calls
- Prompt building or rule-engine logic beyond the basic stub
- Async execution
- Integration with `demo/`, `train/`, `preprocess/`, or other packages
- Automated tests (planned after Phase 8)

## Running the CLI
Examples:

```bash
python -m brain.cli --keywords "আমি ভাত খাওয়া" --emotion neutral
python -m brain.cli --tokens "আমি ভাত খাওয়া neutral"
```

## Configuration via Environment
Optional environment variables with safe defaults:

- `BRAIN_MODEL_NAME` (default: `gemini-1.5-flash`)
- `BRAIN_TIMEOUT_S` (default: `8.0` seconds)
- `BRAIN_DEBUG` (default: `False`)
- `BRAIN_MAX_WORDS` (default: `40`)

Invalid values automatically fall back to defaults to keep the module robust.
