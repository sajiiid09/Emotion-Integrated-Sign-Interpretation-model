# Brain Module – Phase 4 (Prompt Builder)

## Overview
Phase 4 introduces a modular prompt builder that converts a `ResolvedIntent`
(from Phase 3) into a structured prompt ready for an LLM provider (Gemini in
Phase 5). The builder produces a Bangla-only system message, a dynamic
instruction based on the resolved emotion, and a concise user payload
containing keywords and resolution context.

## System and Dynamic Instructions
- **System layer:** Defines the empathetic tutor persona for Bangladeshi Deaf
  students, constrains output length (under ~40 words, 2–3 sentences), and
  keeps responses Bangla-only without markdown or bullet lists.
- **Dynamic layer:** Tailors tone per `resolved_emotion`:
  - `question`: answer directly and simply.
  - `happy`: energetic, upbeat tone (can use `!`).
  - `sad`: gentle, validating, and supportive.
  - `negation`: treat as denial and ask a clarifying follow-up if needed.
  - `neutral`: standard helpful tutor tone.
- **Output constraints:** Reiterates Bangla-only, 40-word maximum, and short
  multi-sentence replies.

## User Payload
The payload includes joined keywords, the resolved emotion tag, a note when
face-tag tone differs from the resolved tone, and a brief instruction to help
in 2–3 sentences. It remains compact for provider-friendly prompts.

## Debug / CLI Visibility
- `BrainOutput.debug` now includes a prompt preview by default, plus full
  prompt text when `BRAIN_DEBUG` is true. The CLI can display the full prompt
  via `--show-prompt` or run in prompt-only mode.
- Rule traces remain visible under debug to show which contradiction rules
  fired.

## Examples
- `python -m brain.cli --tokens "মহাবিশ্ব question" --show-prompt`
  - Shows the question-specific dynamic instruction and output constraints.
- `python -m brain.cli --tokens "খারাপ sad" --prompt-only`
  - Prints only the prompt, revealing the supportive tone guidance.
- `python -m brain.cli --tokens "খুশি negation" --show-prompt`
  - Displays the negation dynamic rule and the resolved emotion in the payload.

## What’s Still Pending
- No Gemini/LLM invocation yet; Phase 5 will connect the prompt builder to the
  provider client.
- No UI/demo integration yet; CLI remains the manual verification path.
- No automated tests yet (planned after Phase 8).
