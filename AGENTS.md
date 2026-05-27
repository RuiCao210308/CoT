# Codex project instructions

## RTK usage

Use RTK for noisy shell commands when possible to reduce context noise.

Prefer:
- `rtk git status` instead of raw `git status`
- `rtk git diff` instead of raw `git diff`
- `rtk rg "<pattern>" <path>` instead of raw `rg`
- `rtk find <path> ...` instead of raw `find`

If RTK output is too compact or hides necessary details, rerun the original raw command.

## Project constraints

Do not train models unless explicitly asked.
Do not call Qwen generate unless explicitly asked.
Do not change JSONL schema unless explicitly asked.
Do not overwrite existing experiment results.
Do not automatically run git commit.
