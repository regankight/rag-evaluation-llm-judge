# Security notes

Checked 2026-10-06, before the repository was published.

## Secrets
- The OpenAI key lives only in `.env`, which is git-ignored and has never been committed (checked across the full git history). Use a **restricted** key with model access only, and a project budget cap.
- `.env.example` contains placeholders only.
- Committed run artifacts were scanned for usernames, local paths and API keys before being added.

## Dependencies
`pip-audit` reports two advisories with **no fixed version available** (checked 2026-10-06). Neither code path is used here:

| Package | Advisory | Why it doesn't apply |
|---|---|---|
| `ragas` 0.4.3 | CVE-2026-6587 / GHSA-95ww-475f-pr4f: server-side request forgery via `retrieved_contexts` in the **multimodal** faithfulness metric | Only text `Faithfulness`, `ContextRecall` and `AnswerAccuracy` are used. Contexts come from the committed TyDi corpus, never from user input. |
| `diskcache` 5.6.3 (pulled in by RAGAS) | CVE-2025-69872 / GHSA-w8v5-vhqr-4h9v: pickle deserialization from a writable cache directory | RAGAS's disk cache is never enabled. |

If this code were exposed to untrusted input, both should be re-checked first.

## CI
- The workflow runs with `permissions: contents: read` and `persist-credentials: false`.
- Third-party actions are pinned to full commit SHAs, not movable tags.
- CI needs no secrets: tests and `make reproduce` run fully offline.

## Local review page
`make_review.py` embeds dataset text in a `<script>` block; `</` is escaped so passage text can't close the block and inject HTML (tested).
