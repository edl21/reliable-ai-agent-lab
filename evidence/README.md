# Tracked evidence bundle

Sanitized JSONL examples for the reliability capability suite. Regenerate with:

```bash
env -u OPENAI_API_KEY -u OPENAI_BASE_URL -u MODEL_NAME \
  .venv/bin/python -m checks.run_all
.venv/bin/python scripts/refresh_evidence_bundle.py
```

Credentials and absolute repository paths are redacted. Runtime `traces/`
remain ignored by Git.
