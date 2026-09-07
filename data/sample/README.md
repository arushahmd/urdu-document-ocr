# Safe synthetic Urdu fixtures

This directory contains 24 current-project synthetic line images and 10 synthetic pages.
The Urdu phrases are project-authored neutral fixture content; no historical text, scan,
annotation, or private data is present. `manifest.jsonl` uses the Phase 7 schema. The split
is document-grouped, and `synthetic-fixture-vocabulary.json` is a train-only **synthetic
fixture vocabulary**, not a historical or production vocabulary.

Regenerate from the repository root:

```console
python scripts/generate_sample_data.py --output data/sample --overwrite
```

Existing unrelated files are rejected and no network access occurs.
See `generation-manifest.json`, `text-provenance.json`, `provenance.json`, and
`artifact-manifest.json` for frozen identities and hashes.
