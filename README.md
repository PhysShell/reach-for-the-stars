# reach-for-the-stars

Small Rust CLI that exports a GitHub user's stargazed repositories to a JSON
envelope. Designed for a daily GitHub Action: cheap incremental scans most of
the time, periodic full reconciles that catch unstars, and an envelope schema
that records its own sync state.

It does three things and nothing else:

1. Fetch starred repos from the GitHub REST API (`star+json`, paginated).
2. Merge them with the previous envelope using a hybrid sync model.
3. Write the new envelope as pretty JSON.

Encryption and remote upload (R2/S3) are deliberate non-goals at this stage.

## Usage

```bash
# Build (one-off; CI builds release)
cargo build --release

# Export, picking sync mode automatically based on the previous envelope
GITHUB_TOKEN=ghp_xxx ./target/release/reach-stars export \
  --username PhysShell \
  --output assets/data/stars.json \
  --sync-mode auto
```

`reach-stars export --help` lists every flag.

### Sync modes

| Mode          | Cost       | Catches unstars? | When to use                        |
| ------------- | ---------- | ---------------- | ---------------------------------- |
| `incremental` | 1 page-ish | no               | most runs                          |
| `full`        | every page | yes              | first run, periodic reconcile      |
| `auto`        | hybrid     | sometimes        | default; full once a day, else inc |

`auto` runs `full` when there's no previous envelope, or when the recorded
`last_full_reconcile_at` is older than `--full-reconcile-interval-hours`
(default 24).

### Envelope schema (v2)

```jsonc
{
  "schema_version": 2,
  "subject": { "kind": "user_starred_repositories", "username": "PhysShell" },
  "exported_at": "2026-05-09T17:40:00Z",
  "sync": {
    "mode": "incremental",
    "last_full_reconcile_at": "2026-05-08T00:00:00Z",
    "watermark_starred_at": "2026-05-09T12:34:56Z",
    "watermark_repo_id": 123456789
  },
  "stats": { "active_count": 120, "tombstone_count": 4, "total_count": 124 },
  "data_hash": "sha256:...",       // hash of the semantic data only
  "stars": [
    {
      "repo_id": 123456789,
      "full_name": "owner/repo",
      "html_url": "https://github.com/owner/repo",
      "description": "Something useful",
      "language": "Rust",
      "starred_at": "2026-05-09T12:34:56Z",
      "first_seen_at": "2026-05-09T12:35:10Z",
      "last_seen_at":  "2026-05-09T12:35:10Z",
      "unstarred_at": null,
      "status": "active"   // or "unstarred"
    }
  ]
}
```

`data_hash` deliberately excludes `exported_at`, `first_seen_at`, and
`last_seen_at`, so it stays stable when the underlying star set is stable.

Unstars are tombstones (`status: "unstarred"`, plus an `unstarred_at`
timestamp) — they are never deleted, so historical snapshots remain
auditable.

## GitHub Action

Two workflows:

- `.github/workflows/ci.yml` — on every PR and push to `main`. Runs
  `cargo fmt --check`, `cargo clippy -D warnings`, `cargo test`,
  `cargo doc -D warnings`, `cargo audit`, and `cargo deny check`. No
  secrets, no upload.
- `.github/workflows/reach-for-the-stars.yml` — on schedule (every 4h) and
  manual dispatch. Builds the release binary, runs `reach-stars export`,
  commits `assets/data/stars.json` if it changed.

The export workflow is gated on `github.repository == 'PhysShell/reach-for-the-stars'`
so it doesn't run on forks.

## Development

```bash
cargo fmt --all
cargo clippy --all-targets --all-features -- -D warnings
cargo test
```

Strictness knobs (in `Cargo.toml`):

- `unsafe_code = "forbid"`
- `clippy::all = "deny"`, `clippy::pedantic = "warn"`, `clippy::cargo = "warn"`
- `unwrap_used`, `expect_used`, `panic`, `todo`, `dbg_macro`, `print_stderr`
  are denied in non-test code; tests opt out per-module.

Supply-chain checks (`cargo audit`, `cargo deny`) run in CI. `deny.toml`
restricts allowed licenses and the source registry.

## Migrating from the previous Python scraper

The old script wrote a flat list of strings (`stars: "1,234"`, etc) scraped
from HTML. The new envelope is a different schema. On first run the Rust
exporter sees the absent (or unparseable) file and runs a full reconcile to
build a fresh v2 envelope.

## License

MIT — see `LICENSE`.

Originally based on a small project by @taylow:
<https://taylo.dev/posts/adding-github-stars-to-my-site>.
