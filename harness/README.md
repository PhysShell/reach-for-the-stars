# harness

**Persistent browser-session capsule for ephemeral CI runners.**

A small Rust binary that restores an authenticated browser session inside a
fresh ephemeral runner via an encrypted, signed, content-addressed checkpoint,
runs work against the live session, and saves the updated checkpoint back —
so a once-logged-in browser context survives across runs without persisting
raw credentials anywhere durable.

Status: research project. Compiles, smoke-tested offline; the chromium-bound
end-to-end path is exercised manually from `nix develop` and via the
`scripts/smoke.sh` wrapper.

---

## Why this exists

Cookie-based sessions for sites that don't expose long-lived service tokens
(OAuth refresh, machine accounts) typically last weeks to months as long as
the cookie jar lives somewhere and gets revalidated. Throwing that jar at
a one-shot CI runner means either:

1. Re-logging in every run (interactive, won't work in cron).
2. Stashing the live cookie in a GitHub Secret and re-injecting (fragile —
   secrets are long-lived plaintext-on-disk inside the runner; cookie
   rotation means you keep bumping the secret).
3. Renting a persistent VM with the browser pre-logged-in (works, but
   adds a managed host you didn't want, plus a fingerprint surface).

This harness picks a fourth path: keep the runner ephemeral, treat the
**session state** as the durable artifact, encrypt+sign it, store it in
S3-compatible / git-native backends, and have the runner restore from
that artifact each run.

---

## Layered architecture

The codebase has three conceptual layers. They are **coupled by design for now** —
abstracting them prematurely would introduce traits with one implementor and
add maintenance cost for no current benefit. The layering is documented here
so the seams are visible when the second use case appears.

```
┌─────────────────────────────────────────────┐
│ Layer 3:  workload                          │  ← "do the work"
│   currently a tracing::info! placeholder    │
│   in src/commands/run.rs                    │
├─────────────────────────────────────────────┤
│ Layer 2:  browser session                   │  ← "drive Chromium"
│   src/browser/cdp.rs (chromiumoxide)        │
│   src/browser/canary.rs (JSON-endpoint probe)│
├─────────────────────────────────────────────┤
│ Layer 1:  session-state lifecycle           │  ← "carry state through time"
│   restore → canary → save                   │
│   src/commands/run.rs + crypto + stores     │
│   + lock + health                           │
└─────────────────────────────────────────────┘
```

### When to actually decouple

- **Layer 3 first**: when there is a second workload (e.g. export from
  different service). Introduce `trait Workload`, pass `&dyn Workload` into
  `run::run`. Existing `tracing::info!("workload placeholder")` line becomes
  `workload.execute(&session).await?`.
- **Layer 2 second**: when there is a non-browser session target (IMAP,
  proprietary RPC). Introduce `trait Session<State>`. `BrowserSession` becomes
  one impl over `StorageState`.
- **Layer 1 last**: only if the storage shape becomes wrong (multi-tenant,
  per-cookie versioning). The current `LatestPointer` + append-only snapshots
  layout handles every case we have visibility into.

Until those triggers fire, the layers stay collapsed into the linear flow in
`src/commands/run.rs`.

---

## How it works

```
seed (one-time, interactive):
  open headed Chromium → manual login → Storage.getCookies
    → zstd → age encrypt → ed25519 sign(ciphertext)
    → put snapshot + sig + latest.json to primary store (CAS-guarded)
    → fan out to mirrors

run (daily cron):
  acquire CAS lock on primary
    ↓
  GET latest.json (capture its etag for the closing CAS write)
    ↓
  GET snapshot + verify_digest(pointer.sha256)
    ↓
  GET signature + ed25519 verify(ciphertext)    ← verify BEFORE decrypt
    ↓
  age decrypt → zstd decompress → StorageState
    ↓
  launch headless Chromium → Storage.setCookies
    ↓
  HTTP canary against an authenticated JSON endpoint
    │     │
    │     └── fail → bail, latest.json untouched, health-log "canary_failed"
    ↓
  workload (currently a placeholder)
    ↓
  Storage.getCookies → refuse if empty
    ↓
  compress → encrypt → sign(new ciphertext)
    ↓
  ensure_monotonic(prev_pointer, new_pointer)
    ↓
  PUT new snapshot + sig (append-only)
    ↓
  put_if_unmodified(latest.json, If-Match=pointer_etag)
    ↓
  fan out snapshot + sig + latest.json to mirrors
  release lock; write health-log record

verify (no-browser health probe):
  GET latest.json → verify_digest → ed25519 verify → age decrypt
    → decompress → "OK version=... cookies=N"
```

Every cross-step transition has either a CAS check, a precondition, or a
cryptographic verification. Failures abandon the run without touching the
durable pointer — the previous good snapshot stays authoritative.

---

## Quick start

Inside `nix develop` (the dev shell provides `cargo`, `age`, `gh`, `jq`,
pinned ungoogled-chromium, and MinIO for smoke tests):

```bash
# One-time setup: generate keys, seal into GitHub Secrets, commit public bits.
./scripts/bootstrap.sh

# Configure: copy and edit canary endpoint, primary store, etc.
cp config.example.toml config.toml
$EDITOR config.toml

# First login: headed browser opens, you log in, press Enter when done.
cargo run --release -- seed

# Verify the snapshot is round-trippable end-to-end (no browser).
cargo run --release -- verify

# Same thing GitHub Actions runs daily.
cargo run --release -- run
```

---

## Commands

| Command           | Browser? | Network writes? | Purpose                                                |
|-------------------|----------|-----------------|--------------------------------------------------------|
| `gen-keys`        | No       | No              | Generate ed25519 keypair; `--json` for scripting.      |
| `seed`            | Headed   | Yes             | Interactive first-time login; uploads initial state.   |
| `run`             | Headless | Yes             | Restore → canary → workload → save; daily lifecycle.   |
| `verify`          | No       | Read-only       | Decrypt latest snapshot; one-line OK or non-zero exit. |

---

## Configuration

`config.toml` (TOML, validated at load):

- `browser.chromium_bin` — path to a Chromium binary (typically `/nix/store/.../bin/chromium`).
- `browser.user_data_dir` — scratch profile directory.
- `browser.headless` — `false` for seed, `true` for run.
- `seed.login_url` — first page Chromium opens on seed.
- `canary.url` — authenticated JSON endpoint that returns a stable
  identity field. Find it in DevTools → Network on the logged-in page.
- `canary.field` — JSON Pointer (RFC 6901) to a value that must equal
  `canary.expected_value` byte-for-byte.
- `crypto.recipient_file` / `crypto.verify_pubkey_file` — public-only,
  commit alongside the code.
- `lock.ttl_seconds` — single-writer lock TTL on the primary store.
- `[[stores]]` — first entry is the canonical source of truth, rest are
  best-effort mirrors. Mixing `s3`, `git_branch`, and `github_release`
  in any order is supported.

Credentials (`HARNESS_AGE_IDENTITY`, `HARNESS_SIGN_SECRET`, plus per-store
`access_key_env` / `token_env`) come from environment variables and are
scrubbed from the process environment immediately on read so child
processes (Chromium) cannot inherit them.

See `config.example.toml` for a fully populated reference.

---

## Security model

**Threat: GitHub Actions runner compromise during a single run.**
The runner sees the plaintext snapshot for the duration of the job. This
is the irreducible exposure window. Mitigations: pin the runner image,
pin actions by commit SHA, no `${{ inputs.* }}` substitution in shell.

**Threat: GitHub Secret leak (long-lived key compromise).**
An attacker with `HARNESS_AGE_IDENTITY` can decrypt every past snapshot
in storage. Rotation: regenerate the age keypair, re-encrypt the latest
snapshot with the new recipient, push to primary, rotate the secret.
Old snapshots remain decryptable with the old key — by design (audit
trail). The `verify_pubkey_file` is public and never sensitive.

**Threat: Storage backend compromise (R2 token leaked, repo write
access leaked).**
An attacker can replace ciphertext blobs. The ed25519 signature catches
this: `run` and `verify` reject anything that doesn't verify against
the in-repo pubkey, so an unauthorised writer can corrupt the store
but cannot forge a snapshot that the harness will trust.

**Threat: Snapshot rollback (downgrade attack).**
`ensure_monotonic(prev_pointer, new_pointer)` rejects any new pointer
whose `created_at` is not strictly newer than the one we just read.
A storage attacker who restores an old (decryptable) snapshot still
gets caught.

**Threat: Concurrent writers (two cron runs, manual + automatic).**
CAS lock on the primary serializes writers. The closing
`put_if_unmodified(latest.json, If-Match=pointer_etag)` is independent
belt-and-suspenders — even if the lock is broken, the etag check
catches concurrent updates.

**Out of scope: site-side detection of automation.**
This harness preserves session state, it does not hide automation.
A site detecting `navigator.webdriver` or a datacenter IP is a layer
above. See "Optional: Tailscale" below.

---

## Design notes / known tradeoffs

**CDP rather than Playwright.** We don't use Playwright at runtime — the
Rust code talks Chrome DevTools Protocol directly via `chromiumoxide`.
Storage state is serialised in a Playwright-compatible JSON shape
(`cookies` + `origins`) so the on-disk format is interoperable with
tooling people already trust, but there is no Node.js dependency in
the runtime path.

**Alternative browser binaries (future).** `browser.chromium_bin` is
already a config knob, so swapping for a hardened/stealth Chromium fork
(e.g. CloakBrowser) is a config change today, modulo Nix packaging
caveats (some forks download their own binary on first launch, which
breaks hermeticity). The cleaner future seam is `browser.mode = "launch"
| "cdp"` so a Docker-hosted browser server can be addressed via CDP
endpoint instead of a launched child process. Not implemented yet —
there's no second backend that needs it.

**Workload is a placeholder.** `commands::run::run` has a single
`tracing::info!` line where the actual work would go. The layered
architecture section above describes the seam; until a second workload
needs to ride this pipeline, the placeholder stays. Forking the binary
and putting work directly in that function is the supported path right
now.

**Health log is best-effort.** Each run appends one JSON record under
`health/` on the primary store. Failures to serialise or upload the
record are logged at warn and swallowed — the audit trail must never
be the reason the actual snapshot pipeline fails.

**Optional: Tailscale exit node for residential IP.** GitHub Actions
runners come from Azure datacenter IP ranges. For sites with strict
"unusual sign-in" heuristics (Google SSO is a common one), the
combination of datacenter IP + headless Chrome triggers reauth
challenges. Routing the runner's egress through a home-based Tailscale
exit node makes the request appear from a familiar residential IP.
This is a workflow-level concern (`tailscale up --exit-node=...`
before `run`), not a harness concern.

**`age` + `ed25519` rather than one combined primitive.** Encryption
and signing use disjoint keys on purpose. A compromised decrypt key
(`HARNESS_AGE_IDENTITY`) does not let an attacker forge new snapshots;
a compromised signing key does not let an attacker decrypt past
snapshots. Both keys would need to leak for either capability.

**Verification happens before decryption in `run`/`verify`.** A tampered
ciphertext never enters the age decryptor, only the ed25519 verifier.
Defence in depth against bugs in the decryption code path.

---

## Storage backends

Three implementations of `trait SnapshotStore`:

- **`s3`** — any S3-compatible object store. Cloudflare R2, Backblaze B2,
  MinIO, AWS S3. Full CAS via `If-Match` / `If-None-Match`. Primary or mirror.
- **`git_branch`** — any git remote with push access. CAS is `git push
  --force-with-lease=<branch>:<expected-sha>`, which is git's native
  test-and-set at the ref level. Zero-vendor-lock fallback: needs only a
  `GITHUB_TOKEN`, no S3-style credentials. Primary or mirror.
- **`github_release`** — GitHub Releases. Mirror-only: no atomic CAS at
  the asset level. Routing collapses our key hierarchy onto a fixed set
  of releases (`harness-snapshots`, `harness-latest`, `harness-health`)
  so the repo's release page doesn't get spammed with one entry per run.

The first `[[stores]]` entry in config is the canonical source of truth.
The rest receive best-effort fanout writes; a mirror falling behind or
being misconfigured warns but does not fail the run.

---

## Development

```bash
nix develop                            # provides chromium, minio, age, gh, jq, cargo
cd harness

cargo build --release
cargo clippy --all-targets -- -D warnings
cargo test                             # 4 smoke tests; 2 skip without env

scripts/smoke.sh                       # spawns MinIO, picks up CHROMIUM_BIN, runs all
```

Smoke tests:

- `crypto_archive_roundtrip` — no external deps; runs on every `cargo test`.
- `s3_cas_semantics` — needs `SMOKE_S3_ENDPOINT` (MinIO 2024-09+).
- `git_branch_cas_semantics` — needs `git` on PATH; uses a tempdir bare repo.
- `full_cycle_with_chromium` — needs `CHROMIUM_BIN`; wiremock + headless
  Chromium round-trip + canary inheritance.

---

## What this is not

- Not a stealth browser. It does not hide automation; it preserves auth.
- Not a credential manager. There are no passwords stored anywhere; only
  the resulting session state.
- Not a multi-tenant service. The layout assumes one canonical pointer
  per backend; concurrent runs against the same primary are serialised
  by the CAS lock, not partitioned.
- Not production-tested. The mechanism is real, the threat-model is
  thought through, the cryptographic primitives are well-chosen — but
  this is a research project, not battle-tested infrastructure.
