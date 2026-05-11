# Ephemeral persistent-agent harness

This crate is a standalone utility for running a browser automation job on an
ephemeral runner while keeping just enough authenticated state to behave like a
long-lived agent. A better name for the first layer is a **persistent session
capsule**: the VM/container is disposable, but the signed and encrypted browser
session snapshot is durable.

## Architecture layers

The design is intentionally layered rather than one tightly-coupled pipeline:

1. **Session-capsule lifecycle** — `seed`, `run`, and `verify` coordinate a
   durable snapshot pointer, single-writer lock, age encryption, ed25519
   signatures, and mirror fan-out. This layer should not know what business task
   the agent performs.
2. **Browser/session adapter** — Chromium is only responsible for importing and
   exporting Playwright-compatible storage state. The archive format is explicit
   in `latest.json`, so a future `user_data_dir` snapshot can be introduced
   without changing store semantics.
3. **Workload/export layer** — the current `run` command has a placeholder for
   the actual job (for example, exporting chats). That code should call into a
   small trait or command module and should not read/write snapshot objects
   directly.

Keeping these boundaries lets us test crypto/storage without Chromium, test the
browser adapter without real object storage, and run a canary before overwriting
state.

## Storage model

The primary store is authoritative and must support compare-and-swap writes for
`lock.json` and `latest.json`. Snapshot blobs are append-only. Mirrors are
best-effort copies of the snapshot, detached signature, and latest pointer; a
mirror may lag without failing the primary run.

Supported store types:

- `s3`: canonical choice for R2, MinIO, Backblaze B2 S3, or another
  S3-compatible backend with `If-Match` / `If-None-Match` support.
- `git_branch`: a zero-vendor-lock fallback that stores state on a dedicated
  branch and uses `git push --force-with-lease` as the CAS primitive.
- `github_release`: sketched as mirror-only, but not implemented yet.

## Optional Tailscale

Tailscale is not required for the snapshot mechanism. Add it only when the
workload needs private network reachability: internal web UIs, databases,
admin-only APIs, or an exit node with a stable egress identity.

Recommended placement:

1. Start Tailscale in the runner before `harness run`.
2. Authenticate with an ephemeral, tagged auth key scoped to the smallest ACL
   surface needed by the workload.
3. Keep snapshot storage credentials separate from Tailscale credentials.
4. Prefer userspace networking when available in CI; use kernel `tun` only when
   the runner supports it.

Minimal CI shape:

```bash
sudo tailscaled --state=mem: --socket=/tmp/tailscaled.sock &
tailscale --socket=/tmp/tailscaled.sock up \
  --auth-key="$TAILSCALE_AUTHKEY" \
  --hostname="agent-${GITHUB_RUN_ID:-local}" \
  --accept-routes=false
cargo run --manifest-path harness/Cargo.toml -- run --config harness/config.toml
tailscale --socket=/tmp/tailscaled.sock logout || true
```

Do not put Tailscale into the browser/session adapter. Treat it as runner
network plumbing around the workload layer; that keeps the harness useful in
plain public-internet jobs and avoids coupling storage, Chromium, and VPN state.

## Review notes / immediate improvements

- The mirror fan-out must include detached signature objects as well as snapshot
  blobs and `latest.json`; otherwise a mirror can receive a pointer to a missing
  signature.
- `verify` should remain a no-browser health check for the storage/crypto
  boundary.
- Keep workload code outside the store and crypto modules. If chat export grows,
  introduce a `Workload` trait or a separate `commands/export_chats.rs` module
  that receives an already-authenticated `BrowserSession`.
- Avoid making Tailscale a Rust dependency unless the program itself has to
  manage tailnet state. For CI, shell-level setup is simpler and less coupled.
