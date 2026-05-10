#!/usr/bin/env bash
# Bootstrap: generate keypairs and seal them into GitHub Secrets.
# Run from inside `nix develop`.
set -euo pipefail
umask 077

REPO_ROOT="$(git rev-parse --show-toplevel)"
HARNESS_DIR="$REPO_ROOT/harness"
SECRETS_DIR="$HARNESS_DIR/.secrets"

require() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "missing dependency: $1 (run from inside 'nix develop')" >&2
    exit 1
  }
}
require age-keygen
require minisign
require gh
require xxd

mkdir -p "$SECRETS_DIR"
cd "$HARNESS_DIR"

if [[ -f "$SECRETS_DIR/age-key.txt" ]]; then
  echo "$SECRETS_DIR/age-key.txt already exists; refusing to overwrite." >&2
  echo "If you really want to rotate, move it aside manually first." >&2
  exit 1
fi

echo "==> Generating age keypair"
age-keygen -o "$SECRETS_DIR/age-key.txt"
age-keygen -y "$SECRETS_DIR/age-key.txt" > "$HARNESS_DIR/age-recipient.txt"
chmod 600 "$SECRETS_DIR/age-key.txt"

echo "==> Generating ed25519 signing keypair (raw hex, not minisign on-disk format)"
# Two 32-byte halves of an ed25519 keypair, hex-encoded. Matches what
# crypto::sign::parse_signing_key / parse_verifying_key expect.
SIGN_SECRET=$(head -c 32 /dev/urandom | xxd -p -c 64)
echo "$SIGN_SECRET" > "$SECRETS_DIR/sign-secret.hex"
chmod 600 "$SECRETS_DIR/sign-secret.hex"

# Derive pubkey via the harness binary (so the format matches exactly).
if cargo build --release --quiet 2>/dev/null; then
  ./target/release/harness gen-keys >/dev/null 2>&1 || true
fi

echo
echo "==> Public artefacts (commit these):"
echo "    harness/age-recipient.txt"
echo "    harness/sign-pubkey.hex   (derive: see TODO below)"
echo
echo "==> Sealing private artefacts into GitHub Secrets"

read -rp "GitHub repo (owner/name): " GH_REPO
read -rp "R2 Access Key ID:        " R2_AK
read -rsp "R2 Secret Access Key:    " R2_SK; echo
read -rp "R2 Endpoint URL:         " R2_ENDPOINT
read -rp "R2 Bucket:               " R2_BUCKET

gh secret set HARNESS_AGE_IDENTITY  --repo "$GH_REPO" < "$SECRETS_DIR/age-key.txt"
gh secret set HARNESS_SIGN_SECRET   --repo "$GH_REPO" < "$SECRETS_DIR/sign-secret.hex"
gh secret set HARNESS_R2_ACCESS_KEY --repo "$GH_REPO" -b "$R2_AK"
gh secret set HARNESS_R2_SECRET_KEY --repo "$GH_REPO" -b "$R2_SK"
gh secret set HARNESS_R2_ENDPOINT   --repo "$GH_REPO" -b "$R2_ENDPOINT"
gh secret set HARNESS_R2_BUCKET     --repo "$GH_REPO" -b "$R2_BUCKET"

echo
echo "Done."
echo
echo "Next steps:"
echo "  1. Edit harness/config.toml (copy from config.example.toml)."
echo "  2. cargo run --release -- seed     # interactive login, headed Chromium"
echo "  3. Verify: cargo run --release -- verify"
echo "  4. shred -u $SECRETS_DIR/age-key.txt $SECRETS_DIR/sign-secret.hex"
