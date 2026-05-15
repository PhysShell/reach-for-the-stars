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
require gh
require jq
require nix

if ! gh auth status --active >/dev/null 2>&1; then
  echo "error: gh is not authenticated. Run 'gh auth login' first." >&2
  exit 1
fi

mkdir -p "$SECRETS_DIR"
cd "$HARNESS_DIR"

KEYS_EXIST=false
if [[ -f "$SECRETS_DIR/age-key.txt" && -f "$SECRETS_DIR/sign-secret.hex" ]]; then
  KEYS_EXIST=true
  echo "==> Keypairs already exist in $SECRETS_DIR — skipping keygen, will re-upload."
else
  if [[ -f "$SECRETS_DIR/age-key.txt" || -f "$SECRETS_DIR/sign-secret.hex" ]]; then
    echo "error: partial keygen state in $SECRETS_DIR (only one key file found)." >&2
    echo "       Remove both age-key.txt and sign-secret.hex to start fresh." >&2
    exit 1
  fi

  echo "==> Generating age keypair"
  age-keygen -o "$SECRETS_DIR/age-key.txt"
  age-keygen -y "$SECRETS_DIR/age-key.txt" > "$HARNESS_DIR/age-recipient.txt"
  chmod 600 "$SECRETS_DIR/age-key.txt"

  echo "==> Generating ed25519 signing keypair"
  KEYS_JSON=$(nix run "$HARNESS_DIR#harness" -- gen-keys --json)
  SIGN_SECRET=$(echo "$KEYS_JSON" | jq -r .secret)
  SIGN_PUBKEY=$(echo "$KEYS_JSON" | jq -r .pubkey)

  echo "$SIGN_PUBKEY" > "$HARNESS_DIR/sign-pubkey.hex"
  printf '%s\n' "$SIGN_SECRET" > "$SECRETS_DIR/sign-secret.hex"
  chmod 600 "$SECRETS_DIR/sign-secret.hex"
  unset KEYS_JSON SIGN_SECRET SIGN_PUBKEY

  echo
  echo "=== Public artefacts (commit these) ==="
  echo "  harness/age-recipient.txt"
  echo "  harness/sign-pubkey.hex"
  echo
fi
echo "=== Sealing private artefacts into GitHub Secrets ==="

read -rp "GitHub repo (owner/name): " GH_REPO
read -rp "R2 Access Key ID:         " R2_AK
read -rsp "R2 Secret Access Key:     " R2_SK; echo
read -rp "R2 Endpoint URL:          " R2_ENDPOINT
read -rp "R2 Bucket:                " R2_BUCKET

gh secret set HARNESS_AGE_IDENTITY  --repo "$GH_REPO" < "$SECRETS_DIR/age-key.txt"
gh secret set HARNESS_SIGN_SECRET   --repo "$GH_REPO" < "$SECRETS_DIR/sign-secret.hex"
printf '%s' "$R2_AK"       | gh secret set HARNESS_R2_ACCESS_KEY --repo "$GH_REPO"
printf '%s' "$R2_SK"       | gh secret set HARNESS_R2_SECRET_KEY --repo "$GH_REPO"
printf '%s' "$R2_ENDPOINT" | gh secret set HARNESS_R2_ENDPOINT   --repo "$GH_REPO"
printf '%s' "$R2_BUCKET"   | gh secret set HARNESS_R2_BUCKET     --repo "$GH_REPO"

echo
echo "Done."
echo
echo "Next steps:"
echo "  1. cp config.example.toml config.toml   # edit canary URL etc."
echo "  2. cargo run --release -- seed          # interactive login, headed Chromium"
echo "  3. cargo run --release -- verify        # sanity-check the snapshot"
echo "  4. shred -u $SECRETS_DIR/age-key.txt $SECRETS_DIR/sign-secret.hex"
