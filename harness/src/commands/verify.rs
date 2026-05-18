use anyhow::{Context, Result};

use crate::archive;
use crate::config::Config;
use crate::crypto::{encrypt, sign};
use crate::secrets::AgeIdentity;
use crate::snapshot::{verify_digest, LatestPointer, LATEST_KEY};
use crate::stores;

/// Read latest pointer from primary, verify digest + signature, decrypt, and
/// confirm the inner StorageState parses. No browser, no canary, no writes.
///
/// Output is a single line on stdout (`OK version=...`) so the command can
/// double as a health-check probe in CI without log scraping.
pub async fn run(cfg: &Config) -> Result<()> {
    let identity = AgeIdentity::take_from_env("HARNESS_AGE_IDENTITY")?.parse()?;
    let verify_pubkey_hex = std::fs::read_to_string(&cfg.crypto.verify_pubkey_file)
        .with_context(|| {
            format!(
                "read verify pubkey {}",
                cfg.crypto.verify_pubkey_file.display()
            )
        })?;
    let verify_pubkey = sign::parse_verifying_key(&verify_pubkey_hex)?;

    let primary = stores::build_store(cfg.primary()).await?;

    let (pointer_bytes, _) = primary
        .get(LATEST_KEY)
        .await
        .context("get latest.json")?;
    let pointer: LatestPointer =
        serde_json::from_slice(&pointer_bytes).context("parse latest.json")?;

    let (ciphertext, _) = primary
        .get(&pointer.object)
        .await
        .with_context(|| format!("get snapshot {}", pointer.object))?;
    verify_digest(&pointer.object_sha256, &ciphertext)?;

    let (sig_bytes, _) = primary
        .get(&pointer.signature_object)
        .await
        .with_context(|| format!("get signature {}", pointer.signature_object))?;
    let sig_hex = std::str::from_utf8(&sig_bytes)
        .context("signature is not utf-8")?
        .trim();
    let sig = hex::decode(sig_hex).context("decode signature hex")?;
    sign::verify(&verify_pubkey, &ciphertext, &sig)?;

    let plaintext = encrypt::decrypt(&ciphertext, &identity)?;
    let state = archive::decompress(&plaintext)?;

    println!(
        "OK version={} cookies={} created_at={}",
        pointer.version,
        state.cookies.len(),
        pointer.created_at.to_rfc3339()
    );
    Ok(())
}
