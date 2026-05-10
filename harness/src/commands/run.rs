use anyhow::{bail, Context, Result};
use bytes::Bytes;
use chrono::{DateTime, Utc};
use std::time::Duration;

use crate::archive;
use crate::browser::canary;
use crate::browser::cdp::{self, BrowserSession};
use crate::config::Config;
use crate::crypto::{encrypt, sign};
use crate::health;
use crate::lock;
use crate::secrets::{AgeIdentity, MinisignKey};
use crate::snapshot::{
    ensure_monotonic, sha256_hex, verify_digest, LatestPointer, SnapshotFormat, LATEST_KEY,
};
use crate::stores::{self, SnapshotStore};

enum Outcome {
    Saved { version: String },
    CanaryFailed { status: u16, observed: Option<String> },
}

pub async fn run(cfg: &Config) -> Result<()> {
    let identity = AgeIdentity::take_from_env("HARNESS_AGE_IDENTITY")?.parse()?;
    let recipient = encrypt::read_recipient(&cfg.crypto.recipient_file)?;
    let signing_key_raw = MinisignKey::take_from_env("HARNESS_SIGN_SECRET")?;
    let signing_key = sign::parse_signing_key(signing_key_raw.expose())?;
    let verify_pubkey_hex = std::fs::read_to_string(&cfg.crypto.verify_pubkey_file)
        .with_context(|| {
            format!(
                "read verify pubkey {}",
                cfg.crypto.verify_pubkey_file.display()
            )
        })?;
    let verify_pubkey = sign::parse_verifying_key(&verify_pubkey_hex)?;

    let primary = stores::build_store(cfg.primary()).await?;
    let mut mirrors: Vec<Box<dyn SnapshotStore>> = Vec::new();
    for mirror_cfg in cfg.mirrors() {
        match stores::build_store(mirror_cfg).await {
            Ok(s) => mirrors.push(s),
            Err(e) => {
                tracing::warn!(store = mirror_cfg.name(), error = %e, "mirror unavailable, skipping")
            }
        }
    }

    let run_id = std::env::var("GITHUB_RUN_ID")
        .unwrap_or_else(|_| format!("local-{}", Utc::now().timestamp()));
    let started = Utc::now();
    let ttl = Duration::from_secs(cfg.lock.ttl_seconds);

    let guard = lock::acquire(&*primary, run_id.clone(), ttl).await?;
    tracing::info!("lock acquired");

    let result = run_locked(
        cfg,
        &*primary,
        &mirrors,
        &identity,
        &recipient,
        &signing_key,
        &verify_pubkey,
    )
    .await;

    if let Err(e) = guard.release().await {
        tracing::warn!(error = %e, "lock release failed (TTL will reclaim)");
    }

    let record = match &result {
        Ok(Outcome::Saved { version }) => {
            health::ok(run_id.clone(), started, version.clone())
        }
        Ok(Outcome::CanaryFailed { status, observed }) => {
            health::canary_failed(run_id.clone(), started, *status, observed.clone())
        }
        Err(e) => health::error(run_id.clone(), started, format!("{e:#}")),
    };
    health::write(&*primary, &record).await;

    match result {
        Ok(Outcome::Saved { version }) => {
            tracing::info!(version = %version, "run complete");
            Ok(())
        }
        Ok(Outcome::CanaryFailed { status, observed }) => bail!(
            "canary failed: status={status}, observed={observed:?}; refusing to overwrite snapshot"
        ),
        Err(e) => Err(e),
    }
}

#[allow(clippy::too_many_arguments)]
async fn run_locked(
    cfg: &Config,
    primary: &dyn SnapshotStore,
    mirrors: &[Box<dyn SnapshotStore>],
    identity: &age::x25519::Identity,
    recipient: &age::x25519::Recipient,
    signing_key: &ed25519_dalek::SigningKey,
    verify_pubkey: &ed25519_dalek::VerifyingKey,
) -> Result<Outcome> {
    let prev = restore(primary, identity, verify_pubkey).await?;
    tracing::info!(version = %prev.pointer.version, "pointer loaded");

    let session = BrowserSession::launch(
        &cfg.browser.chromium_bin,
        &cfg.browser.user_data_dir,
        cfg.browser.headless,
    )
    .await?;
    session.import_storage_state(&prev.state).await?;

    let jar = cdp::cookies_to_reqwest_jar(&prev.state);
    let http = reqwest::Client::builder()
        .cookie_provider(jar)
        .build()
        .context("build canary client")?;

    let canary_result = canary::check(
        &http,
        &cfg.canary.url,
        cfg.canary.expected_status,
        &cfg.canary.field,
        &cfg.canary.expected_value,
    )
    .await?;

    if !canary_result.ok {
        let _ = session.close().await;
        return Ok(Outcome::CanaryFailed {
            status: canary_result.status,
            observed: canary_result.observed,
        });
    }
    tracing::info!("canary ok");

    // Workload extension point: this is where the actual job runs.
    // Kept empty so the harness lifecycle (restore/save) can be exercised
    // independently of any specific automation; fill in here.
    tracing::info!("workload placeholder");

    let new_state = session.export_storage_state().await?;
    session.close().await?;
    if new_state.cookies.is_empty() {
        bail!("post-run cookies empty; refusing to save");
    }

    let new_version = save(
        primary,
        mirrors,
        recipient,
        signing_key,
        &new_state,
        &prev.pointer,
        &prev.pointer_etag,
    )
    .await?;

    Ok(Outcome::Saved {
        version: new_version,
    })
}

struct Restored {
    pointer: LatestPointer,
    pointer_etag: String,
    state: cdp::StorageState,
}

async fn restore(
    primary: &dyn SnapshotStore,
    identity: &age::x25519::Identity,
    verify_pubkey: &ed25519_dalek::VerifyingKey,
) -> Result<Restored> {
    tracing::info!("loading latest pointer");
    let (pointer_bytes, pointer_etag) =
        primary.get(LATEST_KEY).await.context("get latest.json")?;
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
    sign::verify(verify_pubkey, &ciphertext, &sig)?;
    tracing::info!("signature verified");

    let plaintext = encrypt::decrypt(&ciphertext, identity)?;
    let state = archive::decompress(&plaintext)?;
    tracing::info!(cookies = state.cookies.len(), "storage state restored");

    Ok(Restored {
        pointer,
        pointer_etag,
        state,
    })
}

#[allow(clippy::too_many_arguments)]
async fn save(
    primary: &dyn SnapshotStore,
    mirrors: &[Box<dyn SnapshotStore>],
    recipient: &age::x25519::Recipient,
    signing_key: &ed25519_dalek::SigningKey,
    new_state: &cdp::StorageState,
    prev_pointer: &LatestPointer,
    prev_pointer_etag: &str,
) -> Result<String> {
    let new_plaintext = archive::compress(new_state)?;
    let new_ciphertext: Bytes = encrypt::encrypt(&new_plaintext, recipient)?;
    let new_sig = sign::sign(signing_key, &new_ciphertext);

    let now: DateTime<Utc> = Utc::now();
    let ts = now.format("%Y-%m-%dT%H-%M-%SZ").to_string();
    let new_object_key = format!("snapshots/{ts}.zst.age");
    let new_sig_key = format!("snapshots/{ts}.zst.age.sig");
    let new_sha = sha256_hex(&new_ciphertext);

    let new_pointer = LatestPointer {
        version: ts.clone(),
        object: new_object_key.clone(),
        object_sha256: new_sha,
        signature_object: new_sig_key.clone(),
        created_at: now,
        format: SnapshotFormat::StorageStateV1,
    };
    ensure_monotonic(Some(prev_pointer), &new_pointer)?;

    tracing::info!(object = %new_object_key, "uploading snapshot");
    primary.put(&new_object_key, new_ciphertext.clone()).await?;
    primary
        .put(&new_sig_key, Bytes::from(hex::encode(&new_sig)))
        .await?;

    let new_pointer_bytes = Bytes::from(serde_json::to_vec_pretty(&new_pointer)?);
    primary
        .put_if_unmodified(LATEST_KEY, new_pointer_bytes.clone(), Some(prev_pointer_etag))
        .await
        .context("CAS update of latest.json (someone else wrote concurrently?)")?;

    stores::fanout(mirrors, &new_object_key, new_ciphertext).await;
    stores::fanout(mirrors, LATEST_KEY, new_pointer_bytes).await;

    Ok(ts)
}
