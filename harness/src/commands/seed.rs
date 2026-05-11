use anyhow::{Context, Result};
use bytes::Bytes;
use chrono::Utc;

use crate::archive;
use crate::browser::cdp::BrowserSession;
use crate::config::Config;
use crate::crypto::{encrypt, sign};
use crate::secrets::MinisignKey;
use crate::snapshot::{sha256_hex, LatestPointer, SnapshotFormat, LATEST_KEY};
use crate::stores;

pub async fn run(cfg: &Config) -> Result<()> {
    let recipient = encrypt::read_recipient(&cfg.crypto.recipient_file)?;
    let signing_key_raw = MinisignKey::take_from_env("HARNESS_SIGN_SECRET")?;
    let signing_key = sign::parse_signing_key(signing_key_raw.expose())?;

    let primary = stores::build_store(cfg.primary()).await?;
    let mut mirrors: Vec<Box<dyn stores::SnapshotStore>> = Vec::new();
    for mirror_cfg in cfg.mirrors() {
        match stores::build_store(mirror_cfg).await {
            Ok(s) => mirrors.push(s),
            Err(e) => {
                tracing::warn!(store = mirror_cfg.name(), error = %e, "mirror unavailable, skipping")
            }
        }
    }

    tracing::info!(url = %cfg.seed.login_url, "launching headed browser");
    let session =
        BrowserSession::launch(&cfg.browser.chromium_bin, &cfg.browser.user_data_dir, false)
            .await?;
    let _page = session.open(&cfg.seed.login_url).await?;

    eprintln!();
    eprintln!("  Browser is open. Log in, then press Enter here to snapshot.");
    eprintln!("  (All tabs are captured — you can navigate freely before pressing Enter.)");
    eprintln!();
    let mut _line = String::new();
    std::io::stdin()
        .read_line(&mut _line)
        .context("read Enter from stdin")?;

    tracing::info!("snapshotting browser cookies");
    let state = session.export_storage_state().await?;
    session.close().await?;

    if state.cookies.is_empty() {
        anyhow::bail!("no cookies captured — was the login successful?");
    }
    tracing::info!(cookies = state.cookies.len(), "cookies captured");

    let plaintext = archive::compress(&state)?;
    let ciphertext: Bytes = encrypt::encrypt(&plaintext, &recipient)?;
    let sig_bytes = sign::sign(&signing_key, &ciphertext);
    let sig_hex = hex::encode(&sig_bytes);

    let now = Utc::now();
    let ts = now.format("%Y-%m-%dT%H-%M-%SZ").to_string();
    let object_key = format!("snapshots/{ts}.zst.age");
    let sig_key = format!("snapshots/{ts}.zst.age.sig");
    let sha = sha256_hex(&ciphertext);

    tracing::info!(object = %object_key, sha256 = %&sha[..16], "uploading to primary");
    primary
        .put(&object_key, ciphertext.clone())
        .await
        .context("put snapshot")?;
    primary
        .put(&sig_key, Bytes::from(sig_hex.clone()))
        .await
        .context("put signature")?;

    let pointer = LatestPointer {
        version: ts.clone(),
        object: object_key.clone(),
        object_sha256: sha,
        signature_object: sig_key.clone(),
        created_at: now,
        format: SnapshotFormat::StorageStateV1,
    };
    let pointer_bytes = Bytes::from(serde_json::to_vec_pretty(&pointer)?);

    let existing_etag = primary.head(LATEST_KEY).await?;
    match existing_etag {
        None => {
            primary
                .put_if_unmodified(LATEST_KEY, pointer_bytes.clone(), None)
                .await
        }
        Some(etag) => {
            primary
                .put_if_unmodified(LATEST_KEY, pointer_bytes.clone(), Some(&etag))
                .await
        }
    }
    .context("put latest.json")?;

    stores::fanout(&mirrors, &object_key, ciphertext).await;
    stores::fanout(&mirrors, &sig_key, Bytes::from(sig_hex)).await;
    stores::fanout(&mirrors, LATEST_KEY, pointer_bytes).await;

    tracing::info!(
        "seed complete: version={ts}, cookies={}",
        state.cookies.len()
    );
    Ok(())
}
