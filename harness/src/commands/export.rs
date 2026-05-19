use anyhow::{Context, Result};
use bytes::Bytes;
use std::time::Duration;

use crate::archive;
use crate::browser::cdp;
use crate::config::Config;
use crate::crypto::{encrypt, sign};
use crate::secrets::AgeIdentity;
use crate::snapshot::{verify_digest, LatestPointer, LATEST_KEY};
use crate::stores;
use crate::workload::chatgpt::ChatGptClient;

pub async fn run(cfg: &Config) -> Result<()> {
    // ── 1. Restore snapshot (same path as verify) ─────────────────────────
    let identity = AgeIdentity::take_from_env("HARNESS_AGE_IDENTITY")?.parse()?;
    let verify_pubkey_hex =
        std::fs::read_to_string(&cfg.crypto.verify_pubkey_file).with_context(|| {
            format!(
                "read verify pubkey {}",
                cfg.crypto.verify_pubkey_file.display()
            )
        })?;
    let verify_pubkey = sign::parse_verifying_key(verify_pubkey_hex.trim())?;

    let primary = stores::build_store(cfg.primary()).await?;

    let (pointer_bytes, _) = primary.get(LATEST_KEY).await.context("get latest.json")?;
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
        .context("signature not utf-8")?
        .trim();
    let sig = hex::decode(sig_hex).context("decode signature hex")?;
    sign::verify(&verify_pubkey, &ciphertext, &sig)?;

    let plaintext = encrypt::decrypt(&ciphertext, &identity)?;
    let state = archive::decompress(&plaintext)?;
    tracing::info!(cookies = state.cookies.len(), "session restored");

    // ── 2. Authenticated HTTP client with session cookies ─────────────────
    let jar = cdp::cookies_to_reqwest_jar(&state);
    let http = reqwest::Client::builder()
        .cookie_provider(jar)
        .user_agent(&cfg.canary.user_agent)
        .build()
        .context("build http client")?;

    // ── 3. ChatGPT client → access token ─────────────────────────────────
    let export_cfg = &cfg.export;
    let client = ChatGptClient::new(
        http,
        &export_cfg.chatgpt_base_url,
        Duration::from_millis(export_cfg.rate_limit_delay_ms),
        export_cfg.max_retries,
    )
    .await?;

    // ── 4. Fetch the first accessible conversation ────────────────────────
    // Try up to 5 most recent conversations — some may be project/gizmo chats
    // that return conversation_inaccessible via the regular endpoint.
    let page = client.list_conversations(0, 5).await?;
    if page.items.is_empty() {
        println!("No conversations found.");
        return Ok(());
    }

    let mut detail = None;
    let mut chosen_summary = None;
    for summary in &page.items {
        tracing::info!(id = %summary.id, title = ?summary.title, "fetching conversation");
        match client.get_conversation(&summary.id).await {
            Ok(d) => {
                detail = Some(d);
                chosen_summary = Some(summary);
                break;
            }
            Err(e) => {
                tracing::warn!(id = %summary.id, error = %e, "skipping inaccessible conversation");
            }
        }
    }

    let (detail, summary) = match (detail, chosen_summary) {
        (Some(d), Some(s)) => (d, s),
        _ => {
            println!("No accessible conversations found in the first 5.");
            return Ok(());
        }
    };
    let _ = summary;

    // ── 5. Save JSON + Markdown to the primary store ──────────────────────
    let prefix = &export_cfg.prefix;

    let json_key = format!("{prefix}{}.json", detail.id);
    let json_bytes = Bytes::from(serde_json::to_vec_pretty(&detail)?);
    primary
        .put(&json_key, json_bytes)
        .await
        .with_context(|| format!("put {json_key}"))?;
    tracing::info!(key = %json_key, "saved JSON");

    let md = crate::workload::chatgpt::to_markdown(&detail);
    let md_key = format!("{prefix}{}.md", detail.id);
    primary
        .put(&md_key, Bytes::from(md.into_bytes()))
        .await
        .with_context(|| format!("put {md_key}"))?;
    tracing::info!(key = %md_key, "saved markdown");

    println!(
        "Exported: {} — {}",
        detail.id,
        detail.title.as_deref().unwrap_or("Untitled")
    );
    Ok(())
}
