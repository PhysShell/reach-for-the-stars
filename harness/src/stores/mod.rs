use anyhow::{Context, Result};
use async_trait::async_trait;
use bytes::Bytes;
use secrecy::SecretString;

use crate::config::StoreCfg;
use crate::secrets::StoreCredential;

pub mod git_branch;
pub mod github_release;
pub mod s3;

#[derive(Debug, Clone)]
pub struct PutResult {
    pub etag: String,
}

#[derive(Debug, Clone)]
pub struct ObjectMeta {
    pub key: String,
    pub etag: String,
    pub size: u64,
}

#[async_trait]
pub trait SnapshotStore: Send + Sync {
    fn name(&self) -> &str;

    /// Conditional put.
    /// `expected_etag = None`        → fail if object exists ("create-only").
    /// `expected_etag = Some(etag)`  → fail if object's etag differs ("update-only").
    async fn put_if_unmodified(
        &self,
        key: &str,
        body: Bytes,
        expected_etag: Option<&str>,
    ) -> Result<PutResult>;

    /// Unconditional put. Reserved for append-only paths (snapshots/{ts}.tar.zst.age).
    async fn put(&self, key: &str, body: Bytes) -> Result<PutResult>;

    async fn get(&self, key: &str) -> Result<(Bytes, String)>;

    async fn head(&self, key: &str) -> Result<Option<String>>;

    async fn list(&self, prefix: &str) -> Result<Vec<ObjectMeta>>;

    async fn delete_if_match(&self, key: &str, etag: &str) -> Result<()>;
}

/// Construct a `SnapshotStore` from config, pulling credentials from env.
///
/// Env vars named in the config are consumed (removed) so child processes cannot
/// inherit them. Call for primary first, then iterate mirrors — each store's
/// env vars are expected to be distinct.
pub async fn build_store(cfg: &StoreCfg) -> Result<Box<dyn SnapshotStore>> {
    match cfg {
        StoreCfg::S3 {
            name,
            endpoint,
            region,
            bucket,
            prefix,
            access_key_env,
            secret_key_env,
        } => {
            let creds = StoreCredential::take_from_env(access_key_env, secret_key_env)
                .with_context(|| format!("store {name}: read credentials"))?;
            let store = s3::S3Store::new(
                name.clone(),
                endpoint.clone(),
                region.clone(),
                bucket.clone(),
                prefix.clone(),
                creds,
            )
            .await?;
            Ok(Box::new(store))
        }
        StoreCfg::GithubRelease {
            name,
            repo,
            token_env,
        } => {
            let raw = std::env::var(token_env)
                .with_context(|| format!("store {name}: env var {token_env} not set"))?;
            std::env::remove_var(token_env);
            let store = github_release::GithubReleaseStore::new(
                name.clone(),
                repo.clone(),
                SecretString::from(raw),
            )?;
            Ok(Box::new(store))
        }
        StoreCfg::GitBranch {
            name,
            repo_url,
            branch,
            token_env,
        } => {
            let raw = std::env::var(token_env)
                .with_context(|| format!("store {name}: env var {token_env} not set"))?;
            std::env::remove_var(token_env);
            let store = git_branch::GitBranchStore::new(
                name.clone(),
                repo_url.clone(),
                branch.clone(),
                SecretString::from(raw),
            )?;
            Ok(Box::new(store))
        }
    }
}

/// Best-effort fan-out write to mirrors after a successful primary write.
/// Logs failures but does not propagate them: mirrors are eventual.
pub async fn fanout(
    mirrors: &[Box<dyn SnapshotStore>],
    key: &str,
    body: Bytes,
) {
    for m in mirrors {
        match m.put(key, body.clone()).await {
            Ok(_) => tracing::info!(store = m.name(), key, "mirror put ok"),
            Err(e) => tracing::warn!(store = m.name(), key, error = %e, "mirror put failed"),
        }
    }
}
