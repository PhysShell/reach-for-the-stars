use anyhow::Result;
use async_trait::async_trait;
use bytes::Bytes;

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
