use super::{ObjectMeta, PutResult, SnapshotStore};
use anyhow::Result;
use async_trait::async_trait;
use bytes::Bytes;

/// Stores snapshots as binary blobs on an orphan branch of a git repo.
/// CAS is implemented via `git push --force-with-lease=<branch>:<expected-sha>`.
///
/// Useful as a zero-vendor-lock fallback that costs nothing on top of an existing
/// GitHub repo and only needs `GITHUB_TOKEN` (no R2 / S3 credentials).
pub struct GitBranchStore {
    name: String,
    pub repo_url: String,
    pub branch: String,
}

impl GitBranchStore {
    pub fn new(
        name: String,
        repo_url: String,
        branch: String,
        _token: secrecy::SecretString,
    ) -> Result<Self> {
        Ok(Self {
            name,
            repo_url,
            branch,
        })
    }
}

#[async_trait]
impl SnapshotStore for GitBranchStore {
    fn name(&self) -> &str {
        &self.name
    }

    async fn put_if_unmodified(
        &self,
        _key: &str,
        _body: Bytes,
        _expected_etag: Option<&str>,
    ) -> Result<PutResult> {
        anyhow::bail!("git_branch.put_if_unmodified: not yet implemented");
    }

    async fn put(&self, _key: &str, _body: Bytes) -> Result<PutResult> {
        anyhow::bail!("git_branch.put: not yet implemented");
    }

    async fn get(&self, _key: &str) -> Result<(Bytes, String)> {
        anyhow::bail!("git_branch.get: not yet implemented");
    }

    async fn head(&self, _key: &str) -> Result<Option<String>> {
        anyhow::bail!("git_branch.head: not yet implemented");
    }

    async fn list(&self, _prefix: &str) -> Result<Vec<ObjectMeta>> {
        anyhow::bail!("git_branch.list: not yet implemented");
    }

    async fn delete_if_match(&self, _key: &str, _etag: &str) -> Result<()> {
        anyhow::bail!("git_branch.delete_if_match: not yet implemented");
    }
}
