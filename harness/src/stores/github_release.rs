use super::{ObjectMeta, PutResult, SnapshotStore};
use anyhow::Result;
use async_trait::async_trait;
use bytes::Bytes;

pub struct GithubReleaseStore {
    name: String,
    pub repo: String,
}

impl GithubReleaseStore {
    pub fn new(name: String, repo: String, _token: secrecy::SecretString) -> Result<Self> {
        Ok(Self { name, repo })
    }
}

#[async_trait]
impl SnapshotStore for GithubReleaseStore {
    fn name(&self) -> &str {
        &self.name
    }

    async fn put_if_unmodified(
        &self,
        _key: &str,
        _body: Bytes,
        _expected_etag: Option<&str>,
    ) -> Result<PutResult> {
        // GitHub Releases have no atomic CAS. Use as append-only mirror only.
        anyhow::bail!("github_release: CAS writes not supported; mirror-only store");
    }

    async fn put(&self, _key: &str, _body: Bytes) -> Result<PutResult> {
        // TODO: octocrab create_release(tag = derived from key) then upload_asset
        anyhow::bail!("github_release.put: not yet implemented");
    }

    async fn get(&self, _key: &str) -> Result<(Bytes, String)> {
        anyhow::bail!("github_release.get: not yet implemented");
    }

    async fn head(&self, _key: &str) -> Result<Option<String>> {
        anyhow::bail!("github_release.head: not yet implemented");
    }

    async fn list(&self, _prefix: &str) -> Result<Vec<ObjectMeta>> {
        anyhow::bail!("github_release.list: not yet implemented");
    }

    async fn delete_if_match(&self, _key: &str, _etag: &str) -> Result<()> {
        anyhow::bail!("github_release: deletion not supported by design");
    }
}
