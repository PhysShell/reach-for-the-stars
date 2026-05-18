use anyhow::{anyhow, bail, Context, Result};
use async_trait::async_trait;
use bytes::Bytes;
use secrecy::{ExposeSecret, SecretString};
use serde::Deserialize;

use super::{ObjectMeta, PutResult, SnapshotStore};

/// GitHub Releases as a mirror-only snapshot store.
///
/// Routing collapses our virtual hierarchy onto a small fixed set of releases
/// so we don't pollute the repo's release list with one entry per snapshot:
///
///   key starts with "snapshots/"  → release tag "harness-snapshots"
///   key starts with "health/"     → release tag "harness-health"
///   key == "latest.json"          → release tag "harness-latest"
///   anything else                 → release tag "harness-misc"
///
/// Within each release the asset name is the key's basename. For latest.json
/// `put` deletes the existing asset with the same name before upload — this
/// is the closest GitHub Releases gets to "atomic update", and it's why this
/// store is mirror-only: there is no test-and-set primitive at the asset
/// level. `put_if_unmodified` returns an error.
pub struct GithubReleaseStore {
    name: String,
    owner: String,
    repo: String,
    token: SecretString,
    http: reqwest::Client,
}

#[derive(Debug, Deserialize)]
struct Release {
    id: u64,
}

#[derive(Debug, Deserialize)]
struct Asset {
    id: u64,
    name: String,
    size: u64,
}

const ACCEPT: &str = "application/vnd.github+json";
const API_VERSION: &str = "2022-11-28";
const USER_AGENT: &str = "harness/0.1 (+https://github.com)";

impl GithubReleaseStore {
    pub fn new(name: String, repo: String, token: SecretString) -> Result<Self> {
        let (owner, repo_name) = repo
            .split_once('/')
            .with_context(|| format!("expected owner/name, got {repo}"))?;
        let http = reqwest::Client::builder()
            .user_agent(USER_AGENT)
            .build()
            .context("build http client")?;
        Ok(Self {
            name,
            owner: owner.into(),
            repo: repo_name.into(),
            token,
            http,
        })
    }

    fn route(&self, key: &str) -> (String, String) {
        if key == "latest.json" {
            ("harness-latest".into(), "latest.json".into())
        } else if let Some(rest) = key.strip_prefix("snapshots/") {
            ("harness-snapshots".into(), rest.into())
        } else if let Some(rest) = key.strip_prefix("health/") {
            ("harness-health".into(), rest.into())
        } else {
            ("harness-misc".into(), key.replace('/', "_"))
        }
    }

    fn auth_header(&self) -> String {
        format!("Bearer {}", self.token.expose_secret())
    }

    async fn release_by_tag(&self, tag: &str) -> Result<Option<Release>> {
        let url = format!(
            "https://api.github.com/repos/{}/{}/releases/tags/{tag}",
            self.owner, self.repo
        );
        let res = self
            .http
            .get(&url)
            .header("Accept", ACCEPT)
            .header("X-GitHub-Api-Version", API_VERSION)
            .header("Authorization", self.auth_header())
            .send()
            .await
            .context("GET release by tag")?;
        match res.status().as_u16() {
            200 => Ok(Some(res.json::<Release>().await?)),
            404 => Ok(None),
            other => bail!(
                "release by tag {tag}: status {other} body {}",
                res.text().await.unwrap_or_default()
            ),
        }
    }

    async fn ensure_release(&self, tag: &str) -> Result<Release> {
        if let Some(r) = self.release_by_tag(tag).await? {
            return Ok(r);
        }
        let url = format!(
            "https://api.github.com/repos/{}/{}/releases",
            self.owner, self.repo
        );
        let body = serde_json::json!({
            "tag_name": tag,
            "name": tag,
            "draft": false,
            "prerelease": true,
            "make_latest": "false",
        });
        let res = self
            .http
            .post(&url)
            .header("Accept", ACCEPT)
            .header("X-GitHub-Api-Version", API_VERSION)
            .header("Authorization", self.auth_header())
            .json(&body)
            .send()
            .await
            .context("POST create release")?;
        if !res.status().is_success() {
            bail!(
                "create release {tag}: status {} body {}",
                res.status(),
                res.text().await.unwrap_or_default()
            );
        }
        Ok(res.json::<Release>().await?)
    }

    async fn list_assets(&self, release_id: u64) -> Result<Vec<Asset>> {
        let mut all = Vec::new();
        let mut page = 1u32;
        loop {
            let url = format!(
                "https://api.github.com/repos/{}/{}/releases/{release_id}/assets?per_page=100&page={page}",
                self.owner, self.repo
            );
            let res = self
                .http
                .get(&url)
                .header("Accept", ACCEPT)
                .header("X-GitHub-Api-Version", API_VERSION)
                .header("Authorization", self.auth_header())
                .send()
                .await
                .context("GET assets")?;
            if !res.status().is_success() {
                bail!("list assets: status {}", res.status());
            }
            let batch: Vec<Asset> = res.json().await?;
            let len = batch.len();
            all.extend(batch);
            if len < 100 {
                break;
            }
            page += 1;
        }
        Ok(all)
    }

    async fn delete_asset(&self, asset_id: u64) -> Result<()> {
        let url = format!(
            "https://api.github.com/repos/{}/{}/releases/assets/{asset_id}",
            self.owner, self.repo
        );
        let res = self
            .http
            .delete(&url)
            .header("Accept", ACCEPT)
            .header("X-GitHub-Api-Version", API_VERSION)
            .header("Authorization", self.auth_header())
            .send()
            .await
            .context("DELETE asset")?;
        if !res.status().is_success() {
            bail!("delete asset {asset_id}: status {}", res.status());
        }
        Ok(())
    }

    async fn upload_asset(
        &self,
        release_id: u64,
        name: &str,
        body: Bytes,
    ) -> Result<Asset> {
        let url = format!(
            "https://uploads.github.com/repos/{}/{}/releases/{release_id}/assets?name={}",
            self.owner,
            self.repo,
            urlencoding::encode(name)
        );
        let res = self
            .http
            .post(&url)
            .header("Accept", ACCEPT)
            .header("X-GitHub-Api-Version", API_VERSION)
            .header("Authorization", self.auth_header())
            .header("Content-Type", "application/octet-stream")
            .body(body)
            .send()
            .await
            .context("POST upload asset")?;
        if !res.status().is_success() {
            bail!(
                "upload asset {name}: status {} body {}",
                res.status(),
                res.text().await.unwrap_or_default()
            );
        }
        Ok(res.json().await?)
    }

    async fn download_asset(&self, asset_id: u64) -> Result<Bytes> {
        let url = format!(
            "https://api.github.com/repos/{}/{}/releases/assets/{asset_id}",
            self.owner, self.repo
        );
        let res = self
            .http
            .get(&url)
            .header("Accept", "application/octet-stream")
            .header("X-GitHub-Api-Version", API_VERSION)
            .header("Authorization", self.auth_header())
            .send()
            .await
            .context("GET asset bytes")?;
        if !res.status().is_success() {
            bail!("download asset {asset_id}: status {}", res.status());
        }
        Ok(res.bytes().await?)
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
        bail!("github_release: CAS writes not supported; configure as mirror only");
    }

    async fn put(&self, key: &str, body: Bytes) -> Result<PutResult> {
        let (tag, asset_name) = self.route(key);
        let release = self.ensure_release(&tag).await?;
        // Idempotent overwrite: if an asset with the same name exists, drop it.
        for a in self.list_assets(release.id).await? {
            if a.name == asset_name {
                self.delete_asset(a.id).await?;
            }
        }
        let asset = self.upload_asset(release.id, &asset_name, body).await?;
        Ok(PutResult {
            etag: asset.id.to_string(),
        })
    }

    async fn get(&self, key: &str) -> Result<(Bytes, String)> {
        let (tag, asset_name) = self.route(key);
        let release = self
            .release_by_tag(&tag)
            .await?
            .ok_or_else(|| anyhow!("github_release: release {tag} not found"))?;
        let asset = self
            .list_assets(release.id)
            .await?
            .into_iter()
            .find(|a| a.name == asset_name)
            .ok_or_else(|| anyhow!("github_release: asset {asset_name} not found in {tag}"))?;
        let bytes = self.download_asset(asset.id).await?;
        Ok((bytes, asset.id.to_string()))
    }

    async fn head(&self, key: &str) -> Result<Option<String>> {
        let (tag, asset_name) = self.route(key);
        let Some(release) = self.release_by_tag(&tag).await? else {
            return Ok(None);
        };
        Ok(self
            .list_assets(release.id)
            .await?
            .into_iter()
            .find(|a| a.name == asset_name)
            .map(|a| a.id.to_string()))
    }

    async fn list(&self, prefix: &str) -> Result<Vec<ObjectMeta>> {
        let (tag, name_prefix) = self.route(prefix);
        let Some(release) = self.release_by_tag(&tag).await? else {
            return Ok(vec![]);
        };
        Ok(self
            .list_assets(release.id)
            .await?
            .into_iter()
            .filter(|a| a.name.starts_with(&name_prefix))
            .map(|a| ObjectMeta {
                key: a.name.clone(),
                etag: a.id.to_string(),
                size: a.size,
            })
            .collect())
    }

    async fn delete_if_match(&self, key: &str, etag: &str) -> Result<()> {
        let (tag, asset_name) = self.route(key);
        let Some(release) = self.release_by_tag(&tag).await? else {
            return Ok(());
        };
        let target = self
            .list_assets(release.id)
            .await?
            .into_iter()
            .find(|a| a.name == asset_name);
        if let Some(a) = target {
            if a.id.to_string() != etag {
                bail!("github_release delete: etag mismatch ({etag} vs {})", a.id);
            }
            self.delete_asset(a.id).await?;
        }
        Ok(())
    }
}
