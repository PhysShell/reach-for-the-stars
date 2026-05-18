use anyhow::{bail, Context, Result};
use async_trait::async_trait;
use bytes::Bytes;
use secrecy::{ExposeSecret, SecretString};
use std::path::{Path, PathBuf};
use tempfile::TempDir;
use tokio::process::Command;
use tokio::sync::OnceCell;

use super::{ObjectMeta, PutResult, SnapshotStore};

/// Stores snapshots as binary blobs on an orphan branch of a git repo.
/// CAS is implemented via `git push --force-with-lease=<branch>:<expected-sha>`,
/// which is git's native test-and-set primitive at the ref level.
///
/// Etag semantics differ from S3: the etag returned for any file is the
/// **branch HEAD commit SHA** at the time of read, not a content hash.
/// `put_if_unmodified(key, body, Some(etag))` therefore means "succeed only
/// if the branch hasn't moved since I last read etag" — which is exactly
/// what we want for a single-pointer + append-only-snapshots layout.
///
/// Useful as a zero-vendor-lock fallback that costs nothing on top of an
/// existing GitHub repo and only needs a `GITHUB_TOKEN` (no R2/S3 keys).
pub struct GitBranchStore {
    name: String,
    repo_url: String,
    branch: String,
    token: SecretString,
    workdir: OnceCell<TempDir>,
}

impl GitBranchStore {
    pub fn new(
        name: String,
        repo_url: String,
        branch: String,
        token: SecretString,
    ) -> Result<Self> {
        Ok(Self {
            name,
            repo_url,
            branch,
            token,
            workdir: OnceCell::new(),
        })
    }

    async fn workdir(&self) -> Result<&Path> {
        let td = self
            .workdir
            .get_or_try_init(|| async { self.init_workdir().await })
            .await?;
        Ok(td.path())
    }

    async fn init_workdir(&self) -> Result<TempDir> {
        let td = tempfile::tempdir().context("create tempdir for git_branch workdir")?;
        match self.try_clone(td.path()).await {
            Ok(()) => Ok(td),
            Err(e) => {
                tracing::info!(
                    store = %self.name,
                    branch = %self.branch,
                    error = %e,
                    "remote branch absent, initialising orphan locally"
                );
                self.init_orphan(td.path()).await?;
                Ok(td)
            }
        }
    }

    async fn try_clone(&self, dir: &Path) -> Result<()> {
        let auth_url = self.auth_url();
        let out = Command::new("git")
            .args([
                "-c",
                "commit.gpgsign=false",
                "-c",
                "tag.gpgsign=false",
                "clone",
                "--branch",
                &self.branch,
                "--single-branch",
                "--depth",
                "1",
                &auth_url,
                dir.to_str().context("workdir not utf-8")?,
            ])
            .output()
            .await
            .context("spawn git clone")?;
        if !out.status.success() {
            bail!(
                "git clone failed: {}",
                String::from_utf8_lossy(&out.stderr).trim()
            );
        }
        self.configure_identity(dir).await?;
        Ok(())
    }

    async fn init_orphan(&self, dir: &Path) -> Result<()> {
        std::fs::create_dir_all(dir)?;
        run_git(dir, &["init", "--initial-branch", &self.branch]).await?;
        run_git(dir, &["remote", "add", "origin", &self.auth_url()]).await?;
        self.configure_identity(dir).await?;
        Ok(())
    }

    async fn configure_identity(&self, dir: &Path) -> Result<()> {
        run_git(dir, &["config", "user.email", "harness@invalid"]).await?;
        run_git(dir, &["config", "user.name", "harness"]).await?;
        Ok(())
    }

    fn auth_url(&self) -> String {
        // Embed token as Basic-auth username; works for github.com over HTTPS.
        // (We avoid http.extraHeader because it leaks via `git config --list`
        // into any subsequent `git` invocation reusing the workdir.)
        let token = self.token.expose_secret();
        if let Some(rest) = self.repo_url.strip_prefix("https://") {
            format!("https://x-access-token:{token}@{rest}")
        } else {
            self.repo_url.clone()
        }
    }

    async fn fetch_and_reset(&self, dir: &Path) -> Result<bool> {
        // Returns true if the remote branch exists, false if it does not yet.
        let fetch = git_command(dir)
            .args(["fetch", "origin", &self.branch])
            .output()
            .await
            .context("git fetch")?;
        if !fetch.status.success() {
            return Ok(false);
        }
        run_git(
            dir,
            &["reset", "--hard", &format!("origin/{}", self.branch)],
        )
        .await?;
        Ok(true)
    }

    async fn current_sha(&self, dir: &Path) -> Result<Option<String>> {
        let out = git_command(dir)
            .args(["rev-parse", "HEAD"])
            .output()
            .await
            .context("git rev-parse HEAD")?;
        if !out.status.success() {
            // No commits yet (orphan branch, never pushed).
            return Ok(None);
        }
        Ok(Some(String::from_utf8(out.stdout)?.trim().to_string()))
    }

    async fn commit_and_push(
        &self,
        dir: &Path,
        message: &str,
        force_with_lease: Option<&str>,
    ) -> Result<String> {
        run_git(dir, &["add", "-A"]).await?;
        let st = git_command(dir)
            .args(["status", "--porcelain"])
            .output()
            .await?;
        if st.stdout.is_empty() {
            // Nothing to commit; return current SHA.
            return self
                .current_sha(dir)
                .await?
                .ok_or_else(|| anyhow::anyhow!("no changes and no HEAD"));
        }
        run_git(dir, &["commit", "-m", message]).await?;
        let mut push_args: Vec<String> = vec!["push".into()];
        match force_with_lease {
            Some(prev_sha) => {
                push_args.push(format!("--force-with-lease={}:{}", self.branch, prev_sha));
                push_args.push("origin".into());
                push_args.push(self.branch.clone());
            }
            None => {
                // First push or unconditional create.
                push_args.push("--set-upstream".into());
                push_args.push("origin".into());
                push_args.push(self.branch.clone());
            }
        }
        let push_args_ref: Vec<&str> = push_args.iter().map(String::as_str).collect();
        let push = git_command(dir)
            .args(&push_args_ref)
            .output()
            .await
            .context("git push")?;
        if !push.status.success() {
            bail!(
                "git push failed (CAS or auth): {}",
                String::from_utf8_lossy(&push.stderr).trim()
            );
        }
        self.current_sha(dir)
            .await?
            .ok_or_else(|| anyhow::anyhow!("post-push HEAD missing"))
    }

    fn safe_path(dir: &Path, key: &str) -> Result<PathBuf> {
        if key.contains("..") || key.starts_with('/') {
            bail!("git_branch: invalid key {key}");
        }
        Ok(dir.join(key))
    }
}

/// Run git with signing globally disabled.
///
/// The harness writes to a dedicated state branch as machine automation;
/// commit/tag signing requires keys that are not available in CI runners and
/// would gain us nothing (the state is already protected by our own
/// ed25519 signature over the encrypted blob). Overriding here keeps the
/// store working regardless of the operator's `~/.gitconfig`.
fn git_command(dir: &Path) -> Command {
    let mut c = Command::new("git");
    c.current_dir(dir)
        .args(["-c", "commit.gpgsign=false", "-c", "tag.gpgsign=false"]);
    c
}

async fn run_git(dir: &Path, args: &[&str]) -> Result<()> {
    let out = git_command(dir)
        .args(args)
        .output()
        .await
        .with_context(|| format!("spawn git {args:?}"))?;
    if !out.status.success() {
        bail!(
            "git {args:?} failed: {}",
            String::from_utf8_lossy(&out.stderr).trim()
        );
    }
    Ok(())
}

#[async_trait]
impl SnapshotStore for GitBranchStore {
    fn name(&self) -> &str {
        &self.name
    }

    async fn put_if_unmodified(
        &self,
        key: &str,
        body: Bytes,
        expected_etag: Option<&str>,
    ) -> Result<PutResult> {
        let dir = self.workdir().await?.to_path_buf();
        let _ = self.fetch_and_reset(&dir).await?;
        let current = self.current_sha(&dir).await?;

        let target = Self::safe_path(&dir, key)?;
        match (expected_etag, current.as_deref(), target.exists()) {
            (None, _, true) => bail!("git_branch: create-only on existing key {key}"),
            (Some(_), None, _) => bail!("git_branch: branch is empty, no etag to match"),
            (Some(etag), Some(cur), _) if etag != cur => {
                bail!("git_branch: etag mismatch (expected {etag}, current {cur})")
            }
            _ => {}
        }

        if let Some(parent) = target.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(&target, &body)?;
        let new_sha = self
            .commit_and_push(
                &dir,
                &format!(
                    "harness: {} {key}",
                    if expected_etag.is_some() {
                        "update"
                    } else {
                        "create"
                    }
                ),
                current.as_deref(),
            )
            .await?;
        Ok(PutResult { etag: new_sha })
    }

    async fn put(&self, key: &str, body: Bytes) -> Result<PutResult> {
        let dir = self.workdir().await?.to_path_buf();
        let _ = self.fetch_and_reset(&dir).await?;
        let target = Self::safe_path(&dir, key)?;
        if let Some(parent) = target.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(&target, &body)?;
        let prev = self.current_sha(&dir).await?;
        let new_sha = self
            .commit_and_push(&dir, &format!("harness: put {key}"), prev.as_deref())
            .await?;
        Ok(PutResult { etag: new_sha })
    }

    async fn get(&self, key: &str) -> Result<(Bytes, String)> {
        let dir = self.workdir().await?.to_path_buf();
        let _ = self.fetch_and_reset(&dir).await?;
        let target = Self::safe_path(&dir, key)?;
        let bytes = std::fs::read(&target)
            .with_context(|| format!("git_branch get {key}: file missing"))?;
        let sha = self
            .current_sha(&dir)
            .await?
            .ok_or_else(|| anyhow::anyhow!("git_branch get {key}: no HEAD"))?;
        Ok((Bytes::from(bytes), sha))
    }

    async fn head(&self, key: &str) -> Result<Option<String>> {
        let dir = self.workdir().await?.to_path_buf();
        let exists = self.fetch_and_reset(&dir).await?;
        if !exists {
            return Ok(None);
        }
        let target = Self::safe_path(&dir, key)?;
        if !target.exists() {
            return Ok(None);
        }
        Ok(self.current_sha(&dir).await?)
    }

    async fn list(&self, prefix: &str) -> Result<Vec<ObjectMeta>> {
        let dir = self.workdir().await?.to_path_buf();
        let exists = self.fetch_and_reset(&dir).await?;
        if !exists {
            return Ok(vec![]);
        }
        let sha = self.current_sha(&dir).await?.unwrap_or_default();
        let out = git_command(&dir)
            .args(["ls-tree", "-r", "--long", "HEAD", prefix])
            .output()
            .await?;
        if !out.status.success() {
            return Ok(vec![]);
        }
        let stdout = String::from_utf8(out.stdout)?;
        let mut metas = Vec::new();
        for line in stdout.lines() {
            // format: "<mode> blob <sha> <size>\t<path>"
            let mut parts = line.splitn(2, '\t');
            let meta_part = parts.next().unwrap_or("");
            let path = parts.next().unwrap_or("").to_string();
            let cols: Vec<&str> = meta_part.split_whitespace().collect();
            let size: u64 = cols.get(3).and_then(|s| s.parse().ok()).unwrap_or(0);
            metas.push(ObjectMeta {
                key: path,
                etag: sha.clone(),
                size,
            });
        }
        Ok(metas)
    }

    async fn delete_if_match(&self, key: &str, etag: &str) -> Result<()> {
        let dir = self.workdir().await?.to_path_buf();
        let _ = self.fetch_and_reset(&dir).await?;
        let current = self
            .current_sha(&dir)
            .await?
            .ok_or_else(|| anyhow::anyhow!("git_branch delete: no HEAD"))?;
        if current != etag {
            bail!("git_branch delete: etag mismatch (expected {etag}, current {current})");
        }
        let target = Self::safe_path(&dir, key)?;
        if !target.exists() {
            return Ok(());
        }
        run_git(&dir, &["rm", key]).await?;
        let _ = self
            .commit_and_push(&dir, &format!("harness: delete {key}"), Some(&current))
            .await?;
        Ok(())
    }
}
