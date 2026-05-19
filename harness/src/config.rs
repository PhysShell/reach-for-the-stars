use anyhow::{Context, Result};
use serde::Deserialize;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

#[derive(Debug, Deserialize)]
pub struct Config {
    pub browser: Browser,
    pub seed: Seed,
    pub canary: Canary,
    pub crypto: Crypto,
    pub lock: LockCfg,
    pub stores: Vec<StoreCfg>,
    #[serde(default)]
    pub export: ExportCfg,
}

#[derive(Debug, Deserialize)]
pub struct Browser {
    pub chromium_bin: PathBuf,
    pub user_data_dir: PathBuf,
    #[serde(default = "default_true")]
    pub headless: bool,
}

#[derive(Debug, Deserialize)]
pub struct Seed {
    pub login_url: String,
}

#[derive(Debug, Deserialize)]
pub struct Canary {
    pub url: String,
    pub expected_status: u16,
    pub field: String,
    pub expected_value: String,
    /// Sent as the canary request's User-Agent. A bare reqwest UA gets 403/404
    /// from WAFs and many JSON APIs; a realistic browser UA avoids that. Not
    /// an anti-bot bypass — just "don't look like a broken curl".
    #[serde(default = "default_user_agent")]
    pub user_agent: String,
    /// Extra request headers (e.g. `Accept`, `Origin`). Anything set here
    /// overrides the built-in defaults for the same header name.
    #[serde(default)]
    pub headers: HashMap<String, String>,
}

#[derive(Debug, Deserialize)]
pub struct Crypto {
    pub recipient_file: PathBuf,
    pub verify_pubkey_file: PathBuf,
}

#[derive(Debug, Deserialize)]
pub struct LockCfg {
    pub ttl_seconds: u64,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum StoreCfg {
    S3 {
        name: String,
        endpoint: String,
        region: String,
        bucket: String,
        prefix: String,
        access_key_env: String,
        secret_key_env: String,
    },
    GithubRelease {
        name: String,
        repo: String,
        token_env: String,
    },
    GitBranch {
        name: String,
        repo_url: String,
        branch: String,
        token_env: String,
    },
}

impl StoreCfg {
    pub fn name(&self) -> &str {
        match self {
            StoreCfg::S3 { name, .. }
            | StoreCfg::GithubRelease { name, .. }
            | StoreCfg::GitBranch { name, .. } => name,
        }
    }
}

impl Config {
    pub fn load(path: &Path) -> Result<Self> {
        let raw = std::fs::read_to_string(path)
            .with_context(|| format!("read config {}", path.display()))?;
        let mut cfg: Self = toml::from_str(&raw)
            .with_context(|| format!("parse config {}", path.display()))?;

        // Resolve relative paths against the config file's directory so the
        // harness can be invoked from any working directory.
        //
        // canonicalize() can only fail here if the file vanished between the
        // read above and now (TOCTOU) or on a permission/symlink anomaly.
        // Surface that explicitly instead of silently leaving paths relative
        // to CWD — a silent fallback would resolve crypto/key files against
        // the wrong directory and fail later with a confusing "file not found".
        let base = path
            .canonicalize()
            .with_context(|| format!("canonicalize config path {}", path.display()))?;
        let base = base.parent().unwrap_or(Path::new("."));
        let resolve = |p: PathBuf| if p.is_relative() { base.join(p) } else { p };
        cfg.browser.chromium_bin = resolve(cfg.browser.chromium_bin);
        cfg.browser.user_data_dir = resolve(cfg.browser.user_data_dir);
        cfg.crypto.recipient_file = resolve(cfg.crypto.recipient_file);
        cfg.crypto.verify_pubkey_file = resolve(cfg.crypto.verify_pubkey_file);

        if let Ok(bin) = std::env::var("CHROMIUM_BIN") {
            cfg.browser.chromium_bin = PathBuf::from(bin);
        }
        cfg.validate()?;
        Ok(cfg)
    }

    /// Surface configuration mistakes that would otherwise blow up deep in the
    /// pipeline (`stores[0]` panicking on empty list, zero-TTL locks expiring
    /// immediately, etc.) as a single descriptive error at startup.
    fn validate(&self) -> Result<()> {
        anyhow::ensure!(
            !self.stores.is_empty(),
            "config: stores must contain at least one entry (the primary)"
        );
        anyhow::ensure!(
            self.lock.ttl_seconds > 0,
            "config: lock.ttl_seconds must be > 0 (lock with TTL=0 expires immediately and provides no mutual exclusion)"
        );
        Ok(())
    }

    pub fn primary(&self) -> &StoreCfg {
        &self.stores[0]
    }

    pub fn mirrors(&self) -> &[StoreCfg] {
        &self.stores[1..]
    }
}

fn default_true() -> bool {
    true
}

/// A current, realistic desktop Chrome UA. Bump occasionally; a stale UA is
/// itself a (mild) signal, but any plausible browser UA beats `reqwest/x.y`.
pub fn default_user_agent() -> String {
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) \
     Chrome/131.0.0.0 Safari/537.36"
        .to_string()
}

#[derive(Debug, Deserialize)]
pub struct ExportCfg {
    #[serde(default = "default_chatgpt_base_url")]
    pub chatgpt_base_url: String,
    /// Minimum delay between consecutive API calls (ms).
    #[serde(default = "default_rate_limit_delay_ms")]
    pub rate_limit_delay_ms: u64,
    /// Max retries on HTTP 429 before giving up.
    #[serde(default = "default_max_retries")]
    pub max_retries: u32,
    /// Object-store key prefix for exported files.
    #[serde(default = "default_export_prefix")]
    pub prefix: String,
}

impl Default for ExportCfg {
    fn default() -> Self {
        Self {
            chatgpt_base_url: default_chatgpt_base_url(),
            rate_limit_delay_ms: default_rate_limit_delay_ms(),
            max_retries: default_max_retries(),
            prefix: default_export_prefix(),
        }
    }
}

fn default_chatgpt_base_url() -> String { "https://chatgpt.com".into() }
fn default_rate_limit_delay_ms() -> u64 { 2000 }
fn default_max_retries() -> u32 { 5 }
fn default_export_prefix() -> String { "exports/".into() }
