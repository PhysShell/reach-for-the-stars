use anyhow::{ensure, Result};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

pub const LATEST_KEY: &str = "latest.json";
pub const LOCK_KEY: &str = "lock.json";

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct LatestPointer {
    pub version: String,
    pub object: String,
    pub object_sha256: String,
    pub signature_object: String,
    pub created_at: DateTime<Utc>,
    pub format: SnapshotFormat,
}

#[derive(Debug, Serialize, Deserialize, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum SnapshotFormat {
    /// Playwright-compatible storage_state.json (cookies + origins).
    StorageStateV1,
    /// Full Chromium user_data_dir tarball.
    UserDataDirV1,
}

pub fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    hex::encode(digest)
}

/// Verify a freshly-fetched snapshot matches the pointer's recorded digest.
pub fn verify_digest(expected_hex: &str, actual_bytes: &[u8]) -> Result<()> {
    let actual = sha256_hex(actual_bytes);
    ensure!(
        actual == expected_hex,
        "snapshot digest mismatch: expected {expected_hex}, got {actual}"
    );
    Ok(())
}

/// Reject pointer rollback: a new pointer must be strictly newer than the previous.
pub fn ensure_monotonic(prev: Option<&LatestPointer>, next: &LatestPointer) -> Result<()> {
    if let Some(p) = prev {
        ensure!(
            next.created_at > p.created_at,
            "pointer rollback detected: prev created_at = {}, next created_at = {}",
            p.created_at,
            next.created_at
        );
    }
    Ok(())
}
