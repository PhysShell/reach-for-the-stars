//! Canonical content hash for the semantic part of the envelope.
//!
//! `data_hash` deliberately excludes:
//! - `exported_at` (changes every run, even when nothing else did),
//! - `first_seen_at` / `last_seen_at` (operational metadata),
//! - the envelope `stats` (derived from the records).
//!
//! That way the hash is stable when the underlying star set is stable,
//! which lets a downstream uploader skip writing a new archive snapshot
//! when nothing has actually changed.

use anyhow::{Context, Result};
use serde::Serialize;
use sha2::{Digest, Sha256};
use time::OffsetDateTime;

use crate::model::{StarRecord, StarStatus};

#[derive(Debug, Serialize)]
struct CanonicalStar<'a> {
    repo_id: u64,
    full_name: &'a str,
    html_url: &'a str,
    description: Option<&'a str>,
    language: Option<&'a str>,
    #[serde(with = "time::serde::rfc3339")]
    starred_at: OffsetDateTime,
    #[serde(with = "time::serde::rfc3339::option")]
    unstarred_at: Option<OffsetDateTime>,
    status: StarStatus,
}

pub fn compute_data_hash(stars: &[StarRecord]) -> Result<String> {
    // Sort by repo_id so the hash is independent of insertion order.
    let mut indices: Vec<usize> = (0..stars.len()).collect();
    indices.sort_by_key(|&i| stars.get(i).map_or(0, |r| r.repo_id));

    let canonical: Vec<CanonicalStar<'_>> = indices
        .into_iter()
        .filter_map(|i| stars.get(i))
        .map(|r| CanonicalStar {
            repo_id: r.repo_id,
            full_name: &r.full_name,
            html_url: &r.html_url,
            description: r.description.as_deref(),
            language: r.language.as_deref(),
            starred_at: r.starred_at,
            unstarred_at: r.unstarred_at,
            status: r.status,
        })
        .collect();

    let bytes = serde_json::to_vec(&canonical).context("failed to serialize canonical stars")?;
    let mut hasher = Sha256::new();
    hasher.update(&bytes);
    let digest = hasher.finalize();
    Ok(format!("sha256:{}", hex::encode(digest)))
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing
)]
mod tests {
    use super::*;
    use time::macros::datetime;

    fn record(repo_id: u64, status: StarStatus) -> StarRecord {
        let t = datetime!(2026-05-09 12:00:00 UTC);
        StarRecord {
            repo_id,
            full_name: format!("owner/repo-{repo_id}"),
            html_url: format!("https://github.com/owner/repo-{repo_id}"),
            description: None,
            language: None,
            starred_at: t,
            first_seen_at: t,
            last_seen_at: t,
            unstarred_at: None,
            status,
        }
    }

    #[test]
    fn hash_is_independent_of_record_order() {
        let a = vec![
            record(1, StarStatus::Active),
            record(2, StarStatus::Active),
            record(3, StarStatus::Active),
        ];
        let b = vec![
            record(3, StarStatus::Active),
            record(1, StarStatus::Active),
            record(2, StarStatus::Active),
        ];
        assert_eq!(
            compute_data_hash(&a).unwrap(),
            compute_data_hash(&b).unwrap()
        );
    }

    #[test]
    fn hash_changes_when_status_changes() {
        let mut a = vec![record(1, StarStatus::Active)];
        let h1 = compute_data_hash(&a).unwrap();
        a[0].status = StarStatus::Unstarred;
        a[0].unstarred_at = Some(datetime!(2026-05-09 13:00:00 UTC));
        let h2 = compute_data_hash(&a).unwrap();
        assert_ne!(h1, h2);
    }

    #[test]
    fn hash_does_not_change_when_only_seen_timestamps_change() {
        let mut a = vec![record(1, StarStatus::Active)];
        let h1 = compute_data_hash(&a).unwrap();
        a[0].first_seen_at = datetime!(2026-05-10 00:00:00 UTC);
        a[0].last_seen_at = datetime!(2026-05-10 00:00:00 UTC);
        let h2 = compute_data_hash(&a).unwrap();
        assert_eq!(h1, h2);
    }

    #[test]
    fn hash_format_is_prefixed() {
        let h = compute_data_hash(&[record(1, StarStatus::Active)]).unwrap();
        assert!(h.starts_with("sha256:"));
        assert_eq!(h.len(), "sha256:".len() + 64);
    }
}
