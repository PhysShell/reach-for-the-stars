use serde::{Deserialize, Serialize};
use time::OffsetDateTime;

pub const SCHEMA_VERSION: u32 = 2;
pub const SUBJECT_KIND_USER_STARRED: &str = "user_starred_repositories";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StarsEnvelope {
    pub schema_version: u32,
    pub subject: Subject,
    #[serde(with = "time::serde::rfc3339")]
    pub exported_at: OffsetDateTime,
    pub sync: SyncState,
    pub stats: Stats,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub data_hash: Option<String>,
    pub stars: Vec<StarRecord>,
}

impl StarsEnvelope {
    pub fn empty(username: &str, now: OffsetDateTime) -> Self {
        Self {
            schema_version: SCHEMA_VERSION,
            subject: Subject {
                kind: SUBJECT_KIND_USER_STARRED.to_owned(),
                username: username.to_owned(),
            },
            exported_at: now,
            sync: SyncState::default(),
            stats: Stats::default(),
            data_hash: None,
            stars: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Subject {
    pub kind: String,
    pub username: String,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SyncState {
    pub mode: SyncModeRecorded,
    #[serde(
        default,
        with = "time::serde::rfc3339::option",
        skip_serializing_if = "Option::is_none"
    )]
    pub last_full_reconcile_at: Option<OffsetDateTime>,
    #[serde(
        default,
        with = "time::serde::rfc3339::option",
        skip_serializing_if = "Option::is_none"
    )]
    pub watermark_starred_at: Option<OffsetDateTime>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub watermark_repo_id: Option<u64>,
}

/// Mode actually used on the most recent run, recorded in the envelope.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SyncModeRecorded {
    #[default]
    Full,
    Incremental,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[allow(clippy::struct_field_names)]
pub struct Stats {
    pub active_count: u64,
    pub tombstone_count: u64,
    pub total_count: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct StarRecord {
    pub repo_id: u64,
    pub full_name: String,
    pub html_url: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub language: Option<String>,
    #[serde(with = "time::serde::rfc3339")]
    pub starred_at: OffsetDateTime,
    #[serde(with = "time::serde::rfc3339")]
    pub first_seen_at: OffsetDateTime,
    #[serde(with = "time::serde::rfc3339")]
    pub last_seen_at: OffsetDateTime,
    #[serde(
        default,
        with = "time::serde::rfc3339::option",
        skip_serializing_if = "Option::is_none"
    )]
    pub unstarred_at: Option<OffsetDateTime>,
    pub status: StarStatus,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum StarStatus {
    Active,
    Unstarred,
}

pub fn compute_stats(stars: &[StarRecord]) -> Stats {
    let mut active: u64 = 0;
    let mut tombstone: u64 = 0;
    for record in stars {
        match record.status {
            StarStatus::Active => active += 1,
            StarStatus::Unstarred => tombstone += 1,
        }
    }
    Stats {
        active_count: active,
        tombstone_count: tombstone,
        total_count: active + tombstone,
    }
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
    fn stats_counts_active_and_tombstones() {
        let stars = vec![
            record(1, StarStatus::Active),
            record(2, StarStatus::Active),
            record(3, StarStatus::Unstarred),
        ];
        let stats = compute_stats(&stars);
        assert_eq!(stats.active_count, 2);
        assert_eq!(stats.tombstone_count, 1);
        assert_eq!(stats.total_count, 3);
    }

    #[test]
    fn envelope_round_trips_through_json() {
        let now = datetime!(2026-05-09 12:00:00 UTC);
        let mut env = StarsEnvelope::empty("octocat", now);
        env.stars.push(record(42, StarStatus::Active));
        let json = serde_json::to_string(&env).unwrap();
        let parsed: StarsEnvelope = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.schema_version, SCHEMA_VERSION);
        assert_eq!(parsed.subject.username, "octocat");
        assert_eq!(parsed.stars.len(), 1);
        assert_eq!(parsed.stars[0].repo_id, 42);
    }
}
