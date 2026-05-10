use bytes::Bytes;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::stores::SnapshotStore;

#[derive(Debug, Serialize, Deserialize, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum HealthStatus {
    Ok,
    CanaryFailed,
    Error,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct HealthRecord {
    pub run_id: String,
    pub started_at: DateTime<Utc>,
    pub ended_at: DateTime<Utc>,
    pub status: HealthStatus,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub snapshot_version: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub canary_status: Option<u16>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub canary_observed: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// Append-only log: each run gets its own object under `health/`.
///
/// Best-effort by design — any failure here is logged at warn level but
/// must never propagate up to fail the run itself. The audit trail is
/// strictly less critical than the actual snapshot pipeline.
pub async fn write(store: &dyn SnapshotStore, record: &HealthRecord) {
    let key = format!(
        "health/{}-{}.json",
        record.started_at.format("%Y-%m-%dT%H-%M-%SZ"),
        sanitize_run_id(&record.run_id)
    );
    let body = match serde_json::to_vec_pretty(record) {
        Ok(b) => Bytes::from(b),
        Err(e) => {
            tracing::warn!(error = %e, "health: serialise");
            return;
        }
    };
    if let Err(e) = store.put(&key, body).await {
        tracing::warn!(error = %e, "health: put failed");
    }
}

fn sanitize_run_id(id: &str) -> String {
    id.chars()
        .map(|c| if c.is_ascii_alphanumeric() || c == '-' { c } else { '-' })
        .collect()
}

pub fn ok(run_id: String, started: DateTime<Utc>, snapshot_version: String) -> HealthRecord {
    HealthRecord {
        run_id,
        started_at: started,
        ended_at: Utc::now(),
        status: HealthStatus::Ok,
        snapshot_version: Some(snapshot_version),
        canary_status: None,
        canary_observed: None,
        error: None,
    }
}

pub fn canary_failed(
    run_id: String,
    started: DateTime<Utc>,
    canary_status: u16,
    canary_observed: Option<String>,
) -> HealthRecord {
    HealthRecord {
        run_id,
        started_at: started,
        ended_at: Utc::now(),
        status: HealthStatus::CanaryFailed,
        snapshot_version: None,
        canary_status: Some(canary_status),
        canary_observed,
        error: None,
    }
}

pub fn error(run_id: String, started: DateTime<Utc>, error: String) -> HealthRecord {
    HealthRecord {
        run_id,
        started_at: started,
        ended_at: Utc::now(),
        status: HealthStatus::Error,
        snapshot_version: None,
        canary_status: None,
        canary_observed: None,
        error: Some(error),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn run_id_is_sanitised_for_paths() {
        assert_eq!(sanitize_run_id("github-12345"), "github-12345");
        assert_eq!(sanitize_run_id("local/run.42"), "local-run-42");
        assert_eq!(sanitize_run_id("a b\tc"), "a-b-c");
    }
}
