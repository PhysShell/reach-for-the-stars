use crate::snapshot::LOCK_KEY;
use crate::stores::SnapshotStore;
use anyhow::{bail, Result};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
pub struct Lock {
    pub run_id: String,
    pub created_at: DateTime<Utc>,
    pub expires_at: DateTime<Utc>,
}

pub struct LockGuard<'a> {
    store: &'a dyn SnapshotStore,
    etag: String,
}

/// Acquire a single-writer lock on the primary store.
///
/// Uses store-native CAS (`If-None-Match: *` create, then `If-Match: <etag>` if expired).
/// This is race-free; a TTL-based "is it stale?" check is only consulted to decide
/// whether to overwrite an existing lock, not to decide whether to write.
pub async fn acquire<'a>(
    store: &'a dyn SnapshotStore,
    run_id: String,
    ttl: std::time::Duration,
) -> Result<LockGuard<'a>> {
    let now = Utc::now();
    let lock = Lock {
        run_id,
        created_at: now,
        expires_at: now + chrono::Duration::from_std(ttl)?,
    };
    let body = bytes::Bytes::from(serde_json::to_vec_pretty(&lock)?);

    let existing_etag = store.head(LOCK_KEY).await?;

    let put = match existing_etag {
        None => store.put_if_unmodified(LOCK_KEY, body, None).await?,
        Some(etag) => {
            let (existing_body, _) = store.get(LOCK_KEY).await?;
            let prev: Lock = serde_json::from_slice(&existing_body)?;
            if prev.expires_at > now {
                bail!(
                    "lock held by run_id={} until {} (now {})",
                    prev.run_id,
                    prev.expires_at,
                    now
                );
            }
            store.put_if_unmodified(LOCK_KEY, body, Some(&etag)).await?
        }
    };

    Ok(LockGuard {
        store,
        etag: put.etag,
    })
}

impl LockGuard<'_> {
    pub async fn release(self) -> Result<()> {
        self.store.delete_if_match(LOCK_KEY, &self.etag).await
    }
}
