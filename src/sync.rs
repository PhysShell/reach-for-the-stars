//! Sync logic: hybrid incremental scan + periodic full reconcile.
//!
//! - Incremental: walk newest-first until we hit an already-known
//!   `(starred_at, repo_id)` pair. Cheap; cannot detect unstars.
//! - Full reconcile: walk all pages, then mark anything we used to know
//!   about but no longer see as `Unstarred`.
//!
//! The sync is driven by a small async iterator over GitHub pages, which
//! is mocked out in unit tests so the merging logic can be exercised
//! without making any network calls.

use std::collections::{HashMap, HashSet};

use anyhow::Result;
use async_trait::async_trait;
use time::{Duration, OffsetDateTime};
use tracing::{debug, info};

use crate::github::{ApiStar, GitHubClient, StarPage};
use crate::model::{StarRecord, StarStatus, StarsEnvelope, SyncModeRecorded};

/// CLI-facing sync mode selector. clap's `ValueEnum` default rename
/// strategy (CamelCase -> kebab-case) gives us `auto` / `incremental` /
/// `full` on the command line.
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum SyncModeArg {
    /// Pick `full` if the last full reconcile is older than the configured
    /// interval (or has never run), otherwise `incremental`.
    Auto,
    /// Walk newest-first and stop at the previous watermark. Cheap.
    Incremental,
    /// Walk every page and mark missing repos as unstarred. Authoritative.
    Full,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResolvedSyncMode {
    Incremental,
    Full,
}

impl From<ResolvedSyncMode> for SyncModeRecorded {
    fn from(value: ResolvedSyncMode) -> Self {
        match value {
            ResolvedSyncMode::Incremental => Self::Incremental,
            ResolvedSyncMode::Full => Self::Full,
        }
    }
}

/// Resolve `auto` against the envelope's recorded full-reconcile time.
pub fn resolve_sync_mode(
    requested: SyncModeArg,
    envelope: &StarsEnvelope,
    now: OffsetDateTime,
    full_interval: Duration,
) -> ResolvedSyncMode {
    match requested {
        SyncModeArg::Incremental => ResolvedSyncMode::Incremental,
        SyncModeArg::Full => ResolvedSyncMode::Full,
        SyncModeArg::Auto => match envelope.sync.last_full_reconcile_at {
            Some(last) if now - last < full_interval => ResolvedSyncMode::Incremental,
            _ => ResolvedSyncMode::Full,
        },
    }
}

/// Abstraction over a paginated source of stars; lets us mock the network
/// in tests without dragging in a full HTTP mocking framework.
#[async_trait]
pub trait StarPageSource: Send + Sync {
    async fn fetch_page(&self, page: u32) -> Result<StarPage>;
}

pub struct GitHubStarSource<'a> {
    client: &'a GitHubClient,
    username: &'a str,
    per_page: u32,
}

impl<'a> GitHubStarSource<'a> {
    pub fn new(client: &'a GitHubClient, username: &'a str, per_page: u32) -> Self {
        Self {
            client,
            username,
            per_page,
        }
    }
}

#[async_trait]
impl StarPageSource for GitHubStarSource<'_> {
    async fn fetch_page(&self, page: u32) -> Result<StarPage> {
        self.client
            .fetch_starred_page(self.username, page, self.per_page)
            .await
    }
}

/// Incremental sync: walk newest-first until we hit a record we already
/// have at the same `starred_at`. Updates metadata and watermark; does NOT
/// produce tombstones for unstars (full reconcile handles those).
pub async fn incremental_sync<S: StarPageSource>(
    envelope: &mut StarsEnvelope,
    source: &S,
    now: OffsetDateTime,
) -> Result<()> {
    let index: HashMap<u64, usize> = envelope
        .stars
        .iter()
        .enumerate()
        .map(|(i, r)| (r.repo_id, i))
        .collect();
    let mut new_records: Vec<StarRecord> = Vec::new();
    let mut max_watermark: Option<(OffsetDateTime, u64)> = None;
    let mut pages_seen: u32 = 0;
    let mut stars_processed: u64 = 0;

    let mut page: u32 = 1;
    'outer: loop {
        let StarPage { stars, next_page } = source.fetch_page(page).await?;
        pages_seen += 1;
        if stars.is_empty() {
            break;
        }
        for star in &stars {
            stars_processed += 1;
            let candidate = (star.starred_at, star.repo.id);
            max_watermark = Some(max_watermark.map_or(candidate, |w| w.max(candidate)));

            if let Some(&i) = index.get(&star.repo.id) {
                if let Some(record) = envelope.stars.get_mut(i) {
                    if record.starred_at == star.starred_at && record.status == StarStatus::Active {
                        debug!(
                            page,
                            position = i,
                            repo_id = star.repo.id,
                            "incremental sync hit known record; stopping"
                        );
                        break 'outer;
                    }
                    apply_api_update(record, star, now);
                }
            } else {
                new_records.push(record_from_api(star, now));
            }
        }
        match next_page {
            Some(p) => page = p,
            None => break,
        }
    }

    let added = new_records.len();
    envelope.stars.extend(new_records);

    if let Some((sa, rid)) = max_watermark {
        envelope.sync.watermark_starred_at = Some(sa);
        envelope.sync.watermark_repo_id = Some(rid);
    }

    info!(
        pages = pages_seen,
        stars = stars_processed,
        added,
        "incremental sync complete"
    );
    Ok(())
}

/// Full reconcile: walk every page, then mark anything we used to know
/// about but no longer see as `Unstarred`.
pub async fn full_reconcile<S: StarPageSource>(
    envelope: &mut StarsEnvelope,
    source: &S,
    now: OffsetDateTime,
) -> Result<()> {
    let index: HashMap<u64, usize> = envelope
        .stars
        .iter()
        .enumerate()
        .map(|(i, r)| (r.repo_id, i))
        .collect();
    let mut new_records: Vec<StarRecord> = Vec::new();
    let mut current_ids: HashSet<u64> = HashSet::new();
    let mut max_watermark: Option<(OffsetDateTime, u64)> = None;
    let mut pages_seen: u32 = 0;

    let mut page: u32 = 1;
    loop {
        let StarPage { stars, next_page } = source.fetch_page(page).await?;
        pages_seen += 1;
        for star in &stars {
            current_ids.insert(star.repo.id);
            let candidate = (star.starred_at, star.repo.id);
            max_watermark = Some(max_watermark.map_or(candidate, |w| w.max(candidate)));

            if let Some(&i) = index.get(&star.repo.id) {
                if let Some(record) = envelope.stars.get_mut(i) {
                    apply_api_update(record, star, now);
                }
            } else {
                new_records.push(record_from_api(star, now));
            }
        }
        match next_page {
            Some(p) => page = p,
            None => break,
        }
    }

    // Tombstone everything we know about that wasn't in the full listing.
    let mut tombstoned: u64 = 0;
    for record in &mut envelope.stars {
        if !current_ids.contains(&record.repo_id) && record.status == StarStatus::Active {
            record.status = StarStatus::Unstarred;
            record.unstarred_at = Some(now);
            tombstoned += 1;
        }
    }

    let added = new_records.len();
    envelope.stars.extend(new_records);

    if let Some((sa, rid)) = max_watermark {
        envelope.sync.watermark_starred_at = Some(sa);
        envelope.sync.watermark_repo_id = Some(rid);
    }
    envelope.sync.last_full_reconcile_at = Some(now);

    info!(
        pages = pages_seen,
        active_seen = current_ids.len(),
        tombstoned,
        added,
        "full reconcile complete"
    );
    Ok(())
}

fn record_from_api(star: &ApiStar, now: OffsetDateTime) -> StarRecord {
    StarRecord {
        repo_id: star.repo.id,
        full_name: star.repo.full_name.clone(),
        html_url: star.repo.html_url.clone(),
        description: star.repo.description.clone(),
        language: star.repo.language.clone(),
        starred_at: star.starred_at,
        first_seen_at: now,
        last_seen_at: now,
        unstarred_at: None,
        status: StarStatus::Active,
    }
}

fn apply_api_update(record: &mut StarRecord, star: &ApiStar, now: OffsetDateTime) {
    record.status = StarStatus::Active;
    record.unstarred_at = None;
    record.last_seen_at = now;
    record.starred_at = star.starred_at;
    record.full_name.clone_from(&star.repo.full_name);
    record.html_url.clone_from(&star.repo.html_url);
    record.description.clone_from(&star.repo.description);
    record.language.clone_from(&star.repo.language);
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
    use crate::github::ApiRepo;
    use crate::model::Subject;
    use std::sync::Mutex;
    use time::macros::datetime;

    struct FakeSource {
        pages: Mutex<Vec<StarPage>>,
    }

    impl FakeSource {
        fn new(pages: Vec<StarPage>) -> Self {
            Self {
                pages: Mutex::new(pages),
            }
        }
    }

    #[async_trait]
    impl StarPageSource for FakeSource {
        async fn fetch_page(&self, _page: u32) -> Result<StarPage> {
            let mut guard = self.pages.lock().unwrap();
            if guard.is_empty() {
                Ok(StarPage {
                    stars: Vec::new(),
                    next_page: None,
                })
            } else {
                Ok(guard.remove(0))
            }
        }
    }

    fn star(id: u64, name: &str, t: OffsetDateTime) -> ApiStar {
        ApiStar {
            starred_at: t,
            repo: ApiRepo {
                id,
                full_name: name.to_owned(),
                html_url: format!("https://github.com/{name}"),
                description: None,
                language: None,
            },
        }
    }

    fn page(stars: Vec<ApiStar>, next: Option<u32>) -> StarPage {
        StarPage {
            stars,
            next_page: next,
        }
    }

    fn fresh_envelope() -> StarsEnvelope {
        StarsEnvelope {
            schema_version: crate::model::SCHEMA_VERSION,
            subject: Subject {
                kind: crate::model::SUBJECT_KIND_USER_STARRED.to_owned(),
                username: "octocat".to_owned(),
            },
            exported_at: datetime!(2026-05-09 00:00:00 UTC),
            sync: crate::model::SyncState::default(),
            stats: crate::model::Stats::default(),
            data_hash: None,
            stars: Vec::new(),
        }
    }

    #[tokio::test]
    async fn incremental_appends_new_stars() {
        let now = datetime!(2026-05-09 12:00:00 UTC);
        let mut env = fresh_envelope();
        let pages = vec![page(
            vec![
                star(1, "owner/a", datetime!(2026-05-09 10:00:00 UTC)),
                star(2, "owner/b", datetime!(2026-05-08 10:00:00 UTC)),
            ],
            None,
        )];
        let src = FakeSource::new(pages);
        incremental_sync(&mut env, &src, now).await.unwrap();
        assert_eq!(env.stars.len(), 2);
        assert_eq!(
            env.sync.watermark_starred_at,
            Some(datetime!(2026-05-09 10:00:00 UTC))
        );
        assert_eq!(env.sync.watermark_repo_id, Some(1));
    }

    #[tokio::test]
    async fn incremental_stops_at_known_watermark() {
        let now = datetime!(2026-05-09 12:00:00 UTC);
        let t_old = datetime!(2026-05-08 10:00:00 UTC);
        let t_new = datetime!(2026-05-09 10:00:00 UTC);

        let mut env = fresh_envelope();
        env.stars.push(StarRecord {
            repo_id: 2,
            full_name: "owner/b".to_owned(),
            html_url: "https://github.com/owner/b".to_owned(),
            description: None,
            language: None,
            starred_at: t_old,
            first_seen_at: t_old,
            last_seen_at: t_old,
            unstarred_at: None,
            status: StarStatus::Active,
        });

        // Page 1 has a new star, then the known record. Page 2 must not be fetched.
        let pages = vec![
            page(
                vec![star(1, "owner/a", t_new), star(2, "owner/b", t_old)],
                Some(2),
            ),
            page(vec![star(99, "owner/should-not-load", t_old)], None),
        ];
        let src = FakeSource::new(pages);
        incremental_sync(&mut env, &src, now).await.unwrap();

        // We should have one new record, and the loop must have stopped before page 2.
        assert_eq!(env.stars.len(), 2);
        assert!(env.stars.iter().all(|s| s.repo_id != 99));
    }

    #[tokio::test]
    async fn full_reconcile_marks_missing_records_as_unstarred() {
        let now = datetime!(2026-05-09 12:00:00 UTC);
        let t = datetime!(2026-05-08 10:00:00 UTC);

        let mut env = fresh_envelope();
        env.stars.push(StarRecord {
            repo_id: 1,
            full_name: "owner/a".to_owned(),
            html_url: "https://github.com/owner/a".to_owned(),
            description: None,
            language: None,
            starred_at: t,
            first_seen_at: t,
            last_seen_at: t,
            unstarred_at: None,
            status: StarStatus::Active,
        });
        env.stars.push(StarRecord {
            repo_id: 2,
            full_name: "owner/b".to_owned(),
            html_url: "https://github.com/owner/b".to_owned(),
            description: None,
            language: None,
            starred_at: t,
            first_seen_at: t,
            last_seen_at: t,
            unstarred_at: None,
            status: StarStatus::Active,
        });

        // Only repo 1 is still starred; repo 2 should be tombstoned.
        let pages = vec![page(vec![star(1, "owner/a", t)], None)];
        let src = FakeSource::new(pages);
        full_reconcile(&mut env, &src, now).await.unwrap();

        let r1 = env.stars.iter().find(|r| r.repo_id == 1).unwrap();
        let r2 = env.stars.iter().find(|r| r.repo_id == 2).unwrap();
        assert_eq!(r1.status, StarStatus::Active);
        assert_eq!(r2.status, StarStatus::Unstarred);
        assert_eq!(r2.unstarred_at, Some(now));
        assert_eq!(env.sync.last_full_reconcile_at, Some(now));
    }

    #[tokio::test]
    async fn full_reconcile_revives_unstarred_record_when_re_seen() {
        let now = datetime!(2026-05-09 12:00:00 UTC);
        let t_old = datetime!(2026-05-08 10:00:00 UTC);
        let t_new = datetime!(2026-05-09 09:00:00 UTC);

        let mut env = fresh_envelope();
        env.stars.push(StarRecord {
            repo_id: 1,
            full_name: "owner/a".to_owned(),
            html_url: "https://github.com/owner/a".to_owned(),
            description: None,
            language: None,
            starred_at: t_old,
            first_seen_at: t_old,
            last_seen_at: t_old,
            unstarred_at: Some(t_old),
            status: StarStatus::Unstarred,
        });

        let pages = vec![page(vec![star(1, "owner/a", t_new)], None)];
        let src = FakeSource::new(pages);
        full_reconcile(&mut env, &src, now).await.unwrap();

        let r = env.stars.iter().find(|r| r.repo_id == 1).unwrap();
        assert_eq!(r.status, StarStatus::Active);
        assert_eq!(r.unstarred_at, None);
        assert_eq!(r.starred_at, t_new);
        assert_eq!(r.last_seen_at, now);
    }

    #[test]
    fn resolve_sync_mode_uses_age_of_last_reconcile() {
        let now = datetime!(2026-05-09 12:00:00 UTC);
        let mut env = fresh_envelope();

        // No prior reconcile -> full
        assert_eq!(
            resolve_sync_mode(SyncModeArg::Auto, &env, now, Duration::hours(24)),
            ResolvedSyncMode::Full
        );

        // Recent reconcile -> incremental
        env.sync.last_full_reconcile_at = Some(now - Duration::hours(2));
        assert_eq!(
            resolve_sync_mode(SyncModeArg::Auto, &env, now, Duration::hours(24)),
            ResolvedSyncMode::Incremental
        );

        // Stale reconcile -> full
        env.sync.last_full_reconcile_at = Some(now - Duration::hours(48));
        assert_eq!(
            resolve_sync_mode(SyncModeArg::Auto, &env, now, Duration::hours(24)),
            ResolvedSyncMode::Full
        );

        // Explicit overrides win
        assert_eq!(
            resolve_sync_mode(SyncModeArg::Incremental, &env, now, Duration::hours(24)),
            ResolvedSyncMode::Incremental
        );
        assert_eq!(
            resolve_sync_mode(SyncModeArg::Full, &env, now, Duration::hours(1)),
            ResolvedSyncMode::Full
        );
    }
}
