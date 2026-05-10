//! `reach-stars` — export GitHub stargazers to a JSON envelope with hybrid
//! incremental + periodic-reconcile sync.
//!
//! See `docs/`-equivalent in `README.md` for usage. This crate intentionally
//! ships only the `export` subcommand for now; encryption and remote upload
//! are deliberate non-goals at this stage.

#![cfg_attr(
    test,
    allow(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::panic,
        clippy::indexing_slicing
    )
)]

mod github;
mod hash;
mod io;
mod model;
mod sync;

use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Duration;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use secrecy::{SecretBox, SecretString};
use time::{Duration as TimeDuration, OffsetDateTime};
use tracing::{error, info};
use tracing_subscriber::EnvFilter;
use url::Url;

use crate::github::GitHubClient;
use crate::hash::compute_data_hash;
use crate::io::{read_envelope, write_envelope};
use crate::model::{compute_stats, StarsEnvelope};
use crate::sync::{
    full_reconcile, incremental_sync, resolve_sync_mode, GitHubStarSource, ResolvedSyncMode,
    SyncModeArg,
};

#[derive(Debug, Parser)]
#[command(
    name = "reach-stars",
    version,
    about = "Export GitHub stargazers to a JSON envelope with hybrid incremental + reconcile sync.",
    long_about = None,
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Fetch stars from the GitHub API and write the envelope.
    Export(ExportArgs),
}

#[derive(Debug, clap::Args)]
struct ExportArgs {
    /// GitHub username whose stars should be exported.
    #[arg(short, long, env = "GITHUB_USERNAME")]
    username: String,

    /// Output path for the JSON envelope.
    #[arg(short, long, default_value = "assets/data/stars.json")]
    output: PathBuf,

    /// GitHub API token. Optional, but raises the rate limit from 60 to
    /// 5000 req/h. Reads from `GITHUB_TOKEN` env var if not passed.
    #[arg(long, env = "GITHUB_TOKEN", hide_env_values = true)]
    token: Option<String>,

    /// Sync mode: `auto`, `incremental`, or `full`.
    #[arg(long, value_enum, default_value = "auto")]
    sync_mode: SyncModeArg,

    /// Base URL for the GitHub REST API (override for GitHub Enterprise).
    #[arg(long, default_value = "https://api.github.com/")]
    api_base: Url,

    /// Items per page (1..=100). Larger values mean fewer requests.
    #[arg(long, default_value_t = 100)]
    per_page: u32,

    /// HTTP timeout in seconds for each GitHub request.
    #[arg(long, default_value_t = 30)]
    timeout_secs: u64,

    /// Hours between automatic full reconciles when sync mode is `auto`.
    #[arg(long, default_value_t = 24)]
    full_reconcile_interval_hours: i64,

    /// User-Agent header sent to GitHub.
    #[arg(long, default_value = concat!("reach-stars/", env!("CARGO_PKG_VERSION")))]
    user_agent: String,
}

fn main() -> ExitCode {
    init_tracing();
    let cli = Cli::parse();

    let runtime = match tokio::runtime::Builder::new_multi_thread()
        .enable_io()
        .enable_time()
        .build()
    {
        Ok(rt) => rt,
        Err(err) => {
            error!(error = %err, "failed to start tokio runtime");
            return ExitCode::FAILURE;
        }
    };

    let result = runtime.block_on(async {
        match cli.command {
            Command::Export(args) => run_export(args).await,
        }
    });

    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            error!(error = ?err, "reach-stars failed");
            ExitCode::FAILURE
        }
    }
}

fn init_tracing() {
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_writer(std::io::stderr)
        .init();
}

async fn run_export(args: ExportArgs) -> Result<()> {
    if !(1..=100).contains(&args.per_page) {
        anyhow::bail!("--per-page must be in 1..=100, got {}", args.per_page);
    }

    let now = OffsetDateTime::now_utc();
    let token: Option<SecretString> = args
        .token
        .filter(|t| !t.is_empty())
        .map(|t| SecretBox::new(t.into_boxed_str()));

    let client = GitHubClient::new(
        args.api_base.clone(),
        token,
        &args.user_agent,
        Duration::from_secs(args.timeout_secs),
    )?;

    let mut envelope = read_envelope(&args.output)
        .with_context(|| {
            format!(
                "failed to read existing envelope at {}",
                args.output.display()
            )
        })?
        .unwrap_or_else(|| StarsEnvelope::empty(&args.username, now));

    if envelope.subject.username != args.username {
        info!(
            previous = %envelope.subject.username,
            requested = %args.username,
            "envelope was for a different user; resetting"
        );
        envelope = StarsEnvelope::empty(&args.username, now);
    }

    let mode = resolve_sync_mode(
        args.sync_mode,
        &envelope,
        now,
        TimeDuration::hours(args.full_reconcile_interval_hours),
    );
    info!(?mode, requested = ?args.sync_mode, "resolved sync mode");

    let source = GitHubStarSource::new(&client, &args.username, args.per_page);

    match mode {
        ResolvedSyncMode::Incremental => incremental_sync(&mut envelope, &source, now).await?,
        ResolvedSyncMode::Full => full_reconcile(&mut envelope, &source, now).await?,
    }

    envelope.exported_at = now;
    envelope.sync.mode = mode.into();
    envelope.stats = compute_stats(&envelope.stars);
    envelope.data_hash = Some(compute_data_hash(&envelope.stars)?);

    write_envelope(&args.output, &envelope)?;
    info!(
        path = %args.output.display(),
        active = envelope.stats.active_count,
        tombstones = envelope.stats.tombstone_count,
        "wrote envelope"
    );

    Ok(())
}
