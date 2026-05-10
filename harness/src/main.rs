use anyhow::Result;
use clap::{Parser, Subcommand};
use std::path::PathBuf;

use harness::config::Config;

#[derive(Parser)]
#[command(name = "harness", version)]
struct Cli {
    #[arg(long, env = "HARNESS_CONFIG", default_value = "config.toml")]
    config: PathBuf,

    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Print setup instructions for generating age + minisign keypairs.
    GenKeys,
    /// Open a headed browser, wait for manual login, then snapshot+upload.
    Seed,
    /// Daily run: restore snapshot, verify canary, do work, save snapshot.
    Run,
    /// Verify the latest snapshot decrypts and signature is valid.
    Verify,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("harness=info")),
        )
        .with_writer(std::io::stderr)
        .init();

    let cli = Cli::parse();

    match cli.cmd {
        Cmd::GenKeys => commands::gen_keys(),
        Cmd::Seed => commands::seed(&Config::load(&cli.config)?).await,
        Cmd::Run => commands::run(&Config::load(&cli.config)?).await,
        Cmd::Verify => commands::verify(&Config::load(&cli.config)?).await,
    }
}

mod commands {
    use anyhow::Result;
    use harness::config::Config;

    pub fn gen_keys() -> Result<()> {
        println!("Run scripts/bootstrap.sh from inside `nix develop`.");
        println!("It will generate age + minisign keypairs and seal them into GitHub Secrets.");
        Ok(())
    }

    pub async fn seed(_cfg: &Config) -> Result<()> {
        anyhow::bail!("seed: not yet implemented");
    }

    pub async fn run(_cfg: &Config) -> Result<()> {
        anyhow::bail!("run: not yet implemented");
    }

    pub async fn verify(_cfg: &Config) -> Result<()> {
        anyhow::bail!("verify: not yet implemented");
    }
}
