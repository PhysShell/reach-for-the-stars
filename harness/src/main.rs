use anyhow::Result;
use clap::{Parser, Subcommand};
use ed25519_dalek::SigningKey;
use rand::rngs::OsRng;
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
    /// Generate an ed25519 keypair; print secret (→ GitHub Secret) and pubkey (→ repo).
    GenKeys {
        /// Emit a single JSON line `{"secret": "...", "pubkey": "..."}` on stdout
        /// instead of the human-readable two-stream output. Intended for scripts.
        #[arg(long)]
        json: bool,
    },
    /// Open a headed browser, wait for manual login, then snapshot and upload.
    Seed,
    /// Daily run: restore snapshot, verify canary, do work, save snapshot.
    Run,
    /// Verify the latest snapshot decrypts and signature is valid (no browser).
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
        Cmd::GenKeys { json } => gen_keys(json),
        Cmd::Seed => harness::commands::seed::run(&Config::load(&cli.config)?).await,
        Cmd::Run => harness::commands::run::run(&Config::load(&cli.config)?).await,
        Cmd::Verify => harness::commands::verify::run(&Config::load(&cli.config)?).await,
    }
}

fn gen_keys(json: bool) -> Result<()> {
    let signing_key = SigningKey::generate(&mut OsRng);
    let verifying_key = signing_key.verifying_key();

    let secret_hex = hex::encode(signing_key.to_bytes());
    let pubkey_hex = hex::encode(verifying_key.to_bytes());

    if json {
        let out = serde_json::json!({
            "secret": secret_hex,
            "pubkey": pubkey_hex,
        });
        println!("{}", serde_json::to_string(&out)?);
    } else {
        eprintln!("==> ed25519 keypair generated");
        eprintln!();
        eprintln!("  Secret (→ HARNESS_SIGN_SECRET in GitHub Secrets):");
        eprintln!("  {secret_hex}");
        eprintln!();
        eprintln!("  Public key (→ sign-pubkey.hex in repo, commit this):");
        println!("{pubkey_hex}");
    }

    Ok(())
}
