use anyhow::{Context, Result};
use secrecy::{ExposeSecret, SecretString};
use std::str::FromStr;

pub struct AgeIdentity(SecretString);

impl AgeIdentity {
    /// Read identity from env, then immediately remove it from process env so any
    /// subsequently-spawned children cannot inherit it.
    pub fn take_from_env(var: &str) -> Result<Self> {
        let raw = std::env::var(var).with_context(|| format!("env var {var} not set"))?;
        std::env::remove_var(var);
        Ok(Self(SecretString::from(raw)))
    }

    pub fn parse(&self) -> Result<age::x25519::Identity> {
        let raw = self.0.expose_secret();
        // age-keygen writes a multi-line file with comment headers; GitHub
        // Secrets preserves newlines, so extract the key line explicitly.
        let key_line = raw
            .lines()
            .find(|l| l.starts_with("AGE-SECRET-KEY-"))
            .unwrap_or(raw.trim());
        age::x25519::Identity::from_str(self.0.expose_secret())
            .map_err(|e| anyhow::anyhow!("invalid age identity: {e}"))
    }
}

pub struct MinisignKey(SecretString);

impl MinisignKey {
    pub fn take_from_env(var: &str) -> Result<Self> {
        let raw = std::env::var(var).with_context(|| format!("env var {var} not set"))?;
        std::env::remove_var(var);
        Ok(Self(SecretString::from(raw)))
    }

    pub fn expose(&self) -> &str {
        self.0.expose_secret()
    }
}

pub struct StoreCredential {
    pub access_key: SecretString,
    pub secret_key: SecretString,
}

impl StoreCredential {
    pub fn take_from_env(access_var: &str, secret_var: &str) -> Result<Self> {
        let ak =
            std::env::var(access_var).with_context(|| format!("env var {access_var} not set"))?;
        let sk =
            std::env::var(secret_var).with_context(|| format!("env var {secret_var} not set"))?;
        std::env::remove_var(access_var);
        std::env::remove_var(secret_var);
        Ok(Self {
            access_key: SecretString::from(ak),
            secret_key: SecretString::from(sk),
        })
    }
}
