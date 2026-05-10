use age::x25519::{Identity, Recipient};
use anyhow::{Context, Result};
use bytes::Bytes;
use std::io::{Read, Write};
use std::str::FromStr;

pub fn read_recipient(path: &std::path::Path) -> Result<Recipient> {
    let raw = std::fs::read_to_string(path).context("read recipient file")?;
    let trimmed = raw.trim();
    Recipient::from_str(trimmed).map_err(|e| anyhow::anyhow!("invalid age recipient: {e}"))
}

pub fn encrypt(plaintext: &[u8], recipient: &Recipient) -> Result<Bytes> {
    let r: &dyn age::Recipient = recipient;
    let encryptor = age::Encryptor::with_recipients(std::iter::once(r))
        .map_err(|e| anyhow::anyhow!("encryptor build: {e}"))?;
    let mut out = Vec::new();
    let mut writer = encryptor.wrap_output(&mut out)?;
    writer.write_all(plaintext)?;
    writer.finish()?;
    Ok(out.into())
}

pub fn decrypt(ciphertext: &[u8], identity: &Identity) -> Result<Bytes> {
    let decryptor = age::Decryptor::new(ciphertext)?;
    let id: &dyn age::Identity = identity;
    let mut out = Vec::new();
    let mut reader = decryptor.decrypt(std::iter::once(id))?;
    reader.read_to_end(&mut out)?;
    Ok(out.into())
}
