use anyhow::{Context, Result};
use ed25519_dalek::{Signature, Signer, SigningKey, Verifier, VerifyingKey, SECRET_KEY_LENGTH};

pub fn parse_signing_key(raw: &str) -> Result<SigningKey> {
    let bytes = hex::decode(raw.trim()).context("signing key must be hex-encoded")?;
    let arr: [u8; SECRET_KEY_LENGTH] = bytes
        .as_slice()
        .try_into()
        .map_err(|_| anyhow::anyhow!("signing key wrong length"))?;
    Ok(SigningKey::from_bytes(&arr))
}

pub fn parse_verifying_key(raw: &str) -> Result<VerifyingKey> {
    let bytes = hex::decode(raw.trim()).context("pubkey must be hex-encoded")?;
    let arr: [u8; 32] = bytes
        .as_slice()
        .try_into()
        .map_err(|_| anyhow::anyhow!("pubkey wrong length"))?;
    VerifyingKey::from_bytes(&arr).map_err(|e| anyhow::anyhow!("invalid pubkey: {e}"))
}

pub fn sign(key: &SigningKey, payload: &[u8]) -> Vec<u8> {
    key.sign(payload).to_bytes().to_vec()
}

pub fn verify(pubkey: &VerifyingKey, payload: &[u8], sig: &[u8]) -> Result<()> {
    let s = Signature::from_slice(sig).context("signature wrong length")?;
    pubkey
        .verify(payload, &s)
        .map_err(|e| anyhow::anyhow!("signature verification failed: {e}"))
}
