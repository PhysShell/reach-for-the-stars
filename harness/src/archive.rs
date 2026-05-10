use crate::browser::cdp::StorageState;
use anyhow::{Context, Result};
use bytes::Bytes;

pub fn compress(state: &StorageState) -> Result<Bytes> {
    let json = serde_json::to_vec(state).context("serialize state")?;
    let compressed = zstd::encode_all(json.as_slice(), 3).context("zstd compress")?;
    Ok(Bytes::from(compressed))
}

pub fn decompress(data: &[u8]) -> Result<StorageState> {
    let json = zstd::decode_all(data).context("zstd decompress")?;
    serde_json::from_slice(&json).context("deserialize state")
}
