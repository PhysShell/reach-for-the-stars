use anyhow::{Context, Result};
use serde_json::Value;

#[derive(Debug)]
pub struct CanaryResult {
    pub ok: bool,
    pub status: u16,
    pub observed: Option<String>,
}

/// Hits an authenticated JSON endpoint and asserts a stable identity field.
///
/// The `client` MUST already carry the cookies copied from the live CDP session
/// (via `Network.getAllCookies` → `reqwest::cookie::Jar`); otherwise the canary
/// will see an unauthenticated state and incorrectly report "logged out".
pub async fn check(
    client: &reqwest::Client,
    url: &str,
    expected_status: u16,
    field_pointer: &str,
    expected_value: &str,
) -> Result<CanaryResult> {
    let res = client.get(url).send().await.context("canary GET")?;
    let status = res.status().as_u16();
    if status != expected_status {
        return Ok(CanaryResult {
            ok: false,
            status,
            observed: None,
        });
    }

    let body: Value = res.json().await.context("canary JSON parse")?;
    let observed = body
        .pointer(field_pointer)
        .and_then(|v| v.as_str())
        .map(str::to_string);

    let ok = observed.as_deref() == Some(expected_value);
    Ok(CanaryResult {
        ok,
        status,
        observed,
    })
}
