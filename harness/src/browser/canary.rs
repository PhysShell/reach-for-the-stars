use anyhow::{Context, Result};
use reqwest::header::{HeaderMap, HeaderName, HeaderValue, ACCEPT};
use serde_json::Value;
use std::sync::Arc;

use crate::config::Canary;

#[derive(Debug)]
pub struct CanaryResult {
    pub ok: bool,
    pub status: u16,
    pub observed: Option<String>,
}

/// Build the canary HTTP client: cookie jar + a realistic UA + sensible
/// default headers, with config `headers` overriding the defaults.
///
/// A default reqwest client (UA `reqwest/x.y`, no `Accept`) gets 403/404 from
/// WAFs and many JSON APIs. This makes the probe look like a normal browser
/// XHR. It is NOT an anti-bot bypass — endpoints behind active client
/// attestation (proof-of-work / Turnstile tokens) are still out of reach.
pub fn build_client(
    jar: Arc<reqwest::cookie::Jar>,
    cfg: &Canary,
) -> Result<reqwest::Client> {
    let mut headers = HeaderMap::new();
    // Default Accept matches what a browser XHR sends; overridable below.
    headers.insert(
        ACCEPT,
        HeaderValue::from_static("application/json, text/plain, */*"),
    );
    for (k, v) in &cfg.headers {
        let name = HeaderName::from_bytes(k.as_bytes())
            .with_context(|| format!("invalid canary header name {k:?}"))?;
        let val = HeaderValue::from_str(v)
            .with_context(|| format!("invalid canary header value for {k:?}"))?;
        headers.insert(name, val);
    }
    reqwest::Client::builder()
        .cookie_provider(jar)
        .user_agent(cfg.user_agent.clone())
        .default_headers(headers)
        .build()
        .context("build canary client")
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
