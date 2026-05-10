//! Minimal GitHub REST client for `GET /users/{username}/starred`.
//!
//! Uses the `application/vnd.github.star+json` media type to receive
//! `starred_at` alongside the repo payload, and parses RFC 5988 `Link`
//! headers to drive page-by-page traversal.

use std::time::Duration;

use anyhow::{anyhow, Context, Result};
use reqwest::header::{HeaderMap, HeaderName, HeaderValue, ACCEPT, AUTHORIZATION, LINK};
use reqwest::StatusCode;
use secrecy::{ExposeSecret, SecretString};
use serde::Deserialize;
use time::OffsetDateTime;
use url::Url;

const STAR_ACCEPT: &str = "application/vnd.github.star+json";
const GITHUB_API_VERSION: &str = "2022-11-28";
const X_GITHUB_API_VERSION: HeaderName = HeaderName::from_static("x-github-api-version");

#[derive(Debug, Clone, Deserialize)]
pub struct ApiStar {
    #[serde(with = "time::serde::rfc3339")]
    pub starred_at: OffsetDateTime,
    pub repo: ApiRepo,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ApiRepo {
    pub id: u64,
    pub full_name: String,
    pub html_url: String,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub language: Option<String>,
}

#[derive(Debug)]
pub struct GitHubClient {
    client: reqwest::Client,
    api_base: Url,
}

impl GitHubClient {
    pub fn new(
        api_base: Url,
        token: Option<SecretString>,
        user_agent: &str,
        timeout: Duration,
    ) -> Result<Self> {
        let mut headers = HeaderMap::new();
        headers.insert(ACCEPT, HeaderValue::from_static(STAR_ACCEPT));
        headers.insert(
            X_GITHUB_API_VERSION,
            HeaderValue::from_static(GITHUB_API_VERSION),
        );
        if let Some(token) = token {
            let raw = format!("Bearer {}", token.expose_secret());
            let mut value =
                HeaderValue::from_str(&raw).context("invalid characters in GitHub token")?;
            value.set_sensitive(true);
            headers.insert(AUTHORIZATION, value);
        }

        let client = reqwest::Client::builder()
            .default_headers(headers)
            .user_agent(user_agent)
            .timeout(timeout)
            .build()
            .context("failed to build HTTP client")?;

        Ok(Self { client, api_base })
    }

    /// Fetch a single page of stars in newest-first order.
    ///
    /// Returns the parsed page and the page number of the next page if any.
    pub async fn fetch_starred_page(
        &self,
        username: &str,
        page: u32,
        per_page: u32,
    ) -> Result<StarPage> {
        let mut url = self
            .api_base
            .join(&format!("users/{username}/starred"))
            .with_context(|| format!("failed to build URL for user {username}"))?;
        url.query_pairs_mut()
            .append_pair("sort", "created")
            .append_pair("direction", "desc")
            .append_pair("per_page", &per_page.to_string())
            .append_pair("page", &page.to_string());

        let response = self
            .client
            .get(url.clone())
            .send()
            .await
            .with_context(|| format!("HTTP request to {url} failed"))?;

        let status = response.status();
        let next_page = next_page_from_link_header(response.headers().get(LINK))?;

        if !status.is_success() {
            return Err(translate_error(status, response).await);
        }

        let stars: Vec<ApiStar> = response
            .json()
            .await
            .context("failed to decode GitHub stars JSON")?;
        Ok(StarPage { stars, next_page })
    }
}

#[derive(Debug)]
pub struct StarPage {
    pub stars: Vec<ApiStar>,
    pub next_page: Option<u32>,
}

async fn translate_error(status: StatusCode, response: reqwest::Response) -> anyhow::Error {
    // Pull the rate-limit hint up front so we can include it even if the body
    // read fails for any reason.
    let rate_remaining = response
        .headers()
        .get("x-ratelimit-remaining")
        .and_then(|v| v.to_str().ok())
        .map(str::to_owned);
    let rate_reset = response
        .headers()
        .get("x-ratelimit-reset")
        .and_then(|v| v.to_str().ok())
        .map(str::to_owned);

    let body_snippet = match response.text().await {
        Ok(body) => body.chars().take(512).collect::<String>(),
        Err(err) => format!("<unreadable body: {err}>"),
    };

    if status == StatusCode::FORBIDDEN && rate_remaining.as_deref() == Some("0") {
        return anyhow!(
            "GitHub API rate limit exhausted (status {status}, reset epoch={reset:?}); \
             body snippet: {snippet}",
            reset = rate_reset,
            snippet = body_snippet,
        );
    }
    if status == StatusCode::TOO_MANY_REQUESTS {
        return anyhow!(
            "GitHub API throttled this request (status 429); body snippet: {body_snippet}"
        );
    }
    if status == StatusCode::UNAUTHORIZED {
        return anyhow!("GitHub API rejected the credentials (status 401)");
    }
    if status == StatusCode::NOT_FOUND {
        return anyhow!("GitHub API returned 404; user not found or not visible");
    }
    anyhow!("GitHub API request failed: status {status}; body snippet: {body_snippet}")
}

/// Parse the `Link` header to extract the page number for `rel="next"`.
fn next_page_from_link_header(value: Option<&HeaderValue>) -> Result<Option<u32>> {
    let Some(value) = value else {
        return Ok(None);
    };
    let raw = value.to_str().context("Link header is not valid UTF-8")?;

    for part in raw.split(',') {
        let part = part.trim();
        let Some(rel_pos) = part.find("rel=") else {
            continue;
        };
        let rel = &part[rel_pos + "rel=".len()..];
        let rel = rel.trim_matches(|c: char| c == '"' || c.is_whitespace());
        if rel != "next" {
            continue;
        }
        let Some(start) = part.find('<') else {
            continue;
        };
        let Some(end) = part[start + 1..].find('>') else {
            continue;
        };
        let raw_url = &part[start + 1..start + 1 + end];
        let parsed = Url::parse(raw_url)
            .with_context(|| format!("invalid URL in Link header: {raw_url}"))?;
        for (key, val) in parsed.query_pairs() {
            if key == "page" {
                return val
                    .parse::<u32>()
                    .map(Some)
                    .with_context(|| format!("invalid page value in Link header: {val}"));
            }
        }
    }
    Ok(None)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;

    fn link(value: &'static str) -> HeaderValue {
        HeaderValue::from_static(value)
    }

    #[test]
    fn link_header_with_next_returns_next_page_number() {
        let h = link(
            "<https://api.github.com/user/starred?page=2>; rel=\"next\", \
             <https://api.github.com/user/starred?page=10>; rel=\"last\"",
        );
        assert_eq!(next_page_from_link_header(Some(&h)).unwrap(), Some(2));
    }

    #[test]
    fn link_header_without_next_returns_none() {
        let h = link("<https://api.github.com/user/starred?page=10>; rel=\"last\"");
        assert_eq!(next_page_from_link_header(Some(&h)).unwrap(), None);
    }

    #[test]
    fn missing_link_header_returns_none() {
        assert_eq!(next_page_from_link_header(None).unwrap(), None);
    }

    #[test]
    fn link_header_with_extra_params_in_next_url_still_parses() {
        let h = link(
            "<https://api.github.com/user/starred?per_page=100&page=3&sort=created>; \
             rel=\"next\"",
        );
        assert_eq!(next_page_from_link_header(Some(&h)).unwrap(), Some(3));
    }
}
