use anyhow::{bail, Context, Result};
use chrono::{DateTime, TimeZone, Utc};
use reqwest::{Client, StatusCode};
use serde::{Deserialize, Deserializer, Serialize};
use std::collections::HashMap;
use std::time::Duration;
use tracing::{info, warn};

// ── Timestamp deserializer ────────────────────────────────────────────────────
//
// ChatGPT's API returns timestamps as either Unix f64 seconds or ISO 8601
// strings depending on the endpoint and API version. This handles both.

fn deserialize_ts<'de, D>(d: D) -> Result<Option<f64>, D::Error>
where
    D: Deserializer<'de>,
{
    let v = serde_json::Value::deserialize(d)?;
    match v {
        serde_json::Value::Null => Ok(None),
        serde_json::Value::Number(n) => Ok(n.as_f64()),
        serde_json::Value::String(s) => DateTime::parse_from_rfc3339(&s)
            .map(|dt| Some(dt.timestamp() as f64))
            .map_err(serde::de::Error::custom),
        other => Err(serde::de::Error::custom(format!(
            "expected timestamp, got {other}"
        ))),
    }
}

// ── API types ─────────────────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct SessionResponse {
    #[serde(rename = "accessToken")]
    pub access_token: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct ConversationListResponse {
    pub items: Vec<ConversationSummary>,
}

#[derive(Debug, Deserialize, Clone)]
pub struct ConversationSummary {
    pub id: String,
    #[serde(default)]
    pub title: Option<String>,
    #[serde(default, deserialize_with = "deserialize_ts")]
    pub update_time: Option<f64>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct ConversationDetail {
    // List endpoint uses "id"; detail endpoint uses "conversation_id".
    #[serde(alias = "conversation_id")]
    pub id: String,
    #[serde(default)]
    pub title: Option<String>,
    #[serde(default, deserialize_with = "deserialize_ts")]
    pub create_time: Option<f64>,
    #[serde(default, deserialize_with = "deserialize_ts")]
    pub update_time: Option<f64>,
    #[serde(default)]
    pub mapping: Option<HashMap<String, MappingNode>>,
    #[serde(default)]
    pub current_node: Option<String>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct MappingNode {
    // The id is the HashMap key; some nodes omit it from the body.
    #[serde(default)]
    pub id: String,
    #[serde(default)]
    pub message: Option<Message>,
    #[serde(default)]
    pub parent: Option<String>,
    #[serde(default)]
    pub children: Vec<String>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct Message {
    #[serde(default)]
    pub id: String,
    #[serde(default)]
    pub author: Author,
    #[serde(default, deserialize_with = "deserialize_ts")]
    pub create_time: Option<f64>,
    #[serde(default)]
    pub content: Content,
    #[serde(default)]
    pub status: Option<String>,
}

#[derive(Debug, Deserialize, Serialize, Default)]
pub struct Author {
    #[serde(default)]
    pub role: String,
}

#[derive(Debug, Deserialize, Serialize, Default)]
pub struct Content {
    #[serde(default)]
    pub content_type: String,
    #[serde(default)]
    pub parts: Option<Vec<serde_json::Value>>,
}

// ── Client ───────────────────────────────────────────────────────────────────

pub struct ChatGptClient {
    http: Client,
    base_url: String,
    access_token: String,
    rate_delay: Duration,
    max_retries: u32,
}

impl ChatGptClient {
    pub async fn new(
        http: Client,
        base_url: &str,
        rate_delay: Duration,
        max_retries: u32,
    ) -> Result<Self> {
        let url = format!("{base_url}/api/auth/session");
        let resp = http
            .get(&url)
            .send()
            .await
            .context("fetch /api/auth/session")?;

        if !resp.status().is_success() {
            bail!(
                "/api/auth/session returned {} — session may have expired; \
                 re-run `harness seed`",
                resp.status().as_u16()
            );
        }

        let session: SessionResponse = resp
            .json()
            .await
            .context("parse /api/auth/session JSON")?;

        let access_token = session
            .access_token
            .filter(|t| !t.is_empty())
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "no accessToken in session response — session may be invalid or expired; \
                     re-run `harness seed`"
                )
            })?;

        info!(preview = %&access_token[..20.min(access_token.len())], "access token acquired");

        Ok(Self {
            http,
            base_url: base_url.to_string(),
            access_token,
            rate_delay,
            max_retries,
        })
    }

    async fn get_json<T: serde::de::DeserializeOwned>(&self, url: &str) -> Result<T> {
        let mut attempt = 0u32;
        loop {
            tokio::time::sleep(self.rate_delay).await;

            let resp = self
                .http
                .get(url)
                .header("Authorization", format!("Bearer {}", self.access_token))
                .header("Accept", "application/json")
                .header("Accept-Language", "en-US,en;q=0.9")
                .header("Referer", "https://chatgpt.com/")
                .header("sec-fetch-dest", "empty")
                .header("sec-fetch-mode", "cors")
                .header("sec-fetch-site", "same-origin")
                .send()
                .await
                .with_context(|| format!("GET {url}"))?;

            let status = resp.status();

            if status == StatusCode::TOO_MANY_REQUESTS {
                attempt += 1;
                if attempt > self.max_retries {
                    bail!("GET {url}: rate limited after {} retries", self.max_retries);
                }
                let backoff = resp
                    .headers()
                    .get("retry-after")
                    .and_then(|v| v.to_str().ok())
                    .and_then(|v| v.parse::<u64>().ok())
                    .map(Duration::from_secs)
                    .unwrap_or_else(|| Duration::from_secs(30 * 2u64.pow(attempt.min(4))));
                warn!(attempt, wait_secs = backoff.as_secs(), "429 — backing off");
                tokio::time::sleep(backoff).await;
                continue;
            }

            if status == StatusCode::UNAUTHORIZED {
                bail!(
                    "GET {url}: 401 — session expired; re-run `harness seed`"
                );
            }

            if !status.is_success() {
                let body = resp.text().await.unwrap_or_default();
                bail!("GET {url}: status {} body: {body}", status.as_u16());
            }

            // Parse as Value first so we can detect API-level errors that
            // arrive as HTTP 200 with a {"detail": ...} body.
            let raw = resp
                .text()
                .await
                .with_context(|| format!("read body from {url}"))?;

            tracing::debug!(url, body = %&raw[..raw.len().min(500)], "response body");

            let body: serde_json::Value = serde_json::from_str(&raw)
                .with_context(|| format!("parse JSON from {url}"))?;

            // ChatGPT returns 200 with error bodies in several shapes:
            //   {"detail": {"code": "conversation_inaccessible", "message": "..."}}
            //   {"detail": "some string message"}
            let api_err = body.get("detail").and_then(|d| match d {
                serde_json::Value::String(s) => Some(s.clone()),
                serde_json::Value::Object(_) => {
                    let code = d.get("code").and_then(|c| c.as_str()).unwrap_or("api_error");
                    let msg  = d.get("message").and_then(|m| m.as_str()).unwrap_or("");
                    Some(format!("{code}: {msg}"))
                }
                _ => None,
            });
            if let Some(err) = api_err {
                bail!("API error from {url}: {err}");
            }

            return serde_json::from_value(body)
                .with_context(|| format!("deserialize response from {url}"));
        }
    }

    pub async fn list_conversations(
        &self,
        offset: u32,
        limit: u32,
    ) -> Result<ConversationListResponse> {
        let url = format!(
            "{}/backend-api/conversations?offset={offset}&limit={limit}",
            self.base_url
        );
        self.get_json(&url).await
    }

    pub async fn get_conversation(&self, id: &str) -> Result<ConversationDetail> {
        // Note: detail endpoint is singular /conversation/, not /conversations/
        let url = format!("{}/backend-api/conversation/{id}", self.base_url);
        self.get_json(&url).await
    }
}

// ── Markdown rendering ───────────────────────────────────────────────────────

fn walk_to_current<'a>(
    mapping: &'a HashMap<String, MappingNode>,
    current: &str,
) -> Vec<&'a MappingNode> {
    let mut path = Vec::new();
    let mut id = current;
    loop {
        let Some(node) = mapping.get(id) else { break };
        path.push(node);
        match &node.parent {
            Some(p) => id = p.as_str(),
            None => break,
        }
    }
    path.reverse();
    path
}

fn ts_fmt(ts: f64) -> String {
    Utc.timestamp_opt(ts as i64, 0)
        .single()
        .map(|dt| dt.to_rfc3339())
        .unwrap_or_else(|| format!("{ts}"))
}

pub fn to_markdown(conv: &ConversationDetail) -> String {
    let mut md = String::new();
    md.push_str(&format!(
        "# {}\n\n",
        conv.title.as_deref().unwrap_or("Untitled")
    ));
    if let Some(ct) = conv.create_time {
        md.push_str(&format!("*Created: {}*\n\n", ts_fmt(ct)));
    }
    if let Some(ut) = conv.update_time {
        md.push_str(&format!("*Updated: {}*\n\n", ts_fmt(ut)));
    }

    let (Some(mapping), Some(current)) = (&conv.mapping, &conv.current_node) else {
        md.push_str("_No messages._\n");
        return md;
    };

    for node in walk_to_current(mapping, current) {
        let Some(msg) = &node.message else { continue };
        if msg.status.as_deref() == Some("finished_partially") {
            continue;
        }
        let role = match msg.author.role.as_str() {
            "user" => "You",
            "assistant" => "ChatGPT",
            "system" => "System",
            "tool" => "Tool",
            other => other,
        };
        let text = msg
            .content
            .parts
            .as_deref()
            .map(|parts| {
                parts
                    .iter()
                    .filter_map(|p| p.as_str())
                    .collect::<Vec<_>>()
                    .join("\n")
            })
            .filter(|t| !t.is_empty())
            .unwrap_or_else(|| "[non-text content]".into());

        if text.trim().is_empty() {
            continue;
        }
        if let Some(ct) = msg.create_time {
            md.push_str(&format!("<!-- {} | {} -->\n", ts_fmt(ct), msg.id));
        }
        md.push_str(&format!("**{role}:**\n\n{text}\n\n---\n\n"));
    }

    md
}
