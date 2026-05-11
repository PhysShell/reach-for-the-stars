use anyhow::{Context, Result};
use chromiumoxide::cdp::browser_protocol::network::{CookieParam, CookieSameSite, TimeSinceEpoch};
use chromiumoxide::cdp::browser_protocol::storage::{
    ClearCookiesParams, GetCookiesParams, SetCookiesParams,
};
use chromiumoxide::{Browser, BrowserConfig, Page};
use futures::StreamExt;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use tokio::task::JoinHandle;

#[derive(Debug, Serialize, Deserialize, Default, Clone)]
pub struct StorageState {
    pub cookies: Vec<StoredCookie>,
    #[serde(default)]
    pub origins: Vec<Origin>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct StoredCookie {
    pub name: String,
    pub value: String,
    pub domain: String,
    pub path: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expires: Option<f64>,
    #[serde(default, rename = "httpOnly")]
    pub http_only: bool,
    #[serde(default)]
    pub secure: bool,
    #[serde(default, rename = "sameSite", skip_serializing_if = "Option::is_none")]
    pub same_site: Option<String>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Origin {
    pub origin: String,
    #[serde(rename = "localStorage")]
    pub local_storage: Vec<KV>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct KV {
    pub name: String,
    pub value: String,
}

pub struct BrowserSession {
    browser: Browser,
    pump: JoinHandle<()>,
    user_data_dir: PathBuf,
}

impl BrowserSession {
    pub async fn launch(chromium_bin: &Path, user_data_dir: &Path, headless: bool) -> Result<Self> {
        std::fs::create_dir_all(user_data_dir).context("create user_data_dir")?;

        let mut builder = BrowserConfig::builder()
            .chrome_executable(chromium_bin)
            .user_data_dir(user_data_dir)
            .arg("--no-default-browser-check")
            .arg("--no-first-run")
            .arg("--disable-dev-shm-usage")
            .arg("--disable-background-networking");

        if !headless {
            builder = builder.with_head();
        }

        let cfg = builder
            .build()
            .map_err(|e| anyhow::anyhow!("browser config: {e}"))?;
        let (browser, mut handler) = Browser::launch(cfg).await?;

        // The handler stream must be drained continuously, otherwise CDP commands stall.
        let pump = tokio::spawn(async move {
            while let Some(res) = handler.next().await {
                if let Err(e) = res {
                    tracing::debug!(error = %e, "cdp event");
                }
            }
        });

        Ok(Self {
            browser,
            pump,
            user_data_dir: user_data_dir.to_path_buf(),
        })
    }

    pub fn user_data_dir(&self) -> &Path {
        &self.user_data_dir
    }

    pub async fn open(&self, url: &str) -> Result<Page> {
        let page = self.browser.new_page(url).await?;
        Ok(page)
    }

    pub async fn export_storage_state(&self) -> Result<StorageState> {
        let cookies = self.export_cookies().await?;
        // localStorage export requires navigating to each origin (DOMStorage CDP needs
        // a security-origin-bound execution context). Skipped in v1; cookies cover the
        // 90% case for cookie-based session auth. Fill in when a target needs it.
        Ok(StorageState {
            cookies,
            origins: vec![],
        })
    }

    async fn export_cookies(&self) -> Result<Vec<StoredCookie>> {
        let resp = self
            .browser
            .execute(GetCookiesParams::default())
            .await
            .context("Storage.getCookies")?;

        let cookies = resp
            .result
            .cookies
            .iter()
            .map(|c| StoredCookie {
                name: c.name.clone(),
                value: c.value.clone(),
                domain: c.domain.clone(),
                path: c.path.clone(),
                expires: if c.expires < 0.0 {
                    None
                } else {
                    Some(c.expires)
                },
                http_only: c.http_only,
                secure: c.secure,
                same_site: c.same_site.as_ref().map(same_site_to_str),
            })
            .collect();
        Ok(cookies)
    }

    pub async fn import_storage_state(&self, state: &StorageState) -> Result<()> {
        if state.cookies.is_empty() {
            return Ok(());
        }

        // Clear browser-context cookies first so an old expired snapshot can't
        // shadow a newer one we're about to inject.
        self.browser
            .execute(ClearCookiesParams::default())
            .await
            .context("Storage.clearCookies")?;

        let cookies: Vec<CookieParam> = state
            .cookies
            .iter()
            .map(|c| {
                let mut p = CookieParam::new(c.name.clone(), c.value.clone());
                p.domain = Some(c.domain.clone());
                p.path = Some(c.path.clone());
                p.http_only = Some(c.http_only);
                p.secure = Some(c.secure);
                if let Some(exp) = c.expires {
                    p.expires = Some(TimeSinceEpoch::new(exp));
                }
                if let Some(s) = &c.same_site {
                    p.same_site = same_site_from_str(s);
                }
                p
            })
            .collect();

        self.browser
            .execute(SetCookiesParams::new(cookies))
            .await
            .context("Network.setCookies")?;

        // origins / localStorage import: see export_storage_state comment.
        Ok(())
    }

    pub async fn close(mut self) -> Result<()> {
        let _ = self.browser.close().await;
        let _ = self.browser.wait().await;
        self.pump.abort();
        let _ = self.pump.await;
        Ok(())
    }
}

fn same_site_to_str(s: &CookieSameSite) -> String {
    match s {
        CookieSameSite::Strict => "Strict".into(),
        CookieSameSite::Lax => "Lax".into(),
        CookieSameSite::None => "None".into(),
    }
}

fn same_site_from_str(s: &str) -> Option<CookieSameSite> {
    match s {
        "Strict" => Some(CookieSameSite::Strict),
        "Lax" => Some(CookieSameSite::Lax),
        "None" => Some(CookieSameSite::None),
        _ => None,
    }
}

/// Discover an origin's cookies by visiting an authenticated endpoint and copying
/// the live cookie jar into a `reqwest::cookie::Jar`. Used by the canary check.
pub fn cookies_to_reqwest_jar(state: &StorageState) -> std::sync::Arc<reqwest::cookie::Jar> {
    use reqwest::cookie::Jar;
    let jar = Jar::default();
    for c in &state.cookies {
        let mut header = format!(
            "{}={}; Path={}; Domain={}",
            c.name, c.value, c.path, c.domain
        );
        if c.secure {
            header.push_str("; Secure");
        }
        if c.http_only {
            header.push_str("; HttpOnly");
        }
        if let Some(s) = &c.same_site {
            header.push_str(&format!("; SameSite={s}"));
        }
        let scheme = if c.secure { "https" } else { "http" };
        let url_str = format!("{scheme}://{}", c.domain.trim_start_matches('.'));
        if let Ok(url) = url::Url::parse(&url_str) {
            jar.add_cookie_str(&header, &url);
        }
    }
    std::sync::Arc::new(jar)
}
