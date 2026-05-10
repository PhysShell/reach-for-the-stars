use anyhow::Result;
use std::path::Path;

/// Owns a Chromium child + a CDP connection, plus the on-disk user_data_dir
/// it was launched against. Caller is responsible for archiving user_data_dir
/// after `close()`.
pub struct BrowserSession {
    // chromiumoxide::Browser + Handler join handle would live here
}

impl BrowserSession {
    pub async fn launch(
        _chromium_bin: &Path,
        _user_data_dir: &Path,
        _headless: bool,
    ) -> Result<Self> {
        // chromiumoxide::BrowserConfig::builder()
        //     .chrome_executable(_chromium_bin)
        //     .user_data_dir(_user_data_dir)
        //     .with_head() // if !headless
        //     .arg("--no-default-browser-check")
        //     .arg("--no-first-run")
        //     .build()?
        // then chromiumoxide::Browser::launch(cfg).await?
        anyhow::bail!("BrowserSession::launch not yet implemented");
    }

    /// Snapshot cookies + localStorage via CDP into a Playwright-compatible
    /// storage_state.json shape (cookies[], origins[]).
    pub async fn export_storage_state(&self) -> Result<serde_json::Value> {
        anyhow::bail!("export_storage_state not yet implemented");
    }

    /// Inject cookies + localStorage from a previously-exported storage state
    /// before the first navigation.
    pub async fn import_storage_state(&self, _state: &serde_json::Value) -> Result<()> {
        anyhow::bail!("import_storage_state not yet implemented");
    }

    pub async fn close(self) -> Result<()> {
        Ok(())
    }
}
