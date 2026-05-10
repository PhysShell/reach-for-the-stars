//! Read/write the envelope from disk with atomic replace semantics.

use std::fs::{self, File};
use std::io::{BufReader, BufWriter, Write};
use std::path::Path;

use anyhow::{Context, Result};
use tracing::warn;

use crate::model::StarsEnvelope;

/// Try to load an existing envelope. Returns `None` when the file does not
/// exist OR when the file exists but cannot be parsed as a v2 envelope
/// (e.g. it is the old Python-script schema). The caller should treat
/// either case as "no prior state" and force a full reconcile.
pub fn read_envelope(path: &Path) -> Result<Option<StarsEnvelope>> {
    match File::open(path) {
        Ok(file) => {
            let reader = BufReader::new(file);
            match serde_json::from_reader::<_, StarsEnvelope>(reader) {
                Ok(env) => Ok(Some(env)),
                Err(err) => {
                    warn!(
                        path = %path.display(),
                        error = %err,
                        "existing file is not a v2 envelope; starting fresh"
                    );
                    Ok(None)
                }
            }
        }
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(err) => {
            Err(err).with_context(|| format!("failed to open envelope at {}", path.display()))
        }
    }
}

/// Write the envelope as pretty-printed JSON via a tempfile + rename so the
/// destination either holds the previous version or the new one — never a
/// half-written file.
pub fn write_envelope(path: &Path, envelope: &StarsEnvelope) -> Result<()> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent).with_context(|| {
                format!("failed to create parent directory {}", parent.display())
            })?;
        }
    }

    let parent = path.parent().filter(|p| !p.as_os_str().is_empty());
    let mut tmp = match parent {
        Some(dir) => tempfile::NamedTempFile::new_in(dir),
        None => tempfile::NamedTempFile::new(),
    }
    .with_context(|| format!("failed to create tempfile next to {}", path.display()))?;

    {
        let mut writer = BufWriter::new(tmp.as_file_mut());
        serde_json::to_writer_pretty(&mut writer, envelope)
            .context("failed to serialize envelope as JSON")?;
        writer
            .write_all(b"\n")
            .context("failed to write trailing newline")?;
        writer.flush().context("failed to flush envelope writer")?;
    }

    tmp.persist(path)
        .with_context(|| format!("failed to atomically replace {}", path.display()))?;
    Ok(())
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing
)]
mod tests {
    use super::*;
    use crate::model::StarsEnvelope;
    use time::macros::datetime;

    #[test]
    fn read_returns_none_for_missing_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("missing.json");
        assert!(read_envelope(&path).unwrap().is_none());
    }

    #[test]
    fn read_returns_none_for_unparseable_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("legacy.json");
        std::fs::write(&path, b"{\"username\": \"old\", \"stars\": []}").unwrap();
        assert!(read_envelope(&path).unwrap().is_none());
    }

    #[test]
    fn round_trip_via_disk_preserves_data() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("nested").join("stars.json");
        let env = StarsEnvelope::empty("octocat", datetime!(2026-05-09 12:00:00 UTC));
        write_envelope(&path, &env).unwrap();
        let loaded = read_envelope(&path).unwrap().unwrap();
        assert_eq!(loaded.subject.username, "octocat");
    }
}
