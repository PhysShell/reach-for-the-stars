use super::{ObjectMeta, PutResult, SnapshotStore};
use anyhow::{Context, Result};
use async_trait::async_trait;
use aws_config::BehaviorVersion;
use aws_credential_types::Credentials;
use aws_sdk_s3::config::Region;
use aws_sdk_s3::primitives::ByteStream;
use aws_sdk_s3::Client;
use bytes::Bytes;
use secrecy::ExposeSecret;

use crate::secrets::StoreCredential;

pub struct S3Store {
    name: String,
    bucket: String,
    prefix: String,
    client: Client,
}

impl S3Store {
    pub async fn new(
        name: String,
        endpoint: String,
        region: String,
        bucket: String,
        prefix: String,
        creds: StoreCredential,
    ) -> Result<Self> {
        let aws_creds = Credentials::new(
            creds.access_key.expose_secret(),
            creds.secret_key.expose_secret(),
            None,
            None,
            "harness-config",
        );

        let cfg = aws_config::defaults(BehaviorVersion::latest())
            .endpoint_url(endpoint)
            .region(Region::new(region))
            .credentials_provider(aws_creds)
            .load()
            .await;

        let s3_cfg = aws_sdk_s3::config::Builder::from(&cfg)
            .force_path_style(true)
            .build();

        let client = Client::from_conf(s3_cfg);

        Ok(Self {
            name,
            bucket,
            prefix,
            client,
        })
    }

    fn full_key(&self, key: &str) -> String {
        format!("{}{}", self.prefix, key)
    }
}

#[async_trait]
impl SnapshotStore for S3Store {
    fn name(&self) -> &str {
        &self.name
    }

    async fn put_if_unmodified(
        &self,
        key: &str,
        body: Bytes,
        expected_etag: Option<&str>,
    ) -> Result<PutResult> {
        let mut req = self
            .client
            .put_object()
            .bucket(&self.bucket)
            .key(self.full_key(key))
            .body(ByteStream::from(body));

        req = match expected_etag {
            Some(etag) => req.if_match(etag),
            None => req.if_none_match("*"),
        };

        let out = req.send().await.context("s3 put_if_unmodified")?;
        Ok(PutResult {
            etag: out.e_tag().unwrap_or_default().to_string(),
        })
    }

    async fn put(&self, key: &str, body: Bytes) -> Result<PutResult> {
        let out = self
            .client
            .put_object()
            .bucket(&self.bucket)
            .key(self.full_key(key))
            .body(ByteStream::from(body))
            .send()
            .await
            .context("s3 put")?;
        Ok(PutResult {
            etag: out.e_tag().unwrap_or_default().to_string(),
        })
    }

    async fn get(&self, key: &str) -> Result<(Bytes, String)> {
        let out = self
            .client
            .get_object()
            .bucket(&self.bucket)
            .key(self.full_key(key))
            .send()
            .await
            .context("s3 get")?;
        let etag = out.e_tag().unwrap_or_default().to_string();
        let body = out.body.collect().await?.into_bytes();
        Ok((body, etag))
    }

    async fn head(&self, key: &str) -> Result<Option<String>> {
        let res = self
            .client
            .head_object()
            .bucket(&self.bucket)
            .key(self.full_key(key))
            .send()
            .await;
        match res {
            Ok(out) => Ok(Some(out.e_tag().unwrap_or_default().to_string())),
            Err(e) => match e.into_service_error() {
                aws_sdk_s3::operation::head_object::HeadObjectError::NotFound(_) => Ok(None),
                other => Err(anyhow::anyhow!("s3 head: {other:?}")),
            },
        }
    }

    async fn list(&self, prefix: &str) -> Result<Vec<ObjectMeta>> {
        let full = format!("{}{}", self.prefix, prefix);
        let out = self
            .client
            .list_objects_v2()
            .bucket(&self.bucket)
            .prefix(full)
            .send()
            .await
            .context("s3 list")?;

        Ok(out
            .contents()
            .iter()
            .map(|o| ObjectMeta {
                key: o.key().unwrap_or_default().to_string(),
                etag: o.e_tag().unwrap_or_default().to_string(),
                size: o.size().unwrap_or_default() as u64,
            })
            .collect())
    }

    async fn delete_if_match(&self, key: &str, etag: &str) -> Result<()> {
        // R2 and recent MinIO honour If-Match on DeleteObject. Older S3-compat
        // backends may ignore the precondition silently — that's a backend
        // bug, not ours; the lock TTL still provides a safety net.
        self.client
            .delete_object()
            .bucket(&self.bucket)
            .key(self.full_key(key))
            .if_match(etag)
            .send()
            .await
            .context("s3 delete")?;
        Ok(())
    }
}
