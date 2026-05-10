//! End-to-end smoke tests.
//!
//! Three independent checks:
//!   1. `crypto_archive_roundtrip` — pure data pipeline, no external deps.
//!   2. `s3_cas_semantics` — needs MinIO (or any S3-compat backend that
//!      honours If-Match / If-None-Match).
//!   3. `full_cycle_with_chromium` — needs MinIO + a Chromium binary.
//!
//! Tests that need external services skip cleanly when the relevant env vars
//! are absent. Run via `scripts/smoke.sh` from inside `nix develop` to get
//! all three exercised.

use bytes::Bytes;
use rand::rngs::OsRng;
use secrecy::SecretString;

use harness::archive;
use harness::browser::canary;
use harness::browser::cdp::{cookies_to_reqwest_jar, BrowserSession, StorageState, StoredCookie};
use harness::crypto::{encrypt, sign};
use harness::secrets::StoreCredential;
use harness::stores::s3::S3Store;
use harness::stores::SnapshotStore;

fn fixture_state() -> StorageState {
    StorageState {
        cookies: vec![StoredCookie {
            name: "session".into(),
            value: "abc123-roundtrip".into(),
            domain: "example.com".into(),
            path: "/".into(),
            expires: None,
            http_only: true,
            secure: true,
            same_site: Some("Lax".into()),
        }],
        origins: vec![],
    }
}

#[tokio::test]
async fn crypto_archive_roundtrip() {
    let identity = age::x25519::Identity::generate();
    let recipient = identity.to_public();
    let signing_key = ed25519_dalek::SigningKey::generate(&mut OsRng);
    let pubkey = signing_key.verifying_key();

    let state = fixture_state();
    let plaintext = archive::compress(&state).unwrap();
    let ciphertext = encrypt::encrypt(&plaintext, &recipient).unwrap();
    let sig = sign::sign(&signing_key, &ciphertext);

    sign::verify(&pubkey, &ciphertext, &sig).expect("sig should verify");

    let decrypted = encrypt::decrypt(&ciphertext, &identity).unwrap();
    let restored = archive::decompress(&decrypted).unwrap();

    assert_eq!(restored.cookies.len(), 1);
    assert_eq!(restored.cookies[0].name, "session");
    assert_eq!(restored.cookies[0].value, "abc123-roundtrip");
    assert!(restored.cookies[0].http_only);
    assert!(restored.cookies[0].secure);

    // Tamper detection: flipping a byte must invalidate the signature.
    let mut tampered = ciphertext.to_vec();
    tampered[10] ^= 0x01;
    assert!(
        sign::verify(&pubkey, &tampered, &sig).is_err(),
        "tampered ciphertext should not verify"
    );
}

#[tokio::test]
async fn s3_cas_semantics() {
    let Some((endpoint, access_key, secret_key)) = s3_env() else {
        return;
    };

    let bucket = format!("smoke-cas-{}", chrono::Utc::now().timestamp_millis());
    create_bucket(&endpoint, &access_key, &secret_key, &bucket).await;

    let creds = StoreCredential {
        access_key: SecretString::from(access_key),
        secret_key: SecretString::from(secret_key),
    };
    let store = S3Store::new(
        "smoke".into(),
        endpoint,
        "us-east-1".into(),
        bucket,
        "state/".into(),
        creds,
    )
    .await
    .unwrap();

    let v1 = Bytes::from_static(b"hello v1");
    let v2 = Bytes::from_static(b"hello v2");
    let v3 = Bytes::from_static(b"hello v3");

    // create-only succeeds when object is absent.
    let r1 = store
        .put_if_unmodified("test.txt", v1.clone(), None)
        .await
        .expect("create-only on absent should succeed");

    // create-only must fail when object already exists.
    let dup = store.put_if_unmodified("test.txt", v1.clone(), None).await;
    assert!(
        dup.is_err(),
        "create-only on existing object should fail (got {dup:?})"
    );

    // update-only with correct etag succeeds.
    let r2 = store
        .put_if_unmodified("test.txt", v2.clone(), Some(&r1.etag))
        .await
        .expect("update-only with matching etag should succeed");
    assert_ne!(r1.etag, r2.etag, "etag should change on update");

    // update-only with stale etag must fail.
    let stale = store
        .put_if_unmodified("test.txt", v3, Some(&r1.etag))
        .await;
    assert!(
        stale.is_err(),
        "update-only with stale etag should fail (got {stale:?})"
    );

    let (got, _) = store.get("test.txt").await.unwrap();
    assert_eq!(&got[..], &b"hello v2"[..], "store should hold v2");
}

#[tokio::test]
async fn full_cycle_with_chromium() {
    let Some(chromium_bin) = std::env::var("CHROMIUM_BIN").ok() else {
        eprintln!("[skip] CHROMIUM_BIN not set");
        return;
    };

    let mock = wiremock::MockServer::start().await;

    use wiremock::matchers::{method, path};
    use wiremock::{Mock, ResponseTemplate};

    Mock::given(method("GET"))
        .and(path("/api/me"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(serde_json::json!({"user": {"id": "smoke-user-id"}})),
        )
        .mount(&mock)
        .await;

    let canary_url = format!("{}/api/me", mock.uri());
    let host = "127.0.0.1";

    let state = StorageState {
        cookies: vec![StoredCookie {
            name: "smoke_session".into(),
            value: "valid-token".into(),
            domain: host.into(),
            path: "/".into(),
            expires: None,
            http_only: false,
            secure: false,
            same_site: None,
        }],
        origins: vec![],
    };

    let temp = tempfile::tempdir().unwrap();
    let session = BrowserSession::launch(
        std::path::Path::new(&chromium_bin),
        temp.path(),
        true,
    )
    .await
    .expect("chromium launch");

    session
        .import_storage_state(&state)
        .await
        .expect("import cookies");

    let exported = session
        .export_storage_state()
        .await
        .expect("export cookies");
    session.close().await.ok();

    assert!(
        exported
            .cookies
            .iter()
            .any(|c| c.name == "smoke_session" && c.value == "valid-token"),
        "imported cookie missing in re-export: {:?}",
        exported.cookies
    );

    let jar = cookies_to_reqwest_jar(&state);
    let client = reqwest::Client::builder()
        .cookie_provider(jar)
        .build()
        .unwrap();
    let result = canary::check(&client, &canary_url, 200, "/user/id", "smoke-user-id")
        .await
        .expect("canary call");

    assert!(
        result.ok,
        "canary should pass with imported cookie: {result:?}"
    );
}

fn s3_env() -> Option<(String, String, String)> {
    let endpoint = match std::env::var("SMOKE_S3_ENDPOINT") {
        Ok(v) => v,
        Err(_) => {
            eprintln!("[skip] SMOKE_S3_ENDPOINT not set");
            return None;
        }
    };
    let ak = std::env::var("SMOKE_S3_ACCESS_KEY").unwrap_or_else(|_| "minioadmin".into());
    let sk = std::env::var("SMOKE_S3_SECRET_KEY").unwrap_or_else(|_| "minioadmin".into());
    Some((endpoint, ak, sk))
}

async fn create_bucket(endpoint: &str, ak: &str, sk: &str, bucket: &str) {
    use aws_config::BehaviorVersion;
    use aws_credential_types::Credentials;
    use aws_sdk_s3::config::Region;
    use aws_sdk_s3::Client;

    let creds = Credentials::new(ak, sk, None, None, "smoke");
    let cfg = aws_config::defaults(BehaviorVersion::latest())
        .endpoint_url(endpoint)
        .region(Region::new("us-east-1"))
        .credentials_provider(creds)
        .load()
        .await;
    let s3_cfg = aws_sdk_s3::config::Builder::from(&cfg)
        .force_path_style(true)
        .build();
    let s3 = Client::from_conf(s3_cfg);
    let _ = s3.create_bucket().bucket(bucket).send().await;
}
