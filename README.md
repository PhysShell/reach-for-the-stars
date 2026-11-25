Idea from @taylow's post: https://taylo.dev/posts/adding-github-stars-to-my-site.

# GitHub Stars

Lists a user's GitHub stars through the GraphQL API (no HTML scraping) and keeps an encrypted, searchable knowledge base of the starred repositories. Python standard library only.

## Star list

```bash
GITHUB_TOKEN=... python scripts/stars_kb.py list --user YOUR_GITHUB_USERNAME --out assets/data/stars.json
```

The workflow `.github/workflows/reach-for-the-stars.yml` runs this nightly with the built-in `GITHUB_TOKEN` and commits `assets/data/stars.json`. The list is paged 100 at a time, newest star first. A list shorter than 99% of GitHub's `totalCount` is not written.

Output shape:

```json
{
  "username": "example",
  "last_updated": "2026-11-24T12:00:00Z",
  "stars": [
    {
      "repo": "owner/repo",
      "id": 123456789,
      "node_id": "R_kgDOxxxxxx",
      "link": "https://github.com/owner/repo",
      "description": "Example description",
      "stars": 1234,
      "forks": 123,
      "language": "Go",
      "language_color": "#00ADD8",
      "updated": "2026-11-20T10:00:00Z",
      "starred_at": "2026-11-24T09:00:00Z"
    }
  ]
}
```

## Stars knowledge base (encrypted)

`scripts/stars_kb.py` keeps a searchable SQLite database of every starred repository: description, topics, license, stars, last push, language, and the first 8000 characters of the README, with a full-text index.

The data lives in the `stars-kb` release as files encrypted with [age](https://age-encryption.org) to a post-quantum hybrid recipient (ML-KEM-768 + X25519, `age-keygen -pq`). The public recipient is `assets/stars-kb.pub`. Only the owner of the matching private key can read the files, and the private key is never stored in GitHub.

- **Nightly** (`reach-for-the-stars.yml`): the repositories that are new since the previous `stars.json` are fetched through GraphQL, and the ones that left the list are checked and marked `unstarred`, `unavailable` or `renamed`. The batch is encrypted and uploaded as `delta-<timestamp>.jsonl.gz.age`.
- **Rebase** (`stars-kb-rebase.yml`, manual): fetches every starred repository again, and uploads `base-<timestamp>.jsonl.gz.age`. Older files stay, so repositories that are no longer starred keep their data. Use it to refresh star counts and dates, or after replacing a lost key.
- **Locally**, with the age CLI ≥ 1.3 and the Bitwarden CLI (`bw`) on `PATH`. The private key is the notes field of the Bitwarden item `stars-kb`; it is piped to age and never written to disk.

  ```bash
  export BW_SESSION=$(bw unlock --raw)
  python scripts/stars_kb.py sync                              # download, decrypt base + deltas, build .stars-kb/stars.db
  python scripts/stars_kb.py sync --key path/to/stars-kb.key   # or read the key from a file
  python scripts/stars_kb.py search "beat detection" -n 20
  python scripts/stars_kb.py search "matting" --strict --license MIT
  python scripts/stars_kb.py search "tts" --starred                # only repositories still starred
  python scripts/stars_kb.py show owner/repo
  python scripts/stars_kb.py stats
  ```

  Nothing is dropped: every repository ever stored has a status, `starred`, `unstarred`, `unavailable` (GitHub answers "not found": deleted, made private or blocked) or `renamed`, shown by `search`, `show` and `stats`. Records are keyed by GitHub's repository id: the nightly run compares star lists by id, fetches repositories by their global id, and `sync` checks that every starred id has a record, so a name that another repository takes over after a rename keeps both histories and cannot hide a new star. If a starred repository is missing from the release, `sync` stops and leaves the local database unchanged.

The nightly run pushes `stars.json` only after its delta is in the release, and both workflows share one concurrency group, so a failed run is redone by the next one. A GraphQL batch with errors or missing repositories fails the run instead of writing a partial file.

**New or lost key:**
1. Run `age-keygen -pq -o stars-kb.key`.
2. Put the `# public key:` line (without the prefix) into `assets/stars-kb.pub`.
3. Replace the notes of the Bitwarden item `stars-kb` with the contents of `stars-kb.key`, then delete the file.
4. Delete the old `base-*` and `delta-*` files from the `stars-kb` release (`sync` cannot read them any more), then run the rebase workflow. History kept under the old key is lost with it.

**Limit:** a release holds at most 1000 files, about 2.7 years of nightly deltas. Before that, deltas need compacting locally: decrypt everything, merge, check coverage, upload one encrypted base, delete the old files. The same step would carry history across a key change.
