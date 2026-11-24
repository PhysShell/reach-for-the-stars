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

- **Nightly** (`reach-for-the-stars.yml`): after `stars.json` is updated, the repositories that are new since the previous `stars.json` are fetched through GraphQL. The batch is encrypted and uploaded as `delta-<timestamp>.jsonl.gz.age`.
- **Rebase** (`stars-kb-rebase.yml`, manual): fetches every starred repository again, uploads `base-<timestamp>.jsonl.gz.age`, and deletes older files. Use it to refresh star counts and dates, or after replacing a lost key.
- **Locally**, with the age CLI ≥ 1.3 on `PATH`:

  ```bash
  python scripts/stars_kb.py sync --key path/to/stars-kb.key   # download, decrypt base + deltas, build .stars-kb/stars.db
  python scripts/stars_kb.py search "beat detection" -n 20
  python scripts/stars_kb.py search "matting" --strict --license MIT
  python scripts/stars_kb.py show owner/repo
  python scripts/stars_kb.py stats
  ```

  Repositories that are no longer in `stars.json` are dropped during `sync`.

**New or lost key:**
1. Run `age-keygen -pq -o stars-kb.key`.
2. Put the `# public key:` line (without the prefix) into `assets/stars-kb.pub`.
3. Run the rebase workflow.

Files encrypted to the old key become unreadable and are deleted by the rebase.
