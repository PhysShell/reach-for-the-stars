# Prologue

Based on great simple project by @taylow: https://taylo.dev/posts/adding-github-stars-to-my-site.

If it's not appropriate for me to have this repo (license issues, etc.) then contact me here in and I'd gladly remove it.

For now I used MIT license as it's used the most and hope that I won't get sued

# GitHub Stars Crawler

Fetch a user's GitHub stars and save them as JSON.

## Setup (pipenv)

```bash
pipenv install
```

## Run

```bash
pipenv run python scripts/star_scraper.py -u YOUR_GITHUB_USERNAME --output assets/data/stars.json
```

Optional flags:

- Use `--verbose` for stage-by-stage logging. Default output is progress-only.
- Rate-limit controls: `--min-interval` (default `1.0` seconds, use `0` to disable delays),
  `--max-retries`, `--backoff-base`, `--backoff-max`, `--circuit-breaker-threshold`,
  `--circuit-breaker-cooldown`.

```
When I tested it on profile with 3.4k stars in started giving me `429 Client Error: Too Many Requests` halfway through
and that's the reason --min-interval exists. With its default value it's able to complete crawling.
```

## Automation (GitHub Actions)

This repo includes a scheduled workflow that runs nightly and updates `assets/data/stars.json`:
`.github/workflows/reach-for-the-stars.yml`. It uses `GITHUB_USERNAME` (defaults to the repo owner).

## Tests

Install dev dependencies and run pytest:

```bash
pipenv install --dev
pipenv run pytest
```

## Output shape

```json
{
  "username": "example",
  "last_updated": "2026-01-29T12:00:00Z",
  "stars": [
    {
      "link": "https://github.com/owner/repo",
      "name": "owner / repo",
      "description": "Example description",
      "stars": "1,234",
      "language": "Go",
      "language_color": "#00ADD8",
      "forks": "123",
      "updated": "2026-01-28T10:00:00Z"
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
