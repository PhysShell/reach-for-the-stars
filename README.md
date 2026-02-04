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
