import argparse
import json
import os
import random
import re
import sys
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

REPO_PATH_RE = re.compile(r"^/[^/]+/[^/]+$")
LIST_PATH_TEMPLATE = r"^/stars/{username}/lists/[^/]+$"
COLOR_RE = re.compile(r"#(?:[0-9a-fA-F]{3}){1,2}")

class Logger:
    def __init__(self, verbose: bool) -> None:
        self.verbose = verbose
        self._last_progress = None

    def info(self, message: str) -> None:
        if self.verbose:
            print(message, file=sys.stderr)

    def warn(self, message: str) -> None:
        if self.verbose:
            print(f"warning: {message}", file=sys.stderr)

    def progress(self, current: int, total: int | None) -> None:
        if not total:
            return
        percent = int(min(100, (current / total) * 100))
        if percent == self._last_progress:
            return
        self._last_progress = percent
        message = f"Progress: {percent}% ({current}/{total})"
        if self.verbose:
            print(message, file=sys.stderr)
            return
        end = "\r" if percent < 100 else "\n"
        print(message, file=sys.stderr, end=end, flush=True)


class RequestManager:
    def __init__(
        self,
        session: requests.Session,
        timeout: int,
        logger: Logger,
        max_retries: int,
        backoff_base: float,
        backoff_max: float,
        min_interval: float,
        cb_threshold: int,
        cb_cooldown: float,
    ) -> None:
        self.session = session
        self.timeout = timeout
        self.logger = logger
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.backoff_max = backoff_max
        self.min_interval = min_interval
        self.cb_threshold = cb_threshold
        self.cb_cooldown = cb_cooldown
        self._last_request_time = 0.0
        self._consecutive_failures = 0
        self._circuit_open_until = 0.0

    def _sleep(self, seconds: float) -> None:
        if seconds > 0:
            time.sleep(seconds)

    def _enforce_min_interval(self) -> None:
        if self.min_interval <= 0:
            return
        now = time.monotonic()
        elapsed = now - self._last_request_time
        if elapsed < self.min_interval:
            self._sleep(self.min_interval - elapsed)

    def _maybe_wait_for_circuit(self) -> None:
        now = time.monotonic()
        if now < self._circuit_open_until:
            wait = self._circuit_open_until - now
            self.logger.warn(f"circuit breaker open, waiting {wait:.1f}s")
            self._sleep(wait)

    def _record_failure(self) -> None:
        self._consecutive_failures += 1
        if self.cb_threshold > 0 and self._consecutive_failures >= self.cb_threshold:
            self._circuit_open_until = time.monotonic() + self.cb_cooldown
            self._consecutive_failures = 0

    def _record_success(self) -> None:
        self._consecutive_failures = 0

    def _parse_retry_after(self, response: requests.Response | None) -> float | None:
        if response is None:
            return None
        retry_after = response.headers.get("Retry-After")
        if retry_after:
            try:
                return max(0.0, float(retry_after))
            except ValueError:
                try:
                    parsed = parsedate_to_datetime(retry_after)
                    return max(0.0, parsed.timestamp() - time.time())
                except (TypeError, ValueError):
                    return None
        reset = response.headers.get("X-RateLimit-Reset")
        if reset:
            try:
                return max(0.0, float(reset) - time.time())
            except ValueError:
                return None
        return None

    def _compute_backoff(self, attempt: int) -> float:
        base = min(self.backoff_base * (2 ** attempt), self.backoff_max)
        jitter = base * (0.5 + random.random() / 2)
        return min(jitter, self.backoff_max)

    def _is_rate_limited(self, response: requests.Response) -> bool:
        if response.status_code == 429:
            return True
        if response.status_code == 403 and response.headers.get("X-RateLimit-Remaining") == "0":
            return True
        return False

    def _should_retry(self, response: requests.Response | None, exc: Exception | None) -> bool:
        if exc is not None:
            return True
        if response is None:
            return False
        if self._is_rate_limited(response):
            return True
        if response.status_code in (408, 429):
            return True
        if 500 <= response.status_code < 600:
            return True
        return False

    def get(self, url: str) -> requests.Response:
        last_exc: Exception | None = None
        last_response: requests.Response | None = None

        for attempt in range(self.max_retries + 1):
            self._maybe_wait_for_circuit()
            self._enforce_min_interval()
            try:
                response = self.session.get(url, timeout=self.timeout)
                self._last_request_time = time.monotonic()
            except requests.RequestException as exc:
                last_exc = exc
                if attempt >= self.max_retries:
                    raise
                self._record_failure()
                delay = self._compute_backoff(attempt)
                self.logger.warn(f"request failed ({exc}); retrying in {delay:.1f}s")
                self._sleep(delay)
                continue

            last_response = response
            if not self._should_retry(response, None):
                response.raise_for_status()
                self._record_success()
                return response

            if attempt >= self.max_retries:
                response.raise_for_status()
                return response

            self._record_failure()
            retry_after = self._parse_retry_after(response)
            backoff = self._compute_backoff(attempt)
            delay_candidates = [backoff]
            if retry_after is not None:
                delay_candidates.append(retry_after)
            delay = max(delay_candidates)
            self.logger.warn(
                f"request returned {response.status_code}; retrying in {delay:.1f}s"
            )
            self._sleep(delay)

        if last_exc:
            raise last_exc
        if last_response is not None:
            last_response.raise_for_status()
        raise RuntimeError("request failed without response")


def build_session(user_agent: str) -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": user_agent})
    return session


def fetch_soup(requests_manager: RequestManager, url: str) -> BeautifulSoup:
    response = requests_manager.get(url)
    return BeautifulSoup(response.text, "html.parser")


def base_url_from(url: str) -> str:
    parsed = urlparse(url)
    return f"{parsed.scheme}://{parsed.netloc}"


def extract_color(style: str) -> str:
    match = COLOR_RE.search(style or "")
    return match.group(0) if match else ""


def parse_count(value: str) -> int | None:
    if not value:
        return None
    raw = value.strip().lower().replace(",", "")
    match = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*([km]?)", raw)
    if not match:
        return None
    number = float(match.group(1))
    suffix = match.group(2)
    if suffix == "k":
        number *= 1_000
    elif suffix == "m":
        number *= 1_000_000
    return int(number)


def extract_total_stars(soup: BeautifulSoup) -> int | None:
    tab = soup.find("a", attrs={"data-tab-item": "stars"})
    if not tab:
        tab = soup.find("a", href=lambda href: href and "tab=stars" in href)
    if tab:
        counter = tab.find("span", class_="Counter")
        if counter:
            return parse_count(counter.get_text(strip=True))

    for counter in soup.find_all("span", class_="Counter"):
        parent = counter.find_parent("a")
        if not parent:
            continue
        text = parent.get_text(" ", strip=True).lower()
        if "star" in text:
            return parse_count(counter.get_text(strip=True))
    return None


def parse_repo_entry(entry: BeautifulSoup, base_url: str) -> dict | None:
    repo_link_tag = entry.find("a", href=REPO_PATH_RE)
    if not repo_link_tag:
        return None

    repo_href = repo_link_tag.get("href", "").strip()
    if not repo_href:
        return None

    repo_link = urljoin(base_url, repo_href)
    repo_name = repo_link_tag.get_text(strip=True)

    description_tag = entry.find("p", itemprop="description")
    description = description_tag.get_text(strip=True) if description_tag else ""

    star_tag = entry.find("a", href=lambda href: href and "stargazers" in href)
    stars = star_tag.get_text(strip=True) if star_tag else ""

    fork_tag = entry.find("a", href=lambda href: href and ("forks" in href or "network/members" in href))
    forks = fork_tag.get_text(strip=True) if fork_tag else ""

    language_tag = entry.find("span", itemprop="programmingLanguage")
    language = language_tag.get_text(strip=True) if language_tag else ""

    color_tag = entry.find("span", class_="repo-language-color")
    language_color = extract_color(color_tag.get("style", "")) if color_tag else ""

    updated_tag = entry.find("relative-time")
    updated = updated_tag.get("datetime", "") if updated_tag else ""

    return {
        "link": repo_link,
        "name": repo_name,
        "description": description,
        "stars": stars,
        "language": language,
        "language_color": language_color,
        "forks": forks,
        "updated": updated,
    }


def extract_star_entries(soup: BeautifulSoup, base_url: str) -> list[dict]:
    frame = soup.find(id="user-starred-repos") or soup
    entries = frame.find_all("div", class_="d-block")
    stars: list[dict] = []
    for entry in entries:
        repo = parse_repo_entry(entry, base_url)
        if repo:
            stars.append(repo)
    return stars


def find_next_page_url(soup: BeautifulSoup, current_url: str) -> str | None:
    next_link = soup.find(
        "a",
        rel=lambda rel: rel and ("next" in rel if isinstance(rel, list) else "next" in rel),
    )
    if not next_link:
        next_link = soup.find("a", attrs={"aria-label": "Next"})
    if not next_link:
        next_link = soup.find("a", href=lambda href: href and "after=" in href)
    if not next_link:
        return None

    href = next_link.get("href")
    if not href:
        return None

    if href.startswith("?"):
        parsed = urlparse(current_url)
        query = href[1:]
        if "tab=" not in query and parsed.query:
            query = f"{parsed.query}&{query}"
        return f"{parsed.scheme}://{parsed.netloc}{parsed.path}?{query}"

    return urljoin(current_url, href)


def fetch_star_page(
    requests_manager: RequestManager,
    url: str,
) -> tuple[list[dict], str | None]:
    soup = fetch_soup(requests_manager, url)
    base_url = base_url_from(url)
    stars = extract_star_entries(soup, base_url)
    next_url = find_next_page_url(soup, url)
    return stars, next_url


def fetch_all_stars(
    requests_manager: RequestManager,
    start_url: str,
    logger: Logger,
    show_progress: bool = True,
) -> list[dict]:
    stars: list[dict] = []
    seen_urls: set[str] = set()
    next_url = start_url
    total_stars: int | None = None
    page_index = 0
    while next_url and next_url not in seen_urls:
        page_index += 1
        seen_urls.add(next_url)
        logger.info(f"fetching page {page_index}: {next_url}")
        soup = fetch_soup(requests_manager, next_url)
        if total_stars is None:
            total_stars = extract_total_stars(soup)
            if total_stars:
                logger.info(f"detected {total_stars} total stars")
        base_url = base_url_from(next_url)
        page_stars = extract_star_entries(soup, base_url)
        next_url = find_next_page_url(soup, next_url)
        stars.extend(page_stars)
        if show_progress:
            logger.progress(len(stars), total_stars)
    return stars


def extract_list_links(soup: BeautifulSoup, username: str, base_url: str) -> list[dict]:
    pattern = re.compile(LIST_PATH_TEMPLATE.format(username=re.escape(username)))
    links: dict[str, str] = {}
    for anchor in soup.find_all("a", href=pattern):
        href = anchor.get("href", "")
        if not href:
            continue
        name = anchor.get_text(strip=True) or href.rsplit("/", 1)[-1]
        full_url = urljoin(base_url, href)
        if full_url not in links or len(name) > len(links[full_url]):
            links[full_url] = name
    return [{"name": name, "url": url} for url, name in sorted(links.items())]


def fetch_star_lists(
    requests_manager: RequestManager,
    username: str,
    base_url: str,
    logger: Logger,
) -> list[dict]:
    lists_url = f"{base_url}/stars/{username}/lists"
    try:
        soup = fetch_soup(requests_manager, lists_url)
    except requests.RequestException as exc:
        logger.warn(f"unable to fetch star lists ({lists_url}): {exc}")
        return []

    list_links = extract_list_links(soup, username, base_url)
    lists: list[dict] = []
    for item in list_links:
        logger.info(f"fetching list {item['name']} ({item['url']})")
        try:
            stars = fetch_all_stars(requests_manager, item["url"], logger, show_progress=False)
        except requests.RequestException as exc:
            logger.warn(f"unable to fetch list {item['url']}: {exc}")
            stars = []
        lists.append(
            {
                "name": item["name"],
                "url": item["url"],
                "stars": stars,
            }
        )
    return lists


def resolve_username(explicit: str | None) -> str | None:
    if explicit:
        return explicit
    env_username = os.environ.get("GITHUB_USERNAME")
    if env_username:
        return env_username
    repo = os.environ.get("GITHUB_REPOSITORY")
    if repo and "/" in repo:
        return repo.split("/", 1)[0]
    actor = os.environ.get("GITHUB_ACTOR")
    if actor:
        return actor
    return None


def write_json(data: dict, output_path: str) -> None:
    if output_path == "-":
        json.dump(data, sys.stdout, indent=2)
        sys.stdout.write("\n")
        return

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)
        handle.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch GitHub stars for a user.")
    parser.add_argument("-u", "--username", help="GitHub username.")
    parser.add_argument(
        "-o",
        "--output",
        default="assets/data/stars.json",
        help="Output path for JSON. Use '-' for stdout.",
    )
    parser.add_argument(
        "--include-lists",
        action="store_true",
        help="Attempt to fetch star lists (best effort).",
    )
    parser.add_argument(
        "--base-url",
        default="https://github.com",
        help="GitHub base URL for enterprise instances.",
    )
    parser.add_argument("--timeout", type=int, default=20, help="HTTP timeout in seconds.")
    parser.add_argument(
        "--min-interval",
        type=float,
        default=1.0,
        help="Minimum seconds between requests.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=5,
        help="Maximum retry attempts for failed requests.",
    )
    parser.add_argument(
        "--backoff-base",
        type=float,
        default=1.0,
        help="Base seconds for exponential backoff.",
    )
    parser.add_argument(
        "--backoff-max",
        type=float,
        default=60.0,
        help="Maximum seconds for backoff delay.",
    )
    parser.add_argument(
        "--circuit-breaker-threshold",
        type=int,
        default=5,
        help="Consecutive failures before opening the circuit breaker.",
    )
    parser.add_argument(
        "--circuit-breaker-cooldown",
        type=float,
        default=60.0,
        help="Seconds to wait after circuit breaker opens.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging (default shows progress only).",
    )
    parser.add_argument(
        "--user-agent",
        default="github-stars-crawler/1.0",
        help="User-Agent header value.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    username = resolve_username(args.username)
    if not username:
        raise SystemExit("error: username is required (use --username or GITHUB_USERNAME).")

    logger = Logger(args.verbose)
    base_url = args.base_url.rstrip("/")
    start_url = f"{base_url}/{username}?tab=stars"

    session = build_session(args.user_agent)
    requests_manager = RequestManager(
        session=session,
        timeout=args.timeout,
        logger=logger,
        max_retries=args.max_retries,
        backoff_base=args.backoff_base,
        backoff_max=args.backoff_max,
        min_interval=args.min_interval,
        cb_threshold=args.circuit_breaker_threshold,
        cb_cooldown=args.circuit_breaker_cooldown,
    )

    logger.info(f"fetching stars from {start_url}")
    stars = fetch_all_stars(requests_manager, start_url, logger)
    logger.info(f"fetched {len(stars)} starred repositories")

    data: dict = {
        "username": username,
        "last_updated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "stars": stars,
    }

    if args.include_lists:
        lists = fetch_star_lists(requests_manager, username, base_url, logger)
        data["lists"] = lists
        logger.info(f"fetched {len(lists)} star lists")

    write_json(data, args.output)


if __name__ == "__main__":
    main()
