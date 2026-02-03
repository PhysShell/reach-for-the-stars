import sys
from pathlib import Path

import pytest
import requests
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import star_scraper  # noqa: E402


def make_response(status_code: int, text: str = "", headers: dict | None = None) -> requests.Response:
    response = requests.Response()
    response.status_code = status_code
    response._content = text.encode("utf-8")
    response.headers = headers or {}
    response.url = "https://example.com"
    return response


class DummySession:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def get(self, url, timeout=None):
        self.calls.append((url, timeout))
        if not self._responses:
            raise RuntimeError("no more responses")
        item = self._responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class CaptureLogger:
    def __init__(self):
        self.messages = []
        self.progress_updates = []

    def info(self, message: str) -> None:
        self.messages.append(message)

    def warn(self, message: str) -> None:
        self.messages.append(f"warning: {message}")

    def progress(self, current: int, total: int | None) -> None:
        self.progress_updates.append((current, total))


def test_extract_color():
    assert star_scraper.extract_color("color: #00ADD8;") == "#00ADD8"
    assert star_scraper.extract_color("background: red;") == ""


def test_parse_count():
    assert star_scraper.parse_count("1,234") == 1234
    assert star_scraper.parse_count("1.2k") == 1200
    assert star_scraper.parse_count("3M") == 3_000_000
    assert star_scraper.parse_count("") is None


def test_extract_total_stars_from_tab():
    html = """
    <html>
      <a data-tab-item="stars"><span class="Counter">3,456</span> Stars</a>
    </html>
    """
    soup = BeautifulSoup(html, "html.parser")
    assert star_scraper.extract_total_stars(soup) == 3456


def test_find_next_page_url_query_merge():
    html = '<a rel="next" href="?after=abc"></a>'
    soup = BeautifulSoup(html, "html.parser")
    current_url = "https://github.com/user?tab=stars"
    next_url = star_scraper.find_next_page_url(soup, current_url)
    assert next_url == "https://github.com/user?tab=stars&after=abc"


def test_extract_star_entries_single_repo():
    html = """
    <div id="user-starred-repos">
      <div class="d-block">
        <a href="/owner/repo">owner / repo</a>
        <p itemprop="description">Example</p>
        <a href="/owner/repo/stargazers">1,234</a>
        <a href="/owner/repo/forks">12</a>
        <span itemprop="programmingLanguage">Go</span>
        <span class="repo-language-color" style="color: #00ADD8;"></span>
        <relative-time datetime="2026-01-01T00:00:00Z"></relative-time>
      </div>
    </div>
    """
    soup = BeautifulSoup(html, "html.parser")
    stars = star_scraper.extract_star_entries(soup, "https://github.com")
    assert len(stars) == 1
    repo = stars[0]
    assert repo["link"] == "https://github.com/owner/repo"
    assert repo["name"] == "owner / repo"
    assert repo["description"] == "Example"
    assert repo["stars"] == "1,234"
    assert repo["language"] == "Go"
    assert repo["language_color"] == "#00ADD8"
    assert repo["forks"] == "12"
    assert repo["updated"] == "2026-01-01T00:00:00Z"


def test_extract_list_links():
    html = """
    <a href="/stars/user/lists/cool-stuff">Cool Stuff</a>
    <a href="/stars/user/lists/another">Another</a>
    """
    soup = BeautifulSoup(html, "html.parser")
    links = star_scraper.extract_list_links(soup, "user", "https://github.com")
    assert links == [
        {"name": "Another", "url": "https://github.com/stars/user/lists/another"},
        {"name": "Cool Stuff", "url": "https://github.com/stars/user/lists/cool-stuff"},
    ]


def test_request_manager_retries_on_rate_limit(monkeypatch):
    session = DummySession(
        [
            make_response(429, headers={"Retry-After": "2"}),
            make_response(200, text="ok"),
        ]
    )
    logger = star_scraper.Logger(verbose=True)
    sleeps = []

    monkeypatch.setattr(star_scraper.random, "random", lambda: 0)
    monkeypatch.setattr(star_scraper.time, "sleep", lambda s: sleeps.append(s))
    monkeypatch.setattr(star_scraper.time, "monotonic", lambda: 0)
    monkeypatch.setattr(star_scraper.time, "time", lambda: 0)

    manager = star_scraper.RequestManager(
        session=session,
        timeout=5,
        logger=logger,
        max_retries=2,
        backoff_base=1.0,
        backoff_max=1.0,
        min_interval=0.0,
        cb_threshold=5,
        cb_cooldown=1.0,
    )
    response = manager.get("https://example.com")
    assert response.status_code == 200
    assert sleeps and sleeps[0] == pytest.approx(2.0)


def test_circuit_breaker_waits(monkeypatch):
    responses = [
        make_response(500),
        make_response(500),
        make_response(500),
        make_response(500),
        make_response(200, text="ok"),
    ]
    session = DummySession(responses)
    logger = star_scraper.Logger(verbose=True)
    sleeps = []

    monkeypatch.setattr(star_scraper.random, "random", lambda: 0)
    monkeypatch.setattr(star_scraper.time, "sleep", lambda s: sleeps.append(s))
    monkeypatch.setattr(star_scraper.time, "monotonic", lambda: 0)
    monkeypatch.setattr(star_scraper.time, "time", lambda: 0)

    manager = star_scraper.RequestManager(
        session=session,
        timeout=5,
        logger=logger,
        max_retries=1,
        backoff_base=0.1,
        backoff_max=0.1,
        min_interval=0.0,
        cb_threshold=2,
        cb_cooldown=5.0,
    )

    with pytest.raises(requests.HTTPError):
        manager.get("https://example.com")
    with pytest.raises(requests.HTTPError):
        manager.get("https://example.com")

    response = manager.get("https://example.com")
    assert response.status_code == 200
    assert sleeps and sleeps[-1] == pytest.approx(5.0)


def test_fetch_all_stars_reports_progress(monkeypatch):
    html = """
    <html>
      <a data-tab-item="stars"><span class="Counter">2</span> Stars</a>
      <div id="user-starred-repos">
        <div class="d-block">
          <a href="/owner/repo">owner / repo</a>
        </div>
        <div class="d-block">
          <a href="/owner/repo2">owner / repo2</a>
        </div>
      </div>
    </html>
    """
    session = DummySession([make_response(200, text=html)])
    logger = CaptureLogger()

    monkeypatch.setattr(star_scraper.time, "monotonic", lambda: 0)

    manager = star_scraper.RequestManager(
        session=session,
        timeout=5,
        logger=logger,
        max_retries=0,
        backoff_base=0.1,
        backoff_max=0.1,
        min_interval=0.0,
        cb_threshold=5,
        cb_cooldown=1.0,
    )

    stars = star_scraper.fetch_all_stars(manager, "https://github.com/user?tab=stars", logger)
    assert len(stars) == 2
    assert logger.progress_updates[-1] == (2, 2)
