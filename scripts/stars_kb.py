"""Encrypted knowledge base of a user's GitHub stars.

CI (GITHUB_TOKEN set; output is plain gzip, the workflow encrypts it with age):
  python scripts/stars_kb.py list --user NAME --out assets/data/stars.json
  python scripts/stars_kb.py delta --old OLD.json --new assets/data/stars.json --out delta.jsonl.gz
  python scripts/stars_kb.py full --stars assets/data/stars.json --out base.jsonl.gz

Local (needs the age CLI >= 1.3 and the private key):
  python scripts/stars_kb.py sync [--bw ITEM | --key PATH] [--repo OWNER/NAME] [--data DIR]
  python scripts/stars_kb.py import --corpus corpus.jsonl [--data DIR]
  python scripts/stars_kb.py search "beat detection" [-n 20] [--strict] [--license MIT]
  python scripts/stars_kb.py show owner/repo
  python scripts/stars_kb.py stats
"""
import argparse
import gzip
import json
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / ".stars-kb"
RELEASE_TAG = "stars-kb"
BATCH = 50
README_LIMIT = 8000
FIELDS = ("databaseId nameWithOwner description stargazerCount pushedAt isArchived primaryLanguage{name} "
          "licenseInfo{spdxId} repositoryTopics(first:20){nodes{topic{name}}} "
          "a:object(expression:\"HEAD:README.md\"){...on Blob{text}} "
          "b:object(expression:\"HEAD:readme.md\"){...on Blob{text}} "
          "c:object(expression:\"HEAD:README.rst\"){...on Blob{text}}")


def cmd_list(a):
    token = os.environ["GITHUB_TOKEN"]
    stars, after, totals = [], "", set()
    while True:
        q = ('query{user(login:%s){starredRepositories(first:100, orderBy:{field:STARRED_AT, direction:DESC}%s)'
             '{totalCount pageInfo{hasNextPage endCursor} edges{starredAt node{nameWithOwner url description '
             'stargazerCount forkCount pushedAt primaryLanguage{name color}}}}}}') % (json.dumps(a.user), after)
        page = graphql(q, token)
        conn = ((page or {}).get("data") or {}).get("user", {}).get("starredRepositories")
        if not conn:
            sys.exit(f"starred list failed: {json.dumps(page)[:300]}")
        totals.add(conn["totalCount"])
        for e in conn["edges"]:
            n = e["node"]
            stars.append({"repo": n["nameWithOwner"], "link": n["url"], "description": n["description"] or "",
                          "stars": n["stargazerCount"], "forks": n["forkCount"],
                          "language": (n["primaryLanguage"] or {}).get("name"),
                          "language_color": (n["primaryLanguage"] or {}).get("color"),
                          "updated": n["pushedAt"], "starred_at": e["starredAt"]})
        if not conn["pageInfo"]["hasNextPage"]:
            break
        after = ", after:" + json.dumps(conn["pageInfo"]["endCursor"])
    unique = len({s["repo"].lower() for s in stars})
    if len(totals) != 1 or unique != len(stars) or unique != conn["totalCount"]:
        sys.exit(f"listed {unique} unique of {len(stars)} rows, totalCount {sorted(totals)}; "
                 "stars changed during listing or the list is incomplete, not writing it")
    out = {"username": a.user, "last_updated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "stars": stars}
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"{len(stars)} stars -> {a.out}", file=sys.stderr)


def repo_names(stars_json):
    data = json.loads(Path(stars_json).read_text(encoding="utf-8"))
    out = []
    for s in data["stars"]:
        parts = s["link"].rstrip("/").split("/")
        out.append(f"{parts[-2]}/{parts[-1]}")
    return out


def readme_text(raw):
    raw = re.sub(r"<!--.*?-->", " ", raw or "", flags=re.S)
    raw = re.sub(r"!\[[^\]]*\]", " ", raw)
    raw = re.sub(r"<[^>]+>", " ", raw)
    return " ".join(raw.split())[:README_LIMIT]


def to_row(d):
    raw = next((d[k]["text"] for k in "abc" if d.get(k) and d[k].get("text")), "")
    return {"repo": d["nameWithOwner"], "id": d["databaseId"], "desc": d["description"] or "",
            "topics": [t["topic"]["name"] for t in d["repositoryTopics"]["nodes"]],
            "license": (d["licenseInfo"] or {}).get("spdxId"), "stars": d["stargazerCount"],
            "pushed": d["pushedAt"] or "", "archived": d["isArchived"],
            "lang": (d["primaryLanguage"] or {}).get("name"), "readme": readme_text(raw)}


def graphql(query, token):
    req = urllib.request.Request("https://api.github.com/graphql", data=json.dumps({"query": query}).encode(),
                                 headers={"Authorization": f"bearer {token}", "Content-Type": "application/json"})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.load(r)
        except (urllib.error.URLError, TimeoutError) as e:
            if attempt == 3:
                raise
            time.sleep(5 * (attempt + 1))
            print(f"retry after {e}", file=sys.stderr)


def query_repos(repos, fields, token):
    """Yield (name, node); node is None only when GitHub answers NOT_FOUND for that repository."""
    for i in range(0, len(repos), BATCH):
        chunk = repos[i:i + BATCH]
        q = "query{" + " ".join(f'r{k}:repository(owner:{json.dumps(o)},name:{json.dumps(n)}){{{fields}}}'
                                for k, (o, n) in enumerate(r.split("/", 1) for r in chunk)) + "}"
        resp = graphql(q, token) or {}
        data = resp.get("data") or {}
        errors = resp.get("errors") or []
        gone = {e["path"][0] for e in errors if e.get("type") == "NOT_FOUND" and e.get("path")}
        other = [e for e in errors if e.get("type") != "NOT_FOUND"]
        bad = [r for k, r in enumerate(chunk) if not data.get(f"r{k}") and f"r{k}" not in gone]
        if bad or other:
            sys.exit(f"{len(bad)} of {len(chunk)} repositories not returned ({', '.join(bad[:5])}); "
                     f"errors: {json.dumps(other)[:300]}")
        yield from ((r, data.get(f"r{k}")) for k, r in enumerate(chunk))
        print(f"queried {min(i + BATCH, len(repos))}/{len(repos)}", file=sys.stderr)


def mark(repo, status, **extra):
    return {"repo": repo, "status": status, "status_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **extra}


def fetch(repos, token):
    return [dict(to_row(d), repo=r) if d else mark(r, "unavailable") for r, d in query_repos(repos, FIELDS, token)]


def probe_removed(repos, token):
    out = []
    for r, d in query_repos(repos, "databaseId nameWithOwner", token):
        if not d:
            out.append(mark(r, "unavailable"))
        elif d["nameWithOwner"].lower() != r.lower():
            out.append(mark(r, "renamed", id=d["databaseId"], renamed_to=d["nameWithOwner"]))
        else:
            out.append(mark(r, "unstarred", id=d["databaseId"]))
    return out


def write_gz(rows, out):
    with gzip.open(out, "wt", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{len(rows)} rows -> {out}", file=sys.stderr)


def cmd_delta(a):
    old = repo_names(a.old) if a.old and Path(a.old).exists() else []
    cur = repo_names(a.new)
    old_keys, cur_keys = {r.lower() for r in old}, {r.lower() for r in cur}
    new = [r for r in cur if r.lower() not in old_keys]
    removed = [r for r in old if r.lower() not in cur_keys]
    print(f"new stars: {len(new)}, removed: {len(removed)}", file=sys.stderr)
    if not new and not removed:
        return
    token = os.environ["GITHUB_TOKEN"]
    write_gz(fetch(new, token) + probe_removed(removed, token), a.out)


def cmd_full(a):
    write_gz(fetch(repo_names(a.stars), os.environ["GITHUB_TOKEN"]), a.out)


def http_json(url):
    req = urllib.request.Request(url, headers={"Accept": "application/vnd.github+json"})
    tok = os.environ.get("GITHUB_TOKEN")
    if tok:
        req.add_header("Authorization", f"bearer {tok}")
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def read_key(a):
    if a.key:
        return Path(a.key).read_bytes()
    bw = shutil.which("bw") or sys.exit("Bitwarden CLI `bw` not found on PATH")
    r = subprocess.run([bw, "get", "notes", a.bw], capture_output=True)
    if r.returncode or b"AGE-SECRET-KEY-" not in r.stdout:
        sys.exit(f"bw get notes {a.bw} failed; run `bw unlock` and set BW_SESSION. {r.stderr.decode(errors='replace').strip()}")
    return r.stdout


def cmd_sync(a):
    key = read_key(a)
    data = Path(a.data)
    enc = data / "encrypted"
    enc.mkdir(parents=True, exist_ok=True)
    rel = http_json(f"https://api.github.com/repos/{a.repo}/releases/tags/{RELEASE_TAG}")
    assets = sorted((x for x in rel["assets"] if re.match(r"(base|delta)-.*\.jsonl\.gz\.age$", x["name"])),
                    key=lambda x: (x["name"].split("-", 1)[1], not x["name"].startswith("base-")))
    if not any(x["name"].startswith("base-") for x in assets):
        sys.exit("no base-*.jsonl.gz.age in the release; run the rebase workflow or `import` first")
    kb = Corpus()
    for x in assets:
        path = enc / x["name"]
        if not path.exists() or path.stat().st_size != x["size"]:
            urllib.request.urlretrieve(x["browser_download_url"], path)
        plain = subprocess.run(["age", "-d", "-i", "-", str(path)], input=key, capture_output=True, check=True).stdout
        for line in gzip.decompress(plain).decode("utf-8").split("\n"):
            if line.strip():
                kb.merge(json.loads(line))
        print(f"{x['name']}: {len(kb.rows)} repos so far", file=sys.stderr)
    live = repo_names_from_url(f"https://raw.githubusercontent.com/{a.repo}/main/assets/data/stars.json")
    missing = sorted(live - kb.names.keys())
    if missing:
        sys.exit(f"{len(missing)} starred repositories are not in the release ({', '.join(missing[:5])}); "
                 "local database left unchanged. Wait for the nightly run or run the rebase workflow.")
    kb.settle(live)
    rows = kb.rows
    corpus = data / "corpus.jsonl"
    with open(corpus, "w", encoding="utf-8") as f:
        for r in rows.values():
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    counts = {}
    for r in rows.values():
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    print(f"corpus: {len(rows)} repos {counts}", file=sys.stderr)
    build(corpus, data / "stars.db")


BLANK = {"desc": "", "topics": [], "license": None, "stars": None, "pushed": "", "archived": False, "lang": None,
         "readme": ""}


class Corpus:
    """Records keyed by GitHub repository id, so a name taken over by another repository keeps two histories."""

    def __init__(self):
        self.rows = {}
        self.names = {}

    def key_for(self, r):
        name = r["repo"].lower()
        if r.get("id") is None:
            return self.names.get(name, name)
        key = f"id:{r['id']}"
        old = self.names.get(name)
        if key not in self.rows and old in self.rows and self.rows[old].get("id") is None:
            self.rows[key] = self.rows.pop(old)
        return key

    def merge(self, r):
        key = self.key_for(r)
        cur = self.rows.get(key)
        if "status" not in r:
            rec = {**r, "status": "starred"}
            if cur and cur["repo"].lower() != r["repo"].lower():
                rec["renamed_from"] = cur["repo"]
            self.rows[key] = rec
            self.names[r["repo"].lower()] = key
        elif cur and r["status"] == "renamed" and cur["repo"].lower() == r["renamed_to"].lower():
            return
        else:
            self.rows[key] = {**BLANK, "repo": r["repo"], **(cur or {}),
                              **{k: v for k, v in r.items() if k != "repo"}}
            self.names.setdefault(r["repo"].lower(), key)

    def settle(self, live):
        for key, r in self.rows.items():
            if r["repo"].lower() in live and self.names.get(r["repo"].lower()) == key:
                r["status"] = "starred"
            elif r["status"] == "starred":
                r["status"] = "unstarred"


def repo_names_from_url(url):
    with urllib.request.urlopen(url, timeout=60) as r:
        data = json.load(r)
    return {"/".join(s["link"].rstrip("/").split("/")[-2:]).lower() for s in data["stars"]}


def cmd_import(a):
    data = Path(a.data)
    data.mkdir(parents=True, exist_ok=True)
    build(Path(a.corpus), data / "stars.db")


def build(corpus, db):
    tmp = db.with_suffix(".db.tmp")
    if tmp.exists():
        tmp.unlink()
    con = sqlite3.connect(tmp)
    con.executescript("""
    CREATE TABLE repo(rid INTEGER PRIMARY KEY, repo TEXT, stars INT, license TEXT, lang TEXT, pushed TEXT, archived INT,
                      topics TEXT, descr TEXT, readme TEXT, status TEXT, status_at TEXT, renamed_to TEXT,
                      gh_id INT);
    CREATE VIRTUAL TABLE fts USING fts5(repo, topics, descr, readme, tokenize='porter unicode61');
    """)
    n = 0
    with open(corpus, encoding="utf-8") as f:
        for line in f:
            r = {**BLANK, "status": "starred", **json.loads(line)}
            topics = " ".join(r["topics"])
            rid = con.execute("INSERT INTO repo VALUES (NULL,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                              (r["repo"], r["stars"], r["license"], r["lang"], (r["pushed"] or "")[:10],
                               int(r["archived"]), topics, r["desc"], r["readme"], r["status"],
                               r.get("status_at"), r.get("renamed_to"), r.get("id"))).lastrowid
            con.execute("INSERT INTO fts(rowid, repo, topics, descr, readme) VALUES (?,?,?,?,?)",
                        (rid, r["repo"], topics, r["desc"], r["readme"]))
            n += 1
    con.commit()
    con.close()
    os.replace(tmp, db)
    print(f"{n} repos indexed -> {db}", file=sys.stderr)


def connect(a):
    db = Path(a.data) / "stars.db"
    if not db.exists():
        sys.exit(f"{db} not found; run `sync` or `import` first")
    return sqlite3.connect(db)


def cmd_search(a):
    con = connect(a)
    q = a.query if not a.strict else " AND ".join(f"{{topics descr}}: {w}" for w in a.query.split())
    sql = ("SELECT r.repo, r.stars, r.license, r.lang, r.descr, r.status FROM fts JOIN repo r ON r.rid = fts.rowid "
           "WHERE fts MATCH ?" + (" AND r.license = ?" if a.license else "")
           + (" AND r.status = 'starred'" if a.starred else "") + " ORDER BY bm25(fts) LIMIT ?")
    args = [q] + ([a.license] if a.license else []) + [a.n]
    for repo, stars, lic, lang, descr, status in con.execute(sql, args):
        tag = "" if status == "starred" else f"  [{status.upper()}]"
        print(f"{repo}  ★{stars}  {lic or '-'}  {lang or '-'}{tag}\n    {(descr or '')[:160]}")


def cmd_show(a):
    rows = connect(a).execute("SELECT * FROM repo WHERE lower(repo) = lower(?) ORDER BY status = 'starred' DESC",
                              (a.repo,)).fetchall()
    if not rows:
        sys.exit("not found")
    for _, repo, stars, lic, lang, pushed, archived, topics, descr, readme, status, status_at, renamed_to, gh_id in rows:
        print(f"{repo}  ★{stars}  {lic or '-'}  {lang or '-'}  pushed {pushed}{'  ARCHIVED' if archived else ''}"
              f"{f'  id {gh_id}' if gh_id else ''}")
        if status != "starred":
            print(f"{status.upper()} {status_at or ''}{f' -> {renamed_to}' if renamed_to else ''}")
        print(f"topics: {topics}\n{descr}\n\n{readme[:1500]}\n")


def cmd_stats(a):
    con = connect(a)
    print("repos:", con.execute("SELECT count(*) FROM repo").fetchone()[0])
    for status, n in con.execute("SELECT status, count(*) FROM repo GROUP BY status ORDER BY 2 DESC"):
        print(f"  {status}: {n}")
    for lang, n in con.execute("SELECT lang, count(*) c FROM repo GROUP BY lang ORDER BY c DESC LIMIT 10"):
        print(f"  {lang or '-'}: {n}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default=str(DEFAULT_DATA))
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("list"); s.add_argument("--user", required=True); s.add_argument("--out", required=True)
    s = sub.add_parser("delta"); s.add_argument("--old"); s.add_argument("--new", required=True); s.add_argument("--out", required=True)
    s = sub.add_parser("full"); s.add_argument("--stars", required=True); s.add_argument("--out", required=True)
    s = sub.add_parser("sync"); s.add_argument("--repo", default="PhysShell/reach-for-the-stars")
    k = s.add_mutually_exclusive_group(); k.add_argument("--key"); k.add_argument("--bw", default="stars-kb")
    s = sub.add_parser("import"); s.add_argument("--corpus", required=True)
    s = sub.add_parser("search"); s.add_argument("query"); s.add_argument("-n", type=int, default=20)
    s.add_argument("--strict", action="store_true"); s.add_argument("--license")
    s.add_argument("--starred", action="store_true", help="only repositories that are still starred")
    s = sub.add_parser("show"); s.add_argument("repo")
    sub.add_parser("stats")
    a = p.parse_args()
    {"list": cmd_list, "delta": cmd_delta, "full": cmd_full, "sync": cmd_sync, "import": cmd_import,
     "search": cmd_search, "show": cmd_show, "stats": cmd_stats}[a.cmd](a)


if __name__ == "__main__":
    main()
