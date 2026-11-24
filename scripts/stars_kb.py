"""Encrypted knowledge base of a user's GitHub stars.

CI (GITHUB_TOKEN set; output is plain gzip, the workflow encrypts it with age):
  python scripts/stars_kb.py delta --old OLD.json --new assets/data/stars.json --out delta.jsonl.gz
  python scripts/stars_kb.py full --stars assets/data/stars.json --out base.jsonl.gz

Local (needs the age CLI >= 1.3 and the private key):
  python scripts/stars_kb.py sync --key PATH/stars-kb.key [--repo OWNER/NAME] [--data DIR]
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
FIELDS = ("nameWithOwner description stargazerCount pushedAt isArchived primaryLanguage{name} "
          "licenseInfo{spdxId} repositoryTopics(first:20){nodes{topic{name}}} "
          "a:object(expression:\"HEAD:README.md\"){...on Blob{text}} "
          "b:object(expression:\"HEAD:readme.md\"){...on Blob{text}} "
          "c:object(expression:\"HEAD:README.rst\"){...on Blob{text}}")


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
    return {"repo": d["nameWithOwner"], "desc": d["description"] or "",
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


def fetch(repos, token):
    rows = []
    for i in range(0, len(repos), BATCH):
        chunk = repos[i:i + BATCH]
        q = "query{" + " ".join(f'r{k}:repository(owner:{json.dumps(o)},name:{json.dumps(n)}){{{FIELDS}}}'
                                for k, (o, n) in enumerate(r.split("/", 1) for r in chunk)) + "}"
        data = (graphql(q, token) or {}).get("data") or {}
        rows += [to_row(d) for d in data.values() if d]
        print(f"fetched {min(i + BATCH, len(repos))}/{len(repos)}", file=sys.stderr)
    return rows


def write_gz(rows, out):
    with gzip.open(out, "wt", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{len(rows)} rows -> {out}", file=sys.stderr)


def cmd_delta(a):
    old = set(r.lower() for r in repo_names(a.old)) if a.old and Path(a.old).exists() else set()
    new = [r for r in repo_names(a.new) if r.lower() not in old]
    print(f"new stars: {len(new)}", file=sys.stderr)
    if not new:
        return
    write_gz(fetch(new, os.environ["GITHUB_TOKEN"]), a.out)


def cmd_full(a):
    write_gz(fetch(repo_names(a.stars), os.environ["GITHUB_TOKEN"]), a.out)


def http_json(url):
    req = urllib.request.Request(url, headers={"Accept": "application/vnd.github+json"})
    tok = os.environ.get("GITHUB_TOKEN")
    if tok:
        req.add_header("Authorization", f"bearer {tok}")
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def cmd_sync(a):
    data = Path(a.data)
    enc = data / "encrypted"
    enc.mkdir(parents=True, exist_ok=True)
    rel = http_json(f"https://api.github.com/repos/{a.repo}/releases/tags/{RELEASE_TAG}")
    assets = sorted((x for x in rel["assets"] if x["name"].endswith(".jsonl.gz.age")), key=lambda x: x["name"])
    bases = [x for x in assets if x["name"].startswith("base-")]
    if not bases:
        sys.exit("no base-*.jsonl.gz.age in the release; run the rebase workflow or `import` first")
    base = bases[-1]
    chain = [base] + [x for x in assets if x["name"].startswith("delta-") and x["name"][6:] > base["name"][5:]]
    rows = {}
    for x in chain:
        path = enc / x["name"]
        if not path.exists() or path.stat().st_size != x["size"]:
            urllib.request.urlretrieve(x["browser_download_url"], path)
        plain = subprocess.run(["age", "-d", "-i", a.key, str(path)], capture_output=True, check=True).stdout
        for line in gzip.decompress(plain).decode("utf-8").splitlines():
            r = json.loads(line)
            rows[r["repo"].lower()] = r
        print(f"{x['name']}: {len(rows)} repos so far", file=sys.stderr)
    live = repo_names_from_url(f"https://raw.githubusercontent.com/{a.repo}/main/assets/data/stars.json")
    kept = [r for k, r in rows.items() if k in live]
    corpus = data / "corpus.jsonl"
    with open(corpus, "w", encoding="utf-8") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"corpus: {len(kept)} starred ({len(rows) - len(kept)} no longer starred dropped)", file=sys.stderr)
    build(corpus, data / "stars.db")


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
    CREATE TABLE repo(repo TEXT PRIMARY KEY, stars INT, license TEXT, lang TEXT, pushed TEXT, archived INT,
                      topics TEXT, descr TEXT, readme TEXT);
    CREATE VIRTUAL TABLE fts USING fts5(repo, topics, descr, readme, tokenize='porter unicode61');
    """)
    n = 0
    with open(corpus, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            topics = " ".join(r["topics"])
            con.execute("INSERT OR REPLACE INTO repo VALUES (?,?,?,?,?,?,?,?,?)",
                        (r["repo"], r["stars"], r["license"], r["lang"], (r["pushed"] or "")[:10],
                         int(r["archived"]), topics, r["desc"], r["readme"]))
            con.execute("INSERT INTO fts VALUES (?,?,?,?)", (r["repo"], topics, r["desc"], r["readme"]))
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
    sql = ("SELECT r.repo, r.stars, r.license, r.lang, r.descr FROM fts JOIN repo r ON r.repo = fts.repo "
           "WHERE fts MATCH ?" + (" AND r.license = ?" if a.license else "") + " ORDER BY bm25(fts) LIMIT ?")
    args = [q] + ([a.license] if a.license else []) + [a.n]
    for repo, stars, lic, lang, descr in con.execute(sql, args):
        print(f"{repo}  ★{stars}  {lic or '-'}  {lang or '-'}\n    {(descr or '')[:160]}")


def cmd_show(a):
    row = connect(a).execute("SELECT * FROM repo WHERE lower(repo) = lower(?)", (a.repo,)).fetchone()
    if not row:
        sys.exit("not found")
    repo, stars, lic, lang, pushed, archived, topics, descr, readme = row
    print(f"{repo}  ★{stars}  {lic or '-'}  {lang or '-'}  pushed {pushed}{'  ARCHIVED' if archived else ''}")
    print(f"topics: {topics}\n{descr}\n\n{readme[:1500]}")


def cmd_stats(a):
    con = connect(a)
    print("repos:", con.execute("SELECT count(*) FROM repo").fetchone()[0])
    for lang, n in con.execute("SELECT lang, count(*) c FROM repo GROUP BY lang ORDER BY c DESC LIMIT 10"):
        print(f"  {lang or '-'}: {n}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default=str(DEFAULT_DATA))
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("delta"); s.add_argument("--old"); s.add_argument("--new", required=True); s.add_argument("--out", required=True)
    s = sub.add_parser("full"); s.add_argument("--stars", required=True); s.add_argument("--out", required=True)
    s = sub.add_parser("sync"); s.add_argument("--key", required=True); s.add_argument("--repo", default="PhysShell/reach-for-the-stars")
    s = sub.add_parser("import"); s.add_argument("--corpus", required=True)
    s = sub.add_parser("search"); s.add_argument("query"); s.add_argument("-n", type=int, default=20)
    s.add_argument("--strict", action="store_true"); s.add_argument("--license")
    s = sub.add_parser("show"); s.add_argument("repo")
    sub.add_parser("stats")
    a = p.parse_args()
    {"delta": cmd_delta, "full": cmd_full, "sync": cmd_sync, "import": cmd_import,
     "search": cmd_search, "show": cmd_show, "stats": cmd_stats}[a.cmd](a)


if __name__ == "__main__":
    main()
