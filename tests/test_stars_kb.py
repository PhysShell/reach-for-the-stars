import argparse
import gzip
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import stars_kb as kb


def repo(name):
    return {"nameWithOwner": name, "description": "d", "stargazerCount": 1, "pushedAt": "2026-01-01T00:00:00Z",
            "isArchived": False, "primaryLanguage": None, "licenseInfo": None,
            "repositoryTopics": {"nodes": []}, "a": {"text": "readme"}, "b": None, "c": None}


class Fetch(unittest.TestCase):
    def run_fetch(self, response, repos=("o/a", "o/b")):
        with mock.patch.object(kb, "graphql", return_value=response):
            return kb.fetch(list(repos), "t")

    def test_all_returned(self):
        rows = self.run_fetch({"data": {"r0": repo("o/a"), "r1": repo("o/b")}})
        self.assertEqual([r["repo"] for r in rows], ["o/a", "o/b"])

    def test_missing_alias_fails(self):
        with self.assertRaises(SystemExit):
            self.run_fetch({"data": {"r0": repo("o/a")}})

    def test_not_found_is_marked_deleted(self):
        rows = self.run_fetch({"data": {"r0": repo("o/a"), "r1": None},
                               "errors": [{"type": "NOT_FOUND", "path": ["r1"]}]})
        self.assertEqual(rows[1]["repo"], "o/b")
        self.assertEqual(rows[1]["status"], "deleted")

    def test_null_with_other_error_fails(self):
        with self.assertRaises(SystemExit):
            self.run_fetch({"data": {"r0": repo("o/a"), "r1": None},
                            "errors": [{"type": "FORBIDDEN", "path": ["r1"]}]})

    def test_errors_only_fails(self):
        with self.assertRaises(SystemExit):
            self.run_fetch({"errors": [{"type": "RATE_LIMITED"}]})

    def test_renamed_repo_keeps_listed_name(self):
        rows = self.run_fetch({"data": {"r0": repo("new/a")}}, repos=("old/a",))
        self.assertEqual(rows[0]["repo"], "old/a")


class Delta(unittest.TestCase):
    def test_removed_repositories_are_classified(self):
        tmp = Path(tempfile.mkdtemp())

        def stars(path, names):
            path.write_text(json.dumps({"stars": [{"link": f"https://github.com/{n}"} for n in names]}),
                            encoding="utf-8")
            return str(path)

        old = stars(tmp / "old.json", ["o/keep", "o/gone", "o/moved", "o/unstar"])
        new = stars(tmp / "new.json", ["o/keep", "o/new"])
        answers = [{"data": {"r0": repo("o/new")}},
                   {"data": {"r0": None, "r1": {"nameWithOwner": "x/moved"}, "r2": {"nameWithOwner": "o/unstar"}},
                    "errors": [{"type": "NOT_FOUND", "path": ["r0"]}]}]
        out = tmp / "d.jsonl.gz"
        with mock.patch.object(kb, "graphql", side_effect=answers), mock.patch.dict(os.environ, {"GITHUB_TOKEN": "t"}):
            kb.cmd_delta(argparse.Namespace(old=old, new=new, out=str(out)))
        rows = {r["repo"]: r for r in map(json.loads, gzip.decompress(out.read_bytes()).decode().splitlines())}
        self.assertNotIn("status", rows["o/new"])
        self.assertEqual(rows["o/gone"]["status"], "deleted")
        self.assertEqual((rows["o/moved"]["status"], rows["o/moved"]["renamed_to"]), ("renamed", "x/moved"))
        self.assertEqual(rows["o/unstar"]["status"], "unstarred")


def star_page(names, total, more, cursor="c"):
    edges = [{"starredAt": "2026-01-01T00:00:00Z",
              "node": {"nameWithOwner": n, "url": f"https://github.com/{n}", "description": None,
                       "stargazerCount": 1, "forkCount": 0, "pushedAt": None, "primaryLanguage": None}}
             for n in names]
    return {"data": {"user": {"starredRepositories": {
        "totalCount": total, "pageInfo": {"hasNextPage": more, "endCursor": cursor}, "edges": edges}}}}


class List(unittest.TestCase):
    def run_list(self, pages):
        out = Path(tempfile.mkdtemp()) / "stars.json"
        with mock.patch.object(kb, "graphql", side_effect=pages), mock.patch.dict(os.environ, {"GITHUB_TOKEN": "t"}):
            kb.cmd_list(argparse.Namespace(user="u", out=str(out)))
        return out

    def test_complete_list_written(self):
        out = self.run_list([star_page(["o/a"], 2, True), star_page(["o/b"], 2, False)])
        self.assertEqual(len(json.loads(out.read_text(encoding="utf-8"))["stars"]), 2)

    def test_short_list_refused(self):
        with self.assertRaises(SystemExit):
            self.run_list([star_page(["o/a"], 2, False)])

    def test_duplicate_hiding_a_gap_refused(self):
        with self.assertRaises(SystemExit):
            self.run_list([star_page(["o/a", "o/b"], 3, True), star_page(["o/b"], 3, False)])

    def test_total_changed_during_listing_refused(self):
        with self.assertRaises(SystemExit):
            self.run_list([star_page(["o/a"], 2, True), star_page(["o/b", "o/c"], 3, False)])


class Sync(unittest.TestCase):
    def setUp(self):
        self.data = Path(tempfile.mkdtemp())
        self.files = {}

    def asset(self, name, rows):
        full = dict(desc="", topics=[], license=None, stars=0, pushed="", archived=False, lang=None, readme="")
        self.files[name] = gzip.compress("".join(json.dumps(r if "status" in r else {**full, **r}) + "\n"
                                                 for r in rows).encode())
        return {"name": name, "size": len(self.files[name]), "browser_download_url": name}

    def run_sync(self, assets, live):
        def retrieve(url, path):
            Path(path).write_bytes(self.files[url])

        def age(cmd, **kw):
            return subprocess.CompletedProcess(cmd, 0, stdout=Path(cmd[-1]).read_bytes())

        with mock.patch.object(kb, "read_key", return_value=b"k"), \
                mock.patch.object(kb, "http_json", return_value={"assets": assets}), \
                mock.patch.object(kb.urllib.request, "urlretrieve", side_effect=retrieve), \
                mock.patch.object(kb.subprocess, "run", side_effect=age), \
                mock.patch.object(kb, "repo_names_from_url", return_value=set(live)):
            kb.cmd_sync(argparse.Namespace(data=str(self.data), repo="x/y", key=None, bw="stars-kb"))

    def indexed(self):
        return set(self.status())

    def status(self):
        import sqlite3
        return dict(sqlite3.connect(self.data / "stars.db").execute("SELECT repo, status FROM repo"))

    def test_removed_repositories_are_kept_with_status(self):
        self.run_sync([self.asset("base-20260101T000000.jsonl.gz.age",
                                  [{"repo": "o/a"}, {"repo": "o/b", "readme": "kept text"}, {"repo": "o/c"},
                                   {"repo": "o/d"}]),
                       self.asset("delta-20260102T000000.jsonl.gz.age",
                                  [{"repo": "o/b", "status": "deleted"},
                                   {"repo": "o/c", "status": "renamed", "renamed_to": "x/c"},
                                   {"repo": "x/c"}])],
                      ["o/a", "x/c"])
        self.assertEqual(self.status(), {"o/a": "starred", "o/b": "deleted", "o/c": "renamed",
                                         "o/d": "unstarred", "x/c": "starred"})
        import sqlite3
        con = sqlite3.connect(self.data / "stars.db")
        self.assertEqual(con.execute("SELECT readme FROM repo WHERE repo='o/b'").fetchone()[0], "kept text")
        self.assertEqual(con.execute("SELECT count(*) FROM fts WHERE fts MATCH 'kept'").fetchone()[0], 1)

    def test_newer_base_does_not_erase_older_history(self):
        self.run_sync([self.asset("base-20260101T000000.jsonl.gz.age", [{"repo": "o/a"}, {"repo": "o/old"}]),
                       self.asset("base-20260105T000000.jsonl.gz.age", [{"repo": "o/a", "stars": 9}])],
                      ["o/a"])
        self.assertEqual(self.status(), {"o/a": "starred", "o/old": "unstarred"})

    def test_restarred_repository_is_starred_again(self):
        self.run_sync([self.asset("base-20260101T000000.jsonl.gz.age", [{"repo": "o/a"}]),
                       self.asset("delta-20260102T000000.jsonl.gz.age", [{"repo": "o/a", "status": "unstarred"}]),
                       self.asset("delta-20260103T000000.jsonl.gz.age", [{"repo": "o/a"}])],
                      ["o/a"])
        self.assertEqual(self.status(), {"o/a": "starred"})

    def test_base_and_deltas_cover_live(self):
        self.run_sync([self.asset("base-20260101T000000.jsonl.gz.age", [{"repo": "o/a"}]),
                       self.asset("delta-20260101T000000.jsonl.gz.age", [{"repo": "o/b"}]),
                       self.asset("delta-20260102T000000.jsonl.gz.age", [{"repo": "o/c"}])],
                      ["o/a", "o/b", "o/c"])
        self.assertEqual(self.indexed(), {"o/a", "o/b", "o/c"})

    def test_missing_repo_keeps_existing_database(self):
        self.run_sync([self.asset("base-20260101T000000.jsonl.gz.age", [{"repo": "o/a"}])], ["o/a"])
        with self.assertRaises(SystemExit):
            self.run_sync([self.asset("base-20260102T000000.jsonl.gz.age", [{"repo": "o/a"}])], ["o/a", "o/b"])
        self.assertEqual(self.indexed(), {"o/a"})


if __name__ == "__main__":
    unittest.main()
