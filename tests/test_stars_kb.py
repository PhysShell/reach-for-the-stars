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


def repo(name, gid=None):
    return {"databaseId": gid or abs(hash(name)) % 10**9, "nameWithOwner": name, "description": "d", "stargazerCount": 1, "pushedAt": "2026-01-01T00:00:00Z",
            "isArchived": False, "primaryLanguage": None, "licenseInfo": None,
            "repositoryTopics": {"nodes": []}, "a": {"text": "readme"}, "b": None, "c": None}


class Fetch(unittest.TestCase):
    def run_fetch(self, response, repos=("o/a", "o/b")):
        with mock.patch.object(kb, "graphql", return_value=response):
            return kb.fetch([live_entry(r) for r in repos], "t")

    def test_all_returned(self):
        rows = self.run_fetch({"data": {"r0": repo("o/a"), "r1": repo("o/b")}})
        self.assertEqual([r["repo"] for r in rows], ["o/a", "o/b"])

    def test_missing_alias_fails(self):
        with self.assertRaises(SystemExit):
            self.run_fetch({"data": {"r0": repo("o/a")}})

    def test_not_found_is_marked_unavailable(self):
        rows = self.run_fetch({"data": {"r0": repo("o/a"), "r1": None},
                               "errors": [{"type": "NOT_FOUND", "path": ["r1"]}]})
        self.assertEqual(rows[1]["repo"], "o/b")
        self.assertEqual(rows[1]["status"], "unavailable")

    def test_not_found_together_with_rate_limit_fails(self):
        with self.assertRaises(SystemExit):
            self.run_fetch({"data": {"r0": None, "r1": None},
                            "errors": [{"type": "NOT_FOUND", "path": ["r0"]}, {"type": "RATE_LIMITED"}]})

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
                   {"data": {"r0": None, "r1": {"databaseId": 7, "nameWithOwner": "x/moved"},
                             "r2": {"databaseId": 8, "nameWithOwner": "o/unstar"}},
                    "errors": [{"type": "NOT_FOUND", "path": ["r0"]}]}]
        out = tmp / "d.jsonl.gz"
        with mock.patch.object(kb, "graphql", side_effect=answers), mock.patch.dict(os.environ, {"GITHUB_TOKEN": "t"}):
            kb.cmd_delta(argparse.Namespace(old=old, new=new, out=str(out)))
        rows = {r["repo"]: r for r in map(json.loads, gzip.decompress(out.read_bytes()).decode().splitlines())}
        self.assertNotIn("status", rows["o/new"])
        self.assertEqual(rows["o/gone"]["status"], "unavailable")
        self.assertEqual((rows["o/moved"]["status"], rows["o/moved"]["renamed_to"]), ("renamed", "x/moved"))
        self.assertEqual((rows["o/unstar"]["status"], rows["o/unstar"]["id"]), ("unstarred", 8))


def star_page(names, total, more, cursor="c"):
    edges = [{"starredAt": "2026-01-01T00:00:00Z",
              "node": {"id": f"N_{n}", "databaseId": abs(hash(n)) % 10**9, "nameWithOwner": n, "url": f"https://github.com/{n}", "description": None,
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


def live_entry(x):
    if isinstance(x, dict):
        return x
    name, gid = x if isinstance(x, tuple) else (x, None)
    return {"repo": name, "id": gid, "node_id": None}


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
                mock.patch.object(kb, "live_entries", return_value=[live_entry(x) for x in live]):
            kb.cmd_sync(argparse.Namespace(data=str(self.data), repo="x/y", key=None, bw="stars-kb"))

    def indexed(self):
        return set(self.status())

    def status(self):
        return dict(self.db().execute("SELECT repo, status FROM repo"))

    def db(self):
        import contextlib
        import sqlite3

        class Q:
            def execute(_, sql):
                with contextlib.closing(sqlite3.connect(self.data / "stars.db")) as con:
                    return con.execute(sql).fetchall()
        return Q()

    def test_removed_repositories_are_kept_with_status(self):
        self.run_sync([self.asset("base-20260101T000000.jsonl.gz.age",
                                  [{"repo": "o/a"}, {"repo": "o/b", "readme": "kept text"}, {"repo": "o/c"},
                                   {"repo": "o/d"}]),
                       self.asset("delta-20260102T000000.jsonl.gz.age",
                                  [{"repo": "o/b", "status": "unavailable"},
                                   {"repo": "o/c", "status": "renamed", "renamed_to": "x/c"},
                                   {"repo": "x/c"}])],
                      ["o/a", "x/c"])
        self.assertEqual(self.status(), {"o/a": "starred", "o/b": "unavailable", "o/c": "renamed",
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


    def test_reused_name_keeps_two_histories(self):
        self.run_sync([self.asset("base-20260101T000000.jsonl.gz.age", [{"repo": "o/p", "id": 1, "readme": "first"}]),
                       self.asset("delta-20260102T000000.jsonl.gz.age",
                                  [{"repo": "o/p", "status": "renamed", "id": 1, "renamed_to": "o/q"}]),
                       self.asset("delta-20260103T000000.jsonl.gz.age", [{"repo": "o/p", "id": 2, "readme": "second"}])],
                      ["o/p"])
        rows = sorted(self.db().execute("SELECT repo, gh_id, status, readme FROM repo"))
        self.assertEqual(rows, [("o/p", 1, "renamed", "first"), ("o/p", 2, "starred", "second")])

    def test_rename_followed_by_new_name_is_one_record(self):
        self.run_sync([self.asset("base-20260101T000000.jsonl.gz.age", [{"repo": "o/p", "id": 1, "readme": "x"}]),
                       self.asset("delta-20260102T000000.jsonl.gz.age",
                                  [{"repo": "n/p", "id": 1, "readme": "y"},
                                   {"repo": "o/p", "status": "renamed", "id": 1, "renamed_to": "n/p"}])],
                      ["n/p"])
        self.assertEqual(self.status(), {"n/p": "starred"})

    def test_legacy_row_without_id_joins_its_id(self):
        self.run_sync([self.asset("base-20260101T000000.jsonl.gz.age", [{"repo": "o/p", "readme": "old"}]),
                       self.asset("delta-20260102T000000.jsonl.gz.age", [{"repo": "o/p", "status": "unstarred", "id": 5}])],
                      [])
        self.assertEqual(list(self.db().execute("SELECT repo, gh_id, status, readme FROM repo")),
                         [("o/p", 5, "unstarred", "old")])

    def test_unavailable_then_starred_again(self):
        self.run_sync([self.asset("base-20260101T000000.jsonl.gz.age", [{"repo": "o/p", "id": 1, "readme": "kept"}]),
                       self.asset("delta-20260102T000000.jsonl.gz.age", [{"repo": "o/p", "status": "unavailable"}])],
                      [])
        self.assertEqual(list(self.db().execute("SELECT status, readme FROM repo")), [("unavailable", "kept")])
        self.run_sync([self.asset("base-20260101T000000.jsonl.gz.age", [{"repo": "o/p", "id": 1, "readme": "kept"}]),
                       self.asset("delta-20260102T000000.jsonl.gz.age", [{"repo": "o/p", "status": "unavailable"}]),
                       self.asset("delta-20260103T000000.jsonl.gz.age", [{"repo": "o/p", "id": 1, "readme": "back"}])],
                      ["o/p"])
        self.assertEqual(list(self.db().execute("SELECT status, readme FROM repo")), [("starred", "back")])

    def test_sync_twice_gives_same_database(self):
        assets = [self.asset("base-20260101T000000.jsonl.gz.age", [{"repo": "o/a", "id": 1}, {"repo": "o/b", "id": 2}]),
                  self.asset("delta-20260102T000000.jsonl.gz.age", [{"repo": "o/b", "status": "unstarred", "id": 2}])]
        self.run_sync(assets, ["o/a"])
        first = list(self.db().execute("SELECT * FROM repo ORDER BY rid"))
        self.run_sync(assets, ["o/a"])
        self.assertEqual(list(self.db().execute("SELECT * FROM repo ORDER BY rid")), first)


    def test_star_with_known_id_but_no_record_is_missing(self):
        self.run_sync([self.asset("base-20260101T000000.jsonl.gz.age", [{"repo": "o/p", "id": 1}])], [("o/p", 1)])
        with self.assertRaises(SystemExit):
            self.run_sync([self.asset("base-20260101T000000.jsonl.gz.age", [{"repo": "o/p", "id": 1}])],
                          [("o/p", 1), ("o/p2", 2), ("x/p", 3)])
        with self.assertRaises(SystemExit):
            self.run_sync([self.asset("base-20260101T000000.jsonl.gz.age", [{"repo": "o/p", "id": 1}])], [("o/p", 2)])

    def test_new_repository_under_a_renamed_name_reaches_the_index(self):
        tmp = Path(tempfile.mkdtemp())

        def stars(path, items):
            path.write_text(json.dumps({"stars": [{"link": f"https://github.com/{n}", "id": i, "node_id": f"N{i}"}
                                                  for n, i in items]}), encoding="utf-8")
            return path

        old = stars(tmp / "old.json", [("o/p", 1)])
        new = stars(tmp / "new.json", [("o/q", 1), ("o/p", 2)])
        out = tmp / "d.jsonl.gz"
        sent = []

        def answer(q, token):
            sent.append(q)
            return {"data": {"r0": repo("o/p", 2)}}

        with mock.patch.object(kb, "graphql", side_effect=answer), mock.patch.dict(os.environ, {"GITHUB_TOKEN": "t"}):
            kb.cmd_delta(argparse.Namespace(old=str(old), new=str(new), out=str(out)))
        self.assertEqual(len(sent), 1)
        self.assertIn('node(id:"N2")', sent[0])
        self.files["delta"] = out.read_bytes()
        delta = {"name": "delta-20260102T000000.jsonl.gz.age", "size": len(self.files["delta"]),
                 "browser_download_url": "delta"}
        self.run_sync([self.asset("base-20260101T000000.jsonl.gz.age", [{"repo": "o/p", "id": 1, "readme": "old"}]),
                       delta], kb.load_entries(new))
        rows = sorted(self.db().execute("SELECT repo, gh_id, status, readme FROM repo"))
        self.assertEqual(rows, [("o/p", 2, "starred", "readme"), ("o/q", 1, "starred", "old")])


if __name__ == "__main__":
    unittest.main()
