"""ui_graph.py
====================================
A **thin Neo4j driver wrapper** that lets the DP‑autotest stack treat the
UI as a directed graph – each distinct screen (SHA‑1 hash) is a node;
any command that moves from BF → AF is an *edge*.

Key public methods
------------------
* `add_transition(bf_hash, af_hash, cmd, meta)`
* `state_exists(hash)`
* `shortest_path(src_hash, dst_hash, max_len)`

It also auto‑creates uniqueness constraints (`:Screen(hash)`) on first
run so you don’t have to manage schema manually.

Environment variables
---------------------
```
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASS=top‑secret
NEO4J_DB=neo4j   # optional
```

If those aren’t set you can pass credentials explicitly to
`UIGraph(uri, user, password, database)`.

Dependencies:  ``neo4j>=5.14``  (add to *requirements.txt*).
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import AbstractContextManager
from typing import List, Optional

from neo4j import GraphDatabase, basic_auth  # type: ignore – external wheel

logger = logging.getLogger("ui_graph")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s – %(message)s"))
logger.addHandler(handler)

# ---------------------------------------------------------------------------
# Helper – get env or default
# ---------------------------------------------------------------------------

def _env(key: str, default: str | None = None) -> str | None:
    return os.getenv(key, default)

# ---------------------------------------------------------------------------
# UIGraph wrapper
# ---------------------------------------------------------------------------

class UIGraph(AbstractContextManager):
    """Lightweight Neo4j wrapper for UI screen graph."""

    def __init__(
        self,
        uri: str | None = None,
        user: str | None = None,
        password: str | None = None,
        database: str | None = None,
    ) -> None:
        uri = uri or _env("NEO4J_URI", "bolt://10.74.139.250:7687")
        user = user or _env("NEO4J_USER", "neo4j")
        password = password or _env("NEO4J_PASS", "NEO4J_PASS=mango-metal-moral-bronze-prague-8964")
        database = database or _env("NEO4J_DB", "neo4j")

        self._driver = GraphDatabase.driver(uri, auth=basic_auth(user, password))
        self._db = database
        self._ensure_constraints()
        logger.info("Connected to %s (db=%s)", uri, database)

    # ------------------------------------------------------------------
    # Schema bootstrap
    # ------------------------------------------------------------------

    def _ensure_constraints(self):
        cypher = """
        CREATE CONSTRAINT IF NOT EXISTS FOR (s:Screen)
        REQUIRE s.hash IS UNIQUE
        """
        with self._driver.session(database=self._db) as sess:
            sess.run(cypher)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_transition(
        self,
        bf_hash: str,
        af_hash: str,
        cmd: str,
        meta: Optional[dict] = None,
    ) -> None:
        """MERGE two Screen nodes and the edge between them.

        ``meta`` may contain ``latency_ms`` and will be stored on the
        relationship the **first** time the edge is created.
        Subsequent calls increment a ``count`` property.
        """
        meta = meta or {}
        params = {
            "bf": bf_hash,
            "af": af_hash,
            "cmd": cmd.upper(),
            "ts": int(time.time() * 1000),
            "lat": int(meta.get("latency_ms", -1)),
        }

        cypher = """
        MERGE (a:Screen {hash: $bf})
          ON CREATE SET a.first_seen = $ts
          ON MATCH  SET a.last_seen  = $ts

        MERGE (b:Screen {hash: $af})
          ON CREATE SET b.first_seen = $ts
          ON MATCH  SET b.last_seen  = $ts

        MERGE (a)-[r:CMD {cmd: $cmd}]->(b)
          ON CREATE SET r.count = 1, r.first_ts = $ts, r.latency_ms = $lat
          ON MATCH  SET r.count = r.count + 1, r.last_ts = $ts
        """
        with self._driver.session(database=self._db) as sess:
            sess.run(cypher, params)

    # ------------------------------------------------------------------

    def state_exists(self, hash_str: str) -> bool:
        cypher = "MATCH (s:Screen {hash: $h}) RETURN s LIMIT 1"
        with self._driver.session(database=self._db) as sess:
            rec = sess.run(cypher, h=hash_str).single()
            return rec is not None

    # ------------------------------------------------------------------

    def shortest_path(
        self,
        src_hash: str,
        dst_hash: str,
        max_len: int = 10,
    ) -> List[str]:
        """Return command list for the shortest path (BF→…→AF).

        Returns a list of ``cmd`` strings, or empty list if no path.
        """
        cypher = f"""
        MATCH (src:Screen {{hash: $src}}), (dst:Screen {{hash: $dst}}),
        p = shortestPath( (src)-[:CMD*..{max_len}]->(dst) )
        WITH p
        UNWIND relationships(p) AS rel
        RETURN rel.cmd AS cmd
        """
        with self._driver.session(database=self._db) as sess:
            result = sess.run(cypher, src=src_hash, dst=dst_hash)
            cmds = [record["cmd"] for record in result]
        return cmds

    # ------------------------------------------------------------------

    def close(self):
        self._driver.close()

    __exit__ = lambda self, exc_type, exc, tb: self.close()
    __enter__ = lambda self: self

# ---------------------------------------------------------------------------
# Quick CLI self‑test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse, random, hashlib

    def rand_hash() -> str:
        return hashlib.sha1(str(random.random()).encode()).hexdigest()

    ap = argparse.ArgumentParser(description="UIGraph smoke test")
    ap.add_argument("cmd", nargs="?", default="RIGHT")
    args = ap.parse_args()

    with UIGraph() as g:
        h1, h2 = rand_hash(), rand_hash()
        g.add_transition(h1, h2, args.cmd, {"latency_ms": 120})
        print("exists?", g.state_exists(h1))
        path = g.shortest_path(h1, h2)
        print("shortest path", path)
