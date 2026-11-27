"""Client for the SweMetrics DuckDB HTTP endpoint (https://swemetrics.se/duckdb/).

The endpoint takes a SQL string as the POST body and answers JSON {data, meta, rows, statistics}. It
serves several databases in one namespace: `oa` (a full OpenAlex mirror), `swepub` (the Swedish
national publication index), `gdp`, `openaire`, `snd`, ...

Authentication: the key is read from the environment variable SWEMETRICS_API_KEY, loaded from the
repo's .env (git-ignored) like the other pipeline credentials. It is sent only as the X-API-Key header
and is never printed or put in an exception message.

Things worth knowing (measured, see docs/METHODOLOGY.md):
  * The big `oa` tables (works_authorships 1.3 billion rows, works_topics, ...) take minutes to scan.
    Never join a long literal ID list against them; query by a selective filter, or compare against
    a local file and use only the small `swepub` tables for ID lists.
  * `snapshot` is a reserved word; do not use it as an alias.
  * The source data has bad dates (a publication_date of 2050, a swepub year of 2301): filter years.

    from swemetrics import query
    rows = query("select count(*) as n from swepub.works")      # [{'n': 1868887}]
"""

import os
import sys
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

URL = os.environ.get("SWEMETRICS_URL", "https://swemetrics.se/duckdb/")
NORDIC_BALTIC = ("SE", "NO", "DK", "IS", "FI", "EE", "LV", "LT")


class SweMetricsError(RuntimeError):
    pass


def _key() -> str:
    key = os.environ.get("SWEMETRICS_API_KEY")
    if not key:
        raise SweMetricsError("SWEMETRICS_API_KEY is not set (add it to open-alice-data/.env)")
    return key


def query_raw(sql: str, timeout: int = 600, retries: int = 3) -> dict:
    """POST one SQL statement and return the decoded JSON. Retries timeouts and 5xx with a back-off."""
    last = ""
    for attempt in range(retries + 1):
        try:
            r = requests.post(
                URL, data=sql.encode("utf-8"), headers={"X-API-Key": _key(), "Content-Type": "text/plain"}, timeout=timeout
            )
        except requests.RequestException as e:
            last = type(e).__name__
        else:
            if r.status_code == 200:
                return r.json()
            last = f"HTTP {r.status_code}: {r.text[:400]}"
            if r.status_code < 500 or "Parser Error" in r.text or "Binder Error" in r.text:
                break  # our SQL is wrong: retrying will not help
        if attempt < retries:
            time.sleep(5 * (attempt + 1))
    raise SweMetricsError(last)


def query(sql: str, timeout: int = 600, retries: int = 3) -> list[dict]:
    """Run SQL and return the rows as a list of {column: value} dicts."""
    j = query_raw(sql, timeout, retries)
    cols = [m["name"] for m in j["meta"]]
    return [dict(zip(cols, row)) for row in j["data"]]


def in_list(values) -> str:
    """A SQL list literal of integers or strings: `[1, 2, 3]` / `['a', 'b']`."""
    out = []
    for v in values:
        out.append(str(int(v)) if isinstance(v, int) else "'" + str(v).replace("'", "''") + "'")
    return "[" + ",".join(out) + "]"


def chunks(seq, n):
    seq = list(seq)
    for i in range(0, len(seq), n):
        yield seq[i : i + n]


if __name__ == "__main__":  # smoke test: python swemetrics.py "select 1"
    sql = sys.argv[1] if len(sys.argv) > 1 else "select 1 as ok"
    print(query(sql))
