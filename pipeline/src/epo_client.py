"""EPO OPS API client with auth, search, abstract, claims, and biblio endpoints."""

from __future__ import annotations

import base64
import random
import re
import time
from typing import Any

import requests

EPO_AUTH_URL = "https://ops.epo.org/3.2/auth/accesstoken"
EPO_SEARCH_URL = "https://ops.epo.org/3.2/rest-services/published-data/search"
EPO_PUB_URL = "https://ops.epo.org/3.2/rest-services/published-data/publication/epodoc"


class EPORequestError(RuntimeError):
    """Raised when a retriable OPS request exhausts its retry budget."""


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value) if value not in (None, "") else default
    except (TypeError, ValueError):
        return default


def _clean_space(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _flatten_text(value: Any) -> str:
    chunks: list[str] = []
    if isinstance(value, str):
        chunks.append(value)
    elif isinstance(value, dict):
        found = False
        for key in ("text", "$", "plain_text", "value"):
            if isinstance(value.get(key), str):
                chunks.append(value[key])
                found = True
                break
        if not found:
            for k, child in value.items():
                if k.startswith("@"):
                    continue
                chunks.append(_flatten_text(child))
    elif isinstance(value, list):
        for child in value:
            chunks.append(_flatten_text(child))
    return " ".join(c for c in chunks if c)


def _preferred_lang_text(value: Any, lang: str = "en") -> str:
    pairs: list[tuple[str, str]] = []

    def collect(node: Any, inherited_lang: str = "") -> None:
        if isinstance(node, dict):
            node_lang = str(node.get("@lang") or node.get("lang") or inherited_lang).lower()
            text = node.get("text") or node.get("$") or node.get("plain_text") or node.get("value")
            if isinstance(text, str):
                pairs.append((node_lang, text))
            for k, child in node.items():
                if k.startswith("@") or k in {"text", "$", "plain_text", "value", "lang"}:
                    continue
                collect(child, node_lang)
        elif isinstance(node, list):
            for child in node:
                collect(child, inherited_lang)
        elif isinstance(node, str):
            pairs.append((inherited_lang, node))

    collect(value)
    chosen = [t for l, t in pairs if l == lang]
    if not chosen:
        chosen = [t for _, t in pairs if re.search(r"[A-Za-z]", t)]
    if not chosen:
        chosen = [t for _, t in pairs]
    deduped: list[str] = []
    for item in chosen:
        item = _clean_space(item)
        if item and item not in deduped:
            deduped.append(item)
    return " ".join(deduped)


def _nested_values(value: Any, keys: set[str]) -> list[Any]:
    found: list[Any] = []
    if isinstance(value, dict):
        for k, child in value.items():
            if k in keys:
                found.append(child)
            found.extend(_nested_values(child, keys))
    elif isinstance(value, list):
        for child in value:
            found.extend(_nested_values(child, keys))
    return found


class EPOClient:
    def __init__(
        self,
        consumer_key: str,
        consumer_secret: str,
        delay: float = 0.5,
        max_retries: int = 6,
        backoff_base: float = 2.0,
        backoff_cap: float = 120.0,
        jitter_max: float = 1.25,
        search_delay: float = 2.1,
    ):
        self.consumer_key = consumer_key
        self.consumer_secret = consumer_secret
        self.delay = delay
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.backoff_cap = backoff_cap
        self.jitter_max = jitter_max
        self.search_delay = search_delay
        self.session = requests.Session()
        self._token: str = ""
        self._token_expiry: float = 0.0
        self._last_search_at: float = 0.0

    def _get_token(self) -> str:
        if self._token and time.time() < self._token_expiry - 60:
            return self._token
        credentials = base64.b64encode(
            f"{self.consumer_key}:{self.consumer_secret}".encode()
        ).decode()
        r = self.session.post(
            EPO_AUTH_URL,
            headers={
                "Authorization": f"Basic {credentials}",
                "Content-Type": "application/x-www-form-urlencoded",
            },
            data={"grant_type": "client_credentials"},
            timeout=20,
        )
        r.raise_for_status()
        data = r.json()
        self._token = data["access_token"]
        self._token_expiry = time.time() + int(data.get("expires_in", 1200))
        return self._token

    def _headers(self, range_: str = "1-10") -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._get_token()}",
            "Accept": "application/json",
            "X-OPS-Range": range_,
        }

    def _retry_wait(self, response: requests.Response | None, attempt: int) -> float:
        retry_after = ""
        if response is not None:
            retry_after = str(response.headers.get("Retry-After", "")).strip()
        if retry_after.isdigit():
            wait = float(retry_after)
        else:
            wait = self.backoff_base * (2 ** attempt)
        wait += self.delay + random.uniform(0, self.jitter_max)
        return min(self.backoff_cap, max(self.delay, wait))

    def _get(
        self,
        url: str,
        params: dict | None = None,
        retries: int | None = None,
        range_: str = "1-10",
    ) -> dict[str, Any]:
        retries = self.max_retries if retries is None else retries
        for attempt in range(retries):
            try:
                r = self.session.get(url, params=params, headers=self._headers(range_), timeout=30)
                if r.status_code == 404:
                    return {}
                if r.status_code in {403, 429}:
                    self._token = ""
                    wait = self._retry_wait(r, attempt)
                    print(
                        f"  EPO rate limit ({r.status_code}) — retry {attempt + 1}/{retries} "
                        f"after {wait:.1f}s",
                        flush=True,
                    )
                    time.sleep(wait)
                    continue
                r.raise_for_status()
                time.sleep(self.delay)
                return r.json()
            except requests.HTTPError as exc:
                print(f"  EPO HTTP error (attempt {attempt + 1}): {exc}", flush=True)
                time.sleep(min(self.backoff_cap, 1 + attempt))
            except requests.RequestException as exc:
                wait = self._retry_wait(None, attempt)
                print(
                    f"  EPO request error (attempt {attempt + 1}/{retries}): {str(exc)[:160]} "
                    f"— sleeping {wait:.1f}s",
                    flush=True,
                )
                time.sleep(wait)
        raise EPORequestError(f"EPO request exhausted {retries} retries: {url}")

    def search_by_inventor(self, name: str, since_year: int, max_results: int = 25) -> tuple[int, list[dict[str, str]]]:
        """CQL: IN="LASTNAME FIRSTNAME". Name must be pre-reversed (lastname first). Returns (total, refs)."""
        clean = name.strip()
        cql = f'IN="{clean}"'

        # OPS exposes a much tighter search bucket (typically 30 req/min) than
        # its retrieval bucket. The general 0.5s request delay is safe for
        # biblio/text retrieval but would send 120 searches/minute for authors
        # with no patents, causing repeated 403 + Retry-After=120 responses.
        elapsed = time.monotonic() - self._last_search_at
        if self._last_search_at and elapsed < self.search_delay:
            time.sleep(self.search_delay - elapsed)
        self._last_search_at = time.monotonic()

        n = min(max_results, 25)
        range_str = f"1-{n}"
        data = self._get(EPO_SEARCH_URL, params={"q": cql, "Range": range_str, "Output": "biblio"}, range_=range_str)
        ops = data.get("ops:world-patent-data", {})
        results = ops.get("ops:biblio-search", {})
        total = _as_int(results.get("@total-result-count"))
        refs_raw = results.get("ops:search-result", {}).get("ops:publication-reference", [])
        if isinstance(refs_raw, dict):
            refs_raw = [refs_raw]

        refs: list[dict[str, str]] = []
        for ref in refs_raw:
            doc_ids = ref.get("document-id", {})
            if isinstance(doc_ids, list):
                doc = next((d for d in doc_ids if d.get("@document-id-type") == "docdb"), doc_ids[0] if doc_ids else {})
            else:
                doc = doc_ids
            refs.append({
                "country": _flatten_text(doc.get("country", "")).upper(),
                "doc_number": _flatten_text(doc.get("doc-number", "")),
                "kind": _flatten_text(doc.get("kind", "")),
                "date": _flatten_text(doc.get("date", "")),
            })
        return total, refs

    def fetch_abstract(self, country: str, doc_number: str) -> str:
        """Fetch abstract text from EPO full-text API."""
        ref = f"{country}{doc_number}"
        data = self._get(f"{EPO_PUB_URL}/{ref}/abstract")
        if not data:
            return ""
        docs = data.get("ops:world-patent-data", {}).get("exchange-documents", {})
        exchange = docs.get("exchange-document", {})
        if isinstance(exchange, list):
            exchange = exchange[0] if exchange else {}
        abstract = exchange.get("abstract", {})
        return _preferred_lang_text(abstract).strip()

    def fetch_claims(self, country: str, doc_number: str) -> str:
        """Fetch claims text from EPO full-text API."""
        ref = f"{country}{doc_number}"
        data = self._get(f"{EPO_PUB_URL}/{ref}/claims")
        if not data:
            return ""
        docs = data.get("ops:world-patent-data", {}).get("exchange-documents", {})
        exchange = docs.get("exchange-document", {})
        if isinstance(exchange, list):
            exchange = exchange[0] if exchange else {}
        claims = exchange.get("claims", {})
        return _clean_space(_flatten_text(claims))[:5000]

    def fetch_biblio(self, country: str, doc_number: str) -> dict[str, Any]:
        """Fetch biblio (title, inventors, applicants, date)."""
        ref = f"{country}{doc_number}"
        data = self._get(f"{EPO_PUB_URL}/{ref}/biblio")
        if not data:
            return {}
        docs = data.get("ops:world-patent-data", {}).get("exchange-documents", {})
        exchange = docs.get("exchange-document", {})
        if isinstance(exchange, list):
            exchange = exchange[0] if exchange else {}
        return exchange


if __name__ == "__main__":
    # Verify EPOClient can be instantiated (won't test live API without creds)
    import os

    key = os.getenv("EPO_CONSUMER_KEY", "")
    secret = os.getenv("EPO_CONSUMER_SECRET_KEY", "")
    if key and secret:
        client = EPOClient(key, secret)
        print("epo_client smoke test passed: EPOClient instantiated")
    else:
        print("No EPO creds in env; skipping smoke test")
