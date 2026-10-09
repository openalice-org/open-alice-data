"""Download a complete, reviewable snapshot of the OpenAlex topic taxonomy."""

import datetime as dt
import hashlib
import json
import os
import urllib.parse

from utils.openalex_utils import fetch_page
from utils.settings import settings

TOPICS_URL = "https://api.openalex.org/topics"
OUTPUT = "data/openalex_topics_snapshot.jsonl"
MANIFEST = "data/openalex_topics_snapshot_manifest.json"
PER_PAGE = 100
SELECT_FIELDS = (
    "id",
    "display_name",
    "keywords",
    "description",
    "domain",
    "field",
    "subfield",
    "works_count",
    "cited_by_count",
    "created_date",
    "updated_date",
)


def topic_page_url(cursor, api_key="", mailto=""):
    """Build a topics URL. An explicit api_key takes priority over mailto."""
    params = {
        "cursor": cursor,
        "per_page": str(PER_PAGE),
        "select": ",".join(SELECT_FIELDS),
    }
    if api_key:
        params["api_key"] = api_key
    elif mailto:
        params["mailto"] = mailto
    return TOPICS_URL + "?" + urllib.parse.urlencode(params)


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_topic_snapshot(
    api_key, output=OUTPUT, manifest=MANIFEST, fetch_json=fetch_page, mailto=""
):
    """Fetch every topic page and atomically replace the snapshot artifacts."""
    output_tmp = output + ".tmp"
    manifest_tmp = manifest + ".tmp"
    cursor = "*"
    seen_cursors = set()
    seen_topic_ids = set()
    expected_count = None
    topic_count = 0
    page_count = 0

    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(manifest) or ".", exist_ok=True)

    try:
        with open(output_tmp, "w", encoding="utf-8") as out:
            while cursor:
                if cursor in seen_cursors:
                    raise RuntimeError(f"OpenAlex repeated cursor {cursor!r}")
                seen_cursors.add(cursor)

                data = fetch_json(topic_page_url(cursor, api_key, mailto))
                results = data.get("results")
                meta = data.get("meta") or {}
                if not isinstance(results, list):
                    raise RuntimeError("OpenAlex topic response has no results list")

                if expected_count is None:
                    expected_count = meta.get("count")
                    if not isinstance(expected_count, int):
                        raise RuntimeError(
                            "OpenAlex topic response has no integer meta.count"
                        )

                page_count += 1
                for topic in results:
                    topic_id = topic.get("id")
                    if not topic_id:
                        raise RuntimeError("OpenAlex returned a topic without an ID")
                    if topic_id in seen_topic_ids:
                        raise RuntimeError(
                            f"OpenAlex returned duplicate topic ID {topic_id}"
                        )
                    seen_topic_ids.add(topic_id)
                    out.write(
                        json.dumps(topic, ensure_ascii=False, sort_keys=True) + "\n"
                    )
                    topic_count += 1

                next_cursor = meta.get("next_cursor")
                if next_cursor and not results:
                    raise RuntimeError(
                        "OpenAlex returned an empty page with a next cursor"
                    )
                cursor = next_cursor
                print(
                    f"  page {page_count} — {topic_count} topics", end="\r", flush=True
                )

        if topic_count != expected_count:
            raise RuntimeError(
                f"Incomplete topic snapshot: downloaded {topic_count}, "
                f"OpenAlex reported {expected_count}"
            )

        output_hash = file_sha256(output_tmp)
        created_at = dt.datetime.now(dt.timezone.utc).isoformat()
        manifest_data = {
            "schema_version": 1,
            "created_at": created_at,
            "endpoint": TOPICS_URL,
            "select_fields": list(SELECT_FIELDS),
            "page_count": page_count,
            "topic_count": topic_count,
            "api_reported_count": expected_count,
            "output": output,
            "output_sha256": output_hash,
            "taxonomy_snapshot_id": output_hash,
        }
        with open(manifest_tmp, "w", encoding="utf-8") as f:
            json.dump(manifest_data, f, ensure_ascii=False, indent=2, sort_keys=True)
            f.write("\n")

        os.replace(output_tmp, output)
        os.replace(manifest_tmp, manifest)
        print()
        return manifest_data
    except Exception:
        for path in (output_tmp, manifest_tmp):
            try:
                os.remove(path)
            except FileNotFoundError:
                pass
        raise


def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)) + "/..")
    api_key = settings.openalex_api_key
    mailto = settings.openalex_mailto
    if not api_key and not mailto:
        raise SystemExit(
            "Set OPENALEX_API_KEY or OPENALEX_MAILTO before downloading the topic "
            "taxonomy."
        )

    result = write_topic_snapshot(api_key, mailto=mailto)
    print(f"Saved {result['topic_count']} topics to {OUTPUT}")
    print(f"Snapshot manifest: {MANIFEST}")


if __name__ == "__main__":
    main()
