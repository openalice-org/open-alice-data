"""Shared OpenAlex helpers with strict, auditable retrieval guarantees."""

import datetime as dt
import hashlib
import json
import os
import time
import urllib.parse
import urllib.request

BASE_URL = "https://api.openalex.org/works"
_MAILTO = os.environ.get("OPENALEX_MAILTO", "")
_USER_AGENT = "kth-quantum-pipeline/1.0" + (f" (mailto:{_MAILTO})" if _MAILTO else "")


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_page(url, retries=3):
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=90) as resp:
                return json.loads(resp.read())
        except Exception as exc:
            if attempt == retries - 1:
                raise
            print(f"\n  retrying after error: {exc}")
            time.sleep(2**attempt)


def build_works_url(api_key="", mailto="", **params):
    """Build an OpenAlex works URL, omitting unset parameters.

    An explicit ``api_key`` takes priority; ``mailto`` (OpenAlex's polite
    pool) is only attached when no api_key is supplied.
    """
    clean_params = {
        key: str(value)
        for key, value in params.items()
        if value is not None and value != ""
    }
    if api_key:
        clean_params["api_key"] = api_key
    elif mailto:
        clean_params["mailto"] = mailto
    return BASE_URL + "?" + urllib.parse.urlencode(clean_params)


def join_filters(*parts):
    return ",".join(part for part in parts if part)


def _validated_count(data, context):
    meta = data.get("meta") or {}
    count = meta.get("count")
    if not isinstance(count, int) or count < 0:
        raise RuntimeError(f"OpenAlex returned no valid meta.count for {context}")
    return count


def work_count(
    filter_str,
    api_key,
    search=None,
    search_parameter="search",
    fetch_json=fetch_page,
    mailto="",
):
    """Return a verified count without downloading the result set."""
    params = {"filter": filter_str, "per_page": 1, "select": "id"}
    if search:
        params[search_parameter] = search
    data = fetch_json(build_works_url(api_key, mailto=mailto, **params))
    if not isinstance(data.get("results"), list):
        raise RuntimeError("OpenAlex count response has no results list")
    return _validated_count(data, filter_str or search or "unfiltered works")


def grouped_work_counts(
    filter_str,
    group_by,
    api_key,
    search=None,
    search_parameter="search",
    fetch_json=fetch_page,
    mailto="",
):
    """Return all aggregation buckets for a low-cardinality Work field."""
    params = {"filter": filter_str, "group_by": group_by, "per_page": 100}
    if search:
        params[search_parameter] = search
    data = fetch_json(build_works_url(api_key, mailto=mailto, **params))
    _validated_count(data, f"group_by={group_by}")
    groups = data.get("group_by")
    if not isinstance(groups, list):
        raise RuntimeError(f"OpenAlex response has no group_by list for {group_by}")
    result = {}
    for group in groups:
        key = group.get("key")
        count = group.get("count")
        if key in result:
            raise RuntimeError(f"OpenAlex repeated group key {key!r} for {group_by}")
        if not isinstance(count, int) or count < 0:
            raise RuntimeError(
                f"OpenAlex returned an invalid group count for {group_by}"
            )
        result[str(key)] = count
    return result


def sample_works(
    filter_str,
    select,
    api_key,
    sample_size,
    seed,
    expected_population,
    search=None,
    search_parameter="search",
    fetch_json=fetch_page,
    mailto="",
):
    """Fetch a reproducible sample and fail on malformed or duplicate records."""
    target = min(sample_size, expected_population)
    if target == 0:
        return []
    params = {
        "filter": filter_str,
        "select": select,
        "sample": target,
        "per_page": target,
        "seed": seed,
    }
    if search:
        params[search_parameter] = search
    data = fetch_json(build_works_url(api_key, mailto=mailto, **params))
    results = data.get("results")
    if not isinstance(results, list):
        raise RuntimeError("OpenAlex sample response has no results list")
    if len(results) != target:
        raise RuntimeError(
            f"Incomplete OpenAlex sample: requested {target}, received {len(results)}"
        )
    ids = [work.get("id") for work in results]
    if any(not work_id for work_id in ids):
        raise RuntimeError("OpenAlex sample contains a work without an ID")
    if len(ids) != len(set(ids)):
        raise RuntimeError("OpenAlex sample contains duplicate work IDs")
    return results


def paginate(
    filter_str,
    select,
    api_key,
    sort=None,
    max_pages=None,
    label="",
    search=None,
    search_parameter="search",
    fetch_json=fetch_page,
    mailto="",
):
    """Strictly cursor-paginate works and yield each unique record.

    The iterator validates ``meta.count``, repeated cursors, empty intermediate
    pages, work IDs, duplicate work IDs, final counts, and page limits. A safety
    limit is therefore a hard failure rather than a warning that can silently
    produce a truncated corpus.
    """
    cursor = "*"
    seen_cursors = set()
    seen_work_ids = set()
    page = 0
    total = 0
    expected_count = None
    prefix = f"[{label}] " if label else ""

    while cursor:
        if max_pages is not None and page >= max_pages:
            raise RuntimeError(
                f"{prefix}hit max_pages={max_pages} before OpenAlex pagination completed"
            )
        if cursor in seen_cursors:
            raise RuntimeError(f"OpenAlex repeated cursor {cursor!r}")
        seen_cursors.add(cursor)

        params = {
            "filter": filter_str,
            "select": select,
            "per_page": 100,
            "cursor": cursor,
        }
        if sort:
            params["sort"] = sort
        if search:
            params[search_parameter] = search
        data = fetch_json(build_works_url(api_key, mailto=mailto, **params))
        results = data.get("results")
        meta = data.get("meta") or {}
        if not isinstance(results, list):
            raise RuntimeError("OpenAlex works response has no results list")
        response_count = _validated_count(data, filter_str or search or "works")
        if expected_count is None:
            expected_count = response_count
        elif response_count != expected_count:
            raise RuntimeError(
                f"OpenAlex meta.count changed during pagination: "
                f"{expected_count} to {response_count}"
            )

        page += 1
        for work in results:
            work_id = work.get("id")
            if not work_id:
                raise RuntimeError("OpenAlex returned a work without an ID")
            if work_id in seen_work_ids:
                raise RuntimeError(f"OpenAlex returned duplicate work ID {work_id}")
            seen_work_ids.add(work_id)
            total += 1
            yield work

        next_cursor = meta.get("next_cursor")
        if next_cursor and not results:
            raise RuntimeError(
                "OpenAlex returned an empty works page with a next cursor"
            )
        cursor = next_cursor
        print(f"  {prefix}page {page} — {total} works", end="\r", flush=True)
        if results and cursor:
            time.sleep(0.1)

    print()
    if expected_count is None:
        raise RuntimeError("OpenAlex pagination returned no response")
    if total != expected_count:
        raise RuntimeError(
            f"Incomplete OpenAlex retrieval: downloaded {total}, "
            f"OpenAlex reported {expected_count}"
        )


def _read_valid_part(
    part_path, part_manifest_path, query_filter, expected_count, select
):
    try:
        with open(part_manifest_path, encoding="utf-8") as f:
            manifest = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    try:
        actual_hash = file_sha256(part_path)
    except OSError:
        return None
    if (
        manifest.get("filter") != query_filter
        or manifest.get("expected_count") != expected_count
        or manifest.get("select") != select
        or manifest.get("output_sha256") != actual_hash
    ):
        return None
    return manifest


def write_year_batched_work_snapshot(
    base_filter,
    years,
    select,
    api_key,
    output,
    manifest_path,
    batch_dir,
    config_paths=(),
    query_metadata=None,
    max_pages=None,
    fetch_json=fetch_page,
    mailto="",
):
    """Download a complete work snapshot as resumable, verified year batches."""
    years = sorted(set(years))
    if not years or any(not isinstance(year, int) for year in years):
        raise ValueError("years must contain at least one integer year")

    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(manifest_path) or ".", exist_ok=True)
    os.makedirs(batch_dir, exist_ok=True)
    batch_manifests = []

    for year in years:
        query_filter = join_filters(base_filter, f"publication_year:{year}")
        expected_count = work_count(
            query_filter, api_key, fetch_json=fetch_json, mailto=mailto
        )
        part_path = os.path.join(batch_dir, f"{year}.jsonl")
        part_manifest_path = os.path.join(batch_dir, f"{year}.manifest.json")
        saved = None
        if os.path.exists(part_path):
            saved = _read_valid_part(
                part_path, part_manifest_path, query_filter, expected_count, select
            )
        if saved:
            print(f"  {year}: reusing verified {expected_count}-work batch")
            batch_manifests.append(saved)
            continue

        part_tmp = part_path + ".tmp"
        part_manifest_tmp = part_manifest_path + ".tmp"
        try:
            downloaded = 0
            with open(part_tmp, "w", encoding="utf-8") as out:
                for work in paginate(
                    query_filter,
                    select,
                    api_key,
                    max_pages=max_pages,
                    label=str(year),
                    fetch_json=fetch_json,
                    mailto=mailto,
                ):
                    out.write(
                        json.dumps(work, ensure_ascii=False, sort_keys=True) + "\n"
                    )
                    downloaded += 1
            if downloaded != expected_count:
                raise RuntimeError(
                    f"Year {year}: counted {expected_count} before download but retrieved "
                    f"{downloaded}"
                )
            part_manifest = {
                "schema_version": 1,
                "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                "year": year,
                "filter": query_filter,
                "select": select,
                "expected_count": expected_count,
                "retrieved_count": downloaded,
                "output": part_path,
                "output_sha256": file_sha256(part_tmp),
            }
            with open(part_manifest_tmp, "w", encoding="utf-8") as f:
                json.dump(
                    part_manifest, f, ensure_ascii=False, indent=2, sort_keys=True
                )
                f.write("\n")
            os.replace(part_tmp, part_path)
            os.replace(part_manifest_tmp, part_manifest_path)
            batch_manifests.append(part_manifest)
        except Exception:
            for path in (part_tmp, part_manifest_tmp):
                try:
                    os.remove(path)
                except FileNotFoundError:
                    pass
            raise

    output_tmp = output + ".tmp"
    manifest_tmp = manifest_path + ".tmp"
    total = 0
    seen_ids = set()
    try:
        with open(output_tmp, "w", encoding="utf-8") as out:
            for batch_manifest in batch_manifests:
                part_path = batch_manifest["output"]
                if file_sha256(part_path) != batch_manifest.get("output_sha256"):
                    raise RuntimeError(
                        f"Year batch hash changed before combine: {part_path}"
                    )
                with open(part_path, encoding="utf-8") as part:
                    for line_number, line in enumerate(part, start=1):
                        try:
                            work = json.loads(line)
                        except json.JSONDecodeError as exc:
                            raise RuntimeError(
                                f"Invalid JSON in {part_path} line {line_number}"
                            ) from exc
                        work_id = work.get("id")
                        if not work_id:
                            raise RuntimeError(f"Work without ID in {part_path}")
                        if work_id in seen_ids:
                            raise RuntimeError(
                                f"Duplicate work across year batches: {work_id}"
                            )
                        seen_ids.add(work_id)
                        out.write(
                            json.dumps(work, ensure_ascii=False, sort_keys=True) + "\n"
                        )
                        total += 1

        expected_total = sum(batch["expected_count"] for batch in batch_manifests)
        if total != expected_total:
            raise RuntimeError(
                f"Combined snapshot has {total} works; batches report {expected_total}"
            )
        manifest = {
            "schema_version": 1,
            "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "endpoint": BASE_URL,
            "query": {
                "base_filter": base_filter,
                "years": years,
                "select": select,
                **(query_metadata or {}),
            },
            "config_sha256": {path: file_sha256(path) for path in config_paths},
            "batch_dir": batch_dir,
            "batches": batch_manifests,
            "work_count": total,
            "output": output,
            "output_sha256": file_sha256(output_tmp),
        }
        with open(manifest_tmp, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2, sort_keys=True)
            f.write("\n")
        os.replace(output_tmp, output)
        os.replace(manifest_tmp, manifest_path)
        return manifest
    except Exception:
        for path in (output_tmp, manifest_tmp):
            try:
                os.remove(path)
            except FileNotFoundError:
                pass
        raise
