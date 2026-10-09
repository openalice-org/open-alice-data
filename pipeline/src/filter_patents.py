"""Patent location filter: SE inventor OR KTH applicant."""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Any

from src.epo_client import _clean_space, _flatten_text

_COUNTRY_CODE_RE = re.compile(r"\[([A-Z]{2})\]$")


def _extract_inventor_countries(biblio: dict[str, Any]) -> list[str]:
    countries: list[str] = []
    bdata = biblio.get("bibliographic-data", {})
    parties = bdata.get("parties", {})
    inventors = parties.get("inventors", {})
    inv_list = inventors.get("inventor", [])
    if isinstance(inv_list, dict):
        inv_list = [inv_list]
    for inv in inv_list:
        country = ""
        # Try address block first
        address = inv.get("address", {})
        if isinstance(address, list):
            address = address[0] if address else {}
        if address:
            country_node = address.get("country", {})
            if isinstance(country_node, dict):
                country = _flatten_text(country_node.get("$", ""))
            else:
                country = _flatten_text(country_node)
        # Fallback: extract [XX] suffix from epodoc inventor name
        if not country:
            name_node = inv.get("inventor-name", {})
            if isinstance(name_node, list):
                name_node = name_node[0] if name_node else {}
            name = _flatten_text(name_node.get("name", {}) if isinstance(name_node, dict) else name_node)
            m = _COUNTRY_CODE_RE.search(name)
            if m:
                country = m.group(1)
        if country:
            countries.append(country.upper())
    return countries


def _extract_applicant_names(biblio: dict[str, Any]) -> list[str]:
    names: list[str] = []
    bdata = biblio.get("bibliographic-data", {})
    parties = bdata.get("parties", {})
    applicants = parties.get("applicants", {})
    app_list = applicants.get("applicant", [])
    if isinstance(app_list, dict):
        app_list = [app_list]
    for app in app_list:
        app_name = app.get("applicant-name", {})
        if isinstance(app_name, list):
            app_name = app_name[0] if app_name else {}
        name = _flatten_text(app_name.get("name", {}) if isinstance(app_name, dict) else app_name)
        if name:
            names.append(name)
    return names


def filter_patents(
    patents: list[dict[str, Any]],
    biblio_map: dict[str, dict[str, Any]],
    outputs_dir: Path,
    applicant_regex: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Apply SE-inventor or KTH-applicant filter.

    Writes:
      outputs/filtered_patents.csv
      outputs/rejected_patents.csv

    Returns (filtered, rejected).
    """
    uni_re = re.compile(applicant_regex, re.I)
    filtered: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []

    for patent in patents:
        epo_id = patent.get("epo_id", "")
        biblio = biblio_map.get(epo_id, {})
        countries = _extract_inventor_countries(biblio)
        applicants = _extract_applicant_names(biblio)

        se_inventor = any(c == "SE" for c in countries)
        kth_applicant = any(uni_re.search(name) for name in applicants)

        record = {
            "epo_id": epo_id,
            "doc_number": patent.get("doc_number", ""),
            "jurisdiction": patent.get("jurisdiction", ""),
            "year_published": patent.get("year_published", ""),
            "title": patent.get("title", []),
            "se_inventor": se_inventor,
            "kth_applicant": kth_applicant,
            "inventor_countries": "|".join(countries),
            "applicant_names": "|".join(applicants),
        }

        if se_inventor or kth_applicant:
            filtered.append(record)
        else:
            rejected.append(record)

    outputs_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "epo_id",
        "doc_number",
        "jurisdiction",
        "year_published",
        "title",
        "se_inventor",
        "kth_applicant",
        "inventor_countries",
        "applicant_names",
    ]

    def _write(path: Path, rows: list[dict[str, Any]]) -> None:
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                row_copy = dict(row)
                title = row_copy.get("title", [])
                if isinstance(title, list):
                    row_copy["title"] = " | ".join(
                        _clean_space(str(t.get("text", t))) for t in title
                    )
                writer.writerow(row_copy)

    _write(outputs_dir / "filtered_patents.csv", filtered)
    _write(outputs_dir / "rejected_patents.csv", rejected)

    return filtered, rejected


if __name__ == "__main__":
    # Inline assert with synthetic biblio records
    synthetic_se = {
        "bibliographic-data": {
            "parties": {
                "inventors": {
                    "inventor": [
                        {
                            "address": {"country": {"$": "SE"}},
                            "inventor-name": {"name": {"$": "Alice Example"}},
                        }
                    ]
                },
                "applicants": {
                    "applicant": [
                        {
                            "applicant-name": {"name": {"$": "Some Corp"}},
                        }
                    ]
                },
            }
        }
    }
    synthetic_kth = {
        "bibliographic-data": {
            "parties": {
                "inventors": {
                    "inventor": [
                        {
                            "address": {"country": {"$": "US"}},
                            "inventor-name": {"name": {"$": "Bob Example"}},
                        }
                    ]
                },
                "applicants": {
                    "applicant": [
                        {
                            "applicant-name": {"name": {"$": "KTH Royal Institute"}},
                        }
                    ]
                },
            }
        }
    }
    synthetic_reject = {
        "bibliographic-data": {
            "parties": {
                "inventors": {
                    "inventor": [
                        {
                            "address": {"country": {"$": "US"}},
                            "inventor-name": {"name": {"$": "Charlie Example"}},
                        }
                    ]
                },
                "applicants": {
                    "applicant": [
                        {
                            "applicant-name": {"name": {"$": "Acme Inc"}},
                        }
                    ]
                },
            }
        }
    }

    _test_re = re.compile(r"KTH|KUNGLIGA TEKNISKA|ROYAL INSTITUTE OF TECHNOLOGY", re.I)

    assert _extract_inventor_countries(synthetic_se) == ["SE"]
    assert _extract_applicant_names(synthetic_kth) == ["KTH Royal Institute"]
    assert not any(_test_re.search(n) for n in _extract_applicant_names(synthetic_reject))

    # Test full filter with synthetic patents
    import tempfile

    patents = [
        {"epo_id": "SE123", "doc_number": "123", "jurisdiction": "SE", "year_published": 2020, "title": []},
        {"epo_id": "US456", "doc_number": "456", "jurisdiction": "US", "year_published": 2021, "title": []},
        {"epo_id": "WO789", "doc_number": "789", "jurisdiction": "WO", "year_published": 2022, "title": []},
    ]
    biblio_map = {
        "SE123": synthetic_se,
        "US456": synthetic_kth,
        "WO789": synthetic_reject,
    }

    with tempfile.TemporaryDirectory() as td:
        out = Path(td)
        filtered, rejected = filter_patents(
            patents, biblio_map, out,
            applicant_regex=r"KTH|KUNGLIGA TEKNISKA|ROYAL INSTITUTE OF TECHNOLOGY",
        )
        assert len(filtered) == 2, f"Expected 2 filtered, got {len(filtered)}"
        assert len(rejected) == 1, f"Expected 1 rejected, got {len(rejected)}"
        assert (out / "filtered_patents.csv").exists()
        assert (out / "rejected_patents.csv").exists()
        print("filter_patents inline assertions passed")
