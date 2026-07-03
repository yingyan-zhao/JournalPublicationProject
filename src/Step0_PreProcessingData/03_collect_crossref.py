import requests
from pathlib import Path
import csv
import json
from datetime import date
from typing import Any, Iterable
from time import sleep

import os

os.chdir("/Users/yingyan_zhao/Dropbox/JournalPublicationProject")

from top_journals import TOP_FIVE_JOURNALS

CROSSREF_WORKS_URL = "https://api.crossref.org/journals/{issn}/works"
OUTPUT_JSONL = Path("data/raw/Crossref_FullData_Jsonl.jsonl")
OUTPUT_CSV = Path("data/processed/Crossref_FullData_CSV.csv")
FROM_YEAR = 1950
TO_YEAR = 2026
CROSSREF_ROWS_PER_PAGE = 1000


def main() -> None:
    collection_date = date.today().isoformat()

    works = collect_works_by_year(
        from_year=FROM_YEAR,
        to_year=TO_YEAR,
        rows=CROSSREF_ROWS_PER_PAGE,
    )

    write_jsonl(works, OUTPUT_JSONL, collection_date)
    print(f"Wrote {len(works)} raw Crossref records to {OUTPUT_JSONL}")
    print_jsonl_variables(OUTPUT_JSONL)

def write_jsonl(works: list[dict[str, Any]], output_jsonl: Path, collection_date: str) -> None:
    """Write raw Crossref works to a JSON Lines file."""
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    with output_jsonl.open("w", encoding="utf-8") as file:
        for work in works:
            work_with_date = work.copy()
            work_with_date["collection_date"] = collection_date
            file.write(json.dumps(work_with_date, ensure_ascii=False) + "\n")


def request_works(journal: str, issn: str, from_year: int, to_year: int, rows: int) -> list[dict[str, Any]]:
    """Ask Crossref for journal articles from one journal ISSN."""
    cursor = "*"
    works = []

    filters = [
        "type:journal-article",
        f"from-pub-date:{from_year}-01-01",
        f"until-pub-date:{to_year}-12-31",
    ]

    while cursor:
        params = {
            "filter": ",".join(filters),
            "cursor": cursor,
            "rows": rows,
        }

        response = requests.get(CROSSREF_WORKS_URL.format(issn=issn), params=params, timeout=60)
        response.raise_for_status()
        payload = response.json()

        items = payload.get("message", {}).get("items", [])
        for work in items:
            work["_query_journal"] = journal
            work["_query_issn"] = issn
            works.append(work)

        next_cursor = payload.get("message", {}).get("next-cursor")
        if not next_cursor or next_cursor == cursor or not items:
            break

        cursor = next_cursor
        sleep(0.15)

    return works


def collect_works(from_year: int, to_year: int, rows: int) -> list[dict[str, Any]]:
    """Collect Crossref records for all top-five journal ISSNs and remove duplicates."""
    seen_dois: set[str] = set()
    works: list[dict[str, Any]] = []

    for journal, info in TOP_FIVE_JOURNALS.items():
        for issn in info["issns"]:
            for work in request_works(journal, issn, from_year, to_year, rows):
                doi = normalize_doi(work.get("DOI", ""))
                if doi and doi not in seen_dois:
                    works.append(work)
                    seen_dois.add(doi)

    return works


def collect_works_by_year(from_year: int, to_year: int, rows: int) -> list[dict[str, Any]]:
    """Collect Crossref records year-by-year and deduplicate across all years."""
    seen_dois: set[str] = set()
    all_works: list[dict[str, Any]] = []

    for year in range(from_year, to_year + 1):
        yearly_works = collect_works(
            from_year=year,
            to_year=year,
            rows=rows,
        )
        print(f"Collected {len(yearly_works)} unique Crossref records for {year}.")

        for work in yearly_works:
            doi = normalize_doi(work.get("DOI", ""))
            if doi and doi not in seen_dois:
                all_works.append(work)
                seen_dois.add(doi)

    return all_works

#
def normalize_doi(doi: str) -> str:
    """Normalize DOI text for deduplication."""
    return (
        doi.lower()
        .replace("https://doi.org/", "")
        .replace("http://dx.doi.org/", "")
        .strip()
    )


def print_jsonl_variables(path: Path) -> None:
    variables = set()
    record_count = 0

    with path.open(encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            record = json.loads(line)
            variables.update(record.keys())
            record_count += 1

    print(f"Variables in {path} ({record_count} records):")
    for variable in sorted(variables):
        print(variable)


if __name__ == "__main__":
    main()
