import requests
from pathlib import Path
import csv
import json
from datetime import date
from typing import Any, Iterable
from time import sleep
import os

os.chdir("/Users/yingyan_zhao/Dropbox/JournalPublicationProject")

OUTPUT_Jason = Path("data/raw/OpenAlex_FullData_Jsonl.jsonl")
OUTPUT_csv = Path("data/processed/OpenAlex_FullData_CSV.csv")
FROM_YEAR = 2000
TO_YEAR = 2026
OPENALEX_PER_PAGE = 50

from top_journals import TOP_FIVE_JOURNALS
print(TOP_FIVE_JOURNALS)

OPENALEX_WORKS_URL = "https://api.openalex.org/works"

def main() -> None:

    collection_date = date.today().isoformat()
    all_works = collect_works_by_year(
        from_year=FROM_YEAR,
        to_year=TO_YEAR,
        per_page=OPENALEX_PER_PAGE,
    )
    #
    write_jsonl(all_works, OUTPUT_Jason, collection_date)
    print(f"Wrote {len(all_works)} raw OpenAlex records to {OUTPUT_Jason}")



def write_jsonl(all_works: list[dict[str, Any]], output_jsonl: Path, collection_date: str) -> None:
    """Write raw OpenAlex works to a JSON Lines file."""
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    with output_jsonl.open("w", encoding="utf-8") as file:
        for work in all_works:
            work_with_date = work.copy()
            work_with_date["collection_date"] = collection_date
            file.write(json.dumps(work_with_date, ensure_ascii=False) + "\n")


def request_works(journal: str, issn: str, from_year: int, to_year: int, per_page: int) -> Iterable[dict[str, Any]]:

    cursor = "*"

    filters = [
        f"primary_location.source.issn:{issn}",
        "type:article",
        f"from_publication_date:{from_year}-01-01",
        f"to_publication_date:{to_year}-12-31",
    ]

    works = []
    while cursor:
        params = {
            "filter": ",".join(filters),
            "cursor": cursor,
            "per-page": per_page,
        }

        response = requests.get(OPENALEX_WORKS_URL, params=params, timeout=60)
        response.raise_for_status()
        payload = response.json()

        for work in payload.get("results", []):
            work["_query_journal"] = journal
            work["_query_issn"] = issn
            works.append(work)

        next_cursor = (payload.get("meta") or {}).get("next_cursor")
        if not next_cursor or next_cursor == cursor:
            break
        cursor = next_cursor
        sleep(0.15)

    return works

def collect_works(from_year: int, to_year: int, per_page: int) -> list[dict[str, Any]]:
    seen_ids: set[str] = set()
    works: list[dict[str, Any]] = []
    for journal, info in TOP_FIVE_JOURNALS.items():
        for issn in info["issns"]:
            for work in request_works(journal, issn, from_year, to_year, per_page):
                work_id = work.get("id")
                if work_id and work_id not in seen_ids:
                    works.append(work)
                    seen_ids.add(work_id)
    return works


def collect_works_by_year(from_year: int, to_year: int, per_page: int) -> list[dict[str, Any]]:
    """Collect OpenAlex records year-by-year and deduplicate across all years."""
    seen_ids: set[str] = set()
    all_works: list[dict[str, Any]] = []

    for year in range(from_year, to_year + 1):
        yearly_works = collect_works(
            from_year=year,
            to_year=year,
            per_page=per_page,
        )
        print(f"Collected {len(yearly_works)} unique OpenAlex records for {year}.")

        for work in yearly_works:
            work_id = work.get("id")
            if work_id and work_id not in seen_ids:
                all_works.append(work)
                seen_ids.add(work_id)

    return all_works


if __name__ == "__main__":
    main()
