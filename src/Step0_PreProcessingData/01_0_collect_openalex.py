import requests
from pathlib import Path
import json
from datetime import date
from typing import Any, Iterable
from time import sleep
import os

os.chdir("/Users/yingyan_zhao/Dropbox/JournalPublicationProject")

OUTPUT_BY_YEAR_DIR = Path("data/raw/openalex_by_year")
FROM_YEAR = 1950
TO_YEAR = 2026
OPENALEX_PER_PAGE = 200
REQUEST_SLEEP_SECONDS = 3.0
MAX_RETRIES = 8
OVERWRITE_EXISTING_YEAR_FILES = False
OPENALEX_MAILTO = os.getenv("OPENALEX_MAILTO", "")
OPENALEX_API_KEY = os.getenv("OPENALEX_API_KEY", "")

from top_journals import TOP_FIVE_JOURNALS
print(TOP_FIVE_JOURNALS)

OPENALEX_WORKS_URL = "https://api.openalex.org/works"

def main() -> None:

    collection_date = date.today().isoformat()
    years_written = collect_works_by_year(
        from_year=FROM_YEAR,
        to_year=TO_YEAR,
        per_page=OPENALEX_PER_PAGE,
        collection_date=collection_date,
    )
    print(f"Finished collecting OpenAlex data for {years_written} year files.")



def write_jsonl(
    all_works: list[dict[str, Any]],
    output_jsonl: Path,
    collection_date: str,
    mode: str = "w",
) -> None:
    """Write raw OpenAlex works to a JSON Lines file."""
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    with output_jsonl.open(mode, encoding="utf-8") as file:
        for work in all_works:
            work_with_date = work.copy()
            work_with_date.setdefault("collection_date", collection_date)
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
        if OPENALEX_MAILTO:
            params["mailto"] = OPENALEX_MAILTO
        if OPENALEX_API_KEY:
            params["api_key"] = OPENALEX_API_KEY

        response = get_openalex_response(params)
        payload = response.json()

        for work in payload.get("results", []):
            work["_query_journal"] = journal
            work["_query_issn"] = issn
            works.append(work)

        next_cursor = (payload.get("meta") or {}).get("next_cursor")
        if not next_cursor or next_cursor == cursor:
            break
        cursor = next_cursor
        sleep(REQUEST_SLEEP_SECONDS)

    return works


def get_openalex_response(params: dict[str, Any]) -> requests.Response:
    """Request OpenAlex and retry politely when the API rate-limits us."""
    for attempt in range(1, MAX_RETRIES + 1):
        response = requests.get(OPENALEX_WORKS_URL, params=params, timeout=60)
        if response.status_code != 429:
            response.raise_for_status()
            return response

        wait_seconds = retry_wait_seconds(response, attempt)
        print(
            "OpenAlex returned 429 Too Many Requests. "
            f"Waiting {wait_seconds:g} seconds before retry {attempt}/{MAX_RETRIES}."
        )
        sleep(wait_seconds)

    response.raise_for_status()
    return response


def retry_wait_seconds(response: requests.Response, attempt: int) -> float:
    retry_after = response.headers.get("Retry-After")
    if retry_after:
        try:
            return max(float(retry_after), REQUEST_SLEEP_SECONDS)
        except ValueError:
            pass
    return min(REQUEST_SLEEP_SECONDS * (2 ** attempt), 300)

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


def collect_works_by_year(
    from_year: int,
    to_year: int,
    per_page: int,
    collection_date: str,
) -> int:
    """Collect OpenAlex records year-by-year and save each year immediately."""
    years_written = 0

    for year in range(from_year, to_year + 1):
        year_output = yearly_output_path(year)
        if year_output.exists() and not OVERWRITE_EXISTING_YEAR_FILES:
            yearly_works = read_jsonl(year_output)
            print(
                f"Using existing OpenAlex year file for {year}: "
                f"{len(yearly_works)} records from {year_output}."
            )
        else:
            yearly_works = collect_works(
                from_year=year,
                to_year=year,
                per_page=per_page,
            )
            print(f"Collected {len(yearly_works)} unique OpenAlex records for {year}.")
            write_jsonl(yearly_works, year_output, collection_date)
            print(f"Wrote {len(yearly_works)} records for {year} to {year_output}.")
        years_written += 1

    return years_written


def yearly_output_path(year: int) -> Path:
    return OUTPUT_BY_YEAR_DIR / f"OpenAlex_Year{year}_Jsonl.jsonl"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open(encoding="utf-8") as file:
        for line in file:
            if line.strip():
                records.append(json.loads(line))
    return records


if __name__ == "__main__":
    main()
