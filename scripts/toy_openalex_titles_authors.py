"""Toy OpenAlex example: collect titles and authors from top-five economics journals.

Run:
    python scripts/toy_openalex_titles_authors.py
"""

import csv
import json
from pathlib import Path

import requests


OPENALEX_WORKS_URL = "https://api.openalex.org/works"
OUTPUT_JSONL = Path("data/raw/toy_openalex_works.jsonl")
OUTPUT_CSV = Path("data/processed/toy_openalex_titles_authors.csv")

TOP_FIVE_ECON_JOURNALS = {
    "American Economic Review": ["0002-8282", "1944-7981"],
    "Journal of Political Economy": ["0022-3808", "1537-534X"],
    "Econometrica": ["0012-9682", "1468-0262"],
    "Review of Economic Studies": ["0034-6527", "1467-937X"],
    "Quarterly Journal of Economics": ["0033-5533", "1531-4650"],
}


def get_openalex_works(journal_name, issn, year=2020, per_page=5):
    """Ask OpenAlex for a few works from one journal ISSN."""
    params = {
        "filter": (
            f"primary_location.source.issn:{issn},"
            f"from_publication_date:{year}-01-01,"
            f"to_publication_date:{year}-12-31,"
            "type:article"
        ),
        "select": "title,authorships",
        "per-page": per_page,
    }

    response = requests.get(OPENALEX_WORKS_URL, params=params, timeout=30)
    response.raise_for_status()

    works = response.json()["results"]

    rows = []
    for work in works:
        authors = []
        for authorship in work.get("authorships", []):
            author = authorship.get("author", {})
            authors.append(author.get("display_name", ""))

        rows.append(
            {
                "journal": journal_name,
                "year": year,
                "title": work.get("title", ""),
                "authors": "; ".join(authors),
            }
        )

    return works, rows


def write_jsonl(works, output_jsonl):
    """Write raw OpenAlex works to a JSON Lines file."""
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    with output_jsonl.open("w", encoding="utf-8") as file:
        for work in works:
            file.write(json.dumps(work, ensure_ascii=False) + "\n")


def write_csv(rows, output_csv):
    """Write rows to a CSV file."""
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    with output_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["journal", "year", "title", "authors"])
        writer.writeheader()
        writer.writerows(rows)


def main():
    all_works = []
    all_rows = []
    for journal_name, issns in TOP_FIVE_ECON_JOURNALS.items():
        first_issn = issns[0]
        works, rows = get_openalex_works(journal_name, first_issn)
        all_works.extend(works)
        all_rows.extend(rows)

    write_jsonl(all_works, OUTPUT_JSONL)
    write_csv(all_rows, OUTPUT_CSV)
    print(f"Wrote {len(all_works)} raw OpenAlex records to {OUTPUT_JSONL}")
    print(f"Wrote {len(all_rows)} rows to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
