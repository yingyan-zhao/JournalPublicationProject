import csv
import json
from pathlib import Path
from typing import Any

import os


os.chdir("/Users/yingyan_zhao/Dropbox/JournalPublicationProject")

INPUT_JSONL = Path("data/raw/Crossref_FullData_Jsonl.jsonl")
OUTPUT_CSV = Path("data/raw_csv/Crossref_Works.csv")

CSV_COLUMNS = [
    "DOI",
    "title",
    "container-title",
    "short-container-title",
    "publisher",
    "published_year",
    "published",
    "published-print",
    "published-online",
    "issued",
    "created",
    "deposited",
    "indexed",
    "volume",
    "issue",
    "page",
    "type",
    "abstract",
    "reference-count",
    "references-count",
    "is-referenced-by-count",
    "ISSN",
    "issn-type",
    "URL",
    "resource",
    "alternative-id",
    "subtitle",
    "language",
    "source",
    "prefix",
    "member",
    "score",
    "authors",
    "author_given",
    "author_family",
    "author_sequence",
    "author_affiliations",
    "author_orcids",
    "reference",
    "relation",
    "license",
    "funder",
    "collection_date",
    "_query_journal",
    "_query_issn",
]


def main() -> None:
    rows, skipped_no_doi, included_no_year, duplicate_dois = collect_rows(INPUT_JSONL)
    write_csv(rows, OUTPUT_CSV)

    print(f"Wrote {len(rows)} Crossref DOI-level records.")
    print(f"Skipped {skipped_no_doi} records without DOI.")
    print(f"Included {included_no_year} records without publication year.")
    print(f"Skipped {duplicate_dois} duplicate DOI records.")
    print(f"Wrote one CSV file to {OUTPUT_CSV}.")


def collect_rows(path: Path) -> tuple[list[dict[str, Any]], int, int, int]:
    rows = []
    seen_dois = set()
    skipped_no_doi = 0
    included_no_year = 0
    duplicate_dois = 0

    for record in read_jsonl(path):
        doi = normalize_doi(record.get("DOI", ""))
        if doi == "":
            skipped_no_doi += 1
            continue

        if doi in seen_dois:
            duplicate_dois += 1
            continue

        year = publication_year(record)
        if year is None:
            included_no_year += 1

        seen_dois.add(doi)
        rows.append(select_fields(record, doi, year))

    return rows, skipped_no_doi, included_no_year, duplicate_dois


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as file:
        for line in file:
            if line.strip():
                yield json.loads(line)


def select_fields(record: dict[str, Any], doi: str, year: int | None) -> dict[str, Any]:
    author_summary = summarize_authors(record.get("author") or [])

    row = {}
    for column in CSV_COLUMNS:
        if column == "DOI":
            value = doi
        elif column == "published_year":
            value = year if year is not None else ""
        elif column in author_summary:
            value = author_summary[column]
        else:
            value = record.get(column, "")
        row[column] = value_for_csv(value)

    return row


def summarize_authors(authors: list[dict[str, Any]]) -> dict[str, str]:
    author_names = []
    author_given = []
    author_family = []
    author_sequence = []
    author_affiliations = []
    author_orcids = []

    for author in authors:
        given = clean_text(author.get("given", ""))
        family = clean_text(author.get("family", ""))
        name = clean_text(author.get("name", ""))
        if name == "":
            name = " ".join(part for part in [given, family] if part)

        append_if_present(author_names, name)
        append_if_present(author_given, given)
        append_if_present(author_family, family)
        append_if_present(author_sequence, author.get("sequence", ""))
        append_if_present(author_orcids, author.get("ORCID", ""))

        affiliations = author.get("affiliation") or []
        affiliation_names = [
            clean_text(affiliation.get("name", ""))
            for affiliation in affiliations
            if clean_text(affiliation.get("name", "")) != ""
        ]
        append_if_present(author_affiliations, " | ".join(affiliation_names))

    return {
        "authors": "; ".join(author_names),
        "author_given": "; ".join(author_given),
        "author_family": "; ".join(author_family),
        "author_sequence": "; ".join(author_sequence),
        "author_affiliations": "; ".join(author_affiliations),
        "author_orcids": "; ".join(author_orcids),
    }


def publication_year(record: dict[str, Any]) -> int | None:
    for field in ["published", "published-print", "published-online", "issued", "created"]:
        year = year_from_date_parts(record.get(field))
        if year is not None:
            return year
    return None


def year_from_date_parts(value: Any) -> int | None:
    if not isinstance(value, dict):
        return None

    date_parts = value.get("date-parts") or []
    if not date_parts or not date_parts[0]:
        return None

    try:
        return int(date_parts[0][0])
    except (TypeError, ValueError):
        return None


def normalize_doi(doi: Any) -> str:
    if doi is None:
        return ""
    return (
        str(doi)
        .strip()
        .lower()
        .replace("https://doi.org/", "")
        .replace("http://doi.org/", "")
        .replace("http://dx.doi.org/", "")
    )


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).strip().split())


def append_if_present(values: list[str], value: Any) -> None:
    value = clean_text(value)
    if value:
        values.append(value)


def write_csv(rows: list[dict[str, Any]], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def value_for_csv(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    if value is None:
        return ""
    return value


if __name__ == "__main__":
    main()
