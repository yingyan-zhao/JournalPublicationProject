import csv
import json
from pathlib import Path
from typing import Any

import os


os.chdir("/Users/yingyan_zhao/Dropbox/JournalPublicationProject")

INPUT_JSONL = Path("data/raw/OpenAlex_FullData_Jsonl.jsonl")
OUTPUT_CSV = Path("data/raw_csv/OpenAlex_Works.csv")

CSV_COLUMNS = [
    "id",
    "doi",
    "title",
    "display_name",
    "publication_year",
    "publication_date",
    "type",
    "type_crossref",
    "authors",
    "author_ids",
    "raw_author_names",
    "orcid_ids",
    "author_positions",
    "author_institutions",
    "author_institution_ids",
    "raw_affiliation_strings",
    "institutions",
    "keywords",
    "concepts",
    "topics",
    "primary_topic",
    "abstract_inverted_index",
    "cited_by_count",
    "counts_by_year",
    "collection_date",
    "_query_journal",
    "_query_issn",
]


def main() -> None:
    rows, included_no_doi, skipped_no_id, duplicate_ids = collect_rows(INPUT_JSONL)
    write_csv(rows, OUTPUT_CSV)

    print(f"Wrote {len(rows)} OpenAlex ID-level records.")
    print(f"Included {included_no_doi} records without DOI.")
    print(f"Skipped {skipped_no_id} records without OpenAlex ID.")
    print(f"Skipped {duplicate_ids} duplicate OpenAlex ID records.")
    print(f"Wrote one CSV file to {OUTPUT_CSV}.")


def collect_rows(path: Path) -> tuple[list[dict[str, Any]], int, int, int]:
    rows = []
    seen_ids = set()
    included_no_doi = 0
    skipped_no_id = 0
    duplicate_ids = 0

    for record in read_jsonl(path):
        openalex_id = normalize_openalex_id(record.get("id", ""))
        if openalex_id == "":
            skipped_no_id += 1
            continue

        if openalex_id in seen_ids:
            duplicate_ids += 1
            continue

        doi = normalize_doi(record.get("doi", ""))
        if doi == "":
            included_no_doi += 1

        title = clean_title(record)

        seen_ids.add(openalex_id)
        rows.append(select_fields(record, doi, title))

    return rows, included_no_doi, skipped_no_id, duplicate_ids


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as file:
        for line in file:
            if line.strip():
                yield json.loads(line)


def select_fields(record: dict[str, Any], doi: str, title: str) -> dict[str, Any]:
    author_summary = summarize_authorships(record.get("authorships") or [])

    row = {}
    for column in CSV_COLUMNS:
        if column == "doi":
            value = doi
        elif column == "title":
            value = title
        elif column in author_summary:
            value = author_summary[column]
        else:
            value = record.get(column, "")
        row[column] = value_for_csv(value)

    return row


def summarize_authorships(authorships: list[dict[str, Any]]) -> dict[str, str]:
    authors = []
    author_ids = []
    raw_author_names = []
    orcid_ids = []
    author_positions = []
    author_institutions = []
    author_institution_ids = []
    raw_affiliation_strings = []

    for authorship in authorships:
        author = authorship.get("author") or {}
        institutions = authorship.get("institutions") or []

        append_if_present(authors, author.get("display_name", ""))
        append_if_present(author_ids, author.get("id", ""))
        append_if_present(raw_author_names, authorship.get("raw_author_name", ""))
        append_if_present(orcid_ids, authorship.get("raw_orcid", ""))
        append_if_present(author_positions, authorship.get("author_position", ""))

        institution_names = [
            institution.get("display_name", "")
            for institution in institutions
            if institution.get("display_name")
        ]
        institution_ids = [
            institution.get("id", "")
            for institution in institutions
            if institution.get("id")
        ]
        raw_affiliations = authorship.get("raw_affiliation_strings") or []

        append_if_present(author_institutions, " | ".join(institution_names))
        append_if_present(author_institution_ids, " | ".join(institution_ids))
        append_if_present(raw_affiliation_strings, " | ".join(raw_affiliations))

    return {
        "authors": "; ".join(authors),
        "author_ids": "; ".join(author_ids),
        "raw_author_names": "; ".join(raw_author_names),
        "orcid_ids": "; ".join(orcid_ids),
        "author_positions": "; ".join(author_positions),
        "author_institutions": "; ".join(author_institutions),
        "author_institution_ids": "; ".join(author_institution_ids),
        "raw_affiliation_strings": "; ".join(raw_affiliation_strings),
    }


def append_if_present(values: list[str], value: Any) -> None:
    if value is None:
        return
    value = str(value).strip()
    if value:
        values.append(value)


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


def normalize_openalex_id(openalex_id: Any) -> str:
    if openalex_id is None:
        return ""
    return str(openalex_id).strip().lower()


def clean_title(record: dict[str, Any]) -> str:
    title = record.get("title") or record.get("display_name") or ""
    return " ".join(str(title).strip().split())


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
