from pathlib import Path
import csv
import os


os.chdir("/Users/yingyan_zhao/Dropbox/JournalPublicationProject")

INPUT_DIR = Path(
    os.environ.get(
        "REPEC_INPUT_DIR",
        "/Users/yingyan_zhao/Dropbox/JournalPublicationProject/data/raw/repec/RePEc-ReDIF",
    )
)
OUTPUT_DIR = Path("data/raw_csv/repec_by_year")
REDIF_SUFFIXES = {".rdf", ".redif"}
MIN_YEAR = 1950
MAX_YEAR = 2026

REPEC_FIELDS_TO_KEEP = [
    "source_file",
    "template_type",
    "handle",
    "title",
    "author_name",
    "author_name_first",
    "author_name_last",
    "author_person",
    "author_workplace_name",
    "year",
    "abstract",
    "keywords",
    "classification_jel",
    "journal",
    "doi",
    "x_doi",
    "issn",
    "isbn",
    "publication_status",
    "language",
    "creation_date",
    "revision_date",
    "publisher_name",
    "editor_name",
    "editor_email",
    "editor_workplace_name",
]


def main() -> None:
    redif_files = list_redif_files(INPUT_DIR)
    print(f"Found {len(redif_files)} ReDIF files.")

    year_summary, skipped_no_year = write_records_by_year(
        paths=redif_files,
        output_dir=OUTPUT_DIR,
        fieldnames=REPEC_FIELDS_TO_KEEP,
    )

    total_written = sum(summary["written"] for summary in year_summary.values())
    total_missing_handle = sum(
        summary["missing_handle"] for summary in year_summary.values()
    )
    total_missing_title = sum(
        summary["missing_title"] for summary in year_summary.values()
    )
    total_missing_author = sum(
        summary["missing_author"] for summary in year_summary.values()
    )
    total_duplicate_handle_title_authors = sum(
        summary["duplicate_handle_title_author"] for summary in year_summary.values()
    )

    print(f"Wrote {total_written} RePEc handle-title-author-level records.")
    print(f"Found {total_missing_handle} records with blank handle.")
    print(f"Found {total_missing_title} records with blank title.")
    print(f"Found {total_missing_author} records with blank author.")
    print(
        "Skipped "
        f"{skipped_no_year} records without year or outside {MIN_YEAR}-{MAX_YEAR}."
    )
    print(
        "Skipped "
        f"{total_duplicate_handle_title_authors} duplicate handle-title-author records."
    )
    print(f"Wrote {len(year_summary)} yearly CSV files to {OUTPUT_DIR}.")
    print_year_summary(year_summary)


def list_redif_files(input_dir: Path) -> list[Path]:
    if not input_dir.exists():
        raise FileNotFoundError(
            f"{input_dir} does not exist. Run src/05_download_repec.py first."
        )

    return [
        path
        for path in input_dir.rglob("*")
        if path.is_file()
        and not path.name.startswith(".")
        and path.suffix.lower() in REDIF_SUFFIXES
    ]


def write_records_by_year(
    paths: list[Path],
    output_dir: Path,
    fieldnames: list[str],
) -> tuple[dict[int, dict[str, int]], int]:
    output_dir.mkdir(parents=True, exist_ok=True)
    remove_out_of_range_outputs(output_dir)

    writer_bundle_by_year = {}
    year_summary = {}
    seen_record_keys = set()
    skipped_no_year = 0

    try:
        for path in paths:
            for record in parse_redif_file(path):
                year = record_year(record)
                if year is None or year < MIN_YEAR or year > MAX_YEAR:
                    skipped_no_year += 1
                    continue

                handle = clean_text(record.get("handle", ""))
                title = clean_text(record.get("title", ""))
                author = clean_text(record.get("author_name", ""))

                if handle == "":
                    increment_year_summary(year_summary, year, "missing_handle")
                if title == "":
                    increment_year_summary(year_summary, year, "missing_title")
                if author == "":
                    increment_year_summary(year_summary, year, "missing_author")

                record_key = make_record_key(handle, title, author)
                if record_key in seen_record_keys:
                    increment_year_summary(
                        year_summary,
                        year,
                        "duplicate_handle_title_author",
                    )
                    continue

                writer_bundle = writer_for_year(
                    year=year,
                    output_dir=output_dir,
                    fieldnames=fieldnames,
                    writer_bundle_by_year=writer_bundle_by_year,
                )

                row = {"source_file": str(path.relative_to(INPUT_DIR))}
                row["handle"] = handle
                row["title"] = title
                row["author_name"] = author
                row.update(record)
                row["handle"] = handle
                row["title"] = title
                row["author_name"] = author
                writer_bundle["writer"].writerow(row)
                increment_year_summary(year_summary, year, "written")
                seen_record_keys.add(record_key)
    finally:
        close_writer_bundles(writer_bundle_by_year)

    return year_summary, skipped_no_year


def remove_out_of_range_outputs(output_dir: Path) -> None:
    removed_files = []
    for path in output_dir.glob("RePEc_ReDIF_*.csv"):
        year = year_from_filename(path)
        if year is not None and (year < MIN_YEAR or year > MAX_YEAR):
            path.unlink()
            removed_files.append(path.name)

    if removed_files:
        print(
            "Removed old RePEc yearly files outside "
            f"{MIN_YEAR}-{MAX_YEAR}: {removed_files}"
        )


def year_from_filename(path: Path) -> int | None:
    name = path.name
    for part in name.replace(".", "_").split("_"):
        if part.isdigit() and len(part) == 4:
            return int(part)
    return None


def writer_for_year(
    year: int,
    output_dir: Path,
    fieldnames: list[str],
    writer_bundle_by_year: dict[int, dict[str, object]],
) -> dict[str, object]:
    if year in writer_bundle_by_year:
        return writer_bundle_by_year[year]

    output_path = output_dir / f"RePEc_ReDIF_{year}.csv"
    file = output_path.open("w", newline="", encoding="utf-8")
    writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()

    writer_bundle = {"file": file, "writer": writer}
    writer_bundle_by_year[year] = writer_bundle
    return writer_bundle


def close_writer_bundles(writer_bundle_by_year: dict[int, dict[str, object]]) -> None:
    for writer_bundle in writer_bundle_by_year.values():
        writer_bundle["file"].close()


def increment_year_summary(
    year_summary: dict[int, dict[str, int]],
    year: int,
    column: str,
) -> None:
    if year not in year_summary:
        year_summary[year] = {
            "written": 0,
            "missing_handle": 0,
            "missing_title": 0,
            "missing_author": 0,
            "duplicate_handle_title_author": 0,
        }

    year_summary[year][column] += 1


def print_year_summary(year_summary: dict[int, dict[str, int]]) -> None:
    print("Year summary:")
    print(
        "year,written,missing_handle,missing_title,missing_author,"
        "duplicate_handle_title_author"
    )
    for year in sorted(year_summary):
        summary = year_summary[year]
        print(
            f"{year},"
            f"{summary['written']},"
            f"{summary['missing_handle']},"
            f"{summary['missing_title']},"
            f"{summary['missing_author']},"
            f"{summary['duplicate_handle_title_author']}"
        )


def make_record_key(handle: str, title: str, author: str) -> tuple[str, str, str]:
    return (
        normalize_key_text(handle),
        normalize_key_text(title),
        normalize_key_text(author),
    )


def normalize_key_text(value: str) -> str:
    return clean_text(value).lower()


def clean_text(value: str) -> str:
    return " ".join(str(value or "").strip().split())


def record_year(record: dict[str, str]) -> int | None:
    year = record.get("year", "").strip()
    if year == "":
        return None

    try:
        return int(year[:4])
    except ValueError:
        return None


def parse_redif_file(path: Path) -> list[dict[str, str]]:
    records = []
    current_record = {}
    current_field = None

    with path.open(encoding="utf-8", errors="replace") as file:
        for line in file:
            line = line.rstrip("\n")

            if line.strip() == "":
                save_record(records, current_record)
                current_record = {}
                current_field = None
                continue

            if line.startswith((" ", "\t")) and current_field is not None:
                current_record[current_field][-1] += " " + line.strip()
                continue

            if ":" not in line:
                continue

            field, value = line.split(":", 1)
            field = clean_field_name(field)
            value = value.strip()

            if field == "":
                continue

            current_record.setdefault(field, []).append(value)
            current_field = field

    save_record(records, current_record)
    return [flatten_record(record) for record in records]


def save_record(records: list[dict[str, list[str]]], record: dict[str, list[str]]) -> None:
    if record:
        records.append(record)


def flatten_record(record: dict[str, list[str]]) -> dict[str, str]:
    return {
        field: "; ".join(values)
        for field, values in record.items()
    }


def clean_field_name(field: str) -> str:
    return field.strip().lower().replace("-", "_")


if __name__ == "__main__":
    main()
