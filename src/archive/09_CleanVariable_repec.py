from pathlib import Path
import os
import re

import pandas as pd


os.chdir("/Users/yingyan_zhao/Dropbox/JournalPublicationProject")

INPUT_DIR = Path("data/processed/repec_by_year")
OUTPUT_DIR = Path("data/processed/repec_by_year_cleaned")
INPUT_PATTERN = "RePEc_ReDIF_*.csv"
CHUNK_SIZE = 100_000

TOP_FIVE_JOURNALS = {
    "american economic review": "American Economic Review",
    "journal of political economy": "Journal of Political Economy",
    "econometrica": "Econometrica",
    "review of economic studies": "Review of Economic Studies",
    "quarterly journal of economics": "Quarterly Journal of Economics",
}

DROP_COLUMNS = [
    "source_file",
    "template_type",
    "handle",
    "publication_status",
    "language",
    "creation_date",
    "revision_date",
    "publisher_name",
    "editor_name",
    "editor_email",
    "editor_workplace_name",
    "repec_journal_clean",
]


def main() -> None:
    input_files = sorted(INPUT_DIR.glob(INPUT_PATTERN))
    if not input_files:
        raise FileNotFoundError(f"No RePEc yearly CSV files found in {INPUT_DIR}.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    for input_path in input_files:
        year = year_from_filename(input_path)
        if year is None:
            print(f"Skipping file without year in name: {input_path}")
            continue

        output_path = OUTPUT_DIR / f"RePEc_ReDIF_Cleaned_{year}.csv"
        summary = clean_one_year(input_path, output_path, year)
        summary_rows.append(summary)
        print(
            f"{year}: read={summary['read_rows']}, "
            f"kept={summary['kept_rows']}, "
            f"dropped={summary['dropped_rows']}"
        )

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUTPUT_DIR / "RePEc_Cleaning_Summary_By_Year.csv", index=False)
    print_cleaning_summary(summary)


def year_from_filename(path: Path) -> int | None:
    match = re.search(r"(19|20)\d{2}", path.name)
    if not match:
        return None
    return int(match.group(0))


def clean_one_year(input_path: Path, output_path: Path, year: int) -> dict[str, int]:
    read_rows = 0
    kept_rows = 0
    blank_journal_rows = 0
    top_five_journal_rows = 0
    editor_author_rows = 0
    wrote_header = False

    for chunk in pd.read_csv(
        input_path,
        chunksize=CHUNK_SIZE,
        dtype=str,
        keep_default_na=False,
    ):
        if "journal" not in chunk.columns:
            raise ValueError(f"{input_path} does not have a journal column.")

        read_rows += len(chunk)
        cleaned_journals = chunk["journal"].apply(clean_journal_name)
        keep_rows = (chunk["journal"].apply(clean_text) == "") | (
            cleaned_journals != ""
        )
        editor_author = chunk["author_name"].apply(author_contains_editors)
        keep_rows = keep_rows & ~editor_author
        editor_author_rows += int(editor_author.sum())

        cleaned_chunk = chunk.loc[keep_rows].copy()
        cleaned_chunk["repec_journal_clean"] = cleaned_journals.loc[keep_rows]

        blank_journal_rows += int(
            (cleaned_chunk["journal"].apply(clean_text) == "").sum()
        )
        top_five_journal_rows += int(
            (cleaned_chunk["repec_journal_clean"] != "").sum()
        )
        kept_rows += len(cleaned_chunk)
        cleaned_chunk = drop_unwanted_columns(cleaned_chunk)
        cleaned_chunk = add_repec_prefix(cleaned_chunk)

        cleaned_chunk.to_csv(
            output_path,
            mode="w" if not wrote_header else "a",
            header=not wrote_header,
            index=False,
        )
        wrote_header = True

    if not wrote_header:
        pd.DataFrame().to_csv(output_path, index=False)

    return {
        "year": year,
        "read_rows": read_rows,
        "kept_rows": kept_rows,
        "dropped_rows": read_rows - kept_rows,
        "blank_journal_rows": blank_journal_rows,
        "top_five_journal_rows": top_five_journal_rows,
        "editor_author_rows": editor_author_rows,
    }


def drop_unwanted_columns(data: pd.DataFrame) -> pd.DataFrame:
    columns_to_drop = [column for column in DROP_COLUMNS if column in data.columns]
    return data.drop(columns=columns_to_drop)


def add_repec_prefix(data: pd.DataFrame) -> pd.DataFrame:
    return data.rename(
        columns={
            column: f"repec_{column}"
            for column in data.columns
            if not column.startswith("repec_")
        }
    )


def author_contains_editors(value: str) -> bool:
    return "editors" in clean_text(value).lower()


def clean_journal_name(value: str) -> str:
    journal = clean_text(value)
    if journal == "":
        return ""

    normalized = normalize_journal_name(journal)
    for phrase, standard_name in TOP_FIVE_JOURNALS.items():
        if phrase in normalized:
            return standard_name

    return ""


def normalize_journal_name(value: str) -> str:
    value = clean_text(value).lower()
    value = re.sub(r"^the\s+", "", value)
    return value


def clean_text(value: str) -> str:
    return " ".join(str(value or "").strip().split())


def print_cleaning_summary(summary: pd.DataFrame) -> None:
    report_columns = [
        "year",
        "read_rows",
        "kept_rows",
        "dropped_rows",
        "blank_journal_rows",
        "top_five_journal_rows",
        "editor_author_rows",
    ]

    print("\nYear-by-year RePEc cleaning summary:")
    print(summary[report_columns].to_string(index=False))

    print("\nTotals:")
    print(summary[report_columns[1:]].sum().to_string())
    print(f"Wrote cleaned yearly files to {OUTPUT_DIR}.")


if __name__ == "__main__":
    main()
