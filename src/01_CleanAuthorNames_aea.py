from pathlib import Path
import html
import os
import re
import unicodedata

import pandas as pd


os.chdir("/Users/yingyan_zhao/Dropbox/JournalPublicationProject")

INPUT_CSV = Path("data/processed/OpenAlex_Crossref_Webscrape_NBER_Repec_AEA_Merged.csv")
OUTPUT_DIR = Path("data/processed/author_names")
OUTPUT_AUTHOR_CSV = OUTPUT_DIR / "JEL_Training_Data_AEA_AuthorRows.csv"

AEA_AUTHOR_COLUMN = "aea_authors"
AEA_INSTITUTION_COLUMN = "aea_author_institutions"
AEA_AUTHOR_INSTITUTION_PAIR_COLUMN = "aea_author_institution_pairs"


def main() -> None:
    data = pd.read_csv(INPUT_CSV, dtype=str).fillna("")
    data["complete_row_id"] = data.index.astype(str)

    author_rows = build_aea_author_rows(data)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    author_rows.to_csv(OUTPUT_AUTHOR_CSV, index=False)

    print("AEA author-name cleaning summary:")
    print(f"  Input CSV: {INPUT_CSV}")
    print(f"  Input rows: {len(data)}")
    print(f"  Rows with nonblank aea_authors: {count_nonblank(data, AEA_AUTHOR_COLUMN)}")
    print(f"  Author-level output CSV: {OUTPUT_AUTHOR_CSV}")
    print(f"  Author-level rows: {len(author_rows)}")
    print(
        "  Unique AEA first-last names: "
        f"{count_unique_first_last_names(author_rows, 'aea_author_first_name', 'aea_author_last_name')}"
    )


def build_aea_author_rows(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, record in data.iterrows():
        authors = split_author_cell(record.get(AEA_AUTHOR_COLUMN, ""))

        for position, author_name_raw in enumerate(authors, start=1):
            author_name_clean = clean_aea_author_name(author_name_raw)
            author_name_ascii = clean_author_name_ascii(to_ascii(author_name_clean))
            first_name, last_name = split_first_last_name(author_name_ascii)
            rows.append(
                {
                    "doi_full": record.get("doi_full", ""),
                    "aea_author_first_name": first_name,
                    "aea_author_first_name_length": len(first_name),
                    "aea_author_last_name": last_name,
                    "aea_author_last_name_length": len(last_name),
                }
            )

    return pd.DataFrame(rows)


def split_author_cell(value: object) -> list[str]:
    return [clean_author_name(name) for name in split_semicolon_cell(value) if clean_author_name(name)]


def split_semicolon_cell(value: object) -> list[str]:
    text = clean_text(value)
    if not text:
        return []
    return [part.strip() for part in text.split(";") if part.strip()]


def clean_aea_author_name(name: object) -> str:
    text = clean_author_name(name)
    return remove_leading_initials(text)


def clean_author_name(name: object) -> str:
    text = clean_text(name)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def remove_leading_initials(name: object) -> str:
    text = clean_author_name(name)
    return re.sub(r"^(?:[A-Z]\.\s+)+", "", text).strip()


def clean_text(value: object) -> str:
    if pd.isna(value):
        return ""
    text = html.unescape(str(value))
    text = text.replace("\u00a0", " ")
    text = re.sub(r"[\r\n\t]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def to_ascii(value: object) -> str:
    text = clean_text(value)
    normalized = unicodedata.normalize("NFKD", text)
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
    return re.sub(r"\s+", " ", ascii_text).strip()


def clean_author_name_ascii(name: object) -> str:
    text = clean_author_name(name)
    if text == "Erlend E. B":
        return text

    text = re.sub(r"^(?:[A-Z]\.\s+)+", "", text).strip()
    text = re.sub(r"\sr$", "", text).strip()
    text = re.sub(r"(?:\s+[A-Z]\.)+$", "", text).strip()
    return re.sub(r"\s+", " ", text).strip()


def split_first_last_name(name: object) -> tuple[str, str]:
    text = clean_author_name(name)
    if not text:
        return "", ""

    parts = text.split()
    suffixes = {"jr", "jr.", "sr", "sr.", "ii", "iii", "iv"}
    while len(parts) > 1 and parts[-1].lower() in suffixes:
        parts = parts[:-1]

    if len(parts) == 1:
        return "", parts[0].replace(".", "")
    return parts[0].replace(".", ""), parts[-1].replace(".", "")


def value_at_position(values: list[str], position: int) -> str:
    index = position - 1
    if 0 <= index < len(values):
        return values[index]
    return ""


def first_nonblank(record: pd.Series, columns: list[str]) -> str:
    for column in columns:
        value = clean_text(record.get(column, ""))
        if value:
            return value
    return ""


def count_nonblank(data: pd.DataFrame, column: str) -> int:
    if column not in data.columns:
        return 0
    return int(data[column].fillna("").astype(str).str.strip().ne("").sum())


def count_unique_first_last_names(data: pd.DataFrame, first_column: str, last_column: str) -> int:
    first = data[first_column].fillna("").astype(str).str.strip()
    last = data[last_column].fillna("").astype(str).str.strip()
    full_name = (first + " " + last).str.strip()
    return int(full_name.loc[full_name != ""].nunique())


if __name__ == "__main__":
    main()
