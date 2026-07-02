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


## #########################################################################
# In 01_CleanAuthorNames_aea.py, do the following steps only
#
# Step 1. from OpenAlex_Crossref_Webscrape_NBER_Repec_AEA_Merged.csv, only take doi_full and aea_authors.
# Step 2. Convert aea_authors to ASCII.
# Step 3. Splits aea_authors by semicolon ";", so one paper with multiple authors becomes multiple author rows.
#
# Step 4. Split first name and last name, also calculate the length of first and last name.
#
# Step 5. If aea_author_first_name has only 1 single letter, then take the second sequence in aea_authors. If the second sequence is not equal to last name, and the second sequence is not single letter, and take the second sequence as the first name. Otherwise keep the original single letter as aea_author_first_name.
#
# Step 6. If aea_author_last_name has only 1 single letter, then take the second to the last sequence in aea_authors. If the second to the last sequence is not equal to first name, and the second to the last sequence is not single letter, and take the second to the last sequence as the last name. Otherwise keep the original single letter as aea_author_last_name.
## #########################################################################


def main() -> None:
    data = pd.read_csv(INPUT_CSV, dtype=str).fillna("")

    # Step 1. from OpenAlex_Crossref_Webscrape_NBER_Repec_AEA_Merged.csv, only take doi_full and aea_authors.
    selected = select_aea_author_columns(data)

    # Step 2. Convert aea_authors to ASCII.
    selected[AEA_AUTHOR_COLUMN] = selected[AEA_AUTHOR_COLUMN].apply(to_ascii)

    # Step 3. Splits aea_authors by semicolon ";", so one paper with multiple authors becomes multiple author rows.
    author_rows = split_aea_authors(selected)

    # Step 4. Split first name and last name, also calculate the length of first and last name.
    author_rows = add_first_last_name_columns(author_rows)

    # Step 5. If aea_author_first_name has only 1 single letter, then take the second sequence in aea_authors. If the second sequence is not equal to last name, and the second sequence is not single letter, and take the second sequence as the first name. Otherwise keep the original single letter as aea_author_first_name.
    author_rows["aea_author_first_name"] = author_rows.apply(
        lambda row: replace_single_letter_first_name(
            row[AEA_AUTHOR_COLUMN],
            row["aea_author_first_name"],
            row["aea_author_last_name"],
        ),
        axis=1,
    )
    author_rows["aea_author_first_name_length"] = author_rows[
        "aea_author_first_name"
    ].str.len()

    # Step 6. If aea_author_last_name has only 1 single letter, then take the second to the last sequence in aea_authors. If the second to the last sequence is not equal to first name, and the second to the last sequence is not single letter, and take the second to the last sequence as the last name. Otherwise keep the original single letter as aea_author_last_name.
    author_rows["aea_author_last_name"] = author_rows.apply(
        lambda row: replace_single_letter_last_name(
            row[AEA_AUTHOR_COLUMN],
            row["aea_author_first_name"],
            row["aea_author_last_name"],
        ),
        axis=1,
    )
    author_rows["aea_author_last_name_length"] = author_rows[
        "aea_author_last_name"
    ].str.len()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    author_rows.to_csv(OUTPUT_AUTHOR_CSV, index=False)

    print("AEA author-name cleaning summary:")
    print(f"  Input CSV: {INPUT_CSV}")
    print(f"  Input rows: {len(data)}")
    print(f"  Selected columns: doi_full, aea_authors")
    print(f"  Rows with nonblank aea_authors: {count_nonblank(selected, AEA_AUTHOR_COLUMN)}")
    print(f"  Author-level output CSV: {OUTPUT_AUTHOR_CSV}")
    print(f"  Author-level rows: {len(author_rows)}")


def select_aea_author_columns(data: pd.DataFrame) -> pd.DataFrame:
    missing_columns = [
        column for column in ["doi_full", AEA_AUTHOR_COLUMN] if column not in data.columns
    ]
    if missing_columns:
        raise ValueError(f"Missing columns: {missing_columns}")

    return data[["doi_full", AEA_AUTHOR_COLUMN]].copy()


def split_aea_authors(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, record in data.iterrows():
        authors = split_semicolon_cell(record.get(AEA_AUTHOR_COLUMN, ""))
        for author in authors:
            rows.append(
                {
                    "doi_full": record.get("doi_full", ""),
                    AEA_AUTHOR_COLUMN: author,
                }
            )

    return pd.DataFrame(rows, columns=["doi_full", AEA_AUTHOR_COLUMN])


def add_first_last_name_columns(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    names = data[AEA_AUTHOR_COLUMN].apply(split_first_last_name)
    data["aea_author_first_name"] = names.apply(lambda name: name[0])
    data["aea_author_first_name_length"] = data["aea_author_first_name"].str.len()
    data["aea_author_last_name"] = names.apply(lambda name: name[1])
    data["aea_author_last_name_length"] = data["aea_author_last_name"].str.len()
    return data[
        [
            "doi_full",
            AEA_AUTHOR_COLUMN,
            "aea_author_first_name",
            "aea_author_first_name_length",
            "aea_author_last_name",
            "aea_author_last_name_length",
        ]
    ].copy()


def split_first_last_name(name: object) -> tuple[str, str]:
    text = clean_text(name)
    if not text:
        return "", ""

    parts = text.split()
    suffixes = {"jr", "jr.", "sr", "sr.", "ii", "iii", "iv"}
    while len(parts) > 1 and parts[-1].lower() in suffixes:
        parts = parts[:-1]

    if len(parts) == 1:
        return "", clean_name_part(parts[0])
    return clean_name_part(parts[0]), clean_name_part(parts[-1])


def replace_single_letter_first_name(author: object, first_name: str, last_name: str) -> str:
    if len(first_name) != 1:
        return first_name

    parts = clean_text(author).split()
    if len(parts) < 2:
        return first_name

    second_sequence = clean_name_part(parts[1])
    if not second_sequence:
        return first_name
    if second_sequence == last_name:
        return first_name
    if len(second_sequence) == 1:
        return first_name
    return second_sequence


def replace_single_letter_last_name(author: object, first_name: str, last_name: str) -> str:
    if len(last_name) != 1:
        return last_name

    parts = clean_text(author).split()
    if len(parts) < 2:
        return last_name

    second_to_last_sequence = clean_name_part(parts[-2])
    if not second_to_last_sequence:
        return last_name
    if second_to_last_sequence == first_name:
        return last_name
    if len(second_to_last_sequence) == 1:
        return last_name
    return second_to_last_sequence


def clean_name_part(name: object) -> str:
    text = clean_text(name)
    text = text.replace(".", "")
    text = text.replace(",", "")
    return re.sub(r"\s+", " ", text).strip()


def split_semicolon_cell(value: object) -> list[str]:
    text = clean_text(value)
    if not text:
        return []
    return [part.strip() for part in text.split(";") if part.strip()]


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


def count_nonblank(data: pd.DataFrame, column: str) -> int:
    if column not in data.columns:
        return 0
    return int(data[column].fillna("").astype(str).str.strip().ne("").sum())


if __name__ == "__main__":
    main()
