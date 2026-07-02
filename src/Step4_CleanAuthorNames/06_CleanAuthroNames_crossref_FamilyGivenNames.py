from pathlib import Path
import html
import os
import re
import unicodedata

import pandas as pd


os.chdir("/Users/yingyan_zhao/Dropbox/JournalPublicationProject")

INPUT_CSV = Path("data/processed/OpenAlex_Crossref_Webscrape_NBER_Repec_AEA_Merged.csv")
OUTPUT_DIR = Path("data/processed/author_names")
OUTPUT_AUTHOR_CSV = OUTPUT_DIR / "JEL_Training_Data_Crossref_GivenFamily_AuthorRows.csv"

CROSSREF_GIVEN_COLUMN = "crossref_author_given"
CROSSREF_FAMILY_COLUMN = "crossref_author_family"


## #########################################################################
# In 06_CleanAuthroNames_crossref_FamilyGivenNames.py, do the following steps
#
# Step 1. from OpenAlex_Crossref_Webscrape_NBER_Repec_AEA_Merged.csv, only take doi_full and crossref_author_given	crossref_author_family
# Step 2. Convert crossref_author_given	crossref_author_family to ASCII.
# Step 3. Split crossref_author_given and crossref_author_family,  so one paper with multiple authors becomes multiple author rows.
# Step 4. for the record 10.1257/000282803321947317, given name Jomo K should be Jomo, family name S should be KS.
# Step 5. Clean first and last name. Only take the first sequence of the first name. Only take the last sequence of the first name. Calculate the length of  first and last name
# Step 6. If crossref_author_first_name has only 1 single letter, then take the second sequence  in crossref_author_given. If the second sequence is not single letter, and take the second sequence as the first name. Otherwise keep the original single letter as in crossref_author_first_name
# Step 7. If crossref_author_last_name has only 1 single letter, then take the second to the last sequence in crossref_author_family. If the second to the last sequence is not single letter, and take the second to the last sequence as the last name. Otherwise keep the original single letter as in crossref_author_last_name
## #########################################################################


def main() -> None:
    data = pd.read_csv(INPUT_CSV, dtype=str).fillna("")

    # Step 1. from OpenAlex_Crossref_Webscrape_NBER_Repec_AEA_Merged.csv, only take doi_full and crossref_author_given crossref_author_family.
    selected = select_crossref_given_family_columns(data)

    # Step 2. Convert crossref_author_given crossref_author_family to ASCII.
    selected[CROSSREF_GIVEN_COLUMN] = selected[CROSSREF_GIVEN_COLUMN].apply(to_ascii)
    selected[CROSSREF_FAMILY_COLUMN] = selected[CROSSREF_FAMILY_COLUMN].apply(to_ascii)

    # Step 3. Split crossref_author_given and crossref_author_family, so one paper with multiple authors becomes multiple author rows.
    author_rows = split_crossref_given_family_names(selected)

    # Step 4. For the record 10.1257/000282803321947317, given name Jomo K should be Jomo, family name S should be KS.
    author_rows = apply_manual_given_family_corrections(author_rows)

    # Step 5. Clean first and last name. Only take the first sequence of the first name. Only take the last sequence of the last name. Calculate the length of first and last name.
    author_rows = add_first_last_name_columns(author_rows)

    # Step 6. If crossref_author_first_name has only 1 single letter, then take the second sequence in crossref_author_given. If the second sequence is not single letter, and take the second sequence as the first name. Otherwise keep the original single letter as in crossref_author_first_name.
    author_rows["crossref_author_first_name"] = author_rows.apply(
        lambda row: replace_single_letter_first_name(
            row[CROSSREF_GIVEN_COLUMN],
            row["crossref_author_first_name"],
        ),
        axis=1,
    )
    author_rows["crossref_author_first_name_length"] = author_rows[
        "crossref_author_first_name"
    ].str.len()

    # Step 7. If crossref_author_last_name has only 1 single letter, then take the second to the last sequence in crossref_author_family. If the second to the last sequence is not single letter, and take the second to the last sequence as the last name. Otherwise keep the original single letter as in crossref_author_last_name.
    author_rows["crossref_author_last_name"] = author_rows.apply(
        lambda row: replace_single_letter_last_name(
            row[CROSSREF_FAMILY_COLUMN],
            row["crossref_author_last_name"],
        ),
        axis=1,
    )
    author_rows["crossref_author_last_name_length"] = author_rows[
        "crossref_author_last_name"
    ].str.len()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    author_rows.to_csv(OUTPUT_AUTHOR_CSV, index=False)

    print("Crossref given/family author-name cleaning summary:")
    print(f"  Input CSV: {INPUT_CSV}")
    print(f"  Input rows: {len(data)}")
    print("  Selected columns: doi_full, crossref_author_given, crossref_author_family")
    print(f"  Rows with nonblank crossref_author_given: {count_nonblank(selected, CROSSREF_GIVEN_COLUMN)}")
    print(f"  Rows with nonblank crossref_author_family: {count_nonblank(selected, CROSSREF_FAMILY_COLUMN)}")
    print(f"  Author-level output CSV: {OUTPUT_AUTHOR_CSV}")
    print(f"  Author-level rows: {len(author_rows)}")


def select_crossref_given_family_columns(data: pd.DataFrame) -> pd.DataFrame:
    columns = ["doi_full", CROSSREF_GIVEN_COLUMN, CROSSREF_FAMILY_COLUMN]
    missing_columns = [column for column in columns if column not in data.columns]
    if missing_columns:
        raise ValueError(f"Missing columns: {missing_columns}")

    return data[columns].copy()


def split_crossref_given_family_names(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, record in data.iterrows():
        given_names = split_semicolon_cell(record.get(CROSSREF_GIVEN_COLUMN, ""))
        family_names = split_semicolon_cell(record.get(CROSSREF_FAMILY_COLUMN, ""))
        author_count = max(len(given_names), len(family_names))

        for position in range(author_count):
            rows.append(
                {
                    "doi_full": record.get("doi_full", ""),
                    CROSSREF_GIVEN_COLUMN: value_at_position(given_names, position),
                    CROSSREF_FAMILY_COLUMN: value_at_position(family_names, position),
                }
            )

    return pd.DataFrame(
        rows,
        columns=["doi_full", CROSSREF_GIVEN_COLUMN, CROSSREF_FAMILY_COLUMN],
    )


def apply_manual_given_family_corrections(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    target_row = (
        (data["doi_full"].fillna("").astype(str).str.strip() == "10.1257/000282803321947317")
        & (data[CROSSREF_GIVEN_COLUMN].fillna("").astype(str).str.strip() == "Jomo K")
        & (data[CROSSREF_FAMILY_COLUMN].fillna("").astype(str).str.strip() == "S")
    )
    data.loc[target_row, CROSSREF_GIVEN_COLUMN] = "Jomo"
    data.loc[target_row, CROSSREF_FAMILY_COLUMN] = "KS"
    return data


def add_first_last_name_columns(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    data["crossref_author_first_name"] = data[CROSSREF_GIVEN_COLUMN].apply(first_sequence)
    data["crossref_author_first_name_length"] = data["crossref_author_first_name"].str.len()
    data["crossref_author_last_name"] = data[CROSSREF_FAMILY_COLUMN].apply(last_sequence)
    data["crossref_author_last_name_length"] = data["crossref_author_last_name"].str.len()
    return data[
        [
            "doi_full",
            CROSSREF_GIVEN_COLUMN,
            CROSSREF_FAMILY_COLUMN,
            "crossref_author_first_name",
            "crossref_author_first_name_length",
            "crossref_author_last_name",
            "crossref_author_last_name_length",
        ]
    ].copy()


def first_sequence(value: object) -> str:
    parts = clean_name_part(value).split()
    if not parts:
        return ""
    return parts[0]


def replace_single_letter_first_name(given_name: object, first_name: str) -> str:
    if len(first_name) != 1:
        return first_name

    parts = clean_name_part(given_name).split()
    if len(parts) < 2:
        return first_name

    second_sequence = parts[1]
    if len(second_sequence) == 1:
        return first_name
    return second_sequence


def replace_single_letter_last_name(family_name: object, last_name: str) -> str:
    if len(last_name) != 1:
        return last_name

    parts = clean_name_part(family_name).split()
    if len(parts) < 2:
        return last_name

    second_to_last_sequence = parts[-2]
    if len(second_to_last_sequence) == 1:
        return last_name
    return second_to_last_sequence


def last_sequence(value: object) -> str:
    parts = clean_name_part(value).split()
    if not parts:
        return ""
    return parts[-1]


def clean_name_part(value: object) -> str:
    text = clean_text(value)
    text = text.replace(".", "")
    text = text.replace(",", "")
    return re.sub(r"\s+", " ", text).strip()


def split_semicolon_cell(value: object) -> list[str]:
    text = clean_text(value)
    if not text:
        return []
    return [part.strip() for part in text.split(";") if part.strip()]


def value_at_position(values: list[str], position: int) -> str:
    if 0 <= position < len(values):
        return values[position]
    return ""


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
