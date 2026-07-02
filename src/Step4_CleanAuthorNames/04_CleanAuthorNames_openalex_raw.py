from pathlib import Path
import html
import os
import re
import unicodedata

import pandas as pd


os.chdir("/Users/yingyan_zhao/Dropbox/JournalPublicationProject")

INPUT_CSV = Path("data/processed/OpenAlex_Crossref_Webscrape_NBER_Repec_AEA_Merged.csv")
OUTPUT_DIR = Path("data/processed/author_names")
OUTPUT_AUTHOR_CSV = OUTPUT_DIR / "JEL_Training_Data_OpenAlex_Raw_AuthorRows.csv"

OPENALEX_RAW_AUTHOR_COLUMN = "openalex_raw_author_names"


## #########################################################################
# In this file, do the following steps
#
# Step 1. from OpenAlex_Crossref_Webscrape_NBER_Repec_AEA_Merged.csv, only take doi_full and openalex_raw_author_names.
# Step 2. Convert openalex_raw_author_names to ASCII.
# Step 3. Splits openalex_raw_author_names by semicolon “;”, so one paper with multiple authors becomes multiple author rows.
# Step 4. Drop “, Jr.” in openalex_raw_author_names, also drop “Jr.“ in openalex_raw_author_names
#
# Step 5. If openalex_raw_author_names has “,” in it, it means openalex_raw_author_names stores last name first, and then first name, you need to reverse it
# Step 6. For these two (ALVAREZ F. E., ALVAREZ F) , it should be F. E. ALVAREZ ; Jomo K S should be Jomo KS; Amit G should be G Amit , LIPPI F should be F LIPPI

# Step 7. Split first name and last name, also calculate the length of  first and last name
# Step 8. If openalex_author_first_name has only 1 single letter, then take the second sequence  in openalex_raw_author_names. If the second sequence is not equal to last name, and the second sequence is not single letter, and take the second sequence as the first name. Otherwise keep the original single letter as openalex_author_first_name
# Step 9. If openalex_author_last_name has only 1 single letter, then take the second to the last sequence in openalex_raw_author_names. If the second to the last sequence is not equal to first name, and the second to the last sequence is not single letter, and take the second to the last sequence as the last name. Otherwise keep the original single letter as openalex_author_last_name.
## #########################################################################


def main() -> None:
    data = pd.read_csv(INPUT_CSV, dtype=str).fillna("")

    # Step 1. from OpenAlex_Crossref_Webscrape_NBER_Repec_AEA_Merged.csv, only take doi_full and openalex_raw_author_names.
    selected = select_openalex_raw_author_columns(data)

    # Step 2. Convert openalex_raw_author_names to ASCII.
    selected[OPENALEX_RAW_AUTHOR_COLUMN] = selected[OPENALEX_RAW_AUTHOR_COLUMN].apply(to_ascii)

    # Step 3. Splits openalex_raw_author_names by semicolon ";", so one paper with multiple authors becomes multiple author rows.
    author_rows = split_openalex_raw_authors(selected)

    # Step 4. Drop ", Jr." in openalex_raw_author_names, also drop "Jr." in openalex_raw_author_names.
    author_rows[OPENALEX_RAW_AUTHOR_COLUMN] = author_rows[OPENALEX_RAW_AUTHOR_COLUMN].apply(
        drop_jr
    )

    # Step 5. If openalex_raw_author_names has "," in it, it means openalex_raw_author_names stores last name first, and then first name, you need to reverse it.
    author_rows[OPENALEX_RAW_AUTHOR_COLUMN] = author_rows[OPENALEX_RAW_AUTHOR_COLUMN].apply(
        reverse_comma_name
    )

    # Step 6. For these two (ALVAREZ F. E., ALVAREZ F), it should be F. E. ALVAREZ; Jomo K S should be Jomo KS; Amit G should be G Amit, LIPPI F should be F LIPPI.
    author_rows[OPENALEX_RAW_AUTHOR_COLUMN] = author_rows[OPENALEX_RAW_AUTHOR_COLUMN].apply(
        apply_manual_author_corrections
    )

    # Step 7. Split first name and last name, also calculate the length of first and last name.
    author_rows = add_first_last_name_columns(author_rows)

    # Step 8. If openalex_raw_author_first_name has only 1 single letter, then take the second sequence in openalex_raw_author_names. If the second sequence is not equal to last name, and the second sequence is not single letter, and take the second sequence as the first name. Otherwise keep the original single letter as openalex_raw_author_first_name.
    author_rows["openalex_raw_author_first_name"] = author_rows.apply(
        lambda row: replace_single_letter_first_name(
            row[OPENALEX_RAW_AUTHOR_COLUMN],
            row["openalex_raw_author_first_name"],
            row["openalex_raw_author_last_name"],
        ),
        axis=1,
    )
    author_rows["openalex_raw_author_first_name_length"] = author_rows[
        "openalex_raw_author_first_name"
    ].str.len()

    # Step 9. If openalex_raw_author_last_name has only 1 single letter, then take the second to the last sequence in openalex_raw_author_names. If the second to the last sequence is not equal to first name, and the second to the last sequence is not single letter, and take the second to the last sequence as the last name. Otherwise keep the original single letter as openalex_raw_author_last_name.
    author_rows["openalex_raw_author_last_name"] = author_rows.apply(
        lambda row: replace_single_letter_last_name(
            row[OPENALEX_RAW_AUTHOR_COLUMN],
            row["openalex_raw_author_first_name"],
            row["openalex_raw_author_last_name"],
        ),
        axis=1,
    )
    author_rows["openalex_raw_author_last_name_length"] = author_rows[
        "openalex_raw_author_last_name"
    ].str.len()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    author_rows.to_csv(OUTPUT_AUTHOR_CSV, index=False)

    print("OpenAlex raw author-name cleaning summary:")
    print(f"  Input CSV: {INPUT_CSV}")
    print(f"  Input rows: {len(data)}")
    print(f"  Selected columns: doi_full, openalex_raw_author_names")
    print(f"  Rows with nonblank openalex_raw_author_names: {count_nonblank(selected, OPENALEX_RAW_AUTHOR_COLUMN)}")
    print(f"  Author-level output CSV: {OUTPUT_AUTHOR_CSV}")
    print(f"  Author-level rows: {len(author_rows)}")


def select_openalex_raw_author_columns(data: pd.DataFrame) -> pd.DataFrame:
    missing_columns = [
        column for column in ["doi_full", OPENALEX_RAW_AUTHOR_COLUMN] if column not in data.columns
    ]
    if missing_columns:
        raise ValueError(f"Missing columns: {missing_columns}")

    return data[["doi_full", OPENALEX_RAW_AUTHOR_COLUMN]].copy()


def split_openalex_raw_authors(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, record in data.iterrows():
        authors = split_semicolon_cell(record.get(OPENALEX_RAW_AUTHOR_COLUMN, ""))
        for author in authors:
            rows.append(
                {
                    "doi_full": record.get("doi_full", ""),
                    OPENALEX_RAW_AUTHOR_COLUMN: author,
                }
            )

    return pd.DataFrame(rows, columns=["doi_full", OPENALEX_RAW_AUTHOR_COLUMN])


def split_semicolon_cell(value: object) -> list[str]:
    text = clean_text(value)
    if not text:
        return []
    return [part.strip() for part in text.split(";") if part.strip()]


def drop_jr(name: object) -> str:
    text = clean_text(name)
    text = re.sub(r",\s*Jr\.?", "", text)
    text = re.sub(r"\bJr\.?", "", text)
    return re.sub(r"\s+", " ", text).strip()


def reverse_comma_name(name: object) -> str:
    text = clean_text(name)
    if "," not in text:
        return text

    last_name, first_name = text.split(",", 1)
    first_name = clean_text(first_name)
    last_name = clean_text(last_name)
    if first_name and last_name:
        return f"{first_name} {last_name}"
    return clean_text(f"{first_name} {last_name}")


def apply_manual_author_corrections(name: object) -> str:
    text = clean_text(name)
    corrections = {
        "ALVAREZ F. E.": "F. E. ALVAREZ",
        "ALVAREZ F": "F. E. ALVAREZ",
        "Jomo K S": "Jomo KS",
        "Amit G": "G Amit",
        "LIPPI F": "F LIPPI",
    }
    return corrections.get(text, text)


def add_first_last_name_columns(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    names = data[OPENALEX_RAW_AUTHOR_COLUMN].apply(split_first_last_name)
    data["openalex_raw_author_first_name"] = names.apply(lambda name: name[0])
    data["openalex_raw_author_first_name_length"] = data["openalex_raw_author_first_name"].str.len()
    data["openalex_raw_author_last_name"] = names.apply(lambda name: name[1])
    data["openalex_raw_author_last_name_length"] = data["openalex_raw_author_last_name"].str.len()
    return data


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


def clean_name_part(name: object) -> str:
    text = clean_text(name)
    text = text.replace(".", "")
    text = text.replace(",", "")
    return re.sub(r"\s+", " ", text).strip()


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
