from pathlib import Path
import html
import os
import re
import unicodedata

import pandas as pd


os.chdir("/Users/yingyan_zhao/Dropbox/JournalPublicationProject")

INPUT_CSV = Path("data/processed/OpenAlex_Crossref_Webscrape_NBER_Repec_AEA_Merged.csv")
OUTPUT_DIR = Path("data/processed/author_names")
OUTPUT_AUTHOR_CSV = OUTPUT_DIR / "JEL_Training_Data_OpenAlex_AuthorRows.csv"

OPENALEX_AUTHOR_COLUMN = "openalex_authors"

## #########################################################################
# In this file, I am going to clean openalex_authors with the following steps

# Step 1. from OpenAlex_Crossref_Webscrape_NBER_Repec_AEA_Merged.csv, only take doi_full and openalex_authors.
# Step 2. Convert openalex_authors to ASCII.
# Step 3. Splits openalex_authors by semicolon “;”, so one paper with multiple authors becomes multiple author rows.
# Step 4. If openalex_authors has “,” in it, it means openalex_authors stores last name first, and then first name, you need to reverse it
# Step 5. For these two (ALVAREZ F. E., ALVAREZ F) , it should be F. E. ALVAREZ ; for Li Guo, it should be Guo Li; LIPPI F should be F LIPPI; K S Jomo should be Jomo KS; Professor Wenhao Li should be Wenhao Li
# Step 6. Split first name and last name, also calculate the length of  first and last name
# Step 7. If openalex_author_first_name has only 1 single letter, then take the second sequence in openalex_authors. If the second sequence is not equal to last name, and the second sequence is not single letter, then take the second sequence as the first name. Otherwise keep the original single letter as openalex_author_first_name.
## #########################################################################


def main() -> None:
    data = pd.read_csv(INPUT_CSV, dtype=str).fillna("")

    # Step 1. from OpenAlex_Crossref_Webscrape_NBER_Repec_AEA_Merged.csv, only take doi_full and openalex_authors.
    selected = select_openalex_author_columns(data)

    # Step 2. Convert openalex_authors to ASCII.
    selected[OPENALEX_AUTHOR_COLUMN] = selected[OPENALEX_AUTHOR_COLUMN].apply(to_ascii)

    # Step 3. Splits openalex_authors by semicolon ";", so one paper with multiple authors becomes multiple author rows.
    author_rows = split_openalex_authors(selected)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    author_rows.to_csv(OUTPUT_AUTHOR_CSV, index=False)

    print("OpenAlex author-name cleaning summary:")
    print(f"  Input CSV: {INPUT_CSV}")
    print(f"  Input rows: {len(data)}")
    print(f"  Selected columns: doi_full, openalex_authors")
    print(f"  Rows with nonblank openalex_authors: {count_nonblank(selected, OPENALEX_AUTHOR_COLUMN)}")
    print(f"  Author-level output CSV: {OUTPUT_AUTHOR_CSV}")
    print(f"  Author-level rows: {len(author_rows)}")


def select_openalex_author_columns(data: pd.DataFrame) -> pd.DataFrame:
    missing_columns = [
        column for column in ["doi_full", OPENALEX_AUTHOR_COLUMN] if column not in data.columns
    ]
    if missing_columns:
        raise ValueError(f"Missing columns: {missing_columns}")

    return data[["doi_full", OPENALEX_AUTHOR_COLUMN]].copy()


def split_openalex_authors(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, record in data.iterrows():
        authors = split_semicolon_cell(record.get(OPENALEX_AUTHOR_COLUMN, ""))
        for author in authors:
            # Step 4. If openalex_authors has "," in it, it means openalex_authors stores last name first, and then first name, you need to reverse it.
            author = reverse_comma_name(author)

            # Step 5. For these two (ALVAREZ F. E., ALVAREZ F), it should be F. E. ALVAREZ; for Li Guo, it should be Guo Li; LIPPI F should be F LIPPI; K S Jomo should be Jomo KS.
            author = apply_manual_author_corrections(author)

            # Step 6. Split first name and last name, also calculate the length of first and last name.
            first_name, last_name = split_first_last_name(author)

            # Step 7. If openalex_author_first_name has only 1 single letter, then take the second sequence in openalex_authors. If the second sequence is not equal to last name, and the second sequence is not single letter, and take the second sequence as the first name. Otherwise keep the original single letter as openalex_author_first_name.
            first_name = replace_single_letter_first_name(author, first_name, last_name)
            rows.append(
                {
                    "doi_full": record.get("doi_full", ""),
                    OPENALEX_AUTHOR_COLUMN: author,
                    "openalex_author_first_name": first_name,
                    "openalex_author_first_name_length": len(first_name),
                    "openalex_author_last_name": last_name,
                    "openalex_author_last_name_length": len(last_name),
                }
            )

    return pd.DataFrame(
        rows,
        columns=[
            "doi_full",
            OPENALEX_AUTHOR_COLUMN,
            "openalex_author_first_name",
            "openalex_author_first_name_length",
            "openalex_author_last_name",
            "openalex_author_last_name_length",
        ],
    )


def split_semicolon_cell(value: object) -> list[str]:
    text = clean_text(value)
    if not text:
        return []
    return [part.strip() for part in text.split(";") if part.strip()]


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
        "Li Guo": "Guo Li",
        "LIPPI F": "F LIPPI",
        "K S Jomo": "Jomo KS",
    }
    return corrections.get(text, text)


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
