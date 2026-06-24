from pathlib import Path
import os
import re

import pandas as pd


os.chdir("/Users/yingyan_zhao/Dropbox/JournalPublicationProject")

INPUT_CSV = Path("data/processed/OpenAlex_Crossref_Webscrape_NBER_Repec_AEA_Merged.csv")
OUTPUT_CSV = Path("data/processed/JEL_Training_Data.csv")
OUTPUT_WITH_JEL_CSV = Path("data/processed/JEL_Training_Data_With_JEL.csv")
OUTPUT_WITHOUT_JEL_CSV = Path("data/processed/JEL_Training_Data_Without_JEL.csv")

TRAINING_COLUMNS = [
    "doi",
    "title",
    "journalname",
    "publication_year",
    "abstract",
    "openalex_primary_domain",
    "openalex_primary_field",
    "openalex_primary_subfield",
    "openalex_primary_topic",
    "openalex_keywords",
    "openalex_top3_keywords",
    "openalex_concepts",
    "openalex_top3_concepts",
    "openalex_level0_concepts",
    "scrape_keywords",
    "jel_context",
    "nber_keywords",
    "repec_keywords",
    "aea_keywords",
    "aea_jel_descriptions",
    "jel_codes",
]


def main() -> None:
    data = read_data(INPUT_CSV)
    training_data = keep_columns(data, TRAINING_COLUMNS)
    training_data["jel_codes"] = training_data["jel_codes"].apply(clean_jel_codes_to_letters)
    training_data["jel_codes_1"] = training_data["jel_codes"].apply(first_jel_letter)
    with_jel = keep_rows_with_jel_codes(training_data)
    without_jel = keep_rows_without_jel_codes(training_data)

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    training_data.to_csv(OUTPUT_CSV, index=False)
    with_jel.to_csv(OUTPUT_WITH_JEL_CSV, index=False)
    without_jel.to_csv(OUTPUT_WITHOUT_JEL_CSV, index=False)

    print("JEL training data summary:")
    print(f"  Input CSV: {INPUT_CSV}")
    print(f"  Output CSV: {OUTPUT_CSV}")
    print(f"  With-JEL output CSV: {OUTPUT_WITH_JEL_CSV}")
    print(f"  Without-JEL output CSV: {OUTPUT_WITHOUT_JEL_CSV}")
    print(f"  Rows: {len(training_data)}")
    print(f"  Columns: {list(training_data.columns)}")
    print(f"  Rows with JEL codes: {len(with_jel)}")
    print(f"  Rows without JEL codes: {len(without_jel)}")


def read_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist.")

    return pd.read_csv(path, dtype=str, keep_default_na=False)


def keep_columns(data: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    missing_columns = [
        column for column in columns
        if column not in data.columns
    ]
    if missing_columns:
        raise ValueError(f"Missing columns: {missing_columns}")

    return data[columns].copy()


def keep_rows_with_jel_codes(data: pd.DataFrame) -> pd.DataFrame:
    jel_codes = data["jel_codes"].fillna("").astype(str).str.strip()
    return data.loc[jel_codes != ""].copy()


def keep_rows_without_jel_codes(data: pd.DataFrame) -> pd.DataFrame:
    jel_codes = data["jel_codes"].fillna("").astype(str).str.strip()
    return data.loc[jel_codes == ""].copy()


def clean_jel_codes_to_letters(value: object) -> str:
    if pd.isna(value):
        return ""

    text = str(value).upper()
    codes = re.findall(r"\b[A-Z]\d{0,2}\b", text)

    letters = []
    seen_letters = set()
    for code in codes:
        letter = code[0]
        if letter not in seen_letters:
            letters.append(letter)
            seen_letters.add(letter)

    return "; ".join(letters)


def first_jel_letter(value: object) -> str:
    if pd.isna(value):
        return ""

    text = str(value).strip()
    if not text:
        return ""

    return text.split(";")[0].strip()


def count_nonblank(data: pd.DataFrame, column: str) -> int:
    return int(data[column].fillna("").astype(str).str.strip().ne("").sum())


def count_blank(data: pd.DataFrame, column: str) -> int:
    return int(data[column].fillna("").astype(str).str.strip().eq("").sum())


if __name__ == "__main__":
    main()
