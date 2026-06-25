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
    "keywords_openalex",
    "keywords",
    "jel_codes",
    "jel_code_1",
    "jel_code_full",
    "openalex_authors",
    "openalex_author_ids",
    "openalex_raw_author_names",
    "openalex_author_institutions",
    "openalex_author_institution_ids",
    "openalex_raw_affiliation_strings",
    "openalex_institutions",
    "openalex_primary_topic",
    "openalex_keywords",
    "openalex_concepts",
    "crossref_authors",
    "crossref_author_given",
    "crossref_author_family",
    "crossref_author_affiliations",
    "scrape_authors",
    "scrape_author_institutions",
    "nber_authors",
    "nber_author_institutions",
    "aea_authors",
    "aea_author_institutions",
    "aea_author_institution_pairs",
]

DROP_AFTER_CLEANING = [
    "doi_list",
    "doi_1",
    "doi_2",
    "doi_3",
    "aea_doi",
    "scrape_publication_year",
    "aea_publication_year",
    "aea_title_duplicate_tag",
    "aea_match_strategy",
    "duplicate_doi_tag",
    "duplicate_title_tag",
    "aea_abstract",
    "aea_jel_codes",
    "aea_jel_descriptions",
    "jel_context",
    "scrape_keywords",
    "aea_keywords",
]


def main() -> None:
    data = read_data(INPUT_CSV)
    data = clean_data(data)
    training_data = keep_columns(data, TRAINING_COLUMNS)
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
    print("  Final columns:")
    for column in training_data.columns:
        print(f"    {column}")


def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    cleaned = data.copy()
    if "doi" not in cleaned.columns:
        cleaned["doi"] = get_column(cleaned, "doi_full")
    cleaned["journalname"] = prefer_nonblank_source_column(
        cleaned,
        target_column="journalname",
        source_column="aea_journal",
    )
    cleaned["title"] = prefer_nonblank_source_column(
        cleaned,
        target_column="title",
        source_column="aea_title",
    )
    cleaned["publication_year"] = coalesce_columns(
        cleaned,
        ["scrape_publication_year", "aea_publication_year", "publication_year"],
        clean_year,
    )
    cleaned["abstract"] = prefer_nonblank_source_column(
        cleaned,
        target_column="abstract",
        source_column="aea_abstract",
    )
    cleaned["jel_codes"] = prefer_nonblank_source_column(
        cleaned,
        target_column="jel_codes",
        source_column="aea_jel_codes",
    )
    cleaned["jel_code_1"] = cleaned["jel_codes"].apply(first_jel_code_letter)
    cleaned["jel_code_full"] = cleaned["jel_codes"].apply(all_jel_code_letters)
    cleaned["keywords"] = coalesce_columns(
        cleaned,
        ["aea_keywords", "scrape_keywords", "keywords"],
        clean_text,
    )
    cleaned["keywords_openalex"] = cleaned.apply(keywords_openalex_from_row, axis=1)
    if "aea_journal" in cleaned.columns:
        cleaned = cleaned.drop(columns=["aea_journal"])
    if "aea_title" in cleaned.columns:
        cleaned = cleaned.drop(columns=["aea_title"])
    cleaned = drop_columns(cleaned, DROP_AFTER_CLEANING)
    return cleaned


def read_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist.")

    return pd.read_csv(path, dtype=str, keep_default_na=False)


def keep_columns(data: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    cleaned = data.copy()
    for column in columns:
        if column not in cleaned.columns:
            cleaned[column] = ""
    return cleaned[columns].copy()


def prefer_nonblank_source_column(
    data: pd.DataFrame,
    target_column: str,
    source_column: str,
) -> pd.Series:
    target = get_column(data, target_column).fillna("").astype(str)
    source = get_column(data, source_column).fillna("").astype(str)
    source_nonblank = source.str.strip() != ""
    return target.mask(source_nonblank, source)


def coalesce_columns(data: pd.DataFrame, columns: list[str], cleaner) -> pd.Series:
    values = pd.Series([""] * len(data), index=data.index)
    for column in columns:
        if column not in data.columns:
            continue
        candidate = data[column].apply(cleaner)
        values = values.mask(values == "", candidate)
    return values


def clean_year(value: object) -> str:
    if pd.isna(value):
        return ""
    match = re.search(r"\b(19|20)\d{2}\b", str(value))
    if match:
        return match.group(0)
    return ""


def clean_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return " ".join(str(value).strip().split())


def keywords_openalex_from_row(row: pd.Series) -> str:
    values = []
    seen = set()
    for column in ["openalex_primary_topic", "openalex_keywords", "openalex_concepts"]:
        for value in split_keyword_values(row.get(column, "")):
            cleaned_value = clean_openalex_keyword(value)
            if cleaned_value and cleaned_value not in seen:
                values.append(cleaned_value)
                seen.add(cleaned_value)
    return "; ".join(values)


def split_keyword_values(value: object) -> list[str]:
    text = clean_text(value)
    if not text:
        return []
    return [
        clean_text(piece)
        for piece in re.split(r";|,", text)
        if clean_text(piece)
    ]


def clean_openalex_keyword(value: object) -> str:
    text = clean_text(value)
    text = re.sub(r"\(\s*\d+(?:\.\d+)?\s*\)", " ", text)
    text = text.replace("(", " ").replace(")", " ")
    return clean_text(text)


def get_column(data: pd.DataFrame, column: str) -> pd.Series:
    if column in data.columns:
        return data[column]
    return pd.Series([""] * len(data), index=data.index)


def drop_columns(data: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    columns_to_drop = [column for column in columns if column in data.columns]
    return data.drop(columns=columns_to_drop).copy()


def keep_rows_with_jel_codes(data: pd.DataFrame) -> pd.DataFrame:
    jel_codes = data["jel_codes"].fillna("").astype(str).str.strip()
    return data.loc[jel_codes != ""].copy()


def keep_rows_without_jel_codes(data: pd.DataFrame) -> pd.DataFrame:
    jel_codes = data["jel_codes"].fillna("").astype(str).str.strip()
    return data.loc[jel_codes == ""].copy()


def first_jel_code_letter(value: object) -> str:
    if pd.isna(value):
        return ""
    match = re.search(r"\b([A-Z])\d{0,2}\b", str(value).upper())
    if match:
        return match.group(1)
    return ""


def all_jel_code_letters(value: object) -> str:
    if pd.isna(value):
        return ""
    letters = []
    seen = set()
    for match in re.finditer(r"\b([A-Z])\d{0,2}\b", str(value).upper()):
        letter = match.group(1)
        if letter not in seen:
            letters.append(letter)
            seen.add(letter)
    return "; ".join(letters)


def count_nonblank(data: pd.DataFrame, column: str) -> int:
    return int(data[column].fillna("").astype(str).str.strip().ne("").sum())


def count_blank(data: pd.DataFrame, column: str) -> int:
    return int(data[column].fillna("").astype(str).str.strip().eq("").sum())


if __name__ == "__main__":
    main()
