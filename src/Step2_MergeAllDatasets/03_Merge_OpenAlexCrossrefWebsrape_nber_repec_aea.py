from pathlib import Path
import os
import re

import pandas as pd


os.chdir("/Users/yingyan_zhao/Dropbox/JournalPublicationProject")

AEA_INPUT_CSV = Path("data/raw_csv/AEA_Journals_Papers.csv")
BASE_INPUT_CSV = Path("data/processed/OpenAlex_Crossref_Webscrape_NBER_Repec_Merged.csv")
MERGED_OUTPUT_CSV = Path("data/processed/OpenAlex_Crossref_Webscrape_NBER_Repec_AEA_Merged.csv")
AEA_COLUMNS_TO_DROP_AFTER_MERGE = [
    "aea_doi",
    "aea_title",
    "aea_journal",
    "aea_publication_year",
    "aea_abstract",
    "aea_jel_codes",
    "aea_match_strategy",
]

AEA_COLUMNS_TO_DROP = [
    "journal_slug",
    "publication_month",
    "volume",
    "issue",
    "pages",
    "article_url",
    "issue_url",
    "issue_label",
    "collection_date",
    "scraped_at",
    "http_status",
    "error",
]


def main() -> None:
    aea = read_aea_data(AEA_INPUT_CSV)
    cleaned = clean_aea_data(aea)
    base = read_base_data(BASE_INPUT_CSV)
    merged, merge_summary = merge_base_with_aea(base, cleaned)
    merged.to_csv(MERGED_OUTPUT_CSV, index=False)

    print("AEA cleaning summary:")
    print(f"  Input CSV: {AEA_INPUT_CSV}")
    print(f"  Input rows: {len(aea)}")
    print(f"  Output rows: {len(cleaned)}")
    print(f"  Dropped blank-author rows: {len(aea) - len(cleaned)}")
    print(f"  Columns: {list(cleaned.columns)}")
    print()
    print("OpenAlex/Crossref/Webscrape/NBER/RePEc + AEA merge summary:")
    print(f"  Base input CSV: {BASE_INPUT_CSV}")
    print(f"  Base rows: {len(base)}")
    print(f"  Cleaned AEA rows: {len(cleaned)}")
    print(f"  Output CSV: {MERGED_OUTPUT_CSV}")
    print(f"  Output rows: {len(merged)}")
    print(f"  Matched by DOI: {merge_summary['matched_by_doi']}")
    print(f"  Matched by title: {merge_summary['matched_by_title']}")
    print(f"  Base-only rows: {merge_summary['base_only']}")
    print(f"  AEA-only rows: {merge_summary['aea_only']}")
    print(f"  Duplicate base DOI rows: {merge_summary['duplicate_base_doi_rows']}")
    print(f"  Duplicate AEA DOI rows: {merge_summary['duplicate_aea_doi_rows']}")
    print(f"  Duplicate base title rows: {merge_summary['duplicate_base_title_rows']}")
    print(f"  Duplicate AEA title rows: {merge_summary['duplicate_aea_title_rows']}")


def read_aea_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist.")

    return pd.read_csv(path, dtype=str, keep_default_na=False)


def read_base_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist.")

    return pd.read_csv(path, dtype=str, keep_default_na=False)


def clean_aea_data(data: pd.DataFrame) -> pd.DataFrame:
    cleaned = data.copy()
    cleaned = drop_columns(cleaned, AEA_COLUMNS_TO_DROP)
    cleaned = drop_blank_authors(cleaned)
    cleaned = add_prefix_to_columns(cleaned, "aea_")
    return cleaned.reset_index(drop=True)


def merge_base_with_aea(
    base: pd.DataFrame,
    aea: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, int]]:
    base_prepared = base.copy()
    aea_prepared = aea.copy()

    base_prepared["_base_row_id"] = range(len(base_prepared))
    aea_prepared["_aea_row_id"] = range(len(aea_prepared))

    base_prepared["base_merge_doi"] = get_column(base_prepared, "doi").apply(normalize_doi)
    base_prepared["base_merge_title"] = get_column(base_prepared, "title").apply(normalize_title)
    aea_prepared["aea_merge_doi"] = get_column(aea_prepared, "aea_doi").apply(normalize_doi)
    aea_prepared["aea_merge_title"] = get_column(aea_prepared, "aea_title").apply(normalize_title)

    summary = {
        "matched_by_doi": 0,
        "matched_by_title": 0,
        "base_only": 0,
        "aea_only": 0,
        "duplicate_base_doi_rows": duplicate_key_rows(base_prepared, "base_merge_doi"),
        "duplicate_aea_doi_rows": duplicate_key_rows(aea_prepared, "aea_merge_doi"),
        "duplicate_base_title_rows": duplicate_key_rows(base_prepared, "base_merge_title"),
        "duplicate_aea_title_rows": duplicate_key_rows(aea_prepared, "aea_merge_title"),
    }

    matched_frames = []
    matched, base_remaining, aea_remaining = match_unique_stage(
        base_prepared,
        aea_prepared,
        base_key_column="base_merge_doi",
        aea_key_column="aea_merge_doi",
        strategy="doi",
    )
    summary["matched_by_doi"] = len(matched)
    matched_frames.append(matched)

    matched, base_remaining, aea_remaining = match_unique_stage(
        base_remaining,
        aea_remaining,
        base_key_column="base_merge_title",
        aea_key_column="aea_merge_title",
        strategy="title",
    )
    summary["matched_by_title"] = len(matched)
    matched_frames.append(matched)

    base_only = add_empty_aea_columns(base_remaining, aea_prepared)
    base_only["aea_match_strategy"] = "base_only"
    summary["base_only"] = len(base_only)
    matched_frames.append(base_only)

    aea_only = add_empty_base_columns(aea_remaining, base_prepared)
    aea_only["aea_match_strategy"] = "aea_only"
    summary["aea_only"] = len(aea_only)
    matched_frames.append(aea_only)

    merged = pd.concat(matched_frames, ignore_index=True, sort=False)
    merged = sort_outer_merged_rows(merged)
    merged["doi"] = fill_blank_from_column(
        merged,
        target_column="doi",
        source_column="aea_doi",
    )
    merged["title"] = fill_blank_from_column(
        merged,
        target_column="title",
        source_column="aea_title",
    )
    merged["journalname"] = fill_blank_from_column(
        merged,
        target_column="journalname",
        source_column="aea_journal",
    )
    merged["publication_year"] = fill_blank_from_column(
        merged,
        target_column="publication_year",
        source_column="aea_publication_year",
    )
    merged["abstract"] = fill_blank_from_column(
        merged,
        target_column="abstract",
        source_column="aea_abstract",
    )
    merged["jel_codes"] = fill_blank_from_column(
        merged,
        target_column="jel_codes",
        source_column="aea_jel_codes",
    )
    merged = drop_columns(merged, AEA_COLUMNS_TO_DROP_AFTER_MERGE)
    merged = drop_merge_helper_columns(merged)
    return merged.reset_index(drop=True), summary


def match_unique_stage(
    base_unmatched: pd.DataFrame,
    aea_unmatched: pd.DataFrame,
    base_key_column: str,
    aea_key_column: str,
    strategy: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    base_unique = unique_key_rows(base_unmatched, base_key_column)
    aea_unique = unique_key_rows(aea_unmatched, aea_key_column)

    if base_unique.empty or aea_unique.empty:
        return pd.DataFrame(), base_unmatched, aea_unmatched

    matched = base_unique.merge(
        aea_unique,
        left_on=base_key_column,
        right_on=aea_key_column,
        how="inner",
    )
    if matched.empty:
        return matched, base_unmatched, aea_unmatched

    matched["aea_match_strategy"] = strategy
    matched_base_ids = set(matched["_base_row_id"])
    matched_aea_ids = set(matched["_aea_row_id"])

    base_remaining = base_unmatched.loc[
        ~base_unmatched["_base_row_id"].isin(matched_base_ids)
    ].copy()
    aea_remaining = aea_unmatched.loc[
        ~aea_unmatched["_aea_row_id"].isin(matched_aea_ids)
    ].copy()
    return matched, base_remaining, aea_remaining


def unique_key_rows(data: pd.DataFrame, key_column: str) -> pd.DataFrame:
    key = data[key_column].fillna("").astype(str)
    unique_key = (key != "") & ~key.duplicated(keep=False)
    return data.loc[unique_key].copy()


def add_empty_aea_columns(base_rows: pd.DataFrame, aea_columns_source: pd.DataFrame) -> pd.DataFrame:
    rows = base_rows.copy()
    for column in aea_columns_source.columns:
        if column not in rows.columns:
            rows[column] = ""
    return rows


def add_empty_base_columns(aea_rows: pd.DataFrame, base_columns_source: pd.DataFrame) -> pd.DataFrame:
    rows = aea_rows.copy()
    for column in base_columns_source.columns:
        if column not in rows.columns:
            rows[column] = ""
    return rows


def sort_outer_merged_rows(data: pd.DataFrame) -> pd.DataFrame:
    sorted_data = data.copy()
    sorted_data["_sort_base"] = pd.to_numeric(sorted_data["_base_row_id"], errors="coerce")
    sorted_data["_sort_aea"] = pd.to_numeric(sorted_data["_aea_row_id"], errors="coerce")
    sorted_data["_sort_base"] = sorted_data["_sort_base"].fillna(10**12)
    sorted_data["_sort_aea"] = sorted_data["_sort_aea"].fillna(10**12)
    return sorted_data.sort_values(["_sort_base", "_sort_aea"])


def drop_merge_helper_columns(data: pd.DataFrame) -> pd.DataFrame:
    helper_columns = [
        "_base_row_id",
        "_aea_row_id",
        "base_merge_doi",
        "base_merge_title",
        "aea_merge_doi",
        "aea_merge_title",
        "_sort_base",
        "_sort_aea",
    ]
    return data.drop(columns=[column for column in helper_columns if column in data.columns])


def fill_blank_from_column(
    data: pd.DataFrame,
    target_column: str,
    source_column: str,
) -> pd.Series:
    target = get_column(data, target_column).fillna("").astype(str)
    source = get_column(data, source_column).fillna("").astype(str)
    target_blank = target.str.strip() == ""
    source_nonblank = source.str.strip() != ""
    return target.mask(target_blank & source_nonblank, source)


def normalize_doi(value: object) -> str:
    if pd.isna(value):
        return ""

    text = str(value).strip().lower()
    text = re.sub(r"^https?://(dx\.)?doi\.org/", "", text)
    text = re.sub(r"^doi:\s*", "", text)
    return text.strip().rstrip(".")


def normalize_title(value: object) -> str:
    if pd.isna(value):
        return ""

    text = str(value).lower().strip()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def duplicate_key_rows(data: pd.DataFrame, key_column: str) -> int:
    key = data[key_column].fillna("").astype(str)
    return int(key.loc[key != ""].duplicated(keep=False).sum())


def get_column(data: pd.DataFrame, column: str) -> pd.Series:
    if column in data.columns:
        return data[column]

    return pd.Series([""] * len(data), index=data.index)


def drop_columns(data: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    columns_to_drop = [
        column for column in columns
        if column in data.columns
    ]
    return data.drop(columns=columns_to_drop).copy()


def drop_blank_authors(data: pd.DataFrame) -> pd.DataFrame:
    if "authors" not in data.columns:
        raise ValueError("Missing required column: authors")

    authors = data["authors"].fillna("").astype(str).str.strip()
    return data.loc[authors != ""].copy()


def add_prefix_to_columns(data: pd.DataFrame, prefix: str) -> pd.DataFrame:
    renamed = data.copy()
    renamed.columns = [
        column if column.startswith(prefix) else f"{prefix}{column}"
        for column in renamed.columns
    ]
    return renamed


if __name__ == "__main__":
    main()
