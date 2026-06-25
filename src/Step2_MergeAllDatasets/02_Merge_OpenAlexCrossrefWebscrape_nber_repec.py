from pathlib import Path
import os
import re
from typing import Any

import pandas as pd


os.chdir("/Users/yingyan_zhao/Dropbox/JournalPublicationProject")

BASE_INPUT_CSV = Path("data/processed/OpenAlex_Crossref_Webscrape_NBER_Merged.csv")
MERGED_OUTPUT_CSV = Path("data/processed/OpenAlex_Crossref_Webscrape_NBER_Repec_Merged.csv")
REPEC_INPUT_DIR = Path("data/raw_csv/repec_by_year")
REPEC_OUTPUT_DIR = Path("data/processed/repec_by_year_cleaned")

REPEC_COLUMNS_TO_DROP = [
    "source_file",
    "template_type",
    "handle",
    "year",
    "journal",
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

REPEC_COLUMNS_TO_DROP_AFTER_MERGE = [
    "repec_doi",
    "repec_x_doi",
    "repec_match_strategy",
    "repec_title",
    "repec_classification_jel",
    "repec_abstract",
    "repec_keywords",
]

BASE_ROW_ID = "_base_row_order"
REPEC_ROW_ID = "_repec_row_id"


def main() -> None:
    repec_cleaning_summaries = clean_repec_data_by_year()

    base = read_csv(BASE_INPUT_CSV)
    base = ensure_publication_year(base)
    merged, merge_summaries = merge_base_with_repec_by_year(base)

    MERGED_OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(MERGED_OUTPUT_CSV, index=False)

    print("OpenAlex/Crossref/Webscrape/NBER + RePEc merge summary:")
    print(f"  Base input CSV: {BASE_INPUT_CSV}")
    print(f"  Base rows: {len(base)}")
    print(f"  RePEc raw input dir: {REPEC_INPUT_DIR}")
    print(f"  RePEc cleaned output dir: {REPEC_OUTPUT_DIR}")
    print(f"  RePEc year files cleaned: {len(repec_cleaning_summaries)}")
    print(f"  RePEc cleaned rows: {total_repec_cleaned_rows(repec_cleaning_summaries)}")
    print(f"  RePEc rows dropped for duplicated doi/x_doi: {total_repec_duplicate_doi_drops(repec_cleaning_summaries)}")
    print(f"  Output CSV: {MERGED_OUTPUT_CSV}")
    print(f"  Output rows: {len(merged)}")
    print(f"  Rows matched to RePEc: {total_repec_matches(merge_summaries)}")
    print(f"  Rows matched to RePEc by doi: {total_matched_by_strategy(merge_summaries, 'matched_by_doi')}")
    print(f"  Rows matched to RePEc by x_doi: {total_matched_by_strategy(merge_summaries, 'matched_by_x_doi')}")
    print(f"  Rows matched to RePEc by title: {total_matched_by_strategy(merge_summaries, 'matched_by_title')}")
    print(f"  Abstracts filled from RePEc: {total_filled_from_repec(merge_summaries, 'abstract')}")
    print(f"  JEL codes filled from RePEc: {total_filled_from_repec(merge_summaries, 'jel_codes')}")
    print(f"  Keywords filled from RePEc: {total_filled_from_repec(merge_summaries, 'keywords')}")

    for summary in merge_summaries:
        print(
            f"  {summary['year']}: "
            f"base={summary['base_rows']}, "
            f"repec={summary['repec_rows']}, "
            f"doi={summary['matched_by_doi']}, "
            f"x_doi={summary['matched_by_x_doi']}, "
            f"title={summary['matched_by_title']}, "
            f"unmatched_base={summary['unmatched_base']}, "
            f"abstract_filled={summary['abstracts_filled_from_repec']}, "
            f"jel_filled={summary['jel_codes_filled_from_repec']}, "
            f"keywords_filled={summary['keywords_filled_from_repec']}, "
            f"dropped_repec_duplicate_title_rows_for_title_match="
            f"{summary['dropped_repec_duplicate_title_rows_for_title_match']}"
        )


def clean_repec_data_by_year() -> list[dict[str, Any]]:
    if not REPEC_INPUT_DIR.exists():
        raise FileNotFoundError(f"{REPEC_INPUT_DIR} does not exist.")

    REPEC_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    summaries = []
    for input_file in sorted(REPEC_INPUT_DIR.glob("*.csv")):
        year = year_from_path(input_file)
        output_file = REPEC_OUTPUT_DIR / f"RePEc_ReDIF_{year}_Cleaned.csv"
        summaries.append(clean_repec_year_file(input_file, output_file, year))
    return summaries


def clean_repec_year_file(
    input_file: Path,
    output_file: Path,
    year: str,
) -> dict[str, Any]:
    data = read_csv(input_file)
    cleaned = drop_repec_columns(data)

    if "title" in cleaned.columns:
        cleaned["title"] = cleaned["title"].apply(keep_letters_and_numbers)
    cleaned = rename_specific_repec_titles(cleaned, year)

    duplicate_records = duplicate_doi_records(cleaned)
    cleaned = cleaned.loc[~cleaned.index.isin(duplicate_records.index)].copy()

    cleaned = add_prefix_to_columns(cleaned, "repec_")
    cleaned.to_csv(output_file, index=False)

    return {
        "year": year,
        "input_rows": len(data),
        "output_rows": len(cleaned),
        "dropped_duplicate_doi_rows": len(duplicate_records),
        "output_file": output_file,
    }


def merge_base_with_repec_by_year(
    base: pd.DataFrame,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    base_for_merge = base.copy()
    base_for_merge[BASE_ROW_ID] = range(len(base_for_merge))

    merged_years = []
    summaries = []

    for year, base_year in base_for_merge.groupby("publication_year", sort=True):
        repec_file = REPEC_OUTPUT_DIR / f"RePEc_ReDIF_{year}_Cleaned.csv"
        if repec_file.exists():
            repec = read_csv(repec_file)
        else:
            repec = pd.DataFrame()

        merged_year, summary = merge_one_year(base_year.copy(), repec, str(year))
        merged_years.append(merged_year)
        summaries.append(summary)

    merged = pd.concat(merged_years, ignore_index=True, sort=False)
    merged = merged.sort_values(BASE_ROW_ID, kind="mergesort")
    merged = merged.drop(columns=[BASE_ROW_ID])
    return merged.reset_index(drop=True), summaries


def merge_one_year(
    base_year: pd.DataFrame,
    repec: pd.DataFrame,
    year: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    repec = prepare_repec_for_merge(repec)
    base_year = prepare_base_for_merge(base_year)

    summary = {
        "year": year,
        "base_rows": len(base_year),
        "repec_rows": len(repec),
        "matched_by_doi": 0,
        "matched_by_x_doi": 0,
        "matched_by_title": 0,
        "unmatched_base": 0,
        "dropped_repec_duplicate_title_rows_for_title_match": 0,
        "abstracts_filled_from_repec": 0,
        "jel_codes_filled_from_repec": 0,
        "keywords_filled_from_repec": 0,
    }

    matched_frames = []
    remaining_base = base_year.copy()
    remaining_repec = repec.copy()

    matched, remaining_base, remaining_repec = match_by_repec_doi_column(
        base_unmatched=remaining_base,
        repec_pool=remaining_repec,
        repec_key_column="repec_merge_doi",
        strategy="doi",
    )
    summary["matched_by_doi"] = len(matched)
    matched_frames.append(matched)

    matched, remaining_base, remaining_repec = match_by_repec_doi_column(
        base_unmatched=remaining_base,
        repec_pool=remaining_repec,
        repec_key_column="repec_merge_x_doi",
        strategy="x_doi",
    )
    summary["matched_by_x_doi"] = len(matched)
    matched_frames.append(matched)

    title_repec, dropped_duplicate_title_rows = keep_unique_repec_titles(remaining_repec)
    summary["dropped_repec_duplicate_title_rows_for_title_match"] = dropped_duplicate_title_rows
    matched, remaining_base = match_by_title(
        base_unmatched=remaining_base,
        repec_pool=title_repec,
    )
    summary["matched_by_title"] = len(matched)
    matched_frames.append(matched)

    remaining_base = remaining_base.copy()
    remaining_base["repec_match_strategy"] = ""
    summary["unmatched_base"] = len(remaining_base)
    matched_frames.append(remaining_base)

    merged = pd.concat([frame for frame in matched_frames if not frame.empty], ignore_index=True, sort=False)
    merged = merged.sort_values(BASE_ROW_ID, kind="mergesort")
    merged = clean_after_repec_merge(merged)
    filled_counts = merged.attrs.get("repec_filled_counts", {})
    summary["abstracts_filled_from_repec"] = filled_counts.get("abstract", 0)
    summary["jel_codes_filled_from_repec"] = filled_counts.get("jel_codes", 0)
    summary["keywords_filled_from_repec"] = filled_counts.get("keywords", 0)
    return merged, summary


def prepare_base_for_merge(base: pd.DataFrame) -> pd.DataFrame:
    prepared = base.copy()
    prepared["base_merge_title"] = prepared["title"].apply(normalize_title)
    return prepared


def prepare_repec_for_merge(repec: pd.DataFrame) -> pd.DataFrame:
    prepared = repec.copy()
    if prepared.empty:
        for column in [
            REPEC_ROW_ID,
            "repec_merge_doi",
            "repec_merge_x_doi",
            "repec_merge_title",
        ]:
            prepared[column] = ""
        return prepared

    prepared[REPEC_ROW_ID] = range(len(prepared))
    prepared["repec_merge_doi"] = get_column(prepared, "repec_doi").apply(normalize_doi)
    prepared["repec_merge_x_doi"] = get_column(prepared, "repec_x_doi").apply(normalize_doi)
    prepared["repec_merge_title"] = get_column(prepared, "repec_title").apply(normalize_title)
    return prepared


def match_by_repec_doi_column(
    base_unmatched: pd.DataFrame,
    repec_pool: pd.DataFrame,
    repec_key_column: str,
    strategy: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if base_unmatched.empty or repec_pool.empty:
        return pd.DataFrame(), base_unmatched, repec_pool

    base_doi_keys = base_doi_key_rows(base_unmatched)
    repec_lookup = unique_repec_lookup(repec_pool, repec_key_column)
    if base_doi_keys.empty or repec_lookup.empty:
        return pd.DataFrame(), base_unmatched, repec_pool

    candidates = base_doi_keys.merge(
        repec_lookup[[REPEC_ROW_ID, repec_key_column]],
        left_on="base_merge_doi",
        right_on=repec_key_column,
        how="inner",
    )
    if candidates.empty:
        return pd.DataFrame(), base_unmatched, repec_pool

    candidates = candidates.sort_values(
        [BASE_ROW_ID, "base_doi_order", REPEC_ROW_ID],
        kind="mergesort",
    ).drop_duplicates(subset=[BASE_ROW_ID], keep="first")

    matched = base_unmatched.merge(candidates[[BASE_ROW_ID, REPEC_ROW_ID]], on=BASE_ROW_ID, how="inner")
    matched = matched.merge(repec_pool, on=REPEC_ROW_ID, how="left")
    matched["repec_match_strategy"] = strategy

    matched_base_ids = set(candidates[BASE_ROW_ID])
    matched_repec_ids = set(candidates[REPEC_ROW_ID])
    remaining_base = base_unmatched.loc[~base_unmatched[BASE_ROW_ID].isin(matched_base_ids)].copy()
    remaining_repec = repec_pool.loc[~repec_pool[REPEC_ROW_ID].isin(matched_repec_ids)].copy()
    return matched, remaining_base, remaining_repec


def match_by_title(
    base_unmatched: pd.DataFrame,
    repec_pool: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if base_unmatched.empty or repec_pool.empty:
        return pd.DataFrame(), base_unmatched

    base_title = base_unmatched.loc[base_unmatched["base_merge_title"] != ""].copy()
    repec_title = repec_pool.loc[repec_pool["repec_merge_title"] != ""].copy()
    if base_title.empty or repec_title.empty:
        return pd.DataFrame(), base_unmatched

    matched = base_title.merge(
        repec_title,
        left_on="base_merge_title",
        right_on="repec_merge_title",
        how="inner",
    )
    if matched.empty:
        return pd.DataFrame(), base_unmatched

    matched = matched.sort_values([BASE_ROW_ID, REPEC_ROW_ID], kind="mergesort")
    matched = matched.drop_duplicates(subset=[BASE_ROW_ID], keep="first")
    matched["repec_match_strategy"] = "title"

    matched_base_ids = set(matched[BASE_ROW_ID])
    remaining_base = base_unmatched.loc[~base_unmatched[BASE_ROW_ID].isin(matched_base_ids)].copy()
    return matched, remaining_base


def keep_unique_repec_titles(repec: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    if repec.empty or "repec_merge_title" not in repec.columns:
        return repec.copy(), 0

    nonblank_title = repec["repec_merge_title"] != ""
    duplicated_title = nonblank_title & repec["repec_merge_title"].duplicated(keep=False)
    dropped_rows = int(duplicated_title.sum())
    return repec.loc[~duplicated_title].copy(), dropped_rows


def base_doi_key_rows(base: pd.DataFrame) -> pd.DataFrame:
    rows = []
    doi_columns = base_doi_columns(base)

    for _, row in base.iterrows():
        dois = []
        for column in doi_columns:
            if column == "doi_list":
                dois.extend(split_doi_list(row.get(column, "")))
            else:
                doi = normalize_doi(row.get(column, ""))
                if doi:
                    dois.append(doi)

        for order, doi in enumerate(unique_values(dois), start=1):
            rows.append(
                {
                    BASE_ROW_ID: row[BASE_ROW_ID],
                    "base_merge_doi": doi,
                    "base_doi_order": order,
                }
            )

    return pd.DataFrame(rows)


def base_doi_columns(data: pd.DataFrame) -> list[str]:
    columns = []
    if "doi" in data.columns:
        columns.append("doi")
    numbered_doi_columns = sorted(
        [column for column in data.columns if re.fullmatch(r"doi_\d+", column)],
        key=lambda column: int(column.split("_")[1]),
    )
    columns.extend(numbered_doi_columns)
    if "doi_list" in data.columns:
        columns.append("doi_list")
    return columns


def unique_repec_lookup(repec: pd.DataFrame, key_column: str) -> pd.DataFrame:
    if repec.empty or key_column not in repec.columns:
        return pd.DataFrame()

    lookup = repec.loc[repec[key_column] != ""].copy()
    duplicated_key = lookup[key_column].duplicated(keep=False)
    return lookup.loc[~duplicated_key].copy()


def clean_after_repec_merge(data: pd.DataFrame) -> pd.DataFrame:
    cleaned = data.copy()
    before_counts = {
        "abstract": count_nonblank(cleaned, "abstract"),
        "jel_codes": count_nonblank(cleaned, "jel_codes"),
        "keywords": count_nonblank(cleaned, "keywords"),
    }
    cleaned["abstract"] = fill_blank_from_column(cleaned, "abstract", "repec_abstract")
    cleaned["jel_codes"] = fill_blank_from_column(
        cleaned,
        "jel_codes",
        "repec_classification_jel",
    )
    cleaned["keywords"] = fill_blank_from_column(cleaned, "keywords", "repec_keywords")
    filled_counts = {
        column: count_nonblank(cleaned, column) - before_count
        for column, before_count in before_counts.items()
    }

    helper_columns = [
        REPEC_ROW_ID,
        "base_merge_title",
        "repec_merge_doi",
        "repec_merge_x_doi",
        "repec_merge_title",
    ]
    repec_columns = [column for column in cleaned.columns if column.startswith("repec_")]
    columns_to_drop = helper_columns + repec_columns
    cleaned = cleaned.drop(columns=[column for column in columns_to_drop if column in cleaned.columns])
    cleaned.attrs["repec_filled_counts"] = filled_counts
    return cleaned


def duplicate_doi_records(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    doi_normalized = get_column(data, "doi").apply(normalize_doi)
    x_doi_normalized = get_column(data, "x_doi").apply(normalize_doi)

    duplicate_doi = (doi_normalized != "") & doi_normalized.duplicated(keep=False)
    duplicate_x_doi = (x_doi_normalized != "") & x_doi_normalized.duplicated(keep=False)

    data["duplicate_repec_doi"] = duplicate_doi.astype(int)
    data["duplicate_repec_x_doi"] = duplicate_x_doi.astype(int)
    data["normalized_repec_doi"] = doi_normalized
    data["normalized_repec_x_doi"] = x_doi_normalized

    duplicate_records = data.loc[duplicate_doi | duplicate_x_doi].copy()
    if duplicate_records.empty:
        return duplicate_records

    return duplicate_records.sort_values(
        ["normalized_repec_doi", "normalized_repec_x_doi", "title"],
        kind="mergesort",
    )


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


def ensure_publication_year(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    if "publication_year" in data.columns:
        data["publication_year"] = data["publication_year"].apply(clean_year)
        return data

    data["publication_year"] = coalesce_columns(
        data,
        ["openalex_publication_year", "crossref_published_year"],
        clean_year,
    )
    return data


def coalesce_columns(data: pd.DataFrame, columns: list[str], cleaner) -> pd.Series:
    values = pd.Series([""] * len(data), index=data.index)
    for column in columns:
        if column not in data.columns:
            continue
        candidate = data[column].apply(cleaner)
        values = values.mask(values == "", candidate)
    return values


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist.")
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def drop_repec_columns(data: pd.DataFrame) -> pd.DataFrame:
    columns_to_drop = [column for column in REPEC_COLUMNS_TO_DROP if column in data.columns]
    return data.drop(columns=columns_to_drop).copy()


def add_prefix_to_columns(data: pd.DataFrame, prefix: str) -> pd.DataFrame:
    renamed = data.copy()
    renamed.columns = [
        column if column.startswith(prefix) else f"{prefix}{column}"
        for column in renamed.columns
    ]
    return renamed


def year_from_path(path: Path) -> str:
    match = re.search(r"(\d{4})", path.stem)
    if not match:
        raise ValueError(f"Could not find a four-digit year in {path}.")
    return match.group(1)


def split_doi_list(value: Any) -> list[str]:
    return [normalize_doi(doi) for doi in str(value or "").split(";") if normalize_doi(doi)]


def normalize_doi(value: Any) -> str:
    if pd.isna(value):
        return ""

    text = str(value).strip().lower()
    text = re.sub(r"^https?://(dx\.)?doi\.org/", "", text)
    text = re.sub(r"^doi:\s*", "", text)
    return text.strip().rstrip(".")


def clean_year(value: Any) -> str:
    if pd.isna(value):
        return ""

    match = re.search(r"\b(19|20)\d{2}\b", str(value))
    if match:
        return match.group(0)
    return ""


def normalize_title(value: Any) -> str:
    if pd.isna(value):
        return ""

    text = str(value).lower().strip()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def keep_letters_and_numbers(value: Any) -> str:
    if pd.isna(value):
        return ""

    text = str(value)
    text = re.sub(r"[^A-Za-z0-9]+", " ", text)
    return " ".join(text.split())


def rename_specific_repec_titles(data: pd.DataFrame, year: str) -> pd.DataFrame:
    cleaned = data.copy()
    if "title" not in cleaned.columns:
        return cleaned

    title = cleaned["title"].fillna("").astype(str).str.strip()
    target_row = (title == "Human Capital and Growth") & (str(year) == "2015")
    cleaned.loc[target_row, "title"] = "Human Capital and Growth 2015"
    return cleaned


def get_column(data: pd.DataFrame, column: str) -> pd.Series:
    if column in data.columns:
        return data[column]
    return pd.Series([""] * len(data), index=data.index)


def count_nonblank(data: pd.DataFrame, column: str) -> int:
    if column not in data.columns:
        return 0

    return int(data[column].fillna("").astype(str).str.strip().ne("").sum())


def unique_values(values: list[str]) -> list[str]:
    seen = set()
    unique = []
    for value in values:
        value = normalize_doi(value)
        if value and value not in seen:
            unique.append(value)
            seen.add(value)
    return unique


def total_repec_cleaned_rows(summaries: list[dict[str, Any]]) -> int:
    return sum(int(summary["output_rows"]) for summary in summaries)


def total_repec_duplicate_doi_drops(summaries: list[dict[str, Any]]) -> int:
    return sum(int(summary["dropped_duplicate_doi_rows"]) for summary in summaries)


def total_repec_matches(summaries: list[dict[str, Any]]) -> int:
    return sum(
        int(summary["matched_by_doi"])
        + int(summary["matched_by_x_doi"])
        + int(summary["matched_by_title"])
        for summary in summaries
    )


def total_matched_by_strategy(summaries: list[dict[str, Any]], strategy_key: str) -> int:
    return sum(int(summary[strategy_key]) for summary in summaries)


def total_filled_from_repec(summaries: list[dict[str, Any]], column: str) -> int:
    summary_key = {
        "abstract": "abstracts_filled_from_repec",
        "jel_codes": "jel_codes_filled_from_repec",
        "keywords": "keywords_filled_from_repec",
    }[column]
    return sum(int(summary[summary_key]) for summary in summaries)


if __name__ == "__main__":
    main()
