from pathlib import Path
import os
import re

import pandas as pd


os.chdir("/Users/yingyan_zhao/Dropbox/JournalPublicationProject")

BASE_INPUT_CSV = Path("data/processed/OpenAlex_Crossref_Webscrape_NBER_Merged.csv")
MERGED_OUTPUT_CSV = Path("data/processed/OpenAlex_Crossref_Webscrape_NBER_Repec_Merged.csv")
REPEC_INPUT_DIR = Path("data/raw_csv/repec_by_year")
REPEC_OUTPUT_DIR = Path("data/processed/repec_by_year_cleaned")
REPEC_COLUMNS_TO_DROP_AFTER_MERGE = [
    "repec_doi",
    "repec_x_doi",
    "repec_match_strategy",
    "repec_title",
    "repec_classification_jel",
    "repec_abstract",
]

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


def main() -> None:
    base = read_base_data(BASE_INPUT_CSV)
    merged, merge_summaries = merge_base_with_repec_by_year(base)
    MERGED_OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(MERGED_OUTPUT_CSV, index=False)

    print("OpenAlex/Crossref/Webscrape + RePEc year-by-year merge summary:")
    print(f"  Base input CSV: {BASE_INPUT_CSV}")
    print(f"  Base rows: {len(base)}")
    print(f"  Output CSV: {MERGED_OUTPUT_CSV}")
    print(f"  Output rows: {len(merged)}")
    print(f"  Total RePEc matches: {total_repec_matches(merge_summaries)}")
    for summary in merge_summaries:
        print(
            f"  {summary['year']}: "
            f"base={summary['base_rows']}, "
            f"repec={summary['repec_rows']}, "
            f"doi={summary['matched_by_doi']}, "
            f"x_doi={summary['matched_by_x_doi']}, "
            f"title={summary['matched_by_title']}, "
            f"unmatched={summary['unmatched']}, "
            f"duplicate_repec_doi_rows={summary['duplicate_repec_doi_rows']}, "
            f"duplicate_repec_x_doi_rows={summary['duplicate_repec_x_doi_rows']}, "
            f"duplicate_repec_title_rows={summary['duplicate_repec_title_rows']}"
        )


def clean_repec_year_file(input_file: Path, output_file: Path, year: str) -> dict[str, object]:
    data = pd.read_csv(input_file, dtype=str, keep_default_na=False)
    cleaned = drop_repec_columns(data)
    cleaned = add_prefix_to_columns(cleaned, "repec_")
    cleaned.to_csv(output_file, index=False)

    dropped_columns = [
        column for column in REPEC_COLUMNS_TO_DROP
        if column in data.columns
    ]
    return {
        "year": year,
        "input_rows": len(data),
        "output_rows": len(cleaned),
        "dropped_columns": len(dropped_columns),
        "output_file": output_file,
    }


def read_base_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist.")

    return pd.read_csv(path, dtype=str, keep_default_na=False)


def merge_base_with_repec_by_year(base: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    base_for_merge = base.copy()
    base_for_merge["_base_row_order"] = range(len(base_for_merge))

    merged_years = []
    summaries = []
    for year, base_year in base_for_merge.groupby("publication_year", sort=True):
        repec_file = REPEC_OUTPUT_DIR / f"RePEc_ReDIF_{year}_Cleaned.csv"
        if repec_file.exists():
            repec = pd.read_csv(repec_file, dtype=str, keep_default_na=False)
        else:
            repec = pd.DataFrame()

        merged_year, summary = merge_one_year(base_year.copy(), repec, str(year))
        merged_years.append(merged_year)
        summaries.append(summary)

    merged = pd.concat(merged_years, ignore_index=True)
    merged = merged.sort_values("_base_row_order").drop(columns=["_base_row_order"])
    return merged.reset_index(drop=True), summaries


def merge_one_year(
    base_year: pd.DataFrame,
    repec: pd.DataFrame,
    year: str,
) -> tuple[pd.DataFrame, dict[str, object]]:
    base_year["base_merge_doi"] = base_year["doi"].apply(normalize_doi)
    base_year["base_merge_title"] = base_year["title"].apply(normalize_title)

    repec_prepared = prepare_repec_for_merge(repec)
    summary = {
        "year": year,
        "base_rows": len(base_year),
        "repec_rows": len(repec_prepared),
        "matched_by_doi": 0,
        "matched_by_x_doi": 0,
        "matched_by_title": 0,
        "unmatched": 0,
        "duplicate_repec_doi_rows": duplicate_key_rows(repec_prepared, "repec_merge_doi"),
        "duplicate_repec_x_doi_rows": duplicate_key_rows(repec_prepared, "repec_merge_x_doi"),
        "duplicate_repec_title_rows": duplicate_key_rows(repec_prepared, "repec_merge_title"),
    }

    doi_lookup = deduplicate_repec_lookup(repec_prepared, "repec_merge_doi")
    x_doi_lookup = deduplicate_repec_lookup(repec_prepared, "repec_merge_x_doi")
    title_lookup = deduplicate_repec_lookup(repec_prepared, "repec_merge_title")

    matched_frames = []
    remaining = base_year
    matched, remaining = match_repec_stage(
        remaining,
        doi_lookup,
        base_key_column="base_merge_doi",
        repec_key_column="repec_merge_doi",
        strategy="doi",
    )
    summary["matched_by_doi"] = len(matched)
    matched_frames.append(matched)

    matched, remaining = match_repec_stage(
        remaining,
        x_doi_lookup,
        base_key_column="base_merge_doi",
        repec_key_column="repec_merge_x_doi",
        strategy="x_doi",
    )
    summary["matched_by_x_doi"] = len(matched)
    matched_frames.append(matched)

    matched, remaining = match_repec_stage(
        remaining,
        title_lookup,
        base_key_column="base_merge_title",
        repec_key_column="repec_merge_title",
        strategy="title",
    )
    summary["matched_by_title"] = len(matched)
    matched_frames.append(matched)

    remaining = remaining.copy()
    remaining["repec_match_strategy"] = ""
    summary["unmatched"] = len(remaining)
    matched_frames.append(remaining)

    merged = pd.concat(matched_frames, ignore_index=True, sort=False)
    merged = merged.sort_values("_base_row_order")
    merged = drop_merge_helper_columns(merged)
    merged["abstract"] = fill_blank_from_column(
        merged,
        target_column="abstract",
        source_column="repec_abstract",
    )
    merged["jel_codes"] = fill_blank_from_column(
        merged,
        target_column="jel_codes",
        source_column="repec_classification_jel",
    )
    merged = drop_repec_columns_after_merge(merged)
    return merged, summary


def prepare_repec_for_merge(repec: pd.DataFrame) -> pd.DataFrame:
    prepared = repec.copy()
    if prepared.empty:
        for column in ["repec_merge_doi", "repec_merge_x_doi", "repec_merge_title"]:
            prepared[column] = ""
        return prepared

    prepared["repec_merge_doi"] = get_column(prepared, "repec_doi").apply(normalize_doi)
    prepared["repec_merge_x_doi"] = get_column(prepared, "repec_x_doi").apply(normalize_doi)
    prepared["repec_merge_title"] = get_column(prepared, "repec_title").apply(normalize_title)
    return prepared


def match_repec_stage(
    base_unmatched: pd.DataFrame,
    repec_lookup: pd.DataFrame,
    base_key_column: str,
    repec_key_column: str,
    strategy: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if base_unmatched.empty or repec_lookup.empty:
        return pd.DataFrame(), base_unmatched

    eligible = base_unmatched.loc[base_unmatched[base_key_column] != ""].copy()
    ineligible = base_unmatched.loc[base_unmatched[base_key_column] == ""].copy()
    if eligible.empty:
        return pd.DataFrame(), base_unmatched

    stage = eligible.merge(
        repec_lookup,
        left_on=base_key_column,
        right_on=repec_key_column,
        how="left",
        indicator=True,
    )
    matched = stage.loc[stage["_merge"] == "both"].drop(columns=["_merge"]).copy()
    matched["repec_match_strategy"] = strategy

    original_columns = list(base_unmatched.columns)
    still_unmatched = stage.loc[stage["_merge"] == "left_only", original_columns].copy()
    still_unmatched = pd.concat([still_unmatched, ineligible], ignore_index=True, sort=False)
    return matched, still_unmatched


def deduplicate_repec_lookup(repec: pd.DataFrame, key_column: str) -> pd.DataFrame:
    if repec.empty or key_column not in repec.columns:
        return pd.DataFrame()

    lookup = repec.loc[repec[key_column] != ""].copy()
    lookup = lookup.sort_values(
        [key_column, "repec_doi", "repec_x_doi", "repec_title"],
        na_position="last",
    )
    return lookup.drop_duplicates(subset=[key_column], keep="first")


def duplicate_key_rows(data: pd.DataFrame, key_column: str) -> int:
    if data.empty or key_column not in data.columns:
        return 0

    key = data[key_column].fillna("").astype(str)
    return int(key.loc[key != ""].duplicated(keep=False).sum())


def drop_merge_helper_columns(data: pd.DataFrame) -> pd.DataFrame:
    helper_columns = [
        "base_merge_doi",
        "base_merge_title",
        "repec_merge_doi",
        "repec_merge_x_doi",
        "repec_merge_title",
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


def drop_repec_columns_after_merge(data: pd.DataFrame) -> pd.DataFrame:
    columns_to_drop = [
        column for column in REPEC_COLUMNS_TO_DROP_AFTER_MERGE
        if column in data.columns
    ]
    return data.drop(columns=columns_to_drop)


def drop_repec_columns(data: pd.DataFrame) -> pd.DataFrame:
    columns_to_drop = [
        column for column in REPEC_COLUMNS_TO_DROP
        if column in data.columns
    ]
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


def get_column(data: pd.DataFrame, column: str) -> pd.Series:
    if column in data.columns:
        return data[column]

    return pd.Series([""] * len(data), index=data.index)


def count_nonblank(data: pd.DataFrame, column: str) -> int:
    if column not in data.columns:
        return 0

    return int(data[column].fillna("").astype(str).str.strip().ne("").sum())


def total_repec_matches(summaries: list[dict[str, object]]) -> int:
    return sum(
        int(summary["matched_by_doi"])
        + int(summary["matched_by_x_doi"])
        + int(summary["matched_by_title"])
        for summary in summaries
    )


if __name__ == "__main__":
    main()
