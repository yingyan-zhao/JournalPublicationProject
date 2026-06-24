from pathlib import Path
import os
import re
import unicodedata

import pandas as pd


os.chdir("/Users/yingyan_zhao/Dropbox/JournalPublicationProject")

OPENALEX_CROSSREF_BY_YEAR_DIR = Path("data/processed/openalex_crossref_merged/by_year")
REPEC_BY_YEAR_DIR = Path("data/processed/repec_by_year_cleaned")
OUTPUT_DIR = Path("data/processed/openalex_crossref_repec_merged")
OUTPUT_BY_YEAR_DIR = OUTPUT_DIR / "by_year"
SUMMARY_CSV = OUTPUT_DIR / "OpenAlex_Crossref_RePEc_Merge_Summary_By_Year.csv"
OUTPUT_ALL_CSV = OUTPUT_DIR / "OpenAlex_Crossref_RePEc_All.csv"

MIN_YEAR = 2000
MAX_YEAR = 2026
OPENALEX_CROSSREF_ROW_ID = "_openalex_crossref_row_id"
REPEC_ROW_ID = "_repec_row_id"
TEMP_COLUMNS = [
    OPENALEX_CROSSREF_ROW_ID,
    REPEC_ROW_ID,
    "merge_title",
    "merge_title_x",
    "merge_title_y",
    "merge_doi_openalex_crossref",
    "merge_doi_repec",
]


def main() -> None:
    openalex_crossref_files = yearly_files(
        OPENALEX_CROSSREF_BY_YEAR_DIR,
        "*/OpenAlex_Crossref_All_*.csv",
    )
    repec_files = yearly_files(REPEC_BY_YEAR_DIR, "RePEc_ReDIF_Cleaned_*.csv")
    years = [
        year
        for year in sorted(set(openalex_crossref_files) | set(repec_files))
        if MIN_YEAR <= year <= MAX_YEAR
    ]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_BY_YEAR_DIR.mkdir(parents=True, exist_ok=True)

    all_years = []
    summary_rows = []
    for year in years:
        openalex_crossref = read_year_file(openalex_crossref_files.get(year))
        repec = read_year_file(repec_files.get(year))

        merged, summary = merge_one_year(openalex_crossref, repec, year)
        write_year_output(merged, year)
        all_years.append(merged)
        summary_rows.append(summary)

        print(
            f"{year}: OpenAlex/Crossref={summary['openalex_crossref_rows']}, "
            f"RePEc={summary['repec_rows']}, "
            f"matched DOI rows={summary['matched_by_doi']}, "
            f"matched title rows={summary['matched_by_title']}, "
            f"OpenAlex/Crossref only={summary['openalex_crossref_only']}"
        )

    all_merged = concat_frames(all_years)
    all_merged.to_csv(OUTPUT_ALL_CSV, index=False)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(SUMMARY_CSV, index=False)
    print_merge_summary(summary, all_merged)


def yearly_files(input_dir: Path, pattern: str) -> dict[int, Path]:
    files_by_year = {}
    for path in sorted(input_dir.glob(pattern)):
        year = year_from_filename(path)
        if year is not None:
            files_by_year[year] = path
    return files_by_year


def year_from_filename(path: Path) -> int | None:
    match = re.search(r"(19|20)\d{2}", path.name)
    if not match:
        return None
    return int(match.group(0))


def read_year_file(path: Path | None) -> pd.DataFrame:
    if path is None:
        return pd.DataFrame()
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def merge_one_year(
    openalex_crossref: pd.DataFrame,
    repec: pd.DataFrame,
    year: int,
) -> tuple[pd.DataFrame, dict[str, int]]:
    openalex_crossref = prepare_openalex_crossref(openalex_crossref)
    repec = prepare_repec(repec)
    openalex_crossref[OPENALEX_CROSSREF_ROW_ID] = range(len(openalex_crossref))
    repec[REPEC_ROW_ID] = range(len(repec))

    openalex_crossref_with_doi = openalex_crossref.loc[
        openalex_crossref["merge_doi"] != ""
    ].copy()
    repec_with_doi = repec.loc[repec["merge_doi"] != ""].copy()

    matched_by_doi = openalex_crossref_with_doi.merge(
        repec_with_doi,
        on="merge_doi",
        how="inner",
    )
    matched_by_doi.insert(0, "repec_merge_status", "matched_by_doi")

    matched_openalex_crossref_ids = matched_by_doi[
        OPENALEX_CROSSREF_ROW_ID
    ].dropna().unique()
    matched_repec_ids = matched_by_doi[REPEC_ROW_ID].dropna().unique()

    remaining_openalex_crossref = openalex_crossref.loc[
        ~openalex_crossref[OPENALEX_CROSSREF_ROW_ID].isin(
            matched_openalex_crossref_ids
        )
    ].copy()
    remaining_repec = repec.loc[
        ~repec[REPEC_ROW_ID].isin(matched_repec_ids)
    ].copy()

    (
        matched_by_title,
        only_openalex_crossref,
        only_repec,
        ambiguous_openalex_crossref_title_rows,
        ambiguous_repec_title_rows,
    ) = match_remaining_by_title(remaining_openalex_crossref, remaining_repec)

    merged = concat_frames([matched_by_doi, matched_by_title, only_openalex_crossref])
    merged["merge_year"] = year
    merged = move_column_to_front(merged, "merge_year")
    merged = drop_temp_columns(merged)

    summary = {
        "year": year,
        "openalex_crossref_rows": len(openalex_crossref),
        "repec_rows": len(repec),
        "openalex_crossref_nonblank_doi_rows": len(openalex_crossref_with_doi),
        "repec_nonblank_doi_rows": len(repec_with_doi),
        "openalex_crossref_unique_nonblank_doi": openalex_crossref_with_doi[
            "merge_doi"
        ].nunique(),
        "repec_unique_nonblank_doi": repec_with_doi["merge_doi"].nunique(),
        "openalex_crossref_duplicate_nonblank_doi_rows": int(
            openalex_crossref_with_doi["merge_doi"].duplicated().sum()
        ),
        "repec_duplicate_nonblank_doi_rows": int(
            repec_with_doi["merge_doi"].duplicated().sum()
        ),
        "matched_by_doi": int(
            (merged["repec_merge_status"] == "matched_by_doi").sum()
        ),
        "matched_by_title": int(
            (merged["repec_merge_status"] == "matched_by_title").sum()
        ),
        "openalex_crossref_only": int(
            (merged["repec_merge_status"] == "openalex_crossref_only").sum()
        ),
        "repec_unmatched_not_written": len(only_repec),
        "ambiguous_openalex_crossref_title_rows": ambiguous_openalex_crossref_title_rows,
        "ambiguous_repec_title_rows": ambiguous_repec_title_rows,
        "merged_rows": len(merged),
    }

    return merged, summary


def prepare_openalex_crossref(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    data["merge_doi"] = first_nonblank_normalized_doi(data, ["doi_oa", "doi_cr", "doi"])
    data["merge_title"] = first_nonblank_normalized_title(
        data,
        ["openalex_title", "crossref_title"],
    )
    return data


def prepare_repec(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    data["merge_doi"] = first_nonblank_normalized_doi(data, ["repec_x_doi"])
    data["merge_title"] = first_nonblank_normalized_title(data, ["repec_title"])
    return data


def match_remaining_by_title(
    openalex_crossref: pd.DataFrame,
    repec: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, int, int]:
    if openalex_crossref.empty or repec.empty:
        openalex_crossref = add_merge_status(
            openalex_crossref,
            "openalex_crossref_only",
        )
        repec = add_merge_status(repec, "repec_only")
        return pd.DataFrame(), openalex_crossref, repec, 0, 0

    ambiguous_openalex_crossref = count_ambiguous_keys(
        openalex_crossref,
        "merge_title",
    )
    ambiguous_repec = count_ambiguous_keys(repec, "merge_title")

    openalex_crossref_matchable = keep_unique_nonblank_key(
        openalex_crossref,
        "merge_title",
    )
    repec_matchable = keep_unique_nonblank_key(repec, "merge_title")

    matched_by_title = openalex_crossref_matchable.merge(
        repec_matchable,
        on="merge_title",
        how="inner",
        suffixes=("_openalex_crossref", "_repec"),
    )
    matched_by_title.insert(0, "repec_merge_status", "matched_by_title")
    matched_by_title["merge_doi"] = first_nonblank_normalized_doi(
        matched_by_title,
        ["merge_doi_openalex_crossref", "merge_doi_repec"],
    )

    matched_openalex_crossref_ids = matched_by_title[
        OPENALEX_CROSSREF_ROW_ID
    ].dropna().unique()
    matched_repec_ids = matched_by_title[REPEC_ROW_ID].dropna().unique()

    only_openalex_crossref = openalex_crossref.loc[
        ~openalex_crossref[OPENALEX_CROSSREF_ROW_ID].isin(
            matched_openalex_crossref_ids
        )
    ].copy()
    only_repec = repec.loc[
        ~repec[REPEC_ROW_ID].isin(matched_repec_ids)
    ].copy()

    only_openalex_crossref = add_merge_status(
        only_openalex_crossref,
        "openalex_crossref_only",
    )
    only_repec = add_merge_status(only_repec, "repec_only")

    return (
        matched_by_title,
        only_openalex_crossref,
        only_repec,
        ambiguous_openalex_crossref,
        ambiguous_repec,
    )


def add_merge_status(data: pd.DataFrame, status: str) -> pd.DataFrame:
    data = data.copy()
    data.insert(0, "repec_merge_status", status)
    return data


def keep_unique_nonblank_key(data: pd.DataFrame, key_column: str) -> pd.DataFrame:
    nonblank = data.loc[data[key_column] != ""].copy()
    key_counts = nonblank[key_column].value_counts()
    unique_keys = key_counts.loc[key_counts == 1].index
    return nonblank.loc[nonblank[key_column].isin(unique_keys)].copy()


def count_ambiguous_keys(data: pd.DataFrame, key_column: str) -> int:
    nonblank = data.loc[data[key_column] != ""]
    key_counts = nonblank[key_column].value_counts()
    ambiguous_keys = key_counts.loc[key_counts > 1].index
    return int(nonblank[key_column].isin(ambiguous_keys).sum())


def first_nonblank_normalized_doi(
    data: pd.DataFrame,
    columns: list[str],
) -> pd.Series:
    doi = pd.Series("", index=data.index, dtype="object")
    for column in columns:
        if column not in data.columns:
            continue
        candidate = data[column].apply(normalize_doi)
        doi = doi.mask((doi == "") & (candidate != ""), candidate)
    return doi


def first_nonblank_normalized_title(
    data: pd.DataFrame,
    columns: list[str],
) -> pd.Series:
    title = pd.Series("", index=data.index, dtype="object")
    for column in columns:
        if column not in data.columns:
            continue
        candidate = data[column].apply(normalize_title)
        title = title.mask((title == "") & (candidate != ""), candidate)
    return title


def normalize_doi(value: str) -> str:
    if pd.isna(value):
        return ""

    return (
        str(value)
        .strip()
        .lower()
        .replace("https://doi.org/", "")
        .replace("http://doi.org/", "")
        .replace("http://dx.doi.org/", "")
    )


def normalize_title(value: str) -> str:
    if pd.isna(value):
        return ""

    text = unicodedata.normalize("NFKD", str(value))
    text = text.encode("ascii", "ignore").decode("ascii")
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "", text)
    return text


def write_year_output(merged: pd.DataFrame, year: int) -> None:
    year_dir = OUTPUT_BY_YEAR_DIR / str(year)
    year_dir.mkdir(parents=True, exist_ok=True)
    output_path = year_dir / f"OpenAlex_Crossref_RePEc_All_{year}.csv"
    merged.to_csv(output_path, index=False)


def move_column_to_front(data: pd.DataFrame, column: str) -> pd.DataFrame:
    columns = [column] + [other for other in data.columns if other != column]
    return data.loc[:, columns]


def drop_temp_columns(data: pd.DataFrame) -> pd.DataFrame:
    columns_to_drop = [column for column in TEMP_COLUMNS if column in data.columns]
    return data.drop(columns=columns_to_drop)


def concat_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    nonempty_frames = [frame for frame in frames if not frame.empty]
    if not nonempty_frames:
        return pd.DataFrame()
    return pd.concat(nonempty_frames, ignore_index=True)


def print_merge_summary(summary: pd.DataFrame, all_merged: pd.DataFrame) -> None:
    report_columns = [
        "year",
        "openalex_crossref_rows",
        "repec_rows",
        "openalex_crossref_duplicate_nonblank_doi_rows",
        "repec_duplicate_nonblank_doi_rows",
        "matched_by_doi",
        "matched_by_title",
        "openalex_crossref_only",
        "repec_unmatched_not_written",
        "ambiguous_openalex_crossref_title_rows",
        "ambiguous_repec_title_rows",
        "merged_rows",
    ]

    print("\nYear-by-year OpenAlex/Crossref-centered RePEc merge summary:")
    print(summary[report_columns].to_string(index=False))

    print("\nTotals:")
    print(summary[report_columns[1:]].sum().to_string())
    print(f"Combined merged rows: {len(all_merged)}")
    print(f"Wrote combined merged file to {OUTPUT_ALL_CSV}.")
    print(f"Wrote yearly merged files to {OUTPUT_BY_YEAR_DIR}.")
    print(f"Wrote summary to {SUMMARY_CSV}.")


if __name__ == "__main__":
    main()
