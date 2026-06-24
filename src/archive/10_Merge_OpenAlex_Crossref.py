from pathlib import Path
import re
import unicodedata

import os
import pandas as pd


os.chdir("/Users/yingyan_zhao/Dropbox/JournalPublicationProject")

INPUT_DIR_OpenAlex = Path("data/processed/openalex_by_year_cleaned")
INPUT_DIR_Crossref = Path("data/processed/crossref_by_year_cleaned")
OUTPUT_DIR = Path("data/processed/openalex_crossref_merged")
OUTPUT_BY_YEAR_DIR = OUTPUT_DIR / "by_year"

OUTPUT_CSV_Matched = OUTPUT_DIR / "OpenAlex_Crossref_Matched.csv"
OUTPUT_CSV_Matched_By_DOI = OUTPUT_DIR / "OpenAlex_Crossref_Matched_By_DOI.csv"
OUTPUT_CSV_Matched_By_Title = OUTPUT_DIR / "OpenAlex_Crossref_Matched_By_Title.csv"
OUTPUT_CSV_Only_OpenAlex = OUTPUT_DIR / "OpenAlex_Only.csv"
OUTPUT_CSV_Only_Crossref = OUTPUT_DIR / "Crossref_Only.csv"

OPENALEX_ROW_ID = "_openalex_row_id"
CROSSREF_ROW_ID = "_crossref_row_id"
TEMP_COLUMNS = [
    OPENALEX_ROW_ID,
    CROSSREF_ROW_ID,
    "merge_doi",
    "merge_title",
]


def main() -> None:
    openalex_files = yearly_files(INPUT_DIR_OpenAlex, "OpenAlex_Works_Cleaned_*.csv")
    crossref_files = yearly_files(INPUT_DIR_Crossref, "Crossref_Works_Cleaned_*.csv")
    years = sorted(set(openalex_files) | set(crossref_files))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_BY_YEAR_DIR.mkdir(parents=True, exist_ok=True)

    all_matched_by_doi = []
    all_matched_by_title = []
    all_only_openalex = []
    all_only_crossref = []
    summary_rows = []

    for year in years:
        openalex = read_year_file(openalex_files.get(year), dataset_name="OpenAlex", year=year)
        crossref = read_year_file(crossref_files.get(year), dataset_name="Crossref", year=year)

        result = merge_one_year(openalex, crossref, year)
        write_year_outputs(result, year)

        all_matched_by_doi.append(result["matched_by_doi"])
        all_matched_by_title.append(result["matched_by_title"])
        all_only_openalex.append(result["only_openalex"])
        all_only_crossref.append(result["only_crossref"])
        summary_rows.append(result["summary"])

    matched_by_doi = concat_frames(all_matched_by_doi)
    matched_by_title = concat_frames(all_matched_by_title)
    matched_all = concat_frames([matched_by_doi, matched_by_title])
    only_openalex = concat_frames(all_only_openalex)
    only_crossref = concat_frames(all_only_crossref)

    matched_all.to_csv(OUTPUT_CSV_Matched, index=False)
    matched_by_doi.to_csv(OUTPUT_CSV_Matched_By_DOI, index=False)
    matched_by_title.to_csv(OUTPUT_CSV_Matched_By_Title, index=False)
    only_openalex.to_csv(OUTPUT_CSV_Only_OpenAlex, index=False)
    only_crossref.to_csv(OUTPUT_CSV_Only_Crossref, index=False)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUTPUT_DIR / "OpenAlex_Crossref_Merge_Summary_By_Year.csv", index=False)

    print_merge_summary(summary, matched_all, matched_by_doi, matched_by_title, only_openalex, only_crossref)


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


def read_year_file(path: Path | None, dataset_name: str, year: int) -> pd.DataFrame:
    if path is None:
        print(f"{dataset_name} has no file for {year}; using empty data.")
        return pd.DataFrame()

    data = pd.read_csv(path)
    data["source_file"] = path.name
    return data


def merge_one_year(openalex: pd.DataFrame, crossref: pd.DataFrame, year: int) -> dict[str, object]:
    check_required_columns(openalex, crossref, year)

    doi_check_openalex = doi_uniqueness_stats(openalex)
    doi_check_crossref = doi_uniqueness_stats(crossref)

    openalex = openalex.copy()
    crossref = crossref.copy()
    openalex[OPENALEX_ROW_ID] = range(len(openalex))
    crossref[CROSSREF_ROW_ID] = range(len(crossref))
    openalex["merge_year"] = year
    crossref["merge_year"] = year

    matched_by_doi, remaining_openalex, remaining_crossref = match_by_doi(openalex, crossref)
    matched_by_title, only_openalex, only_crossref, ambiguous_openalex, ambiguous_crossref = (
        match_by_title(remaining_openalex, remaining_crossref)
    )

    matched_by_doi = drop_temp_columns(matched_by_doi)
    matched_by_title = drop_temp_columns(matched_by_title)
    only_openalex = drop_temp_columns(only_openalex)
    only_crossref = drop_temp_columns(only_crossref)

    summary = {
        "year": year,
        "openalex_rows": len(openalex),
        "crossref_rows": len(crossref),
        "openalex_nonblank_doi_rows": doi_check_openalex["nonblank"],
        "openalex_unique_nonblank_doi": doi_check_openalex["unique"],
        "openalex_duplicate_nonblank_doi_rows": doi_check_openalex["duplicates"],
        "crossref_nonblank_doi_rows": doi_check_crossref["nonblank"],
        "crossref_unique_nonblank_doi": doi_check_crossref["unique"],
        "crossref_duplicate_nonblank_doi_rows": doi_check_crossref["duplicates"],
        "matched_by_doi": len(matched_by_doi),
        "matched_by_title": len(matched_by_title),
        "matched_total": len(matched_by_doi) + len(matched_by_title),
        "only_openalex": len(only_openalex),
        "only_crossref": len(only_crossref),
        "ambiguous_openalex_title_rows": ambiguous_openalex,
        "ambiguous_crossref_title_rows": ambiguous_crossref,
    }

    print_year_summary(summary)

    return {
        "matched_by_doi": matched_by_doi,
        "matched_by_title": matched_by_title,
        "only_openalex": only_openalex,
        "only_crossref": only_crossref,
        "summary": summary,
    }


def check_required_columns(openalex: pd.DataFrame, crossref: pd.DataFrame, year: int) -> None:
    if not openalex.empty:
        missing_openalex = [
            column for column in ["doi", "openalex_title"] if column not in openalex.columns
        ]
        if missing_openalex:
            raise ValueError(f"OpenAlex {year} missing columns: {missing_openalex}")

    if not crossref.empty:
        missing_crossref = [
            column for column in ["doi", "crossref_title"] if column not in crossref.columns
        ]
        if missing_crossref:
            raise ValueError(f"Crossref {year} missing columns: {missing_crossref}")


def doi_uniqueness_stats(data: pd.DataFrame) -> dict[str, int]:
    if data.empty or "doi" not in data.columns:
        return {"nonblank": 0, "unique": 0, "duplicates": 0}

    doi = data["doi"].apply(normalize_doi)
    nonblank_doi = doi.loc[doi != ""]
    return {
        "nonblank": len(nonblank_doi),
        "unique": nonblank_doi.nunique(),
        "duplicates": int(nonblank_doi.duplicated().sum()),
    }


def match_by_doi(
    openalex: pd.DataFrame,
    crossref: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if openalex.empty or crossref.empty:
        return empty_merge(), openalex.copy(), crossref.copy()

    openalex_with_key = openalex.copy()
    crossref_with_key = crossref.copy()
    openalex_with_key["merge_doi"] = openalex_with_key["doi"].apply(normalize_doi)
    crossref_with_key["merge_doi"] = crossref_with_key["doi"].apply(normalize_doi)

    openalex_matchable = openalex_with_key.loc[openalex_with_key["merge_doi"] != ""]
    crossref_matchable = crossref_with_key.loc[crossref_with_key["merge_doi"] != ""]

    matched = openalex_matchable.merge(
        crossref_matchable,
        on="merge_doi",
        how="inner",
        suffixes=("_oa", "_cr"),
    )
    matched["match_strategy"] = "doi"

    matched_openalex_ids = matched[OPENALEX_ROW_ID].dropna().unique()
    matched_crossref_ids = matched[CROSSREF_ROW_ID].dropna().unique()

    remaining_openalex = openalex.loc[
        ~openalex[OPENALEX_ROW_ID].isin(matched_openalex_ids)
    ].copy()
    remaining_crossref = crossref.loc[
        ~crossref[CROSSREF_ROW_ID].isin(matched_crossref_ids)
    ].copy()

    return matched, remaining_openalex, remaining_crossref


def match_by_title(
    openalex: pd.DataFrame,
    crossref: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, int, int]:
    if openalex.empty or crossref.empty:
        return empty_merge(), openalex.copy(), crossref.copy(), 0, 0

    openalex_with_key = openalex.copy()
    crossref_with_key = crossref.copy()
    openalex_with_key["merge_title"] = openalex_with_key["openalex_title"].apply(normalize_title)
    crossref_with_key["merge_title"] = crossref_with_key["crossref_title"].apply(normalize_title)

    openalex_matchable = keep_unique_nonblank_key(openalex_with_key, "merge_title")
    crossref_matchable = keep_unique_nonblank_key(crossref_with_key, "merge_title")

    ambiguous_openalex = count_ambiguous_keys(openalex_with_key, "merge_title")
    ambiguous_crossref = count_ambiguous_keys(crossref_with_key, "merge_title")

    matched = openalex_matchable.merge(
        crossref_matchable,
        on="merge_title",
        how="inner",
        suffixes=("_oa", "_cr"),
    )
    matched["match_strategy"] = "normalized_title"

    matched_openalex_ids = matched[OPENALEX_ROW_ID].dropna().unique()
    matched_crossref_ids = matched[CROSSREF_ROW_ID].dropna().unique()

    only_openalex = openalex.loc[
        ~openalex[OPENALEX_ROW_ID].isin(matched_openalex_ids)
    ].copy()
    only_crossref = crossref.loc[
        ~crossref[CROSSREF_ROW_ID].isin(matched_crossref_ids)
    ].copy()

    return matched, only_openalex, only_crossref, ambiguous_openalex, ambiguous_crossref


def empty_merge() -> pd.DataFrame:
    return pd.DataFrame()


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


def normalize_doi(value) -> str:
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


def normalize_title(value) -> str:
    if pd.isna(value):
        return ""

    text = unicodedata.normalize("NFKD", str(value))
    text = text.encode("ascii", "ignore").decode("ascii")
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "", text)
    return text


def drop_temp_columns(data: pd.DataFrame) -> pd.DataFrame:
    columns_to_drop = [column for column in TEMP_COLUMNS if column in data.columns]
    return data.drop(columns=columns_to_drop)


def write_year_outputs(result: dict[str, object], year: int) -> None:
    year_dir = OUTPUT_BY_YEAR_DIR / str(year)
    year_dir.mkdir(parents=True, exist_ok=True)

    matched_all = concat_frames([result["matched_by_doi"], result["matched_by_title"]])
    all_records = combine_matched_and_unmatched_records(
        matched_all=matched_all,
        only_openalex=result["only_openalex"],
        only_crossref=result["only_crossref"],
    )

    all_records.to_csv(year_dir / f"OpenAlex_Crossref_All_{year}.csv", index=False)
    matched_all.to_csv(year_dir / f"OpenAlex_Crossref_Matched_{year}.csv", index=False)
    result["matched_by_doi"].to_csv(year_dir / f"OpenAlex_Crossref_Matched_By_DOI_{year}.csv", index=False)
    result["matched_by_title"].to_csv(year_dir / f"OpenAlex_Crossref_Matched_By_Title_{year}.csv", index=False)
    result["only_openalex"].to_csv(year_dir / f"OpenAlex_Only_{year}.csv", index=False)
    result["only_crossref"].to_csv(year_dir / f"Crossref_Only_{year}.csv", index=False)


def combine_matched_and_unmatched_records(
    matched_all: pd.DataFrame,
    only_openalex: pd.DataFrame,
    only_crossref: pd.DataFrame,
) -> pd.DataFrame:
    matched_all = add_record_status(matched_all, "matched")
    only_openalex = add_record_status(only_openalex, "openalex_only")
    only_crossref = add_record_status(only_crossref, "crossref_only")

    return concat_frames([matched_all, only_openalex, only_crossref])


def add_record_status(data: pd.DataFrame, record_status: str) -> pd.DataFrame:
    data = data.copy()
    data.insert(0, "record_status", record_status)

    if "match_strategy" not in data.columns:
        data.insert(1, "match_strategy", "")
    else:
        match_strategy = data.pop("match_strategy")
        data.insert(1, "match_strategy", match_strategy)

    return data


def concat_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    nonempty_frames = [frame for frame in frames if not frame.empty]
    if not nonempty_frames:
        return pd.DataFrame()
    return pd.concat(nonempty_frames, ignore_index=True)


def print_year_summary(summary: dict[str, int]) -> None:
    print(
        f"{summary['year']}: "
        f"OpenAlex={summary['openalex_rows']}, "
        f"Crossref={summary['crossref_rows']}, "
        f"DOI matches={summary['matched_by_doi']}, "
        f"title matches={summary['matched_by_title']}, "
        f"OpenAlex only={summary['only_openalex']}, "
        f"Crossref only={summary['only_crossref']}"
    )


def print_merge_summary(
    summary: pd.DataFrame,
    matched_all: pd.DataFrame,
    matched_by_doi: pd.DataFrame,
    matched_by_title: pd.DataFrame,
    only_openalex: pd.DataFrame,
    only_crossref: pd.DataFrame,
) -> None:
    report_columns = [
        "year",
        "openalex_rows",
        "crossref_rows",
        "openalex_duplicate_nonblank_doi_rows",
        "crossref_duplicate_nonblank_doi_rows",
        "matched_by_doi",
        "matched_by_title",
        "matched_total",
        "only_openalex",
        "only_crossref",
    ]

    print("\nYear-by-year summary:")
    print(summary[report_columns].to_string(index=False))

    print("\nTotals:")
    print(summary[report_columns[1:]].sum().to_string())

    print("Merge summary across years:")
    print(f"  Years processed: {len(summary)}")
    print(f"  OpenAlex rows: {summary['openalex_rows'].sum()}")
    print(f"  Crossref rows: {summary['crossref_rows'].sum()}")
    print(f"  Matched by DOI: {len(matched_by_doi)}")
    print(f"  Matched by normalized title after DOI matching: {len(matched_by_title)}")
    print(f"  Matched total: {len(matched_all)}")
    print(f"  Only OpenAlex rows: {len(only_openalex)}")
    print(f"  Only Crossref rows: {len(only_crossref)}")
    print(f"Wrote combined matched rows to {OUTPUT_CSV_Matched}")
    print(f"Wrote per-year outputs to {OUTPUT_BY_YEAR_DIR}")

if __name__ == "__main__":
    main()
