from pathlib import Path
import os
import re

import pandas as pd


os.chdir("/Users/yingyan_zhao/Dropbox/JournalPublicationProject")

AEA_INPUT_CSV = Path("data/raw_csv/AEA_Journals_Papers.csv")
BASE_INPUT_CSV = Path("data/processed/OpenAlex_Crossref_Webscrape_NBER_Repec_Merged.csv")
AEA_CLEANED_OUTPUT_CSV = Path("data/processed/AEA_Journals_Papers_Cleaned.csv")
MERGED_OUTPUT_CSV = Path("data/processed/OpenAlex_Crossref_Webscrape_NBER_Repec_AEA_Merged.csv")
DOI_FULL_TO_DROP_AFTER_MERGE = {
    # "10.1086/342812",
    # "10.1086/339337",
    # "10.1086/342337",
}
AEA_COLUMNS_TO_DROP_AFTER_MERGE = [
    "aea_doi",
    "aea_title",
    "aea_journal",
    "aea_publication_year",
    "aea_abstract",
    "aea_jel_codes",
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
    AEA_CLEANED_OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(AEA_CLEANED_OUTPUT_CSV, index=False)
    aea_duplicate_stats = duplicate_aea_stats(cleaned)

    print("AEA cleaning summary:")
    print(f"  Input CSV: {AEA_INPUT_CSV}")
    print(f"  Input rows: {len(aea)}")
    print(f"  Cleaned CSV: {AEA_CLEANED_OUTPUT_CSV}")
    print(f"  Output rows: {len(cleaned)}")
    print(f"  Dropped blank-author rows: {len(aea) - len(cleaned)}")
    print(f"  Duplicate DOI rows: {aea_duplicate_stats['duplicate_doi_rows']}")
    print(f"  Duplicate DOI groups: {aea_duplicate_stats['duplicate_doi_groups']}")
    print(f"  Duplicate title rows: {aea_duplicate_stats['duplicate_title_rows']}")
    print(f"  Duplicate title groups: {aea_duplicate_stats['duplicate_title_groups']}")
    print(f"  Columns: {list(cleaned.columns)}")

    base = read_base_data(BASE_INPUT_CSV)
    print()
    print("OpenAlex/Crossref/Webscrape/NBER/RePEc + AEA merge summary:")
    print(f"  Base input CSV: {BASE_INPUT_CSV}")
    print(f"  Base rows: {len(base)}")
    print(f"  Base rows with duplicated title: {duplicate_title_rows(base, 'title')}")
    print(f"  Base rows with duplicated DOI: {duplicate_doi_rows(base)}")

    merged, merge_summary = merge_base_with_aea(base, cleaned)
    merged.to_csv(MERGED_OUTPUT_CSV, index=False)
    print(f"  Output CSV: {MERGED_OUTPUT_CSV}")
    print(f"  Output rows: {len(merged)}")
    print(f"  Output rows with duplicated DOI: {duplicate_doi_rows(merged)}")
    print(f"  Output rows with duplicated title: {duplicate_exported_title_rows(merged)}")
    print(f"  Matched by doi_1: {merge_summary['matched_by_doi_1']}")
    print(f"  Matched by doi_2: {merge_summary['matched_by_doi_2']}")
    print(f"  Matched by doi_3: {merge_summary['matched_by_doi_3']}")
    print(f"  Matched by DOI total: {merge_summary['matched_by_doi_total']}")
    print(f"  Matched by normalized title: {merge_summary['matched_by_title']}")
    print(f"  Base-only rows: {merge_summary['base_only']}")
    print(f"  AEA-only rows: {merge_summary['aea_only']}")
    print(f"  Dropped base-only rows with blank doi_list: {merge_summary['dropped_base_only_blank_doi_list']}")
    print(f"  Dropped excluded doi_full rows: {merge_summary['dropped_excluded_doi_full']}")
    print(f"  Dropped Suggested by author rows: {merge_summary['dropped_suggested_by_author_rows']}")


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
    if "title" in cleaned.columns:
        cleaned["title"] = cleaned["title"].apply(keep_letters_and_numbers)
    cleaned = rename_specific_aea_titles(cleaned)
    cleaned = drop_correction_titles(cleaned)
    cleaned["title_duplicate_tag"] = duplicate_title_tag(cleaned, "title")
    cleaned = add_prefix_to_columns(cleaned, "aea_")
    return cleaned.reset_index(drop=True)


def rename_specific_aea_titles(data: pd.DataFrame) -> pd.DataFrame:
    cleaned = data.copy()
    if "title" not in cleaned.columns or "publication_year" not in cleaned.columns:
        return cleaned

    title = cleaned["title"].fillna("").astype(str).str.strip()
    publication_year = cleaned["publication_year"].fillna("").astype(str).str.strip()
    target_row = (title == "Human Capital and Growth") & (publication_year == "2015")
    cleaned.loc[target_row, "title"] = "Human Capital and Growth 2015"
    return cleaned


def add_base_duplicate_title_tag(data: pd.DataFrame) -> pd.DataFrame:
    tagged = data.copy()
    tagged["title_duplicate_tag"] = duplicate_title_tag(tagged, "title")
    return tagged


def merge_base_with_aea(
    base: pd.DataFrame,
    aea: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, int]]:
    base_prepared = base.copy()
    aea_prepared = aea.copy()

    base_prepared["_base_row_id"] = range(len(base_prepared))
    aea_prepared["_aea_row_id"] = range(len(aea_prepared))

    base_prepared["base_merge_doi_1"] = get_column(base_prepared, "doi_1").apply(normalize_doi)
    base_prepared["base_merge_doi_2"] = get_column(base_prepared, "doi_2").apply(normalize_doi)
    base_prepared["base_merge_doi_3"] = get_column(base_prepared, "doi_3").apply(normalize_doi)
    base_prepared["base_merge_title"] = get_column(base_prepared, "title").apply(normalize_title)
    aea_prepared["aea_merge_doi"] = get_column(aea_prepared, "aea_doi").apply(normalize_doi)
    aea_prepared["aea_merge_title"] = get_column(aea_prepared, "aea_title").apply(normalize_title)

    summary = {
        "matched_by_doi_1": 0,
        "matched_by_doi_2": 0,
        "matched_by_doi_3": 0,
        "matched_by_doi_total": 0,
        "matched_by_title": 0,
        "base_only": 0,
        "aea_only": 0,
        "dropped_base_only_blank_doi_list": 0,
        "dropped_excluded_doi_full": 0,
        "dropped_suggested_by_author_rows": 0,
    }

    matched_frames = []
    base_remaining = base_prepared
    aea_remaining = aea_prepared

    for doi_number in [1, 2, 3]:
        matched, base_remaining, aea_remaining = match_unique_stage(
            base_remaining,
            aea_remaining,
            base_key_column=f"base_merge_doi_{doi_number}",
            aea_key_column="aea_merge_doi",
            strategy=f"doi_{doi_number}",
        )
        summary[f"matched_by_doi_{doi_number}"] = len(matched)
        if not matched.empty:
            matched_frames.append(matched)

    summary["matched_by_doi_total"] = (
        summary["matched_by_doi_1"]
        + summary["matched_by_doi_2"]
        + summary["matched_by_doi_3"]
    )

    matched, base_remaining, aea_remaining = match_unique_stage(
        base_remaining,
        aea_remaining,
        base_key_column="base_merge_title",
        aea_key_column="aea_merge_title",
        strategy="normalized_title",
    )
    summary["matched_by_title"] = len(matched)
    if not matched.empty:
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
    merged = drop_merge_helper_columns(merged)
    merged, dropped_rows = drop_base_only_blank_doi_list(merged)
    summary["dropped_base_only_blank_doi_list"] = dropped_rows
    merged["abstract"] = prefer_nonblank_source_column(merged, "abstract", "aea_abstract")
    merged["jel_codes"] = prefer_nonblank_source_column(merged, "jel_codes", "aea_jel_codes")
    merged["doi_full"] = merged.apply(doi_full_from_row, axis=1)
    merged, dropped_excluded_doi_full = drop_excluded_doi_full(merged)
    summary["dropped_excluded_doi_full"] = dropped_excluded_doi_full
    merged, dropped_suggested_by_author_rows = drop_suggested_by_author_rows(merged)
    summary["dropped_suggested_by_author_rows"] = dropped_suggested_by_author_rows
    merged["duplicate_doi_tag"] = duplicate_doi_tag(merged)
    merged["duplicate_title_tag"] = duplicate_exported_title_tag(merged)
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
        "base_merge_doi_1",
        "base_merge_doi_2",
        "base_merge_doi_3",
        "base_merge_title",
        "aea_merge_doi",
        "aea_merge_title",
        "_sort_base",
        "_sort_aea",
    ]
    return data.drop(columns=[column for column in helper_columns if column in data.columns])


def drop_base_only_blank_doi_list(data: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    doi_list = get_column(data, "doi_list").fillna("").astype(str).str.strip()
    match_strategy = get_column(data, "aea_match_strategy").fillna("").astype(str).str.strip()
    drop_rows = (doi_list == "") & (match_strategy == "base_only")
    return data.loc[~drop_rows].copy(), int(drop_rows.sum())


def drop_excluded_doi_full(data: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    doi_full = get_column(data, "doi_full").apply(normalize_doi)
    drop_rows = doi_full.isin(DOI_FULL_TO_DROP_AFTER_MERGE)
    return data.loc[~drop_rows].copy(), int(drop_rows.sum())


def drop_suggested_by_author_rows(data: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    authors = get_column(data, "openalex_authors").fillna("").astype(str)
    raw_author_names = get_column(data, "openalex_raw_author_names").fillna("").astype(str)
    drop_rows = (
        authors.str.contains("Suggested by", case=False, regex=False)
        | raw_author_names.str.contains("Suggested by", case=False, regex=False)
    )
    return data.loc[~drop_rows].copy(), int(drop_rows.sum())


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


def prefer_source_column(
    data: pd.DataFrame,
    target_column: str,
    source_column: str,
) -> pd.Series:
    target = get_column(data, target_column).fillna("").astype(str)
    source = get_column(data, source_column).fillna("").astype(str)
    source_nonblank = source.str.strip() != ""
    return target.mask(source_nonblank, source)


def prefer_nonblank_source_column(
    data: pd.DataFrame,
    target_column: str,
    source_column: str,
) -> pd.Series:
    target = get_column(data, target_column).fillna("").astype(str)
    source = get_column(data, source_column).fillna("").astype(str)
    source_nonblank = source.str.strip() != ""
    return target.mask(source_nonblank, source)


def fill_blank_doi_from_doi_versions(data: pd.DataFrame) -> pd.Series:
    doi = get_column(data, "doi").apply(normalize_doi)
    doi_versions = data.apply(doi_versions_from_row, axis=1)
    return doi.mask(doi == "", doi_versions)


def doi_versions_from_row(row: pd.Series) -> str:
    dois = []
    for column in ["doi_1", "doi_2", "doi_3"]:
        doi = normalize_doi(row.get(column, ""))
        if doi and doi not in dois:
            dois.append(doi)
    return "; ".join(dois)


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


def keep_letters_and_numbers(value: object) -> str:
    if pd.isna(value):
        return ""

    text = str(value)
    text = re.sub(r"[^A-Za-z0-9]+", " ", text)
    return " ".join(text.split())


def duplicate_aea_stats(data: pd.DataFrame) -> dict[str, int]:
    doi = get_column(data, "aea_doi").apply(normalize_doi)
    doi = doi.loc[doi != ""]
    duplicate_doi = doi.duplicated(keep=False)

    title = get_column(data, "aea_title").apply(normalize_title)
    title = title.loc[title != ""]
    duplicate_title = title.duplicated(keep=False)

    return {
        "duplicate_doi_rows": int(duplicate_doi.sum()),
        "duplicate_doi_groups": int(doi.loc[duplicate_doi].nunique()),
        "duplicate_title_rows": int(duplicate_title.sum()),
        "duplicate_title_groups": int(title.loc[duplicate_title].nunique()),
    }


def duplicate_title_tag(data: pd.DataFrame, title_column: str) -> pd.Series:
    title = get_column(data, title_column).apply(normalize_title)
    duplicate_title = (title != "") & title.duplicated(keep=False)
    return duplicate_title.astype(int)


def duplicate_title_groups(data: pd.DataFrame, title_column: str) -> int:
    title = get_column(data, title_column).apply(normalize_title)
    title = title.loc[title != ""]
    duplicate_title = title.duplicated(keep=False)
    return int(title.loc[duplicate_title].nunique())


def duplicate_title_rows(data: pd.DataFrame, title_column: str) -> int:
    title = get_column(data, title_column).apply(normalize_title)
    title = title.loc[title != ""]
    return int(title.duplicated(keep=False).sum())


def duplicate_exported_title_rows(data: pd.DataFrame) -> int:
    return int(duplicate_exported_title_tag(data).sum())


def duplicate_exported_title_tag(data: pd.DataFrame) -> pd.Series:
    title = coalesce_title_for_duplicate_check(data).apply(normalize_title)
    duplicate_title = title.ne("") & title.duplicated(keep=False)
    return duplicate_title.astype(int)


def coalesce_title_for_duplicate_check(data: pd.DataFrame) -> pd.Series:
    title = get_column(data, "title").fillna("").astype(str)
    aea_title = get_column(data, "aea_title").fillna("").astype(str)
    title_blank = title.str.strip() == ""
    return title.mask(title_blank, aea_title)


def duplicate_doi_rows(data: pd.DataFrame) -> int:
    return int(duplicate_doi_tag(data).sum())


def duplicate_doi_tag(data: pd.DataFrame) -> pd.Series:
    rows = []
    for row_id, row in data.iterrows():
        for doi in doi_values_from_row(row):
            rows.append({"row_id": row_id, "doi": doi})

    if not rows:
        return pd.Series([0] * len(data), index=data.index)

    doi_rows = pd.DataFrame(rows).drop_duplicates()
    duplicated_doi = doi_rows["doi"].duplicated(keep=False)
    duplicate_row_ids = set(doi_rows.loc[duplicated_doi, "row_id"])
    return data.index.to_series().isin(duplicate_row_ids).astype(int)


def doi_values_from_row(row: pd.Series) -> list[str]:
    dois = []
    seen = set()
    for column in ["doi", "doi_1", "doi_2", "doi_3", "doi_list", "aea_doi"]:
        value = row.get(column, "")
        if column == "doi_list":
            candidates = str(value or "").split(";")
        else:
            candidates = [value]

        for candidate in candidates:
            doi = normalize_doi(candidate)
            if doi and doi not in seen:
                dois.append(doi)
                seen.add(doi)
    return dois


def doi_full_from_row(row: pd.Series) -> str:
    dois = []
    seen = set()
    for column in ["doi_1", "doi_2", "doi_3", "doi_list", "aea_doi"]:
        value = row.get(column, "")
        candidates = str(value or "").split(";") if column == "doi_list" else [value]
        for candidate in candidates:
            doi = normalize_doi(candidate)
            if doi and doi not in seen:
                dois.append(doi)
                seen.add(doi)
    return "; ".join(dois)


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


def drop_correction_titles(data: pd.DataFrame) -> pd.DataFrame:
    correction_patterns = [
        "Correction:",
        "A Correction",
        "Correction to",
        "Erratum",
        "Corrigendum",
        "comment",
        "report of",
        "Editors' Introduction",
        "Editor's Introduction",
        "Foreword",
        "reply",
        "Editorial Announcement",
        "Ad Hoc Search Committee",
        "Executive Committee",
        "List of Online Reports",
        "Book Review",
        "Frontmatter of Econometrica",
        "Backmatter of Econometrica ",
        "Ad Hoc Committee",
        "Journal of Economic Perspectives",
        "American Economic Journal",
        "Journal of Economic Literature",
        "Committee on",
        "Index to Volume",
        "Recent Referees",
        "Minutes of the Annual Meeting",
        "Journal of Political Economy",
        "John Bates Clark Award",
        "Appendix",
        "American Economic Association",
        "American Economic Review",
        "Job Openings for Economists",
        "OUP accepted manuscript",
        "Minutes of the Annual Business Meeting",
        "Front Matter",
        "The Econometric Society Annual Reports Econometrica",
        "Correction:",
        "A Correction",
        "Correction to",
        "Erratum",
        "Corrigendum",
        "comment",
        "report of",
        "Editors' Introduction",
        "Editor's Introduction",
        "Foreword",
        "reply",
        "Editorial Announcement",
        "Ad Hoc Search Committee",
        "Executive Committee",
        "List of Online Reports",
        "Book Review",
        "Frontmatter of Econometrica",
        "Backmatter of Econometrica",
        "Ad Hoc Committee",
        "Committee on",
        "Index to Volume",
        "Recent Referees",
        "Minutes of the Annual Meeting",
        "John Bates Clark Award",
        "Appendix",
        "Job Openings for Economists",
        "OUP accepted manuscript",
        "Accepted Manuscripts",
        "Acknowledgment of Referees",
        "Acknowledgement of Referees",
        "Acknowledgment to Referees",
        "Acknowledgements to Referees",
        "Abstracts",
        "Minutes of the Annual Business Meeting",
        "Front Matter",
        "The Econometric Society Annual Reports Econometrica",
        "Announcements",
        "Independent Auditors' Report",
        "The Marriage Squeeze Interpretation of Dowry Inflation: Response",
        "Forthcoming Papers",
        "Data on Time to First Decision",
        "Election of Fellows to the Econometric Society",
        "North American Summer Meeting of the Econometric Society",
        "Lucas Prize Announcement",
        "Back Cover",
        "News Notes",
        "Nobel Lecture:",
        "Meeting of the Econometric Society",
        "Submission of Manuscripts to Econometrica",
        "Submission Fees and Response Times in Academic Publishing",
        "Submission of Manuscripts",
        "Subscription Page",
        "Table of Content",
        "The Econometric Society Annual Reports",
        "The Quarterly Journal of Economics",
        "An Astonishing Sixty Years The Legacy of Hiroshima",
        "the Diamond Water Paradox",
        "General Information on the Association",
        "Information on the Association",
        "Private and Social Rates of Return to Education of Academicians Note",
        "Protectionism through Prostitution",
        "Voltaire on Labor Markets and Monetary Policy",
        "Private and Social Rates of Return to Education of Academicians Note",
        "Fellows of the Econometric Society",
        "Galileo on the Diamond/Water Paradox",
        "Independent Auditor's Report",
        "JPE Submissions",
        "JPE Turnaround Times",
        "JPE Turnaround Times, Previous Two Years",
        "Referee List",
        "Title Page",
        "Editors Introduction",
        "Editor s Introduction",
        "Editor s Note",
        "Report by the AEA Data Editor",
        "AEA Data and Code Availability Policy",
        "Note from the AEA Secretary Treasurer about the Proceedings Supplement",
        "INDEPENDENT AUDITOR S REPORT",
        "Independent Auditor s Report",
        "Behavior of the Firm Under Regulatory Constraint",
        "Auditors Report Audited Financial Statements",
        "INDEPENDENT AUDITOR S REPORT",
        "John Bates Clark Medalist"
    ]

    pattern = "|".join(re.escape(phrase) for phrase in correction_patterns)

    correction_title = data["title"].fillna("").str.contains(
        pattern,
        case=False,
        regex=True,
    )

    return data.loc[~correction_title].copy()


if __name__ == "__main__":
    main()
