from pathlib import Path
import html
import json
import math
import os

import pandas as pd


os.chdir("/Users/yingyan_zhao/Dropbox/JournalPublicationProject")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

PAPER_AUTHOR_INPUT_CSV = (
    Path("data/processed/author_names")
    / "JEL_Training_Data_PaperAuthor_WithAuthorID.csv"
)
PAPER_AUTHOR_CLEANED_CSV = (
    Path("data/processed")
    / "JEL_Training_Data_PaperAuthor_WithAuthorID_Cleaned.csv"
)
JEL_COMPLETE_INPUT_CSV = (
    Path("data/trainingmodel")
    / "JEL_Training_Data_Complete_Observed_And_Predicted.csv"
)
JEL_COMPLETE_CLEANED_CSV = (
    Path("data/processed")
    / "JEL_Training_Data_Complete_Observed_And_Predicted_Cleaned.csv"
)
PAPER_AUTHOR_JEL_MERGED_CSV = (
    Path("data/processed")
    / "JEL_Training_Data_PaperAuthor_WithAuthorID_JEL_Merged.csv"
)
PAPER_AUTHOR_JEL_TOP5_CSV = (
    Path("data/processed")
    / "JEL_Training_Data_PaperAuthor_WithAuthorID_JEL_Merged_Top5.csv"
)
ANALYSIS_OUTPUT_DIR = Path("data/processed/author_concentration")
OVERALL_OUTPUT_DIR = ANALYSIS_OUTPUT_DIR / "overall"
FIELD_OUTPUT_DIR = ANALYSIS_OUTPUT_DIR / "by_field"
ANALYSIS_SUMMARY_CSV = ANALYSIS_OUTPUT_DIR / "Analysis_Summary.csv"

PAPER_AUTHOR_COLUMNS_TO_KEEP = [
    "doi_full",
    "final_last_name",
    "final_first_name",
    "author_id",
]
JEL_COMPLETE_COLUMNS_TO_KEEP = [
    "doi",
    "title",
    "journalname",
    "publication_year",
    "jel_code_full",
    "tfidf_predicted_jel_code_full",
    "tfidf_max_confidence",
    "specter2_predicted_jel_code_full",
    "specter2_max_confidence",
    "scibert_predicted_jel_code_full",
    "scibert_max_confidence",
    "ensemble_predicted_jel_code_full",
    "ensemble_max_confidence",
]
TOP_FIVE_JOURNALS = {
    "american economic review": "American Economic Review",
    "journal of political economy": "Journal of Political Economy",
    "econometrica": "Econometrica",
    "review of economic studies": "Review of Economic Studies",
    "quarterly journal of economics": "Quarterly Journal of Economics",
    "the quarterly journal of economics": "Quarterly Journal of Economics",
}
JEL_FIELDS = ("D", "C", "E", "J", "H", "O", "L", "G", "F", "I")
TOP_AUTHOR_PERCENTAGES = (1, 5, 10)
GRAPH1_HIGHLIGHT_PERCENTAGE = 10
ROLLING_WINDOW_YEARS = 20
AUTHOR_ENTRY_COHORTS = [
    (1981, 1990, "1981-1990"),
    (1991, 2000, "1991-2000"),
    (2001, 2010, "2001-2010"),
    (2011, 2020, "2011-2020"),
]


def main() -> None:
    # Step 1. Keep only the requested paper-author columns.
    paper_author = read_csv_data(PAPER_AUTHOR_INPUT_CSV)
    paper_author_cleaned = keep_columns(
        paper_author,
        PAPER_AUTHOR_COLUMNS_TO_KEEP,
    )
    write_csv(paper_author_cleaned, PAPER_AUTHOR_CLEANED_CSV)
    print_progress(
        step=1,
        scope="Data preparation",
        message=(
            f"kept {len(paper_author_cleaned):,} paper-author rows and "
            f"wrote {PAPER_AUTHOR_CLEANED_CSV}"
        ),
    )

    # Step 2. Keep only the requested paper and JEL columns.
    jel_complete = read_csv_data(JEL_COMPLETE_INPUT_CSV)
    jel_complete_cleaned = keep_columns(
        jel_complete,
        JEL_COMPLETE_COLUMNS_TO_KEEP,
    )
    write_csv(jel_complete_cleaned, JEL_COMPLETE_CLEANED_CSV)
    print_progress(
        step=2,
        scope="Data preparation",
        message=(
            f"kept {len(jel_complete_cleaned):,} paper rows and wrote "
            f"{JEL_COMPLETE_CLEANED_CSV}"
        ),
    )

    # Step 3. Merge paper-author rows to paper data in a many-to-one merge.
    merged = merge_paper_author_with_jel(
        paper_author_cleaned,
        jel_complete_cleaned,
    )
    write_csv(merged, PAPER_AUTHOR_JEL_MERGED_CSV)
    matched_rows = int(merged["paper_jel_merge_status"].eq("both").sum())
    unmatched_rows = int(
        merged["paper_jel_merge_status"].eq("left_only").sum()
    )
    print_progress(
        step=3,
        scope="Data preparation",
        message=(
            f"merged {len(merged):,} rows: {matched_rows:,} matched and "
            f"{unmatched_rows:,} paper-author-only"
        ),
    )

    # Step 4. Keep top-five journals and construct the final JEL classification.
    top5 = keep_top_five_journals(merged)
    top5 = add_final_jel_code(top5)
    write_csv(top5, PAPER_AUTHOR_JEL_TOP5_CSV)
    final_jel_nonblank = int(
        top5["Final_jel_code"].fillna("").astype(str).str.strip().ne("").sum()
    )
    print_progress(
        step=4,
        scope="Data preparation",
        message=(
            f"kept {len(top5):,} author-paper rows across "
            f"{top5['doi_full'].nunique():,} top-five papers; "
            f"{final_jel_nonblank:,} rows have Final_jel_code"
        ),
    )

    # Steps 5-9. Produce the five analyses for all top-five papers.
    print("\nStarting Steps 5-9 for All fields...", flush=True)
    summaries = [
        run_analysis(
            data=top5,
            output_dir=OVERALL_OUTPUT_DIR,
            analysis_label="All fields",
            minimum_gap_authors=30,
        )
    ]

    # Step 10. Repeat Steps 5-9 separately for each requested JEL field.
    for field_number, field in enumerate(JEL_FIELDS, start=1):
        field_data = keep_jel_field(top5, field)
        print(
            f"\nStarting Steps 5-9 for JEL {field} "
            f"({field_number}/{len(JEL_FIELDS)})...",
            flush=True,
        )
        summaries.append(
            run_analysis(
                data=field_data,
                output_dir=FIELD_OUTPUT_DIR / field,
                analysis_label=f"JEL {field}",
                minimum_gap_authors=10,
            )
        )
        print_progress(
            step=10,
            scope=f"JEL {field}",
            message=(
                f"finished field {field_number}/{len(JEL_FIELDS)} with "
                f"{field_data['doi_full'].nunique():,} unique papers"
            ),
        )

    summary = pd.DataFrame(summaries)
    write_csv(summary, ANALYSIS_SUMMARY_CSV)
    print_progress(
        step=10,
        scope="All requested fields",
        message=(
            f"completed {len(JEL_FIELDS)} field analyses and wrote "
            f"{ANALYSIS_SUMMARY_CSV}"
        ),
    )
    print_run_summary(
        paper_author=paper_author,
        jel_complete=jel_complete,
        merged=merged,
        top5=top5,
        summary=summary,
    )


def run_analysis(
    data: pd.DataFrame,
    output_dir: Path,
    analysis_label: str,
    minimum_gap_authors: int,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paper_years = prepare_paper_years(data)
    author_papers = prepare_author_papers(data)

    # Step 5 / Graph 1.
    top_author_shares, top_author_list = rolling_top_author_paper_shares(
        paper_years,
        author_papers,
    )
    write_csv(
        top_author_shares,
        output_dir / "Graph1_Rolling20Year_TopAuthorPaperShares.csv",
    )
    write_csv(
        top_author_list,
        output_dir / "Graph1_Rolling20Year_TopAuthors.csv",
    )
    plot_top_author_shares(
        top_author_shares,
        output_dir / "Graph1_TopAuthorPaperShares_After1980.png",
        analysis_label,
        from_year=1980,
    )
    write_interactive_top_author_shares(
        top_author_shares,
        output_dir / "Graph1_TopAuthorPaperShares_After1980.html",
        analysis_label,
        from_year=1980,
    )
    print_progress(
        step=5,
        scope=analysis_label,
        message=(
            f"Graph 1 PNG and interactive HTML complete for "
            f"{paper_years['publication_year'].nunique():,} publication years"
        ),
    )

    # Step 6 / Graphs 2.1 and 2.2.
    new_author_summary = new_author_statistics_by_year(
        paper_years,
        author_papers,
    )
    write_csv(
        new_author_summary,
        output_dir / "Graph2_NewAuthors_ByYear.csv",
    )
    plot_share_only(
        new_author_summary,
        output_dir / "Graph2_1_NewAuthorCountShare_1980_2025.png",
        analysis_label=analysis_label,
        from_year=1980,
        to_year=2025,
        share_min=0.30,
        share_max=0.50,
        share_column="share_of_authors_who_are_new",
        title="New authors account for a smaller share of authors over time.",
        subtitle=(
            f"{analysis_label} \N{MIDDLE DOT} New author defined as first "
            "observed publication in Top 5 journals"
        ),
        share_label="Share of authors who are new",
        end_label="New authors\N{RIGHT SINGLE QUOTATION MARK} share",
        annotation_text=(
            "Despite year-to-year fluctuations, the share of new authors "
            "trends downward\nfrom about 45% in 1980 to about 36% in 2025."
            if analysis_label == "All fields"
            else ""
        ),
        line_color="#087E73",
    )
    write_interactive_new_author_share(
        new_author_summary,
        output_dir / "Graph2_1_NewAuthorShare_1980_2025.html",
        analysis_label=analysis_label,
        from_year=1980,
        to_year=2025,
        share_column="share_of_authors_who_are_new",
    )
    plot_share_and_count(
        new_author_summary,
        output_dir / "Graph2_2_PapersWithNewAuthorCountShare_1980_2025.png",
        analysis_label=analysis_label,
        from_year=1980,
        to_year=2025,
        share_min=0.30,
        share_max=0.70,
        share_column="share_of_papers_with_new_author",
        count_column="unique_papers_with_new_author",
        title="Papers with a New Author",
        share_label="Share of papers with a new author",
        count_label="Number of papers with a new author",
    )
    print_progress(
        step=6,
        scope=analysis_label,
        message=(
            f"Graph 2.1 PNG/interactive HTML and Graph 2.2 PNG complete "
            f"with {len(new_author_summary):,} annual observations"
        ),
    )

    # Step 7 / Graph 3.
    new_author_types, new_author_type_summary = (
        classify_new_author_first_publications(author_papers)
    )
    write_csv(
        new_author_types,
        output_dir / "Graph3_NewAuthorCoauthorType.csv",
    )
    write_csv(
        new_author_type_summary,
        output_dir / "Graph3_NewAuthorCoauthorType_ByYear.csv",
    )
    plot_new_author_coauthor_types(
        new_author_type_summary,
        output_dir / "Graph3_NewAuthorCoauthorType_1980_2025.png",
        analysis_label,
        from_year=1980,
        to_year=2025,
    )
    write_interactive_new_author_coauthor_types(
        new_author_type_summary,
        output_dir / "Graph3_NewAuthorCoauthorType_1980_2025.html",
        analysis_label,
        from_year=1980,
        to_year=2025,
    )
    print_progress(
        step=7,
        scope=analysis_label,
        message=(
            f"Graph 3 PNG and interactive HTML complete for "
            f"{len(new_author_types):,} new authors"
        ),
    )
    # Step 8 / Graph 4.
    publication_gaps = consecutive_publication_gaps(author_papers)
    publication_gap_by_cohort = summarize_publication_gaps_by_cohort(
        publication_gaps
    )
    write_csv(
        publication_gaps,
        output_dir / "Graph4_ConsecutivePublicationGaps.csv",
    )
    write_csv(
        publication_gap_by_cohort,
        output_dir / "Graph4_ConsecutivePublicationGaps_ByCohort.csv",
    )
    plot_publication_gaps_by_cohort(
        publication_gap_by_cohort,
        output_dir / "Graph4_ConsecutivePublicationGaps_ByCohort.png",
        analysis_label,
        minimum_authors=minimum_gap_authors,
    )
    write_interactive_publication_gaps_by_cohort(
        publication_gap_by_cohort,
        output_dir / "Graph4_ConsecutivePublicationGaps_ByCohort.html",
        analysis_label,
        minimum_authors=minimum_gap_authors,
    )
    print_progress(
        step=8,
        scope=analysis_label,
        message=(
            f"Graph 4 PNG and interactive HTML complete from "
            f"{len(publication_gaps):,} consecutive publication gaps"
        ),
    )
    # Step 9 / Graph 5.
    first_to_second_gap = summarize_first_to_second_gap_by_first_year(
        publication_gaps
    )
    write_csv(
        first_to_second_gap,
        output_dir / "Graph5_FirstToSecondPublicationGap_ByFirstYear.csv",
    )
    plot_first_to_second_gap_by_first_year(
        first_to_second_gap,
        output_dir / "Graph5_FirstToSecondPublicationGap_1980_2020.png",
        analysis_label,
        from_year=1980,
        to_year=2020,
    )
    write_interactive_first_to_second_gap_by_first_year(
        first_to_second_gap,
        output_dir / "Graph5_FirstToSecondPublicationGap_1980_2020.html",
        analysis_label,
        from_year=1980,
        to_year=2020,
    )
    authors_with_second_publication = int(
        first_to_second_gap["number_of_authors"].sum()
        if not first_to_second_gap.empty
        else 0
    )
    print_progress(
        step=9,
        scope=analysis_label,
        message=(
            f"Graph 5 PNG and interactive HTML complete for "
            f"{authors_with_second_publication:,} authors with a second publication"
        ),
    )

    return {
        "analysis": analysis_label,
        "paper_author_rows": len(data),
        "unique_papers": int(paper_years["doi_full"].nunique()),
        "unique_authors": int(author_papers["author_id"].nunique()),
        "output_directory": str(output_dir),
    }


def read_csv_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing input CSV: {path}")
    return pd.read_csv(path, dtype=str, keep_default_na=False, low_memory=False)


def print_progress(step: int, scope: str, message: str) -> None:
    print(f"[Step {step} complete] {scope}: {message}", flush=True)


def keep_columns(data: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    missing_columns = [column for column in columns if column not in data.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    return data[columns].copy()


def write_csv(data: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(path, index=False)


def merge_paper_author_with_jel(
    paper_author: pd.DataFrame,
    jel_complete: pd.DataFrame,
) -> pd.DataFrame:
    jel_for_merge = jel_complete.copy()
    jel_for_merge["doi"] = jel_for_merge["doi"].fillna("").astype(str).str.strip()
    jel_for_merge = jel_for_merge.loc[jel_for_merge["doi"].ne("")].copy()
    duplicate_doi = jel_for_merge["doi"].duplicated(keep=False)
    if duplicate_doi.any():
        raise ValueError(
            "Complete JEL data must be unique by nonblank doi for an m:1 merge. "
            f"Duplicated rows: {int(duplicate_doi.sum())}"
        )

    return paper_author.merge(
        jel_for_merge,
        left_on="doi_full",
        right_on="doi",
        how="left",
        validate="many_to_one",
        indicator="paper_jel_merge_status",
    )


def keep_top_five_journals(data: pd.DataFrame) -> pd.DataFrame:
    cleaned = data.copy()
    normalized = cleaned["journalname"].apply(normalize_journalname)
    cleaned["journalname"] = normalized.map(TOP_FIVE_JOURNALS).fillna(
        cleaned["journalname"]
    )
    return cleaned.loc[normalized.isin(TOP_FIVE_JOURNALS)].copy()


def normalize_journalname(value) -> str:
    return " ".join(str(value or "").strip().lower().split())


def add_final_jel_code(data: pd.DataFrame) -> pd.DataFrame:
    cleaned = data.copy()
    observed = cleaned["jel_code_full"].fillna("").astype(str).str.strip()
    predicted = (
        cleaned["scibert_predicted_jel_code_full"]
        .fillna("")
        .astype(str)
        .str.strip()
    )
    cleaned["Final_jel_code"] = observed.mask(observed.eq(""), predicted)
    return cleaned


def keep_jel_field(data: pd.DataFrame, field: str) -> pd.DataFrame:
    field_mask = data["Final_jel_code"].apply(
        lambda value: field in split_jel_fields(value)
    )
    return data.loc[field_mask].copy()


def split_jel_fields(value) -> set[str]:
    return {
        part.strip().upper()
        for part in str(value or "").split(";")
        if part.strip()
    }


def prepare_paper_years(data: pd.DataFrame) -> pd.DataFrame:
    papers = data[["doi_full", "publication_year"]].copy()
    papers["doi_full"] = papers["doi_full"].fillna("").astype(str).str.strip()
    papers["publication_year"] = pd.to_numeric(
        papers["publication_year"],
        errors="coerce",
    )
    papers = papers.loc[
        papers["doi_full"].ne("") & papers["publication_year"].notna()
    ].copy()
    papers["publication_year"] = papers["publication_year"].astype(int)
    return papers.drop_duplicates("doi_full").reset_index(drop=True)


def prepare_author_papers(data: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "doi_full",
        "publication_year",
        "author_id",
        "final_first_name",
        "final_last_name",
    ]
    author_papers = data[columns].copy()
    author_papers["doi_full"] = (
        author_papers["doi_full"].fillna("").astype(str).str.strip()
    )
    author_papers["author_id"] = (
        author_papers["author_id"].fillna("").astype(str).str.strip()
    )
    author_papers["publication_year"] = pd.to_numeric(
        author_papers["publication_year"],
        errors="coerce",
    )
    author_papers = author_papers.loc[
        author_papers["doi_full"].ne("")
        & author_papers["author_id"].ne("")
        & author_papers["publication_year"].notna()
    ].copy()
    author_papers["publication_year"] = author_papers[
        "publication_year"
    ].astype(int)
    return author_papers.drop_duplicates(["author_id", "doi_full"]).reset_index(
        drop=True
    )


def rolling_top_author_paper_shares(
    paper_years: pd.DataFrame,
    author_papers: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows = []
    author_rows = []
    for year in sorted(paper_years["publication_year"].unique()):
        history_start = year - ROLLING_WINDOW_YEARS
        history_end = year - 1
        history = author_papers.loc[
            author_papers["publication_year"].between(
                history_start,
                history_end,
                inclusive="both",
            )
        ]
        current = author_papers.loc[author_papers["publication_year"].eq(year)]
        total_papers = int(
            paper_years.loc[
                paper_years["publication_year"].eq(year),
                "doi_full",
            ].nunique()
        )
        author_counts = (
            history.groupby("author_id")["doi_full"]
            .nunique()
            .sort_values(ascending=False, kind="mergesort")
        )
        names = author_name_map(history)

        for percentage in TOP_AUTHOR_PERCENTAGES:
            selected, cutoff, nominal_count = select_top_authors(
                author_counts,
                percentage,
            )
            papers_with_top_author = int(
                current.loc[current["author_id"].isin(selected), "doi_full"].nunique()
            )
            summary_rows.append(
                {
                    "publication_year": year,
                    "history_start_year": history_start,
                    "history_end_year": history_end,
                    "top_author_percentage": percentage,
                    "number_of_eligible_authors": len(author_counts),
                    "nominal_number_of_top_authors": nominal_count,
                    "number_of_top_authors_including_ties": len(selected),
                    "publication_count_cutoff": cutoff,
                    "total_unique_papers": total_papers,
                    "unique_papers_with_top_author": papers_with_top_author,
                    "share_of_papers_with_top_author": round(
                        papers_with_top_author / total_papers if total_papers else 0,
                        6,
                    ),
                }
            )
            for author_id in selected:
                first_name, last_name = names.get(author_id, ("", ""))
                author_rows.append(
                    {
                        "publication_year": year,
                        "history_start_year": history_start,
                        "history_end_year": history_end,
                        "top_author_percentage": percentage,
                        "author_id": author_id,
                        "final_first_name": first_name,
                        "final_last_name": last_name,
                        "rolling_publication_count": int(
                            author_counts.get(author_id, 0)
                        ),
                    }
                )
    return pd.DataFrame(summary_rows), pd.DataFrame(author_rows)


def select_top_authors(
    author_counts: pd.Series,
    percentage: int,
) -> tuple[list[str], int | str, int]:
    if author_counts.empty:
        return [], "", 0
    nominal_count = max(1, math.ceil(len(author_counts) * percentage / 100))
    cutoff = int(author_counts.iloc[nominal_count - 1])
    selected = author_counts.loc[author_counts.ge(cutoff)]
    return selected.index.astype(str).tolist(), cutoff, nominal_count


def new_author_statistics_by_year(
    paper_years: pd.DataFrame,
    author_papers: pd.DataFrame,
) -> pd.DataFrame:
    first_year = author_papers.groupby("author_id")["publication_year"].min()
    rows = []
    for year in sorted(paper_years["publication_year"].unique()):
        current = author_papers.loc[author_papers["publication_year"].eq(year)]
        current_authors = set(current["author_id"])
        new_authors = {
            author_id
            for author_id in current_authors
            if first_year.get(author_id) == year
        }
        total_papers = int(
            paper_years.loc[
                paper_years["publication_year"].eq(year),
                "doi_full",
            ].nunique()
        )
        papers_with_new_author = int(
            current.loc[current["author_id"].isin(new_authors), "doi_full"].nunique()
        )
        rows.append(
            {
                "publication_year": year,
                "number_of_new_authors": len(new_authors),
                "total_unique_authors": len(current_authors),
                "share_of_authors_who_are_new": round(
                    len(new_authors) / len(current_authors) if current_authors else 0,
                    6,
                ),
                "unique_papers_with_new_author": papers_with_new_author,
                "total_unique_papers": total_papers,
                "share_of_papers_with_new_author": round(
                    papers_with_new_author / total_papers if total_papers else 0,
                    6,
                ),
            }
        )
    return pd.DataFrame(rows)


def classify_new_author_first_publications(
    author_papers: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    first_year = author_papers.groupby("author_id")["publication_year"].min()
    first_year_papers = author_papers.loc[
        author_papers["publication_year"].eq(
            author_papers["author_id"].map(first_year)
        )
    ]
    paper_authors = author_papers.groupby("doi_full")["author_id"].apply(set).to_dict()
    names = author_name_map(author_papers)
    rows = []

    for author_id, group in first_year_papers.groupby("author_id", sort=False):
        entry_year = int(first_year.loc[author_id])
        has_experienced = False
        has_other_new = False
        has_coauthor = False
        for doi in group["doi_full"].unique():
            coauthors = paper_authors.get(doi, set()) - {author_id}
            has_coauthor = has_coauthor or bool(coauthors)
            has_experienced = has_experienced or any(
                first_year.get(coauthor, entry_year) < entry_year
                for coauthor in coauthors
            )
            has_other_new = has_other_new or any(
                first_year.get(coauthor) == entry_year for coauthor in coauthors
            )

        if has_experienced:
            category = "with_experienced_coauthor"
        elif has_other_new:
            category = "only_with_other_new_coauthors"
        else:
            category = "solo"
        first_name, last_name = names.get(author_id, ("", ""))
        rows.append(
            {
                "author_id": author_id,
                "final_first_name": first_name,
                "final_last_name": last_name,
                "first_top5_publication_year": entry_year,
                "number_of_papers_in_first_year": int(group["doi_full"].nunique()),
                "has_any_coauthor": int(has_coauthor),
                "has_experienced_coauthor": int(has_experienced),
                "has_other_new_coauthor": int(has_other_new),
                "first_publication_coauthor_type": category,
            }
        )

    detail = pd.DataFrame(rows)
    if detail.empty:
        return detail, pd.DataFrame()
    counts = (
        detail.groupby(
            ["first_top5_publication_year", "first_publication_coauthor_type"]
        )["author_id"]
        .nunique()
        .unstack(fill_value=0)
    )
    categories = [
        "with_experienced_coauthor",
        "only_with_other_new_coauthors",
        "solo",
    ]
    for category in categories:
        if category not in counts.columns:
            counts[category] = 0
    counts = counts.reset_index()
    counts["total_new_authors"] = counts[categories].sum(axis=1)
    for category in categories:
        counts[f"share_{category}"] = (
            counts[category] / counts["total_new_authors"]
        ).round(6)
    return detail, counts


def consecutive_publication_gaps(author_papers: pd.DataFrame) -> pd.DataFrame:
    papers = author_papers.sort_values(
        ["author_id", "publication_year", "doi_full"],
        kind="mergesort",
    ).copy()
    papers["publication_number"] = papers.groupby("author_id").cumcount() + 1
    papers["previous_publication_year"] = papers.groupby("author_id")[
        "publication_year"
    ].shift(1)
    first_year = papers.groupby("author_id")["publication_year"].min()
    gaps = papers.loc[papers["publication_number"].ge(2)].copy()
    gaps["previous_publication_year"] = gaps[
        "previous_publication_year"
    ].astype(int)
    gaps["gap_years"] = gaps["publication_year"] - gaps["previous_publication_year"]
    gaps["from_publication_number"] = gaps["publication_number"] - 1
    gaps["to_publication_number"] = gaps["publication_number"]
    gaps["publication_transition"] = (
        gaps["from_publication_number"].astype(str)
        + " to "
        + gaps["to_publication_number"].astype(str)
    )
    gaps["first_top5_publication_year"] = gaps["author_id"].map(first_year)
    gaps["entry_cohort"] = gaps["first_top5_publication_year"].apply(
        author_entry_cohort
    )
    return gaps[
        [
            "author_id",
            "final_first_name",
            "final_last_name",
            "first_top5_publication_year",
            "entry_cohort",
            "doi_full",
            "from_publication_number",
            "to_publication_number",
            "publication_transition",
            "previous_publication_year",
            "publication_year",
            "gap_years",
        ]
    ].copy()


def summarize_publication_gaps_by_cohort(gaps: pd.DataFrame) -> pd.DataFrame:
    selected = gaps.loc[gaps["entry_cohort"].ne("")].copy()
    if selected.empty:
        return pd.DataFrame()
    summary = (
        selected.groupby(
            [
                "entry_cohort",
                "from_publication_number",
                "to_publication_number",
                "publication_transition",
            ]
        )
        .agg(
            number_of_authors=("author_id", "nunique"),
            average_gap_years=("gap_years", "mean"),
            median_gap_years=("gap_years", "median"),
            share_same_year=("gap_years", lambda values: values.eq(0).mean()),
        )
        .reset_index()
    )
    cohort_order = {
        label: order
        for order, (_, _, label) in enumerate(AUTHOR_ENTRY_COHORTS)
    }
    summary["_cohort_order"] = summary["entry_cohort"].map(cohort_order)
    summary = summary.sort_values(
        ["_cohort_order", "to_publication_number"],
        kind="mergesort",
    ).drop(columns="_cohort_order")
    return round_columns(
        summary.reset_index(drop=True),
        ["average_gap_years", "median_gap_years", "share_same_year"],
    )


def summarize_first_to_second_gap_by_first_year(
    gaps: pd.DataFrame,
) -> pd.DataFrame:
    selected = gaps.loc[gaps["from_publication_number"].eq(1)].copy()
    if selected.empty:
        return pd.DataFrame()
    summary = (
        selected.groupby(
            [
                "from_publication_number",
                "to_publication_number",
                "publication_transition",
                "previous_publication_year",
            ]
        )
        .agg(
            number_of_authors=("author_id", "nunique"),
            average_gap_years=("gap_years", "mean"),
            median_gap_years=("gap_years", "median"),
            share_same_year=("gap_years", lambda values: values.eq(0).mean()),
        )
        .reset_index()
        .sort_values("previous_publication_year")
    )
    return round_columns(
        summary,
        ["average_gap_years", "median_gap_years", "share_same_year"],
    )


def author_entry_cohort(year) -> str:
    for start_year, end_year, label in AUTHOR_ENTRY_COHORTS:
        if start_year <= int(year) <= end_year:
            return label
    return ""


def author_name_map(data: pd.DataFrame) -> dict[str, tuple[str, str]]:
    names = {}
    for author_id, group in data.groupby("author_id", sort=False):
        names[str(author_id)] = (
            longest_nonblank(group["final_first_name"]),
            longest_nonblank(group["final_last_name"]),
        )
    return names


def longest_nonblank(values: pd.Series) -> str:
    cleaned = values.fillna("").astype(str).str.strip()
    cleaned = cleaned.loc[cleaned.ne("")]
    if cleaned.empty:
        return ""
    return max(cleaned, key=lambda value: (len(value), value))


def round_columns(data: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    rounded = data.copy()
    for column in columns:
        rounded[column] = rounded[column].round(6)
    return rounded


def write_interactive_top_author_shares(
    data: pd.DataFrame,
    output_path: Path,
    analysis_label: str,
    from_year: int,
) -> None:
    plot_data = data.loc[data["publication_year"].ge(from_year)].copy()
    if plot_data.empty:
        return

    series_data = {}
    for percentage in TOP_AUTHOR_PERCENTAGES:
        series = plot_data.loc[
            plot_data["top_author_percentage"].eq(percentage)
        ].sort_values("publication_year")
        series_data[str(percentage)] = [
            [
                int(row.publication_year),
                round(float(row.share_of_papers_with_top_author), 6),
            ]
            for row in series.itertuples(index=False)
        ]

    document = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>A growing share of papers include top-ranked authors</title>
  <style>
    :root {
      color-scheme: light dark;
      --background: #ffffff;
      --foreground: #1e252b;
      --muted-foreground: #68737e;
      --grid: #dce1e6;
      --axis: #aeb7c0;
      --series-1: #b6bdc5;
      --series-5: #e67e22;
      --series-10: #087e73;
      --tooltip-background: #1e252b;
      --tooltip-foreground: #ffffff;
    }
    @media (prefers-color-scheme: dark) {
      :root {
        --background: #171a1d;
        --foreground: #f1f3f4;
        --muted-foreground: #b2bac2;
        --grid: #343a40;
        --axis: #626c75;
        --series-1: #7f8993;
        --series-5: #f3a35c;
        --series-10: #46c6b8;
        --tooltip-background: #f1f3f4;
        --tooltip-foreground: #171a1d;
      }
    }
    * {
      box-sizing: border-box;
      letter-spacing: 0;
    }
    body {
      margin: 0;
      background: var(--background);
      color: var(--foreground);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    .chart {
      position: relative;
      width: 100%;
      max-width: 1100px;
      margin: 0 auto;
      padding: 24px;
    }
    h1 {
      margin: 0 0 6px;
      font-size: 28px;
      font-weight: 500;
    }
    .subtitle {
      margin: 0 0 14px;
      color: var(--muted-foreground);
      font-size: 15px;
    }
    .legend {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-bottom: 8px;
    }
    .legend button {
      display: inline-flex;
      align-items: center;
      gap: 7px;
      min-height: 34px;
      padding: 6px 10px;
      border: 1px solid transparent;
      border-radius: 4px;
      background: transparent;
      color: var(--muted-foreground);
      font: inherit;
      cursor: pointer;
    }
    .legend button:hover,
    .legend button:focus-visible,
    .legend button[aria-pressed="true"] {
      border-color: var(--axis);
      color: var(--foreground);
      outline-offset: 2px;
    }
    .swatch {
      width: 22px;
      height: 3px;
      background: var(--swatch);
    }
    .chart-svg {
      display: block;
      width: 100%;
      height: auto;
      overflow: visible;
    }
    .grid-line {
      stroke: var(--grid);
      stroke-width: 1;
    }
    .axis-line {
      stroke: var(--axis);
      stroke-width: 1;
    }
    .axis-label,
    .tick-label,
    .end-label {
      fill: var(--muted-foreground);
      font-family: inherit;
    }
    .axis-label {
      font-size: 14px;
    }
    .tick-label,
    .end-label {
      font-size: 12px;
    }
    .annotation {
      fill: var(--muted-foreground);
      font-family: inherit;
      font-size: 12px;
      pointer-events: none;
    }
    .annotation-line {
      stroke: var(--muted-foreground);
      stroke-width: 1;
      pointer-events: none;
    }
    .series-line {
      fill: none;
      stroke-linecap: round;
      stroke-linejoin: round;
      stroke-width: 2;
      opacity: 0.32;
      transition: opacity 160ms ease, stroke-width 160ms ease;
    }
    .series-line[data-series="1"] { stroke: var(--series-1); }
    .series-line[data-series="5"] { stroke: var(--series-5); }
    .series-line[data-series="10"] { stroke: var(--series-10); }
    .series-line.is-active {
      stroke-width: 4;
      opacity: 1;
    }
    .hit-line {
      fill: none;
      stroke: transparent;
      stroke-width: 18;
      pointer-events: stroke;
      cursor: crosshair;
    }
    .hover-marker {
      stroke: var(--background);
      stroke-width: 2;
      display: none;
      pointer-events: none;
    }
    .hover-marker[data-series="1"] { fill: var(--series-1); }
    .hover-marker[data-series="5"] { fill: var(--series-5); }
    .hover-marker[data-series="10"] { fill: var(--series-10); }
    .tooltip {
      position: absolute;
      z-index: 2;
      transform: translate(-50%, calc(-100% - 10px));
      max-width: 180px;
      padding: 7px 9px;
      border-radius: 4px;
      background: var(--tooltip-background);
      color: var(--tooltip-foreground);
      font-size: 13px;
      pointer-events: none;
    }
    .tooltip[hidden] {
      display: none;
    }
    @media (max-width: 520px) {
      .chart { padding: 16px 10px; }
      h1 { font-size: 22px; }
      .subtitle { font-size: 13px; }
      .grid-line,
      .axis-line,
      .series-line,
      .hit-line,
      .hover-marker {
        vector-effect: non-scaling-stroke;
      }
      .tick-label { font-size: 26px; }
      .axis-label { font-size: 24px; }
      .end-label { display: none; }
    }
    @media (prefers-reduced-motion: reduce) {
      .series-line { transition: none; }
    }
  </style>
</head>
<body>
  <main id="top-author-share-chart" class="chart">
    <h1>A growing share of papers include top-ranked authors</h1>
    <p class="subtitle">__ANALYSIS_LABEL__ &middot; Author rankings based on publication count over the preceding 20 years</p>
    <div class="legend" aria-label="Choose the emphasized author group">
      <button type="button" data-series="1" aria-pressed="false"><span class="swatch" style="--swatch: var(--series-1)"></span>Top 1% of authors</button>
      <button type="button" data-series="5" aria-pressed="false"><span class="swatch" style="--swatch: var(--series-5)"></span>Top 5% of authors</button>
      <button type="button" data-series="10" aria-pressed="true"><span class="swatch" style="--swatch: var(--series-10)"></span>Top 10% of authors</button>
    </div>
    <svg class="chart-svg" viewBox="0 0 900 520" role="img" aria-labelledby="chart-title chart-description">
      <title id="chart-title">Top author paper shares over time</title>
      <desc id="chart-description">Hover over a line to emphasize it and inspect its annual values.</desc>
    </svg>
    <div class="tooltip" role="status" aria-live="polite" hidden></div>
  </main>
  <script>
    (() => {
      const series = __SERIES_DATA__;
      const showAnnotation = __SHOW_ANNOTATION__;
      const root = document.getElementById("top-author-share-chart");
      const svg = root.querySelector(".chart-svg");
      const tooltip = root.querySelector(".tooltip");
      const ns = "http://www.w3.org/2000/svg";
      const width = 900;
      const height = 520;
      const margin = { top: 18, right: 70, bottom: 62, left: 86 };
      const keys = ["1", "5", "10"];
      const points = Object.values(series).flat();
      const xMin = Math.min(...points.map(point => point[0]));
      const xMax = Math.max(...points.map(point => point[0]));
      const observedMax = Math.max(...points.map(point => point[1]));
      const tickMax = Math.max(.1, Math.ceil(observedMax * 10) / 10);
      const yMax = tickMax + .035;
      const plotRight = width - margin.right;
      const plotBottom = height - margin.bottom;
      const x = year => margin.left + (year - xMin) / (xMax - xMin) * (plotRight - margin.left);
      const y = value => plotBottom - value / yMax * (plotBottom - margin.top);
      const add = (name, attributes, text) => {
        const node = document.createElementNS(ns, name);
        Object.entries(attributes).forEach(([key, value]) => node.setAttribute(key, value));
        if (text !== undefined) node.textContent = text;
        svg.appendChild(node);
        return node;
      };
      for (let value = 0; value <= tickMax + .0001; value += .1) {
        const position = y(value);
        add("line", { class: "grid-line", x1: margin.left, x2: plotRight, y1: position, y2: position });
        add("text", { class: "tick-label", x: margin.left - 12, y: position + 4, "text-anchor": "end" }, `${Math.round(value * 100)}%`);
      }
      for (let year = Math.ceil(xMin / 10) * 10; year <= xMax; year += 10) {
        add("text", { class: "tick-label", x: x(year), y: plotBottom + 25, "text-anchor": "middle" }, year);
      }
      add("line", { class: "axis-line", x1: margin.left, x2: margin.left, y1: margin.top, y2: plotBottom });
      add("line", { class: "axis-line", x1: margin.left, x2: plotRight, y1: plotBottom, y2: plotBottom });
      add("text", { class: "axis-label", x: (margin.left + plotRight) / 2, y: height - 12, "text-anchor": "middle" }, "Publication year");
      add("text", { class: "axis-label", x: 20, y: (margin.top + plotBottom) / 2, "text-anchor": "middle", transform: `rotate(-90 20 ${(margin.top + plotBottom) / 2})` }, "Share of papers with at least one top-ranked author");

      const lineNodes = {};
      const markerNodes = {};
      const hitNodes = {};
      keys.forEach(key => {
        const path = series[key].map(([year, value], index) => `${index ? "L" : "M"}${x(year).toFixed(2)},${y(value).toFixed(2)}`).join(" ");
        lineNodes[key] = add("path", { class: `series-line${key === "10" ? " is-active" : ""}`, "data-series": key, d: path });
        markerNodes[key] = add("circle", { class: "hover-marker", "data-series": key, r: 5 });
        hitNodes[key] = add("path", { class: "hit-line", "data-series": key, d: path });
        const lastPoint = series[key][series[key].length - 1];
        add("text", { class: "end-label", x: x(lastPoint[0]) + 8, y: y(lastPoint[1]) + 4 }, `Top ${key}%`);
      });
      if (showAnnotation) {
        const target = series["10"].find(point => point[0] === 2025) ?? series["10"][series["10"].length - 1];
        add("line", { class: "annotation-line", x1: x(2016), y1: y(0.55), x2: x(target[0]), y2: y(target[1]) });
        add("text", { class: "annotation", x: x(1997), y: y(0.585) }, "The share of papers with a top-10% author");
        add("text", { class: "annotation", x: x(1997), y: y(0.565) }, "rose from 28% in 1980 to 46% in 2025.");
      }

      let selectedSeries = "10";
      const activate = key => {
        keys.forEach(seriesKey => {
          lineNodes[seriesKey].classList.toggle("is-active", seriesKey === key);
        });
        root.querySelectorAll("button[data-series]").forEach(button => {
          button.setAttribute("aria-pressed", button.dataset.series === key ? "true" : "false");
        });
      };
      const hideTooltip = key => {
        markerNodes[key].style.display = "none";
        tooltip.hidden = true;
      };
      const showTooltip = (key, event) => {
        const svgRect = svg.getBoundingClientRect();
        const svgX = (event.clientX - svgRect.left) / svgRect.width * width;
        const nearest = series[key].reduce((best, point) => {
          return Math.abs(x(point[0]) - svgX) < Math.abs(x(best[0]) - svgX) ? point : best;
        });
        const marker = markerNodes[key];
        marker.setAttribute("cx", x(nearest[0]));
        marker.setAttribute("cy", y(nearest[1]));
        marker.style.display = "block";
        tooltip.textContent = `Top ${key}% · ${nearest[0]}: ${(nearest[1] * 100).toFixed(1)}% of papers`;
        tooltip.hidden = false;
        const rootRect = root.getBoundingClientRect();
        const markLeft = svgRect.left - rootRect.left + x(nearest[0]) / width * svgRect.width;
        const markTop = svgRect.top - rootRect.top + y(nearest[1]) / height * svgRect.height;
        const clampedLeft = Math.max(95, Math.min(root.clientWidth - 95, markLeft));
        tooltip.style.left = `${clampedLeft}px`;
        tooltip.style.top = `${Math.max(32, markTop)}px`;
      };

      keys.forEach(key => {
        hitNodes[key].addEventListener("pointerenter", event => {
          activate(key);
          showTooltip(key, event);
        });
        hitNodes[key].addEventListener("pointermove", event => {
          activate(key);
          showTooltip(key, event);
        });
        hitNodes[key].addEventListener("pointerleave", () => {
          hideTooltip(key);
          activate(selectedSeries);
        });
      });
      root.querySelectorAll("button[data-series]").forEach(button => {
        button.addEventListener("mouseenter", () => activate(button.dataset.series));
        button.addEventListener("focus", () => activate(button.dataset.series));
        button.addEventListener("click", () => {
          selectedSeries = button.dataset.series;
          activate(selectedSeries);
        });
        button.addEventListener("mouseleave", () => activate(selectedSeries));
        button.addEventListener("blur", () => activate(selectedSeries));
      });
    })();
  </script>
</body>
</html>
"""
    document = document.replace(
        "__ANALYSIS_LABEL__",
        html.escape(analysis_label),
    ).replace(
        "__SERIES_DATA__",
        json.dumps(series_data, separators=(",", ":"), allow_nan=False),
    ).replace(
        "__SHOW_ANNOTATION__",
        json.dumps(analysis_label == "All fields"),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(document, encoding="utf-8")


def write_interactive_new_author_share(
    data: pd.DataFrame,
    output_path: Path,
    analysis_label: str,
    from_year: int,
    to_year: int,
    share_column: str,
) -> None:
    plot_data = data.loc[
        data["publication_year"].between(from_year, to_year, inclusive="both")
    ].copy()
    plot_data[share_column] = pd.to_numeric(
        plot_data[share_column],
        errors="coerce",
    )
    plot_data = plot_data.dropna(subset=[share_column]).sort_values(
        "publication_year"
    )
    if plot_data.empty:
        return

    series_data = [
        [
            int(row.publication_year),
            round(float(getattr(row, share_column)), 6),
        ]
        for row in plot_data.itertuples(index=False)
    ]
    show_annotation = analysis_label == "All fields"

    document = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>New authors account for a smaller share of authors over time</title>
  <style>
    :root {
      color-scheme: light dark;
      --background: #ffffff;
      --foreground: #1e252b;
      --muted-foreground: #68737e;
      --grid: #dce1e6;
      --axis: #aeb7c0;
      --series: #087e73;
      --tooltip-background: #1e252b;
      --tooltip-foreground: #ffffff;
    }
    @media (prefers-color-scheme: dark) {
      :root {
        --background: #171a1d;
        --foreground: #f1f3f4;
        --muted-foreground: #b2bac2;
        --grid: #343a40;
        --axis: #626c75;
        --series: #46c6b8;
        --tooltip-background: #f1f3f4;
        --tooltip-foreground: #171a1d;
      }
    }
    * {
      box-sizing: border-box;
      letter-spacing: 0;
    }
    body {
      margin: 0;
      background: var(--background);
      color: var(--foreground);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    .chart {
      position: relative;
      width: 100%;
      max-width: 1100px;
      margin: 0 auto;
      padding: 24px;
    }
    h1 {
      margin: 0 0 6px;
      font-size: 28px;
      font-weight: 500;
    }
    .subtitle {
      margin: 0 0 14px;
      color: var(--muted-foreground);
      font-size: 15px;
    }
    .chart-svg {
      display: block;
      width: 100%;
      height: auto;
      overflow: visible;
    }
    .grid-line {
      stroke: var(--grid);
      stroke-width: 1;
    }
    .axis-line {
      stroke: var(--axis);
      stroke-width: 1;
    }
    .axis-label,
    .tick-label,
    .end-label,
    .annotation {
      fill: var(--muted-foreground);
      font-family: inherit;
    }
    .axis-label {
      font-size: 14px;
    }
    .tick-label,
    .end-label,
    .annotation {
      font-size: 12px;
    }
    .end-label {
      fill: var(--series);
      font-weight: 500;
    }
    .annotation-line {
      stroke: var(--muted-foreground);
      stroke-width: 1;
    }
    .share-line {
      fill: none;
      stroke: var(--series);
      stroke-width: 3;
      stroke-linecap: round;
      stroke-linejoin: round;
      transition: stroke-width 140ms ease;
    }
    .share-line.is-active {
      stroke-width: 4;
    }
    .hit-line {
      fill: none;
      stroke: transparent;
      stroke-width: 20;
      pointer-events: stroke;
      cursor: crosshair;
    }
    .hover-marker {
      display: none;
      fill: var(--series);
      stroke: var(--background);
      stroke-width: 2;
      pointer-events: none;
    }
    .tooltip {
      position: absolute;
      z-index: 2;
      transform: translate(-50%, calc(-100% - 10px));
      max-width: 180px;
      padding: 7px 9px;
      border-radius: 4px;
      background: var(--tooltip-background);
      color: var(--tooltip-foreground);
      font-size: 13px;
      white-space: nowrap;
      pointer-events: none;
    }
    .tooltip[hidden] {
      display: none;
    }
    @media (max-width: 520px) {
      .chart { padding: 16px 10px; }
      h1 { font-size: 22px; }
      .subtitle { font-size: 13px; }
      .grid-line,
      .axis-line,
      .share-line,
      .hit-line,
      .hover-marker,
      .annotation-line {
        vector-effect: non-scaling-stroke;
      }
      .tick-label { font-size: 25px; }
      .axis-label { font-size: 23px; }
      .end-label,
      .annotation,
      .annotation-line { display: none; }
    }
    @media (prefers-reduced-motion: reduce) {
      .share-line { transition: none; }
    }
  </style>
</head>
<body>
  <main id="new-author-share-chart" class="chart">
    <h1>New authors account for a smaller share of authors over time.</h1>
    <p class="subtitle">__ANALYSIS_LABEL__ &middot; New author defined as first observed publication in Top 5 journals</p>
    <svg class="chart-svg" viewBox="0 0 900 520" role="img" aria-labelledby="chart-title chart-description">
      <title id="chart-title">Annual share of authors who are new to top-five economics journals</title>
      <desc id="chart-description">Hover over the line to see the year and percentage of authors who are new.</desc>
    </svg>
    <div class="tooltip" role="status" aria-live="polite" hidden></div>
  </main>
  <script>
    (() => {
      const data = __SERIES_DATA__;
      const showAnnotation = __SHOW_ANNOTATION__;
      const root = document.getElementById("new-author-share-chart");
      const svg = root.querySelector(".chart-svg");
      const tooltip = root.querySelector(".tooltip");
      const ns = "http://www.w3.org/2000/svg";
      const width = 900;
      const height = 520;
      const margin = { top: 18, right: 132, bottom: 62, left: 86 };
      const fromYear = __FROM_YEAR__;
      const toYear = __TO_YEAR__;
      const shareMin = 0.30;
      const shareMax = 0.50;
      const plotRight = width - margin.right;
      const plotBottom = height - margin.bottom;
      const x = year => margin.left + (year - fromYear) / (toYear - fromYear) * (plotRight - margin.left);
      const y = value => plotBottom - (value - shareMin) / (shareMax - shareMin) * (plotBottom - margin.top);
      const add = (name, attributes, text) => {
        const node = document.createElementNS(ns, name);
        Object.entries(attributes).forEach(([key, value]) => node.setAttribute(key, value));
        if (text !== undefined) node.textContent = text;
        svg.appendChild(node);
        return node;
      };

      for (let value = shareMin; value <= shareMax + 0.0001; value += 0.05) {
        const position = y(value);
        add("line", { class: "grid-line", x1: margin.left, x2: plotRight, y1: position, y2: position });
        add("text", { class: "tick-label", x: margin.left - 12, y: position + 4, "text-anchor": "end" }, `${Math.round(value * 100)}%`);
      }
      [1980, 1990, 2000, 2010, 2020, 2025].filter(year => year >= fromYear && year <= toYear).forEach(year => {
        add("text", { class: "tick-label", x: x(year), y: plotBottom + 25, "text-anchor": "middle" }, year);
      });
      add("line", { class: "axis-line", x1: margin.left, x2: margin.left, y1: margin.top, y2: plotBottom });
      add("line", { class: "axis-line", x1: margin.left, x2: plotRight, y1: plotBottom, y2: plotBottom });
      add("text", { class: "axis-label", x: (margin.left + plotRight) / 2, y: height - 12, "text-anchor": "middle" }, "Publication year");
      add("text", { class: "axis-label", x: 20, y: (margin.top + plotBottom) / 2, "text-anchor": "middle", transform: `rotate(-90 20 ${(margin.top + plotBottom) / 2})` }, "Share of authors who are new");

      const path = data.map(([year, value], index) => `${index ? "L" : "M"}${x(year).toFixed(2)},${y(value).toFixed(2)}`).join(" ");
      const shareLine = add("path", { class: "share-line", d: path });
      const lastPoint = data[data.length - 1];
      add("text", { class: "end-label", x: x(lastPoint[0]) + 9, y: y(lastPoint[1]) + 4 }, "New authors' share");
      if (showAnnotation) {
        add("line", { class: "annotation-line", x1: x(2014), y1: y(0.463), x2: x(lastPoint[0]), y2: y(lastPoint[1]) });
        add("text", { class: "annotation", x: x(1998), y: y(0.492) }, "Despite year-to-year fluctuations,");
        add("text", { class: "annotation", x: x(1998), y: y(0.482) }, "the share of new authors trends downward");
        add("text", { class: "annotation", x: x(1998), y: y(0.472) }, "from about 45% in 1980 to about 36% in 2025.");
      }
      const marker = add("circle", { class: "hover-marker", r: 5 });
      const hitLine = add("path", { class: "hit-line", d: path });

      const showTooltip = event => {
        const svgRect = svg.getBoundingClientRect();
        const svgX = (event.clientX - svgRect.left) / svgRect.width * width;
        const nearest = data.reduce((best, point) => {
          return Math.abs(x(point[0]) - svgX) < Math.abs(x(best[0]) - svgX) ? point : best;
        });
        marker.setAttribute("cx", x(nearest[0]));
        marker.setAttribute("cy", y(nearest[1]));
        marker.style.display = "block";
        shareLine.classList.add("is-active");
        tooltip.textContent = `${nearest[0]} · ${(nearest[1] * 100).toFixed(1)}%`;
        tooltip.hidden = false;
        const rootRect = root.getBoundingClientRect();
        const markLeft = svgRect.left - rootRect.left + x(nearest[0]) / width * svgRect.width;
        const markTop = svgRect.top - rootRect.top + y(nearest[1]) / height * svgRect.height;
        tooltip.style.left = `${Math.max(90, Math.min(root.clientWidth - 90, markLeft))}px`;
        tooltip.style.top = `${Math.max(32, markTop)}px`;
      };
      const hideTooltip = () => {
        marker.style.display = "none";
        shareLine.classList.remove("is-active");
        tooltip.hidden = true;
      };
      hitLine.addEventListener("pointerenter", showTooltip);
      hitLine.addEventListener("pointermove", showTooltip);
      hitLine.addEventListener("pointerleave", hideTooltip);
    })();
  </script>
</body>
</html>
"""
    document = (
        document.replace("__ANALYSIS_LABEL__", html.escape(analysis_label))
        .replace(
            "__SERIES_DATA__",
            json.dumps(series_data, separators=(",", ":"), allow_nan=False),
        )
        .replace("__SHOW_ANNOTATION__", json.dumps(show_annotation))
        .replace("__FROM_YEAR__", str(from_year))
        .replace("__TO_YEAR__", str(to_year))
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(document, encoding="utf-8")


def plot_top_author_shares(
    data: pd.DataFrame,
    output_path: Path,
    analysis_label: str,
    from_year: int,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    plot_data = data.loc[data["publication_year"].ge(from_year)].copy()
    if plot_data.empty:
        return
    figure, axis = plt.subplots(figsize=(11, 6.5))
    markers = {1: "o", 5: "s", 10: "^"}
    line_styles = {
        1: {
            "color": "#A8AFB7",
            "linewidth": 1.8,
            "alpha": 0.72,
            "zorder": 2,
        },
        5: {
            "color": "#E67E22",
            "linewidth": 2.0,
            "alpha": 0.78,
            "zorder": 2,
        },
        GRAPH1_HIGHLIGHT_PERCENTAGE: {
            "color": "#087E73",
            "linewidth": 3.4,
            "alpha": 1.0,
            "zorder": 3,
        },
    }
    for percentage in TOP_AUTHOR_PERCENTAGES:
        series = plot_data.loc[
            plot_data["top_author_percentage"].eq(percentage)
        ]
        axis.plot(
            series["publication_year"],
            series["share_of_papers_with_top_author"],
            marker=markers[percentage],
            markevery=5,
            markersize=5,
            label=f"Top {percentage}% of authors",
            **line_styles[percentage],
        )
    axis.set_title(
        "A growing share of papers include top-ranked authors",
        loc="left",
        pad=38,
        fontsize=17,
        fontweight="semibold",
    )
    axis.text(
        0,
        1.015,
        (
            f"{analysis_label} \N{MIDDLE DOT} Author rankings based on publication count over the  "
            "preceding 20 years"
        ),
        transform=axis.transAxes,
        color="#69727D",
        fontsize=11,
        ha="left",
        va="bottom",
    )
    axis.set_xlabel("Publication year")
    axis.set_ylabel("Share of papers with at least one top-ranked author")
    axis.yaxis.set_major_formatter(PercentFormatter(1.0))
    axis.set_xlim(left=from_year)
    axis.set_ylim(bottom=0)
    axis.grid(axis="y", color="#D8DDE3", linewidth=0.8, alpha=0.8)
    axis.grid(axis="x", visible=False)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color("#B8C0C8")
    axis.spines["bottom"].set_color("#B8C0C8")
    axis.tick_params(colors="#4E5965")
    axis.legend(frameon=False, loc="upper left")
    if analysis_label == "All fields":
        top_ten = plot_data.loc[
            plot_data["top_author_percentage"].eq(10)
            & plot_data["publication_year"].le(2025)
        ].sort_values("publication_year")
        if not top_ten.empty:
            target = top_ten.iloc[-1]
            axis.annotate(
                (
                    "The share of papers with a top-10% author\n"
                    "rose from 28% in 1980 to 46% in 2025."
                ),
                xy=(
                    int(target["publication_year"]),
                    float(target["share_of_papers_with_top_author"]),
                ),
                xytext=(0.43, 0.96),
                textcoords="axes fraction",
                color="#4E5965",
                fontsize=10,
                ha="left",
                va="top",
                arrowprops={
                    "arrowstyle": "-",
                    "color": "#87919B",
                    "linewidth": 1.0,
                },
            )
    save_figure(figure, output_path)


def plot_share_only(
    data: pd.DataFrame,
    output_path: Path,
    analysis_label: str,
    from_year: int,
    to_year: int,
    share_min: float,
    share_max: float,
    share_column: str,
    title: str,
    subtitle: str,
    share_label: str,
    end_label: str,
    annotation_text: str,
    line_color: str,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    plot_data = data.loc[
        data["publication_year"].between(from_year, to_year, inclusive="both")
    ].copy()
    if plot_data.empty:
        return

    figure, axis = plt.subplots(figsize=(11, 6.5))
    share_line = axis.plot(
        plot_data["publication_year"],
        plot_data[share_column],
        color=line_color,
        linewidth=3.0,
        marker="o",
        markevery=5,
        markersize=5,
    )[0]
    axis.set_title(
        title,
        loc="left",
        pad=38,
        fontsize=17,
        fontweight="semibold",
    )
    axis.text(
        0,
        1.015,
        subtitle,
        transform=axis.transAxes,
        color="#69727D",
        fontsize=11,
        ha="left",
        va="bottom",
    )
    axis.set_xlabel("Publication year")
    axis.set_ylabel(share_label)
    axis.yaxis.set_major_formatter(PercentFormatter(1.0))
    axis.set_xlim(from_year, to_year + 3)
    axis.set_xticks([1980, 1990, 2000, 2010, 2020, 2025])
    axis.set_ylim(share_min, share_max)
    axis.grid(axis="y", alpha=0.25)
    axis.grid(axis="x", visible=False)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)

    last_row = plot_data.sort_values("publication_year").iloc[-1]
    last_year = int(last_row["publication_year"])
    last_share = float(last_row[share_column])
    axis.text(
        last_year + 0.35,
        last_share,
        end_label,
        color=share_line.get_color(),
        fontsize=10,
        fontweight="semibold",
        ha="left",
        va="center",
    )
    if annotation_text:
        axis.annotate(
            annotation_text,
            xy=(last_year, last_share),
            xytext=(2000, 0.492),
            color="#4E5965",
            fontsize=10,
            ha="left",
            va="top",
            arrowprops={
                "arrowstyle": "-",
                "color": "#87919B",
                "linewidth": 1.0,
            },
        )
    save_figure(figure, output_path)


def plot_share_and_count(
    data: pd.DataFrame,
    output_path: Path,
    analysis_label: str,
    from_year: int,
    to_year: int,
    share_min: float,
    share_max: float,
    share_column: str,
    count_column: str,
    title: str,
    share_label: str,
    count_label: str,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    plot_data = data.loc[
        data["publication_year"].between(from_year, to_year, inclusive="both")
    ].copy()
    if plot_data.empty:
        return
    figure, share_axis = plt.subplots(figsize=(11, 6.5))
    count_axis = share_axis.twinx()
    share_line = share_axis.plot(
        plot_data["publication_year"],
        plot_data[share_column],
        color="#1F77B4",
        linewidth=2.2,
        marker="o",
        markevery=5,
        markersize=5,
        label=share_label,
    )[0]
    count_line = count_axis.plot(
        plot_data["publication_year"],
        plot_data[count_column],
        color="#E67E22",
        linewidth=2.2,
        marker="s",
        markevery=5,
        markersize=5,
        label=count_label,
    )[0]
    share_axis.set_title(f"{title} - {analysis_label}")
    share_axis.set_xlabel("Publication year")
    share_axis.set_ylabel(share_label, color=share_line.get_color())
    count_axis.set_ylabel(count_label, color=count_line.get_color())
    share_axis.yaxis.set_major_formatter(PercentFormatter(1.0))
    share_axis.set_xlim(from_year, to_year)
    share_axis.set_ylim(share_min, share_max)
    count_axis.set_ylim(bottom=0)
    share_axis.tick_params(axis="y", colors=share_line.get_color())
    count_axis.tick_params(axis="y", colors=count_line.get_color())
    share_axis.grid(axis="y", alpha=0.25)
    share_axis.legend(
        [share_line, count_line],
        [share_label, count_label],
        frameon=False,
        loc="upper left",
    )
    save_figure(figure, output_path)


def write_interactive_new_author_coauthor_types(
    data: pd.DataFrame,
    output_path: Path,
    analysis_label: str,
    from_year: int,
    to_year: int,
) -> None:
    plot_data = data.loc[
        data["first_top5_publication_year"].between(
            from_year,
            to_year,
            inclusive="both",
        )
    ].copy()
    if plot_data.empty:
        return

    series_columns = {
        "experienced": "share_with_experienced_coauthor",
        "new_only": "share_only_with_other_new_coauthors",
        "solo": "share_solo",
    }
    series_data = {}
    for key, column in series_columns.items():
        series = plot_data[["first_top5_publication_year", column]].copy()
        series[column] = pd.to_numeric(series[column], errors="coerce")
        series = series.dropna(subset=[column]).sort_values(
            "first_top5_publication_year"
        )
        series_data[key] = [
            [
                int(row.first_top5_publication_year),
                round(float(getattr(row, column)), 6),
            ]
            for row in series.itertuples(index=False)
        ]

    subtitle = (
        "Share of new authors by coauthor composition in their first "
        "top-five publication, 1980\N{EN DASH}2025"
    )
    if analysis_label != "All fields":
        subtitle = f"{analysis_label} \N{MIDDLE DOT} {subtitle}"

    document = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>New authors publish their first top-five increasingly with experienced coauthors</title>
  <style>
    :root {
      color-scheme: light dark;
      --background: #ffffff;
      --foreground: #1e252b;
      --muted-foreground: #68737e;
      --grid: #dce1e6;
      --axis: #aeb7c0;
      --experienced: #087e73;
      --new-only: #e67e22;
      --solo: #a8afb7;
      --tooltip-background: #1e252b;
      --tooltip-foreground: #ffffff;
    }
    @media (prefers-color-scheme: dark) {
      :root {
        --background: #171a1d;
        --foreground: #f1f3f4;
        --muted-foreground: #b2bac2;
        --grid: #343a40;
        --axis: #626c75;
        --experienced: #46c6b8;
        --new-only: #f3a35c;
        --solo: #89939d;
        --tooltip-background: #f1f3f4;
        --tooltip-foreground: #171a1d;
      }
    }
    * {
      box-sizing: border-box;
      letter-spacing: 0;
    }
    body {
      margin: 0;
      background: var(--background);
      color: var(--foreground);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    .chart {
      position: relative;
      width: 100%;
      max-width: 1100px;
      margin: 0 auto;
      padding: 24px;
    }
    h1 {
      margin: 0 0 6px;
      font-size: 28px;
      font-weight: 500;
    }
    .subtitle {
      margin: 0 0 14px;
      color: var(--muted-foreground);
      font-size: 15px;
    }
    .legend {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-bottom: 8px;
    }
    .legend button {
      display: inline-flex;
      align-items: center;
      gap: 7px;
      min-height: 34px;
      padding: 6px 10px;
      border: 1px solid transparent;
      border-radius: 4px;
      background: transparent;
      color: var(--muted-foreground);
      font: inherit;
      cursor: pointer;
    }
    .legend button:hover,
    .legend button:focus-visible,
    .legend button[aria-pressed="true"] {
      border-color: var(--axis);
      color: var(--foreground);
      outline-offset: 2px;
    }
    .swatch {
      width: 22px;
      height: 3px;
      background: var(--swatch);
    }
    .chart-svg {
      display: block;
      width: 100%;
      height: auto;
      overflow: visible;
    }
    .grid-line {
      stroke: var(--grid);
      stroke-width: 1;
    }
    .axis-line {
      stroke: var(--axis);
      stroke-width: 1;
    }
    .axis-label,
    .tick-label,
    .annotation {
      fill: var(--muted-foreground);
      font-family: inherit;
    }
    .axis-label { font-size: 14px; }
    .tick-label,
    .annotation { font-size: 12px; }
    .annotation,
    .annotation-line { pointer-events: none; }
    .annotation-line {
      stroke: var(--muted-foreground);
      stroke-width: 1;
    }
    .series-line {
      fill: none;
      stroke-linecap: round;
      stroke-linejoin: round;
      stroke-width: 2;
      opacity: 0.36;
      transition: opacity 160ms ease, stroke-width 160ms ease;
    }
    .series-line[data-series="experienced"] { stroke: var(--experienced); }
    .series-line[data-series="new_only"] { stroke: var(--new-only); }
    .series-line[data-series="solo"] { stroke: var(--solo); }
    .series-line.is-active {
      stroke-width: 4;
      opacity: 1;
    }
    .hit-line {
      fill: none;
      stroke: transparent;
      stroke-width: 20;
      pointer-events: stroke;
      cursor: crosshair;
    }
    .hover-marker {
      display: none;
      stroke: var(--background);
      stroke-width: 2;
      pointer-events: none;
    }
    .hover-marker[data-series="experienced"] { fill: var(--experienced); }
    .hover-marker[data-series="new_only"] { fill: var(--new-only); }
    .hover-marker[data-series="solo"] { fill: var(--solo); }
    .tooltip {
      position: absolute;
      z-index: 2;
      transform: translate(-50%, calc(-100% - 10px));
      max-width: 260px;
      padding: 7px 9px;
      border-radius: 4px;
      background: var(--tooltip-background);
      color: var(--tooltip-foreground);
      font-size: 13px;
      white-space: nowrap;
      pointer-events: none;
    }
    .tooltip[hidden] { display: none; }
    .note {
      margin: 4px 0 0;
      color: var(--muted-foreground);
      font-size: 12px;
    }
    @media (max-width: 520px) {
      .chart { padding: 16px 10px; }
      h1 { font-size: 22px; }
      .subtitle { font-size: 13px; }
      .grid-line,
      .axis-line,
      .series-line,
      .hit-line,
      .hover-marker,
      .annotation-line { vector-effect: non-scaling-stroke; }
      .tick-label { font-size: 25px; }
      .axis-label { font-size: 23px; }
      .annotation,
      .annotation-line { display: none; }
    }
    @media (prefers-reduced-motion: reduce) {
      .series-line { transition: none; }
    }
  </style>
</head>
<body>
  <main id="new-author-coauthor-chart" class="chart">
    <h1>New authors publish their first top-five increasingly with experienced coauthors</h1>
    <p class="subtitle">__SUBTITLE__</p>
    <div class="legend" aria-label="Choose the emphasized coauthor composition">
      <button type="button" data-series="experienced" aria-pressed="true"><span class="swatch" style="--swatch: var(--experienced)"></span>At least one experienced coauthor</button>
      <button type="button" data-series="new_only" aria-pressed="false"><span class="swatch" style="--swatch: var(--new-only)"></span>New coauthors only</button>
      <button type="button" data-series="solo" aria-pressed="false"><span class="swatch" style="--swatch: var(--solo)"></span>Solo-authored</button>
    </div>
    <svg class="chart-svg" viewBox="0 0 900 540" role="img" aria-labelledby="chart-title chart-description">
      <title id="chart-title">Coauthor composition in new authors' first top-five publications</title>
      <desc id="chart-description">Hover over a line to see the year and share of new authors in that coauthor category.</desc>
    </svg>
    <div class="tooltip" role="status" aria-live="polite" hidden></div>
    <p class="note"><strong>Note:</strong> New authors are the ones who published their first top-five in that year. &ldquo;Experienced coauthor&rdquo; means the coauthor published at least one top-five before. Categories are mutually exclusive and sum to 100%.</p>
  </main>
  <script>
    (() => {
      const series = __SERIES_DATA__;
      const showAnnotation = __SHOW_ANNOTATION__;
      const labels = {
        experienced: "At least one experienced coauthor",
        new_only: "New coauthors only",
        solo: "Solo-authored"
      };
      const root = document.getElementById("new-author-coauthor-chart");
      const svg = root.querySelector(".chart-svg");
      const tooltip = root.querySelector(".tooltip");
      const ns = "http://www.w3.org/2000/svg";
      const width = 900;
      const height = 540;
      const margin = { top: 18, right: 32, bottom: 68, left: 90 };
      const fromYear = __FROM_YEAR__;
      const toYear = __TO_YEAR__;
      const plotRight = width - margin.right;
      const plotBottom = height - margin.bottom;
      const x = year => margin.left + (year - fromYear) / (toYear - fromYear) * (plotRight - margin.left);
      const y = value => plotBottom - value * (plotBottom - margin.top);
      const add = (name, attributes, text) => {
        const node = document.createElementNS(ns, name);
        Object.entries(attributes).forEach(([key, value]) => node.setAttribute(key, value));
        if (text !== undefined) node.textContent = text;
        svg.appendChild(node);
        return node;
      };

      for (let value = 0; value <= 1.0001; value += 0.2) {
        const position = y(value);
        add("line", { class: "grid-line", x1: margin.left, x2: plotRight, y1: position, y2: position });
        add("text", { class: "tick-label", x: margin.left - 12, y: position + 4, "text-anchor": "end" }, `${Math.round(value * 100)}%`);
      }
      [1980, 1990, 2000, 2010, 2020, 2025].filter(year => year >= fromYear && year <= toYear).forEach(year => {
        add("text", { class: "tick-label", x: x(year), y: plotBottom + 25, "text-anchor": "middle" }, year);
      });
      add("line", { class: "axis-line", x1: margin.left, x2: margin.left, y1: margin.top, y2: plotBottom });
      add("line", { class: "axis-line", x1: margin.left, x2: plotRight, y1: plotBottom, y2: plotBottom });
      add("text", { class: "axis-label", x: (margin.left + plotRight) / 2, y: height - 14, "text-anchor": "middle" }, "Year of first top-five publication");
      add("text", { class: "axis-label", x: 22, y: (margin.top + plotBottom) / 2, "text-anchor": "middle", transform: `rotate(-90 22 ${(margin.top + plotBottom) / 2})` }, "Share of new authors");

      const keys = ["experienced", "new_only", "solo"];
      const lineNodes = {};
      const markerNodes = {};
      const hitNodes = {};
      keys.forEach(key => {
        const path = series[key].map(([year, value], index) => `${index ? "L" : "M"}${x(year).toFixed(2)},${y(value).toFixed(2)}`).join(" ");
        lineNodes[key] = add("path", { class: `series-line${key === "experienced" ? " is-active" : ""}`, "data-series": key, d: path });
        markerNodes[key] = add("circle", { class: "hover-marker", "data-series": key, r: 5 });
        hitNodes[key] = add("path", { class: "hit-line", "data-series": key, d: path });
      });

      if (showAnnotation) {
        const target = series.experienced.find(point => point[0] === 2025) ?? series.experienced[series.experienced.length - 1];
        const newOnlyTarget = series.new_only.find(point => point[0] === 2025) ?? series.new_only[series.new_only.length - 1];
        const soloTarget = series.solo.find(point => point[0] === 2025) ?? series.solo[series.solo.length - 1];
        add("line", { class: "annotation-line", x1: x(2014), y1: y(0.91), x2: x(target[0]), y2: y(target[1]) });
        add("text", { class: "annotation", x: x(1994), y: y(0.96) }, "First publication with experienced coauthors rose from about 28% to 77%,");
        add("text", { class: "annotation", x: x(1994), y: y(0.915) }, "+49 percentage points since 1980");
        add("line", { class: "annotation-line", x1: x(2018), y1: y(0.29), x2: x(newOnlyTarget[0]), y2: y(newOnlyTarget[1]) });
        add("text", { class: "annotation", x: x(2005), y: y(0.35) }, "New coauthors only: 24% \N{RIGHTWARDS ARROW} 16%");
        add("text", { class: "annotation", x: x(2005), y: y(0.305) }, "(-8 percentage points since 1980)");
        add("line", { class: "annotation-line", x1: x(2016), y1: y(0.12), x2: x(soloTarget[0]), y2: y(soloTarget[1]) });
        add("text", { class: "annotation", x: x(1998), y: y(0.15) }, "Solo-authored: 49% \N{RIGHTWARDS ARROW} 7%");
        add("text", { class: "annotation", x: x(1998), y: y(0.105) }, "(-42 percentage points since 1980)");
      }

      let selectedSeries = "experienced";
      const activate = key => {
        keys.forEach(seriesKey => {
          lineNodes[seriesKey].classList.toggle("is-active", seriesKey === key);
        });
        root.querySelectorAll("button[data-series]").forEach(button => {
          button.setAttribute("aria-pressed", button.dataset.series === key ? "true" : "false");
        });
      };
      const hideTooltip = key => {
        markerNodes[key].style.display = "none";
        tooltip.hidden = true;
      };
      const showTooltip = (key, event) => {
        const svgRect = svg.getBoundingClientRect();
        const svgX = (event.clientX - svgRect.left) / svgRect.width * width;
        const nearest = series[key].reduce((best, point) => {
          return Math.abs(x(point[0]) - svgX) < Math.abs(x(best[0]) - svgX) ? point : best;
        });
        const marker = markerNodes[key];
        marker.setAttribute("cx", x(nearest[0]));
        marker.setAttribute("cy", y(nearest[1]));
        marker.style.display = "block";
        tooltip.textContent = `${labels[key]} - ${nearest[0]}: ${(nearest[1] * 100).toFixed(1)}%`;
        tooltip.hidden = false;
        const rootRect = root.getBoundingClientRect();
        const markLeft = svgRect.left - rootRect.left + x(nearest[0]) / width * svgRect.width;
        const markTop = svgRect.top - rootRect.top + y(nearest[1]) / height * svgRect.height;
        tooltip.style.left = `${Math.max(135, Math.min(root.clientWidth - 135, markLeft))}px`;
        tooltip.style.top = `${Math.max(32, markTop)}px`;
      };

      keys.forEach(key => {
        hitNodes[key].addEventListener("pointerenter", event => {
          activate(key);
          showTooltip(key, event);
        });
        hitNodes[key].addEventListener("pointermove", event => {
          activate(key);
          showTooltip(key, event);
        });
        hitNodes[key].addEventListener("pointerleave", () => {
          hideTooltip(key);
          activate(selectedSeries);
        });
      });
      root.querySelectorAll("button[data-series]").forEach(button => {
        button.addEventListener("mouseenter", () => activate(button.dataset.series));
        button.addEventListener("focus", () => activate(button.dataset.series));
        button.addEventListener("click", () => {
          selectedSeries = button.dataset.series;
          activate(selectedSeries);
        });
        button.addEventListener("mouseleave", () => activate(selectedSeries));
        button.addEventListener("blur", () => activate(selectedSeries));
      });
    })();
  </script>
</body>
</html>
"""
    document = (
        document.replace("__SUBTITLE__", html.escape(subtitle))
        .replace(
            "__SERIES_DATA__",
            json.dumps(series_data, separators=(",", ":"), allow_nan=False),
        )
        .replace(
            "__SHOW_ANNOTATION__",
            json.dumps(analysis_label == "All fields"),
        )
        .replace("__FROM_YEAR__", str(from_year))
        .replace("__TO_YEAR__", str(to_year))
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(document, encoding="utf-8")


def plot_new_author_coauthor_types(
    data: pd.DataFrame,
    output_path: Path,
    analysis_label: str,
    from_year: int,
    to_year: int,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    if data.empty:
        return
    plot_data = data.loc[
        data["first_top5_publication_year"].between(
            from_year,
            to_year,
            inclusive="both",
        )
    ]
    if plot_data.empty:
        return
    figure, axis = plt.subplots(figsize=(11, 6.5))
    series = [
        (
            "share_with_experienced_coauthor",
            "At least one experienced coauthor",
            "o",
            {
                "color": "#087E73",
                "linewidth": 3.4,
                "alpha": 1.0,
                "zorder": 3,
            },
        ),
        (
            "share_only_with_other_new_coauthors",
            "New coauthors only",
            "s",
            {
                "color": "#E67E22",
                "linewidth": 2.0,
                "alpha": 0.78,
                "zorder": 2,
            },
        ),
        (
            "share_solo",
            "Solo-authored",
            "^",
            {
                "color": "#A8AFB7",
                "linewidth": 1.8,
                "alpha": 0.72,
                "zorder": 2,
            },
        ),
    ]
    for column, label, marker, line_style in series:
        axis.plot(
            plot_data["first_top5_publication_year"],
            plot_data[column],
            marker=marker,
            markevery=5,
            markersize=5,
            label=label,
            **line_style,
        )
    axis.set_title(
        (
            "New authors publish their first top-five increasingly "
            "with experienced coauthors"
        ),
        loc="left",
        pad=38,
        fontsize=15,
        fontweight="semibold",
    )
    subtitle = (
        "Share of new authors by coauthor composition in their first "
        f"top-five publication, 1980\N{EN DASH}2025"
    )
    if analysis_label != "All fields":
        subtitle = f"{analysis_label} \N{MIDDLE DOT} {subtitle}"
    axis.text(
        0,
        1.015,
        subtitle,
        transform=axis.transAxes,
        color="#69727D",
        fontsize=11,
        ha="left",
        va="bottom",
    )
    axis.set_xlabel("Year of first top-five publication")
    axis.set_ylabel("Share of new authors")
    axis.yaxis.set_major_formatter(PercentFormatter(1.0))
    axis.set_xlim(from_year, to_year)
    axis.set_ylim(0, 1)
    axis.grid(axis="y", color="#D8DDE3", linewidth=0.8, alpha=0.8)
    axis.grid(axis="x", visible=False)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color("#B8C0C8")
    axis.spines["bottom"].set_color("#B8C0C8")
    axis.tick_params(colors="#4E5965")
    axis.legend(frameon=False, loc="upper left")

    if analysis_label == "All fields":
        experienced = plot_data.loc[
            plot_data["first_top5_publication_year"].le(2025)
        ].sort_values("first_top5_publication_year")
        if not experienced.empty:
            target = experienced.iloc[-1]
            axis.annotate(
                (
                    "First publication with experienced coauthors rose from about 28% "
                    "to 77%,\n+49 percentage points since 1980"
                ),
                xy=(
                    int(target["first_top5_publication_year"]),
                    float(target["share_with_experienced_coauthor"]),
                ),
                xytext=(0.45, 0.94),
                textcoords="axes fraction",
                color="#4E5965",
                fontsize=10,
                ha="left",
                va="top",
                arrowprops={
                    "arrowstyle": "-",
                    "color": "#87919B",
                    "linewidth": 1.0,
                },
            )
            axis.annotate(
                (
                    "New coauthors only: 24% \N{RIGHTWARDS ARROW} 16%\n"
                    "(-8 percentage points since 1980)"
                ),
                xy=(
                    int(target["first_top5_publication_year"]),
                    float(target["share_only_with_other_new_coauthors"]),
                ),
                xytext=(0.56, 0.35),
                textcoords="axes fraction",
                color="#4E5965",
                fontsize=9,
                ha="left",
                va="top",
                arrowprops={
                    "arrowstyle": "-",
                    "color": "#C98A4A",
                    "linewidth": 1.0,
                },
            )
            axis.annotate(
                (
                    "Solo-authored: 49% \N{RIGHTWARDS ARROW} 7%\n"
                    "(-42 percentage points since 1980)"
                ),
                xy=(
                    int(target["first_top5_publication_year"]),
                    float(target["share_solo"]),
                ),
                xytext=(0.47, 0.15),
                textcoords="axes fraction",
                color="#4E5965",
                fontsize=9,
                ha="left",
                va="top",
                arrowprops={
                    "arrowstyle": "-",
                    "color": "#9EA6AE",
                    "linewidth": 1.0,
                },
            )

    figure.text(
        0.115,
        0.015,
        (
            "Note: New authors are the ones who published their first top-five "
            "in that year.\n\"Experienced coauthor\" means the coauthor published "
            "at least one top-five before.\nCategories are mutually exclusive "
            "and sum to 100%."
        ),
        color="#69727D",
        fontsize=9,
        ha="left",
        va="bottom",
    )
    save_figure(
        figure,
        output_path,
        tight_layout_rect=(0, 0.17, 1, 1),
    )


def write_interactive_publication_gaps_by_cohort(
    data: pd.DataFrame,
    output_path: Path,
    analysis_label: str,
    minimum_authors: int,
) -> None:
    if data.empty:
        return
    plot_data = data.loc[
        data["number_of_authors"].ge(minimum_authors)
        & data["to_publication_number"].between(2, 10, inclusive="both")
    ].copy()
    plot_data["average_gap_years"] = pd.to_numeric(
        plot_data["average_gap_years"],
        errors="coerce",
    )
    plot_data["number_of_authors"] = pd.to_numeric(
        plot_data["number_of_authors"],
        errors="coerce",
    )
    plot_data = plot_data.dropna(
        subset=["average_gap_years", "number_of_authors"]
    )
    if plot_data.empty:
        return

    series_data = {}
    for _, _, cohort_label in AUTHOR_ENTRY_COHORTS:
        cohort = plot_data.loc[
            plot_data["entry_cohort"].eq(cohort_label)
        ].sort_values("to_publication_number")
        if cohort.empty:
            continue
        series_data[cohort_label] = [
            [
                int(row.to_publication_number),
                round(float(row.average_gap_years), 6),
                int(row.number_of_authors),
            ]
            for row in cohort.itertuples(index=False)
        ]
    if not series_data:
        return

    subtitle = (
        "Average years between consecutive top-five publications, by cohort "
        "of first publication"
    )
    if analysis_label != "All fields":
        subtitle = f"{analysis_label} \N{MIDDLE DOT} {subtitle}"

    document = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Observed publication gaps are shorter for newer cohorts</title>
  <style>
    :root {
      color-scheme: light dark;
      --background: #ffffff;
      --foreground: #1e252b;
      --muted-foreground: #68737e;
      --grid: #dce1e6;
      --axis: #aeb7c0;
      --cohort-1980s: #e67e22;
      --cohort-1990s: #a8afb7;
      --cohort-2000s: #68737e;
      --cohort-2010s: #087e73;
      --tooltip-background: #1e252b;
      --tooltip-foreground: #ffffff;
    }
    @media (prefers-color-scheme: dark) {
      :root {
        --background: #171a1d;
        --foreground: #f1f3f4;
        --muted-foreground: #b2bac2;
        --grid: #343a40;
        --axis: #626c75;
        --cohort-1980s: #f3a35c;
        --cohort-1990s: #89939d;
        --cohort-2000s: #aeb7c0;
        --cohort-2010s: #46c6b8;
        --tooltip-background: #f1f3f4;
        --tooltip-foreground: #171a1d;
      }
    }
    * {
      box-sizing: border-box;
      letter-spacing: 0;
    }
    body {
      margin: 0;
      background: var(--background);
      color: var(--foreground);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    .chart {
      position: relative;
      width: 100%;
      max-width: 1100px;
      margin: 0 auto;
      padding: 24px;
    }
    h1 {
      margin: 0 0 6px;
      font-size: 28px;
      font-weight: 500;
    }
    .subtitle {
      margin: 0 0 14px;
      color: var(--muted-foreground);
      font-size: 15px;
    }
    .legend {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-bottom: 8px;
    }
    .legend button {
      display: inline-flex;
      align-items: center;
      gap: 7px;
      min-height: 34px;
      padding: 6px 10px;
      border: 1px solid transparent;
      border-radius: 4px;
      background: transparent;
      color: var(--muted-foreground);
      font: inherit;
      cursor: pointer;
    }
    .legend button:hover,
    .legend button:focus-visible,
    .legend button[aria-pressed="true"] {
      border-color: var(--axis);
      color: var(--foreground);
      outline-offset: 2px;
    }
    .swatch {
      width: 22px;
      height: 3px;
      background: var(--swatch);
    }
    .chart-svg {
      display: block;
      width: 100%;
      height: auto;
      overflow: visible;
    }
    .grid-line {
      stroke: var(--grid);
      stroke-width: 1;
    }
    .axis-line {
      stroke: var(--axis);
      stroke-width: 1;
    }
    .axis-label,
    .tick-label,
    .annotation {
      fill: var(--muted-foreground);
      font-family: inherit;
    }
    .axis-label { font-size: 14px; }
    .tick-label,
    .annotation { font-size: 12px; }
    .annotation,
    .annotation-line { pointer-events: none; }
    .annotation-line {
      stroke: var(--muted-foreground);
      stroke-width: 1;
    }
    .series-line {
      fill: none;
      stroke-linecap: round;
      stroke-linejoin: round;
      stroke-width: 2;
      opacity: 0.32;
      transition: opacity 160ms ease, stroke-width 160ms ease;
    }
    .series-line[data-series="1981-1990"] { stroke: var(--cohort-1980s); }
    .series-line[data-series="1991-2000"] { stroke: var(--cohort-1990s); }
    .series-line[data-series="2001-2010"] { stroke: var(--cohort-2000s); }
    .series-line[data-series="2011-2020"] { stroke: var(--cohort-2010s); }
    .series-line.is-active {
      stroke-width: 4;
      opacity: 1;
    }
    .swatch { position: relative; }
    .swatch::after {
      position: absolute;
      top: 50%;
      left: 50%;
      width: 9px;
      height: 9px;
      border: 1px solid var(--background);
      background: var(--swatch);
      content: "";
      transform: translate(-50%, -50%);
    }
    .swatch.circle::after { border-radius: 50%; }
    .swatch.diamond::after { transform: translate(-50%, -50%) rotate(45deg); }
    .swatch.triangle::after {
      border: 0;
      clip-path: polygon(50% 0, 100% 100%, 0 100%);
    }
    .point-marker {
      stroke: var(--background);
      stroke-width: 1.3;
      opacity: 0.32;
      pointer-events: none;
      transition: opacity 160ms ease, stroke-width 160ms ease;
    }
    .point-marker[data-series="1981-1990"] { fill: var(--cohort-1980s); }
    .point-marker[data-series="1991-2000"] { fill: var(--cohort-1990s); }
    .point-marker[data-series="2001-2010"] { fill: var(--cohort-2000s); }
    .point-marker[data-series="2011-2020"] { fill: var(--cohort-2010s); }
    .point-marker.is-active {
      stroke-width: 2;
      opacity: 1;
    }
    .point-marker.is-hover {
      stroke-width: 4;
      opacity: 1;
    }
    .hit-line {
      fill: none;
      stroke: transparent;
      stroke-width: 20;
      pointer-events: stroke;
      cursor: crosshair;
    }
    .tooltip {
      position: absolute;
      z-index: 2;
      transform: translate(-50%, calc(-100% - 10px));
      max-width: 300px;
      padding: 7px 9px;
      border-radius: 4px;
      background: var(--tooltip-background);
      color: var(--tooltip-foreground);
      font-size: 13px;
      white-space: nowrap;
      pointer-events: none;
    }
    .tooltip[hidden] { display: none; }
    .note {
      margin: 4px 0 0;
      color: var(--muted-foreground);
      font-size: 12px;
      line-height: 1.45;
    }
    @media (max-width: 520px) {
      .chart { padding: 16px 10px; }
      h1 { font-size: 22px; }
      .subtitle { font-size: 13px; }
      .grid-line,
      .axis-line,
      .series-line,
      .point-marker,
      .hit-line,
      .annotation-line { vector-effect: non-scaling-stroke; }
      .tick-label { font-size: 22px; }
      .axis-label { font-size: 23px; }
      .annotation,
      .annotation-line { display: none; }
      .tooltip { white-space: normal; }
    }
    @media (prefers-reduced-motion: reduce) {
      .series-line,
      .point-marker { transition: none; }
    }
  </style>
</head>
<body>
  <main id="publication-gap-chart" class="chart">
    <h1>Observed publication gaps are shorter for newer cohorts</h1>
    <p class="subtitle">__SUBTITLE__</p>
    <div class="legend" aria-label="Choose the emphasized first-publication cohort">
      <button type="button" data-series="1981-1990" aria-pressed="false"><span class="swatch circle" style="--swatch: var(--cohort-1980s)"></span>1981-1990</button>
      <button type="button" data-series="1991-2000" aria-pressed="false"><span class="swatch square" style="--swatch: var(--cohort-1990s)"></span>1991-2000</button>
      <button type="button" data-series="2001-2010" aria-pressed="false"><span class="swatch diamond" style="--swatch: var(--cohort-2000s)"></span>2001-2010</button>
      <button type="button" data-series="2011-2020" aria-pressed="true"><span class="swatch triangle" style="--swatch: var(--cohort-2010s)"></span>2011-2020</button>
    </div>
    <svg class="chart-svg" viewBox="0 0 980 560" role="img" aria-labelledby="chart-title chart-description">
      <title id="chart-title">Average publication gaps by first-publication cohort</title>
      <desc id="chart-description">Hover over a cohort line to inspect the publication transition, average gap, and number of authors.</desc>
    </svg>
    <div class="tooltip" role="status" aria-live="polite" hidden></div>
    <p class="note"><strong>Note:</strong> Values show mean years between consecutive publications. Cohorts are defined by the year of first top-five publication. Later publication numbers include only authors who reach that stage.</p>
  </main>
  <script>
    (() => {
      const series = __SERIES_DATA__;
      const showAnnotation = __SHOW_ANNOTATION__;
      const transitionLabels = {
        2: "1st to 2nd publication",
        3: "2nd to 3rd publication",
        4: "3rd to 4th publication",
        5: "4th to 5th publication",
        6: "5th to 6th publication",
        7: "6th to 7th publication",
        8: "7th to 8th publication",
        9: "8th to 9th publication",
        10: "9th to 10th publication"
      };
      const shortTransitionLabels = {
        2: "1st-2nd", 3: "2nd-3rd", 4: "3rd-4th",
        5: "4th-5th", 6: "5th-6th", 7: "6th-7th",
        8: "7th-8th", 9: "8th-9th", 10: "9th-10th"
      };
      const root = document.getElementById("publication-gap-chart");
      const svg = root.querySelector(".chart-svg");
      const tooltip = root.querySelector(".tooltip");
      const ns = "http://www.w3.org/2000/svg";
      const width = 980;
      const height = 560;
      const margin = { top: 24, right: 34, bottom: 76, left: 92 };
      const keys = Object.keys(series);
      const defaultSeries = keys.includes("2011-2020") ? "2011-2020" : keys[keys.length - 1];
      const points = Object.values(series).flat();
      const observedMax = Math.max(...points.map(point => point[1]));
      const tickMax = Math.max(1, Math.ceil(observedMax));
      const yMax = tickMax + 0.45;
      const plotRight = width - margin.right;
      const plotBottom = height - margin.bottom;
      const x = publicationNumber => margin.left + (publicationNumber - 2) / 8 * (plotRight - margin.left);
      const y = gap => plotBottom - gap / yMax * (plotBottom - margin.top);
      const add = (name, attributes, text) => {
        const node = document.createElementNS(ns, name);
        Object.entries(attributes).forEach(([key, value]) => node.setAttribute(key, value));
        if (text !== undefined) node.textContent = text;
        svg.appendChild(node);
        return node;
      };

      for (let value = 0; value <= tickMax; value += 1) {
        const position = y(value);
        add("line", { class: "grid-line", x1: margin.left, x2: plotRight, y1: position, y2: position });
        add("text", { class: "tick-label", x: margin.left - 12, y: position + 4, "text-anchor": "end" }, value.toFixed(0));
      }
      for (let publicationNumber = 2; publicationNumber <= 10; publicationNumber += 1) {
        add("text", { class: "tick-label x-tick", x: x(publicationNumber), y: plotBottom + 25, "text-anchor": "middle" }, shortTransitionLabels[publicationNumber]);
      }
      add("line", { class: "axis-line", x1: margin.left, x2: margin.left, y1: margin.top, y2: plotBottom });
      add("line", { class: "axis-line", x1: margin.left, x2: plotRight, y1: plotBottom, y2: plotBottom });
      add("text", { class: "axis-label", x: (margin.left + plotRight) / 2, y: height - 14, "text-anchor": "middle" }, "Publication number");
      add("text", { class: "axis-label", x: 22, y: (margin.top + plotBottom) / 2, "text-anchor": "middle", transform: `rotate(-90 22 ${(margin.top + plotBottom) / 2})` }, "Average gap from previous publication (years)");

      const lineNodes = {};
      const pointNodes = {};
      const hitNodes = {};
      const addPointMarker = (key, publicationNumber, gap) => {
        const pointX = x(publicationNumber);
        const pointY = y(gap);
        const common = {
          class: `point-marker${key === defaultSeries ? " is-active" : ""}`,
          "data-series": key,
          "data-publication-number": publicationNumber
        };
        if (key === "1981-1990") {
          return add("circle", { ...common, cx: pointX, cy: pointY, r: 4.4 });
        }
        if (key === "1991-2000") {
          return add("rect", { ...common, x: pointX - 4.2, y: pointY - 4.2, width: 8.4, height: 8.4 });
        }
        if (key === "2001-2010") {
          return add("rect", { ...common, x: pointX - 3.8, y: pointY - 3.8, width: 7.6, height: 7.6, transform: `rotate(45 ${pointX} ${pointY})` });
        }
        return add("polygon", { ...common, points: `${pointX},${pointY - 5.2} ${pointX + 5.0},${pointY + 4.1} ${pointX - 5.0},${pointY + 4.1}` });
      };
      keys.forEach(key => {
        const path = series[key].map(([publicationNumber, gap], index) => `${index ? "L" : "M"}${x(publicationNumber).toFixed(2)},${y(gap).toFixed(2)}`).join(" ");
        lineNodes[key] = add("path", { class: `series-line${key === defaultSeries ? " is-active" : ""}`, "data-series": key, d: path });
        pointNodes[key] = series[key].map(([publicationNumber, gap]) => addPointMarker(key, publicationNumber, gap));
        hitNodes[key] = add("path", { class: "hit-line", "data-series": key, d: path });
      });

      root.querySelectorAll("button[data-series]").forEach(button => {
        if (!keys.includes(button.dataset.series)) button.hidden = true;
        button.setAttribute("aria-pressed", button.dataset.series === defaultSeries ? "true" : "false");
      });

      if (showAnnotation && series["1981-1990"] && series["2011-2020"]) {
        const oldest = series["1981-1990"].find(point => point[0] === 2);
        const newest = series["2011-2020"].find(point => point[0] === 2);
        if (oldest && newest) {
          const annotationGap = Math.min(yMax - 0.15, Math.max(oldest[1], newest[1]) + 1.0);
          add("line", { class: "annotation-line", x1: x(3.45), y1: y(annotationGap - 0.15), x2: x(newest[0]), y2: y(newest[1]) });
          add("text", { class: "annotation", x: x(3.55), y: y(annotationGap) }, "The observed gap from the 1st to 2nd publication declined");
          add("text", { class: "annotation", x: x(3.55), y: y(annotationGap - 0.32) }, `from about ${oldest[1].toFixed(1)} to ${newest[1].toFixed(1)} years`);
        }
      }

      let selectedSeries = defaultSeries;
      const activate = key => {
        keys.forEach(seriesKey => {
          lineNodes[seriesKey].classList.toggle("is-active", seriesKey === key);
          pointNodes[seriesKey].forEach(point => {
            point.classList.toggle("is-active", seriesKey === key);
          });
        });
        root.querySelectorAll("button[data-series]").forEach(button => {
          button.setAttribute("aria-pressed", button.dataset.series === key ? "true" : "false");
        });
      };
      const hideTooltip = key => {
        pointNodes[key].forEach(point => point.classList.remove("is-hover"));
        tooltip.hidden = true;
      };
      const showTooltip = (key, event) => {
        const svgRect = svg.getBoundingClientRect();
        const svgX = (event.clientX - svgRect.left) / svgRect.width * width;
        let nearestIndex = 0;
        series[key].forEach((point, index) => {
          if (Math.abs(x(point[0]) - svgX) < Math.abs(x(series[key][nearestIndex][0]) - svgX)) {
            nearestIndex = index;
          }
        });
        const nearest = series[key][nearestIndex];
        pointNodes[key].forEach((point, index) => {
          point.classList.toggle("is-hover", index === nearestIndex);
        });
        tooltip.textContent = `${key} cohort - ${transitionLabels[nearest[0]]}: ${nearest[1].toFixed(2)} years (${nearest[2].toLocaleString()} authors)`;
        tooltip.hidden = false;
        const rootRect = root.getBoundingClientRect();
        const markLeft = svgRect.left - rootRect.left + x(nearest[0]) / width * svgRect.width;
        const markTop = svgRect.top - rootRect.top + y(nearest[1]) / height * svgRect.height;
        tooltip.style.left = `${Math.max(150, Math.min(root.clientWidth - 150, markLeft))}px`;
        tooltip.style.top = `${Math.max(32, markTop)}px`;
      };

      keys.forEach(key => {
        hitNodes[key].addEventListener("pointerenter", event => {
          activate(key);
          showTooltip(key, event);
        });
        hitNodes[key].addEventListener("pointermove", event => {
          activate(key);
          showTooltip(key, event);
        });
        hitNodes[key].addEventListener("pointerleave", () => {
          hideTooltip(key);
          activate(selectedSeries);
        });
      });
      root.querySelectorAll("button[data-series]").forEach(button => {
        if (button.hidden) return;
        button.addEventListener("mouseenter", () => activate(button.dataset.series));
        button.addEventListener("focus", () => activate(button.dataset.series));
        button.addEventListener("click", () => {
          selectedSeries = button.dataset.series;
          activate(selectedSeries);
        });
        button.addEventListener("mouseleave", () => activate(selectedSeries));
        button.addEventListener("blur", () => activate(selectedSeries));
      });
    })();
  </script>
</body>
</html>
"""
    document = (
        document.replace("__SUBTITLE__", html.escape(subtitle))
        .replace(
            "__SERIES_DATA__",
            json.dumps(series_data, separators=(",", ":"), allow_nan=False),
        )
        .replace(
            "__SHOW_ANNOTATION__",
            json.dumps(analysis_label == "All fields"),
        )
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(document, encoding="utf-8")


def plot_publication_gaps_by_cohort(
    data: pd.DataFrame,
    output_path: Path,
    analysis_label: str,
    minimum_authors: int,
) -> None:
    import matplotlib.pyplot as plt

    if data.empty:
        return
    plot_data = data.loc[
        data["number_of_authors"].ge(minimum_authors)
        & data["to_publication_number"].between(2, 10, inclusive="both")
    ].copy()
    if plot_data.empty:
        return
    figure, axis = plt.subplots(figsize=(11, 6.5))
    line_styles = {
        "1981-1990": {
            "color": "#E67E22",
            "markerfacecolor": "#E67E22",
            "linewidth": 2.2,
            "alpha": 0.82,
            "zorder": 2,
            "marker": "o",
            "markersize": 5.8,
            "markeredgecolor": "white",
            "markeredgewidth": 0.8,
        },
        "1991-2000": {
            "color": "#A8AFB7",
            "markerfacecolor": "#A8AFB7",
            "linewidth": 1.8,
            "alpha": 0.72,
            "zorder": 2,
            "marker": "s",
            "markersize": 5.6,
            "markeredgecolor": "white",
            "markeredgewidth": 0.8,
        },
        "2001-2010": {
            "color": "#68737E",
            "markerfacecolor": "#68737E",
            "linewidth": 2.0,
            "alpha": 0.80,
            "zorder": 2,
            "marker": "D",
            "markersize": 5.5,
            "markeredgecolor": "white",
            "markeredgewidth": 0.8,
        },
        "2011-2020": {
            "color": "#087E73",
            "markerfacecolor": "#087E73",
            "linewidth": 3.4,
            "alpha": 1.0,
            "zorder": 3,
            "marker": "^",
            "markersize": 7.0,
            "markeredgecolor": "white",
            "markeredgewidth": 0.9,
        },
    }
    for _, _, cohort_label in AUTHOR_ENTRY_COHORTS:
        cohort = plot_data.loc[plot_data["entry_cohort"].eq(cohort_label)]
        if cohort.empty:
            continue
        axis.plot(
            cohort["to_publication_number"],
            cohort["average_gap_years"],
            label=cohort_label,
            **line_styles[cohort_label],
        )
    axis.set_title(
        "Observed publication gaps are shorter for newer cohorts",
        loc="left",
        pad=38,
        fontsize=17,
        fontweight="semibold",
    )
    subtitle = (
        "Average years between consecutive top-five publications, by cohort "
        "of first publication"
    )
    if analysis_label != "All fields":
        subtitle = f"{analysis_label} \N{MIDDLE DOT} {subtitle}"
    axis.text(
        0,
        1.015,
        subtitle,
        transform=axis.transAxes,
        color="#69727D",
        fontsize=11,
        ha="left",
        va="bottom",
    )
    axis.set_xlabel("Publication number")
    axis.set_ylabel("Average gap from previous publication (years)")
    axis.set_xticks(range(2, 11))
    axis.set_xticklabels(
        [
            "1st\N{RIGHTWARDS ARROW}2nd",
            "2nd\N{RIGHTWARDS ARROW}3rd",
            "3rd\N{RIGHTWARDS ARROW}4th",
            "4th\N{RIGHTWARDS ARROW}5th",
            "5th\N{RIGHTWARDS ARROW}6th",
            "6th\N{RIGHTWARDS ARROW}7th",
            "7th\N{RIGHTWARDS ARROW}8th",
            "8th\N{RIGHTWARDS ARROW}9th",
            "9th\N{RIGHTWARDS ARROW}10th",
        ],
    )
    axis.set_xlim(1.8, 10.2)
    axis.set_ylim(bottom=0)
    axis.grid(axis="y", color="#D8DDE3", linewidth=0.8, alpha=0.8)
    axis.grid(axis="x", visible=False)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color("#B8C0C8")
    axis.spines["bottom"].set_color("#B8C0C8")
    axis.tick_params(colors="#4E5965")
    axis.legend(
        title="First-publication cohort",
        frameon=False,
        loc="upper right",
    )

    if analysis_label == "All fields":
        newest_first_gap = plot_data.loc[
            plot_data["entry_cohort"].eq("2011-2020")
            & plot_data["to_publication_number"].eq(2)
        ]
        if not newest_first_gap.empty:
            target = newest_first_gap.iloc[0]
            axis.annotate(
                (
                    "The observed gap from the 1st to 2nd publication "
                    "declined\nfrom about 5.9 to 3.5 years"
                ),
                xy=(2, float(target["average_gap_years"])),
                xytext=(3.35, 6.15),
                color="#4E5965",
                fontsize=10,
                ha="left",
                va="top",
                arrowprops={
                    "arrowstyle": "-",
                    "color": "#87919B",
                    "linewidth": 1.0,
                },
            )

    figure.text(
        0.115,
        0.025,
        (
            "Note: Values show mean years between consecutive publications. "
            "Cohorts are defined by the year of first top-five publication.\n"
            "Later publication numbers include only authors who reach that stage."
        ),
        color="#69727D",
        fontsize=9,
        ha="left",
        va="bottom",
    )
    save_figure(
        figure,
        output_path,
        tight_layout_rect=(0, 0.12, 1, 1),
    )


def write_interactive_first_to_second_gap_by_first_year(
    data: pd.DataFrame,
    output_path: Path,
    analysis_label: str,
    from_year: int,
    to_year: int,
) -> None:
    if data.empty:
        return
    plot_data = data.loc[
        data["previous_publication_year"].between(
            from_year,
            to_year,
            inclusive="both",
        )
    ].copy()
    for column in ["average_gap_years", "number_of_authors"]:
        plot_data[column] = pd.to_numeric(plot_data[column], errors="coerce")
    plot_data = plot_data.dropna(
        subset=["average_gap_years", "number_of_authors"]
    ).sort_values("previous_publication_year")
    if plot_data.empty:
        return

    series_data = [
        [
            int(row.previous_publication_year),
            round(float(row.average_gap_years), 6),
            int(row.number_of_authors),
        ]
        for row in plot_data.itertuples(index=False)
    ]
    subtitle = (
        "Average years between authors' first and second top-five publications, "
        "by year of first publication"
    )
    if analysis_label != "All fields":
        subtitle = f"{subtitle} \N{MIDDLE DOT} {analysis_label}"

    document = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>The observed gap to a second top-five publication has narrowed</title>
  <style>
    :root {
      color-scheme: light dark;
      --background: #ffffff;
      --foreground: #1e252b;
      --muted-foreground: #68737e;
      --grid: #dce1e6;
      --axis: #aeb7c0;
      --series: #087e73;
      --tooltip-background: #1e252b;
      --tooltip-foreground: #ffffff;
    }
    @media (prefers-color-scheme: dark) {
      :root {
        --background: #171a1d;
        --foreground: #f1f3f4;
        --muted-foreground: #b2bac2;
        --grid: #343a40;
        --axis: #626c75;
        --series: #46c6b8;
        --tooltip-background: #f1f3f4;
        --tooltip-foreground: #171a1d;
      }
    }
    * {
      box-sizing: border-box;
      letter-spacing: 0;
    }
    body {
      margin: 0;
      background: var(--background);
      color: var(--foreground);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    .chart {
      position: relative;
      width: 100%;
      max-width: 1100px;
      margin: 0 auto;
      padding: 24px;
    }
    h1 {
      margin: 0 0 6px;
      font-size: 28px;
      font-weight: 500;
    }
    .subtitle {
      margin: 0 0 14px;
      color: var(--muted-foreground);
      font-size: 15px;
    }
    .chart-svg {
      display: block;
      width: 100%;
      height: auto;
      overflow: visible;
    }
    .grid-line {
      stroke: var(--grid);
      stroke-width: 1;
    }
    .axis-line {
      stroke: var(--axis);
      stroke-width: 1;
    }
    .axis-label,
    .tick-label,
    .annotation {
      fill: var(--muted-foreground);
      font-family: inherit;
    }
    .axis-label { font-size: 14px; }
    .tick-label,
    .annotation { font-size: 12px; }
    .annotation,
    .annotation-line { pointer-events: none; }
    .annotation-line {
      stroke: var(--muted-foreground);
      stroke-width: 1;
    }
    .gap-line {
      fill: none;
      stroke: var(--series);
      stroke-width: 3;
      stroke-linecap: round;
      stroke-linejoin: round;
      transition: stroke-width 140ms ease;
    }
    .gap-line.is-active { stroke-width: 4; }
    .hit-line {
      fill: none;
      stroke: transparent;
      stroke-width: 20;
      pointer-events: stroke;
      cursor: crosshair;
    }
    .hover-marker {
      display: none;
      fill: var(--series);
      stroke: var(--background);
      stroke-width: 2;
      pointer-events: none;
    }
    .tooltip {
      position: absolute;
      z-index: 2;
      transform: translate(-50%, calc(-100% - 10px));
      max-width: 280px;
      padding: 7px 9px;
      border-radius: 4px;
      background: var(--tooltip-background);
      color: var(--tooltip-foreground);
      font-size: 13px;
      white-space: nowrap;
      pointer-events: none;
    }
    .tooltip[hidden] { display: none; }
    .note {
      margin: 4px 0 0;
      color: var(--muted-foreground);
      font-size: 12px;
      line-height: 1.45;
    }
    @media (max-width: 520px) {
      .chart { padding: 16px 10px; }
      h1 { font-size: 22px; }
      .subtitle { font-size: 13px; }
      .grid-line,
      .axis-line,
      .gap-line,
      .hit-line,
      .hover-marker,
      .annotation-line { vector-effect: non-scaling-stroke; }
      .tick-label { font-size: 22px; }
      .axis-label { font-size: 23px; }
      .annotation,
      .annotation-line { display: none; }
      .tooltip { white-space: normal; }
    }
    @media (prefers-reduced-motion: reduce) {
      .gap-line { transition: none; }
    }
  </style>
</head>
<body>
  <main id="first-to-second-gap-chart" class="chart">
    <h1>The observed gap to a second top-five publication has narrowed</h1>
    <p class="subtitle">__SUBTITLE__</p>
    <svg class="chart-svg" viewBox="0 0 900 520" role="img" aria-labelledby="chart-title chart-description">
      <title id="chart-title">Average years between first and second top-five publications</title>
      <desc id="chart-description">Hover over the line to inspect the first-publication year, average gap to a second publication, and number of authors.</desc>
    </svg>
    <div class="tooltip" role="status" aria-live="polite" hidden></div>
    <p class="note"><strong>Note:</strong> Cohorts are defined by the year of an author&rsquo;s first top-five publication. Values show the mean observed time to a second top-five publication. Recent cohorts may have incomplete follow-up.</p>
  </main>
  <script>
    (() => {
      const data = __SERIES_DATA__;
      const showAnnotation = __SHOW_ANNOTATION__;
      const root = document.getElementById("first-to-second-gap-chart");
      const svg = root.querySelector(".chart-svg");
      const tooltip = root.querySelector(".tooltip");
      const ns = "http://www.w3.org/2000/svg";
      const width = 900;
      const height = 520;
      const margin = { top: 18, right: 38, bottom: 62, left: 92 };
      const fromYear = __FROM_YEAR__;
      const toYear = __TO_YEAR__;
      const gapMin = 2;
      const gapMax = 7;
      const plotRight = width - margin.right;
      const plotBottom = height - margin.bottom;
      const x = year => margin.left + (year - fromYear) / (toYear - fromYear) * (plotRight - margin.left);
      const y = value => plotBottom - (value - gapMin) / (gapMax - gapMin) * (plotBottom - margin.top);
      const add = (name, attributes, text) => {
        const node = document.createElementNS(ns, name);
        Object.entries(attributes).forEach(([key, value]) => node.setAttribute(key, value));
        if (text !== undefined) node.textContent = text;
        svg.appendChild(node);
        return node;
      };
      const definitions = document.createElementNS(ns, "defs");
      const clipPath = document.createElementNS(ns, "clipPath");
      const clipRectangle = document.createElementNS(ns, "rect");
      clipPath.setAttribute("id", "graph5-plot-clip");
      clipRectangle.setAttribute("x", margin.left);
      clipRectangle.setAttribute("y", margin.top);
      clipRectangle.setAttribute("width", plotRight - margin.left);
      clipRectangle.setAttribute("height", plotBottom - margin.top);
      clipPath.appendChild(clipRectangle);
      definitions.appendChild(clipPath);
      svg.appendChild(definitions);

      for (let value = gapMin; value <= gapMax; value += 1) {
        const position = y(value);
        add("line", { class: "grid-line", x1: margin.left, x2: plotRight, y1: position, y2: position });
        add("text", { class: "tick-label", x: margin.left - 12, y: position + 4, "text-anchor": "end" }, value);
      }
      [1980, 1990, 2000, 2010, 2020].filter(year => year >= fromYear && year <= toYear).forEach(year => {
        add("text", { class: "tick-label", x: x(year), y: plotBottom + 25, "text-anchor": "middle" }, year);
      });
      add("line", { class: "axis-line", x1: margin.left, x2: margin.left, y1: margin.top, y2: plotBottom });
      add("line", { class: "axis-line", x1: margin.left, x2: plotRight, y1: plotBottom, y2: plotBottom });
      add("text", { class: "axis-label", x: (margin.left + plotRight) / 2, y: height - 12, "text-anchor": "middle" }, "Year of first top-five publication");
      add("text", { class: "axis-label", x: 22, y: (margin.top + plotBottom) / 2, "text-anchor": "middle", transform: `rotate(-90 22 ${(margin.top + plotBottom) / 2})` }, "Years between first and second publication");

      const path = data.map(([year, gap], index) => `${index ? "L" : "M"}${x(year).toFixed(2)},${y(gap).toFixed(2)}`).join(" ");
      const gapLine = add("path", { class: "gap-line", d: path, "clip-path": "url(#graph5-plot-clip)" });
      if (showAnnotation) {
        const firstPoint = data.find(point => point[0] === fromYear);
        const lastPoint = data.find(point => point[0] === toYear);
        if (firstPoint && lastPoint) {
          const change = lastPoint[1] - firstPoint[1];
          const sign = change < 0 ? "\u2212" : "+";
          add("line", { class: "annotation-line", x1: x(2007), y1: y(6.0), x2: x(lastPoint[0]), y2: y(lastPoint[1]) });
          add("text", { class: "annotation", x: x(2002), y: y(6.35) }, `${sign}${Math.abs(change).toFixed(1)} years`);
          add("text", { class: "annotation", x: x(2002), y: y(6.05) }, `change from ${fromYear} to ${toYear}`);
        }
      }
      const marker = add("circle", { class: "hover-marker", r: 5 });
      const hitLine = add("path", { class: "hit-line", d: path, "clip-path": "url(#graph5-plot-clip)" });

      const showTooltip = event => {
        const svgRect = svg.getBoundingClientRect();
        const svgX = (event.clientX - svgRect.left) / svgRect.width * width;
        const nearest = data.reduce((best, point) => {
          return Math.abs(x(point[0]) - svgX) < Math.abs(x(best[0]) - svgX) ? point : best;
        });
        marker.setAttribute("cx", x(nearest[0]));
        marker.setAttribute("cy", y(nearest[1]));
        marker.style.display = "block";
        gapLine.classList.add("is-active");
        tooltip.textContent = `${nearest[0]} \u00b7 ${nearest[1].toFixed(2)} years \u00b7 ${nearest[2].toLocaleString()} authors`;
        tooltip.hidden = false;
        const rootRect = root.getBoundingClientRect();
        const markLeft = svgRect.left - rootRect.left + x(nearest[0]) / width * svgRect.width;
        const markTop = svgRect.top - rootRect.top + y(nearest[1]) / height * svgRect.height;
        tooltip.style.left = `${Math.max(135, Math.min(root.clientWidth - 135, markLeft))}px`;
        tooltip.style.top = `${Math.max(32, markTop)}px`;
      };
      const hideTooltip = () => {
        marker.style.display = "none";
        gapLine.classList.remove("is-active");
        tooltip.hidden = true;
      };
      hitLine.addEventListener("pointerenter", showTooltip);
      hitLine.addEventListener("pointermove", showTooltip);
      hitLine.addEventListener("pointerleave", hideTooltip);
    })();
  </script>
</body>
</html>
"""
    document = (
        document.replace("__SUBTITLE__", html.escape(subtitle))
        .replace(
            "__SERIES_DATA__",
            json.dumps(series_data, separators=(",", ":"), allow_nan=False),
        )
        .replace(
            "__SHOW_ANNOTATION__",
            json.dumps(analysis_label == "All fields"),
        )
        .replace("__FROM_YEAR__", str(from_year))
        .replace("__TO_YEAR__", str(to_year))
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(document, encoding="utf-8")


def plot_first_to_second_gap_by_first_year(
    data: pd.DataFrame,
    output_path: Path,
    analysis_label: str,
    from_year: int,
    to_year: int,
) -> None:
    import matplotlib.pyplot as plt

    if data.empty:
        return
    plot_data = data.loc[
        data["previous_publication_year"].between(
            from_year,
            to_year,
            inclusive="both",
        )
    ].copy()
    if plot_data.empty:
        return
    plot_data = plot_data.sort_values("previous_publication_year")
    figure, axis = plt.subplots(figsize=(11, 6.5))
    axis.plot(
        plot_data["previous_publication_year"],
        plot_data["average_gap_years"],
        color="#087E73",
        linewidth=3.2,
        marker="o",
        markevery=5,
        markersize=4.8,
        markerfacecolor="#087E73",
        markeredgecolor="white",
        markeredgewidth=0.8,
        zorder=3,
    )
    axis.set_title(
        "The observed gap to a second top-five publication has narrowed",
        loc="left",
        pad=38,
        fontsize=17,
        fontweight="semibold",
    )
    subtitle = (
        "Average years between authors' first and second top-five publications, "
        "by year of first publication"
    )
    if analysis_label != "All fields":
        subtitle = f"{subtitle} \N{MIDDLE DOT} {analysis_label}"
    axis.text(
        0,
        1.015,
        subtitle,
        transform=axis.transAxes,
        color="#69727D",
        fontsize=11,
        ha="left",
        va="bottom",
    )
    axis.set_xlabel("Year of first top-five publication")
    axis.set_ylabel("Years between first and second publication")
    axis.set_xlim(from_year, to_year)
    axis.set_ylim(2, 7)
    axis.grid(axis="y", color="#D8DDE3", linewidth=0.8, alpha=0.8)
    axis.grid(axis="x", visible=False)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color("#B8C0C8")
    axis.spines["bottom"].set_color("#B8C0C8")
    axis.tick_params(colors="#4E5965")

    if analysis_label == "All fields":
        start = plot_data.loc[
            plot_data["previous_publication_year"].eq(from_year)
        ]
        end = plot_data.loc[
            plot_data["previous_publication_year"].eq(to_year)
        ]
        if not start.empty and not end.empty:
            start_gap = float(start.iloc[0]["average_gap_years"])
            end_gap = float(end.iloc[0]["average_gap_years"])
            change = end_gap - start_gap
            axis.annotate(
                (
                    f"{change:.1f} years\n"
                    f"change from {from_year} to {to_year}"
                ).replace("-", "\N{MINUS SIGN}", 1),
                xy=(to_year, end_gap),
                xytext=(2002, 6.25),
                color="#4E5965",
                fontsize=11,
                ha="left",
                va="center",
                arrowprops={
                    "arrowstyle": "-",
                    "color": "#87919B",
                    "linewidth": 1.0,
                },
            )

    figure.text(
        0.115,
        0.025,
        (
            "Note: Cohorts are defined by the year of an author's first top-five publication. "
            "Values show the mean observed time to a second top-five publication.\n"
            "Recent cohorts may have incomplete follow-up."
        ),
        color="#69727D",
        fontsize=9,
        ha="left",
        va="bottom",
    )
    save_figure(
        figure,
        output_path,
        tight_layout_rect=(0, 0.10, 1, 1),
    )


def save_figure(
    figure,
    output_path: Path,
    tight_layout_rect: tuple[float, float, float, float] | None = None,
) -> None:
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout(rect=tight_layout_rect)
    figure.savefig(output_path, dpi=300)
    plt.close(figure)


def print_run_summary(
    paper_author: pd.DataFrame,
    jel_complete: pd.DataFrame,
    merged: pd.DataFrame,
    top5: pd.DataFrame,
    summary: pd.DataFrame,
) -> None:
    print("Author concentration analysis completed:")
    print(f"  Paper-author input rows: {len(paper_author)}")
    print(f"  JEL input rows: {len(jel_complete)}")
    print(f"  Merged paper-author rows: {len(merged)}")
    print(f"  Top-five paper-author rows: {len(top5)}")
    print(
        "  Top-five rows with Final_jel_code: "
        f"{top5['Final_jel_code'].fillna('').astype(str).str.strip().ne('').sum()}"
    )
    print()
    print(summary.to_string(index=False))
    print()
    print(f"Analysis summary CSV: {ANALYSIS_SUMMARY_CSV}")


if __name__ == "__main__":
    main()
