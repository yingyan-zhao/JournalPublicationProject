from pathlib import Path
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

    # Step 2. Keep only the requested paper and JEL columns.
    jel_complete = read_csv_data(JEL_COMPLETE_INPUT_CSV)
    jel_complete_cleaned = keep_columns(
        jel_complete,
        JEL_COMPLETE_COLUMNS_TO_KEEP,
    )
    write_csv(jel_complete_cleaned, JEL_COMPLETE_CLEANED_CSV)

    # Step 3. Merge paper-author rows to paper data in a many-to-one merge.
    merged = merge_paper_author_with_jel(
        paper_author_cleaned,
        jel_complete_cleaned,
    )
    write_csv(merged, PAPER_AUTHOR_JEL_MERGED_CSV)

    # Step 4. Keep top-five journals and construct the final JEL classification.
    top5 = keep_top_five_journals(merged)
    top5 = add_final_jel_code(top5)
    write_csv(top5, PAPER_AUTHOR_JEL_TOP5_CSV)

    # Steps 5-9. Produce the five analyses for all top-five papers.
    summaries = [
        run_analysis(
            data=top5,
            output_dir=OVERALL_OUTPUT_DIR,
            analysis_label="All fields",
            minimum_gap_authors=30,
        )
    ]

    # Step 10. Repeat Steps 5-9 separately for each requested JEL field.
    for field in JEL_FIELDS:
        field_data = keep_jel_field(top5, field)
        summaries.append(
            run_analysis(
                data=field_data,
                output_dir=FIELD_OUTPUT_DIR / field,
                analysis_label=f"JEL {field}",
                minimum_gap_authors=10,
            )
        )

    summary = pd.DataFrame(summaries)
    write_csv(summary, ANALYSIS_SUMMARY_CSV)
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

    # Step 6 / Graphs 2.1 and 2.2.
    new_author_summary = new_author_statistics_by_year(
        paper_years,
        author_papers,
    )
    write_csv(
        new_author_summary,
        output_dir / "Graph2_NewAuthors_ByYear.csv",
    )
    plot_share_and_count(
        new_author_summary,
        output_dir / "Graph2_1_NewAuthorCountShare_1980_2025.png",
        analysis_label=analysis_label,
        from_year=1980,
        to_year=2025,
        share_min=0.30,
        share_max=0.70,
        share_column="share_of_authors_who_are_new",
        count_column="number_of_new_authors",
        title="New Authors",
        share_label="Share of authors who are new",
        count_label="Number of new authors",
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
    for percentage in TOP_AUTHOR_PERCENTAGES:
        series = plot_data.loc[
            plot_data["top_author_percentage"].eq(percentage)
        ]
        axis.plot(
            series["publication_year"],
            series["share_of_papers_with_top_author"],
            linewidth=2.2,
            marker=markers[percentage],
            markevery=5,
            markersize=5,
            label=f"Top {percentage}%",
        )
    axis.set_title(f"Share of Papers with a Top Author - {analysis_label}")
    axis.set_xlabel("Publication year")
    axis.set_ylabel("Share of unique papers")
    axis.yaxis.set_major_formatter(PercentFormatter(1.0))
    axis.set_xlim(left=from_year)
    axis.set_ylim(bottom=0)
    axis.grid(True, alpha=0.25)
    axis.legend(title="Rank over prior 20 years", frameon=False)
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
        ("share_with_experienced_coauthor", "With experienced coauthor", "o"),
        (
            "share_only_with_other_new_coauthors",
            "Only with other new coauthors",
            "s",
        ),
        ("share_solo", "Solo", "^"),
    ]
    for column, label, marker in series:
        axis.plot(
            plot_data["first_top5_publication_year"],
            plot_data[column],
            linewidth=2.2,
            marker=marker,
            markevery=5,
            markersize=5,
            label=label,
        )
    axis.set_title(f"How New Authors Enter - {analysis_label}")
    axis.set_xlabel("First top-five publication year")
    axis.set_ylabel("Share of new authors")
    axis.yaxis.set_major_formatter(PercentFormatter(1.0))
    axis.set_xlim(from_year, to_year)
    axis.set_ylim(0, 1)
    axis.grid(True, alpha=0.25)
    axis.legend(frameon=False)
    save_figure(figure, output_path)


def plot_publication_gaps_by_cohort(
    data: pd.DataFrame,
    output_path: Path,
    analysis_label: str,
    minimum_authors: int,
) -> None:
    import matplotlib.pyplot as plt

    if data.empty:
        return
    plot_data = data.loc[data["number_of_authors"].ge(minimum_authors)].copy()
    if plot_data.empty:
        return
    figure, axis = plt.subplots(figsize=(11, 6.5))
    markers = ["o", "s", "^", "D"]
    for marker, (_, _, cohort_label) in zip(markers, AUTHOR_ENTRY_COHORTS):
        cohort = plot_data.loc[plot_data["entry_cohort"].eq(cohort_label)]
        if cohort.empty:
            continue
        axis.plot(
            cohort["to_publication_number"],
            cohort["average_gap_years"],
            linewidth=2.2,
            marker=marker,
            markersize=5,
            label=cohort_label,
        )
    maximum_transition = int(plot_data["to_publication_number"].max())
    axis.set_title(f"Average Publication Gaps by Entry Cohort - {analysis_label}")
    axis.set_xlabel("Later publication in the sequence")
    axis.set_ylabel("Average gap since previous publication (years)")
    axis.set_xticks(range(2, maximum_transition + 1))
    axis.set_xticklabels(
        [f"{number - 1} to {number}" for number in range(2, maximum_transition + 1)],
        rotation=45,
        ha="right",
    )
    axis.set_ylim(bottom=0)
    axis.grid(True, alpha=0.25)
    axis.legend(title="First-publication cohort", frameon=False)
    save_figure(figure, output_path)


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
    ]
    if plot_data.empty:
        return
    figure, axis = plt.subplots(figsize=(11, 6.5))
    axis.plot(
        plot_data["previous_publication_year"],
        plot_data["average_gap_years"],
        linewidth=2.2,
        marker="o",
        markevery=5,
        markersize=5,
        label="First to second publication",
    )
    axis.set_title(f"Average First-to-Second Publication Gap - {analysis_label}")
    axis.set_xlabel("Year of first top-five publication")
    axis.set_ylabel("Average years to second top-five publication")
    axis.set_xlim(from_year, to_year)
    axis.set_ylim(bottom=0)
    axis.grid(True, alpha=0.25)
    axis.legend(frameon=False)
    save_figure(figure, output_path)


def save_figure(figure, output_path: Path) -> None:
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
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
