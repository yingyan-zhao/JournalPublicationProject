from pathlib import Path
import os
import re

import pandas as pd


os.chdir("/Users/yingyan_zhao/Dropbox/JournalPublicationProject")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

INPUT_CSV = (
    Path("data/trainingmodel")
    / "JEL_Training_Data_Complete_Observed_And_Predicted.csv"
)
OUTPUT_DIR = Path("data/processed/field_concentration")
OUTPUT_CLEANED_TOP5_CSV = OUTPUT_DIR / "JEL_Training_Data_Top5_FieldCleaned.csv"
OUTPUT_FIELD_RANKING_CSV = OUTPUT_DIR / "JEL_Field_Ranking_Top5.csv"
OUTPUT_FIELD_YEAR_TREND_CSV = OUTPUT_DIR / "JEL_Field_YearTrend_Top5.csv"
OUTPUT_FIELD_YEAR_TREND_PNG = OUTPUT_DIR / "JEL_Field_YearTrend_Top5.png"
OUTPUT_FIELD_YEAR_SHARE_TREND_CSV = OUTPUT_DIR / "JEL_Field_YearShareTrend_After2000_Top5.csv"
OUTPUT_FIELD_YEAR_SHARE_TREND_PNG = OUTPUT_DIR / "JEL_Field_YearShareTrend_After2000_Top5.png"

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
FIELDS_FOR_TREND = ["D", "C", "E", "J", "H", "O", "L", "G", "F", "I"]

########################################################################
# In 01_ExploringConcentrationField.py,
# Step 1. Clean JEL_Training_Data_Complete_Observed_And_Predicted.csv, by keeping doi title journalname publication_year jel_code_full tfidf_predicted_jel_code_full tfidf_max_confidence specter2_predicted_jel_code_full specter2_max_confidence scibert_predicted_jel_code_full scibert_max_confidence ensemble_predicted_jel_code_full ensemble_max_confidence. Generate jel_code_final. It equals to jel_code_full if jel_code_full is not blank. It equals  to scibert_predicted_jel_code_full if jel_code_full is blank or missing.
# Step 2. Keep papers in only top 5 journals.
# Step 3. Rank fields by how many number of  in each fields (fields are defined by jel_code_full)
# Step 4. For fields D,C,E,J,H,O,L,G,F,I, divide the number of papers in each field by the total number of papers in each year. Then draw the line graph for the trend after 2000.
########################################################################

def main() -> None:
    data = read_input_data(INPUT_CSV)
    cleaned = keep_columns(data, JEL_COMPLETE_COLUMNS_TO_KEEP)
    cleaned["jel_code_final"] = coalesce_text_columns(
        cleaned,
        ["jel_code_full", "scibert_predicted_jel_code_full"],
    )

    top5 = keep_top_five_journals(cleaned)
    top5 = top5.drop_duplicates(["doi", "title"]).copy()

    field_ranking = rank_fields_by_papers(top5)
    field_year_trend = field_year_trend_table(top5, FIELDS_FOR_TREND)
    field_year_share_trend = field_year_share_trend_table(
        top5,
        field_year_trend,
        from_year=2000,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    top5.to_csv(OUTPUT_CLEANED_TOP5_CSV, index=False)
    field_ranking.to_csv(OUTPUT_FIELD_RANKING_CSV, index=False)
    field_year_trend.to_csv(OUTPUT_FIELD_YEAR_TREND_CSV, index=False)
    field_year_share_trend.to_csv(OUTPUT_FIELD_YEAR_SHARE_TREND_CSV, index=False)
    write_field_year_trend_figure(
        field_year_trend,
        OUTPUT_FIELD_YEAR_TREND_PNG,
    )
    write_field_year_share_trend_figure(
        field_year_share_trend,
        OUTPUT_FIELD_YEAR_SHARE_TREND_PNG,
    )

    print("Field concentration exploration summary:")
    print(f"  Input CSV: {INPUT_CSV}")
    print(f"  Input rows: {len(data)}")
    print(f"  Kept columns: {JEL_COMPLETE_COLUMNS_TO_KEEP}")
    print(f"  Top-five paper rows: {len(top5)}")
    print(f"  Papers with observed JEL: {count_nonblank(top5, 'jel_code_full')}")
    print(
        "  Papers using SciBERT JEL because observed JEL is blank: "
        f"{count_scibert_fallback(top5)}"
    )
    print(f"  Cleaned top-five output CSV: {OUTPUT_CLEANED_TOP5_CSV}")
    print()
    print("JEL field ranking:")
    print(field_ranking.to_string(index=False))
    print(f"  Field ranking output CSV: {OUTPUT_FIELD_RANKING_CSV}")
    print()
    print("JEL field year trend:")
    print(field_year_trend.head(30).to_string(index=False))
    print(f"  Full table rows: {len(field_year_trend)}")
    print(f"  Field-year trend output CSV: {OUTPUT_FIELD_YEAR_TREND_CSV}")
    print(f"  Field-year trend figure PNG: {OUTPUT_FIELD_YEAR_TREND_PNG}")
    print()
    print("JEL field yearly share trend after 2000:")
    print(field_year_share_trend.head(30).to_string(index=False))
    print(f"  Full table rows: {len(field_year_share_trend)}")
    print(f"  Field-year share trend output CSV: {OUTPUT_FIELD_YEAR_SHARE_TREND_CSV}")
    print(f"  Field-year share trend figure PNG: {OUTPUT_FIELD_YEAR_SHARE_TREND_PNG}")


def read_input_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing input CSV: {path}")
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def keep_columns(data: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    missing_columns = [column for column in columns if column not in data.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    return data[columns].copy()


def coalesce_text_columns(data: pd.DataFrame, columns: list[str]) -> pd.Series:
    values = pd.Series([""] * len(data), index=data.index)
    for column in columns:
        candidate = data[column].fillna("").astype(str).str.strip()
        values = values.mask(values.eq(""), candidate)
    return values


def keep_top_five_journals(data: pd.DataFrame) -> pd.DataFrame:
    cleaned = data.copy()
    normalized_journal = cleaned["journalname"].apply(normalize_journalname)
    cleaned["journalname"] = normalized_journal.map(TOP_FIVE_JOURNALS).fillna(
        cleaned["journalname"]
    )
    return cleaned.loc[normalized_journal.isin(TOP_FIVE_JOURNALS)].copy()


def normalize_journalname(value) -> str:
    return " ".join(str(value or "").strip().lower().split())


def rank_fields_by_papers(data: pd.DataFrame) -> pd.DataFrame:
    paper_fields = []
    for _, row in data.iterrows():
        doi = str(row.get("doi", "")).strip()
        title = str(row.get("title", "")).strip()
        paper_id = doi if doi else title
        for field in extract_jel_fields(row.get("jel_code_final", "")):
            paper_fields.append(
                {
                    "paper_id": paper_id,
                    "field": field,
                }
            )

    if not paper_fields:
        return pd.DataFrame(
            columns=["rank", "field", "n_papers", "share_of_field_paper_counts"]
        )

    field_data = pd.DataFrame(paper_fields).drop_duplicates(["paper_id", "field"])
    ranking = (
        field_data.groupby("field", as_index=False)
        .agg(n_papers=("paper_id", "nunique"))
        .sort_values(["n_papers", "field"], ascending=[False, True])
        .reset_index(drop=True)
    )
    total_field_paper_counts = ranking["n_papers"].sum()
    ranking.insert(0, "rank", range(1, len(ranking) + 1))
    ranking["share_of_field_paper_counts"] = (
        ranking["n_papers"] / total_field_paper_counts
    ).round(4)
    return ranking


def field_year_trend_table(data: pd.DataFrame, fields_to_keep: list[str]) -> pd.DataFrame:
    paper_fields = []
    for _, row in data.iterrows():
        year = str(row.get("publication_year", "")).strip()
        if not year:
            continue
        doi = str(row.get("doi", "")).strip()
        title = str(row.get("title", "")).strip()
        paper_id = doi if doi else title
        for field in extract_jel_fields(row.get("jel_code_final", "")):
            if field in fields_to_keep:
                paper_fields.append(
                    {
                        "publication_year": year,
                        "field": field,
                        "paper_id": paper_id,
                    }
                )

    if not paper_fields:
        return pd.DataFrame(columns=["publication_year", "field", "n_papers"])

    field_data = pd.DataFrame(paper_fields).drop_duplicates(
        ["publication_year", "field", "paper_id"]
    )
    trend = (
        field_data.groupby(["publication_year", "field"], as_index=False)
        .agg(n_papers=("paper_id", "nunique"))
    )
    trend["publication_year"] = pd.to_numeric(
        trend["publication_year"],
        errors="coerce",
    )
    trend = trend.loc[trend["publication_year"].notna()].copy()
    trend["publication_year"] = trend["publication_year"].astype(int)

    all_years = range(trend["publication_year"].min(), trend["publication_year"].max() + 1)
    complete_index = pd.MultiIndex.from_product(
        [all_years, fields_to_keep],
        names=["publication_year", "field"],
    )
    trend = (
        trend.set_index(["publication_year", "field"])
        .reindex(complete_index, fill_value=0)
        .reset_index()
        .sort_values(["publication_year", "field"], kind="mergesort")
    )
    return trend


def write_field_year_trend_figure(data: pd.DataFrame, output_path: Path) -> None:
    import matplotlib.pyplot as plt

    if data.empty:
        return

    fig, ax = plt.subplots(figsize=(11, 6))
    for field in FIELDS_FOR_TREND:
        field_data = data.loc[data["field"].eq(field)].copy()
        if field_data.empty:
            continue
        ax.plot(
            field_data["publication_year"],
            field_data["n_papers"],
            linewidth=2,
            label=field,
        )

    ax.set_title("Number of Top-Five Papers by JEL Field Over Time")
    ax.set_xlabel("Publication year")
    ax.set_ylabel("Number of papers")
    ax.set_xlim(data["publication_year"].min(), data["publication_year"].max())
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.25)
    ax.legend(title="JEL field", frameon=False, ncol=5)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def field_year_share_trend_table(
    top5: pd.DataFrame,
    field_year_trend: pd.DataFrame,
    from_year: int,
) -> pd.DataFrame:
    yearly_totals = yearly_total_papers(top5)
    share_trend = field_year_trend.merge(
        yearly_totals,
        on="publication_year",
        how="left",
        validate="many_to_one",
    )
    share_trend["field_share_of_year_papers"] = (
        share_trend["n_papers"] / share_trend["total_papers_in_year"]
    ).round(4)
    return share_trend.loc[share_trend["publication_year"].ge(from_year)].copy()


def yearly_total_papers(data: pd.DataFrame) -> pd.DataFrame:
    cleaned = data.copy()
    cleaned["publication_year"] = pd.to_numeric(
        cleaned["publication_year"],
        errors="coerce",
    )
    cleaned = cleaned.loc[cleaned["publication_year"].notna()].copy()
    cleaned["publication_year"] = cleaned["publication_year"].astype(int)
    cleaned["paper_id"] = cleaned.apply(paper_identifier, axis=1)
    return (
        cleaned.groupby("publication_year", as_index=False)
        .agg(total_papers_in_year=("paper_id", "nunique"))
        .sort_values("publication_year", kind="mergesort")
    )


def paper_identifier(row: pd.Series) -> str:
    doi = str(row.get("doi", "")).strip()
    title = str(row.get("title", "")).strip()
    return doi if doi else title


def write_field_year_share_trend_figure(data: pd.DataFrame, output_path: Path) -> None:
    import matplotlib.pyplot as plt

    if data.empty:
        return

    fig, ax = plt.subplots(figsize=(11, 6))
    for field in FIELDS_FOR_TREND:
        field_data = data.loc[data["field"].eq(field)].copy()
        if field_data.empty:
            continue
        ax.plot(
            field_data["publication_year"],
            field_data["field_share_of_year_papers"],
            linewidth=2,
            label=field,
        )

    ax.set_title("Share of Top-Five Papers by JEL Field After 2000")
    ax.set_xlabel("Publication year")
    ax.set_ylabel("Field papers / total papers in year")
    ax.set_xlim(data["publication_year"].min(), data["publication_year"].max())
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.25)
    ax.legend(title="JEL field", frameon=False, ncol=5)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def extract_jel_fields(value) -> list[str]:
    text = str(value or "").upper()
    fields = []
    seen = set()
    for token in re.findall(r"\b[A-Z](?:\d{2})?\b", text):
        field = token[:1]
        if field and field not in seen:
            fields.append(field)
            seen.add(field)
    return fields


def count_nonblank(data: pd.DataFrame, column: str) -> int:
    return int(data[column].fillna("").astype(str).str.strip().ne("").sum())


def count_scibert_fallback(data: pd.DataFrame) -> int:
    observed_blank = data["jel_code_full"].fillna("").astype(str).str.strip().eq("")
    scibert_nonblank = (
        data["scibert_predicted_jel_code_full"]
        .fillna("")
        .astype(str)
        .str.strip()
        .ne("")
    )
    return int((observed_blank & scibert_nonblank).sum())


if __name__ == "__main__":
    main()
