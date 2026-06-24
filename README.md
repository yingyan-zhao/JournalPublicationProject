# Economics Top-Journal Publication Concentration

This project studies how publications in top economics journals are concentrated among authors over time.

## Research Question

How concentrated are publications in top economics journals among individual authors, and how has that concentration changed across cohorts, journals, fields, and institutions?

## Starting Scope

The initial unit of analysis is an author-journal-year publication record. A single article with multiple authors should contribute fractional publication credit by default, while full-count credit can be used as a robustness check.

Candidate journal set:

- American Economic Review
- Econometrica
- Journal of Political Economy
- Quarterly Journal of Economics
- Review of Economic Studies

Optional extensions:

- Add Review of Economics and Statistics, Journal of Finance, Journal of Monetary Economics, or field top journals.
- Compare economics with adjacent disciplines.
- Study concentration by institution, gender, PhD institution, country, or field.

## Data Schema

Place raw source files in `data/raw/`. The starter scripts expect a processed file at:

`data/processed/publications.csv`

Required columns:

- `publication_id`: stable article identifier
- `year`: publication year
- `journal`: journal name
- `author_id`: stable author identifier
- `author_name`: display name
- `author_position`: author order, if available
- `n_authors`: number of authors on the publication
- `abstract`: article abstract
- `keywords`: article keywords
- `jel_codes`: JEL classification codes
- `acknowledgments`: thanks, acknowledgments, funding notes, and related article notes when available

Recommended columns:

- `title`
- `doi`
- `institution`
- `field`: broad economics field derived from JEL classification
- `first_submission_date`
- `accepted_date`
- `online_publication_date`
- `issue_publication_date`
- `first_working_paper_date`
- `first_working_paper_source`
- `first_working_paper_url`
- `first_working_paper_repository`
- `citation_count`
- `citation_source`
- `citation_date`
- `citation_url`
- `openalex_primary_domain`
- `openalex_primary_field`
- `openalex_primary_subfield`
- `openalex_primary_topic`
- `openalex_topics`
- `submission_date_source`
- `acknowledgments_source`
- `source`

## Concentration Measures

The core analysis will estimate:

- Share of publications by top 1%, 5%, and 10% of authors
- Herfindahl-Hirschman Index
- Gini coefficient
- Lorenz curves
- Author productivity distribution
- Entry, persistence, and repeat-publication rates

## Reproducible Workflow

1. Collect the first version of the dataset from OpenAlex:

```bash
python -m src.econ_pub_concentration.collect_openalex --from-year 2015 --to-year 2020 --mailto your_email@example.com
```

2. Enrich the OpenAlex records with Crossref metadata using DOI matches.
3. Add fields not fully covered by OpenAlex or Crossref, such as JEL codes, acknowledgments, submission dates, and first working-paper dates.
4. Clean and harmonize the data into `data/processed/publications.csv`.
5. Run:

```bash
python -m src.econ_pub_concentration.analyze data/processed/publications.csv
```

Outputs are written to:

- `outputs/tables/`
- `outputs/figures/`

See [docs/openalex_data_plan.md](docs/openalex_data_plan.md) for the OpenAlex collection plan.

## Project Structure

```text
data/
  raw/          # Original downloaded or hand-collected files
  processed/    # Analysis-ready CSV files
docs/           # Notes, definitions, and design choices
notebooks/      # Exploratory notebooks
outputs/
  figures/      # Generated plots
  tables/       # Generated CSV summaries
src/
  econ_pub_concentration/
tests/
```
