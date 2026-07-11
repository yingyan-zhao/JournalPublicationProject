# Has It Become Harder to Break Into Economics' Leading Journals Without an Established Publication Record?

## Overview

Has it become harder for researchers without an established publication record to break into economics' leading journals? This project examines how authorship and entry into the journals commonly known as the "Top Five" have changed over time.[^1]

A Top Five publication is widely viewed as an important marker of research success in economics. Publication outcomes are likely to reflect genuine differences in research quality, ability, experience, persistence, and access to productive collaborators and feedback. Established scholars may produce stronger work because they have accumulated knowledge, learned how to identify promising questions, and developed more effective research processes.

Prior success may also create advantages that extend beyond research ability alone. Reputation can increase a paper's visibility, facilitate collaboration, and provide access to professional networks and institutional resources. Editors and referees may also treat an author's publication record as a signal of quality when evaluating uncertain or highly specialized work. These mechanisms need not involve deliberate favoritism, and publication data alone cannot distinguish them from the effects of experience and skill.

I therefore treat the analysis as descriptive rather than causal. The project asks whether publication in the Top Five has become more concentrated among experienced authors and whether first-time authors increasingly enter through collaboration with scholars who already have a Top Five publication.

[^1]: The conventional Top Five journals in economics are the *American Economic Review*, *Econometrica*, *Journal of Political Economy*, *Quarterly Journal of Economics*, and *Review of Economic Studies*.

## Data and Methods

### Data Sources

I construct a paper-level dataset covering Top Five publications from 1950 through 2026. Core bibliographic records come from the OpenAlex and Crossref APIs. I enrich these records with metadata from RePEc, AEA journal pages, and NBER, and use targeted web scraping when abstracts, keywords, author information, or JEL codes are unavailable from structured sources.

### Record Linkage and Deduplication

Records are linked across sources primarily by normalized Digital Object Identifiers (DOIs). When a DOI is unavailable or unsuccessful, standardized article titles provide a secondary matching key. The pipeline reconciles DOI variants, consolidates duplicate records, and standardizes journal names, publication dates, titles, author names, and institutional affiliations.

### Author Disambiguation

To construct author-level publication histories, I normalize names across sources and combine exact matching with cautious fuzzy matching. The procedure addresses differences in accents, initials, name order, punctuation, and formatting. Ambiguous matches are flagged for manual review, and each consolidated author is assigned a stable `author_id`.

### Classification of Missing JEL Codes

Approximately 72% of papers in the current Top Five analytic sample do not have an observed broad JEL category. I treat JEL assignment as a multi-label text-classification problem because a paper may belong to several fields.

I compare three approaches using article titles, abstracts, and keywords:

1. TF-IDF features with one-vs-rest logistic regression;
2. SPECTER2 embeddings with one-vs-rest logistic regression; and
3. A fine-tuned SciBERT classifier.

I also evaluate a weighted ensemble of the three models. Model performance is assessed on papers with observed JEL codes using micro and macro F1, precision, recall, Hamming loss, and subset accuracy. Observed and predicted codes remain separately identified. The current field analysis uses an observed JEL code when available and a SciBERT prediction otherwise.

### Current Analytic Sample

The cleaned Top Five sample currently contains:

- **23,486 distinct papers**;
- **15,608 disambiguated authors**; and
- **42,468 paper-author observations**.

The sample covers 1950-2026. Because 2026 is incomplete, observations from that year are provisional. Coverage of abstracts, JEL codes, and institutional affiliations varies across journals and over time.

### Key Definitions

- **Publication counting:** The author rankings use full counting, so a coauthored paper contributes one publication to each author. Paper-level shares count each paper only once.
- **Top-ranked author:** An author in the top 1%, 5%, or 10% of the publication-count distribution calculated from the preceding 20 years.
- **New author:** An author whose first observed Top Five publication occurs in the indicated year.
- **Experienced coauthor:** A coauthor with at least one observed Top Five publication before the focal author's first Top Five publication.
- **Field:** A broad JEL category based on the observed code when available and the SciBERT-predicted code otherwise.

## Research Question

How have entry, persistence, and the concentration of authorship in economics' Top Five journals changed over time, and has publication without a prior Top Five record become less common?

## Preliminary Main Findings

Each figure links to its companion HTML file. Download and open that file locally to use the interactive hover details.

### Finding 1: Top-ranked authors appear on a growing share of papers

The share of papers with at least one author ranked in the preceding 20-year top 10% increased from **27.8% in 1980 to 46.4% in 2025**. The corresponding share increased from **3.4% to 12.9%** for the top 1% and from **18.4% to 29.7%** for the top 5%. By this measure, authorship has become increasingly concentrated around scholars with strong recent Top Five publication records.

[![Share of papers with a top-ranked author](outputs/figures/overall/Graph1_TopAuthorPaperShares_After1980.png)](outputs/figures/overall/Graph1_TopAuthorPaperShares_After1980.html)

*Figure 1. Share of papers with at least one top-ranked author. Rankings are based on publication counts during the preceding 20 years.*

### Finding 2: New authors account for a smaller share of authors

New authors represented **44.9% of authors publishing in 1980** and **35.9% in 2025**. The annual series fluctuates, but its long-run direction is downward. This finding concerns the composition of authors, not the number of new authors, which can increase as both the number of papers and average team size grow.

[![Annual share of authors who are new](outputs/figures/overall/Graph2_1_NewAuthorCountShare_1980_2025.png)](outputs/figures/overall/Graph2_1_NewAuthorShare_1980_2025.html)

*Figure 2. Share of authors whose first observed Top Five publication occurs in each year.*

### Finding 3: New authors increasingly publish with experienced coauthors

Among authors entering the Top Five, the share whose first publication included at least one experienced coauthor increased from **27.3% in 1980 to 76.8% in 2025**. Over the same period, the solo-authored share fell from **48.8% to 6.9%**, while the share publishing only with other new authors declined from **23.9% to 16.3%**. Entry has therefore become much more closely associated with collaboration with authors who already have Top Five experience.

[![Coauthor composition of new authors' first publications](outputs/figures/overall/Graph3_NewAuthorCoauthorType_1980_2025.png)](outputs/figures/overall/Graph3_NewAuthorCoauthorType_1980_2025.html)

*Figure 3. Coauthor composition of new authors' first Top Five publications. The three categories are mutually exclusive.*

### Finding 4: Observed publication gaps are shorter for newer cohorts

Among authors who publish at least twice in the Top Five, the mean observed gap between the first and second publication declined from **5.9 years for the 1981-1990 entry cohort** to **3.5 years for the 2011-2020 cohort**. Later publication gaps are also generally shorter for newer cohorts. These comparisons are conditional on reaching the relevant publication number and should not be interpreted as unconditional career persistence rates.

[![Publication gaps by entry cohort](outputs/figures/overall/Graph4_ConsecutivePublicationGaps_ByCohort.png)](outputs/figures/overall/Graph4_ConsecutivePublicationGaps_ByCohort.html)

*Figure 4. Mean years between consecutive Top Five publications by cohort of first publication.*

### Finding 5: The observed gap to a second publication has narrowed

The mean observed time from an author's first to second Top Five publication fell from **5.5 years for authors entering in 1980** to **2.5 years for authors entering in 2020**, a decline of approximately **3.0 years**. This result is consistent with faster repeat publication among recent entrants, but right-censoring is important: recent cohorts have had less time to produce a second publication.

[![Gap between first and second publications](outputs/figures/overall/Graph5_FirstToSecondPublicationGap_1980_2020.png)](outputs/figures/overall/Graph5_FirstToSecondPublicationGap_1980_2020.html)

*Figure 5. Mean observed years between authors' first and second Top Five publications, by year of first publication.*

## Interpretation and Limitations

The findings consistently show that prior Top Five experience has become more closely associated with authorship and entry. They do **not**, by themselves, show that reputation causes publication success or that editorial decisions are biased. Several alternative mechanisms could generate the same patterns, including increasing team size, greater specialization, rising research complexity, changes in the submission pool, and shifts in the distribution of research ability.

The analysis also has five important measurement limitations:

1. It observes published papers rather than submissions, rejections, or acceptance probabilities.
2. Authors whose careers began before 1950 may be incorrectly classified as new near the beginning of the sample.
3. Recent cohorts are right-censored, especially in analyses of repeat publication.
4. Name-based author disambiguation may retain false matches or missed matches.
5. Field analyses partly rely on model-predicted JEL categories.

The defensible conclusion is therefore narrower than the title's motivating question: entering the Top Five without a prior Top Five record has become less common, and first-time authors are increasingly connected to experienced Top Five authors through coauthorship.

## Technical Stack

- **Data collection:** Python, Requests, Beautiful Soup, OpenAlex API, and Crossref API
- **Data processing and record linkage:** pandas, NumPy, DOI normalization, exact matching, and fuzzy name matching
- **Machine learning:** scikit-learn, TF-IDF, logistic regression, PyTorch, Hugging Face Transformers, SPECTER2, and SciBERT
- **Visualization:** Matplotlib and custom interactive HTML/CSS/JavaScript charts
- **Testing and version control:** pytest, Git, and GitHub

## Project Structure

```text
data/                         # Raw and processed data (not tracked by Git)
outputs/
  figures/                    # Static PNGs, interactive HTML, and figure data
  tables/                     # Summary tables
src/
  Step0_PreProcessingData/    # Data collection and parsing
  Step1_CleanOpenalexCrossrefData/
  Step2_MergeAllDatasets/
  Step3_TrainingModelClassifyJELCodes/
  Step4_CleanAuthorNames/
  Step5_EnrichAuthorEducation/
tests/                        # Regression and pipeline tests
```

## Author PhD Enrichment

After author IDs are created, the Step 5 pipeline can use the ORCID Public API to enrich the unique author list with PhD institutions and years while retaining source evidence and ambiguous cases for review:

```bash
python src/Step5_EnrichAuthorEducation/01_EnrichAuthorPhD.py --limit 25
```

See [`src/Step5_EnrichAuthorEducation/README.md`](src/Step5_EnrichAuthorEducation/README.md) for API setup, confidence rules, resumable runs, and output definitions.
