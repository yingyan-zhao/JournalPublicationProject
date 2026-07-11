
# Tech Stack Overview:

Core Python & ML:
API & Deployment: FastAPI, Pydantic, Uvicorn, Docker, pytest


# Has it become harder to break into economics’ leading journals without an established publication record?

Has it become harder for researchers without an established publication record to break into economics’ leading journals? This project examines how entry into the journals commonly known as the “Top Five” has changed over time.^1

A publication in one of these journals is widely viewed as an important marker of research success in economics. The standards are demanding, and publication outcomes are likely to reflect genuine differences in research quality, ability, experience, and persistence. Established scholars may produce stronger work because they have accumulated knowledge, learned how to identify promising questions, developed more effective research processes, and gained access to valuable collaborators and feedback.

Prior success, however, may also create advantages that extend beyond research ability alone. An established reputation can make a paper more visible, facilitate collaboration with other prominent researchers, and provide access to professional networks and institutional resources. When evaluating uncertain or highly specialized work, editors and referees may also treat an author’s previous record as a signal of quality. These mechanisms need not involve deliberate favoritism, and they are difficult to distinguish from the effects of experience and skill. Nevertheless, they raise an important question: has access to top journals become increasingly associated with prior standing in the profession?

To investigate this question, I construct long-run publication histories for authors in the Top Five journals. I examine several dimensions of entry and persistence: the share of papers written by established authors, the proportion of authors publishing in the Top Five for the first time, whether new entrants publish alone or with experienced coauthors, and the time between an author’s first and subsequent Top Five publications. Together, these measures describe how the composition of authorship and the pathways into top-journal publishing have evolved.

The results point to a gradual shift toward a system in which prior experience and established connections are increasingly associated with publication in the Top Five. Established authors appear on a growing share of papers, first-time authors account for a smaller share of contributors, and new entrants increasingly publish alongside experienced coauthors rather than alone or exclusively with other newcomers. These patterns do not, by themselves, establish that reputation causes publication success or that editorial decisions are biased. They are also consistent with changes in collaboration, specialization, research complexity, and the distribution of scholarly ability. The narrower conclusion is that breaking into the Top Five without an established publication record has become less common, while prior standing within the profession has become more closely connected to entry.

This project therefore provides descriptive evidence on the changing structure of elite economics publishing. Rather than treating reputation and research quality as competing explanations, it asks how experience, collaboration, and professional standing have become intertwined—and what those changes may mean for researchers attempting to enter the field’s most selective publication venues.

⸻
^1 Following the conventional definition used in economics, the “Top Five” journals are the American Economic Review, Econometrica, Journal of Political Economy, Quarterly Journal of Economics, and Review of Economic Studies.

# Data Sources and Dataset Construction

I construct an article-level dataset covering publications in the five journals commonly referred to as the Top Five in economics from 1950 through 2026. Core bibliographic records are obtained from OpenAlex and Crossref. I enrich these records with abstracts, keywords, and Journal of Economic Literature (JEL) classification codes from RePEc, the AEA Journals API, and NBER. When structured metadata are unavailable, I retrieve missing information directly from journal and article webpages through targeted web scraping.

## Record Linkage and Author Disambiguation

Records are linked across data sources primarily using Digital Object Identifiers (DOIs). When a DOI is unavailable, I use standardized article titles as a fallback matching criterion. I then remove duplicate records and standardize journal names, publication dates, author names, and institutional affiliations.

To construct author-level publication histories, I disambiguate author identities using normalized names and fuzzy matching. This procedure is designed to reconcile differences in spelling, initials, name order, and formatting across data sources.

## Classification of Missing JEL Codes

Approximately XX% of articles do not have reported JEL classification codes. For these articles, I formulate JEL-code assignment as a multi-label text-classification problem, since a single article may be associated with multiple JEL categories.

I compare three multi-label text-classification approaches:
* a traditional machine-learning baseline using TF–IDF features with a [logistic-regression/linear-SVM] classifier;
* a SPECTER2 embedding-based classifier; and
* a fine-tuned SciBERT classifier.

The models use available textual information—including article titles, abstracts, and keywords—to predict the relevant JEL categories. Model performance is evaluated on a held-out sample of articles with observed JEL codes. In the final dataset, reported and model-predicted JEL codes are identified separately.

## Final Dataset

After record linkage, deduplication, author disambiguation, and metadata enrichment, the final dataset contains XXXX publications and XXX unique authors between 1950 and 2026. It provides a harmonized longitudinal record of publishing in the Top Five and includes:

* Publication metadata: article title, journal, publication year, and DOI;
* Research metadata: abstract, keywords, and reported or predicted JEL classification codes; and
* Author metadata: author names and institutional affiliations, where available.

Because 2026 is not yet a complete publication year, observations from 2026 are treated as provisional. Metadata coverage, particularly for abstracts, JEL codes, and institutional affiliations, also varies across journals and over time.

# Research Question

Identify the change in the pattern of economics publications: Has it become harder to break into economics’ leading journals without an established publication record?

# Main Findings
# A growing share of papers include top-ranked authors (Author rankings based on publication count over the preceding 20 years)
<img src="outputs/Graph1_TopAuthorPaperShares_After1980.html">

# New authors account for a smaller share of authors over time.
# New authors publish their first top-five increasingly with experienced coauthors, Share of new authors by coauthor composition in their first top-five publication, 1980–2025
# Observed publication gaps are shorter for newer cohorts: Average years between consecutive top-five publications, by cohort of first publication
# The observed gap to a second top-five publication has narrowed


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

## Author PhD Enrichment

After author IDs are created, the Step 5 pipeline can use the ORCID Public API
to enrich the unique author list with PhD institutions and years while
retaining source evidence and ambiguous cases for review:

```bash
python src/Step5_EnrichAuthorEducation/01_EnrichAuthorPhD.py --limit 25
```

See
[`src/Step5_EnrichAuthorEducation/README.md`](src/Step5_EnrichAuthorEducation/README.md)
for API setup, confidence rules, resuming a run, and output definitions.
