# Higher Entry Barriers to Break into Economics Top-Five Journal Publication

In this project, I study the long-term trend of economics publications in top-five journals. Publishing in Top-five journals are very hard in economics. The bar is very high. However, I noticed that publications by established researchers are way more easier than new comers. This could be established researchers with higher ability or through learning by doing, are more capable to produce high quality researches. However, other factors play big roles as well, for example networking, reputations of big names. These factors may bias referee and editors' opinions and making established researchers are more easier to publish. The entry barriers for new commers are higher. For new commers, their researches are under a stricter scrutize than established ones.

In this project, I study the trend of publications in top-five journals over a long time and showing that the entry barriers for new commers are getting higher and higher. Reputations play an increasing role in this publishing games.

A footnote Top-Five Journals in economics are:

- American Economic Review
- Econometrica
- Journal of Political Economy
- Quarterly Journal of Economics
- Review of Economic Studies

## Data Collection

- Publications in Top-Five Journals (1950-2026) are collected through OpenAlex and CrossRef API
- I supplement the information in Keywords, Abstract, Jel Codes by the dataset: Repec API, AEA Journals API and NBER, webscraping each paper when these information are missing.

After integrating all the information from different data sources and cleaning the data, the final dataset includes XXXX total publications from Top-Five Journals in economics, XXX unique authors, covering the period from 1950 to 2026. This dataset provides a comprehensive view of the publications in top journals in economics, including:
- Publication information: publication year, journal
- Authors information: names, institutions

## Research Question

Identify the change in the pattern of economics publications. 

## Tech Stack Overview:

Core Python & ML: 
API & Deployment: FastAPI, Pydantic, Uvicorn, Docker, pytest

## Main Findings
# A growing share of papers include top-ranked authors (Author rankings based on publication count over the preceding 20 years)
<img src="path_to_graph.png">
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
