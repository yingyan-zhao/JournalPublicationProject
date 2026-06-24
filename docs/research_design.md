# Research Design Notes

## Research Question

The central question is whether publications in top economics journals are concentrated among a relatively small group of highly visible authors. In other words, do a few "big names" account for a large share of articles in the top five economics journals, or is publication credit broadly distributed across the profession?

The project will describe the extent of concentration, how it changes over time, whether concentration differs across journals, and whether concentration varies across fields in economics.

## Main Outcome

The main outcome is author-level publication credit in top economics journals.

A secondary outcome is institution-level publication credit. This sideline analysis asks whether top-journal publications are also concentrated among a small number of universities or research institutions.

Baseline credit:

```text
credit = 1 / number_of_authors_on_article
```

Robustness:

```text
credit = 1
```

## Key Design Choices

### Journal Definition

The project focuses on the conventional top five journals in economics:

- American Economic Review (AER)
- Journal of Political Economy (JPE)
- Econometrica
- Review of Economic Studies (ReStud)
- Quarterly Journal of Economics (QJE)

Keep the journal list explicit in any paper or presentation, because concentration estimates can move when field journals are included.

### Initial Data Source

The first version of the dataset will use OpenAlex as the backbone source. OpenAlex provides structured metadata for articles, authors, institutions, publication dates, journals, DOIs, abstracts when available, and citation counts.

After OpenAlex, the next enrichment source will be Crossref, matched primarily by DOI. Crossref can help verify publisher-deposited metadata, publication dates, abstracts when available, references, reference counts, funding information, and ORCID metadata.

OpenAlex and Crossref will be supplemented with publisher webpages, PDFs, NBER, CEPR, SSRN, RePEc/IDEAS, IZA, and author websites for fields that these APIs do not consistently provide, such as JEL codes, acknowledgments, submission dates, accepted dates, and first online working-paper dates.

### Time Windows

Useful cuts:

- Annual concentration
- Rolling five-year windows
- Decade cohorts
- Pre/post major data coverage changes

### Field Differences

The project will also examine whether publication concentration differs across fields in economics, such as microeconomics, macroeconomics, econometrics, labor, development, public finance, industrial organization, international economics, and economic history.

Field classification will follow the Journal of Economic Literature (JEL) classification system. Each article should be assigned to one or more JEL codes when available.

The data should also include the article abstract and author- or journal-provided keywords. These text fields can help audit JEL-based field assignments, identify ambiguous cases, and support later descriptive work on research topics.

OpenAlex topic metadata will also be collected when available. OpenAlex maps works into a hierarchy of domain, field, subfield, and topic. These topic fields are useful for auditing and exploratory analysis, but the main field classification for this project will follow JEL codes.

For analysis, narrow JEL codes can be grouped into broader economics fields. For example, JEL D can be treated as microeconomics, JEL E as macroeconomics and monetary economics, JEL C as econometrics and quantitative methods, JEL J as labor and demographic economics, JEL O as development and growth, and JEL H as public economics.

Field-level concentration should be interpreted carefully because some fields are larger than others, and some fields may publish more often in general-interest top journals than others.

### Institution Concentration

As a secondary result, the project will examine concentration across institutions. The goal is to describe whether top-five economics journal publications are concentrated among a small set of universities, business schools, central banks, research institutes, or policy institutions.

The preferred institution measure is the author's institutional affiliation at the time of publication. If multiple affiliations are listed, the analysis should define whether credit is assigned to all listed institutions or only to the primary affiliation.

Institution-level publication credit can be calculated using the same logic as author-level credit:

```text
institution_credit = publication_credit assigned to authors affiliated with the institution
```

Institution names should be standardized before analysis, since the same institution may appear under multiple names.

### Submission and Acknowledgment Information

When available, the dataset should capture the first submission date, acceptance date, online publication date, and issue publication date. The first submission date may come from the published article, article webpage, publisher metadata, or manual extraction from the PDF, so the source of the date should be recorded.

For each published article, the project will also search online for the first working-paper version of the paper. This captures when the research first became publicly available, which can differ substantially from the first journal submission date and the final publication date.

Relevant sources for the first online working-paper version include:

- NBER working papers
- CEPR discussion papers
- SSRN
- RePEc and IDEAS
- IZA discussion papers
- university working-paper series
- author personal websites
- archived versions found through persistent URLs or web archives

The dataset should record the first working-paper date, the source, the URL, and the repository or working-paper series. If multiple versions exist, the preferred measure is the earliest verifiable online circulation date. The matching process should use title, authors, abstracts, DOI links, and version notes to avoid linking the published article to an unrelated paper with a similar title.

The dataset should also capture the article's acknowledgments or "thanks" section. This field can include named individuals thanked by the authors, seminar and conference participants, editors, referees, funding sources, research assistants, and institutional support.

Acknowledgment text is article-level information, so it will usually be repeated across author rows in an author-publication dataset. If the thanks section is extracted from a PDF or article webpage, the extraction source should be recorded.

### Citation Information

The project will collect citation information for each paper as an additional measure of scholarly impact. Citation counts should be treated as time-varying metadata rather than fixed article characteristics.

The dataset should record the citation count, citation source, citation collection date, and a URL or identifier for the citation record. Possible sources include Google Scholar, OpenAlex, Crossref, Semantic Scholar, Scopus, Web of Science, RePEc, and publisher webpages.

Citation measures may differ across sources because databases vary in coverage, working-paper matching, citation deduplication, and update frequency. Analyses using citation counts should therefore report the source and collection date.

### Author Disambiguation

Author identity is the main measurement challenge. Name-only matching is not sufficient for final analysis. Prefer stable author identifiers from a bibliographic database, ORCID, OpenAlex, RePEc, Scopus, Web of Science, or a manually audited crosswalk.

### Interpretation

High concentration can reflect several mechanisms:

- Star-author productivity
- Coauthorship networks
- Editorial and referee networks
- Institutional concentration
- Field composition
- Data coverage and author disambiguation artifacts

The analysis should separate descriptive concentration from causal claims.
