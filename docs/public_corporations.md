# Public corporations: sources and raking

The `public-corporations` sector (`src/aie/sectors/public_corporations/`) covers ten public
bodies that are outside the Civil Service, NHS, schools, councils and police samples:

BBC, Bank of England, British Steel, Channel 4, Civil Aviation Authority, National Audit
Office, National Highways, Network Rail, Ordnance Survey and Transport for London.

Most of them are public corporations in the ONS sense. A few are classified elsewhere, so
"public corporations" is a working name, not an ONS category:

- the NAO is a parliamentary body;
- Network Rail and National Highways are classified to central government;
- TfL is classified to local government.

None of these bodies is in another sector's sample.

## Raw adverts

The source is one spreadsheet of adverts scraped from each body's careers site, received in
November 2026. It has 4,825 rows dated 2017 to 2026. The release keeps it unchanged in
`build/sector=public-corporations/release=v1/raw/`.

## Cleaning (L1)

Rows that fail a rule stay in `adverts.parquet` with an `exclusion_reason` and are not sent for
scoring. The rules run in this order, and the first rule that applies sets the reason:

| Rule | What it catches |
|---|---|
| Advert closed | The page text is only "This advert has now closed." or "Sorry, this position has been filled." (all BBC) |
| No job description | Less than 1,000 characters remain once the careers-site header line is removed. This catches Network Rail listing shells and Bank of England "internal only" stubs |
| Unfilled template | The advert still holds the careers-site template text ("Enter a brief description of the vacancy…"); about 200 Network Rail adverts, with no duties to extract |
| Based outside the UK | BBC World Service bureaux, BBC Studios offices abroad and Network Rail Consulting, read from the country code in the advert header, the stated location, the job title or the URL |
| BBC Media Action | A charity, not part of the BBC Group |
| Volunteering or work experience | The careers site's own hours field says so |
| Repeat advert | Same body, same title and same description as a later advert; the latest is kept |

Dates after the scrape (placeholders such as 31/12/2999) are set to missing. The raw value is
kept in `date_raw`.

## Salary

Salary is the midpoint of the advertised range, annual and between £12k and £250k only. When
the range is empty, the salary stated in the advert text is used instead
(`sectors/police/salary_text.py`). Nothing is estimated, so adverts with no stated salary stay
out of the £ figures.

## Raking targets (L3, primary spec `rake_employer_salaried`)

No official table gives staff numbers for each public corporation. ONS Public Sector
Employment publishes only a total for all public corporations, and that total leaves out the
NAO, Network Rail, National Highways and TfL. Each body's own published staff number is
therefore the target.

The method is 1D post-stratification with one cell per body. The BBC has two cells, public
service and commercial, matching its accounts. Each advert's legal entity puts it in one of
them: BBC Studios, BBC Global News and the other commercial companies go to commercial. Every
eligible advert in a cell gets the weight target divided by n. Cells are not pooled across
bodies, because the bodies do unrelated work. Cells with fewer than 30 adverts are flagged in
`rake_diagnostics.json` (`small_cells`), and their figures need that caveat.

**Which adverts share the target.** The primary spec, `rake_employer_salaried`, shares each
organisation's staff number over its adverts that **state a salary**. £ totals therefore scale
up from the salaried adverts to the whole organisation: with 5,000 usable adverts of which
2,000 give a salary, each salaried advert stands for target / 2,000. This assumes adverts
without a salary are like those with one. FTE, exposure and quadrant shares are on the same
salaried adverts. The second spec, `rake_employer`, shares the target over every usable
advert. Adverts without a salary then count in FTE and exposure but add £0, so £ totals cover
only the salaried share of staff (79% overall; Bank of England 27%).

The figures and their details are in `config.toml [targets]`. The PDFs are copied to
`build/.../release=v1/reference/`.

| Cell | Target | Measure | Source |
|---|---:|---|---|
| BBC public service | 17,531 | average FTE, 2024/25 | BBC Group Annual Report and Accounts 2024/25, note D1 |
| BBC commercial | 4,242 | average FTE, 2024/25 | same |
| Bank of England | 5,731 | average headcount, year to Feb 2025 (no FTE published) | Annual Report and Accounts 2025, staff costs note |
| British Steel | 4,568 | direct headcount, 2021 (latest stated) | Oxford Economics for British Steel, *Socioeconomic Footprint* (2026) |
| Channel 4 | 1,284 | average FTE, 2024 | Report and Financial Statements 2024, note 4 |
| Civil Aviation Authority | 1,602 | monthly average employees (Group), 2024/25 | Annual Report and Accounts 2024/25, note 3 |
| National Audit Office | 1,028 | average FTE, 2024-25 | Annual Report and Accounts 2024-25, Figure 15 |
| National Highways | 6,866 | average FTE, 2024-25 | Annual Report and Accounts 2024-25, staff report |
| Network Rail | 41,331 | monthly average FTE (Group), 2024/25 | Annual Report and Accounts 2025, note 6 |
| Ordnance Survey | 1,415 | average monthly FTE (permanent), 2024-25 | Annual Report and Financial Statements 2024-25, note 4 |
| Transport for London | 27,638 | FTE including non-permanent labour, 31 March 2025 | Quarterly Performance Report Q4 2024/25, Headcount |

## Caveats

- **Mixed measures.** The Bank of England and the CAA publish headcount and British Steel's
  figure is a headcount, so those cells are slightly too large in FTE terms. TfL's figure
  includes non-permanent labour.
- **British Steel's figure is from 2021.** It has published no accounts as a public body yet.
- **Small samples.** Ordnance Survey has under 10 adverts. British Steel, Channel 4, the NAO,
  National Highways and TfL each have about 40 to 130. Figures for a single body are only as
  good as its sample, and every advert in a cell carries the same weight, with no occupation
  margin within a body.
- **Overseas BBC staff.** BBC overseas adverts are excluded, but the BBC targets include its
  staff abroad, so UK adverts stand for them as well.
