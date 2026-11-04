# Source: Home Office, Police workforce, England and Wales (open data table)

The raking target for **police** (`sectors/police/workforce.py`).

| | |
|---|---|
| Publisher | Home Office (accredited official statistics) |
| Page | https://www.gov.uk/government/statistics/police-workforce-open-data-tables |
| File used | Workforce open data table, as at 31 March 2007 to 2026 (ODS): https://assets.publishing.service.gov.uk/media/6a58d48c7eb057103bd945ba/open-data-table-police-workforce-220726.ods |
| Release | https://www.gov.uk/government/statistics/police-workforce-england-and-wales-31-march-2026 |
| Published | 22 July 2026 (latest when fetched on 29 October 2026); next update July 2027 |
| Reference date | 31 March 2026 (the latest year in the table; earlier years are ignored) |
| Coverage | The 43 territorial forces of England and Wales plus British Transport Police (BTP has no region, `[z]`, treated as "National"). Not Police Scotland, PSNI, the Civil Nuclear Constabulary or MoD Police |
| Measure used | `Total (FTE)`. Headcount is kept alongside |
| Totals (31 March 2026, incl. BTP) | 239,055 FTE: officers 148,702, police staff 77,002, PCSOs 7,452, designated officers 5,899; special constables and volunteers 0 FTE |
| Copy on S3 | `build/sector=police/release=v1/reference/police_workforce_open_data_2026.ods` |

The table's `Data` sheet has one row per year x force x sex x rank x worker type. `[z]` marks
not-applicable cells (read as 0).

## How it is used (police v1)

`rake_force_staff` (the only spec) is a 1D post-stratification to **FTE** of police staff
(including designated officers) and PCSOs:

- **Officers are not weighted.** They are most of police FTE, but officer recruitment runs
  through national entry routes, so force job boards carry only 366 officer adverts. Those
  are mostly senior or specialist posts (deputy and assistant chief constables, regional
  organised crime units), and few give a salary, so they cannot stand for the officer
  workforce. Officer adverts stay in the data with no weight.
- **Specials and volunteers are not weighted** (no FTE). Volunteer and cadet-leader adverts
  classified as staff are dropped for the same reason.
- **Cells.** Police staff: a force with at least 30 eligible staff adverts is its own cell (15
  forces); the rest are pooled by region, then nationally if the region is still thin. PCSOs:
  one national cell. Surrey, Sussex and the "Surrey & Sussex Police
  Collaboration" adverts are one unit, as are Norfolk, Suffolk and their joint adverts.
- **Result.** The staff and PCSO adverts are weighted to 90,353 FTE, the Home Office total
  for police staff and PCSOs. Every target cell has adverts. Coverage, design effect and the
  largest weight are in the diagnostics file below.

**Known limitations.**
- The weighted population is staff and PCSOs only, so police totals are **not** the whole
  police workforce. Say so wherever police figures are quoted.
- **Salaries are only ones the adverts state.** Most adverts give a salary in their salary
  fields; some forces put it only in the advert text, which `salary_text.py` reads
  (`salary_source = advert_text`). Adverts that state no salary anywhere are left out of the
  £ figures; nothing is estimated. The coverage is reported in the diagnostics.
- A few forces supply most of the adverts. Their own cells contain them, but smaller forces'
  figures rest on pooled regional cells.
- Worker type comes from the job title, so some staff/officer classification will be wrong.

Full diagnostics are in `build/sector=police/release=v1/weights/rake_diagnostics.json`.

## Updating to a newer year

1. Add the new ODS URL under `[reference_sources]` in `sectors/police/config.toml` and change
   `reference_file`, `target_source` and `target_date`.
2. Build a new release so v1 stays reproducible: `aie build police reference|weight --release v2`,
   then `aie publish police --release v2`.
3. Check `coverage.not_raked_forces_top` in the diagnostics for new force name variants and
   add any to `ALIASES` in `workforce.py`.
