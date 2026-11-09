# Source: ONS Quarterly Public Sector Employment Survey (QPSES), local authority tables

The raking target for **local government** (`sectors/local_govt/qpses.py`).

| | |
|---|---|
| Publisher | Local Government Association (LGA) Research and Information Team, from ONS QPSES returns |
| Page | https://www.local.gov.uk/publications/ons-quarterly-public-sector-employment-survey |
| File used | QPSES 2026 Q2, xlsx: https://www.local.gov.uk/sites/default/files/documents/QPSES%202026%20Q2.xlsx |
| Summary | QPSES 2026 Q2, PDF: https://www.local.gov.uk/sites/default/files/documents/QPSES%202026%20Q2%20-%20summary.pdf |
| Published | 16 September 2026 (latest available when fetched on 28 October 2026) |
| Reference period | Quarter 2 2026 |
| Coverage | 344 principal authorities in England and Wales (English unitaries, London boroughs, metropolitan districts, shire counties, shire districts, Welsh unitaries). No Scottish or Northern Irish councils, no town or parish councils, no fire, police or combined authorities |
| Measure used | `Total FTE` for England (900,020), matching full-time salaries and the NHS and Education targets. Headcount is kept alongside |
| Totals | Headcount 1,285,100 England and Wales (1,150,200 England only); FTE 1,006,700 |
| Copy on S3 | `build/sector=local-govt/release=v1/reference/qpses_2026_q2.xlsx` (+ `_summary.pdf`) |

## Sheets

- **Final Individual** (used): one row per authority. Columns are LA number, LGA Name, Type,
  Region, Total Headcount, Total FTE, sex x full/part-time splits, permanent vs
  temporary/casual, and Notes. The Notes column flags `Imputation` and
  `Non-response - imputation` where ONS imputed an authority's return.
- **Final summary**: totals by authority type and region, rounded to the nearest 100. They may
  not equal the sum of authorities.

## Things to know

- **Schools.** The file does not state its scope. Large county and metropolitan figures
  (Hampshire 35,000, Leeds 25,000, Kent 22,000) only make sense if staff in
  council-maintained schools are included, so academies (not council employers) are out. On
  that reading, local government's council-advertised school posts (`is_school_role`) belong
  in the target and stay in the rake. They may overlap with Education's DfE School Workforce
  target, which is still to be discussed.
- **Reorganised areas.** QPSES 2026 Q2 still reports Cumbria under the county and its six
  former districts. Together they total 7,863, which matches Cumberland plus Westmorland and
  Furness. Adverts from either new council are raked to that combined unit. Adverts from
  former North Yorkshire and Somerset districts go to the new unitaries.
- **Zero returns.** Babergh, Wandsworth, West Oxfordshire and Forest of Dean report 0,
  because a partner employs their staff (Mid Suffolk, Richmond, Publica). The first two are
  pooled with their partner. A zero council's adverts are pooled with similar councils.
- **Joint employers.** Adverts such as "Bromsgrove and Redditch Councils" pool those councils
  into one unit when joint adverts are at least 10% of the councils' adverts, or a member
  reports 0. Otherwise the few joint adverts are left unweighted.

## How it is used (local government v1)

`rake_council` (the only spec) is a 1D post-stratification to QPSES **FTE**, for **England**:

- **FTE, not headcount.** Salaries are full-time annual, and NHS and Education rake to FTE
  too. Many council staff are part-time, so weighting to headcount would overstate paybill.
  Headcount is kept in the diagnostics.
- **England only**, like NHS and Education. Our few hundred Welsh adverts are mostly senior
  posts on national boards, too few and too senior to stand for Welsh councils. Welsh
  adverts stay in the data, unweighted.
- **Cells.** A council (unit) with at least 30 eligible adverts is its own cell. Every other council is pooled by region x authority type, or by region if that
  pool has fewer than 30 adverts. That includes councils with no adverts at all. If a
  pool has no adverts (Bradford), every council of that region and type joins it. So a
  council's few adverts never stand for the whole council.
- **Councils with no adverts (v2).** If a council's pool has no adverts at all because its
  neighbours of the same type all have their own cells, its FTE is shared out over those
  neighbours in proportion to their FTE, and they keep their own cells. v1 pooled every
  council of that region and type instead, so a few councils with many adverts stood for the
  whole group.
- **Result.** All 900,020 England FTE (the QPSES total) are represented. The number of
  adverts weighted, the design effect and the largest weight are in the diagnostics.

**Known limitation.** Council cells fix *where* the workforce is, not *what it does*.
Adverts skew to professional and senior posts, so the weighted mean salary is well above
local government's actual average. Paybill and £ figures are therefore
likely overstated. An occupation or pay margin (APS occupation mix, or ASHE pay) is the next
step, and QPSES has neither.

Full diagnostics (coverage, pooled cells, unmatched councils, unit list) are in
`build/sector=local-govt/release=v1/weights/rake_diagnostics.json`. The cell targets used are
in `targets.parquet`.

## Updating to a newer quarter

1. Add the new xlsx and summary URLs to `sectors/local_govt/config.toml` under
   `[reference_sources]`, and change `reference_file`, `target_source` and `target_date`.
2. Rebuild the current release in place (a new release only if the team asks for one):
   `aie build local-govt reference|clean|weight --release v1`, then
   `aie publish local-govt --release v1`.
3. Check `rake_diagnostics.json` → `coverage.unmatched_councils_top` for new name variants,
   and add any to `ALIASES` in `qpses.py`.

## Releases

There is one release, **v1**, updated in place:

- 2026-10-28: council cells, with small councils pooled by region x type.
- 2026-11-04: a council with no adverts is shared out over its region x type neighbours in
  proportion to their FTE (it was first built as a separate v2; folded into v1 on 2026-11-09).
- 2026-11-09: each target is shared over the adverts that **state a salary**, so £ totals scale
  up from them. The earlier basis is kept as the `rake_council_all_adverts` weight.
