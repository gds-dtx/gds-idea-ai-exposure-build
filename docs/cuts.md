# Cuts (`aie.cuts`)

`aie.cuts` turns the curated layer into tables: totals for any grouping and filters, or one row
per job advert. Dashboard and analysis code both call it, so their figures always agree.

From the command line: `aie cut --options`, or `aie cut --spec cut.json [--out file.csv]`.

## Calling it from an app

```python
from aie.cuts import options, values, run_cut, CutSpec

opts = options()            # once at startup: sectors, releases, dimensions, measures, models
values("grade", ["nhs"])    # when a filter dropdown opens
res = run_cut(CutSpec.model_validate(form_state))   # on "Preview" / "Download"
dcc.send_bytes(res.to_csv(), "cut.csv")             # plus res.meta as a .json alongside
```

`CutSpec` fields map one-to-one onto controls: sectors (multi-select), release, group_by
(up to about 3), filters, measures, threshold basis, weight spec (CS: 1D or 3D), and model
(locked to gpt-oss-120b when more than one sector is ticked). Show `res.meta["warnings"]`
above the table. They cover drafts, advert-count sectors and blanked small cells.

## One row per vacancy

`level="vacancy"` lists the adverts themselves instead of grouping them: one row per advert
that passes the filters, with the `columns` asked for (`options()["vacancy_columns"]`, plus a
one-sector cut's own columns). Figures are the advert's own (its weight as `fte`, mean
exposure, quadrant and per-FTE £), so nothing is blanked. `group_by`, `measures` and
`min_cell_adverts` are ignored.

```python
CutSpec(sectors=["civil-service"], level="vacancy",
        filters=[{"column": "occupation_group", "values": ["Communications and Marketing"]}])
```

## Custom quadrant limits

Everything else is fixed at the published methodology. The one thing a cut may change is
where the quadrant lines are drawn: two sliders, from
`options()["custom_thresholds"]` (5-50%, default 25 / 25):

```python
CutSpec(..., threshold_basis="custom",
        custom_thresholds={"mean_top_pct": 30, "var_pct": 30})
```

- **mean_top_pct**: the "high" group is the top X% of roles by mean exposure.
- **var_pct**: within that group, the bottom X% by variance are Displacement-type;
  within the low group, the top X% by variance are Concentrated.

The percentiles are read off the same distribution as the published fixed thresholds (CS's
ensemble roles, primary weights), rounded to 4 dp, and applied to every sector. So 25 / 25
reproduces the headline exactly. Nothing is stored. The engine reclassifies each advert from
`metrics.mean_exposure` / `var_exposure` with the shared methods, which takes about 2 s.
`res.meta["custom_thresholds"]` records the percentiles and the values they became, and a
warning says the result is a sensitivity, not the published figure. Show both on the page.

## Small cells

- The engine loads only the partitions a cut touches and caches them in the process.
- `min_cell_adverts` (default 10) blanks small groups in grouped cuts. Advert-level cuts
  (`level="vacancy"`) are not blanked, since each row is one advert's own figures.
