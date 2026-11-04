# Raking local government and police: options and status

**Status (2026-10-29).**
- **Local government is raked** to council FTE (England) from the ONS Quarterly Public Sector
  Employment Survey, using the LGA's per-authority tables for 2026 Q2. See
  [qpses.md](qpses.md) for the source, scope and matching. The only spec, `rake_council`,
  represents all 900,020 England council FTE. Occupation is still not a margin. The options below for adding an occupation mix
  (APS or Census) still apply.
- **Police is raked (2026-10-29)** to Home Office police workforce FTE by force, for police
  staff and PCSOs only; officer adverts are too few and too senior to weight. See
  [police_workforce.md](police_workforce.md).

The pattern that worked for NHS and Education is to rake 1D (IPF) to the official workforce
statistics for that sector, on the dimensions the adverts can also produce. The question for
each sector below is which statistics, and which advert dimensions line up with them.

## Police: a clear route

**Target:** Home Office, *Police workforce, England and Wales* (published twice a year, with
open data tables). It gives FTE by force and by worker type (police officers, police staff,
PCSOs, designated officers, special constables), and by function. Police Scotland and PSNI
publish their own equivalents.

**What the adverts already give us (v1):**
- `org_unit` = canonical force (normalised in the old cleaning repo), and `region` from the
  force.
- `occupation_group` = worker type, classified from the job title (Police officer / PCSO /
  Special constable / Police staff). This is approximate and needs a check on a sample.

**Proposed rake:** 1D on worker type × force (as two margins) for England and Wales forces,
with targets taken from the Home Office open data table for the latest March or September.

**Caveats to agree first:**
- Officer recruitment mostly runs through national campaigns (PCDA, DHEP), so officer adverts
  are thin, and the recruitment-route pages are already excluded from scope. Officers may
  need to be dropped from the weighted population and the rake restricted to staff and
  PCSOs, or officer weights capped.
- The sample is small (about 3,900 eligible adverts across roughly 45 forces), so a
  force-level margin may produce extreme weights. Region × worker type is the fallback.
- National bodies (BTP, MDP, CNC, NCA) are not in the Home Office territorial tables.

**Effort:** small. Download one table into `build/sector=police/reference/`, write about 40 lines of
target mapping, add a title-classification check.

## Local government: no clean target yet

There is no official local government workforce table broken down by occupation in the way
that HCHS (NHS) or the School Workforce census (Education) is.

| Option | What it gives | Problem |
|---|---|---|
| ONS Public Sector Employment (QPSES) | Local government headcount and FTE totals, by country | Scale only, no structure. Can set the total but not the mix. |
| ONS APS/LFS microdata (UK Data Service) | Occupation (SOC 2020) × region for people working in local government (public-sector flag + industry) | Needs UKDS access, and adverts need SOC codes. These could come from an LLM pass, which would be a batch cost gate, or from ONS's occupation coding tool. |
| LGA Local Government Workforce Survey | Council workforce by service area | Partial coverage, not official statistics, published irregularly |
| Census 2021 custom table | Occupation × industry (SIC 84.11 local government) × LA | 2021 vintage, and SIC 84.11 mixes in some central administration |

**Suggested route:** set the total from QPSES and the occupation × region mix from APS (or
Census 2021 as a cheaper first pass). That needs two things the adverts do not have yet:
1. `region`: a council → ONS region lookup. This is cheap, using the ONS LAD lookup and
   fuzzy-matching about 2,900 council name strings.
2. `occupation_group`: SOC major group (or sub-major) per advert. This is the expensive part.

**Decide first:** the school-type adverts. Some local government adverts
have school-type titles. These are council-advertised posts in council-maintained schools.
The staff are probably already counted in DfE's School Workforce figures that Education rakes
to. They are kept in local government with `is_school_role` / `school_role_type` in
curated `vacancies`. Before raking local government, decide whether they are excluded from the
local government weighted population (likely, to avoid counting them twice against
Education) or kept and raked to a school-staff margin.
