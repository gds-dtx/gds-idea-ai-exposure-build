"""Public corporations: adverts from the careers sites of ten public bodies outside the other
sectors (BBC, Bank of England, British Steel, Channel 4, Civil Aviation Authority, National
Audit Office, National Highways, Network Rail, Ordnance Survey, Transport for London).

Most are public corporations in the ONS sense. A few are classified elsewhere (the NAO is a
parliamentary body; Network Rail and National Highways are classified to central government,
TfL to local government) but none is in the Civil Service, NHS, schools, council or police
samples, so they sit here rather than being lost.

Mapping onto the contract:
  employer           the body
  org_unit           the raking cell: the body, except the BBC, which is split into
                     "BBC public service" and "BBC commercial" (BBC Studios, BBC Global News and
                     other commercial subsidiaries) from the legal entity on the advert
  occupation_group   the job category the careers site gives, where it gives one
  region             not set (locations are free text; kept as `location` in the sector columns)
  salary             advert range midpoint, annual £12k-£250k only; where the salary fields are
                     empty, the salary the advert TEXT states; never estimated

Cleaning (first reason that applies; excluded adverts are kept with exclusion_reason and are
not sent for scoring):
  1. the page says the advert has closed or the post is filled (no description left)
  2. no job description: under MIN_BODY_CHARS characters once the careers-site header line is
     removed (Network Rail listing shells, Bank of England "internal only" stubs)
  3. an unfilled careers-site template ("Enter a brief description of the vacancy…"), which
     Network Rail published for some posts
  4. based outside the UK (BBC World Service bureaux and BBC Studios overseas offices, Network
     Rail Consulting), or BBC Media Action (a charity, not part of the BBC Group)
  5. volunteering or work-experience placements
  6. an exact repeat of an earlier advert from the same body (same title and description);
     the latest posting is kept

Weights: post-stratified to each cell's published staff number (targets.py).
"""

from __future__ import annotations

import io
import re
from pathlib import Path

import pandas as pd

from ... import settings, storage
from ..base import (
    CleanOutput,
    ScoreOutput,
    Sector,
    WeightOutput,
    advert_midpoint_salary,
    year_of,
)
from ..police.salary_text import salary_from_text
from . import targets

MIN_BODY_CHARS = 1_000
# Careers-site template text left in unfinished adverts (Network Rail).
TEMPLATE_PLACEHOLDERS = (
    "Enter a brief description of the vacancy",
    "Enter Department description here",
)
CLOSED_STUBS = ("this advert has now closed", "sorry, this position has been filled")
EMPLOYER_CODE = {
    "BBC": "bbc",
    "Bank of England": "boe",
    "British Steel": "bsl",
    "Channel 4": "c4",
    "Civil Aviation Authority": "caa",
    "National Audit Office": "nao",
    "National Highways": "nh",
    "Network Rail": "nr",
    "Ordnance Survey": "os",
    "Transport for London": "tfl",
}
# UK plus the Crown dependencies, whose BBC local stations are BBC public service posts.
UK_CODES = {"GB", "GBR", "UK", "JE", "GG", "IM"}
# Places outside the UK that appear in BBC job URLs and careers-site location fields.
OVERSEAS_PLACES = (
    "Abuja", "Accra", "Amman", "Amsterdam", "Bangkok", "Beijing", "Beirut", "Bishkek",
    "Brisbane", "Brussels", "Cairo", "Colombo", "Dakar", "Damascus", "Dar es Salaam", "Dar-es",
    "Delhi", "Dhaka", "Dubai", "Hong Kong", "Hong-Kong", "Islamabad", "Istanbul", "Jakarta",
    "Jerusalem", "Johannesburg", "Juba", "Kabul", "Kampala", "Kiev", "Kinshasa", "Kyiv",
    "Lagos", "Los Angeles", "Los-Angeles", "Melbourne", "Mexico City", "Miami", "Moscow",
    "Mumbai", "Nairobi", "New York", "New-York", "New Delhi", "New-Delhi", "Paris",
    "Phnom Penh", "Riga", "San Francisco", "Seoul", "Singapore", "Sydney", "Taipei",
    "Tashkent", "Tbilisi", "Tokyo", "Toronto", "Warsaw", "Washington", "Yangon",
)  # fmt: skip
_OVERSEAS_RE = re.compile(r"\b(?:" + "|".join(re.escape(p) for p in OVERSEAS_PLACES) + r")\b")
# Countries named in BBC job titles for posts abroad ("Senior Journalist - Australia").
OVERSEAS_COUNTRIES = (
    "Afghanistan", "Australia", "Bangladesh", "Egypt", "Ethiopia", "Ghana", "India", "Indonesia",
    "Israel", "Jordan", "Kenya", "Kyrgyzstan", "Lebanon", "Myanmar", "Nigeria", "Pakistan",
    "Senegal", "Singapore", "Somalia", "South Africa", "Sri Lanka", "Thailand", "Turkey",
    "Uganda", "Ukraine", "USA", "United States", "Uzbekistan",
)  # fmt: skip
_TITLE_ABROAD = re.compile(
    r"\b(?:" + "|".join(re.escape(p) for p in OVERSEAS_PLACES + OVERSEAS_COUNTRIES) + r")\b"
)
# BBC header: "<place>, [<area>, ]<CC>[, <postcode>] <legal entity> Job Closing Date"
_BBC_ENTITY = re.compile(r"([^\n]{0,80}?)\s*Job Closing Date")
_CC_COMMA = re.compile(r",\s([A-Z]{2,3})(?=,)")
_CC_BARE = re.compile(
    r"(?:Time|Term|Schemes|Contract|Apprenticeship)\s([A-Z]{2})\s+(?:BBC|PS -|PUBLIC|Media Action)"
)
_BBC_LOCATION = re.compile(r"(?i)\blocation\s?:\s?([^\n|;]{2,80})")
_UK_PLACE = re.compile(r"London|Salford|Glasgow|Cardiff|Belfast|Birmingham|Bristol|\bUK\b", re.I)
_PLACE_BEFORE_CC = re.compile(r"([A-Z][A-Za-z .'\-]{1,30}),\s(?:[A-Za-z .\-]{1,30},\s)?[A-Z]{2,3},")
_COMMERCIAL = re.compile(r"Studios|BBCS\b|Global News|Worldwide", re.I)
_HEADER_FIELD = re.compile(r"^\s*[A-Z][A-Za-z ]{1,30}: ")


def split_header(text: pd.Series) -> tuple[pd.Series, pd.Series]:
    """(header, body). Most careers sites put a 'Location: … | Hours: …' line first."""
    parts = text.fillna("").str.split("\n", n=1)
    first = parts.str[0]
    has = first.str.match(_HEADER_FIELD) & (first.str.len() < 400)
    header = first.where(has, "")
    body = text.fillna("").where(~has, parts.str[1].fillna(""))
    return header, body


def header_field(header: pd.Series, name: str) -> pd.Series:
    """The value of `name:` in the header line (up to the next field or '|')."""
    nxt = (
        r"(?:Geographical Area|Job Category|Category|Function|Asset|Contract Type|Hours|"
        r"Schedule|Organisation|Location)"
    )
    pat = rf"(?<![A-Za-z]){name}: (.+?)(?:\s+{nxt}:|\s*\||$)"
    return header.str.extract(pat)[0].str.strip().replace("", pd.NA)


def bbc_entity(text: pd.Series) -> pd.Series:
    """The legal entity before 'Job Closing Date' in a BBC advert (e.g. BBC PUBLIC SERVICE
    B'CAST, BBC Studios Production, BBC PS - Kenya), or NA."""
    return text.str[:600].str.extract(_BBC_ENTITY)[0].str.strip()


def bbc_countries(text: pd.Series) -> pd.Series:
    """Country codes in a BBC advert header (first 700 characters), as a set per advert:
    'London, GB, W1A 1AA BBC …' -> {GB}; '… Full Time NG BBC PS - Nigeria' -> {NG}."""
    head = text.str[:700]
    return pd.Series(
        [set(_CC_COMMA.findall(h)) | set(_CC_BARE.findall(h)) for h in head], index=text.index
    )


def bbc_cell(entity: pd.Series, text: pd.Series) -> pd.Series:
    """'BBC commercial' for BBC Studios / Global News / other commercial companies, else
    'BBC public service'. Uses the legal entity where the advert gives one, else the header."""
    basis = entity.fillna(text.str[:600])
    return pd.Series(
        ["BBC commercial" if _COMMERCIAL.search(str(b)) else "BBC public service" for b in basis],
        index=text.index,
        dtype="string",
    )


def is_overseas(df: pd.DataFrame, header: pd.Series, entity: pd.Series) -> pd.Series:
    """Posts based outside the UK.

    BBC: a non-UK country code in the advert header, a 'BBC PS - <country>' entity, an overseas
    place before the country code (some Dubai posts carry GB), an overseas place or country in
    the job title, or, for adverts with no header, an overseas city at the start of the job URL.
    Other bodies: an overseas place in the careers-site location field, or Network Rail
    Consulting ('NRC ')."""
    text = df["jd_text"].fillna("").astype(str)
    bbc = df["employer"].eq("BBC")
    codes = bbc_countries(text)
    has_code = codes.map(bool)
    # Abroad only if no UK code at all: "Salford, MAN, GBR" (county code) and multi-site
    # "London, GBR | Nairobi, KEN" adverts stay in.
    foreign_code = codes.map(lambda c: bool(c) and not (c & UK_CODES))
    ps_abroad = entity.fillna("").str.contains(r"PS - (?!UK)", regex=True)
    place = text.str[:700].str.extract(_PLACE_BEFORE_CC)[0].fillna("")
    place_abroad = place.str.contains(_OVERSEAS_RE)
    title_abroad = df["job_title"].fillna("").str.contains(_TITLE_ABROAD)
    url_city = df["url"].fillna("").str.extract(r"/job/([^/]+)/")[0].fillna("")
    url_abroad = url_city.str.match(_OVERSEAS_RE.pattern)
    # Newer adverts state "LOCATION: Nairobi" in the job details instead of a country code.
    stated = text.str[:2000].str.extract(_BBC_LOCATION)[0].fillna("")
    stated_abroad = stated.str.contains(_OVERSEAS_RE) & ~stated.str.contains(_UK_PLACE)
    bbc_abroad = (
        foreign_code
        | ps_abroad
        | place_abroad
        | title_abroad
        | (~has_code & (url_abroad | stated_abroad))
    )
    location = header_field(header, "Location").fillna("")
    other_abroad = location.str.contains(_OVERSEAS_RE) | location.str.contains(r"\bNRC\b")
    return (bbc & bbc_abroad) | (~bbc & other_abroad)


def clean_adverts(raw: pd.DataFrame, today: pd.Timestamp | None = None) -> pd.DataFrame:
    """The cleaning rules on the raw sheet: one row per raw advert with org_unit, location,
    occupation_group, body text and exclusion_reason. Pure function (tested)."""
    today = today or pd.Timestamp.today().normalize()
    d = raw.copy().reset_index(drop=True)
    text = d["jd_text"].fillna("").astype(str)
    header, body = split_header(text)
    entity = bbc_entity(text).where(d["employer"].eq("BBC"))
    d["body"] = body.str.strip()
    d["location"] = header_field(header, "Location")
    category = header_field(header, "Job Category").fillna(header_field(header, "Category"))
    d["occupation_group"] = category
    d["legal_entity"] = entity
    d["org_unit"] = (
        d["employer"].astype("string").mask(d["employer"].eq("BBC"), bbc_cell(entity, text))
    )
    date = pd.to_datetime(d["date"], errors="coerce")
    d["date_raw"] = d["date"].astype("string")
    d["posted_date"] = date.where(date <= today + pd.Timedelta(days=366))

    stub = text.str.strip().str.lower().str.rstrip(".").isin([s for s in CLOSED_STUBS])
    shell = d["body"].str.len() < MIN_BODY_CHARS
    abroad = is_overseas(d, header, entity)
    media_action = entity.fillna("").str.contains("Media Action")
    template = d["body"].str.contains(
        "|".join(re.escape(t) for t in TEMPLATE_PLACEHOLDERS), case=False, regex=True
    )
    unpaid = d["hours"].fillna("").str.strip().str.lower().isin(["volunteering", "work experience"])
    key = (
        d["employer"].astype(str)
        + "|"
        + d["job_title"].fillna("").str.strip().str.lower()
        + "|"
        + d["body"].str.lower().str.replace(r"\s+", " ", regex=True)
    )
    order = d["posted_date"].sort_values(ascending=False, kind="stable", na_position="last")
    dup = pd.Series(False, index=d.index)
    dup.loc[order.index] = key.loc[order.index].duplicated()

    reason = pd.Series(pd.NA, index=d.index, dtype="string")
    rules = [
        (stub, "advert closed: no job description left"),
        (shell, "no job description (listing page only)"),
        (template, "unfilled advert template (placeholder text, no duties)"),
        (abroad, "based outside the UK"),
        (media_action, "BBC Media Action (a charity, not the BBC Group)"),
        (unpaid, "volunteering or work experience"),
        (dup, "repeat of a later advert (same title and description)"),
    ]
    for mask, why in rules:
        reason = reason.mask(reason.isna() & mask.fillna(False), why)
    d["exclusion_reason"] = reason
    d["is_overseas"] = abroad
    return d


class PublicCorporations(Sector):
    slug = "public-corporations"
    package_dir = Path(__file__).parent

    def _raw_path(self, release: str) -> str:
        name = Path(self.config["raw"]["files"][0]).name
        return storage.join(settings.raw_dir(self.slug, release), name)

    def clean(self, release: str) -> CleanOutput:
        src = self._raw_path(release)
        raw = pd.read_excel(io.BytesIO(storage.read_bytes(src)), dtype={"id": str})
        d = clean_adverts(raw)
        code = d["employer"].map(EMPLOYER_CODE)
        if code.isna().any():
            unknown = sorted(d.loc[code.isna(), "employer"].unique())
            raise ValueError(f"unknown employers: {unknown}")
        vid = self.ids(code + "-" + d["id"].astype(str))
        if vid.duplicated().any():
            raise ValueError("duplicate vacancy ids")

        salary = advert_midpoint_salary(d["salary_min"], d["salary_max"])
        from_text = d["body"].map(salary_from_text).astype("float64").where(salary.isna())
        source = pd.Series("advert_midpoint", index=d.index, dtype="string").where(salary.notna())
        source = source.mask(from_text.notna(), "advert_text")
        salary = salary.fillna(from_text)

        in_scope = d["exclusion_reason"].isna()
        adverts = pd.DataFrame(
            {
                "vacancy_id": vid,
                "source_vacancy_id": d["id"],
                "source": d["source"],
                "posted_date": d["posted_date"],
                "year": year_of(d["posted_date"]),
                "job_title": d["job_title"],
                "employer": d["employer"],
                "org_unit": d["org_unit"],
                "occupation_group": d["occupation_group"],
                "grade": pd.NA,
                "region": pd.NA,
                "salary_annual_gbp": salary,
                "salary_source": source,
                "headcount": 1,
                "in_scope": in_scope,
                "exclusion_reason": d["exclusion_reason"],
            }
        )
        ext = pd.DataFrame(
            {
                "vacancy_id": vid,
                "url": d["url"],
                "location": d["location"],
                "legal_entity": d["legal_entity"],
                "is_overseas": d["is_overseas"],
                "hours": d["hours"],
                "date_raw": d["date_raw"],
                "salary_min_raw": d["salary_min"],
                "salary_max_raw": d["salary_max"],
            }
        )
        text = pd.DataFrame({"vacancy_id": vid, "advert_text": d["body"]})[in_scope]
        return CleanOutput(adverts, ext, text.reset_index(drop=True), funnel(d), inputs=[src])

    def score(self, release: str) -> ScoreOutput:
        """Scored by `aie llm public-corporations extract|label ... collect`, which writes the
        scored layer directly; this re-reads it so `aie build ... score` is a no-op re-write."""
        folder = settings.build_dir(self.slug, "scored", release)
        read = lambda n: storage.read_parquet(storage.join(folder, n))  # noqa: E731
        return ScoreOutput(
            read("tasks.parquet"),
            read("model_scores.parquet"),
            read("task_reasoning.parquet"),
            inputs=[folder],
        )

    def weight(
        self, release: str, adverts: pd.DataFrame, ext: pd.DataFrame, n_tasks: pd.Series
    ) -> WeightOutput:
        return targets.rake_to_employers(self, adverts, n_tasks)


def funnel(d: pd.DataFrame) -> pd.DataFrame:
    rows = [{"step": "raw adverts", "employer": "all", "rows": len(d)}]
    for reason, n in d["exclusion_reason"].value_counts().items():
        rows.append({"step": f"excluded: {reason}", "employer": "all", "rows": int(n)})
    kept = d[d["exclusion_reason"].isna()]
    rows.append({"step": "sent for scoring", "employer": "all", "rows": len(kept)})
    for unit, n in kept["org_unit"].value_counts().sort_index().items():
        rows.append({"step": "sent for scoring", "employer": unit, "rows": int(n)})
    return pd.DataFrame(rows)


SECTOR = PublicCorporations
