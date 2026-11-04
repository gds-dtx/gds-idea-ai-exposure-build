"""Annual salary stated in a police advert's own text, for adverts whose salary fields are
empty or unreadable.

Only a figure the advert states is used: the first annual amount (or range midpoint) within
WINDOW characters after a salary keyword. Amounts next to "allowance", "supplement" etc. and
hourly or daily rates are skipped, and anything outside £12k-£250k is ignored. Where nothing
unambiguous is found the salary stays missing: nothing is estimated or imputed. On adverts
that have a salary field too, this agrees with it within 10% for 95% (542 adverts).
"""

from __future__ import annotations

import re

MONEY = r"£\s?(\d{1,3}(?:[,\s]\s?\d{3})+(?:\.\d+)?|\d{4,6}(?:\.\d+)?|\d{1,3}(?:\.\d+)?\s?k)"
RANGE = re.compile(MONEY + r"(?:\s*(?:-|–|—|to)\s*" + MONEY + r")?", re.I)
NOT_ANNUAL = re.compile(r"^\W{0,3}(?:per|an|a|/)\s*(?:hour|hr|day|week|month)|^\W{0,3}p\.?h", re.I)
ALLOWANCE_BEFORE = re.compile(
    r"(allowance|supplement|enhancement|bonus|payment|uplift)\W{0,25}$", re.I
)


def _num(s: str) -> float:
    s = s.lower().replace(",", "").replace(" ", "")
    return float(s[:-1]) * 1000 if s.endswith("k") else float(s)


KEYWORD = re.compile(r"salary|pay\s*(?:scale|band|range)|grade\s*range|starting\s*pay", re.I)
WINDOW = 80


def salary_from_text(text: str, floor=12_000, cap=250_000) -> float | None:
    """Midpoint of the first annual salary (range) stated within WINDOW characters after a
    salary keyword. Amounts next to 'allowance', 'supplement' etc. and hourly/daily rates are
    skipped. None if nothing unambiguous is found: nothing is estimated."""
    if not isinstance(text, str):
        return None
    for k in KEYWORD.finditer(text):
        window = text[k.end() : k.end() + WINDOW]
        for r in RANGE.finditer(window):
            before = window[max(0, r.start() - 40) : r.start()]
            after = window[r.end() : r.end() + 20]
            if ALLOWANCE_BEFORE.search(before) or NOT_ANNUAL.search(after):
                continue
            lo = _num(r.group(1))
            hi = _num(r.group(2)) if r.group(2) else lo
            if floor <= lo <= hi <= cap:
                return (lo + hi) / 2
    return None
