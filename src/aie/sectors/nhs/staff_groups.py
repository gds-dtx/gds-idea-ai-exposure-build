"""First-pass mapping of advert ``job_title`` -> HCHS staff group (2026 release).

Copied from gds-idea-nhs-ai-exposure-cleaning (src/nhs_ai_exposure/staff_groups.py,
commit 95d9bc4), so the build no longer depends on that repo.
The rules are unchanged; comments quoting sample figures were reworded (public repo).

This is deliberately a transparent, keyword-first mapper, not a black box — the whole point
of this pass is to *see how far simple rules get us* and which titles fall through, so we can
judge how realistic it is to rake on staff group.

Structure of the classification:
  1. ``pay_scheme_type`` gives us two groups for free:
       HOSPITAL_MEDICAL_DENTAL -> "HCHS doctors"
       VERY_SENIOR_MANAGER     -> "NHS infrastructure support" (senior managers)
  2. Everything else is Agenda for Change, classified by ordered keyword rules on the title.
     Order matters: more specific / higher-priority groups are tested first (midwife before
     nurse, registered professions before their support roles).

The target groups match the NHS Digital HCHS "staff group" breakdown so the result joins
straight onto the 2026 workforce control totals.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

# HCHS main staff groups (the rake targets).
DOCTORS = "HCHS doctors"
NURSES = "Nurses & health visitors"
MIDWIVES = "Midwives"
AMBULANCE = "Ambulance staff"
STT = "Scientific, therapeutic & technical"
SUPPORT = "Support to clinical staff"
INFRA = "NHS infrastructure support"
UNCLASSIFIED = "Unclassified"
OUTSIDE = "Outside HCHS (excluded)"  # cluster 4 — real roles but not in HCHS workforce totals

# ---------------------------------------------------------------------------------------
# PAY-SCHEME OVERRIDE & POPULATION CAVEATS — FOR REVIEW.
#   - HOSPITAL_MEDICAL_DENTAL -> HCHS doctors: verified clean on 2026 (all consultants/
#     specialty doctors/fellows/locum consultants). Minor contaminant: "salaried GP" (a few
#     ads) is primary care, not HCHS. A doctor advertised under AfC would be missed (rare).
#   - VERY_SENIOR_MANAGER -> Infrastructure: few ads, all chief/director roles. A few are
#     clinically-qualified leaders (CNO/CMO/chief nurse) but HCHS counts senior managers as
#     infrastructure regardless, so this is consistent. NB infrastructure is NOT only from
#     VSM — most of it comes from AfC admin/manager titles via TITLE_RULES below.
#   - POPULATION LEAK: the pay_scheme filter drops adverts on OTHER (care homes/
#     private/primary care), but a small residual non-HCHS still passes because they use AfC
#     pay — mainly charitable HOSPICES + salaried GPs. Cleanable later with an employer/ODS
#     trust-list filter (pairs with the ICB step). Flagged, not yet applied.
#
# First-pass staff-group shares (advert FLOW, pre-rake) were checked against HCHS. Watch: STT
# looks high & Support looks low in the adverts — same assistant/technician
# boundary that Cluster 2 (below) will shift once wired in.
# ---------------------------------------------------------------------------------------

# UNCLASSIFIED TAIL — clusters 2-4 now WIRED IN below; cluster 1 (practitioners) OUTSTANDING.
# Clusters 2-4 + the hospice filter shrink the unclassified share; most of the remainder is the
# cluster-1 practitioner roles.
#
#   Cluster 1 — "... practitioner" roles  (OUTSTANDING — needs a DECISION not a keyword):
#     mental health practitioner (+ senior/community/primary-care/CAMHS/education variants),
#     advanced clinical practitioner, assistant/associate/community practitioner.
#     -> Ambiguous at the HCHS boundary: could be Nurses (RMN-derived) OR Scientific,
#        therapeutic & technical (psychology/AHP). NEEDS A SPLIT RULE. Candidate default:
#        "mental health practitioner" -> Nurses; "advanced clinical practitioner" -> follow
#        pay band / leave for manual review.
#
#   Cluster 2 — generic clinical support / technician roles:
#     medical laboratory assistant, sterile services / decontamination technician,
#     radiology / imaging assistant, dietetic assistant, rehabilitation assistant,
#     ophthalmic technician, newborn hearing screener, donor carer.
#     -> COULD BE CHUCKED INTO: Support to clinical staff (assistants) / STT (technicians).
#
#   Cluster 3 — generic managers / leads / non-clinical:
#     team leader, clinical lead, clinical team leader, management accountant,
#     switchboard operator, personal assistant, hotel services / maintenance assistant.
#     -> COULD BE CHUCKED INTO: NHS infrastructure support.
#
#   Cluster 4 — genuinely OUTSIDE HCHS (flag, do NOT force into a group):
#     social worker (social care), ips employment specialist.
#     -> Recommend leaving as Unclassified / excluding, since HCHS totals won't contain them.
# ---------------------------------------------------------------------------------------

# Ordered (group, regex) rules applied to a lower-cased job title. First match wins.
# Kept readable and easy to extend as we eyeball the unclassified tail.
TITLE_RULES: list[tuple[str, str]] = [
    # Cluster 4 — genuinely outside HCHS. Tested first so they never get pulled into a group.
    (OUTSIDE, r"social worker|employment specialist|\bips employment\b"),
    (MIDWIVES, r"midwif|midwife|labour ward (?:co-?ordinator|manager|sister|lead)"),
    (
        AMBULANCE,
        r"paramedic|ambulance|emergency care assistant|emergency medical technician|\beca\b|\bemt\b",  # noqa: E501
    ),
    # Support-to-clinical must come BEFORE nurses/STT so "nursing assistant", "therapy
    # assistant", "healthcare assistant" don't get pulled into the registered groups.
    (
        SUPPORT,
        r"health\s?care assistant|\bhca\b|nursing assistant|care assistant|support worker|clinical support|therapy assistant|nursery nurse|phlebotom|ward clerk|healthcare support|"  # noqa: E501
        # Cluster 2 — clinical support / technician roles (assistants to the registered
        # professions, plus sterile/decontamination technical support).
        r"laboratory assistant|radiology assistant|imaging assistant|radiology department assistant|dietetic assistant|rehabilitation assistant|pharmacy assistant|sterile services|decontamination|newborn hearing|hearing screener|donor carer|maternity support|healthcare science assistant|ophthalmic technician",  # noqa: E501
    ),
    (
        NURSES,
        r"\bnurse|nursing|health visitor|\brgn\b|\brmn\b|\brn\b|matron|\bsister\b|charge nurse|\bhcsw\b|district nurse|practice nurse|theatre practitioner|"  # noqa: E501
        # Registered clinical leaders that would otherwise be pulled into Infrastructure by
        # the broad 'manager'/'team lead' keywords. Ward managers are senior nurses; clinical
        # team/site/ward/service managers are clinical leadership (predominantly nursing).
        # Bare 'clinical lead' is DELIBERATELY not here - it stays Infra (too ambiguous:
        # could be medical or AHP). Placed before the Infra rule so these win.
        r"ward manager|clinical team (?:lead|leader|manager)|clinical (?:site|ward|service) manager|\bclinical manager\b",  # noqa: E501
    ),
    (
        STT,
        r"physiotherap|occupational therap|\bradiograph|pharmac|dietit|dietic|podiat|chiropod|orthopt|orthotist|prosthet|speech|\bslt\b|psycholog|psychotherap|biomedical scientist|clinical scientist|healthcare scientist|operating department|\bodp\b|sonograph|audiolog|optomet|art therap|music therap|drama therap|perfusion|physiologist|\btherapist\b|dental|\bscientist\b|technologist",  # noqa: E501
    ),
    (
        INFRA,
        r"administrat|\bclerk|receptionist|secretar|\bmanager\b|finance|\bhr\b|human resources|\bit\b|informatics|\bict\b|porter|estates|facilities|cleaner|domestic|catering|chef|hostess|housekeep|payroll|\banalyst\b|\bofficer\b|coordinator|co-ordinator|clerical|data|procurement|governance|director|business|project|communications|workforce|librarian|driver|security|\badmin\b|"  # noqa: E501
        # Cluster 3 — generic managers / leads / non-clinical (only reach here if no clinical
        # profession keyword matched first, e.g. bare "clinical lead" / "team leader").
        r"team lead|clinical lead|switchboard|personal assistant|hotel services|maintenance|accountant|steward|\bstores\b|warehouse",  # noqa: E501
    ),
    # ---------------------------------------------------------------------------------------
    # HIGH-CONFIDENCE TAIL RECOVERY (added after eyeballing the unclassified pool over the
    # full 2021-2026 sample). Appended AFTER the rules above so it only ever fires on titles
    # that fell through to Unclassified; it cannot change an existing classification. Within
    # this block Support/STT precede Infra so patient-facing support roles are not pulled into
    # the (broad) Infra keywords. Recovers part of the unclassified pool.
    #
    # DELIBERATELY LEFT UNCLASSIFIED (decision: title alone cannot distinguish Nurse vs
    # AHP/STT, so we do not guess and risk biasing the staff-group margins). Mainly: mental
    # health practitioner, advanced/urgent clinical practitioner, clinical/practice educator,
    # community practitioner, theatre/anaesthetic/scrub/recovery practitioner, physician
    # associate. These are candidates for a future LLM-over-advert-text pass.
    (
        SUPPORT,
        r"assistant practitioner|associate practitioner|technical instructor|therapy instructor|recovery worker|support[, ]?time[, ]?recovery|patient services assistant|\bcarer\b|domiciliary|care navigator|pathway navigator|discharge (?:facilitator|coordinator)|research assistant|clinical trials assistant|play specialist|medicines management assistant|materials management assistant",  # noqa: E501
    ),
    (
        STT,
        r"mammograph|cardiograph|medical photograph|clinical photograph|counsellor|counselling|psychological wellbeing|\bpwp\b|screening practitioner|pharmacy technician|medicines management technician|dental technician",  # noqa: E501
    ),
    (
        INFRA,
        r"medical pa\b|medical secretary|audio typist|recruitment (?:assistant|advisor|officer)|\bbuyer\b|employee relations|materials management|house ?keep|food service|cleaning|chaplain|telephonist|call handler|fire safety|business support|programme manager|project manager|project support",  # noqa: E501
    ),
    # PASS 2 — deeper keyword-anchored recovery (CNS, clinical technicians/engineers, coders,
    # digital/estates, admin). Same append-only guarantee; raises coverage further.
    # After this the residual is genuinely ambiguous / niche (clinical research practitioner,
    # safeguarding practitioner, crisis clinician, clinical specialist, ...) that title alone
    # cannot resolve -> LLM-over-advert-text territory.
    (NURSES, r"\bcns\b|clinical nurse specialist|nurse specialist|advanced nurse"),
    (
        SUPPORT,
        r"therapy technician|rehabilitation technician|therapy support|physiology assistant|respiratory assistant|cardiac (?:rehab\w* )?support|ophthalmic assistant|dietetic support|nutrition assistant|cancer (?:pathway )?(?:assistant|support)|pathway assistant|ward host|healthcare support worker|clinical support worker|assistant technical officer|generic worker|theatre assistant|maternity assistant",  # noqa: E501
    ),
    (
        STT,
        r"endoscopy technician|cardiac (?:physiolog|technician)|cardiology technician|neurophysiolog|respiratory physiolog|vascular (?:scientist|technician)|renal technician|healthcare science|clinical technologist|(?:medical|clinical|biomedical) engineer|\bebme\b|non.?medical prescriber|medicines optimisation",  # noqa: E501
    ),
    (
        INFRA,
        r"clinical cod|\bcoder\b|\bcook\b|executive assistant|\bdeveloper\b|\bsoftware\b|desktop|network engineer|systems (?:analyst|administrat)|financial|accounts (?:assistant|payable|receivable)|legal (?:assistant|secretary)|organisation\w* development|\bportfolio\b|programme (?:manager|lead|officer|support)|business (?:partner|intelligence|manager)|data (?:analyst|manager|officer|scientist)|quality improvement|\bqi\b|information (?:governance|analyst|officer)|\bengineer\b|craftsperson|catering assistant|domestic assistant|\bpa\b|typist|library assistant|logistics|cleanliness|health and safety|clinical audit|freedom to speak up",  # noqa: E501
    ),
]

_COMPILED = [(group, re.compile(pattern, re.IGNORECASE)) for group, pattern in TITLE_RULES]


def classify_titles(titles: pd.Series) -> pd.Series:
    """Vectorised keyword classification of a Series of job titles (AfC roles)."""
    t = titles.fillna("").astype(str)
    conditions = [t.str.contains(rx) for _, rx in _COMPILED]
    choices = [group for group, _ in _COMPILED]
    return pd.Series(np.select(conditions, choices, default=UNCLASSIFIED), index=titles.index)


def assign_staff_group(df: pd.DataFrame) -> pd.Series:
    """Full staff-group assignment using pay scheme + title rules."""
    out = classify_titles(df["job_title"])
    # pay_scheme_type overrides: these two groups are unambiguous from the scheme alone.
    out = out.mask(df["pay_scheme_type"].eq("HOSPITAL_MEDICAL_DENTAL"), DOCTORS)
    out = out.mask(df["pay_scheme_type"].eq("VERY_SENIOR_MANAGER"), INFRA)
    return out
