"""Map `primary_job_role` (+ `job_title` fallback) to DfE's School Workforce in England
taxonomy, mirroring NHS's `staff_groups.py` keyword-mapping approach.

CRITICAL FINDING (2026-10-07): `Job_Type` ("Teaching Role" / "Non Teaching Role") does NOT
mean what its name implies — it is a near-perfect proxy for which of two data-entry
vocabularies a row's `primary_job_role` uses, not what the job actually is:

    primary_job_role is one of the structured snake_case/group-label values
    (teacher, headteacher, deputy_headteacher, teaching_assistant, sendco, "Other Support", ...)
        -> Job_Type == "Non Teaching Role" for almost all of them, INCLUDING ACTUAL
           TEACHERS AND HEADTEACHERS.

    primary_job_role is a free-text value (Class Teacher, Science, Cover Teacher, Cleaner,
    Exam Invigilator, ...)
        -> Job_Type == "Teaching Role" for all of them, INCLUDING CLEANERS
           AND INVIGILATORS.

So `Job_Type`/`job_type_clean` CANNOT be used to pick Teacher-characteristics vs
Support-staff-characteristics for raking (the original plan). This module ignores it entirely
and classifies teaching-vs-non-teaching directly from role content, in the same pass as the
DfE Grade/Post assignment — the only way this can work correctly on this data.

Target categories (DfE "School Workforce in England", see README "Raking target"):
    Teacher grades:  Head teacher | Deputy head teacher | Assistant head teacher |
                      Classroom teacher
    Support posts:   Administrative staff | Auxiliary staff | Leadership - Non Teacher |
                      Other school support staff | School business professionals |
                      Teaching assistants | Technicians

Several buckets are genuinely ambiguous from title/role text alone and are resolved with an
explicit, documented default rather than silently guessed:
  - Heads of department/year/phase, SENCOs, and generic "Leadership Role" labels are typically
    teaching staff holding a TLR/responsibility allowance in English schools, and DfE's own
    4-grade teacher taxonomy has no separate "middle leader" bucket — these default to
    Classroom teacher.
  - "other_leadership" (the structured-vocabulary label, distinct from free-text "Leadership
    Role") is treated as non-teaching SLT (-> Leadership - Non Teacher) since it sits in the
    Non-Teaching-vocabulary rows alongside Administration/Catering/IT Support, which DO appear
    to be genuinely non-teaching in that vocabulary's own internal consistency.

Copied from gds-idea-education-ai-exposure-cleaning (src/education_ai_exposure/role_groups.py,
commit 6d49b8f) so the build no longer depends on that repo.
The rules are unchanged; comments quoting sample figures were reworded (public repo).
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

HEAD = "Head teacher"
DEPUTY_HEAD = "Deputy head teacher"
ASSISTANT_HEAD = "Assistant head teacher"
CLASSROOM_TEACHER = "Classroom teacher"
ADMIN = "Administrative staff"
AUXILIARY = "Auxiliary staff"
LEADERSHIP_NON_TEACHER = "Leadership - Non Teacher"
OTHER_SUPPORT = "Other school support staff"
BUSINESS_PROF = "School business professionals"
TEACHING_ASSISTANT = "Teaching assistants"
TECHNICIAN = "Technicians"
UNCLASSIFIED = "Unclassified"

TEACHER_GRADES = {HEAD, DEPUTY_HEAD, ASSISTANT_HEAD, CLASSROOM_TEACHER}
SUPPORT_POSTS = {
    ADMIN,
    AUXILIARY,
    LEADERSHIP_NON_TEACHER,
    OTHER_SUPPORT,
    BUSINESS_PROF,
    TEACHING_ASSISTANT,
    TECHNICIAN,
}

# Ordered (category, regex) rules applied to lower-cased `primary_job_role` + `job_title`
# (concatenated). First match wins. Teacher grades are tested first since "head of
# department"-type phrases could otherwise be caught by support-staff "head"/"manager" rules.
_RULES: list[tuple[str, str]] = [
    # --- Teacher grades (also catches "Principal" for UTC/studio-school-style titling) ---
    (
        HEAD,
        r"\bheadteacher\b|\bhead_teacher\b|\bhead of school\b|\bprincipal\b(?!.*vice|.*assistant|.*deputy)",  # noqa: E501
    ),
    (DEPUTY_HEAD, r"\bdeputy_?head\b|\bdeputy head\b|\bdeputy principal\b|\bvice[ -]principal\b"),
    (
        ASSISTANT_HEAD,
        r"\bassistant_?head\b|\bassistant head\b|\bassistant principal\b|\bvice head\b",
    ),
    # --- Support posts (tested before the Classroom-teacher catch-all) ---
    (
        TEACHING_ASSISTANT,
        r"\bteaching assistant\b|\bteaching_assistant\b|\bhlta\b|\bhigher_level_teaching_assistant\b|"  # noqa: E501
        r"\bclassroom assistant\b|\blearning support\b|\b1[ :-]?2[ :-]?1\b.*assistant|\bsen teaching assistant\b",  # noqa: E501
    ),
    (
        AUXILIARY,
        r"catering,? cleaning and site management|\bcatering\b|\bcleaner\b|\bcleaning\b|\bcaretaker\b|"  # noqa: E501
        r"\bsite (?:manager|assistant|officer)\b|\bpremises\b|\bcook\b|\bkitchen\b|\bminibus driver\b|"  # noqa: E501
        r"\bmidday\b|\blunchtime\b|\bmealtime\b|\bbreakfast club\b|\bafter school club\b|\bplayworker\b",  # noqa: E501
    ),
    (
        TECHNICIAN,
        r"\bit support\b|\bit technician\b|\bnetwork (?:manager|engineer|technician)\b|"
        r"\bscience technician\b|\b(?:d&t|dt) technician\b|\bsystems (?:administrator|technician)\b",  # noqa: E501
    ),
    (
        BUSINESS_PROF,
        r"\bbusiness manager\b|\bbursar\b|\bfinance manager\b|\boperations manager\b|"
        r"\bschool business\b",
    ),
    (
        ADMIN,
        r"administration,? hr,? data,? and finance|\badmin(?:istrator|istration|istrative)?\b|\bclerk\b|"  # noqa: E501
        r"\breceptionist\b|\bsecretary\b|\bdata (?:officer|manager)\b|\bpayroll\b|\bhr\b|"
        r"\bexams? officer\b|\bexaminations? officer\b|\boffice (?:manager|administrator)\b|"
        r"\bfinance (?:assistant|officer)\b",
    ),
    (
        LEADERSHIP_NON_TEACHER,
        r"\bother_leadership\b|\bdirector of (?:finance|operations|business)\b",
    ),
    (
        OTHER_SUPPORT,
        r"\bother support\b|\bpatoral health and welfare\b|\beducation_support\b|"
        r"\bother_teaching_support\b|\blearning mentor\b|\bpastoral (?:manager|assistant)\b|"
        r"\bsafeguarding officer\b|\blibrarian\b|\bexam invigilators?\b|\bapprentice\b|"
        r"\bhealth care\b|\bspecial needs\b",
    ),
    # --- Classroom teacher: broad catch-all for anything teaching-shaped, tested after the
    #     support-staff rules so e.g. "teaching assistant" never falls through to here. ---
    (
        CLASSROOM_TEACHER,
        r"\bteacher\b|\bteaching\b|\bclass teacher\b|\bsubject\b|\bcover teacher\b|\bsupport teacher\b|"  # noqa: E501
        r"\bsenco\b|\bsendco\b|\bhead_of_department_or_curriculum\b|\bhead_of_year_or_phase\b|"
        r"\bleadership role\b|\b(?:ks1|ks2|ks3)\b|\bscience\b|\benglish\b|\bmathematics\b|\bmaths\b|"  # noqa: E501
        r"\bgeography\b|\bhistory\b|\bforeign language\b|\bphysical education\b|\bart\b|\bmusic\b|"
        r"\bdrama\b|\bdance\b|\bdesign technology\b|\breligious studies\b|\bearly years teacher\b|"
        r"\breception teacher\b|\bprimary teacher\b|\bsecondary teacher\b|\bnursery teacher\b",
    ),
    # --- TAIL RECOVERY (appended after eyeballing the Unclassified pool on the full cleaned
    #     pool). Append-only: fires only on rows the rules above missed, so it can't change an
    #     existing classification. Shrinks the Unclassified share. ---
    (
        CLASSROOM_TEACHER,
        r"\bhumanities\b|\beconomics\b|\bict\b(?!.*technician)|\bpsychology\b|\bsociology\b",
    ),
    (
        TECHNICIAN,
        r"\bict technician\b|\btechnology technician\b|\b(?:ict|it)\s+technician\b|\btechnician\b",
    ),
    (
        AUXILIARY,
        r"\bwrap(?:\s|-)?around care\b|\bmdsa\b|\bfacilities (?:manager|assistant|officer)\b|"
        r"\bplay\s?worker\b",
    ),
    (ADMIN, r"\bclerical\b|\bdata\b"),
    (
        LEADERSHIP_NON_TEACHER,
        r"\bchief executive\b|\bceo\b|\bdirector\b",
    ),
    (
        OTHER_SUPPORT,
        r"\bschool counsellor\b|\bfamily liaison\b|\bcareers advisor\b|\blearning (?:assistant|coach)\b|"  # noqa: E501
        r"\binclusion manager\b|\beducation welfare\b|\binvigilator\b",
    ),
]
_COMPILED = [(cat, re.compile(pat, re.IGNORECASE)) for cat, pat in _RULES]


def classify_role(primary_job_role: pd.Series, job_title: pd.Series) -> pd.Series:
    """Vectorised keyword classification into the DfE Grade/Post taxonomy."""
    combined = primary_job_role.fillna("").astype(str) + " " + job_title.fillna("").astype(str)
    conditions = [combined.str.contains(rx) for _, rx in _COMPILED]
    choices = [cat for cat, _ in _COMPILED]
    return pd.Series(np.select(conditions, choices, default=UNCLASSIFIED), index=combined.index)


def assign_dfe_category(df: pd.DataFrame) -> pd.DataFrame:
    """Add `dfe_category` (one of TEACHER_GRADES | SUPPORT_POSTS | Unclassified) and the
    derived `is_teaching` boolean (replaces the broken `job_type_clean` for raking purposes)."""
    out = df.copy()
    out["dfe_category"] = classify_role(out["primary_job_role"], out["job_title"])
    out["is_teaching"] = out["dfe_category"].isin(TEACHER_GRADES)
    return out
