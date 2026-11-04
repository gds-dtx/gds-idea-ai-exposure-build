"""Extract-and-score prompt and output schema, shared by every sector.

The prompt body, rubric, schema and validators were identical across the CS, NHS, Education,
local-govt and police repos (the NHS/Education version, with rule 5 GRANULARITY). The only
per-sector difference is the system prompt, which now lives in each sector's config.toml
([llm] extract_system_prompt). Single combined extract + score call.
"""

from __future__ import annotations

import contextvars

from pydantic import BaseModel, Field, ValidationInfo, field_validator

AUTOMATION_BUCKETS: tuple[float, ...] = tuple(round(i / 10, 1) for i in range(11))
_BUCKET_STR = ", ".join(f"{b:.1f}" for b in AUTOMATION_BUCKETS)

_coercion_log: contextvars.ContextVar[list[dict] | None] = contextvars.ContextVar(
    "_coercion_log", default=None
)


def task_extract_and_score_ai_automation(job_description: str) -> str:
    """Extract tasks from a job advert and score each for automation potential (single call).

    Shared by every sector; see module docstring. Construct: the fraction of each
    task's SUBSTANTIVE CONTENT that current AI systems could perform, scored in discrete 0.1
    buckets.
    """
    return f"""
    Extract each work task from the job description below and score it
    for its AUTOMATION POTENTIAL from current AI systems, using the
    provided format.

    TASK EXTRACTION RULES:
    1. Return tasks in the order they appear in the job description.
       Order matters: downstream processing preserves the list order.
    2. Exclude tasks that describe this role's own recruitment,
       onboarding, working hours, or working conditions. If the role
       involves recruiting or onboarding others as part of the job,
       retain those tasks.
    3. Extract the individual role's actionable tasks (what the person
       does), not skills (what the person knows) or team goals.
    4. Do NOT extract items that describe vague outcomes or accountability,
       not actions (e.g. "meet KPIs", "deliver on strategic priorities").
       Test: if you cannot answer "what specifically does the
       postholder DO on a typical day?", it is not a task worth
       extracting.
    5. GRANULARITY: extract tasks at the level of a role's KEY
       RESPONSIBILITIES, not every individual sub-action. Group closely
       related activities that serve the same purpose into ONE task rather
       than listing them separately. For example, "carry out reception
       duties", "carry out administrative duties" and "carry out stock
       control duties" should be a single task; several cleaning or
       room-preparation steps should be a single task. Aim for the
       smallest number of tasks that still captures the DISTINCT kinds of
       work in the role - typically about 5 to 12, and only rarely more
       than 15. Do not split one responsibility into many granular items.

    SCORING DEFINITION (automation_potential):
    Your estimate of the FRACTION OF THE TASK'S SUBSTANTIVE CONTENT
    that could be performed by current AI systems (LLMs, multimodal
    models, AI-powered software, transcription tools, retrieval
    tools). The rubric below describes how AI and human involvement
    combine at different points on the scale.

    - 0.0 means no substantive part of the task can be performed by
      current AI systems.
    - 1.0 means the entire substantive content can be performed by
      current AI systems with negligible human involvement.
    - Score in DISCRETE 0.1 intervals. Return only values in
      {{{_BUCKET_STR}}}.

    RUBRIC:

    | Score     | Share automatable | Character of AI involvement           |
    |-----------|-------------------|---------------------------------------|
    | 0.0       | None              | No substantive part is automatable.   |
    |           |                   | Task requires physical presence,      |
    |           |                   | physical action, in-person live       |
    |           |                   | judgement, or capabilities beyond     |
    |           |                   | current AI systems.                   |
    | 0.1-0.2   | Small share       | The human performs the substantive    |
    |           |                   | work. AI can support with background  |
    |           |                   | preparation, information retrieval,   |
    |           |                   | or partial drafting, but the core of  |
    |           |                   | the task remains with the human.      |
    | 0.3-0.4   | Meaningful share  | AI can produce initial drafts,        |
    |           |                   | intermediate outputs, or process      |
    |           |                   | structured inputs. The human still    |
    |           |                   | performs core reasoning, adaptation,  |
    |           |                   | and substantive decision-making.      |
    | 0.5-0.6   | Roughly half      | Shared execution. AI processes and    |
    |           |                   | drafts a substantial share; the human |
    |           |                   | directs, contextualises, and verifies |
    |           |                   | with materially more than spot-check  |
    |           |                   | involvement.                          |
    | 0.7-0.8   | Majority share    | AI produces the substantive output    |
    |           |                   | (applying codified rules, synthesising|
    |           |                   | evidence, or processing structured    |
    |           |                   | information). The human reviews,      |
    |           |                   | adjusts, and handles exceptions.      |
    | 0.9-1.0   | Near-complete     | AI performs the substantive work      |
    |           |                   | end-to-end. Human role is limited to  |
    |           |                   | setup, sampling, spot-checks, or      |
    |           |                   | handling residual edge cases.         |

    Score continuously across the eleven buckets; the bands above are
    for orientation only. Choose the bucket that best matches your
    substantive-component decomposition.

    OUTPUT INSTRUCTIONS:
    For each extracted task, provide in this order:

    1. task (str): the task text as extracted from the job description.
       Reproduce the substantive activity; do not paraphrase heavily.

    2. reasoning (one sentence): before scoring, name the substantive
       component(s) of the task current AI could perform, and the
       substantive component(s) that require a human. This
       decomposition should determine your score, not justify a
       pre-committed number.

    3. automation_potential (float in {{{_BUCKET_STR}}}): based on
       your reasoning, the fraction of the task's substantive content
       AI could perform. Return exactly one of the eleven allowed
       values.

    <job_description>{job_description}</job_description>
    """


class AutomationScoredTask(BaseModel):
    """A task extracted from a job description and scored for automation potential."""

    task: str = Field(
        ...,
        min_length=3,
        description=(
            "The task text as extracted from the job description. "
            "Reproduce the substantive activity described; do not paraphrase heavily."
        ),
    )
    reasoning: str = Field(
        ...,
        description=(
            "One sentence naming the substantive component(s) of the task current AI could "
            "perform, and the component(s) that require a human. This decomposition should "
            "determine the score, not justify a pre-committed number."
        ),
    )
    automation_potential: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description=(
            "Estimated fraction of the task's substantive content that current AI could "
            f"perform. One of the eleven discrete values: {_BUCKET_STR}."
        ),
    )

    @field_validator("automation_potential", mode="before")
    @classmethod
    def snap_to_nearest_bucket(cls, v: float, info: ValidationInfo) -> float:
        """Snap to the nearest 0.1 bucket; log any real coercion if a log is active."""
        raw = float(v)
        snapped = round(raw * 10) / 10
        if abs(snapped - raw) > 1e-9:
            log = _coercion_log.get()
            if log is not None:
                log.append(
                    {
                        "task_number": info.data.get("task_number"),
                        "raw_value": raw,
                        "snapped_value": snapped,
                        "drift": raw - snapped,
                    }
                )
        return snapped


class AutomationScoredTaskOutput(BaseModel):
    """Structured output: extracted-and-scored tasks for one job description.

    Task numbering is applied downstream via enumerate() over the preserved list order.
    """

    tasks: list[AutomationScoredTask] = Field(
        ...,
        description=(
            "One entry per task extracted from the job description, scored for its automation "
            "potential from current AI. Ordered as tasks appear in the job description."
        ),
    )
