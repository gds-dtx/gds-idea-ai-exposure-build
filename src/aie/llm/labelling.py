"""Second-pass task labelling (substantive / peripheral / generic), shared by every sector.

Runs as a separate classification pass over already-extracted tasks, so boilerplate and
incidental duties can be filtered at analysis time without touching extraction or scoring.
The system prompt is per sector (config.toml [llm] label_system_prompt). The illustrative
examples in the prompt body are the local-government wording; they are generic enough for
every sector.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

Label = Literal["substantive", "peripheral", "generic"]


def label_tasks(job_title: str, tasks: list[str]) -> str:
    """Prompt to classify a role's extracted tasks as substantive / peripheral / generic."""
    numbered = "\n".join(f"{i}. {t}" for i, t in enumerate(tasks, start=1))
    return f"""
    You are given a JOB TITLE and a NUMBERED LIST OF TASKS extracted from one
    job advert. Classify EACH task into exactly one label, so we can filter out items that are
    not meaningful parts of the role.

    LABELS:
    - substantive: a task the postholder would actually spend meaningful day-to-day time on,
      given what this role primarily is. This INCLUDES administrative, clerical, catering,
      cleaning, premises or other support duties WHEN they are the main thrust of the role
      (e.g. for an Business Support Officer, Refuse Loader, or Facilities Caretaker).
    - peripheral: a real but minor or incidental task that takes up little of THIS postholder's
      time and is not part of the role's main purpose (e.g. a social worker occasionally
      attending a corporate away day, or a planning officer helping tidy a shared meeting
      room). If the activity is effectively what the job IS, it is substantive, not
      peripheral.
    - generic: a generic conduct, compliance, or values statement that is not really a discrete
      task the person "does" on a typical day (e.g. "adhere to the employer's dress code",
      "maintain confidentiality in line with policy", "promote equality and diversity",
      "participate in appraisal/CPD", "comply with safeguarding policy", "demonstrate an
      understanding of X").

    THE DECIDING QUESTION for substantive vs peripheral: given what THIS role primarily is, would
    the postholder spend meaningful day-to-day time on this task? The SAME activity can be
    substantive in one role and peripheral in another. For example "carry out reception and
    administrative duties" is SUBSTANTIVE for a Business Support Officer (it is the job),
    whereas "attend relevant meetings" is PERIPHERAL for a Highways Engineer (incidental to the
    engineering role). Do not label a task peripheral merely because it is administrative or support
    work; only label it peripheral if it is minor relative to this role's main purpose.

    RULES:
    - Classify exactly the tasks listed, no more and no fewer. Echo each task_number so labels
      align to the list. Preserve order.
    - Do NOT merge, split, reorder, add, or drop tasks.
    - If the task list below is EMPTY, call the tool with an empty labels list. Do not invent a
      task to label, and do not respond in prose instead of calling the tool.

    RESPOND ONLY by calling the TaskLabellingOutput tool with a "labels" array (use the exact
    lower-case key "labels"). Do NOT reply in plain text and do NOT add any commentary outside
    the tool call. Produce exactly one object per task; each object must have, in this order:
    1. task_number (int): the number of the task from the list below, echoed exactly.
    2. reasoning (one sentence): why this label, i.e. what makes the task substantive, peripheral,
       or generic for THIS role.
    3. label: exactly one of substantive, peripheral, generic (lower-case).

    JOB TITLE: {job_title}

    <tasks_to_label>
    {numbered}
    </tasks_to_label>
    """


class TaskLabelling(BaseModel):
    """A label + reason for one pre-extracted task, keyed by task_number."""

    task_number: int = Field(..., ge=1, description="1-indexed task number, echoed exactly.")
    reasoning: str = Field(..., description="One sentence justifying the label for this role.")
    label: Label = Field(..., description="substantive | peripheral | generic")


class TaskLabellingOutput(BaseModel):
    """Labels for a fixed, pre-extracted task list for one advert."""

    labels: list[TaskLabelling] = Field(
        ..., description="One entry per task in the supplied list, echoing task_number."
    )

    @model_validator(mode="before")
    @classmethod
    def _accept_capitalised_key(cls, data):
        """gpt-oss sometimes returns the key as 'Labels'; accept that too."""
        if isinstance(data, dict) and "labels" not in data and "Labels" in data:
            data = {**data, "labels": data["Labels"]}
        return data

    @model_validator(mode="after")
    def task_numbers_unique(self) -> TaskLabellingOutput:
        nums = [x.task_number for x in self.labels]
        if len(nums) != len(set(nums)):
            raise ValueError(f"Duplicate task_number in response: {nums}")
        return self
