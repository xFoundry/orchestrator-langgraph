"""
Clarification tools for asking structured questions in the chat UI.

Emits clarification artifacts that the frontend can render as inline UI.
"""

from __future__ import annotations

import re
import time
import uuid
from typing import Optional

from langchain_core.callbacks.manager import adispatch_custom_event
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field


def _slugify(text: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", text.strip().lower()).strip("_")
    return slug or "option"


class ClarificationQuestionInput(BaseModel):
    """Single clarification question."""

    prompt: str = Field(..., description="Question text to present to the user.")
    options: list[str] = Field(
        ...,
        description="Multiple choice options for the user to pick from.",
        min_length=1,
    )
    multi_select: bool = Field(
        False,
        description="Allow the user to pick multiple options.",
    )
    allow_other: bool = Field(
        True,
        description="Allow the user to type a custom response.",
    )
    required: bool = Field(
        True,
        description="Whether the user must answer this question before proceeding.",
    )
    question_id: Optional[str] = Field(
        None,
        description="Optional stable identifier for the question.",
    )


class ClarificationRequestInput(BaseModel):
    """Input schema for requesting clarifications."""

    title: Optional[str] = Field(
        None,
        description="Short title for the clarification card.",
    )
    description: Optional[str] = Field(
        None,
        description="Optional helper text shown under the title.",
    )
    questions: list[ClarificationQuestionInput] = Field(
        ...,
        description="List of questions to ask the user.",
        min_length=1,
    )


async def request_clarifications_impl(
    title: Optional[str],
    description: Optional[str],
    questions: list[ClarificationQuestionInput],
) -> str:
    """Emit a clarification UI payload for the frontend to render."""
    request_id = f"clarification_{uuid.uuid4().hex[:8]}"
    normalized_questions: list[dict[str, object]] = []

    for index, question in enumerate(questions):
        question_id = question.question_id or f"question_{index + 1}_{uuid.uuid4().hex[:4]}"
        option_ids: set[str] = set()
        options_payload: list[dict[str, str]] = []

        for opt_index, option in enumerate(question.options):
            label = option.strip()
            if not label:
                continue

            base_id = _slugify(label)
            option_id = base_id
            if option_id in option_ids:
                option_id = f"{base_id}_{opt_index + 1}"
            option_ids.add(option_id)

            options_payload.append({"id": option_id, "label": label})

        if not options_payload:
            continue

        normalized_questions.append(
            {
                "id": question_id,
                "prompt": question.prompt,
                "selectionType": "multi" if question.multi_select else "single",
                "allowOther": question.allow_other,
                "required": question.required,
                "options": options_payload,
            }
        )

    payload = {
        "id": request_id,
        "title": title or "Clarifying questions",
        "description": description,
        "questions": normalized_questions,
    }

    if not normalized_questions:
        return "No valid clarification questions were provided."

    await adispatch_custom_event(
        "artifact",
        {
            "type": "artifact",
            "id": request_id,
            "artifact_type": "clarification",
            "title": payload["title"],
            "summary": description,
            "payload": payload,
            "created_at": time.time(),
        },
    )

    return "Clarification questions requested from the user. Wait for their response."


def get_clarification_tools() -> list[StructuredTool]:
    """Return the clarification tools for the tool registry."""
    request_clarifications_tool = StructuredTool(
        name="request_clarifications",
        description=(
            "Ask the user clarifying questions with multiple-choice options. "
            "Use when key details are missing or ambiguous. "
            "Provide 1-3 questions and include options; avoid open-ended prompts."
        ),
        func=lambda *args, **kwargs: None,
        coroutine=request_clarifications_impl,
        args_schema=ClarificationRequestInput,
    )

    return [request_clarifications_tool]
