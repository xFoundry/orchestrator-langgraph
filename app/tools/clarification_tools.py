"""
Clarification tools for asking structured questions in the chat UI.

Uses LangGraph's interrupt() for proper human-in-the-loop workflow.
The frontend renders these as interactive, selectable UI components.
"""

from __future__ import annotations

import re
import uuid
from typing import Optional

from langchain_core.tools import StructuredTool
from langgraph.types import interrupt
from pydantic import BaseModel, Field


def _slugify(text: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", text.strip().lower()).strip("_")
    return slug or "option"


class ClarificationOption(BaseModel):
    """A single option in a clarification question."""

    label: str = Field(..., description="The display label for this option.")
    description: Optional[str] = Field(
        None, description="Optional description shown below the label."
    )
    value: Optional[str] = Field(
        None, description="Value to return if selected. Defaults to label."
    )


class ClarificationQuestionInput(BaseModel):
    """Single clarification question."""

    prompt: str = Field(..., description="Question text to present to the user.")
    options: list[str | ClarificationOption] = Field(
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


def request_clarifications_impl(
    title: Optional[str],
    description: Optional[str],
    questions: list[ClarificationQuestionInput],
) -> str:
    """
    Request clarifications from the user using LangGraph interrupt.
    
    This pauses the agent execution and waits for user input.
    The frontend renders interactive selection UI.
    """
    request_id = f"clarification_{uuid.uuid4().hex[:8]}"
    normalized_questions: list[dict[str, object]] = []

    for index, question in enumerate(questions):
        question_id = (
            question.question_id or f"question_{index + 1}_{uuid.uuid4().hex[:4]}"
        )
        option_ids: set[str] = set()
        options_payload: list[dict[str, str | None]] = []

        for opt_index, option in enumerate(question.options):
            # Handle both string options and ClarificationOption objects
            if isinstance(option, str):
                label = option.strip()
                option_description = None
                value = label
            else:
                label = option.label.strip()
                option_description = option.description
                value = option.value or label

            if not label:
                continue

            base_id = _slugify(label)
            option_id = base_id
            if option_id in option_ids:
                option_id = f"{base_id}_{opt_index + 1}"
            option_ids.add(option_id)

            options_payload.append({
                "id": option_id,
                "label": label,
                "description": option_description,
                "value": value,
            })

        if not options_payload:
            continue

        normalized_questions.append({
            "id": question_id,
            "prompt": question.prompt,
            "selectionType": "multi" if question.multi_select else "single",
            "allowOther": question.allow_other,
            "required": question.required,
            "options": options_payload,
        })

    if not normalized_questions:
        return "No valid clarification questions were provided."

    # Build the interrupt payload for the frontend
    interrupt_payload = {
        "type": "clarification",
        "id": request_id,
        "title": title or "Asking for clarification",
        "description": description,
        "questions": normalized_questions,
        "total_questions": len(normalized_questions),
        "current_question": 1,
    }

    # Build a lookup from question ID to question prompt
    question_prompts = {q["id"]: q["prompt"] for q in normalized_questions}

    # Use LangGraph interrupt to pause execution and wait for user response
    # The frontend will render this as an interactive UI
    user_response = interrupt(interrupt_payload)

    # Format the user's response for the agent with readable question prompts
    if isinstance(user_response, dict):
        answers = []
        for q_id, answer in user_response.items():
            # Get the original question prompt for readability
            question_text = question_prompts.get(q_id, q_id)
            # Truncate long questions for cleaner output
            if len(question_text) > 60:
                question_text = question_text[:57] + "..."
            
            if isinstance(answer, list):
                answer_text = ", ".join(str(a) for a in answer)
            else:
                answer_text = str(answer)
            
            answers.append(f"{question_text}: {answer_text}")
        return f"User responded with: {'; '.join(answers)}"
    elif isinstance(user_response, str):
        return f"User responded with: {user_response}"
    else:
        return f"User responded with: {user_response}"


def get_clarification_tools() -> list[StructuredTool]:
    """Return the clarification tools for the tool registry."""
    request_clarifications_tool = StructuredTool(
        name="request_clarifications",
        description=(
            "Ask the user clarifying questions with multiple-choice options. "
            "Use when key details are missing or ambiguous. "
            "Provide 1-3 questions with helpful options. "
            "Each option can have a label and optional description. "
            "Include an 'Other' option when appropriate."
        ),
        func=request_clarifications_impl,
        args_schema=ClarificationRequestInput,
    )

    return [request_clarifications_tool]
