"""Utility for auto-generating thread titles."""

from __future__ import annotations

import logging
from typing import Any, Sequence

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.callbacks.manager import adispatch_custom_event
from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)


def _get_message_content(msg: BaseMessage | dict[str, Any]) -> str:
    """Extract content from a message object or dict."""
    if isinstance(msg, BaseMessage):
        content = msg.content
    elif isinstance(msg, dict):
        content = msg.get("content", "")
    else:
        return ""
    
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        # Handle content blocks (text, images, etc.)
        texts = []
        for block in content:
            if isinstance(block, str):
                texts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                texts.append(block.get("text", ""))
        return " ".join(texts)
    return str(content)


def _is_human_message(msg: BaseMessage | dict[str, Any]) -> bool:
    """Check if a message is from the human."""
    if isinstance(msg, HumanMessage):
        return True
    if isinstance(msg, dict):
        return msg.get("type") == "human"
    return False


def _is_ai_message(msg: BaseMessage | dict[str, Any]) -> bool:
    """Check if a message is from the AI."""
    if isinstance(msg, AIMessage):
        return True
    if isinstance(msg, dict):
        return msg.get("type") == "ai"
    return False


async def generate_thread_title(
    messages: Sequence[BaseMessage | dict[str, Any]],
    model: str = "gpt-4o-mini",
) -> str | None:
    """
    Generate a concise title for a conversation thread.
    
    Args:
        messages: The conversation messages
        model: The model to use for title generation
        
    Returns:
        A short title string, or None if generation fails
    """
    if not messages:
        return None
    
    # Find first human and first AI message
    first_human = None
    first_ai = None
    
    for msg in messages:
        if first_human is None and _is_human_message(msg):
            first_human = _get_message_content(msg)
        elif first_ai is None and _is_ai_message(msg):
            first_ai = _get_message_content(msg)
        if first_human and first_ai:
            break
    
    if not first_human:
        return None
    
    try:
        llm = ChatOpenAI(model=model, temperature=0, max_tokens=50)
        
        # Build the prompt
        context = f"User: {first_human[:500]}"
        if first_ai:
            context += f"\n\nAssistant: {first_ai[:500]}"
        
        prompt = f"""Generate a concise 3-6 word title for this conversation. 
Return ONLY the title, no quotes or explanation.

{context}

Title:"""
        
        response = await llm.ainvoke(prompt)
        title = response.content.strip().strip('"').strip("'")
        
        # Validate - should be short
        if len(title) > 60:
            title = title[:57] + "..."
        
        logger.info(f"Generated thread title: {title}")
        return title
        
    except Exception as e:
        logger.warning(f"Failed to generate thread title: {e}")
        return None


async def emit_thread_title_event(title: str) -> None:
    """Emit a UI event with the generated thread title."""
    await adispatch_custom_event(
        "ui",
        {
            "type": "ui",
            "id": "thread_title",
            "name": "thread_title",
            "props": {
                "title": title,
            },
            "metadata": {"merge": True},
        },
    )


async def maybe_generate_and_emit_title(
    messages: Sequence[BaseMessage | dict[str, Any]],
    has_existing_title: bool = False,
) -> str | None:
    """
    Generate and emit a thread title if appropriate.
    
    Only generates a title if:
    - There's no existing title
    - There's at least one human message and one AI message
    
    Args:
        messages: The conversation messages
        has_existing_title: Whether the thread already has a custom title
        
    Returns:
        The generated title, or None
    """
    if has_existing_title:
        return None
    
    # Check if we have enough messages (at least 1 human + 1 AI)
    has_human = any(_is_human_message(m) for m in messages)
    has_ai = any(_is_ai_message(m) for m in messages)
    
    if not (has_human and has_ai):
        return None
    
    title = await generate_thread_title(messages)
    if title:
        await emit_thread_title_event(title)
    
    return title

