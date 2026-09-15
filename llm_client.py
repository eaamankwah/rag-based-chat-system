import os
from typing import Dict, List
from openai import OpenAI


def _resolve_base_url() -> str:
    """Resolve a custom OpenAI-compatible base URL, e.g. for classroom
    proxy keys (Vocareum, etc).

    The `openai` SDK only auto-reads the newer OPENAI_BASE_URL env var;
    older setup guides commonly document the pre-1.0 name OPENAI_API_BASE,
    which the current SDK silently ignores. Checking both here means this
    works regardless of which name your environment/instructor used.
    """
    return os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_API_BASE")

SYSTEM_PROMPT = (
    "You are a NASA mission operations expert and historian. You specialize in "
    "the Apollo 11, Apollo 13, and Space Shuttle Challenger missions. "
    "Answer questions using ONLY the mission context provided to you below - do "
    "not rely on outside knowledge that isn't supported by the context.\n\n"
    "Guidelines:\n"
    "- Ground every claim in the provided context and cite the source "
    "(e.g., mission and document) when you reference specific facts.\n"
    "- If the provided context does not contain enough information to fully "
    "answer the question, say so explicitly instead of guessing or inventing "
    "details.\n"
    "- Be precise, factual, and technical where appropriate, but explain "
    "acronyms and jargon so a curious non-expert can follow along.\n"
    "- Keep answers focused and well-organized. Use short paragraphs or "
    "bullet points for multi-part answers.\n"
)

MAX_HISTORY_TURNS = 6  # keep the last N (role, content) turns to control token usage


def generate_response(openai_key: str, user_message: str, context: str,
                     conversation_history: List[Dict], model: str = "gpt-3.5-turbo") -> str:
    """Generate response using OpenAI with context"""

    # Define system prompt: give the model its NASA-expert persona plus
    # instructions on how to treat retrieved context.
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]

    # Add chat history (trimmed to the most recent turns so we don't blow the
    # context window / cost budget on very long conversations). Only role and
    # content are forwarded - any extra keys (e.g. UI metadata) are dropped.
    if conversation_history:
        trimmed_history = conversation_history[-MAX_HISTORY_TURNS:]
        for turn in trimmed_history:
            role = turn.get("role")
            content = turn.get("content")
            if role in ("user", "assistant") and content:
                messages.append({"role": role, "content": content})

    # Set context in messages: inject the retrieved RAG context as its own
    # system-level note right before the new user question so the model
    # treats it as authoritative grounding material for this turn.
    if context:
        messages.append({
            "role": "system",
            "content": (
                "Use the following retrieved mission documents as your primary "
                "source of truth for the next question:\n\n" + context
            )
        })
    else:
        messages.append({
            "role": "system",
            "content": (
                "No relevant mission documents were retrieved for this "
                "question. Let the user know you don't have grounded context "
                "to answer confidently, then answer as best as you can while "
                "flagging the uncertainty."
            )
        })

    # Add the new user question
    messages.append({"role": "user", "content": user_message})

    # Create OpenAI Client (base_url is explicitly resolved so classroom
    # proxy keys like Vocareum's route to the right endpoint even if only
    # the older OPENAI_API_BASE env var name was set)
    client = OpenAI(api_key=openai_key, base_url=_resolve_base_url())

    # Send request to OpenAI and return response
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.3,
            max_tokens=800,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error generating response from OpenAI: {e}"
