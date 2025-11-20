import os
import json
from typing import Literal, TypedDict, List, Optional

from fastmcp import FastMCP
from google import genai
from google.genai import types as genai_types

# ---------- Config ----------

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    # In FastMCP Cloud this comes from the dashboard env vars
    raise RuntimeError("GEMINI_API_KEY is not set")

client = genai.Client(api_key=api_key)

# Our MCP server instance
mcp = FastMCP(name="PromptSuggestion")

# ---------- Types ----------

class ChatMessage(TypedDict):
    role: Literal["user", "assistant", "system", "tool"]
    content: str


class AnalyzeResult(TypedDict):
    summary: str
    root_causes: List[str]
    suggested_prompt: str
    alternatives: List[str]
    confidence: float


# ---------- Helper: truncate & prompt builder ----------

MAX_MESSAGES = 8  # roughly last 4 turns (user+assistant pairs)


def truncate_messages(messages: List[ChatMessage]) -> List[ChatMessage]:
    """Keep only the last N messages."""
    return messages[-MAX_MESSAGES:]


def build_user_prompt(
    messages: List[ChatMessage],
    user_comment: Optional[str] = None,
    task_hint: Optional[str] = None,
) -> str:
    # Same spirit as your TS version
    trimmed = truncate_messages(messages)

    convo = "\n".join(
        f"{m['role'].upper()}: {m['content']}"
        for m in trimmed
    )

    last_user = next(
        (m["content"] for m in reversed(trimmed) if m["role"] == "user"),
        "(unknown last user message)",
    )

    return f"""
You are a PROMPT REWRITER for LLM chats.

You will receive:
- A short conversation between a USER and an ASSISTANT.
- The USER's most recent message.
- An optional user comment explaining why they disliked the last answer.
- An optional hint about the task domain.

Your job is to:
1. Infer what the user REALLY wanted, especially from the LAST user message.
2. Briefly understand why the last answer failed.
3. Produce a SINGLE, SELF-CONTAINED prompt that the user can send to the SAME assistant to get a much better answer.

CRITICAL RULES FOR "suggested_prompt":
- It MUST be written in the FIRST-PERSON perspective, as if the user is directly talking to the assistant.
  - Use "you" to refer to the assistant.
  - Use "I", "me", and "my" to refer to the user.
- Avoid generic phrases like "the user" or "users" when describing benefits.
  - Prefer phrasing such as "how it benefits me" or "how it improves my experience".
- It MUST NOT mention "the user", "the assistant", "previous discussion", "conversation above", "earlier answer", or thumbs-down feedback.
- It MUST be SELF-CONTAINED: it should make sense even if the assistant never saw the prior conversation.
- It MUST directly request the desired result (explanation, code, design, plan, etc.).
- It CAN add clarifying constraints based on context (e.g. "in simple terms", "with 3 concrete UX improvements", "step-by-step").
- It SHOULD be clear, concise, and specific.

You will output STRICT JSON with these keys:
- "summary": short explanation of what went wrong with the assistant's last answer.
- "root_causes": array of 1–4 bullet-style reasons (strings) explaining the failure.
- "suggested_prompt": the single BEST improved, self-contained FIRST-PERSON prompt the user should send next.
- "alternatives": ALWAYS an empty array [] (do NOT put any prompts here).
- "confidence": number between 0 and 1 (your confidence that the suggested_prompt will work well).

Full conversation:
{convo}

Last user message (for focus):
{last_user}

User comment (may be empty):
{user_comment or "(none)"}

Task hint (may be empty):
{task_hint or "(none)"}
""".strip()


# ---------- Helper: call Gemini + JSON parsing ----------

def call_llm(system: str, user: str) -> str:
    """Call Gemini and return raw text."""
    prompt = f"System:\n{system}\n\nUser:\n{user}"

    resp = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,  # plain text is fine
        config=genai_types.GenerateContentConfig(
            temperature=0.2,
            max_output_tokens=8192,  # Changed from 768 to 8192
        ),
    )

    return (resp.text or "").strip()



def strip_code_fences(text: str) -> str:
    """Remove ``` or ```json fences if the model wraps the JSON."""
    t = text.strip()
    if t.startswith("```"):
        parts = t.split("```", 2)
        if len(parts) >= 2:
            inner = parts[1]
            if inner.lstrip().startswith("json"):
                inner = inner.lstrip()[4:].lstrip()
            return inner.strip()
    return t


def safe_parse_json(text: str) -> dict:
    cleaned = strip_code_fences(text)
    try:
        return json.loads(cleaned)
    except Exception:
        # Last resort: try to find outermost braces
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(cleaned[start : end + 1])
        raise


# ---------- MCP Tool ----------

@mcp.tool()
def analyze_dislike(
    messages: List[ChatMessage],
    user_comment: Optional[str] = None,
    task_hint: Optional[str] = None,
) -> AnalyzeResult:
    """
    Analyze a disliked LLM response and suggest a better follow-up prompt.
    """

    base_system = (
        "Return ONLY a JSON object with keys: "
        "summary, root_causes, suggested_prompt, alternatives, confidence."
    )
    strict_system = (
        "STRICT JSON ONLY. Keys: summary, root_causes, suggested_prompt, "
        "alternatives, confidence. No extra text."
    )

    user_prompt = build_user_prompt(messages, user_comment, task_hint)

    # First attempt
    raw = call_llm(base_system, user_prompt)
    try:
        data = safe_parse_json(raw)
    except Exception:
        # Retry with stricter instructions
        raw2 = call_llm(strict_system, user_prompt)
        data = safe_parse_json(raw2)

    # Build result with sensible defaults
    summary = data.get("summary", "").strip()
    suggested_prompt = data.get("suggested_prompt", "").strip()
    root_causes = data.get("root_causes") or []
    # Ignore whatever the model put in `alternatives` – we want a single best prompt only
    confidence_raw = data.get("confidence", 0.8)

    try:
        confidence = float(confidence_raw)
    except Exception:
        confidence = 0.8

    if confidence < 0:
        confidence = 0.0
    if confidence > 1:
        confidence = 1.0

    result: AnalyzeResult = {
        "summary": summary,
        "root_causes": list(root_causes),
        "suggested_prompt": suggested_prompt,
        # 🔒 Force alternatives to always be an empty list
        "alternatives": [],
        "confidence": confidence,
    }
    return result



# ---------- Local dev entrypoint (FastMCP Cloud ignores this) ----------

if __name__ == "__main__":
    # For local testing: `fastmcp run server.py`
    mcp.run()