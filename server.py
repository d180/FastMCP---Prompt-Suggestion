from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

from fastmcp import FastMCP
from google import genai
# ---------- Config ----------

GEMINI_MODEL = "gemini-2.0-flash"  # adjust if you prefer a different model name


def _get_gemini_client() -> genai.Client:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set in environment variables")
    return genai.Client(api_key=api_key)


# ---------- MCP server ----------

# You can list dependencies here so `fastmcp install` knows what to pull in
mcp = FastMCP(
    name="dislike-coach",
    dependencies=["google-genai"],
)


# ---------- Helper: build the user-facing analysis prompt ----------

def build_user_prompt(
    messages: List[Dict[str, str]],
    user_comment: Optional[str] = None,
    task_hint: Optional[str] = None,
) -> str:
    """
    Build the prompt text for Gemini, matching the TypeScript buildUserPrompt behavior.
    """

    # 🔥 Recommended truncation: last 4 messages = 2 turns
    truncated = messages[-4:]

    # Recreate "ROLE: content" format
    convo_lines: List[str] = []
    for m in truncated:
        role = (m.get("role") or "user").upper()
        content = (m.get("content") or "").strip()
        if not content:
            continue
        convo_lines.append(f"{role}: {content}")
    convo = "\n".join(convo_lines)

    # Last user message
    last_user = "(unknown last user message)"
    for m in reversed(truncated):
        if m.get("role") == "user" and m.get("content"):
            last_user = str(m["content"]).strip()
            break

    # Full rewritten prompt instructions (same as TS)
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

Examples of BAD suggested_prompt (DO NOT WRITE THESE):
- "Based on the previous discussion, can you..."
- "Provide HTML/CSS for a login page for the user..."
- "Explain to the user how Docker works."
- "Explain how this improves the user's experience."

Examples of GOOD suggested_prompt (STYLE TO FOLLOW):
- "Explain Kubernetes pods to me in simple terms using a real-world analogy, and give me 3 concrete use cases."
- "Give me HTML/CSS for a simple login page, and include 3 specific modern UX improvements. Explain how each improvement helps me."
- "Explain how neural networks work to me in beginner-friendly language, using a clear analogy and a short example."

You will output STRICT JSON with these keys:
- "summary"
- "root_causes"
- "suggested_prompt"
- "alternatives"
- "confidence"

Full conversation:
{convo}

Last user message (for focus):
{last_user}

User comment:
{user_comment or "(none)"}

Task hint:
{task_hint or "(none)"}
""".strip()


# ---------- Helper: JSON-safe parsing & clamping ----------

def parse_and_normalize_output(raw_text: str) -> Dict[str, Any]:
    """
    Try to parse model output as JSON and normalize fields so that:
    - root_causes is a list of strings
    - alternatives is a list of strings
    - confidence is in [0, 1]
    """
    text = raw_text.strip()

    # Try to strip accidental ```json ... ``` fences if present
    if text.startswith("```"):
        # crude fence removal
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:].strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # Fallback: wrap in a safe "failed" result
        return {
            "summary": "Model output was not valid JSON.",
            "root_causes": [
                "The coach model did not follow the strict JSON-only instructions."
            ],
            "suggested_prompt": "Please help me debug why my last LLM answer was bad.",
            "alternatives": [],
            "confidence": 0.3,
        }

    # Ensure types / defaults
    summary = str(data.get("summary", "") or "").strip()
    root_causes_raw = data.get("root_causes") or []
    alternatives_raw = data.get("alternatives") or []
    suggested_prompt = str(data.get("suggested_prompt", "") or "").strip()
    confidence_raw = data.get("confidence", 0.5)

    # Normalize lists
    root_causes: List[str] = []
    if isinstance(root_causes_raw, list):
        root_causes = [str(x) for x in root_causes_raw if str(x).strip()]
    else:
        root_causes = [str(root_causes_raw)] if root_causes_raw else []

    alternatives: List[str] = []
    if isinstance(alternatives_raw, list):
        alternatives = [str(x) for x in alternatives_raw if str(x).strip()]
    else:
        alternatives = [str(alternatives_raw)] if alternatives_raw else []

    # Confidence clamp
    try:
        conf_val = float(confidence_raw)
    except (TypeError, ValueError):
        conf_val = 0.5
    if conf_val < 0:
        conf_val = 0.0
    if conf_val > 1:
        conf_val = 1.0

    # If model forgot a good suggested_prompt, make a fallback
    if not suggested_prompt and summary:
        suggested_prompt = f"Help me with the following: {summary}"

    if not suggested_prompt:
        suggested_prompt = (
            "Help me refine my question so I can get a better answer than last time."
        )

    return {
        "summary": summary or "Unknown user goal.",
        "root_causes": root_causes,
        "suggested_prompt": suggested_prompt,
        "alternatives": alternatives,
        "confidence": conf_val,
    }


# ---------- Tool: analyze_dislike ----------

@mcp.tool()
def analyze_dislike(
    messages: List[Dict[str, str]],
    user_comment: Optional[str] = None,
    task_hint: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Analyze a disliked LLM response and suggest a better follow-up prompt.

    Args:
        messages: Recent conversation turns as a list of { "role": "...", "content": "..." }.
                  Roles are typically "user" or "assistant".
        user_comment: Optional free-text explanation from the user about why they disliked
                      the answer.
        task_hint: Optional short hint about the task domain, e.g. "coding", "data analysis",
                   "UI design", "math homework", etc.

    Returns:
        A JSON object with keys:
          - summary: string
          - root_causes: list[string]
          - suggested_prompt: string  (first-person phrasing)
          - alternatives: list[string]
          - confidence: float in [0, 1]
    """
    client = _get_gemini_client()

    user_prompt = build_user_prompt(
        messages=messages,
        user_comment=user_comment,
        task_hint=task_hint,
    )

    # We keep it single-prompt; Gemini will generate the JSON directly.
    resp = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=user_prompt,
    )

    # The python SDK exposes .text with concatenated text parts
    raw_text = getattr(resp, "text", "") or ""

    result = parse_and_normalize_output(raw_text)
    return result


if __name__ == "__main__":
    # Run as a stdio MCP server (FastMCP handles the protocol)
    mcp.run()
