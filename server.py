import os
import json
from typing import List, Literal, Optional

from fastmcp import FastMCP, tool
from google import genai
from google.genai import types

# ---------- Load API key ----------
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY is not set. "
        "Configure it in FastMCP Cloud under Environment Variables."
    )

client = genai.Client(api_key=GEMINI_API_KEY)

# ---------- FastMCP server ----------
mcp = FastMCP(
    name="prompt-suggestion",
    version="1.0.0",
    description="Suggests improved follow-up prompts when a user dislikes an LLM answer.",
)

Role = Literal["user", "assistant", "system", "tool"]


def build_user_prompt(
    messages: List[dict],
    user_comment: Optional[str] = None,
    task_hint: Optional[str] = None,
    window_size: int = 8,
) -> str:
    """
    Build the big system-style prompt we send to Gemini.
    Truncates conversation to last `window_size` messages.
    """
    if len(messages) > window_size:
        messages = messages[-window_size:]

    convo = "\n".join(
        f"{m.get('role','user').upper()}: {m.get('content','')}" for m in messages
    )

    last_user = next(
        (
            m.get("content", "")
            for m in reversed(messages)
            if m.get("role") == "user"
        ),
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
- It MUST be written in the FIRST-PERSON perspective, as if I am directly talking to you (the assistant).
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
- "summary": short explanation of what went wrong with the assistant's last answer.
- "root_causes": array of 1–4 bullet-style reasons (strings) explaining the failure.
- "suggested_prompt": the single improved, self-contained FIRST-PERSON prompt the user should send next.
- "alternatives": array of 0–3 alternative prompts, each also directly sendable, self-contained, and written in first-person.
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


def _parse_gemini_json(raw_text: str) -> dict:
    """
    Gemini sometimes wraps JSON in text or ```json fences.
    Try to robustly extract the JSON object.
    """
    text = raw_text.strip()

    # Strip ``` fences if present
    if text.startswith("```"):
        # find first '{' and last '}'
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            text = text[start : end + 1]

    try:
        data = json.loads(text)
    except Exception:
        # fallback: try to locate first JSON-like block
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            snippet = text[start : end + 1]
            data = json.loads(snippet)
        else:
            raise ValueError("Model did not return valid JSON")

    # Normalize output shape + defaults
    summary = data.get("summary", "").strip()
    suggested = data.get("suggested_prompt", "").strip()
    root_causes = data.get("root_causes") or []
    alternatives = data.get("alternatives") or []
    confidence = data.get("confidence", 0.7)

    # clamp confidence
    try:
        c = float(confidence)
    except Exception:
        c = 0.7
    c = max(0.0, min(1.0, c))

    return {
        "summary": summary,
        "root_causes": root_causes,
        "suggested_prompt": suggested,
        "alternatives": alternatives,
        "confidence": c,
    }


@tool(
    mcp,
    name="analyze_dislike",
    description="Given a short conversation and a disliked answer, suggest a better follow-up prompt.",
)
def analyze_dislike(
    messages: List[dict],
    user_comment: Optional[str] = None,
    task_hint: Optional[str] = None,
) -> dict:
    """
    This is the core tool: same semantics as your Chrome extension.
    - `messages`: small conversation window [{role, content}, ...]
    - `user_comment`: optional explanation from the user
    - `task_hint`: optional domain hint (e.g., 'frontend', 'docker', etc.)
    """

    # Build the big instruction prompt
    prompt = build_user_prompt(messages, user_comment, task_hint)

    # Call Gemini 2.5 Flash
    resp = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.4,
            max_output_tokens=512,
            response_mime_type="text/plain",
        ),
    )

    raw_text = resp.text or ""
    data = _parse_gemini_json(raw_text)
    return data


if __name__ == "__main__":
    # This is what FastMCP calls during preflight
    mcp.run()
