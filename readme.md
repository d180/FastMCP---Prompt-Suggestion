# Prompt Suggestion MCP Server

A Model Context Protocol (MCP) server that helps any LLM (ChatGPT, Claude, etc.) instantly rewrite disappointing responses into **one clean, high-quality prompt**.

This MCP server analyzes the recent conversation, identifies what the user actually wanted, and generates a **single, first-person improved prompt** tailored to produce a better response from the assistant.

No noise.  
No multiple choices.  
Just one perfect next prompt — every time.

---

## ✨ Features

### 🎯 Smart Failure Understanding  
The server reads:
- The recent conversation window  
- The last user message  
- An optional user comment (why they disliked the answer)

And analyzes:
- What the user was trying to accomplish  
- Why the assistant’s prior answer failed  
- What information or structure was missing  

---

### ✍️ High-Quality Prompt Rewriting  
The MCP server generates a **single optimized prompt** that:

- Is written fully in **first person**  
  - (“Explain to me…”, “Give me…”, “Help me…”)  
- Is **self-contained**  
  - The prompt works even if the model never saw the past conversation  
- Is clear and specific  
- Avoids generic references like “the user” or “previous discussion”  
- Adds helpful structure when needed (steps, examples, analogies, UX guidelines, etc.)

Optional: Up to 2 lightweight alternatives may be included.

---

### 🧠 Consistent Output  
No matter how the user triggers the tool —  
“use the tool”, “suggest a better prompt”, “rewrite this”, etc. —  
the server always returns **one single best prompt**, not multiple choices.

---

## 🔗 Connect the MCP Server to ChatGPT (Developer Mode)

### 1️⃣ Enable Developer Mode  
Open ChatGPT:

**Settings → Apps & Connectors → Advanced Settings → Enable Developer Mode**

---

### 2️⃣ Add the MCP Server  
Go to:

**Apps & Connectors → Create → New MCP Connector**

Fill in the following:

| Field | Value |
|-------|--------|
| **Name** | Prompt Suggestion Tool (or any name you want) |
| **Description** | Automatically rewrites your prompts into a higher-quality version |
| **Server URL** | `https://PromptSuggestion.fastmcp.app/mcp` |
| **Authentication** | No Authentication |

Check the trust box → Click **Create**

---

### 3️⃣ Refresh ChatGPT  
Now click the **+** button next to the message box.  
Under **More**, you will see the MCP tool with the name you assigned.

Your prompt-rewriting MCP server is now active 🎉

---

## 🧠 How to Use the MCP Tool

At any time, simply type:

- **“Use the tool and give me a better prompt.”**  
- **“Rewrite this using the MCP server.”**  
- **“Fix my prompt using the tool.”**

ChatGPT will:
1. Detect your request  
2. Call your MCP server  
3. Ask for permission  
4. Return a clean, improved prompt  

Every result contains:
- A summary of what went wrong  
- Root-cause reasons  
- **One single improved prompt** (the one you should send next)  
- Optional small alternative suggestions  
- A confidence score  

---

## 🧩 Output Format  
The MCP server always returns:

```json
{
  "summary": "...",
  "root_causes": ["..."],
  "suggested_prompt": "THE BEST FIRST-PERSON PROMPT",
  "alternatives": ["..."],
  "confidence": 0.0 - 1.0
}
