def build_prompt(mode: str, context: dict, docs: list, query: str) -> str:
    context_block = "\n".join(f"{k}: {v}" for k, v in context.items())
    knowledge_block = "\n\n".join(doc.page_content for doc in docs)

    if mode == "daily_advice":
        system_instruction = """
You are a nutrition coach.
Give ONE concise daily insight.
No questions. No long explanations.
Tone: calm, motivating, human.
"""
        response_structure = """
Daily Insight:
- Insight
- Why it matters
- One small action today
"""

    elif mode == "recommendation":
        system_instruction = """
You are a nutrition coach.
Give a contextual food or habit recommendation.
Be specific, practical, and realistic.
"""
        response_structure = """
Recommendation:
- What to do
- Why this helps
- How to implement today
"""

    else:  # chat / default
        system_instruction = """
You are a nutrition coaching assistant.
Answer clearly and concisely.
Be friendly and supportive.
"""
        response_structure = """
1. Observation
2. Reason
3. Actionable Suggestion
"""

    return f"""
{system_instruction}

USER CONTEXT:
{context_block}

RETRIEVED KNOWLEDGE:
{knowledge_block}

USER QUESTION:
{query}

Respond using this structure ONLY:
{response_structure}
"""
