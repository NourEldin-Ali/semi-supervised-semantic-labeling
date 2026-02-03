question_pairwise_prompt = """
You are given a user need and two generated questions (A and B) that try to satisfy it.

Inputs are formatted as:
- <user_need>...</user_need>
- <question group='A'>...</question>
- <question group='B'>...</question>

Evaluate which question is better overall using these criteria (in order):
1) Alignment with the user's intent
2) Completeness of coverage
3) Consistency with expected question-writing standards
4) Clarity and lack of ambiguity

Important:
- Do NOT prefer a question just because it repeats keywords.
- Prefer questions that explicitly or implicitly capture the user's need.
- Anchor comparisons on coverage AND enforceability across layers (defense-in-depth).

If both questions are correct, prefer the one that is more comprehensive, explicit,
or conservatively phrased. Only declare a tie if they are functionally identical.

Return only the structured data requested by the caller.
"""
