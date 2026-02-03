question_set_pairwise_prompt = """
You are given a user need and TWO sets of generated questions (Set A and Set B).

Inputs are formatted as:
- <user_need>...</user_need>
- <questions group='A'>...</questions>
- <questions group='B'>...</questions>

Evaluate which set is better overall using these criteria (in order):
1) Alignment with the user's intent
2) Completeness of coverage across the set
3) Consistency with expected question-writing standards
4) Clarity and lack of ambiguity

Important:
- Do NOT prefer a set just because questions repeat keywords.
- Prefer sets that explicitly or implicitly capture the user's need.
- Anchor comparisons on coverage AND enforceability across layers (defense-in-depth).

If both sets are correct, prefer the set that is more comprehensive, explicit,
or conservatively phrased. Only declare a tie if they are functionally identical.

Return only the structured data requested by the caller.
"""
