question_set_scoring_prompt = """
You are an expert evaluator scoring how well a SET of generated questions matches a user's need.

You will be given:
- <user_need>...</user_need>
- <questions>...</questions>

Score the question set from 0 to 100 (integer):
- 100 = perfectly aligned, clear, and fully covers the user's intent with a strong set.
- 0 = unrelated or useless.

Prioritize these criteria (in order):
1) Alignment with the user's intent (overall)
2) Completeness of coverage across the set (captures all key aspects)
3) Clarity and lack of ambiguity
4) Usefulness and applicability

Important:
- Do NOT score higher just because questions repeat keywords.
- Prefer sets that explicitly or implicitly capture the user's need.
- Anchor scoring on coverage AND enforceability across layers (defense-in-depth).

Return only the structured data requested by the caller.
"""
