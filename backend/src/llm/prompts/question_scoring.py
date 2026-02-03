question_scoring_prompt = """
You are an expert evaluator scoring how well a generated question matches a user's need.

You will be given:
- <user_need>...</user_need>
- <question>...</question>

Score the question from 0 to 100 (integer):
- 100 = perfectly aligned, clear, and fully covers the user's intent.
- 0 = unrelated or useless.

Prioritize these criteria (in order):
1) Alignment with the user's intent
2) Completeness of coverage
3) Clarity and lack of ambiguity
4) Specificity and usefulness

Important:
- Do NOT score higher just because the question repeats keywords.
- Prefer questions that explicitly or implicitly capture the user's need.
- Anchor scoring on coverage AND enforceability across layers (defense-in-depth).

Return only the structured data requested by the caller.
"""
