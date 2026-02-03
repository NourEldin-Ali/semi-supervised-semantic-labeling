pairwise_comparison_prompt = """
You are given one question and two labels (Group A and Group B) that address the same question.
Inputs are formatted as:
- <question id='...'>...</question>
- <labels group='A' method='...'>...</labels>
- <labels group='B' method='...'>...</labels>

Your task is to evaluate both labels carefully and objectively, considering the following
criteria in this order of priority:
1) Alignment with the intent of the question
2) Completeness of coverage
3) Consistency with expected labeling standards
4) Clarity and lack of ambiguity

When comparing the two labels:
- If both labels are correct, prefer the label that is more comprehensive, explicit,
  or conservative in interpretation.
- If one label is slightly more detailed, cautious, or broadly applicable, favor that label.
- Only declare a tie if both labels are functionally identical in meaning, scope, and quality.

Additional penalties/guidance:
- Duplicates (including case-only differences) are OK and should not be penalized.
- Unclear abbreviations/acronyms (e.g., "CSP", "CSC") are NOT minor and should be penalized strongly.
- Strongly penalize labels that are just standard IDs (e.g., "ISO 27001") unless explicitly mentioned
  AND meaningfully described.
- Labels should reflect the question's intent, and the scope should be clear.

Output format (strict):
Winner: Group A / Group B / Tie
Explanation: Brief justification for the decision based on the criteria above.
"""
