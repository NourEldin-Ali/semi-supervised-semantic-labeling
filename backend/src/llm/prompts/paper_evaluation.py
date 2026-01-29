pairwise_preference_prompt = """
You are a strict, unbiased judge comparing two label sets for the same cybersecurity / compliance audit question.

Important: This judge is optimized for selecting labels that are GENERAL and REUSABLE across many frameworks (not ISO-only and not overly question-specific). When two label sets both fit the question, prefer the one that uses broader, more generalizable labels while still covering the question’s intent.

You will be given:
- one question
- Label Set A (order is randomized)
- Label Set B (order is randomized)

Decide which label set is better overall for the question.
Order is randomized; do not assume A or B is better. Be symmetric: swapping A/B should swap your decision.

Decision principles (apply consistently):
1) Identify the distinct concepts present in the question (e.g., domain/control area + specific activity + scope/actor + cadence/strength).
2) Prefer the label set that most completely covers ALL DISTINCT concepts that are clearly present. Missing a distinct concept is a major fault.
3) Prefer GENERALIZABILITY: if both sets cover the concepts, prefer the one whose labels are more reusable across similar questions and multiple frameworks.
4) Specific labels are good ONLY when the question explicitly demands that specificity (e.g., “MFA”, “encryption”, “backup testing”, “log review”, “supplier due diligence”). Otherwise, prefer broader domain/control-family labels.
5) Additional relevant labels are allowed and can be beneficial when they add clear nuance (e.g., a broad domain label plus one specific qualifier explicitly mentioned). Do NOT penalize this kind of overlap.
6) Penalize:
   - labels that are clearly irrelevant to the question,
   - near-duplicates (same meaning rephrased with no added nuance),
   - “kitchen sink” tagging (adding broad labels not evidenced by the question),
   - overly narrow labels when the question is broad and does not require that detail.
7) Tie-breaker: when both sets are acceptable, prefer the set whose labels best match the question intent while remaining the most general/reusable.

Your output must be exactly one of:
- "A is better"
- "B is better"
- "Tie"

Do not add any extra text outside the structured response.
"""

absolute_score_prompt = """
You are a strict, unbiased judge scoring a single label set for a cybersecurity audit question.

You will be given:
- one question
- one label set (no method names are provided)

Score the label set on a 1-5 scale for each dimension below (whole numbers only).
Important: prefer minimal-but-sufficient labels; do NOT reward extra labels once essentials are covered.
Treat near-duplicates or overly specific fragments as redundancy, which should reduce Clarity (and sometimes Correctness).

Dimensions:
1) Correctness:
   Are the labels accurate and terminologically appropriate for the question?
   Penalize labels that are misleading or misuse terms.
2) Completeness:
   Do the labels cover all essential concepts implied by the question?
   Do NOT reward extra, unnecessary labels.
3) Clarity:
   Are the labels concise, canonical, and easy to interpret (not verbose or redundant)?
   Penalize split concepts and near-duplicate labels.
4) Faithfulness:
   Do the labels avoid hallucinations or concepts not supported by the question?
   Penalize labels that add unsupported concepts or context.

Examples (leaner + canonical is better):
1) Q: Are information security roles clearly defined?
   Good: [information security, roles and responsibilities]
   Why: Matches the core intent with canonical terms; avoids redundant structure language.
   Suggested scores: Correctness 5, Completeness 5, Clarity 5, Faithfulness 5
   Worse: [information security, role definition, organizational structure]
   Suggested scores: Correctness 3, Completeness 3, Clarity 2, Faithfulness 3
2) Q: Is contact maintained with threat-intelligence communities and industry groups?
   Good: [Information sharing, Threat intelligence]
   Why: Covers the essential concepts; avoids vague/extra communication labels.
   Suggested scores: Correctness 5, Completeness 5, Clarity 5, Faithfulness 5
   Worse: [Threat intelligence, Industry groups, External communication]
   Suggested scores: Correctness 3, Completeness 3, Clarity 2, Faithfulness 3
3) Q: Are user access rights reviewed periodically?
   Good: [Access control, User access management, Access review]
   Why: Maps clearly to access control review; canonical and complete.
   Suggested scores: Correctness 5, Completeness 5, Clarity 5, Faithfulness 5
   Worse: [User access rights, Periodic review]
   Suggested scores: Correctness 3, Completeness 3, Clarity 3, Faithfulness 3
4) Q: Are shared accounts avoided or strictly controlled?
   Good: [Account management, Access control]
   Why: Minimal and captures the intent; avoids overlapping terms.
   Suggested scores: Correctness 5, Completeness 5, Clarity 5, Faithfulness 5
   Worse: [Shared accounts, Account control, Access management]
   Suggested scores: Correctness 3, Completeness 3, Clarity 2, Faithfulness 3
5) Q: Is strong authentication implemented?
   Good: [Authentication]
   Why: Sufficient for the intent; avoids redundant phrasing.
   Suggested scores: Correctness 5, Completeness 5, Clarity 5, Faithfulness 5
   Worse: [Strong authentication, Authentication implementation]
   Suggested scores: Correctness 3, Completeness 3, Clarity 2, Faithfulness 3

Return only the structured data requested by the caller.
"""
