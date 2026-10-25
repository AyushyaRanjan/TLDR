# Prompts

Every call has one job, a short role line, a handful of numbered rules, one
tiny example, and a fixed plain-text shape. The document sits inside
`<document>` tags. The prompt says that block is untrusted data. Literal
`<document>` strings in the source are broken before wrapping, so a PDF cannot
close the tag early.

| Call | Job | Output | Cap |
| --- | --- | --- | --- |
| `chunk_summary` | Summarize one chunk | One paragraph | 280 |
| `chunk_notes` | Notes from one chunk | TAKEAWAYS, DEFINITIONS, NUMBERS | 400 |
| `reduce_summary` | Merge summaries without new facts | One paragraph | 500 |
| `summary_30s` | 30-second summary | One paragraph in a word range | 220 |
| `explain_3m` | 3-minute explanation | Why, How, Terms | 900 |
| `key_takeaways` | Top takeaways | 5–7 bullets | 350 |
| `open_questions` | Gaps left by the document | Bullets, or the missing-info sentence | 300 |
| `goal_relevance` | Score one chunk against the goal | `SCORE: 0-5` | 32 |
| `goal_prerequisites` | Tools, knowledge, accounts, versions | Bullets | 400 |
| `goal_steps` | Ordered steps, commands copied verbatim | Numbered list | 500 |
| `goal_relevant_parts` | Which pages matter | Bullets with `(p. N)` | 400 |
| `goal_skip` | What to skip for this goal | Bullets | 300 |
| `goal_gotchas` | Warnings from the text | Bullets | 300 |

Missing information is supposed to come back as `Not covered in the document.`
The pipeline does not trust that sentence for code or numbers. Those are
checked in `postprocess.py`.

If you edit a prompt, keep it to one job. Update the docstring token cap and
`MAX_TOKENS` in `config.py` together. `tests/test_prompts.py` fails when they
drift, and it checks that a full-size chunk prompt still fits in 8192 tokens.
