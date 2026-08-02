Planner experiment - how to run

1. For each entry in prompts.jsonl, send the "prompt" text to the LLM
   (Gemini API or chat window; temperature 0 if you can set it). 75
   prompts in total: 25 scenes x 3 conditions.
2. Save each reply. With an API key this is a 20-line loop; ask for the
   snippet if wanted.
3. Score the plans in scoring_sheet_blind.csv IN SLOT ORDER (the order is
   shuffled so conditions cannot be guessed). Three y/n columns per plan:
   - clears the top object before grasping the target
   - grasps the right object
   - invents nothing (no objects or constraints that are not in the prompt)
4. Return the filled sheet; scoring by condition is automatic from the key.

The comparison that matters: condition C (our labels) vs B (human labels),
with A showing what happens with no relations at all.
