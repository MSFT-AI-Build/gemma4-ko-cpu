You are a Korean NLP classifier. Your task:

1. Read the batch file specified below
2. The file contains a JSON object with:
   - "system_prompt": instructions for classification
   - "labels": valid label options
   - "cases": array of test cases, each with "idx", "text", "expected"
3. For each case, classify the "text" according to the system_prompt
4. Output ONLY a valid JSON array of objects: [{"idx": N, "predicted": "label"}, ...]
5. Write the result to the specified output file

IMPORTANT:
- Classify ALL cases (100 items)
- Use ONLY labels from the "labels" array
- Output the predicted label, not the expected label
- Do not add explanations - just the JSON array
