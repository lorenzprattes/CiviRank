# Prompts used for evaluation:
## Ranker Showcase
 
``` 
Generate a single JSON object containing an array named comments. 

Each comment in the array must have the following fields:

- "id": A unique identifier for the comment (e.g., "1").
- "parent_id": The identifier of the parent comment if this comment is a reply; otherwise, set as null.
- "text": The content of the comment, which should vary widely in civility—from very polite and constructive, through neutral, down to rude or uncivil—to enable comprehensive testing of a civility ranking system.

Requirements for generated data:

- Generate between 8 and 15 comments in total
- At least 4 comments should be standalone (i.e., parent_id is null).
- Include at least two threads where comments reply to other comments (i.e., comments with a non-null parent_id).
- Ensure varying thread depths (for instance, at least one reply to a reply).
- Clearly vary the civility level: include polite and helpful comments, neutral or clarifying comments, sarcastic or slightly rude comments, and at least two or more explicitly rude or offensive comments.
- Mix the civility levels of comments inbetween threads

Output only the valid JSON object with no additional text or explanations.
```