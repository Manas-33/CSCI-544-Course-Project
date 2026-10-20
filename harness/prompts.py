"""Provisional shared wording; only problem and candidate enter the user message."""

TEMPLATES = {
    "T1": "Problem:\n{problem}\n\nCandidate solution:\n{candidate_answer}\n\n"
          "Is the final answer correct? Answer Yes or No.",
    "T2": "You are a grader. Evaluate the candidate solution to the problem below.\n\n"
          "Problem:\n{problem}\n\nCandidate solution:\n{candidate_answer}\n\n"
          "Is the candidate's final answer correct? Answer Yes or No.",
    "T3": "Problem:\n{problem}\n\nCandidate solution:\n{candidate_answer}\n\n"
          "Is the final answer incorrect? Answer Yes or No.",
}


def messages(record, template):
    return [{"role": "user", "content": TEMPLATES[template].format(
        problem=record["problem"], candidate_answer=record["candidate_answer"])}]


def encode_prompt(tokenizer, record, template):
    return tokenizer.apply_chat_template(
        messages(record, template), tokenize=True, add_generation_prompt=True,
        truncation=False,
    )
