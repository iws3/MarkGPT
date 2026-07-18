from utils.config import get_answer

ANSWERER_SYSTEM_PROMPT = """
You are an expert project reporter embedded in a software repository.
Your job is to answer the user's question clearly and honestly
based only on the context provided.

Guidelines:
- Be direct. Lead with the answer, then explain.
- Refer to specific files, commit messages, and changes where relevant.
- Do not guess or invent details not present in the context.
- Write for a developer who wants to understand what is happening,
  not just get a yes/no answer.
""".strip()

def answer(question: str, context: dict) -> str:
    """
    Given a user question and the gathered context, build a prompt
    and ask the model to answer it.
    """
    context_block = "\n\n".join(
        f"--- {key.upper()} ---\n{value}"
        for key, value in context.items()
    )
    messages = [{
        "role": "user",
        "content": f"Here is the current state of the repository:\n\n{context_block}\n\n---\n\nQuestion: {question}"
    }]
    return get_answer(messages, system_prompt=ANSWERER_SYSTEM_PROMPT)
