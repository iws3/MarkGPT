import os
import json
import logging

from time import time
from xml.parsers.expat import errors
from openai import OpenAI, APIConnectionError, APIStatusError, APITimeoutError
from typing import Dict, List
from dotenv import load_dotenv

from typer import prompt

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

ROUTER_SYSTEM_PROMPT = """
You are the planning step of a directory analysis agent.

Your only job is to read a user's question about a code repository
and return a JSON list of tools needed to answer it.

Available tools:
- directory_tree: the list of all tracked files in the repo
- recent_commits: last 20 commits with messages and changed files
- uncommitted_changes: any staged or unstaged work in progress
- last_diff: the full code diff introduced by the most recent commit

Rules:
- Return ONLY a raw JSON array. No explanation, no markdown, no prose.
- Choose the minimum set of tools needed. Don't over-fetch.
- If the question is about recent activity or history, include recent_commits.
- If the question is about current unfinished work, include uncommitted_changes.
- If the question is about what the last commit did, include last_diff.
- If the question is about project structure or what files exist, include directory_tree.
- You may combine tools when the question spans multiple concerns.

Examples:
  "what has been worked on recently?" -> ["recent_commits"]
  "what am i currently working on?" -> ["uncommitted_changes"]
  "what does this project look like?" -> ["directory_tree", "recent_commits"]
  "what did the last commit change?" -> ["last_diff"]
""".strip()


class AIResponseError(Exception):
    """All models exhausted or returned unusable responses."""

class AIClient:
    """Handles API calls with retries per model"""

    def __init__(self):
        self.models = [
        "google/gemini-2.5-flash-lite",   # Primary
        "meta-llama/llama-4-scout",  # Free fallback
        "mistralai/mistral-nemo",
        ]
        self.max_retries: int = 2
        self.retry_delay: int = 2
        self.timeout: int = 60
        self.client = OpenAI(
            base_url=os.getenv("OPENROUTER_BASE_URL"),
            api_key=os.getenv("OPENROUTER_API_KEY"),
        )

    def call_router(self, question: str) -> List[str]:

        errors = {}

        for model in self.models:
            try:
                log.info("Model: %s | Attempt 1/%d", model, self.max_retries)
                response = self.client.chat.completions.create(
                    model=model,
                    messages=[
                    {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
                    {"role": "user",   "content": question}
                    ],
                    timeout=self.timeout,
                )

                raw = response.choices[0].message.content
                parsed = json.loads(raw)
                if isinstance(parsed, list) and all(isinstance(item, str) for item in parsed):
                    return parsed
                if isinstance(parsed, dict):
                # grab the first list value found
                    for v in parsed.values():
                        if isinstance(v, list):
                            return v
                return []
            except Exception as e:
                errors[model] = f"{type(e).__name__}: {e}"
                log.warning("Model %s failed — trying next. (%s)", model, errors[model])

        raise AIResponseError(f"All models failed:\n" + "\n".join(f"  {m}: {e}" for m, e in errors.items()))

    def call_answerer(self, messages: list, system_prompt: str) -> str:
        errors = {}
        # print(messages=[{"role": "system", "content": system_prompt}] + messages)
        for model in self.models:
            try:
                log.info("Model: %s | Attempt 1/%d", model, self.max_retries)
                response = self.client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": system_prompt}] + messages,
                timeout=self.timeout,
                )
                return response.choices[0].message.content

            except Exception as e:
                errors[model] = f"{type(e).__name__}: {e}"
                log.warning("Model %s failed — trying next. (%s)", model, errors[model])

        raise AIResponseError(f"All models failed:\n" + "\n".join(f"  {m}: {e}" for m, e in errors.items()))

    def call_agent(self, messages: list, system_prompt: str) -> str:
        """
        Single model call for the agent loop.
        Unlike call_answerer, this is called repeatedly with growing
        history — the caller manages the conversation state.
        Returns raw text so the agent loop can parse THOUGHT/ACTION/ANSWER.
        """
        errors = {}

        for model in self.models:
            try:
                log.info("Model: %s", model)
                response = self.client.chat.completions.create(
                    model=model,
                    messages=[{"role": "system", "content": system_prompt}] + messages,
                    timeout=self.timeout,
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                errors[model] = f"{type(e).__name__}: {e}"
                log.warning("Model %s failed — trying next. (%s)", model, errors[model])

        raise AIResponseError(f"All models failed:\n" + "\n".join(f"  {m}: {e}" for m, e in errors.items()))

evaluator = AIClient()
        
def get_tools_needed(prompt: str) -> Dict:
    return evaluator.call_router(prompt)
    
def get_answer(messages: list, system_prompt: str = None) -> str:
    if system_prompt is None:
        raise ValueError("system_prompt is required for call_answerer")
    return evaluator.call_answerer(messages, system_prompt)

def call_agent_step(messages: list, system_prompt: str) -> str:
    return evaluator.call_agent(messages, system_prompt)