import json
import re
from rich.console import Console
from rich.panel import Panel
from tools import TOOLS, run_tool
from utils.config import call_agent_step

console = Console()

MAX_ITERATIONS = 8

AGENT_SYSTEM_PROMPT = """
You are a repository reporter agent. You answer questions about a code repository
by thinking step by step and using tools to gather evidence before answering.

You have access to these tools:
{tool_descriptions}

On each turn you must output EXACTLY one of these formats — nothing else:

If you need to use a tool:
THOUGHT: <your reasoning about what you need and why>
ACTION: {{"tool": "<tool_name>", "args": {{"key": "value"}}}}

If the tool takes no arguments, use an empty object:
ACTION: {{"tool": "<tool_name>", "args": {{}}}}

Examples of valid actions:
ACTION: {{"tool": "recent_commits", "args": {{}}}}
ACTION: {{"tool": "read_file", "args": {{"filepath": "src/main.py", "start_line": 1, "end_line": 50}}}}
ACTION: {{"tool": "directory_tree", "args": {{}}}}

If you have enough information to answer:
THOUGHT: <your final reasoning>
ANSWER: <your complete answer to the user's question>

Rules:
- Always start with a THOUGHT.
- Only call one tool per turn.
- After each OBSERVATION, decide whether you have enough to answer or need more.
- For read_file, think about whether you need the whole file or specific lines.
  If the file is large and you only need a section, use start_line and end_line.
- Never invent or assume information not present in your observations.
- When you have enough evidence, stop calling tools and write your ANSWER.
""".strip()


def build_tool_descriptions() -> str:
    """
    Format each tool's name, arguments, and description into
    a readable block the model can scan when deciding what to use.
    """
    lines = []
    for name, tool in TOOLS.items():
        args_str = ""
        if tool["args"]:
            args_str = " Arguments: " + ", ".join(tool["args"])
        lines.append(f"- {name}:{args_str}\n  {tool['desc']}")
    return "\n".join(lines)


def parse_response(text: str) -> dict:
    """
    Extract THOUGHT, ACTION, and ANSWER from the model's raw output.

    Why not ask the model to return pure JSON for everything?
    Because THOUGHT is free-form reasoning — forcing it into JSON
    makes the model spend tokens on escaping rather than thinking.
    We only need the ACTION to be structured, so we parse selectively.
    """
    result = {"thought": None, "action": None, "answer": None}

    # Extract THOUGHT — everything between THOUGHT: and the next keyword
    thought_match = re.search(
        r"THOUGHT:\s*(.+?)(?=ACTION:|ANSWER:|$)", text, re.DOTALL
    )
    if thought_match:
        result["thought"] = thought_match.group(1).strip()

    # Extract ACTION — grab everything from the opening brace onward
    action_match = re.search(r"ACTION:\s*(\{.+)", text, re.DOTALL)
    if action_match:
        raw_action = action_match.group(1).strip()

        # Attempt 1: parse as-is
        try:
            parsed_action = json.loads(raw_action)
            if parsed_action.get("tool") != "error":
                result["action"] = parsed_action
        except json.JSONDecodeError:
            # Fix unclosed braces — some models drop the final }
            # Count the imbalance and append the missing closing braces
            open_count = raw_action.count("{")
            close_count = raw_action.count("}")
            fixed = raw_action + ("}" * (open_count - close_count))
            try:
                parsed_action = json.loads(fixed)
                if parsed_action.get("tool") != "error":
                    result["action"] = parsed_action
                    console.print("[dim yellow]Fixed malformed ACTION JSON[/dim yellow]")
            except json.JSONDecodeError:
                console.print(
                    f"[dim red]Could not parse ACTION JSON: {raw_action}[/dim red]"
                )

    # Extract ANSWER — everything after ANSWER: to end of string
    answer_match = re.search(r"ANSWER:\s*(.+)", text, re.DOTALL)
    if answer_match:
        result["answer"] = answer_match.group(1).strip()

    return result


def run_agent(question: str, repo_path: str) -> None:
    """
    The ReAct loop: Think → Act → Observe → repeat until Answer.

    The conversation_history list is the agent's working memory.
    Every thought, action, and observation is appended to it so the
    model always sees the full picture of what has happened so far.
    This is what makes multi-step reasoning possible — the model isn't
    starting fresh each iteration, it's continuing a conversation.
    """
    system_prompt = AGENT_SYSTEM_PROMPT.format(
        tool_descriptions=build_tool_descriptions()
    )

    # Seed the conversation with the user's question
    conversation_history = [
        {"role": "user", "content": f"Question: {question}"}
    ]

    for iteration in range(1, MAX_ITERATIONS + 1):
        console.print(f"[dim]── step {iteration} of {MAX_ITERATIONS} ──[/dim]")

        # Ask the model what to do next, passing full history each time
        raw_response = call_agent_step(conversation_history, system_prompt)

        conversation_history.append({
            "role": "assistant",
            "content": raw_response
        })

        parsed = parse_response(raw_response)

        if parsed["thought"]:
            console.print(
                f"[dim yellow]Thought:[/dim yellow] {parsed['thought']}\n"
            )

        if parsed["answer"]:
            console.print(Panel(
                parsed["answer"],
                title="[bold cyan]Final Answer[/bold cyan]",
                subtitle="[dim]Based on gathered context[/dim]",
                border_style="cyan"
            ))
            return

        if parsed["action"]:
            tool_name = parsed["action"].get("tool", "")
            tool_args = parsed["action"].get("args", {})

            console.print(
                f"[dim cyan]Action:[/dim cyan] {tool_name} "
                f"[dim]with args {tool_args}[/dim]\n"
            )

            observation = run_tool(tool_name, repo_path, tool_args)

            if len(observation) > 6000:
                observation = (
                    observation[:6000]
                    + "\n\n[... output truncated — ask about specific files "
                    "or sections if you need more detail ...]"
                )

            # Show a preview of the observation in the terminal
            preview = observation[:300]
            if len(observation) > 300:
                preview += "..."
            console.print(
                f"[bold green]Observation:[/bold green] {preview}\n"
            )
            conversation_history.append({
                "role": "user",
                "content": f"OBSERVATION:\n{observation}"
            })

        else:
            console.print(
                "[yellow]Could not parse a valid action. Nudging model...[/yellow]"
            )
            conversation_history.append({
                "role": "user",
                "content": (
                    "Your last response did not contain a valid ACTION or ANSWER. "
                    "No tool was run and you have no new information. "
                    "Do NOT assume or invent any observations. "
                    "Please output a valid ACTION using one of the available tools, "
                    "or output ANSWER if you truly have enough information already gathered."
                )
            })

    # Exhausted all iterations without a final answer
    console.print(
        f"[bold red]Agent reached the maximum of {MAX_ITERATIONS} steps "
        "without producing a final answer. "
        "Try asking a more specific question.[/bold red]"
    )