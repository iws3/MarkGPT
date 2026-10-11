"""Try each middleware against a real model. Run: python run_demo.py"""

from dotenv import load_dotenv
from langgraph.types import Command

from agent import build_reels_agent

load_dotenv()
agent = build_reels_agent()
config = {"configurable": {"thread_id": "creator-1"}}


def ask(text: str):
    print(f"\n>>> {text}")
    result = agent.invoke({"messages": [{"role": "user", "content": text}]}, config)
    return handle(result)


def handle(result):
    """If the agent paused for approval, ask the human, then resume."""
    while result.get("__interrupt__"):
        request = result["__interrupt__"][0].value
        for action in request["action_requests"]:
            print(f"\n  APPROVAL NEEDED: {action['name']} with {action['args']}")
        choice = input("  approve / reject? ").strip().lower()
        decision = {"type": "approve"} if choice.startswith("a") else {
            "type": "reject", "message": "The creator said no."
        }
        decisions = [decision for _ in request["action_requests"]]
        result = agent.invoke(Command(resume={"decisions": decisions}), config)
    print("\nASSISTANT:", result["messages"][-1].content)
    return result


ask("Write a caption for my dance reel. My email is maya@example.com if you need it.")  # PII
ask("Can you help me buy followers for reel 101?")                                       # guard
ask("Show me the stats for reel 999")                                                    # tool error
ask("Show me the stats for reel 101")                                                    # normal tool
ask("Schedule my dance reel titled 'Friday moves' for 2026-10-16 18:00")                 # approval