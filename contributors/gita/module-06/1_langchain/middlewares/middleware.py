"""Custom middleware for the reels assistant, written in three different styles."""

from langchain.agents.middleware import (
    AgentMiddleware,
    AgentState,
    after_model,
    before_model,
    hook_config,
    wrap_model_call,
    wrap_tool_call,
)
from langchain_core.messages import AIMessage, ToolMessage


# ---------------------------------------------------------------------------
# Style 1: a class. Use it when the middleware has settings or several hooks.
# Hook: before_agent, which runs once per invoke(), before anything else.
# ---------------------------------------------------------------------------
class PolicyGuardMiddleware(AgentMiddleware):
    """Stop requests that ask the assistant to fake engagement."""

    BLOCKED_PHRASES = ["buy followers", "buy likes", "fake likes", "fake views", "bot views"]

    # Without this line, "jump_to" below is ignored and the model still runs.
    @hook_config(can_jump_to=["end"])
    def before_agent(self, state: AgentState, runtime):
        text = str(state["messages"][-1].content).lower()
        if any(phrase in text for phrase in self.BLOCKED_PHRASES):
            return {
                "messages": [
                    AIMessage(
                        "I can't help with buying followers, likes or views. "
                        "I can help you write captions, pick hashtags or plan posting times instead."
                    )
                ],
                "jump_to": "end",
            }
        return None  # None means: carry on as normal


# ---------------------------------------------------------------------------
# Style 2: decorators. Use them for a single small hook.
# Hooks: before_model and after_model, which run around EVERY model call.
# ---------------------------------------------------------------------------
@before_model
def log_before_model(state: AgentState, runtime):
    print(f"  [before_model] about to call the model with {len(state['messages'])} messages")
    return None


@after_model
def log_after_model(state: AgentState, runtime):
    last = state["messages"][-1]
    calls = getattr(last, "tool_calls", None) or []
    if calls:
        print(f"  [after_model] model asked for tools: {[c['name'] for c in calls]}")
    else:
        print("  [after_model] model gave a final answer")
    return None


# ---------------------------------------------------------------------------
# Style 3: wrap hooks. They receive a `handler` and decide whether, when
# and how many times to call it. Use them for retries, fallbacks, routing.
# ---------------------------------------------------------------------------
@wrap_tool_call
def handle_tool_errors(request, handler):
    """If a tool raises, tell the model what happened instead of crashing the run."""
    try:
        return handler(request)
    except Exception as exc:
        print(f"  [wrap_tool_call] {request.tool_call['name']} failed: {exc}")
        return ToolMessage(
            content=f"Could not run {request.tool_call['name']}: {exc}. Ask the user to check the input.",
            tool_call_id=request.tool_call["id"],
        )


def make_model_router(long_conversation_model, threshold: int = 10):
    """Use the main model for short chats and a different one once the chat grows."""

    @wrap_model_call
    def router(request, handler):
        if len(request.messages) > threshold:
            print(f"  [wrap_model_call] {len(request.messages)} messages, switching model")
            return handler(request.override(model=long_conversation_model))
        return handler(request)

    return router