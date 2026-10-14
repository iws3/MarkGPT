"""Assemble the reels assistant. The middleware list is the interesting part."""

from langchain.agents import create_agent
from langchain.agents.middleware import (
    HumanInTheLoopMiddleware,
    PIIMiddleware,
    SummarizationMiddleware,
)
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import InMemorySaver

from middleware import (
    PolicyGuardMiddleware,
    handle_tool_errors,
    log_after_model,
    log_before_model,
    make_model_router,
)
from tools import delete_reel, get_reel_stats, get_trending_hashtags, schedule_reel

MODEL = "google_genai:gemini-2.5-flash"

SYSTEM_PROMPT = """You are a reels assistant for short-video creators.
Help with captions, hashtags, posting times and reel stats.
Use get_trending_hashtags for hashtags and get_reel_stats for numbers, never guess them.
Only call schedule_reel or delete_reel when the creator clearly asks for it."""


def build_reels_agent(model=None, summary_model=None, long_model=None, checkpointer=None):
    """Every argument can be replaced, which is how the tests inject a fake model."""
    model = model or init_chat_model(MODEL, temperature=0)
    summary_model = summary_model or model
    long_model = long_model or model

    return create_agent(
        model=model,
        tools=[get_trending_hashtags, get_reel_stats, schedule_reel, delete_reel],
        system_prompt=SYSTEM_PROMPT,
        middleware=[
            PolicyGuardMiddleware(),                                   # before_agent: runs before any model hook, wherever it sits
            PIIMiddleware("email", strategy="redact", apply_to_input=True),  # before_model: keep it ahead of anything that logs messages
            log_before_model,                                          # sees redacted text because PII is listed first
            log_after_model,
            handle_tool_errors,                                        # wraps every tool call
            make_model_router(long_model, threshold=10),               # wraps every model call
            SummarizationMiddleware(                                   # before_model: keeps context small
                model=summary_model,
                trigger=("messages", 12),
                keep=("messages", 4),
            ),
            HumanInTheLoopMiddleware(                                  # after_model: pauses before risky tools run
                interrupt_on={
                    "schedule_reel": {"allowed_decisions": ["approve", "edit", "reject"]},
                    "delete_reel": {"allowed_decisions": ["approve", "reject"]},
                    "get_reel_stats": False,
                    "get_trending_hashtags": False,
                }
            ),
        ],
        checkpointer=checkpointer or InMemorySaver(),  # HITL cannot work without one
    )