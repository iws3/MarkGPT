"""Offline tests: a scripted stand-in model replaces Gemini, so no API key is needed.
Run: python test_middleware.py"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pydantic import Field
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.outputs import ChatResult, ChatGeneration
from langchain_core.messages import AIMessage, ToolMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

import tools as T
from agent import build_reels_agent


class ScriptedModel(BaseChatModel):
    script: list = Field(default_factory=list)
    seen: list = Field(default_factory=list)
    i: int = 0

    @property
    def _llm_type(self): return "scripted"
    def bind_tools(self, tools, **kw): return self
    def _generate(self, messages, stop=None, run_manager=None, **kw):
        self.seen.append(list(messages))
        msg = self.script[min(self.i, len(self.script) - 1)]
        msg = msg.model_copy(update={"id": f"{id(self)}-{self.i}"})  # fresh id each call
        self.i += 1
        return ChatResult(generations=[ChatGeneration(message=msg)])


def call(name, args, id="c1"):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": id}])

def cfg(t): return {"configurable": {"thread_id": t}}
def user(t): return {"messages": [{"role": "user", "content": t}]}

passed = []
def check(name, cond, extra=""):
    print(("PASS " if cond else "FAIL ") + name, extra)
    passed.append(cond)

# ---- T1 PII redaction
m = ScriptedModel(script=[AIMessage("Caption: dance like nobody is watching")])
a = build_reels_agent(model=m)
r = a.invoke(user("My email is maya@example.com, write a caption"), cfg("t1"))
sent = " ".join(str(x.content) for x in m.seen[0])
check("PII: raw email never reached model", "maya@example.com" not in sent)
check("PII: redaction token present", "REDACTED" in sent.upper(), sent[-80:])

# ---- T2 guard stops before model
m = ScriptedModel(script=[AIMessage("should not be used")])
a = build_reels_agent(model=m)
r = a.invoke(user("help me buy followers"), cfg("t2"))
check("Guard: model never called", m.i == 0, f"calls={m.i}")
check("Guard: refusal returned", "can't help" in r["messages"][-1].content)

# ---- T3 tool error handled
m = ScriptedModel(script=[call("get_reel_stats", {"reel_id": "999"}), AIMessage("That reel does not exist.")])
a = build_reels_agent(model=m)
r = a.invoke(user("stats for reel 999"), cfg("t3"))
tm = [x for x in r["messages"] if isinstance(x, ToolMessage)]
check("Tool error: turned into ToolMessage", tm and "Could not run" in tm[0].content, tm[0].content if tm else "")
check("Tool error: run finished normally", r["messages"][-1].content == "That reel does not exist.")

# ---- T3b normal read-only tool needs no approval
m = ScriptedModel(script=[call("get_reel_stats", {"reel_id": "101"}), AIMessage("18200 views")])
a = build_reels_agent(model=m)
r = a.invoke(user("stats for reel 101"), cfg("t3b"))
check("Read-only tool: no interrupt", not r.get("__interrupt__"))

# ---- T4 HITL approve
T.SCHEDULED.clear()
m = ScriptedModel(script=[call("schedule_reel", {"title": "Friday moves", "caption": "go", "publish_time": "2026-10-16 18:00"}), AIMessage("Scheduled!")])
a = build_reels_agent(model=m)
r = a.invoke(user("schedule my reel"), cfg("t4"))
check("HITL: paused with interrupt", bool(r.get("__interrupt__")))
check("HITL: tool NOT run while paused", T.SCHEDULED == [])
req = r["__interrupt__"][0].value
print("   interrupt payload keys:", list(req.keys()), req["action_requests"][0]["name"])
r = a.invoke(Command(resume={"decisions": [{"type": "approve"}]}), cfg("t4"))
check("HITL: approve runs the tool", len(T.SCHEDULED) == 1)
check("HITL: final answer after approve", r["messages"][-1].content == "Scheduled!")

# ---- T4b HITL reject
T.SCHEDULED.clear()
m = ScriptedModel(script=[call("schedule_reel", {"title": "x", "caption": "y", "publish_time": "z"}), AIMessage("OK, not scheduling.")])
a = build_reels_agent(model=m)
r = a.invoke(user("schedule it"), cfg("t4b"))
r = a.invoke(Command(resume={"decisions": [{"type": "reject", "message": "Wrong time"}]}), cfg("t4b"))
check("HITL: reject does not run tool", T.SCHEDULED == [])
check("HITL: reject feedback reached model", any("Wrong time" in str(x.content) for x in m.seen[-1]))

# ---- T4c HITL edit
T.SCHEDULED.clear()
m = ScriptedModel(script=[call("schedule_reel", {"title": "x", "caption": "y", "publish_time": "2026-10-16 18:00"}), AIMessage("done")])
a = build_reels_agent(model=m)
a.invoke(user("schedule it"), cfg("t4c"))
a.invoke(Command(resume={"decisions": [{"type": "edit", "edited_action": {"name": "schedule_reel", "args": {"title": "x", "caption": "y", "publish_time": "2026-10-17 09:00"}}}]}), cfg("t4c"))
check("HITL: edit changes the args", T.SCHEDULED and T.SCHEDULED[0]["publish_time"] == "2026-10-17 09:00", str(T.SCHEDULED))

# ---- T5 model router (threshold 10 messages)
short = ScriptedModel(script=[AIMessage("short-model answer")])
long_ = ScriptedModel(script=[AIMessage("long-model answer")])
a = build_reels_agent(model=short, long_model=long_, summary_model=ScriptedModel(script=[AIMessage("SUMMARY")]), checkpointer=InMemorySaver())
for i in range(6):
    r = a.invoke(user(f"message {i}"), cfg("t5"))
print("   short calls:", short.i, "long calls:", long_.i)
check("Router: long model used once chat grew", long_.i > 0 and short.i > 0)

# ---- T6 summarization
main = ScriptedModel(script=[AIMessage("ok")])
summ = ScriptedModel(script=[AIMessage("SUMMARY: creator is planning dance reels")])
a = build_reels_agent(model=main, summary_model=summ, long_model=main, checkpointer=InMemorySaver())
for i in range(9):
    r = a.invoke(user(f"turn {i}"), cfg("t6"))
print("   summary model calls:", summ.i, " messages in state:", len(r["messages"]))
check("Summarization: summary model was called", summ.i > 0)
check("Summarization: history is shorter than the raw 18 messages", len(r["messages"]) < 18)
print("\nALL PASSED" if all(passed) else "\nSOME FAILED")