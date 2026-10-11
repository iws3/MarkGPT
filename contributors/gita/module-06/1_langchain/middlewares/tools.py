"""Tools for the reels assistant. Two are read-only, two have side effects."""

from langchain_core.tools import tool

# A fake "database" so the lab runs without a real backend.
REEL_STATS = {
    "101": {"title": "Street food tour", "views": 18200, "likes": 2310, "shares": 144},
    "102": {"title": "Morning routine", "views": 5400, "likes": 610, "shares": 23},
}
TRENDING = {
    "dance": ["#dancechallenge", "#afrobeats", "#trending", "#fyp"],
    "food": ["#streetfood", "#foodie", "#recipe", "#fyp"],
}

# Real side effects would go here. We record them so tests can check them.
SCHEDULED: list[dict] = []
DELETED: list[str] = []


@tool
def get_trending_hashtags(topic: str) -> str:
    """Return trending hashtags for a reel topic such as 'dance' or 'food'.

    Args:
        topic: The reel topic, one word.
    """
    tags = TRENDING.get(topic.lower().strip())
    return ", ".join(tags) if tags else "No trending data for that topic."


@tool
def get_reel_stats(reel_id: str) -> str:
    """Return views, likes and shares for a reel. Read-only.

    Args:
        reel_id: The numeric reel id, for example '101'.
    """
    stats = REEL_STATS.get(reel_id.strip())
    if stats is None:
        # Raising on purpose: handle_tool_errors (middleware) catches this.
        raise ValueError(f"No reel with id {reel_id}")
    return f"{stats['title']}: {stats['views']} views, {stats['likes']} likes, {stats['shares']} shares"


@tool
def schedule_reel(title: str, caption: str, publish_time: str) -> str:
    """Schedule a reel for publishing. This has a real effect: it will go public.

    Args:
        title: Short reel title.
        caption: The caption text.
        publish_time: When to publish, for example '2026-10-12 18:00'.
    """
    SCHEDULED.append({"title": title, "caption": caption, "publish_time": publish_time})
    return f"Scheduled '{title}' for {publish_time}."


@tool
def delete_reel(reel_id: str) -> str:
    """Permanently delete a reel. This cannot be undone.

    Args:
        reel_id: The numeric reel id.
    """
    DELETED.append(reel_id)
    return f"Deleted reel {reel_id}."