from gatherer import (
    get_directory_tree,
    get_recent_commits,
    get_uncommitted_changes,
    get_file_content,
    get_last_diff,
    read_readme,
)

"""
Each tool has:
fn - the callable
args - the arguments it takes above the repo path
desc - what the tool reads to decide whether to use it
"""

TOOLS = {
    "directory_tree": {
        "fn": get_directory_tree,
        "args": [],
        "desc": "Lists all tracked files in the repo. Use to understand project structure or find which files exist."
    },
    "recent_commits": {
        "fn": get_recent_commits,
        "args": [],
        "desc": "Returns the last 20 commits with author, date, message, and changed files. Use for questions about history or recent activity."
    },
    "uncommitted_changes": {
        "fn": get_uncommitted_changes,
        "args": [],
        "desc": "Returns staged and unstaged diffs. Use when asked about current work in progress."
    },
    "last_diff": {
        "fn": get_last_diff,
        "args": [],
        "desc": "Returns the full patch of the most recent commit. Use when asked what the last commit changed."
    },
    "read_file": {
        "fn": get_file_content,
        "args": ["filepath", "start_line", "end_line"],
        "desc": (
            "Reads content of a specific file. "
            "Provide filepath (relative to repo root). "
            "Optionally provide start_line and end_line to read only a section. "
            "Use when you need to understand what specific code does."
        )
    },
    "read_readme": {
        "fn": read_readme,
        "args": ["filepath"],
        "desc": (
            "Reads the README file of the project. "
            "Provide filepath (relative to repo root, e.g. 'README.md'). "
            "Use this to get an overview of the project, its purpose, and how to use it."
        )
    },

}

def run_tool (name: str, repo_path: str, args: dict) -> str:
    """
    Execute a tool by name, passing repo_path and any extra args.
    Returns the tool's output as a string, or an error message.
    """

    if name not in TOOLS:
        return f"Error: Unknown tool'{name}'. Available tools: {', '.join(TOOLS.keys())}"
    
    tool = TOOLS[name]

    try:
        if name == "read_file":
            filepath = args.get("filepath")
            if not filepath:
                return "Error: read_file requires a 'filepath' argument."
            
            start_line = args.get("start_line")
            end_line = args.get("end_line")

            content = tool["fn"](repo_path, filepath)

            if start_line or end_line:
                lines = content.splitlines()
                start = max(0, (int(start_line) - 1) if start_line else 0)
                end = int(end_line) if end_line else len(lines)
                content = "\n".join(lines[start:end])
                content = f"[Lines {start+1}-{end} of {filepath}]\n\n{content}"

            return content
        else:
            return tool["fn"](repo_path)
    except Exception as e:
        return f"Error running tool '{name}': {e}"
