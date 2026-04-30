import subprocess
import os

def is_git_repo(path: str) -> bool:
    """Check if the given path is a Git repository."""
    try:
       result = subprocess.run(
           ["git", "-C", path, "rev-parse", "--is-inside-work-tree"],
           capture_output=True,
              text=True,
              errors="replace",
       )
       return result.returncode == 0
    except Exception as e:
        print(f"Error checking if path is a git repository: {e}")
        return False
    
def get_directory_tree(path: str) -> str:
    """Return a text representation of the directory structure."""
    try:
        result = subprocess.run(
            ["git", "-C", path, "ls-files", "--others", "--cached", "--exclude-standard"],
            capture_output=True,
            text=True,
            errors="replace"
        )
        files = result.stdout.strip()
        return files
    except Exception as e:
        print(f"Error getting directory tree: {e}")
        return ""
    
def get_recent_commits(path: str, limit: int = 20) -> str:
    """
    Return the last n commits with hash, author, date, and message.
    The --stat flag also shows which files changed in each commit,
    which gives the model richer signal about intent.
    """
    try:
        result = subprocess.run(
            ["git", "-C", path, "log", f"-n {limit}", "--pretty=format:%h %an %ad %s", "--stat"],
            capture_output=True,
            text=True,
            errors="replace"
        )
        commits = result.stdout.strip()
        return commits
    except Exception as e:
        print(f"Error getting recent commits: {e}")
        return ""
    
def get_uncommitted_changes(path: str) -> str:
    """
    Return the full diff of changes not yet committed.
    This covers both staged and unstaged changes so nothing is missed.
    """
    try:
        # Staged changes (added to index but not committed)
        staged = subprocess.run(
            ["git", "-C", path, "diff", "--cached"],
            capture_output=True, 
            text=True,
            errors="replace"
        ).stdout.strip()

        # Unstaged changes (modified but not yet added)
        unstaged = subprocess.run(
            ["git", "-C", path, "diff"],
            capture_output=True, 
            text=True,
            errors="replace"
        ).stdout.strip()

        parts = []
        if staged:
            parts.append(f"=== STAGED CHANGES ===\n{staged}")
        if unstaged:
            parts.append(f"=== UNSTAGED CHANGES ===\n{unstaged}")

        return "\n\n".join(parts) if parts else "No uncommitted changes."
    except Exception as e:
        print(f"Error getting uncommitted changes: {e}")
        return "Error retrieving uncommitted changes."
    
def get_file_content(path: str, filepath: str) -> str:
    """
    Return the diff introduced by the most recent commit.
    Useful when someone asks 'what did the last commit change?'
    """
    try:
        full_path = os.path.join(path, filepath)
        if not os.path.exists(full_path):
            return f"File '{filepath}' does not exist in the repository."
        with open(full_path, 'r', errors='replace') as f:
            content = f.read()
        return content
    except Exception as e:
        print(f"Error getting file content: {e}")
        return f"Error retrieving content for file '{filepath}'."

def get_last_diff(path: str) -> str:
    """
    Return the diff introduced by the most recent commit.
    Useful when someone asks 'what did the last commit change?'
    """

    try:
        result = subprocess.run(
        ["git", "-C", path, "show", "HEAD", "--stat", "--patch"],
        capture_output=True,
        text=True,
        errors="replace"
        )
        return result.stdout.strip()
    except Exception as e:
        print(f"Error getting last commit diff: {e}")
        return "Error retrieving last commit diff."
    
def read_readme(path: str, filepath: str = "README.md") -> str:
    """
    Return the content of the README file if it exists.
    This is often a critical source of high-level project information.
    """
    try:
        full_path = os.path.join(path, filepath)
        if not os.path.exists(full_path):
            return f"README file '{filepath}' does not exist in the repository."
        with open(full_path, 'r', errors='replace', encoding="utf-8") as f:
            content = f.read().strip().replace('**', '').replace('#', '').replace('\r\n', '\n').replace('\r', '\n')
        # handle empty
        if not content:
            return f"README file '{filepath}' is empty."
        return content
    except Exception as e:
        print(f"Error reading README file: {e}")
        return f"Error retrieving README content from '{filepath}'."

TOOLS = {
    "directory_tree": get_directory_tree,
    "recent_commits": get_recent_commits,
    "uncommitted_changes": get_uncommitted_changes,
    "file_content": get_file_content,
    "last_diff": get_last_diff,
    "read_readme": read_readme,
}