import time
import typer

from rich.rule import Rule
from agent.agent import run_agent
from rich.panel import Panel
from dotenv import load_dotenv
from rich.console import Console
from gatherer import is_git_repo, get_directory_tree, TOOLS

load_dotenv()
console = Console()
app = typer.Typer()

MAX_RETRIES = 3
RETRY_DELAY = 2

def run(query: str, repo_path: str) -> None:
    """
    Execute one full cycle of the agent: route, gather context, and answer.
    Retries the whole process up to MAX_RETRIES times if it fails.
    """

    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            run_agent(query, repo_path)
            return
        except KeyboardInterrupt:
            console.print("\n[red]Process interrupted by user. Exiting.[/red]")
            return
        except Exception as e:
            last_error = e
            if attempt < MAX_RETRIES:
                console.print(
                    f"[yellow] → attempt {attempt} failed: {e}. "
                    f"Retrying in {RETRY_DELAY}s...[/yellow]"
                )
                time.sleep(RETRY_DELAY)
            else:
                console.print(
                    f"[bold red]Error:[/bold red] Failed after {MAX_RETRIES} attempts. "
                    f"Last error: {last_error}"
                )


def query_loop(repo_path: str) -> bool:
    """
    Main loop: ask the user for a question, run the agent, and repeat.
    """
    while True:
        try:
            user_query = console.input("[bold magenta]❯ [/bold magenta]").strip()
        except KeyboardInterrupt:
            console.print("\n[yellow]Goodbye![/yellow]")
            return False
        
        if user_query.lower() == "back":
            console.print("[yellow]Going back to repository selection...[/yellow]")
            return True

        if user_query.lower() in {"exit", "quit"}:
            console.print("[yellow]Goodbye![/yellow]")
            return False
        
        run(user_query, repo_path)

@app.command()
def main():
    """Entry point of the application. Handles repository selection and starts the query loop."""
    console.clear()
    console.print(Panel.fit(
        "[bold cyan]Welcome to the Repository Reporter![/bold cyan]\n\n" \
        "[dim]Monitoring code changes and commit intent[/dim]\n\n",
        border_style="cyan",
        title="Repo Reporter",
    ))

    while True:
        try:
            path_input = console.input("[bold magenta]Enter the path to a git repository (or 'exit' to quit): [/bold magenta]").strip()

            if not path_input:
                continue

            if not is_git_repo(path_input):
                console.print(f"[red]Error: '{path_input}' does not have a valid git repository. Please try again.[/red]")
                continue

            console.print(
                f"\n[bold green]✓[/bold green] Found git repo: "
                f"[bold underline]{path_input}[/bold underline]\n"
            )
            console.print(get_directory_tree(path_input))
            console.print(Rule(style="dim"))
            console.print("[dim]Type your question, 'back' to change directory, or 'exit' to quit.[/dim]\n")

            should_continue = query_loop(path_input)
            if not should_continue:
                break

        except KeyboardInterrupt:
            console.print("\n[red]Process interrupted by user. Exiting.[/red]")
            break

if __name__ == "__main__":
    main()