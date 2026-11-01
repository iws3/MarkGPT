# Connecting Claude Desktop to Filesystem, GitHub, Notion & Gmail with MCP
## What is an MCP
MCP's(model context protocol):it ia a standard for connecting LLM's to Ai applications. it is used to connect tools to an LLM using thesame mechanism

## The goal
Today I set out to give Claude Desktop hands: access to my local files, my GitHub account, my Notion workspace and my Gmail, all through the Model Context Protocol (MCP). I started from the official modelcontextprotocol/servers repository. It turned into a lesson in how fast the MCP ecosystem moves, and in how unforgiving JSON is.

## Step 1: Reality check on the official repo

The first surprise: the repo no longer contains most of the servers I expected. It is now a small set of reference implementations (Filesystem, Git, Fetch, Memory, Time and a few others). The GitHub server has been archived, and Gmail and Notion were never there. So each tool needed a different route:

- Filesystem: official reference server, run with npx.
- GitHub: GitHub's own maintained server (Docker image or hosted endpoint). The old npm package is deprecated.
- Notion: Notion's hosted MCP server at mcp.notion.com, added via Settings > Connectors.
- Gmail: Anthropic's built-in Google Workspace connector (Settings > Connectors), no config file needed.

## Step 2: Editing claude_desktop_config.json

In Claude Desktop, Settings > Developer > Edit Config opens the config file. I added the Filesystem server pointing at a notes folder. On Windows, paths need escaped backslashes, and npx may need to be launched through cmd /c.

```json
"mcpServers": {
  "filesystem": {
    "command": "npx",
    "args": ["-y", "@modelcontextprotocol/server-filesystem", "E:\\ityiyg\\notes"]
  },
  "github": {
    "command": "cmd",
    "args": ["/c", "npx", "-y", "@modelcontextprotocol/server-github"],
    "env": { "GITHUB_PERSONAL_ACCESS_TOKEN": "<your-token>" }
  }
}
```

## Step 3: The mistakes (a debugging log)

My editor lit up with red squiggles. Here is everything that was wrong, in the order I hit it:

- Unescaped backslashes: "E:\ityiyg\notes" is invalid JSON. Every backslash must be doubled.
- A leftover placeholder path (/Users/you/Projects) copied from an example. It does not exist on Windows.
- A trailing comma after the last property in the file. JSON does not allow it.
- The github entry sitting outside mcpServers, plus a stray closing brace. Every server must be a sibling inside the single mcpServers object.
- Overwriting the whole file would have deleted my existing preferences block. Only the mcpServers block should be replaced.

## Why cmd /c instead of npx?

On Windows, npx is a batch script (npx.cmd), and Claude Desktop may fail to launch it directly. Wrapping it as cmd /c npx ... runs it through the command prompt. It is still npx doing the work, just launched through a shell.

## The security lesson

The biggest lesson of the day: I pasted a live GitHub personal access token into a chat while debugging. A token in a chat log is a leaked token. The fix is simple: revoke it immediately in GitHub (Settings > Developer settings > Personal access tokens), create a new fine-grained token limited to specific repos, and keep it only in the local config file. Never paste secrets into chats, screenshots or commits.

## What I can do once it all works

- Filesystem: summarize notes, rename and organize files, scaffold project folders.
- GitHub: triage issues, review pull requests, draft release notes.
- Notion: create meeting notes, query task databases, build project plans.
- Gmail: summarize unread mail, find invoices, draft replies.