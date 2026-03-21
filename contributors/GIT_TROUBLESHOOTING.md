# Git Troubleshooting for Contributors

Got stuck with Git? This guide covers common issues and solutions.

---

## Installation & Setup

### Issue: Git command not found

**Solution:**
```bash
# Check if git is installed
git --version

# If not installed:
# Windows: Download from https://git-scm.com/download/win
# Mac: brew install git
# Linux: sudo apt-get install git (Ubuntu/Debian)
```

### Issue: Git not configured with your name

**Solution:**
```bash
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"

# Verify it worked
git config --list
```

---

## Cloning & Forking

### Issue: "Permission denied" when cloning

**Cause:** SSH key not set up

**Solution:**
```bash
# Use HTTPS instead of SSH
git clone https://github.com/YOUR-USERNAME/MarkGPT-LLM-Curriculum.git

# Or set up SSH:
# 1. Go to https://github.com/settings/keys
# 2. Add new SSH key
# 3. Run: ssh-keygen -t ed25519
# 4. Add the public key to GitHub
```

### Issue: "Repository not found"

**Cause:** You cloned the original, not your fork

**Solution:**
```bash
# Check your remote
git remote -v

# If it shows someone else's repo:
# Delete and re-clone your fork
cd ..
rm -rf MarkGPT-LLM-Curriculum
git clone https://github.com/YOUR-USERNAME/MarkGPT-LLM-Curriculum.git
```

---

## Branches

### Issue: "fatal: Not a valid object name"

**Cause:** Branch doesn't exist locally

**Solution:**
```bash
# Fetch from remote
git fetch origin

# List all branches
git branch -a

# Switch to branch
git checkout branch-name
```

### Issue: Accidentally committed to `main` instead of a feature branch

**Solution:**
```bash
# Create a new branch with your commits
git branch your-feature-branch

# Reset main to upstream
git checkout main
git reset --hard upstream/main

# Switch to your feature branch
git checkout your-feature-branch

# Continue working (your commits are safe!)
```

### Issue: "Your branch is ahead of 'origin/main' by 5 commits"

**Cause:** Normal! You've made commits that aren't pushed yet

**Solution:**
```bash
git push origin your-branch-name
```

---

## Commits

### Issue: Typo in commit message

**Solution (if not pushed yet):**
```bash
# For last commit
git commit --amend -m "correct message here"

# If already pushed
git commit --amend -m "correct message here"
git push origin your-branch --force-with-lease
```

### Issue: "nothing to commit, working tree clean"

**Cause:** You haven't made changes, or they're not staged

**Solution:**
```bash
# Check what changed
git status

# Stage your changes
git add .
git add path/to/specific/file

# Then commit
git commit -m "your message"
```

### Issue: Want to undo your last commit

**Solution:**
```bash
# Keep your changes (safe!)
git reset --soft HEAD~1

# Discard the changes
git reset --hard HEAD~1
```

### Issue: Want to combine multiple commits into one

**Solution:**
```bash
# Count how many commits back
git log --oneline
# Shows: abc123 commit 3
#        def456 commit 2 
#        ghi789 commit 1

# Interactive rebase (combine last 3 commits)
git rebase -i HEAD~3

# In editor: change "pick" to "squash" for commits to combine
# Save and close editor
# Edit final commit message
# Done!
```

---

## Syncing with Upstream

### Issue: Your fork is behind the original repo

**Cause:** Others have made changes to the main repository

**Solution:**
```bash
# Fetch from upstream
git fetch upstream

# Rebase your work on top
git rebase upstream/main

# If you have conflicts, resolve them manually, then:
git add .
git rebase --continue

# Update your fork
git push origin main --force-with-lease
```

### Issue: Merge conflicts!

**Cause:** You and someone else edited the same lines

**Solution:**
```
1. Open the conflicted file in editor
2. Look for sections marked like:
   <<<<<<< HEAD
   your changes
   =======
   their changes
   >>>>>>> branch-name
3. Manually choose which to keep (or combine both!)
4. Delete the conflict markers (<<<, ===, >>>)
5. Save the file
```

Then:
```bash
git add path/to/resolved/file
git rebase --continue
# or
git commit -m "resolve merge conflicts"
```

---

## Pushing & Pull Requests

### Issue: "fatal: The current branch has no upstream branch"

**Cause:** New branch exists locally but not on GitHub

**Solution:**
```bash
# Push and set upstream in one go
git push -u origin your-branch-name

# Or just push normally next time
git push origin your-branch-name
```

### Issue: Can't push - "access denied"

**Cause:** Your credentials expired or fork permissions issue

**Solution:**
```bash
# Re-authenticate (macOS)
git config --global credential.osxkeychain erase https://github.com

# Or use GitHub CLI
gh auth login

# Then try pushing again
git push origin your-branch-name
```

### Issue: "Your push would publish private commits"

**Cause:** You have private email setting enabled in GitHub

**Solution:**
```bash
# Use public email from GitHub settings
git config --global user.email "your-public-github-email@example.com"

# Or turn off private email in GitHub settings
# https://github.com/settings/emails
```

---

## Debugging

### Issue: "Where did my changes go?"

**Solution:**
```bash
# Check reflog (history of HEAD changes)
git reflog

# See a specific commit
git show abc123

# Recover deleted branch
git checkout -b recovered-branch 4c3f33a
```

### Issue: Want to see what changed in a commit

**Solution:**
```bash
# See changes in last commit
git show

# See changes in specific commit
git show abc123

# See differences between branches
git diff main your-branch

# See all commits with their messages
git log --oneline --graph --all
```

### Issue: Accidentally deleted a file

**Solution:**
```bash
# If not committed yet
git checkout path/to/file

# If committed but not pushed
git revert abc123  # Revert the deletion commit

# If in older commits
git reset --hard abc123  # Go back to that commit (careful!)
```

---

## Commands Cheat Sheet

```bash
# Setup
git clone https://github.com/YOUR-USERNAME/MarkGPT-LLM-Curriculum.git
git remote add upstream https://github.com/ORIGINAL/MarkGPT-LLM-Curriculum.git

# Workflow
git fetch upstream
git checkout -b feat/your-feature upstream/main
# ... make changes ...
git add .
git commit -m "feat: your change"
git push origin feat/your-feature

# Sync
git fetch upstream
git rebase upstream/main
git push origin feat/your-feature --force-with-lease

# View history
git log --oneline
git status
git diff

# Undo
git reset --soft HEAD~1    # Undo last commit, keep changes
git reset --hard HEAD~1    # Undo last commit, discard changes
git revert abc123          # Create new commit undoing abc123
```

---

## Still Stuck?

- **GitHub Docs:** https://docs.github.com/en/get-started
- **Git Docs:** https://git-scm.com/doc
- **Interactive Git Learning:** https://learngitbranching.js.org/
- **Ask in an Issue:** Create a GitHub issue describing your problem
- **Email Help:** `iwstechnical@gmail.com`

---

*Remember: Git is designed to not lose your work. Even if you feel like you've made a mistake, your changes are usually recoverable!* 🙌
