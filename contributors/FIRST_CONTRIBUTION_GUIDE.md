# 🎓 Student Contributor Guide: Your First PR

A beginner-friendly guide specifically for students making their first open source contribution.

---

## Why Contribute?

✅ **Build Your Portfolio** — Show employers you can collaborate on real projects  
✅ **Learn Best Practices** — From experienced mentors in code review  
✅ **Help Others** — Your improvements help thousands of students  
✅ **Community Recognition** — Get acknowledged in CONTRIBUTORS.md  
✅ **Make Friends** — Meet other learners and developers  

---

## Before Your First Contribution

### Time Commitment
- **Small contribution** (typo fix, clarification): 30 minutes - 1 hour
- **Medium contribution** (add exercises, improve lesson): 2-4 hours
- **Large contribution** (new lesson, new project): 4-8 hours

### What You Need
- Computer with Windows/Mac/Linux
- GitHub account (free at github.com)
- Git installed (instructions in [GIT_TROUBLESHOOTING.md](GIT_TROUBLESHOOTING.md))
- Text editor or VS Code (free)
- Python 3.10+ (for testing notebooks)

### Knowledge Needed
- **Absolute minimum:** Can read and write Python, understand git basics
- **Helpful:** Understand the module you're improving
- **Not required:** Advanced Git knowledge (we'll help!)

---

## The Student Workflow (Step-by-Step)

### Step 1: Find Something to Improve (15 min)

**Beginner Ideas:**
1. Go through Module 01 or 02
2. Find where YOU got confused
3. Write better explanations/examples
4. Create exercises based on your struggles
5. Fix typos you notice

**Or ask yourself:**
- "Where did I get stuck?"
- "What example would have helped me?"
- "What does the lesson need that it doesn't have?"

**Don't:**
- Worry about finding the "perfect" thing
- Feel like you must fix something huge
- Overthink it — small contributions are valuable!

### Step 2: Fork the Repository (5 min)

1. Go to https://github.com/MarkGPT-LLM-Curriculum
2. Click **Fork** button (top right)
3. You now have your own copy!

### Step 3: Clone to Your Computer (5 min)

```bash
# Open terminal/command prompt
cd Desktop  # or where you want to store it

# Replace YOUR-USERNAME with your GitHub username
git clone https://github.com/YOUR-USERNAME/MarkGPT-LLM-Curriculum.git

# Go into the folder
cd MarkGPT-LLM-Curriculum
```

### Step 4: Create Your Contributor Folder (5 min)

```bash
# Create your workspace folder
mkdir contributors/YOUR-GITHUB-USERNAME
cd contributors/YOUR-GITHUB-USERNAME

# Create the module structure you're working on
# Example: if improving Module 02 lessons
mkdir -p module-02/lessons
```

### Step 5: Copy & Improve Content (1-2 hours)

```bash
# Copy the lesson you want to improve INTO your folder
# Example: copy from the actual curriculum
# On Windows: copy ..\..\..\..\modules\module-02\lessons\pandas_basics.ipynb module-02\lessons\
# On Mac/Linux: cp ../../../modules/module-02/lessons/pandas_basics.ipynb module-02/lessons/

# Now open it and improve it!
# In VS Code: Open the notebook and edit
```

**What to check:**
- [ ] All code cells run without errors
- [ ] Explanations are clear for beginners
- [ ] Examples work correctly
- [ ] No hardcoded file paths

### Step 6: Test Your Work (30 min)

```bash
# Open Jupyter notebook
jupyter notebook

# Or in VS Code:
# Right-click on .ipynb file → Open with... → Jupyter

# Test:
# - Run each cell (make sure no errors)
# - Check outputs are correct
# - Verify visualizations look good
```

### Step 7: Make Your First Commit (10 min)

```bash
# Check what changed
git status

# Add your changes
git add .

# Commit with a clear message
git commit -m "feat(module-02): improve pandas lesson with more examples"
```

**Good commit messages:**
- `feat(module-02): add 5 pandas practice problems`
- `fix(module-03): correct gradient formula in explanation`
- `docs: add troubleshooting section for common errors`

**Bad commit messages:**
- `update` ❌
- `fixes` ❌
- `stuff` ❌

### Step 8: Push to Your Fork (5 min)

```bash
# Send your work to GitHub
git push -u origin main
```

### Step 9: Create a Pull Request (10 min)

1. Go to your fork on GitHub
2. You'll see a banner saying "Compare & pull request"
3. Click it
4. Fill in:
   - **Title:** `feat(module-02): improve pandas lesson with more examples`
   - **Description:** What did you improve and why?

**Good PR description:**
```
## What I improved
I found the pandas lesson confusing because I didn't have enough practice 
problems. I added 5 additional examples using real datasets.

## What I tested
- All code cells run without errors
- Examples work with Python 3.10
- Notebook completes in under 3 minutes

## Files changed
- contributors/YOUR-USERNAME/module-02/lessons/
```

5. Click **Create Pull Request**

### Step 10: Wait for Review (1-7 days)

Maintainers will:
- Review your work
- Ask questions or request changes
- Eventually approve and merge

**If they ask for changes:**
```bash
# Make the changes they requested
# Then:
git add .
git commit -m "fix: address review feedback on pandas examples"
git push origin main

# The PR updates automatically!
```

---

## Common Mistakes (& How to Avoid Them)

### ❌ Mistake 1: "I can't find the original file to copy"

**Fix:**
```bash
# From your contributor folder, list modules
dir ..\..\..\..\modules\  # Windows
ls ../../../modules/      # Mac/Linux

# Or use GUI file explorer
# Navigate to: modules/module-02/lessons/
```

### ❌ Mistake 2: "I committed to main instead of creating a feature branch"

**Fix:**
```bash
git checkout -b feat/my-improvement
# Your commits are safe now!
```

### ❌ Mistake 3: "The notebook cells fail because of a missing import"

**Fix:**
```python
# Add import at TOP of notebook
import numpy as np
import pandas as pd
# etc.
```

### ❌ Mistake 4: "I have a hardcoded filepath like C:/Users/..."

**Fix:**
```python
# Wrong:
df = pd.read_csv('C:/Users/MyName/Desktop/data.csv')

# Right:
from pathlib import Path
data_path = Path(__file__).parent / 'data.csv'
df = pd.read_csv(data_path)
```

### ❌ Mistake 5: "My PR title is too vague"

**Fix:**
```
Wrong: "update module 2"
Right: "feat(module-02): add 5 pandas practice problems with solutions"
```

---

## Success Check list

Before clicking "Create Pull Request", verify:

- [ ] I've read [CONTRIBUTORS_GUIDE.md](CONTRIBUTORS_GUIDE.md)
- [ ] My contribution is isolated in `contributors/YOUR-USERNAME/`
- [ ] All notebook cells run without errors
- [ ] No hardcoded file paths
- [ ] File is in correct module folder
- [ ] Commit message is clear and follows conventions
- [ ] PR title describes what changed
- [ ] PR description explains why it matters
- [ ] I've tested everything thoroughly

---

## After Your PR Merges 🎉

Once maintainers merge your PR:

1. **You're in CONTRIBUTORS.md!** Congrats!
2. Your improvement helps thousands of students
3. Your GitHub profile shows the contribution
4. You can make more contributions!

---

## Getting Unstuck

**It didn't work?** That's normal! Here are resources:

- **[CONTRIBUTORS_GUIDE.md](CONTRIBUTORS_GUIDE.md)** — Full step-by-step guide
- **[GIT_TROUBLESHOOTING.md](GIT_TROUBLESHOOTING.md)** — Git issues
- **[BEST_PRACTICES.md](../BEST_PRACTICES.md)** — Quality standards
- **Open an Issue** on GitHub describing your problem
- **Email:** iwstechnical@gmail.com

**Common questions:**
- **"How long does review take?"** → Usually 3-7 days
- **"What if they ask me to change things?"** → Make the changes, commit, push. The PR updates!
- **"Can my PR be rejected?"** → Yes, but that's okay — we'll help improve it!
- **"Should I work on multiple things?"** → Start with ONE, merge it, then do the next

---

## Next Steps After Your First PR

Once you've made one contribution, you can:

1. ✅ Make more improvements to same module
2. ✅ Work on a different module
3. ✅ Build a mini-project
4. ✅ Create an exercise set
5. ✅ Become a "regular contributor" with multiple merged PRs

---

## You've Got This! 🚀

Remember:
- Small contributions count
- We're here to help
- Every student contribution matters
- You're not alone — reach out if stuck

**Ready to start? Go to [CONTRIBUTORS_GUIDE.md](CONTRIBUTORS_GUIDE.md) and follow Step 1!**
