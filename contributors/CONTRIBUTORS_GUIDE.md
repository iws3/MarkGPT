 # 🤝 MarkGPT Contributors Guide

Welcome! This guide is designed for **beginner and intermediate students** who want to contribute to the MarkGPT LLM Curriculum. Whether you want to fix bugs, improve lessons, add exercises, or enhance documentation, this guide will walk you through the process step by step.

---

## Table of Contents

1. [Before You Start](#before-you-start)
2. [Step-by-Step Contribution Workflow](#step-by-step-contribution-workflow)
3. [Setting Up Your Contributor Folder](#setting-up-your-contributor-folder)
4. [Working on Your Contributions](#working-on-your-contributions)
5. [Ways to Contribute](#ways-to-contribute)
6. [Contribution Best Practices](#contribution-best-practices)
7. [Creating a Pull Request](#creating-a-pull-request)
8. [Getting Help](#getting-help)

---

## Before You Start

### Prerequisites

- **GitHub Account**: Create one at [github.com](https://github.com) if you don't have one
- **Git Installed**: Download from [git-scm.com](https://git-scm.com)
- **Python 3.10+**: Required to work with Jupyter notebooks
- **A Code Editor**: VS Code, PyCharm, or your favorite editor
- **Approximately 30 minutes**: To set up and make your first contribution

### Knowledge Requirements

- Basic command line skills (can follow simple `cd`, `git` commands)
- Comfort reading and trying code
- **No advanced Git knowledge required** — we'll walk through the essentials

---

## Step-by-Step Contribution Workflow

### **Step 1: Fork the Repository**

Forking creates your own copy of the MarkGPT repository on GitHub.

1. Go to [the MarkGPT repository](https://github.com/YOUR-LINK-HERE)
2. Click the **Fork** button (top-right corner)
3. This creates `your-username/MarkGPT-LLM-Curriculum` in your GitHub account
4. You now have permission to make changes without affecting the original project

### **Step 2: Clone Your Forked Repository**

Cloning downloads the repository to your computer.

```bash
# Open your terminal/command prompt and navigate to where you want the folder
cd Desktop  # or any folder where you want to work

# Clone your forked repository (replace YOUR-USERNAME with your actual GitHub username)
git clone https://github.com/YOUR-USERNAME/MarkGPT-LLM-Curriculum.git

# Navigate into the project folder
cd MarkGPT-LLM-Curriculum
```

### **Step 3: Create Your Contributor Folder**

Inside the `contributors/` folder, create a folder with your GitHub username:

```bash
# (Make sure you're in the MarkGPT-LLM-Curriculum folder)

# Create your contributor folder with your username
mkdir contributors/YOUR-GITHUB-USERNAME

# Navigate to your folder
cd contributors/YOUR-GITHUB-USERNAME
```

**Example**: If your GitHub username is `alice-wonder`, your folder structure looks like:

```
contributors/
├── .template/          # Reference structure (read-only)
└── alice-wonder/       # Your working folder
    ├── module-01/
    ├── module-02/
    ├── module-03/
    └── ... (other modules as needed)
```

### **Step 4: Set Up the Module Structure You'll Work On**

Create the exact folder structure that mirrors the modules you want to contribute to.

For example, if you want to improve lessons in **Module 02**, create:

```bash
mkdir -p module-02/lessons
mkdir -p module-02/exercises
mkdir -p module-02/projects
```

Or if you want to add exercises in **Module 03**:

```bash
mkdir -p module-03/exercises
```

**Reference the `.template/` folder** to see the expected structure for each module.

### **Step 5: Make Your First Commit**

Tell Git that you're starting to work on something:

```bash
# Make sure you're in your contributor folder
cd contributors/YOUR-GITHUB-USERNAME

# Tell Git to track this folder
git add .

# Make your first commit
git commit -m "feat: initialize contributor workspace for YOUR-USERNAME"
```

### **Step 6: Work on Your Contribution**

Copy the relevant lesson/exercise/project from `modules/` into your contributor folder, make your improvements, test them (if applicable), and save them.

**Example: Improving a Lesson**

```bash
# Your contributor folder: contributors/YOUR-USERNAME/module-02/lessons/

# You copy improved_lesson.ipynb here
# You make edits, test them
# Then commit your work

git add .
git commit -m "fix: improve module-02 lesson on NumPy arrays with clearer examples"
```

### **Step 7: Sync with the Main Repository**

Before you submit your changes, make sure you have the latest version of the main repository:

```bash
# Add the original repository as "upstream"
git remote add upstream https://github.com/ORIGINAL-REPO-URL/MarkGPT-LLM-Curriculum.git

# Fetch the latest changes from the original
git fetch upstream

# Rebase your changes on top (synchronizes your work)
git rebase upstream/main

# If there are conflicts, resolve them (your editor will help), then:
# git add .
# git rebase --continue
```

### **Step 8: Push Your Changes to Your Fork**

Send your work to your GitHub copy:

```bash
# Push to your forked repository
git push origin main
```

### **Step 9: Create a Pull Request**

This is how you propose your changes to the main project.

1. Go to your forked repository on GitHub
2. You'll see a message: **"Your branch is ahead of the original"**
3. Click **Compare & pull request** button
4. Add a title: `feat: improve module-02 NumPy lesson examples`
5. Add a description of your changes
6. Click **Create Pull Request**

### **Step 10: Wait for Review & Feedback**

The maintainers will review your work, provide feedback, and eventually merge it if everything looks good. Be patient — reviews can take a few days.

---

## Setting Up Your Contributor Folder

### Folder Structure Template

When you create your contributor folder, mirror this structure:

```
contributors/
└── YOUR-GITHUB-USERNAME/
    ├── module-01/
    │   ├── lessons/       # Improved or new Jupyter notebooks
    │   ├── exercises/     # Additional exercises or fixes
    │   └── projects/      # Mini-projects and demos
    │
    ├── module-02/
    │   ├── lessons/
    │   ├── exercises/
    │   └── projects/
    │
    ├── module-03/
    │   ├── lessons/
    │   ├── exercises/
    │   └── projects/
    │
    └── README.md          # (OPTIONAL) Describe your contributions
```

### File Naming Conventions

When creating or modifying files:

- **Notebooks**: Use descriptive names: `improved_numpy_basics.ipynb`, `custom_neural_network_impl.ipynb`
- **Python Files**: Use snake_case: `data_preprocessing.py`, `train_model.py`
- **Markdown**: Use UPPERCASE: `CUSTOM_NOTES.md`, `SOLUTION_GUIDE.md`

---

## Working on Your Contributions

### Types of Contributions

#### 1. **Improving Lessons**
- Clarify confusing explanations
- Add more examples or visualizations
- Add real-world applications
- Fix typos or errors

```bash
# Example: Improving a lesson
contributors/YOUR-USERNAME/module-02/lessons/improved_pandas_guide.ipynb
```

#### 2. **Adding or Improving Exercises**
- Create additional practice problems
- Provide solution walkthroughs
- Add difficulty levels (beginner → advanced)

```bash
# Example: Adding exercises
contributors/YOUR-USERNAME/module-03/exercises/neural_network_challenges.ipynb
```

#### 3. **Building Projects**
- Create mini-capstone projects
- Demonstrate real-world applications
- Build end-to-end examples

```bash
# Example: Project
contributors/YOUR-USERNAME/module-05/projects/sentiment_analysis_nlp.ipynb
```

#### 4. **Improving Documentation**
- Update README files
- Create troubleshooting guides
- Document common issues
- Provide tips and tricks

```bash
# Example: Documentation
contributors/YOUR-USERNAME/module-01/LEARNING_TIPS.md
contributors/YOUR-USERNAME/module-02/COMMON_ISSUES.md
```

#### 5. **Bug Fixes**
- Report and fix errors in existing content
- Correct mathematical explanations
- Fix code that doesn't run

---

## Contribution Best Practices

### ✅ Do's

1. **Start Small**: Your first contribution should be small (1 file, 1 lesson improvement)
   - Small PRs are reviewed faster
   - Easier to get feedback and merge

2. **Test Before Submitting**: 
   - If you modify a Jupyter notebook, run all cells and ensure they execute without errors
   - If you add Python code, test it locally

3. **Follow Existing Patterns**:
   - Look at how existing lessons are structured
   - Follow the same naming conventions
   - Use the same notebook cell structure (Markdown → Code → Output)

4. **Document Your Changes**:
   - Write clear commit messages
   - In your PR description, explain what you changed and why
   - If referring to a specific lesson, include the module and lesson number

5. **Be Descriptive in Commit Messages**:
   ```bash
   # Good commit messages
   git commit -m "feat: add 5 practice problems for transformer attention mechanism"
   git commit -m "fix: correct mathematical notation in backpropagation lesson"
   git commit -m "docs: add common pandas errors and solutions guide"
   
   # Avoid vague commits
   git commit -m "update"
   git commit -m "fixes"
   git commit -m "stuff"
   ```

6. **Keep Your Fork Updated**:
   ```bash
   # Regularly sync with the main repository
   git fetch upstream
   git rebase upstream/main
   git push origin main
   ```

7. **Comment Your Code**:
   - Especially in Jupyter notebooks, explain complex sections
   - Write under "Explanation" Markdown cells
   - Use inline comments for tricky math or algorithms

8. **Check for Existing Work**:
   - Look at open issues and pull requests
   - Avoid duplicating work someone else is doing
   - Leave a comment if you want to work on something

### ❌ Don'ts

1. **Don't Work Directly on `main` Branch**:
   - Always create a feature branch for new work
   - Keeps history clean and organized

2. **Don't Modify Files Outside Your Contributor Folder**:
   - Only edit files in `contributors/YOUR-USERNAME/`
   - The maintainers will decide if your work merges into the main `modules/` folder

3. **Don't Copy-Paste Without Attribution**:
   - If using external resources, cite them
   - Add comments like: `# Adapted from: [URL]`
   - Respect intellectual property

4. **Don't Make Massive PRs**:
   - Avoid changing 20+ files at once
   - Break huge improvements into smaller, focused PRs
   - Easier to review, discuss, and merge incrementally

5. **Don't Commit Sensitive Data**:
   - Never commit API keys or passwords
   - Don't commit large data files (>10MB)
   - Use `.gitignore` to exclude these

6. **Don't Push to the Original Repository**:
   - Always push to your own fork
   - Let the maintainers merge when ready

7. **Don't Ignore Feedback**:
   - Review comments carefully
   - Make requested changes
   - Ask for clarification if unclear

---

## Creating a Pull Request

### Before You Submit

**Checklist:**

- [ ] Your code/notebook runs without errors
- [ ] You've created a clear, descriptive commit message
- [ ] You've synced with the upstream repository
- [ ] Your changes are isolated in your contributor folder
- [ ] You've added comments/documentation where needed
- [ ] You've read through your changes once more

### Pull Request Title

Use clear, concise titles:

- `feat: add 10 advanced exercises for module-04`
- `fix: correct backpropagation math in neural networks lesson`
- `docs: create debugging guide for module-02`
- `refactor: improve code clarity in transformer implementation guide`

### Pull Request Description

Include:

1. **What you changed**: A clear summary
2. **Why you changed it**: The problem you're solving
3. **Which module(s) it affects**: e.g., Module 02, Module 05
4. **Testing**: How you tested your changes
5. **Additional notes**: Any context the maintainers should know

**Example PR Description:**

```markdown
## Description

I improved the Module 02 NumPy lesson by adding 5 additional practice problems 
and rewriting the "Advanced Indexing" section with clearer examples.

## Changes Made

- Added 5 difficulty-tiered practice problems in contributors/alice-wonder/module-02/exercises
- Rewrote "Advanced Indexing" explanation with visual examples
- Fixed typo on Line 45 of the original lesson

## Related Issues

Addresses: #42 (request for more NumPy practice problems)

## Testing

- All cells in the notebook execute without errors (tested on Python 3.10)
- Ran through all exercises myself to ensure they work as expected

## How This Helps

These additional exercises help students solidify NumPy array manipulations before 
moving to pandas in Module 02 Lesson 4.
```

---

## Getting Help

### If You're Stuck

1. **Check the Documentation**:
   - Read [GETTING_STARTED.md](../GETTING_STARTED.md) for setup help
   - Check [docs/FAQ.md](../docs/FAQ.md) for common questions

2. **Search Existing Issues**:
   - Visit the Issues tab on GitHub
   - Someone might have already solved your problem

3. **Ask in a New Issue**:
   - Go to your forked repository's Issues tab
   - Describe your problem clearly
   - Provide error messages or screenshots

4. **Reach Out to Maintainers**:
   - Email: `iwstechnical@gmail.com`
   - Be polite and specific about what you need help with

### Common Git Issues

**"I made changes but forgot to commit them"**:
```bash
git add .
git commit -m "feat: your description here"
```

**"I made a mistake in my last commit message"**:
```bash
git commit --amend -m "correct message here"
git push origin your-branch --force-with-lease
```

**"I have merge conflicts"**:
```bash
# Look for sections marked <<<<<<, ======, >>>>>>
# Edit them manually, keeping the parts you want
git add .
git rebase --continue
```

**"I want to see my commit history"**:
```bash
git log --oneline  # Shows short history
git log            # Shows detailed history
```

---

## Final Thoughts

Contributing to MarkGPT is a journey, not a race. Every contribution — whether it's fixing a typo, improving an explanation, or building a complete new exercise set — helps thousands of students around the world learn about large language models.

Thank you for being part of this community. We're excited to see what you create! 🚀

---

**Questions? Contributions? Suggestions?** 
- Open an issue on GitHub
- Email: `iwstechnical@gmail.com`
- Check existing discussions in our repository

Happy contributing! 🎉
