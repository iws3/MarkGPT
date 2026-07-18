# 📖 Contributor Resources & Learning Materials

A curated list of resources to help you contribute effectively.

---

## Table of Contents

1. [Getting Started](#getting-started)
2. [Git & GitHub Learning](#git--github-learning)
3. [Writing Better Explanations](#writing-better-explanations)
4. [Tools & Environment Setup](#tools--environment-setup)
5. [Curriculum Content](#curriculum-content)
6. [Community & Support](#community--support)

---

## Getting Started

### MarkGPT-Specific Guides
- **[CONTRIBUTORS_GUIDE.md](CONTRIBUTORS_GUIDE.md)** — Complete workflow
- **[FIRST_CONTRIBUTION_GUIDE.md](FIRST_CONTRIBUTION_GUIDE.md)** — For beginners
- **[CONTRIBUTION_CHECKLISTS.md](CONTRIBUTION_CHECKLISTS.md)** — Quick validation
- **[CONTRIBUTION_FAQ.md](CONTRIBUTION_FAQ.md)** — Common questions
- **[../../BEST_PRACTICES.md](../../BEST_PRACTICES.md)** — Quality standards
- **[../../CONTRIBUTOR_EXAMPLES.md](../../CONTRIBUTOR_EXAMPLES.md)** — Real examples

### Quick Read (30 min)
1. [FIRST_CONTRIBUTION_GUIDE.md](FIRST_CONTRIBUTION_GUIDE.md) — Start here
2. [CONTRIBUTION_CHECKLISTS.md](CONTRIBUTION_CHECKLISTS.md) — What to check
3. Make your first contribution!

---

## Git & GitHub Learning

### Interactive Learning
- **Learn Git Branching** — https://learngitbranching.js.org/
  - Interactive visualization of Git concepts
  - Practice branches, merging, rebasing
  - Takes 20-30 minutes

- **GitHub Learning Lab** — https://lab.github.com/
  - Free courses on Git and GitHub
  - Hands-on practice in sandboxed repos

### Practical Guides
- **[GIT_TROUBLESHOOTING.md](GIT_TROUBLESHOOTING.md)** — Common issues & fixes
- **GitHub Official Guides** — https://guides.github.com/
- **Conventional Commits** — https://www.conventionalcommits.org/

### Video Tutorials
- **Git in 15 Minutes** — https://git-scm.com/videos/
- **GitHub Desktop** — GUI alternative to command line

### Cheat Sheets
```bash
# Basic workflow
git clone <url>
git checkout -b feat/name
git add .
git commit -m "feat: description"
git push origin feat/name
```

---

## Writing Better Explanations

### Resources for Educators
- **Bloom's Taxonomy** — https://www.bloomstaxonomy.net/
  - Plan learning outcomes
  - Structure your lesson from simple to complex

- **The Feynman Technique** — Explain concepts simply
  1. Choose a concept
  2. Explain it simply
  3. Identify what you don't understand
  4. Refine and simplify further

### Technical Writing
- **Google Developer Style Guide** — https://developers.google.com/style/
  - Clear, concise technical writing
  - Examples and best practices

- **IBM's API Documentation Guide** — https://www.ibm.com/docs/
  - How to document code and APIs
  - Clear structure and examples

### Mathematical Notation
- **LaTeX Math Mode Reference** — https://en.wikibooks.org/wiki/LaTeX/Mathematics
  - Write equations in Jupyter notebooks
  - Use `$$equation$$` for display math

### Data Visualization
- **Matplotlib Documentation** — https://matplotlib.org/
- **Plotly Documentation** — https://plotly.com/python/
- **Seaborn Gallery** — https://seaborn.pydata.org/examples.html

---

## Tools & Environment Setup

### Essential Tools
- **VS Code** — https://code.visualstudio.com/ (Free)
  - Python extension
  - Jupyter extension
  - Git integration
  
- **PyCharm Community** — https://www.jetbrains.com/pycharm/community/ (Free)
  - Full Python IDE
  - Great notebook support

- **Jupyter Notebook** — https://jupyter.org/
  - Interactive coding environment
  - Required for coursework

### Installation Guides
- **[../../GETTING_STARTED.md](../../GETTING_STARTED.md)** — Full setup guide
- **Python Installation** — https://www.python.org/downloads/
  - Recommend 3.10+
  - Installation guides for Mac, Windows, Linux

- **Virtual Environments** — https://docs.python.org/3/tutorial/venv.html
  - Keep project dependencies isolated
  - `python -m venv venv`

### Testing & Validation
- **Pytest** — Testing Python code — https://pytest.org/
- **Black** — Code formatter — https://github.com/psf/black
- **Ruff** — Linter — https://github.com/astral-sh/ruff

---

## Curriculum Content

### Understanding LLMs
- **Neural Networks from Scratch** — https://github.com/karpathy/makemore
  - Andrej Karpathy's educational series
  - Build models with numpy

- **The Attention Mechanism** — https://arxiv.org/abs/1706.03762
  - Original "Attention is All You Need" paper
  - Transformers architecture

- **GPT from First Principles** — https://karpathy.ai/
  - Detailed explanations by Andrej Karpathy
  - Building language models

### Referenced Textbooks
- **"Speech and Language Processing" (3rd ed.)** — Jurafsky & Martin
  - https://web.stanford.edu/~jurafsky/slp3/
  - Comprehensive NLP reference

- **"Deep Learning" — Goodfellow, Bengio, Courville**
  - Mathematical foundations
  - Free online: https://www.deeplearningbook.org/

### Banso Language Resources
- **Cameroon Languages** — Research on Banso/Nso'
- **Indigenous Language Preservation** — UNESCO resources
- **Linguistic Databases** — OSE (Open Source Ethnography)

---

## Community & Support

### MarkGPT Community
- **GitHub Issues** — Ask questions, report bugs
- **GitHub Discussions** — Share ideas, learn together
- **Email Support** — iwstechnical@gmail.com

### Broader Open Source Community
- **First Timers Only** — https://www.firsttimersonly.com/
  - Beginner-friendly open source projects
  - Resources for first-time contributors

- **Open Source Guides** — https://opensource.guide/
  - How to start, contribute, maintain open source

- **Dev Community** — https://dev.to/
  - Articles on programming and open source

### Mentorship
- **Find a Mentor** — https://www.dataquest.io/blog/open-source-projects-for-beginners/
  - Connect with experienced developers
  - Many projects have mentorship programs

---

## Learning Paths

### Path 1: Beginner Student (Time: 4 weeks)
Week 1: Setup and Git basics
- [GETTING_STARTED.md](../../GETTING_STARTED.md)
- [GIT_TROUBLESHOOTING.md](GIT_TROUBLESHOOTING.md)
- Learn Git Branching (interactive)

Week 2: First contribution
- [FIRST_CONTRIBUTION_GUIDE.md](FIRST_CONTRIBUTION_GUIDE.md)
- Make a small contribution (typo fix, clarification)
- Get it merged!

Week 3: Deeper contribution
- [BEST_PRACTICES.md](../../BEST_PRACTICES.md)
- Improve a lesson or add exercises
- Submit larger PR

Week 4: Become a Regular
- Continue contributing to modules
- Help review other PRs
- Guide new contributors

### Path 2: Experienced Developer (Time: 2-3 days)
- Skim [BEST_PRACTICES.md](../../BEST_PRACTICES.md)
- Review existing modules
- Find area to improve
- Submit PR

### Path 3: Educator (Time: 1 week)
- Review curriculum structure
- Identify gaps
- Create lesson improvements or new exercises
- [BEST_PRACTICES.md](../../BEST_PRACTICES.md) — Educational standards

### Path 4: Banso Language Expert (Time: Ongoing)
- Review linguistic content
- Suggest improvements
- Contribute new Banso materials
- Build dataset
- Contact: iwstechnical@gmail.com

---

## Staying Updated

### Follow Project Changes
- **GitHub Star** — https://github.com/iwstechnical/MarkGPT-LLM-Curriculum
  - Get notifications for releases
  - Track new issues

- **Watch Repository** — Get notifications for PRs and issues
- **Subscribe to Discussions** — Participate in community

### Learning Resources
- **Join a Study Group** — Work through curriculum with others
- **Share Your Improvements** — Blog about what you learned
- **Contribute Back** — Share resources with community

---

## Still Learning?

These resources can help:

1. **Specific Python Skill?**
   - Search "how to [skill] in Python"
   - Check Stack Overflow
   - Look at existing code in MarkGPT

2. **Specific Git Issue?**
   - See [GIT_TROUBLESHOOTING.md](GIT_TROUBLESHOOTING.md)
   - Search "git [error message]"
   - Come ask us in an issue!

3. **Curriculum Content Questions?**
   - Review the original modules
   - Read referenced papers
   - Ask in GitHub discussions

4. **Writing Better Code/Notebooks?**
   - Study existing lessons
   - Read [BEST_PRACTICES.md](../../BEST_PRACTICES.md)
   - Get feedback in PR review

---

## Your Next Step

**Pick one resource and dive in:**
- Want to contribute soon? → [FIRST_CONTRIBUTION_GUIDE.md](FIRST_CONTRIBUTION_GUIDE.md)
- Git troubles? → [GIT_TROUBLESHOOTING.md](GIT_TROUBLESHOOTING.md)
- Quality questions? → [../../BEST_PRACTICES.md](../../BEST_PRACTICES.md)
- Specific question? → [CONTRIBUTION_FAQ.md](CONTRIBUTION_FAQ.md)

Let's go! 🚀
