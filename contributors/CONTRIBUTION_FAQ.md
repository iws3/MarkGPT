# 🎯 FAQ: Frequently Asked Questions About Contributing

Got a question? You're probably not alone! Check here for answers.

---

## General Questions

### Q: Is this project for beginners?
**A:** Yes! The curriculum is FOR beginners, and we welcome contributions from all levels. If you're learning the material, you can improve it for others who learn the same way.

### Q: Do I need to be an expert to contribute?
**A:** No! Beginner and intermediate contributors are welcome. In fact, a beginner's perspective on what's confusing is VERY valuable.

### Q: Can I contribute if I'm not a native English speaker?
**A:** Absolutely! Language corrections and translation contributions are welcome. Pair with a native English speaker for translation, if helpful.

### Q: Are there specific things you most need?
**A:** Yes! We especially need:
- More exercises and practice problems
- Visualizations of complex concepts
- Debugging guides for common errors
- Banso language expertise
- Real-world project examples

### Q: How long does a contribution take?
**A:** Anywhere from 30 minutes (typo fix) to 8+ hours (full lesson). Start small!

---

## First Steps

### Q: I want to contribute but don't know where to start
**A:** Read this in order:
1. [CONTRIBUTORS_GUIDE.md](CONTRIBUTORS_GUIDE.md) — Overview
2. [FIRST_CONTRIBUTION_GUIDE.md](FIRST_CONTRIBUTION_GUIDE.md) — Step-by-step for beginners
3. [CONTRIBUTOR_EXAMPLES.md](../../CONTRIBUTOR_EXAMPLES.md) — See real examples

### Q: What if I don't understand a module?
**A:** That's actually the BEST reason to contribute! Write clearer explanations that would have helped you.

### Q: Can I ask for ideas on what to improve?
**A:** Yes! Open an issue titled "Improvement idea: [topic]" and describe what confused you or what you'd like to see.

### Q: Do I need permission before starting?
**A:** For small improvements, just do it! For large additions, open an issue first to discuss with maintainers.

---

## Technical Questions

### Q: I don't know Git very well — can I still contribute?
**A:** Yes! Check [GIT_TROUBLESHOOTING.md](GIT_TROUBLESHOOTING.md) for help. We also forgive Git mistakes 😊

### Q: What if I make a Git mistake?
**A:** Don't panic! Git is designed so you can't actually lose your work. See GIT_TROUBLESHOOTING.md for recovery steps.

### Q: Do I need to set up a local development environment?
**A:** For most contributions (lessons, exercises), you just need:
- Python 3.10+
- A code editor (VS Code is free)
- Jupyter notebooks

### Q: How do I test my Jupyter notebook?
**A:** 
```bash
# Open it in Jupyter
jupyter notebook

# Or right-click in VS Code and "Open with Jupyter Notebook"

# Key: Run all cells (Kernel → Restart & Run All) and check for errors
```

### Q: The notebook won't run — what's wrong?
**A:** Common issues:
- Missing import at the top (add it!)
- File path is hardcoded (make it relative)
- Package not installed (`pip install package_name`)
- Python version too old (upgrade to 3.10+)

### Q: I don't know Python well — can I still contribute?
**A:** Absolutely! You can:
- Improve writing/explanations
- Fix documentation
- Flag errors or unclear sections
- Help with non-code contributions

---

## About Your Contribution

### Q: What if my contribution overlaps with someone else's?
**A:** No problem! Different perspectives are valuable. We'll figure out the best approach.

### Q: Can I contribute to multiple modules?
**A:** Yes, but start with ONE PR. Get it merged first, then move to the next.

### Q: Should I submit one big PR or many small ones?
**A:** Many small ones! Small PRs are:
- Easier to review
- Faster to merge
- Better for tracking history
- Less overwhelming for reviewers

### Q: How much credit do I get?
**A:** You get:
- Your name in CONTRIBUTORS.md
- GitHub shows your contribution
- Listed in your commit history
- Community recognition

### Q: Can I include my portfolio/website link?
**A:** In your PR description or CONTRIBUTORS.md entry, yes!

---

## Pull Request Process

### Q: How long does PR review take?
**A:** Usually 3-7 days. Thank you for being patient!

### Q: What if they ask me to make changes?
**A:** That's normal! They're helping improve your work. Make the changes:
```bash
git add .
git commit -m "fix: address review feedback"
git push origin your-branch
```

The PR automatically updates!

### Q: Can a PR be rejected?
**A:** Yes, but our goal is to help you make it better, not to reject it. If something can't merge, we'll explain why and how to fix it.

### Q: What if I disagree with feedback?
**A:** Have a respectful conversation! Ask clarifying questions. Most feedback includes reasoning.

### Q: Can I ask questions in my PR?
**A:** Yes! Use comments or explicitly ask in the PR description. Maintainers are happy to help.

---

## Content Guidelines

### Q: How do I make sure my content fits?
**A:** Read [BEST_PRACTICES.md](../../BEST_PRACTICES.md) and look at existing lessons in the module you're improving.

### Q: Should I use Python 3.11 features?
**A:** No, stick to Python 3.10 (the minimum version we support).

### Q: Can I use fancy libraries?
**A:** Check existing imports first. Avoid adding new heavy dependencies without discussion.

### Q: How do I cite something?
**A:** Simple format in Markdown:
```markdown
- [Source name](URL) by Author Name
- See also: [Related topic](link)

# References
- Vaswani et al. (2017). "Attention is All You Need"
```

---

## Working on the Curriculum

### Q: Should I improve something I didn't write?
**A:** Yes! Improving existing content is one of the most valuable contributions.

### Q: Can I create completely new lessons?
**A:** For large new lessons, discuss first. For exercises in existing lessons, just do it!

### Q: Is it okay to suggest completely different approaches?
**A:** Yes, but open an issue first to discuss with maintainers.

### Q: What if I find an error?
**A:** Fix it! That's a valuable contribution:
```bash
git commit -m "fix: correct the softmax formula explanation in module-04"
```

---

## Contributing Banso Language Content

### Q: I'm a native Banso speaker — how can I help?
**A:** We need your expertise! Options:
- Review linguistic accuracy
- Contribute pronunciation guides
- Add cultural context
- Expand the Banso dataset
- Suggest better translations

### Q: I speak Banso but just want to help, not be a lead contributor
**A:** Perfect! You can:
- Comment on existing content for accuracy
- Suggest improvements
- Review PRs
- Contribute specific sections

### Q: How do I ensure proper attribution?
**A:** We'll credit you in:
- CONTRIBUTORS.md
- Module documentation
- Relevant PR/commit messages

---

## Mentorship & Learning

### Q: Can I ask maintainers for help learning the content?
**A:** That's outside the scope of the repository, but:
- Work through the lessons yourself
- Reach out to the community on GitHub discussions
- See GETTING_STARTED.md for resources

### Q: Can I get feedback on my own learning?
**A:** Create a discussion post or open a learning-related issue!

### Q: Can I mentor other contributors?
**A:** Yes! Help with reviews and questions in the community.

---

## Special Situations

### Q: What if I contribute, then can't finish?
**A:** No problem! Just let us know in the PR. We can:
- Help you finish
- Mark as draft
- Close without merging (future contributors can use it as reference)

### Q: Can I contribute confidentially (without my name)?
**A:** Contributions are by default attributed. If you need anonymity, DM a maintainer to discuss.

### Q: What if I have a disabilities or access needs?
**A:** Let us know! We'll work to accommodate you. Email: iwstechnical@gmail.com

### Q: Can I contribute from outside certain countries/regions?
**A:** We welcome contributions from everywhere!

---

## Still Have Questions?

- **GitHub Issues:** Create an issue with your question
- **GitHub Discussions:** Join the community conversation  
- **Email:** iwstechnical@gmail.com

We're here to help! 🙌
