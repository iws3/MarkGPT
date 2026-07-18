# 👥 Contributors Workspace

Welcome to the **Contributors Workspace**! This folder is your safe space to work on improvements to the MarkGPT LLM Curriculum before submitting them to the main repository.

## 📁 Folder Structure

```
contributors/
├── CONTRIBUTORS_GUIDE.md      # ← READ THIS FIRST: Complete step-by-step guide
├── .template/                 # Reference folder structure (do not edit)
│   ├── module-01/
│   ├── module-02/
│   └── ...
│
└── YOUR-GITHUB-USERNAME/      # Your personal contributor workspace
    ├── module-01/             # Work on Module 01
    ├── module-02/             # Work on Module 02
    ├── module-03/             # Work on Module 03
    └── ...up to module-10/
```

## 🚀 Quick Start

### Before Contributing

1. **Read [CONTRIBUTORS_GUIDE.md](CONTRIBUTORS_GUIDE.md)** — comprehensive step-by-step instructions
2. **Fork the repository** on GitHub
3. **Create your contributor folder** using your GitHub username

### Example Workflow

```bash
# 1. Clone your fork
git clone https://github.com/YOUR-USERNAME/MarkGPT-LLM-Curriculum.git
cd MarkGPT-LLM-Curriculum

# 2. Create your contributor folder
mkdir contributors/YOUR-USERNAME
cd contributors/YOUR-USERNAME

# 3. Create the module structure you want to work on
mkdir -p module-02/lessons
mkdir -p module-02/exercises

# 4. Make your changes, test them, and commit
git add .
git commit -m "feat: improve module-02 lesson on numpy basics"
git push origin main

# 5. Create a Pull Request on GitHub
```

## 📋 What Can You Contribute?

- ✅ **Improve lessons** — Clarify explanations, add examples
- ✅ **Add exercises** — New practice problems, solutions
- ✅ **Build projects** — Mini-capstone projects, demos
- ✅ **Fix bugs** — Correct errors in code or explanations
- ✅ **Improve documentation** — Create guides, tips, troubleshooting
- ✅ **Translate content** — Adapt lessons for different languages/dialects

## 🎯 Best Practices

1. **Work in your own folder** (`contributors/YOUR-USERNAME/`)
2. **Mirror the module structure** from the source `modules/` folder
3. **Test before submitting** — Run Jupyter notebooks to ensure they work
4. **Write clear commit messages** — Describe what you changed and why
5. **Create focused PRs** — Start small (1-2 files), expand once confident
6. **Follow existing patterns** — Look at how lessons are structured

## ❓ Frequently Asked Questions

**Q: What if I don't know where to start?**  
A: Check [CONTRIBUTORS_GUIDE.md](CONTRIBUTORS_GUIDE.md#ways-to-contribute) for ideas, or browse the main repository for lessons you found confusing — improve them!

**Q: Will my folder be kept private?**  
A: Your folder is visible in the repository, but only you manage your work there. When you're ready, you submit a Pull Request to have your work reviewed and merged.

**Q: How long does PR review take?**  
A: Typically 3-7 days, depending on maintainer availability. Thank you for your patience!

**Q: Can I request help while working?**  
A: Absolutely! Open an issue describing what you're stuck on, and the community can help.

## 🤝 Getting Help

- **Setup issues?** → Check [GETTING_STARTED.md](../GETTING_STARTED.md)
- **Git problems?** → See ["Common Git Issues"](CONTRIBUTORS_GUIDE.md#common-git-issues) in the guide
- **Content questions?** → Review the original lesson in `modules/`
- **Still stuck?** → Email `iwstechnical@gmail.com` or open a GitHub issue

## 📚 Resources

- [CONTRIBUTORS_GUIDE.md](CONTRIBUTORS_GUIDE.md) — Complete contribution workflow
- [../README.md](../README.md) — Main project overview
- [../GETTING_STARTED.md](../GETTING_STARTED.md) — Environment setup
- [../modules/](../modules/) — Source material to improve upon

## 🎉 Thank You!

Every contribution — no matter how small — helps students around the world learn about large language models. We're grateful for your effort and enthusiasm!

Happy contributing! 🚀
