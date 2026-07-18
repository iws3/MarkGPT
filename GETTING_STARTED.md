# 🚀 Getting Started — MarkGPT LLM Curriculum

Welcome. Before you write a single line of code, read this page completely. It will save you hours of frustration and set you up for success across all 60 days.

---

## Step 1: Choose Your Learning Environment

You have three options, ordered from easiest to most powerful.

### Option A — Google Colab (Recommended for Beginners)
Google Colab gives you a free Python environment with GPU access in your browser. No installation required.
1. Go to https://colab.research.google.com
2. Sign in with a Google account
3. Open any notebook from the `notebooks/` folder using File → Open → GitHub
4. For GPU: Runtime → Change Runtime Type → T4 GPU (free tier, sufficient through Module 7)

### Option B — Local Python Installation
For learners who want full control over their environment.

```bash
# 1. Install Python 3.10 or later from https://python.org

# 2. Clone this repository
git clone https://github.com/your-username/MarkGPT-LLM-Curriculum.git
cd MarkGPT-LLM-Curriculum

# 3. Create and activate a virtual environment
python -m venv markgpt-env
source markgpt-env/bin/activate   # On Windows: markgpt-env\Scripts\activate

# 4. Install all dependencies
pip install -r requirements.txt

# 5. Verify installation
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

### Option C — Kaggle Notebooks
Similar to Colab. Go to https://kaggle.com/code and create a new notebook. Kaggle also offers free GPU hours and is excellent for the training runs in Modules 7–10.

---

## Step 2: Install Required Libraries

The full list is in `requirements.txt`. The most important ones, and why you need them:

**PyTorch** — the deep learning framework. Everything in this course is written in PyTorch because it is transparent and educational. We avoid hiding things behind high-level wrappers until you understand what they wrap.

**Transformers (Hugging Face)** — used for loading pretrained tokenizers and models in Modules 8–10. We do NOT use it as a crutch in Modules 3–7.

**SentencePiece** — Google's tokenizer library. We use it to build the Banso-specific tokenizer in Module 9.

**Datasets (Hugging Face)** — for loading and streaming the Bible corpus efficiently.

**Wandb** — for tracking training experiments. This is optional but strongly recommended for Modules 7–10.

**Gradio** — for building the MarkGPT web demo on Day 59.

---

## Step 3: Download the Bible Corpus

Run this once to set up your data directory:

```bash
python scripts/download_data.py
```

This will download:
- King James Version Bible (plain text, ~4.5MB)
- World English Bible (copyright-free modern translation)
- Available Lamnso'/Banso Bible excerpts
- Banso proverbs and oral literature collection

---

## Step 4: Verify Everything Works

```bash
python scripts/verify_setup.py
```

You should see a green checkmark next to each item. If anything fails, check the `docs/TROUBLESHOOTING.md` file.

---

## Step 5: Understand the Daily Rhythm

Each day follows a consistent structure that mirrors how the best university courses are run:

**Morning (45–60 min): Read the lesson.** Every lesson is in `modules/module-XX/lessons/`. Read it fully, with a pencil. Annotate. Ask "why?" at every step.

**Midday (30–45 min): Do the exercises.** Exercises are in `modules/module-XX/exercises/`. Some are on paper. Some are code. Some are written reflection. All are important.

**Evening (15 min): Update your learning journal.** There is a `journal/` template in each module. Write three sentences: what you learned, what confused you, what you want to explore more.

This rhythm, held consistently, is more effective than any particular technique or shortcut.

---

## Step 6: A Word on Difficulty

This curriculum is honest about difficulty. Some days will be hard. Day 15 (backpropagation from scratch) is genuinely challenging. Day 31 (reading the Attention paper) requires real focus. Day 46 (tokenizer adaptation for tonal languages) requires patience and creativity.

When you hit a wall, the recommended sequence is:
1. Re-read the lesson from the beginning
2. Try the exercise with a simpler example (fewer words, shorter sentences)
3. Look at the hints in `modules/module-XX/exercises/HINTS.md`
4. Check the community forum in `docs/COMMUNITY.md`
5. Only then look at the solutions in `modules/module-XX/exercises/solutions/`

The goal is not to complete 60 days. The goal is to understand deeply enough to build something real.

---

*Good luck. You are about to do something genuinely difficult and genuinely worthwhile.*
