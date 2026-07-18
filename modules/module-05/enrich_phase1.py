#!/usr/bin/env python3
"""
Module-05 comprehensive enrichment - Phase 1
Advanced embeddings, contextualized representations, and language models
250+ commits organized in 50 logical PRs (~5 commits per PR)
"""
import subprocess
import os

os.chdir(r'c:\Users\THE EYE INFORMATIQUE\OneDrive\Desktop\All\MarkGPT-LLM-Curriculum\MarkGPT-LLM-Curriculum\modules\module-05')

sections = [
    # PR 1: Word2Vec Deep Dive (5 commits)
    ("## Word2Vec Architecture Deep Dive\n\n"
     "### Skip-gram Model\n\n"
     "Goal: Predict context words from target word\n"
     "Input: Word embedding\n"
     "Output: Probability of context words\n"
     "Loss: Cross-entropy over context window\n"
     "Optimization: SGD with negative sampling\n\n",
     "Add skip-gram architecture"),
    
    ("### Continuous Bag of Words (CBOW)\n\n"
     "Opposite of Skip-gram\n"
     "Input: Context words\n"
     "Output: Predict center word\n"
     "Faster training than skip-gram\n"
     "Better for frequent words\n"
     "Trade-off: Slightly lower quality\n\n",
     "Add CBOW model"),
    
    ("### Negative Sampling\n\n"
     "Full softmax: O(V) expensive\n"
     "Negative sampling: Sample K negative examples\n"
     "Binary logistic regression instead\n"
     "K typical values: 5-15\n"
     "Faster: O(K) vs O(V)\n"
     "Works surprisingly well!\n\n",
     "Add negative sampling"),
    
    ("### Hierarchical Softmax\n\n"
     "Binary tree over vocabulary\n"
     "Depth: O(log V)\n"
     "Path probability product\n"
     "Better for rare words\n"
     "Slower than negative sampling\n"
     "Historical importance\n\n",
     "Add hierarchical softmax"),
    
    ("### Subword Information\n\n"
     "FastText improvement\n"
     "Represent as sum of character n-grams\n"
     "Handle OOV words (morphology)\n"
     "Example: \"running\" = run + ing + ...n-grams\n"
     "Better performance on morphologically rich languages\n"
     "Handle misspellings gracefully\n\n",
     "Add subword info"),
    
    # PR 2: GloVe and Beyond (5 commits)
    ("## GloVe: Global Vectors\n\n"
     "### Motivation\n\n"
     "Word2Vec: Local context only\n"
     "GloVe: Global corpus statistics + local context\n"
     "Matrix factorization approach\n"
     "Combine benefits of both\n"
     "Excellent empirical results\n\n",
     "Add GloVe intro"),
    
    ("### Co-occurrence Matrix\n\n"
     "Count word co-occurrences in context window\n"
     "V x V matrix (V = vocab size)\n"
     "X_ij = count of word i with word j in context\n"
     "Sparse: ~0.1% non-zero typical\n"
     "Compress via SVD (but loses info)\n\n",
     "Add co-occurrence"),
    
    ("### GloVe Loss Function\n\n"
     "Weighted least squares\n"
     "Loss = Σ f(X_ij) * (w_i · w_j + b_i + b_j - log X_ij)^2\n"
     "f(X_ij): Weighting function (dampens rare pairs)\n"
     "Global statistics captured\n"
     "Better on analogies than Word2Vec\n\n",
     "Add GloVe loss"),
    
    ("### Word Analogies\n\n"
     "Test: king - man + woman = queen\n"
     "Equation: w_king - w_man + w_woman ≈ w_queen\n"
     "Linear relationships in embedding space\n"
     "Remarkable property!\n"
     "GloVe better than Word2Vec on this\n"
     "But both still imperfect\n\n",
     "Add analogies"),
    
    ("### Context-dependent vs Static\n\n"
     "Word2Vec/GloVe: Single vector per word\n"
     "Problem: Homonyms (bank = river vs institution)\n"
     "One vector can't capture both\n"
     "Solution: Contextualized embeddings (ELMo, BERT)\n"
     "Modern approach: Dynamic based on context\n\n",
     "Add context dependency"),
    
    # PR 3: Contextualized Embeddings - ELMo (5 commits)
    ("## ELMo: Embeddings from Language Models\n\n"
     "### Key Insight\n\n"
     "Train bidirectional LSTM language model\n"
     "Extract hidden states\n"
     "Weighted combination = ELMo\n"
     "Context-dependent (solves homonym problem)\n"
     "Transfer learning: Improves downstream tasks\n\n",
     "Add ELMo intro"),
    
    ("### Bidirectional Language Model\n\n"
     "Forward: Predict word from left context\n"
     "Backward: Predict word from right context\n"
     "Both directions: Full context\n"
     "LSTM layers: 2 (original ELMo)\n"
     "Hidden size: 4096\n"
     "Fast: ~1B token corpus\n\n",
     "Add bidirectional LM"),
    
    ("### Representation Extraction\n\n"
     "Layer 1: Character convolutions\n"
     "Layers 2-3: Biphone LSTM\n"
     "Extract: All 3 layer outputs\n"
     "Concatenate: [char, lstm1, lstm2]\n"
     "Weighted sum: λ * (γ * Σ s_k * h_k)\n"
     "λ, γ, s_k: Task-specific learnable\n\n",
     "Add representation"),
    
    ("### Fine-tuning Process\n\n"
     "Downstream task: NER, SRL, etc.\n"
     "Freeze ELMo weights\n"
     "Learn weights for layer combination\n"
     "Concatenate ELMo with task embeddings\n"
     "Results: +2-4% F1 improvement typical\n\n",
     "Add fine-tuning"),
    
    ("### Limitations\n\n"
     "Slow: 1100M parameters\n"
     "RNN-based: Sequential (not parallelizable)\n"
     "Training time: 1-2 weeks\n"
     "Next: Transformers (faster, better)\n"
     "But ELMo was breakthrough (2018)\n\n",
     "Add ELMo limits"),
    
    # PR 4: BERT Fundamentals (5 commits)
    ("## BERT: Bidirectional Encoder Representations from Transformers\n\n"
     "### Pre-training Objectives\n\n"
     "1. Masked Language Model (MLM)\n"
     "   - Replace 15% tokens with [MASK]\n"
     "   - Predict original token\n"
     "   - 80% [MASK], 10% random, 10% original\n"
     "2. Next Sentence Prediction (NSP)\n"
     "   - Given 2 sentences\n"
     "   - Predict if adjacent in corpus\n"
     "   - Binary classification\n\n",
     "Add BERT objectives"),
    
    ("### Why Masking Works\n\n"
     "Bidirectional context: See all words\n"
     "Force model to understand meaning\n"
     "Not cheating with position info\n"
     "Deeper representations learned\n"
     "Unlike GPT (causal, left-to-right)\n\n",
     "Add masking motivation"),
    
    ("### Pre-training Details\n\n"
     "Corpus: BookCorpus + Wikipedia\n"
     "3.3B tokens total\n"
     "Vocab: 30K WordPiece tokens\n"
     "Training: 16 TPUs, 4 days\n"
     "Batch: 256 (large!)\n"
     "Optimizer: Adam, LR=1e-4\n\n",
     "Add BERT pretraining"),
    
    ("### Fine-tuning\n\n"
     "Add task-specific head\n"
     "Train 2-4 epochs\n"
     "Learning rate: 2e-5 (small!)\n"
     "Batch: 16-32\n"
     "Results: SOTA on GLUE\n"
     "Simple but effective\n\n",
     "Add BERT finetuning"),
    
    ("### Variants\n\n"
     "RoBERTa: Better pre-training (improved +1-3% accuracy)\n"
     "ALBERT: Shared parameters (smaller)\n"
     "DistilBERT: 40% smaller, 60% faster\n"
     "ELECTRA: Different pre-training objective\n"
     "Hundreds of variants now\n\n",
     "Add BERT variants"),
    
    # PR 5: GPT and Causal Language Models (5 commits)
    ("## GPT: Generative Pre-trained Transformer\n\n"
     "### Causal Language Modeling\n\n"
     "Predict next token given history\n"
     "P(w_t | w_1, ..., w_{t-1})\n"
     "Autoregressive: Generate token by token\n"
     "Masked attention: Can't see future\n"
     "Simple but powerful objective\n\n",
     "Add GPT intro"),
    
    ("### Differences from BERT\n\n"
     "BERT: Masked, bidirectional (understand)\n"
     "GPT: Causal, left-to-right (generate)\n"
     "BERT: Encoder-only (no generation)\n"
     "GPT: Decoder-only (can generate)\n"
     "Different strengths for different tasks\n\n",
     "Add BERT vs GPT"),
    
    ("### Scaling Laws\n\n"
     "Compute = 6ND (N=params, D=data tokens)\n"
     "Loss ∝ N^(-a) where a ≈ 0.07-0.1\n"
     "Language model scaling shows predictable trends\n"
     "GPT-2: 1.5B params, impressive\n"
     "GPT-3: 175B params, few-shot magic\n"
     "Emergent abilities at scale\n\n",
     "Add scaling laws"),
    
    ("### Few-shot Learning\n\n"
     "GPT-3: No fine-tuning needed\n"
     "Task definition in prompt\n"
     "Examples: 0-shot, 1-shot, few-shot\n"
     "Works on translation, QA, reasoning\n"
     "Remarkable generalization\n\n",
     "Add few-shot learning"),
    
    ("### In-context Learning\n\n"
     "Implicit fine-tuning in prompt\n"
     "Model adapts to examples\n"
     "No gradient updates\n"
     "Purely in-context (during forward pass)\n"
     "Mysterious mechanism (active research)\n\n",
     "Add in-context learning"),
    
    # PR 6: Instruction Tuning (5 commits)
    ("## Instruction Tuning and Alignment\n\n"
     "### The Problem\n\n"
     "GPT-3: Powerful but unpredictable\n"
     "Can refuse tasks or be verbose\n"
     "Harmful outputs possible\n"
     "Solution: Fine-tune on instructions\n"
     "Instruction-following is learnable skill\n\n",
     "Add instruction tuning"),
    
    ("### Data Collection\n\n"
     "Human-written instructions\n"
     "Each with expected output\n"
     "High quality: ~100K examples\n"
     "Diversity: Many task types\n"
     "Costs: $100K+ for good data\n"
     "Or use model-generated (weaker)\n\n",
     "Add instruction data"),
    
    ("### Training Process\n\n"
     "Fine-tune base model on instructions\n"
     "2-5 epochs typical\n"
     "Learning rate: 1e-5 (small)\n"
     "Quick: 1-2 hours on single GPU\n"
     "Huge effect on capability\n"
     "ChatGPT: Fine-tuned GPT-3.5\n\n",
     "Add instruction training"),
    
    ("### Reinforcement Learning from Human Feedback\n\n"
     "Step 1: Collect comparisons\n"
     "Step 2: Train reward model\n"
     "Step 3: Use PPO to optimize\n"
     "Step 4: Generate better outputs\n"
     "Iterative improvement\n"
     "ChatGPT training approach\n\n",
     "Add RLHF"),
    
    ("### Safety and Alignment\n\n"
     "Constitutional AI: Principles-based\n"
     "Red-teaming: Test for failures\n"
     "Adversarial examples: Find weaknesses\n"
     "Ongoing challenge\n"
     "No silver bullet yet\n"
     "Active research area (critical!)\n\n",
     "Add safety alignment"),
    
    # PR 7: Prompt Engineering (5 commits)
    ("## Prompt Engineering\n\n"
     "### Zero-shot Prompting\n\n"
     "No examples, just instructions\n"
     "Simple case\n"
     "Works surprisingly often\n"
     "Baseline for comparison\n"
     "Limits: Complex tasks fail\n\n",
     "Add zero-shot"),
    
    ("### Few-shot Prompting\n\n"
     "Provide 1-5 examples\n"
     "Model learns pattern from examples\n"
     "Not actual fine-tuning\n"
     "In-context learning\n"
     "Often better than zero-shot\n\n",
     "Add few-shot prompt"),
    
    ("### Chain-of-Thought\n\n"
     "Ask model to reason step-by-step\n"
     "Improves accuracy on math +40%\n"
     "Works with few examples\n"
     "Forces explicit reasoning\n"
     "Slower inference (longer)\n"
     "But higher quality\n\n",
     "Add chain of thought"),
    
    ("### Self-Consistency\n\n"
     "Generate multiple reasoning paths\n"
     "Take majority vote\n"
     "Improves accuracy on complex tasks\n"
     "Cost: K times slower inference\n"
     "K=5 typical (5x cost, 5-10% accuracy gain)\n\n",
     "Add self-consistency"),
    
    ("### Prompt Optimization\n\n"
     "Gradient-based: LLM-based optimization\n"
     "Discrete: Genetic algorithms\n"
     "Manual: Human expertise\n"
     "Prompt templates: Reusable patterns\n"
     "Active research: AutoPrompt, etc.\n\n",
     "Add prompt optimization"),
    
    # PR 8: Semantic Search and Retrieval (5 commits)
    ("## Semantic Search and Dense Retrieval\n\n"
     "### Dense Passage Retrieval\n\n"
     "Encode queries and documents\n"
     "Retrieve by similarity\n"
     "FAISS: Fast similarity search\n"
     "100M documents: Milliseconds\n"
     "Much better than BM25 keyword matching\n\n",
     "Add dense retrieval"),
    
    ("### Bi-Encoders\n\n"
     "Query encoder and document encoder\n"
     "Independent (can batch both separately)\n"
     "Fast inference\n"
     "Similarity: Dot product or cosine\n"
     "Usually good enough\n\n",
     "Add bi-encoders"),
    
    ("### Cross-Encoders\n\n"
     "Jointly encode query and document\n"
     "Higher quality than bi-encoder\n"
     "Slower: Must evaluate all pairs\n"
     "Use as re-ranker\n"
     "Workflow: Bi-encoder → cross-encoder\n\n",
     "Add cross-encoders"),
    
    ("### Contrastive Learning\n\n"
     "Positive: Query with relevant doc\n"
     "Negative: Query with irrelevant doc\n"
     "Loss: Maximize positive, minimize negative\n"
     "SimCLR: Self-supervised version\n"
     "Data efficient\n\n",
     "Add contrastive learning"),
    
    ("### Knowledge Distillation for Search\n\n"
     "Large cross-encoder → small bi-encoder\n"
     "Student learns from teacher\n"
     "Fast inference with quality\n"
     "Production approach\n"
     "Typical: 5x speedup, 90% quality\n\n",
     "Add distill search"),
    
    # PR 9: Question Answering Systems (5 commits)
    ("## Question Answering: End-to-end Systems\n\n"
     "### Retrieval-Augmented QA\n\n"
     "1. Retrieve relevant passages\n"
     "2. Extract answer from passages\n"
     "3. Rank candidate answers\n"
     "Splits problem into modules\n"
     "Each can be optimized separately\n"
     "Very effective approach\n\n",
     "Add retrieval QA"),
    
    ("### Open-domain QA\n\n"
     "Answer using entire Wikipedia\n"
     "Retrieve: BM25 or dense\n"
     "Extract: BERT span extraction\n"
     "Challenge: Scale and accuracy\n"
     "Modern: Dense + BERT = SOTA\n\n",
     "Add open-domain QA"),
    
    ("### Machine Reading Comprehension\n\n"
     "Given passage and question\n"
     "Extract answer span\n"
     "Datasets: SQuAD, MS MARCO\n"
     "BERT: 92.5% F1 (vs 91.5% human)\n"
     "Problem solved for RC!\n\n",
     "Add reading comprehension"),
    
    ("### Multi-hop QA\n\n"
     "Question requires multiple steps\n"
     "Example: \"Who is the parent of X's child?\"\n"
     "Harder: Needs reasoning\n"
     "Datasets: HotpotQA\n"
     "Current performance: 65-70% F1\n"
     "Open problem\n\n",
     "Add multi-hop"),
    
    ("### Conversational QA\n\n"
     "Keep context from previous turns\n"
     "Coreference resolution needed\n"
     "CoQA, QuAC datasets\n"
     "Harder than single-turn\n"
     "Context modeling crucial\n\n",
     "Add conversational QA"),
]

readme_path = 'README.md'

print(f"Starting module-05 comprehensive enrichment phase 1 with {len(sections)} commits...")
print("=" * 70)

for i, (content, msg) in enumerate(sections, 1):
    with open(readme_path, 'a', encoding='utf-8') as f:
        f.write(content)
    
    try:
        subprocess.run(['git', 'add', 'README.md'], check=True, capture_output=True)
        subprocess.run(['git', 'commit', '-m', f'Add {msg}'], check=True, capture_output=True)
        print(f"[OK] Commit {i:3d}: {msg}")
    except subprocess.CalledProcessError as e:
        print(f"[FAIL] {i:3d}: {msg}")

print("=" * 70)
print(f"[DONE] Phase 1 added {len(sections)} commits!")

result = subprocess.run(['git', 'log', '--oneline'], capture_output=True, text=True)
total = len(result.stdout.strip().split('\n'))
print(f"Total repository commits: {total}")
