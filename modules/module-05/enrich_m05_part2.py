#!/usr/bin/env python3
"""
Module-05 enrichment part 2 - 42 commits
Word embeddings (Word2Vec, GloVe, fastText)
"""
import subprocess
import os

os.chdir(r'c:\Users\THE EYE INFORMATIQUE\OneDrive\Desktop\All\MarkGPT-LLM-Curriculum\MarkGPT-LLM-Curriculum\modules\module-05')

sections = [
    ("## Word Embeddings Fundamentals\n\n"
     "### What are Word Embeddings?\n\n"
     "Dense vectors representing words\n"
     "Dimension: 50-300 typical\n"
     "Learned from large text corpus\n"
     "Similar words → similar vectors\n"
     "Foundation of modern NLP\n"
     "Input to neural networks\n\n",
     "Add embeddings intro"),
    
    ("### Distributional Hypothesis\n\n"
     "\"You shall know a word by the company it keeps\"\n"
     "Context determines meaning\n"
     "Words in similar contexts → similar meanings\n"
     "Learning principle: Co-occurrence statistics\n"
     "Basis for all embedding methods\n"
     "Remarkably effective!\n\n",
     "Add distributional hypothesis"),
    
    ("### Embedding Dimension\n\n"
     "50D: Very small, fast, limited expressiveness\n"
     "100D: Minimal, basic tasks\n"
     "300D: Standard for word embeddings\n"
     "1000D: Large, rich, slow\n"
     "Larger: More expressive, more parameters\n"
     "Typical: 300D word2vec, 768D BERT\n\n",
     "Add embedding dimension"),
    
    ("## Word2Vec\n\n"
     "### Skip-gram Model\n\n"
     "Predict context from word\n"
     "Input: Center word\n"
     "Output: Surrounding words (window)\n"
     "Loss: Cross-entropy\n"
     "Objective: Maximize P(context|word)\n"
     "Simple but powerful\n\n",
     "Add skip-gram"),
    
    ("### CBOW (Continuous Bag of Words)\n\n"
     "Opposite of Skip-gram\n"
     "Input: Context words\n"
     "Output: Center word\n"
     "Faster to train\n"
     "Better for frequent words\n"
     "Generally worse performance than Skip-gram\n\n",
     "Add CBOW"),
    
    ("### Negative Sampling\n\n"
     "Problem: Softmax over entire vocabulary\n"
     "Huge vocabulary: 1M+ words\n"
     "Computing softmax: O(V)\n"
     "Solution: Negative sampling\n"
     "Sample K negative examples\n"
     "Loss: Binary classification (positive vs negatives)\n"
     "10-15x speedup!\n\n",
     "Add negative sampling"),
    
    ("### Hierarchical Softmax\n\n"
     "Alternative to negative sampling\n"
     "Binary tree of vocabulary\n"
     "Log depth ≈ log(V)\n"
     "Each path: Binary decisions\n"
     "Faster than full softmax\n"
     "Slower than negative sampling\n"
     "Used in some implementations\n\n",
     "Add hierarchical softmax"),
    
    ("### Word2Vec Training\n\n"
     "1. Initialize embeddings randomly\n"
     "2. Iterate through corpus\n"
     "3. For each word-context pair:\n"
     "   - Compute output probability\n"
     "   - Compute loss\n"
     "   - Update embeddings\n"
     "4. Repeat multiple epochs\n"
     "Convergence: ~10-20 billion words\n\n",
     "Add word2vec training"),
    
    ("### Analogies: Additive Property\n\n"
     "\"king - man + woman ≈ queen\"\n"
     "Vector arithmetic works!\n"
     "v(king) - v(man) + v(woman) ≈ v(queen)\n"
     "Shows semantic structure\n"
     "Not always perfect\n"
     "Remarkable emergent property\n\n",
     "Add analogies"),
    
    ("### Why it Works\n\n"
     "Skip-gram objective: Embed words\n"
     "Minimize distance for similar contexts\n"
     "Words in similar positions → close vectors\n"
     "Unsupervised learning from corpus statistics\n"
     "No manual annotations needed\n"
     "Scales to huge corpora\n\n",
     "Add why word2vec works"),
    
    ("## GloVe (Global Vectors)\n\n"
     "### Motivation\n\n"
     "Word2Vec: Local context only\n"
     "GloVe: Combine local and global\n"
     "Use word co-occurrence matrix\n"
     "Factor matrix → embeddings\n"
     "Linear transformation\n"
     "Faster training than Word2Vec\n\n",
     "Add GloVe intro"),
    
    ("### Co-occurrence Matrix\n\n"
     "Count how often words co-occur\n"
     "X_ij = count(word_i near word_j)\n"
     "Huge and sparse: V x V\n"
     "100K vocab: 10 billion entries!\n"
     "Solution: Only store non-zeros\n"
     "Factorize: X ≈ W @ W^T\n\n",
     "Add co-occurrence matrix"),
    
    ("### GloVe Objective\n\n"
     "Weighted least squares:\n"
     "L = Σ_ij f(X_ij) (w_i . w_j - log X_ij)^2\n"
     "f(X_ij): Weight function\n"
     "Prevents rare co-occurrences dominating\n"
     "Closed form optimization possible\n"
     "Faster convergence than Word2Vec\n\n",
     "Add GloVe objective"),
    
    ("### GloVe vs Word2Vec\n\n"
     "GloVe: Explicit global statistics\n"
     "Word2Vec: Implicit via optimization\n"
     "GloVe: Typically better (slightly)\n"
     "Word2Vec: Simpler, faster\n"
     "Both: Great in practice\n"
     "Modern: Transformers replace both\n\n",
     "Add GloVe vs Word2Vec"),
    
    ("## fastText\n\n"
     "### Subword Vectors\n\n"
     "Problem: Word2Vec can't handle OOV\n"
     "Solution: Learn character n-gram embeddings\n"
     "Word = sum of character n-grams\n"
     "v(hello) = v(he) + v(el) + v(ll) + v(lo) + ...\n"
     "Handles unseen words!\n"
     "Handles morphology\n\n",
     "Add fastText"),
    
    ("### Character N-grams\n\n"
     "Typical: 3-6 character n-grams\n"
     "\"hello\" (n=3): [hel, ell, llo]\n"
     "\"hello\" (n=4): [hell, ello]\n"
     "\"hello\" (n=5): [hello]\n"
     "Plus special markers for word start/end\n"
     "Sum: Gives word embedding\n\n",
     "Add character ngrams"),
    
    ("### fastText Training\n\n"
     "1. Compute character n-gram vectors\n"
     "2. Word vector = sum of n-grams\n"
     "3. Train like Word2Vec (Skip-gram)\n"
     "4. Learn both: word vectors and n-grams\n"
     "5. At inference: Can compute for OOV\n\n"
     "Handling spelling: Similar n-grams\n\n",
     "Add fastText training"),
    
    ("### fastText Language Support\n\n"
     "Multilingual: 176+ languages\n"
     "Pre-trained vectors available\n"
     "Works great for morphologically rich\n"
     "Turkish, Finnish, Czech, etc.\n"
     "Better than Word2Vec for these\n"
     "Practical advantage for low-resource\n\n",
     "Add fastText languages"),
    
    ("## Embedding Visualization\n\n"
     "### t-SNE Projection\n\n"
     "300D embedding → 2D visualization\n"
     "Non-linear dimensional reduction\n"
     "Preserves local structure\n"
     "Shows word clusters\n"
     "Similar words close together\n"
     "Beautiful emergent structure\n\n",
     "Add t-SNE"),
    
    ("### UMAP\n\n"
     "Uniform Manifold Approximation\n"
     "Faster than t-SNE\n"
     "Preserves global structure better\n"
     "Two parameters: n_neighbors, min_dist\n"
     "Good for large embedding sets\n"
     "Can scale to millions of vectors\n\n",
     "Add UMAP"),
    
    ("## Training Embeddings\n\n"
     "### Using gensim\n\n"
     "```python\n"
     "from gensim.models import Word2Vec\n\n"
     "model = Word2Vec(sentences, size=300, window=5)\n"
     "vec = model.wv['hello']\n"
     "similar = model.wv.most_similar('king')\n"
     "```\n\n"
     "### Using fastText\n\n"
     "```python\n"
     "import fasttext\n"
     "model = fasttext.train_unsupervised('data.txt')\n"
     "vec = model.get_word_vector('hello')\n"
     "```\n\n",
     "Add embedding code"),
    
    ("## Pre-trained Embeddings\n\n"
     "### When to Use Pre-trained\n\n"
     "Limited data: Always use\n"
     "Data on same domain: Cold-start faster\n"
     "Transfer learning: Few examples needed\n"
     "Stability: Better than random init\n"
     "Typical: Download from model zoos\n"
     "Fine-tune on task data\n\n",
     "Add pretrained"),
    
    ("### Domain Specificity\n\n"
     "Generic (Wikipedia): Good baseline\n"
     "Medical (PubMed): Better for medical\n"
     "Code (GitHub): Better for code\n"
     "Domain adaptation: Fine-tune\n"
     "Task-specific: Train from scratch if big data\n"
     "Usually: Domain > task-specific\n\n",
     "Add domain"),
    
    ("## Embedding Evaluation\n\n"
     "### Intrinsic Evaluation\n\n"
     "Word analogies: King - man + woman = queen\n"
     "Similarity correlation: Compare to human ratings\n"
     "Relatedness tasks: RareWord, SimLex\n"
     "Direct measure of embedding quality\n"
     "Fast to compute\n"
     "Not always predictive of downstream\n\n",
     "Add intrinsic eval"),
    
    ("### Extrinsic Evaluation\n\n"
     "Downstream task performance\n"
     "Text classification\n"
     "Named entity recognition\n"
     "Sentiment analysis\n"
     "Best predictor of real usefulness\n"
     "But: Slower and task-dependent\n\n",
     "Add extrinsic eval"),
    
    ("## Embedding Training Tips\n\n"
     "### Hyperparameters\n\n"
     "Size (dimension): 100-300 typical\n"
     "Window size: 2-15 (larger = preserves syntax)\n"
     "Negative samples: 5-15\n"
     "Epochs: 5-30 (more if larger corpus)\n"
     "Learning rate: Usually 0.025 default\n"
     "Minimum count: 5, drop rare words\n\n",
     "Add training hyperparams"),
    
    ("### Dataset Size\n\n"
     "Small (1M words): Overfitting risk\n"
     "Medium (100M words): Good quality\n"
     "Large (1B+ words): Best quality\n"
     "Diminishing returns after 1B\n"
     "Typical: Train on crawled web text\n"
     "Wikipedia (3.5B words): Excellent base\n\n",
     "Add dataset size"),
    
    ("### Computational Efficiency\n\n"
     "Word2Vec: O(V) with negative sampling\n"
     "Can train on single machine\n"
     "100M words: Minutes\n"
     "1B words: Hours\n"
     "Parallel training: Multiple threads\n"
     "Fast compared to deep learning\n\n",
     "Add efficiency"),
    
    ("## Contextualized Embeddings\n\n"
     "### Static vs Dynamic\n\n"
     "Static (Word2Vec): Same vector for all contexts\n"
     "Dynamic (ELMo, BERT): Different per context\n"
     "Problem with static: \"bank\" is ambiguous\n"
     "ELMo: First dynamic method (2018)\n"
     "Revolution in NLP\n"
     "Transition to transformers\n\n",
     "Add contextualized"),
    
    ("## Common Mistakes\n\n"
     "### Don't\n\n"
     "- Train on tiny corpus (<100K words)\n"
     "- Use embedding size > 500 without need\n"
     "- Ignore OOV problem\n"
     "- Fine-tune randomly (use pre-trained)\n"
     "- Ignore domain mismatch\n"
     "- Train embeddings from scratch if data limited\n\n"
     "### Do\n\n"
     "- Use pre-trained when possible\n"
     "- Match embedding dim to task\n"
     "- Fine-tune on task data\n"
     "- Evaluate on downstream tasks\n"
     "- Consider subword (fastText) for OOV\n"
     "- Use contextualized for modern tasks\n\n",
     "Add common mistakes"),
    
    ("## Embedding Arithmetic\n\n"
     "### Word Analogies\n\n"
     "Task: a is to b as c is to d\n"
     "Find d: d ≈ b - a + c\n"
     "Example: Paris - France + Germany = Berlin\n"
     "Remarkable: Works surprisingly well\n"
     "Emergent property of vector space\n"
     "Not always accurate but conceptually rich\n\n",
     "Add analogies detail"),
    
    ("### Semantic Relationships\n\n"
     "Synonyms: Similar vectors\n"
     "Antonyms: Opposite vectors\n"
     "Hypernyms: Generalization direction\n"
     "Part-of: Also somewhat reflected\n"
     "Relational properties: Encoded implicitly\n"
     "Remarkable unsupervised learning!\n\n",
     "Add relationships"),
]

readme_path = 'README.md'

print(f"Starting module-05 part 2 with {len(sections)} commits...")
print("=" * 60)

for i, (content, msg) in enumerate(sections, 1):
    with open(readme_path, 'a', encoding='utf-8') as f:
        f.write(content)
    
    try:
        subprocess.run(['git', 'add', 'README.md'], check=True, capture_output=True)
        subprocess.run(['git', 'commit', '-m', f'Enrich module-05 part2 {i}: {msg}'], check=True, capture_output=True)
        print(f"[OK] Commit {i:2d}/{len(sections)}: {msg}")
    except subprocess.CalledProcessError:
        print(f"[FAIL] Part 2 {i:2d}: {msg}")

print("=" * 60)
print(f"[DONE] Part 2 added {len(sections)} commits!")

result = subprocess.run(['git', 'log', '--oneline'], capture_output=True, text=True)
total = len(result.stdout.strip().split('\n'))
print(f"Total repository commits: {total}")
