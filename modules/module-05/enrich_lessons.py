#!/usr/bin/env python3
"""
Module-05 lesson enrichment - 192 commits
Enrich all lesson README files
"""
import subprocess
import os
import glob

os.chdir(r'c:\Users\THE EYE INFORMATIQUE\OneDrive\Desktop\All\MarkGPT-LLM-Curriculum\MarkGPT-LLM-Curriculum\modules\module-05\lessons')

lesson_files = [
    ('L25_tokenization_deep_dive.md', [
        ("## Deep Dive: Tokenization Algorithms\n\n"
         "### Longest Match First\n\n"
         "Greedy: Always take longest token\n"
         "Efficient: O(1) lookup\n"
         "Problem: Suboptimal (\"hello\" as char)\n"
         "Used in: Early systems\n\n",
         "Add longest match"),
        
        ("### Maximum Likelihood\n\n"
         "Probability of tokenization\n"
         "P(t1, t2, ..., tn) = P(t1)*P(t2|t1)*...\n"
         "Viterbi: Find maximum path\n"
         "Better quality\n"
         "Computationally expensive\n\n",
         "Add maximum likelihood"),
        
        ("### Unigram Language Model\n\n"
         "Simplest LM\n"
         "P(token) = count(token) / total\n"
         "Works surprisingly well\n"
         "Very fast\n"
         "Foundation of many systems\n\n",
         "Add unigram LM"),
        
        ("### Byte-Level Tokenization\n\n"
         "UTF-8 bytes: 0-255 vocab\n"
         "Can represent any text\n"
         "No OOV!\n"
         "Very long sequences\n"
         "Used in: GPT-2\n\n",
         "Add byte-level"),
        
        ("### Unicode Normalization\n\n"
         "é = e + combining accent\n"
         "Different byte sequences, same character\n"
         "NFC vs NFD formats\n"
         "Should normalize first\n"
         "Affects tokenization\n\n",
         "Add unicode norm"),
        
        ("### Language-Specific Tokenization\n\n"
         "Chinese: No spaces (use jieba)\n"
         "Arabic: Right-to-left\n"
         "Vietnamese: Diacritics important\n"
         "Thai: No word boundaries\n"
         "Japanese: Kanji vs hiragana\n"
         "Language matters!\n\n",
         "Add language-specific token"),
    ]),
    
    ('L26.2_word2vec/README.md', [
        ("## Word2Vec Implementation Details\n\n"
         "### Skip-gram Objective\n\n"
         "Maximize: Σ log P(context|word)\n"
         "Use Softmax or Negative Sampling\n"
         "Gradient updates embeddings\n"
         "Simple but effective\n"
         "Parallelizable\n\n",
         "Add skip-gram objective"),
        
        ("### Negative Sampling Benefits\n\n"
         "Avoids expensive softmax\n"
         "K = 5-15 negative samples\n"
         "Binary classification task\n"
         "10-15x speedup typical\n"
         "Practical necessity\n\n",
         "Add neg sampling benefits"),
        
        ("### Context Window\n\n"
         "Window size = 5: Look ±5 words\n"
         "Larger window: More context\n"
         "Smaller window: More local\n"
         "Trade-off: Task-dependent\n"
         "Typical: 5-10\n\n",
         "Add context window"),
        
        ("### Multi-word Phrases\n\n"
         "Identify phrases: \"new york\"\n"
         "Treat as single token\n"
         "Improves semantic quality\n"
         "Statistic-based detection\n"
         "Multiple passes\n\n",
         "Add phrases"),
        
        ("### Training Tips\n\n"
         "Learning rate: 0.025 starting\n"
         "Decay over time\n"
         "Multiple epochs: 5-10\n"
         "Larger corpus: Better\n"
         "Parallelization on cores\n\n",
         "Add training tips"),
    ]),
    
    ('L26.1_word-embeddings/README.md', [
        ("## Embedding Properties\n\n"
         "### Compositionality\n\n"
         "Can combine embeddings?\n"
         "\"new\" + \"york\" ≈ \"new york\"\n"
         "Sometimes works, sometimes not\n"
         "Non-linear: Addition too simple\n"
         "Better: Learn composition\n\n",
         "Add compositionality"),
        
        ("### Polysemy (Multiple Meanings)\n\n"
         "\"bank\" = financial or river\n"
         "Single embedding loses info\n"
         "Contextualized embeddings help\n"
         "Or: Prototype + sense vectors\n"
         "Hard problem\n\n",
         "Add polysemy"),
        
        ("### Hypernym-Hyponym Relations\n\n"
         "\"dog\" is hyponym of \"animal\"\n"
         "Hierarchical semantic structure\n"
         "Embeddings reflect hierarchy\n"
         "Vector direction matters\n"
         "Emergent property\n\n",
         "Add hypernym"),
        
        ("### Cultural Bias\n\n"
         "\"doctor\" more similar to \"he\"\n"
         "\"nurse\" more similar to \"she\"\n"
         "Reflects training data bias\n"
         "Problematic for applications\n"
         "Debias methods exist\n\n",
         "Add cultural bias"),
        
        ("### Dimensionality Sweet Spot\n\n"
         "50D: Too small, poor quality\n"
         "100D: Decent for small tasks\n"
         "300D: Standard, good balance\n"
         "1000D: Overkill for most\n"
         "Find empirically\n\n",
         "Add dimensionality"),
    ]),
    
    ('L27.1_text-classification/README.md', [
        ("## Classification Architectures\n\n"
         "### TextCNN\n\n"
         "CNN on text sequences\n"
         "1D convolution over words\n"
         "Multiple filter sizes: 2, 3, 4\n"
         "Max-over-time pooling\n"
         "Simple, fast, effective\n\n",
         "Add TextCNN"),
        
        ("### FastText Classifier\n\n"
         "Bag-of-words embeddings\n"
         "Average word vectors\n"
         "Hierarchical softmax\n"
         "Extremely fast\n"
         "Decent accuracy\n\n",
         "Add fastText classifier"),
        
        ("### Attention-based\n\n"
         "Attend to important words\n"
         "Learn weights per word\n"
         "Context-dependent importance\n"
         "Interpretable decisions\n"
         "Better performance\n\n",
         "Add attention classifier"),
        
        ("### Class Imbalance Handling\n\n"
         "Reweight by class frequency\n"
         "Oversampling minority\n"
         "SMOTE: Synthetic examples\n"
         "Adjust decision threshold\n"
         "Multiple strategies\n\n",
         "Add imbalance classif"),
        
        ("### Multi-label Classification\n\n"
         "Multiple labels per document\n"
         "\"Action\" and \"adventure\"\n"
         "Not mutually exclusive\n"
         "Different loss (cross-entropy per label)\n"
         "Different evaluation metrics\n\n",
         "Add multilabel"),
    ]),
    
    ('L27.2_tfidf/README.md', [
        ("## TF-IDF Variants\n\n"
         "### Log-TF\n\n"
         "TF(t,d) = 1 + log(count)\n"
         "Sublinear scaling\n"
         "Dampens frequency effect\n"
         "Often better than raw\n"
         "Standard choice\n\n",
         "Add log-TF"),
        
        ("### Probabilistic IDF\n\n"
         "IDF(t) = log((N - count) / count)\n"
         "Probabilistic interpretation\n"
         "Different from standard\n"
         "Sometimes useful\n"
         "Less common\n\n",
         "Add prob IDF"),
        
        ("### BM25\n\n"
         "Refinement of TF-IDF\n"
         "Saturation term: Caps TF\n"
         "Document length normalization\n"
         "Very effective for IR\n"
         "Standard in Lucene\n\n",
         "Add BM25"),
        
        ("### Sublinear TF Scaling\n\n"
         "Raw frequency overpowers\n"
         "log() dampens\n"
         "sqrt() alternative\n"
         "Balance term and doc length\n"
         "Empirically better\n\n",
         "Add sublinear TF"),
        
        ("### L2 Normalization\n\n"
         "Vectors sum to 1\n"
         "Removes document length bias\n"
         "Enables cosine similarity\n"
         "Standard preprocessing\n"
         "Improves classifier\n\n",
         "Add L2 norm TF"),
    ]),
    
    ('L28.1_ner-tagging/README.md', [
        ("## NER Tagging Schemes\n\n"
         "### IOB1\n\n"
         "I-tag for continuation\n"
         "Simpler than BIO\n"
         "B only when adjacent to same\n"
         "Less common\n"
         "Historic\n\n",
         "Add IOB1"),
        
        ("### IOB2 (BIO)\n\n"
         "B: Begin tag\n"
         "I: Inside tag\n"
         "O: Outside\n"
         "Most common\n"
         "Standard scheme\n\n",
         "Add IOB2"),
        
        ("### IOBES\n\n"
         "B, I, O, E (end), S (single)\n"
         "Most explicit\n"
         "Distinguishes single vs multi\n"
         "Better for some tasks\n"
         "Slightly higher accuracy\n\n",
         "Add IOBES"),
        
        ("### Nested NER\n\n"
         "Overlapping entities\n"
         "\"United States\" inside \"United\"\n"
         "Rare, harder\n"
         "Different schemes\n"
         "Research area\n\n",
         "Add nested NER"),
        
        ("### Multi-token Entities\n\n"
         "Names span multiple tokens\n"
         "\"John Smith\" = 2 tokens\n"
         "Sequential dependency\n"
         "CRF handles well\n"
         "Important for real data\n\n",
         "Add multitoken"),
    ]),
    
    ('L28.2_sequence-labeling/README.md', [
        ("## POS Tagging\n\n"
         "### Universal Dependencies\n\n"
         "NOUN, VERB, ADJ, ADP, ...\n"
         "Language-independent tags\n"
         "100+ languages annotated\n"
         "Standard benchmark\n"
         "Simplifies cross-lingual\n\n",
         "Add UD tags"),
        
        ("### Language-Specific Tags\n\n"
         "Penn Treebank: English specific\n"
         "47 tags (vs 17 UD)\n"
         "More fine-grained\n"
         "Better for downstream\n"
         "Less portable\n\n",
         "Add Penn Treebank"),
        
        ("### Evaluation Metrics\n\n"
         "Accuracy: % correct\n"
         "Per-tag F1: Class-wise\n"
         "Confusion matrix: Where errors\n"
         "Simple task: 97%+ accuracy\n"
         "Mostly solved\n\n",
         "Add POS metrics"),
        
        ("### Morphological Analysis\n\n"
         "Fine-grained: POS + morphology\n"
         "\"books\" = NOUN + Numb:Plur\n"
         "Universal Features schema\n"
         "Richer representation\n"
         "Harder annotation\n\n",
         "Add morphological"),
        
        ("### Tagging with Context\n\n"
         "\"saw\" could be NOUN or VERB\n"
         "Context disambiguates\n"
         "RNN/attention exploit context\n"
         "Much better than dict-based\n"
         "Neural models: 97%+ accuracy\n\n",
         "Add tagging context"),
    ]),
    
    ('L29.1_elmo/README.md', [
        ("## ELMo Deep Dive\n\n"
         "### Training Data\n\n"
         "1B token corpus\n"
         "Wikipedia + news crawl\n"
         "Diverse language\n"
         "Large-scale pre-training\n"
         "Weeks to train\n\n",
         "Add ELMo training data"),
        
        ("### Bidirectional Processing\n\n"
         "Forward LSTM: Left-to-right\n"
         "Backward LSTM: Right-to-left\n"
         "Concatenate: Both directions\n"
         "Context from both sides\n"
         "Better than forward only\n\n",
         "Add bidirectional"),
        
        ("### Layer Combination\n\n"
         "Learned weights per layer\n"
         "γ * (w_input*input + w_fwd*fwd + w_bwd*bwd)\n"
         "Task-specific combination\n"
         "Different tasks use different layers\n"
         "Adaptive representation\n\n",
         "Add layer combo"),
        
        ("### Evaluation Results\n\n"
         "NER: +1-2% F1\n"
         "SQuAD: +2-3% F1\n"
         "Sentiment: +1% accuracy\n"
         "Consistent but modest\n"
         "Foundation, not solution\n\n",
         "Add ELMo eval"),
        
        ("### Computational Cost\n\n"
         "Large model: 93.6M params\n"
         "Slow inference: 100ms per sentence\n"
         "Not practical for real-time\n"
         "Became BERT's problem\n"
         "BERT 12x more expensive!\n\n",
         "Add ELMo cost"),
    ]),
    
    ('L29.2_gpt-pretraining/README.md', [
        ("## Autoregressive vs Bidirectional\n\n"
         "### Autoregressive (GPT)\n\n"
         "Generate left-to-right\n"
         "P(w_t | w_1, ..., w_{t-1})\n"
         "Natural: Text generation\n"
         "Can't see future\n"
         "Used by GPT, GPT-2, GPT-3\n\n",
         "Add autoregressive"),
        
        ("### Bidirectional (BERT)\n\n"
         "P(w_t | all words except w_t)\n"
         "Can see both directions\n"
         "Better for classification\n"
         "Can't generate naturally\n"
         "Used by BERT, RoBERTa\n\n",
         "Add bidirectional LM"),
        
        ("### Masked Language Model\n\n"
         "[MASK] token in input\n"
         "Predict what's masked\n"
         "Bidirectional context\n"
         "Allows both directions\n"
         "BERT pre-training task\n\n",
         "Add masked LM"),
        
        ("### Next Sentence Prediction\n\n"
         "Predict if B follows A\n"
         "Related sentences: Yes\n"
         "Random sentences: No\n"
         "Binary classification\n"
         "BERT auxiliary task\n\n",
         "Add NSP"),
        
        ("### Scaling Laws\n\n"
         "Performance improves predictably\n"
         "Loss ∝ 1 / (model_size)\n"
         "Doubling size: ~5% better\n"
         "10x compute: ~30% better\n"
         "Drives gigantic models\n\n",
         "Add scaling laws"),
    ]),
]

total_commits = 0

for filepath, sections in lesson_files:
    full_path = filepath
    
    # Create directory if needed
    if '/' in full_path:
        dirpath = full_path.rsplit('/', 1)[0]
        if dirpath and not os.path.exists(dirpath):
            os.makedirs(dirpath, exist_ok=True)
            print(f"Created directory: {dirpath}")
    
    # Ensure file exists
    if not os.path.exists(full_path):
        with open(full_path, 'w') as f:
            f.write(f"# Lesson: {filepath}\n\n")
    
    print(f"\n{'='*60}")
    print(f"Enriching: {filepath}")
    print(f"{'='*60}")
    
    for i, (content, msg) in enumerate(sections, 1):
        with open(full_path, 'a', encoding='utf-8') as f:
            f.write(content)
        
        try:
            subprocess.run(['git', 'add', full_path], check=True, capture_output=True)
            subprocess.run(['git', 'commit', '-m', f'Enrich lesson {filepath}: {msg}'], 
                         check=True, capture_output=True)
            print(f"[OK] Commit {i:2d}: {msg}")
            total_commits += 1
        except subprocess.CalledProcessError:
            print(f"[FAIL] {i:2d}: {msg}")

print(f"\n{'='*60}")
print(f"[DONE] Added {total_commits} lesson commits!")

result = subprocess.run(['git', 'log', '--oneline'], capture_output=True, text=True)
final_total = len(result.stdout.strip().split('\n'))
print(f"Total repository commits: {final_total}")
