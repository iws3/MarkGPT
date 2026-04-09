#!/usr/bin/env python3
"""
Module-05 enrichment part 3 - 85 commits
ELMo, contextual embeddings, and pre-Transformer models
"""
import subprocess
import os

os.chdir(r'c:\Users\THE EYE INFORMATIQUE\OneDrive\Desktop\All\MarkGPT-LLM-Curriculum\MarkGPT-LLM-Curriculum\modules\module-05')

sections = [
    ("## ELMo: Embeddings from Language Models\n\n"
     "### The Great Insight (2018)\n\n"
     "Language modeling captures context\n"
     "Hidden states encode rich information\n"
     "Use as embeddings instead of words\n"
     "Different representation per context\n"
     "Solves word sense disambiguation\n"
     "Revolutionized NLP in 2018\n\n",
     "Add ELMo insight"),
    
    ("### Architecture\n\n"
     "Bidirectional LSTM language model\n"
     "Forward LSTM: Left to right\n"
     "Backward LSTM: Right to left\n"
     "3 layers: Input + 2 hidden\n"
     "Each token: Multiple representation options\n"
     "Combine all: Weighted sum of layers\n\n",
     "Add ELMo arch"),
    
    ("### Training\n\n"
     "Language modeling task:\n"
     "Predict next token given context\n"
     "Unsupervised: Only need text\n"
     "Scale: 1B token corpus\n"
     "Result: Rich contextual representations\n"
     "Weights: Learned task-specifically\n\n",
     "Add ELMo training"),
    
    ("### Usage\n\n"
     "1. Pre-train LSTM language model\n"
     "2. Extract hidden states for token\n"
     "3. Concatenate/weight combinations\n"
     "4. Use as features for downstream\n"
     "5. Fine-tune weights on task\n"
     "Simple layer-weighting learned\n\n",
     "Add ELMo usage"),
    
    ("### Results\n\n"
     "NER: +2% F1 on CoNLL\n"
     "Sentiment: +1-2% accuracy\n"
     "QA: +2-3% on SQuAD\n"
     "Consistent improvements\n"
     "Modest but reliable gains\n"
     "Foundation for BERT\n\n",
     "Add ELMo results"),
    
    ("### Limitations\n\n"
     "Unidirectional for future context\n"
     "Backward pass separate\n"
     "Slow to compute\n"
     "Large model size\n"
     "Replaced by BERT (bidirectional)\n"
     "But: Key insight remains (contextual)\n\n",
     "Add ELMo limits"),
    
    ("## Pre-Transformer Context\n\n"
     "### Timeline\n\n"
     "2013: Word2Vec\n"
     "2014: GloVe\n"
     "2016: fastText\n"
     "2018: ELMo (game changer)\n"
     "2018: BERT (even better)\n"
     "2018: GPT (autoregressive)\n"
     "2019: Beyond\n\n",
     "Add SOTA timeline"),
    
    ("## Feature Engineering\n\n"
     "### Traditional NLP Pipeline\n\n"
     "1. Tokenization\n"
     "2. POS tagging\n"
     "3. Parsing\n"
     "4. Named entities\n"
     "5. Manual feature extraction\n"
     "6. Machine learning classifier\n"
     "Was state-of-art pre-deep learning\n\n",
     "Add traditional pipeline"),
    
    ("### Feature Types\n\n"
     "Lexical: Word, lemma, stem\n"
     "Morphological: POS tag, suffixes\n"
     "Syntactic: Dependency relations\n"
     "Semantic: Entity types, meanings\n"
     "Contextual: Surrounding words\n"
     "External: Knowledge bases\n\n",
     "Add feature types"),
    
    ("### TF-IDF\n\n"
     "Term Frequency-Inverse Document Freq\n"
     "TF(t, d) = count(t in d) / |d|\n"
     "IDF(t) = log(N / count(docs with t))\n"
     "TF-IDF(t, d) = TF(t, d) * IDF(t)\n"
     "Downweights common words\n"
     "Simple but effective baseline\n\n",
     "Add TF-IDF"),
    
    ("### Bag of Words\n\n"
     "Simplest representation\n"
     "Count each word occurrence\n"
     "Create feature vector\n"
     "Loses word order\n"
     "High-dimensional but sparse\n"
     "Still baseline for many tasks\n\n",
     "Add BoW"),
    
    ("### N-grams\n\n"
     "Unigrams: Individual words\n"
     "Bigrams: Two consecutive words\n"
     "Trigrams: Three consecutive\n"
     "Encodes local word order\n"
     "Partially solves word order loss\n"
     "Increase dimensionality\n\n",
     "Add n-grams"),
    
    ("## Text Classification\n\n"
     "### Problem Definition\n\n"
     "Input: Document (text)\n"
     "Output: Category label (discrete)\n"
     "Task: Learn function mapping\n"
     "Many real applications\n"
     "Sentiment, spam, topic, intent\n"
     "Foundation for NLP\n\n",
     "Add classification intro"),
    
    ("### Simple Baseline\n\n"
     "1. TF-IDF vectorization\n"
     "2. Logistic regression\n"
     "3. Get predictions\n"
     "4. Evaluate on test set\n"
     "Often surprisingly good\n"
     "Fast, interpretable, reliable\n\n",
     "Add baseline"),
    
    ("### Deep Learning Approach\n\n"
     "1. Tokenize text\n"
     "2. Lookup embeddings\n"
     "3. Pool/RNN over sequence\n"
     "4. Dense layers\n"
     "5. Softmax for class probs\n"
     "Better with large labeled data\n\n",
     "Add DL classification"),
    
    ("### Datasets\n\n"
     "20 newsgroups: 20 categories\n"
     "Movie reviews: Sentiment (2 class)\n"
     "AG News: News categorization\n"
     "DBpedia: Wikipedia categories\n"
     "TREC: Question type classification\n"
     "Standard benchmarks for evaluation\n\n",
     "Add classification data"),
    
    ("### Evaluation Metrics\n\n"
     "Accuracy: % correct (balanced data)\n"
     "Precision: TP / (TP + FP)\n"
     "Recall: TP / (TP + FN)\n"
     "F1: 2 * (P*R) / (P+R) (unbalanced)\n"
     "Macro F1: Average per class\n"
     "Choose based on use case\n\n",
     "Add eval metrics"),
    
    ("## Information Extraction\n\n"
     "### Named Entity Recognition\n\n"
     "Identify entities: People, places, orgs\n"
     "Tag each token with entity type\n"
     "Sequence labeling task\n"
     "BIO tagging scheme\n"
     "B: Begin, I: Inside, O: Outside\n"
     "Fundamental IE task\n\n",
     "Add NER"),
    
    ("### Relation Extraction\n\n"
     "Identify relationships between entities\n"
     "Example: Company_X founded_in Year_Y\n"
     "Given: Named entities\n"
     "Find: Relationship type\n"
     "Applications: Knowledge base construction\n"
     "Harder than NER\n\n",
     "Add relation extraction"),
    
    ("### Dependency Parsing\n\n"
     "Identify grammatical structure\n"
     "Who depends on whom\n"
     "Arc labels: Subject, object, etc.\n"
     "Tree structure (mostly)\n"
     "Transition-based systems\n"
     "Graph-based systems\n\n",
     "Add parsing"),
    
    ("## Sequence Labeling\n\n"
     "### Problem Setup\n\n"
     "Input: Sequence of tokens\n"
     "Output: Label per token\n"
     "Constraints: Local dependencies\n"
     "Examples: POS, NER, chunking\n"
     "More complex than classification\n"
     "Order and structure matter\n\n",
     "Add seq labeling"),
    
    ("### CRF (Conditional Random Fields)\n\n"
     "Probabilistic sequence model\n"
     "Conditions on observations\n"
     "Globally normalized\n"
     "Handles dependencies\n"
     "Can't predict impossible sequences\n"
     "Good baseline for labeling\n\n",
     "Add CRF"),
    
    ("### BiLSTM-CRF\n\n"
     "BiLSTM: Encode sequence\n"
     "CRF: Decode with constraints\n"
     "State-of-art pre-Transformer\n"
     "High performance\n"
     "Good for structured prediction\n"
     "Popular in industry\n\n",
     "Add BiLSTM-CRF"),
    
    ("## Language Models (Pre-Transformer)\n\n"
     "### N-gram Language Models\n\n"
     "P(w_t | w_{t-n+1}, ..., w_{t-1})\n"
     "Count-based: Simple and effective\n"
     "Backoff: Handle OOV\n"
     "Smoothing: Unseen n-grams\n"
     "Fast inference\n"
     "Limited by sparsity\n\n",
     "Add ngram LM"),
    
    ("### RNN Language Models\n\n"
     "Recurrent: Process token by token\n"
     "Hidden state: Encodes context\n"
     "Unbounded context\n"
     "Much better than n-grams\n"
     "Training: Backprop through time\n"
     "Slow but powerful\n\n",
     "Add RNN LM"),
    
    ("### Perplexity\n\n"
     "How surprised model is\n"
     "PP = 2^(cross-entropy)\n"
     "Lower is better\n"
     "Baseline: Unigram LM (~1000)\n"
     "LSTM: ~100-200\n"
     "Good metric for LM quality\n\n",
     "Add perplexity LM"),
    
    ("## Text Similarity\n\n"
     "### Cosine Similarity\n\n"
     "For embeddings or TF-IDF vectors\n"
     "cos(u, v) = (u . v) / (||u|| ||v||)\n"
     "Range: [-1, 1], typically [0, 1]\n"
     "1: Identical direction\n"
     "0: Orthogonal\n"
     "-1: Opposite\n\n",
     "Add cosine similarity"),
    
    ("### Word Movers Distance\n\n"
     "Treat documents as distributions\n"
     "Distance between word clouds\n"
     "Uses word embeddings\n"
     "More semantic than BoW\n"
     "Computationally expensive\n"
     "Good for short documents\n\n",
     "Add WMD"),
    
    ("### Semantic Similarity Tasks\n\n"
     "STS Benchmark: Sentence pairs\n"
     "Rate 0-5 similarity\n"
     "450K+ sentence pairs\n"
     "Standard evaluation benchmark\n"
     "Predicts perceptual similarity\n"
     "Challenging: Requires understanding\n\n",
     "Add STS"),
    
    ("## Paraphrase Detection\n\n"
     "### Problem\n\n"
     "Identify if sentences mean same\n"
     "Binary classification\n"
     "Different words, same meaning\n"
     "Or: Similarity rating\n"
     "Useful for plagiarism detection\n"
     "Helps question answering\n\n",
     "Add paraphrase"),
    
    ("### Approaches\n\n"
     "1. Edit distance: Too simple\n"
     "2. TF-IDF similarity: Better\n"
     "3. Word embeddings: Much better\n"
     "4. ELMo/BERT: State-of-art\n"
     "5. Fine-tuned transformer: Best\n"
     "Modern: >> 95% accuracy\n\n",
     "Add paraphrase approaches"),
    
    ("## Machine Translation (Pre-Seq2seq)\n\n"
     "### Statistical Machine Translation\n\n"
     "P(t|s) = P(t) * P(s|t) / P(s)\n"
     "Language model: P(t)\n"
     "Translation model: P(s|t)\n"
     "Phrase-based capturing\n"
     "Reordering models\n"
     "Dominated until 2016\n\n",
     "Add SMT"),
    
    ("### Seq2Seq Breakthrough\n\n"
     "Encoder-decoder RNNs\n"
     "End-to-end learning\n"
     "Outperformed SMT dramatically\n"
     "BLEU score: 35+ vs SMT 25+\n"
     "Simpler system\n"
     "Single model, not pipeline\n\n",
     "Add seq2seq MT"),
    
    ("## Sentiment Analysis\n\n"
     "### Task Definition\n\n"
     "Classify positive vs negative\n"
     "Or: Rate on scale 1-5\n"
     "Common application\n"
     "Good benchmark problem\n"
     "Many datasets available\n"
     "Progression: BoW → embeddings → transformers\n\n",
     "Add sentiment intro"),
    
    ("### Challenges\n\n"
     "Sarcasm: \"This is great!\" (negative)\n"
     "Negation: \"not bad\" = positive\n"
     "Aspect sentiment: \"Good food, bad service\"\n"
     "Multi-class: Fine-grained opinions\n"
     "Domain: Language varies by domain\n"
     "More complex than it seems\n\n",
     "Add sentiment challenges"),
    
    ("### Datasets\n\n"
     "Movie reviews: Binary, straightforward\n"
     "IMDB: 25K training examples\n"
     "SemEval: Multiple languages\n"
     "Amazon reviews: Large scale\n"
     "Twitter: Informal, sarcasm\n"
     "Different difficulty levels\n\n",
     "Add sentiment data"),
    
    ("## Vector Space Models\n\n"
     "### Distributional Similarity\n\n"
     "Words as points in space\n"
     "Distance encodes similarity\n"
     "Clusters emerge automatically\n"
     "Unsupervised discovery\n"
     "Remarkable regularities\n"
     "Foundation of modern NLP\n\n",
     "Add VSM"),
    
    ("### Locality Sensitive Hashing\n\n"
     "Fast nearest neighbor search\n"
     "Hash similar vectors together\n"
     "Approximate but efficient\n"
     "Million vector queries: Milliseconds\n"
     "Used in production systems\n"
     "Scales to web\n\n",
     "Add LSH"),
    
    ("## Knowledge Distillation Pre-Transformers\n\n"
     "### Compressing Models\n\n"
     "Large model: Better performance\n"
     "Small model: Faster inference\n"
     "Distillation: Transfer knowledge\n"
     "Student learns from teacher\n"
     "Soft targets via temperature\n"
     "Pre-dates transformers\n\n",
     "Add distillation pretransformer"),
    
    ("## Attention Before Transformers\n\n"
     "### Attention for Seq2Seq\n\n"
     "Bottleneck: Single vector from encoder\n"
     "Attention: Look at all encoder states\n"
     "Different attention per decoder step\n"
     "Context vector: Weighted sum\n"
     "Huge improvement for translation\n"
     "Next: Make entire model attention\n\n",
     "Add attention seq2seq"),
    
    ("## Embedding Analysis\n\n"
     "### Bias in Embeddings\n\n"
     "man:programmer ≈ woman:homemaker\n"
     "Reflects training data bias\n"
     "Problematic for applications\n"
     "Detection: Analogy tests\n"
     "Mitigation: Debias embeddings\n"
     "Ongoing research\n\n",
     "Add embedding bias"),
    
    ("### Word Sense Disambiguation\n\n"
     "\"Bank\" = financial vs river\n"
     "Single embedding fails\n"
     "Solution: Contextualized (ELMo+)\n"
     "Different vectors per context\n"
     "Better semantic understanding\n"
     "Jumping off point to BERT\n\n",
     "Add word sense"),
    
    ("## Cross-lingual Transfer\n\n"
     "### Multilingual Embeddings\n\n"
     "Single space across languages\n"
     "Similar concepts align\n"
     "Zero-shot translation\n"
     "mBERT enables this\n"
     "Word2Vec: Language-specific\n"
     "BERT: Unified space\n\n",
     "Add cross-lingual"),
    
    ("## Module 05 Summary\n\n"
     "**Concepts Learned**\n"
     "- Tokenization methods and trade-offs\n"
     "- Word embeddings (Word2Vec, GloVe, fastText)\n"
     "- Static vs contextualized embeddings\n"
     "- ELMo and pre-Transformer models\n"
     "- Classical NLP features (TF-IDF, n-grams)\n"
     "- Text classification and sequence labeling\n"
     "- Information extraction\n"
     "- Language modeling\n"
     "- Machine translation and seq2seq\n"
     "- Production systems\n\n",
     "Add module summary"),
]

readme_path = 'README.md'

print(f"Starting module-05 part 3 with {len(sections)} commits...")
print("=" * 60)

for i, (content, msg) in enumerate(sections, 1):
    with open(readme_path, 'a', encoding='utf-8') as f:
        f.write(content)
    
    try:
        subprocess.run(['git', 'add', 'README.md'], check=True, capture_output=True)
        subprocess.run(['git', 'commit', '-m', f'Enrich module-05 part3 {i}: {msg}'], check=True, capture_output=True)
        print(f"[OK] Commit {i:2d}/{len(sections)}: {msg}")
    except subprocess.CalledProcessError:
        print(f"[FAIL] Part 3 {i:2d}: {msg}")

print("=" * 60)
print(f"[DONE] Part 3 added {len(sections)} commits!")

result = subprocess.run(['git', 'log', '--oneline'], capture_output=True, text=True)
total = len(result.stdout.strip().split('\n'))
print(f"Total repository commits: {total}")
