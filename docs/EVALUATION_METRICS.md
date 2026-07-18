# Evaluation Metrics & Benchmarking

## Automatic Metrics

### Perplexity (Language Model Specific)

$$\text{PPL} = e^{\frac{1}{N} \sum_{i=1}^{N} -\log P(y_i | x_{<i})}$$

Measures "surprise" of model at predicting next token.

```python
def compute_perplexity(model, tokenizer, text):
    """Compute perplexity on text."""
    
    tokens = tokenizer.encode(text, return_tensors='pt').to('cuda')
    
    with torch.no_grad():
        logits = model(tokens).logits
    
    # Shift for predicting next token
    logits = logits[:, :-1, :].contiguous()
    targets = tokens[:, 1:].contiguous()
    
    # Cross-entropy
    ce_loss = F.cross_entropy(
        logits.view(-1, logits.shape[-1]),
        targets.view(-1)
    )
    
    # Perplexity
    ppl = torch.exp(ce_loss)
    
    return ppl.item()

# Interpretation:
# PPL = 20 → Model thinks next token has ~20 equally likely options
# PPL = 100 → Very uncertain
```

### BLEU Score (Machine Translation)

Fraction of generated n-grams matching reference:

$$\text{BLEU} = \text{BP} \cdot \exp\left(\sum_{n=1}^{N} w_n \log p_n\right)$$

```python
from nltk.translate.bleu_score import sentence_bleu

hypothesis = "the cat sat on the mat".split()
reference = "the cat is sitting on the mat".split()

bleu = sentence_bleu(
    [reference],
    hypothesis,
    weights=(0.25, 0.25, 0.25, 0.25)  # 1-grams, 2-grams, ..., 4-grams
)

# Result: 0-1 scale
# BLEU > 0.3: Decent translation
# BLEU > 0.5: Good translation
```

### ROUGE Score (Summarization)

Recall-based (how much of reference is in output):

```python
from rouge_score import rouge_scorer

scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)

reference = "Cats are fluffy animals that purr."
generated = "Cats are cute fluffy animals."

scores = scorer.score(reference, generated)

# ROUGE-1: Unigram overlap
# ROUGE-L: Longest common subsequence
print(scores['rouge1'].fmeasure)  # F1 score
```

### METEOR (Better than BLEU)

Considers synonyms and word order:

```python
from nltk.translate.meteor_score import meteor_score

hypothesis = "the cat sat on the mat"
reference = "the cat is sitting on the mat"

score = meteor_score([reference], hypothesis)  # 0-1 scale
```

---

## Domain-Specific Metrics (MarkGPT)

### Multilingual Evaluation

```python
def multilingual_eval(model, tokenizer):
    """Evaluate on English and Banso separately."""
    
    results = {}
    
    # English benchmark
    en_corpus = load_english_test()
    en_ppl = compute_perplexity(model, tokenizer, en_corpus)
    results['english_ppl'] = en_ppl
    
    # Banso benchmark
    banso_corpus = load_banso_test()
    banso_ppl = compute_perplexity(model, tokenizer, banso_corpus)
    results['banso_ppl'] = banso_ppl
    
    # Translation quality (if parallel data)
    en_banso_pairs = load_en_banso_pairs()
    bleu = evaluate_translation(model, en_banso_pairs)
    results['translation_bleu'] = bleu
    
    # Language identification (should stay in language)
    lang_accuracy = evaluate_language_preservation(model)
    results['language_preservation'] = lang_accuracy
    
    return results
```

### Code-Switching Evaluation

```python
def evaluate_code_switching(model, tokenizer):
    """Evaluate bilingual generation."""
    
    prompts_en_start = ["Write code for", "The algorithm"]
    prompts_banso_start = ["Bhala", "Inkosi"]
    
    # Start in English, should maintain
    en_purity_scores = []
    for prompt in prompts_en_start:
        output = model.generate(prompt, max_tokens=50)
        en_ratio = estimate_english_fraction(output)
        en_purity_scores.append(en_ratio)
    
    # Start in Banso, should maintain  
    banso_purity_scores = []
    for prompt in prompts_banso_start:
        output = model.generate(prompt, max_tokens=50)
        banso_ratio = estimate_banso_fraction(output)
        banso_purity_scores.append(banso_ratio)
    
    return {
        'english_purity': np.mean(en_purity_scores),
        'banso_purity': np.mean(banso_purity_scores),
        'avg_purity': (np.mean(en_purity_scores) + np.mean(banso_purity_scores)) / 2
    }
```

---

## Downstream Tasks (Extrinsic Evaluation)

### Sentiment Analysis

```python
def evaluate_downstream_sentiment(model, tokenizer):
    """Use MarkGPT for sentiment classification (few-shot)."""
    
    # Few-shot examples
    few_shot = """
    Positive: This movie is amazing!
    Negative: I hate waiting in lines.
    Positive: What a wonderful day!
    Negative: The food was terrible.
    
    "The service was excellent" ->
    """
    
    # Generate label
    output = model.generate(few_shot, max_tokens=1)
    
    # Parse
    if 'Positive' in output:
        return 'positive'
    else:
        return 'negative'

# Evaluate on test set
correct = 0
for text, label in test_set:
    pred = evaluate_downstream_sentiment(model, tokenizer)
    if pred == label:
        correct += 1

accuracy = correct / len(test_set)
```

### Reading Comprehension

```python
def evaluate_reading_comprehension(model, tokenizer):
    """Q&A based on context."""
    
    context = "Paris is the capital of France and is known for the Eiffel Tower."
    question = "What is the capital of France?"
    
    prompt = f"""Context: {context}
Question: {question}
Answer:"""
    
    answer = model.generate(prompt, max_tokens=10)
    
    # Check if answer contains correct entity
    if 'Paris' in answer:
        return True
    return False
```

---

## Human Evaluation

### Inter-Rater Agreement (Kappa)

```python
from sklearn.metrics import cohen_kappa_score

# Two annotators rate outputs (e.g., 1-5 scale)
rater1 = [5, 4, 3, 5, 2]
rater2 = [5, 3, 4, 5, 1]

kappa = cohen_kappa_score(rater1, rater2)

# Interpretation:
# kappa > 0.8: Strong agreement
# kappa > 0.6: Moderate agreement
# kappa > 0.4: Fair agreement
```

### Annotation Template

```python
def human_eval_interface():
    """Web interface for human evaluation."""
    
    rubric = {
        'fluency': (1, 5, 'Is the text grammatically correct and natural?'),
        'relevance': (1, 5, 'Does it address the prompt?'),
        'factuality': (1, 5, 'Are facts accurate?'),
        'coherence': (1, 5, 'Is text logically coherent?'),
        'language_preservation': (1, 5, 'Does it stay in the target language?'),
    }
    
    # For each generated output
    scores = {}
    for criterion, (min_val, max_val, question) in rubric.items():
        print(f"{criterion}: {question}")
        scores[criterion] = int(input(f"  Rating ({min_val}-{max_val}): "))
    
    return scores

# Aggregate
evaluations = [human_eval_interface() for _ in range(100_outputs)]
avg_score = np.mean([e['fluency'] for e in evaluations])
```

---

## Benchmark Results (MarkGPT Expected)

### Nano (70M)

```
English:
  - Perplexity: 45-50
  - WIKITEXT: 50.0 PPL
  
Banso:
  - Perplexity: 80-100
  - Translation BLEU: 15-20

Downstream (few-shot):
  - Sentiment: 65% (weak)
  - Q&A: 40% (random)
```

### Small (200M)

```
English:
  - Perplexity: 25-30
  - WIKITEXT: 30.5 PPL
  
Banso:
  - Perplexity: 50-60
  - Translation BLEU: 22-28

Downstream (few-shot):
  - Sentiment: 75% (decent)
  - Q&A: 55% (better)
```

### Base (500M) - TARGET

```
English:
  - Perplexity: 15-20
  - WIKITEXT: 20.2 PPL (competitive!)
  - SQuAD: 75% (strong)
  
Banso:
  - Perplexity: 35-45
  - Translation BLEU: 30-35
  - Language preservation: 85%

Downstream (few-shot):
  - Sentiment: 82%
  - Q&A: 70%
  - Code-switching: Fluent
```

---

## Benchmark Scripts

```python
class MarkGPTBenchmark:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
    
    def run_all_benchmarks(self):
        """Comprehensive evaluation."""
        
        results = {
            'automatic': self.automatic_metrics(),
            'downstream': self.downstream_tasks(),
            'multilingual': self.multilingual_evaluation(),
            'efficiency': self.efficiency_metrics(),
        }
        
        self.save_results(results)
        return results
    
    def automatic_metrics(self):
        """BLEU, ROUGE, Perplexity."""
        return {
            'wikitext_ppl': self.compute_wikitext_ppl(),
            'banso_corpus_ppl': self.compute_banso_ppl(),
            'translation_bleu': self.compute_translation_bleu(),
        }
    
    def downstream_tasks(self):
        """Sentiment, QA, etc."""
        return {
            'sentiment_accuracy': self.eval_sentiment(),
            'qa_accuracy': self.eval_qa(),
        }
    
    def efficiency_metrics(self):
        """Speed, memory."""
        return {
            'tokens_per_second': self.measure_throughput(),
            'memory_gb': self.measure_memory(),
            'size_mb': self.measure_model_size(),
        }

# Run
benchmark = MarkGPTBenchmark(model, tokenizer)
results = benchmark.run_all_benchmarks()

# Save
import json
with open('benchmark_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(json.dumps(results, indent=2))
```

---

**Evaluation Metrics v1.0**
**Last Updated**: 2024
