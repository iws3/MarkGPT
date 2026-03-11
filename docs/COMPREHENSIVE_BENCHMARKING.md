# Comprehensive Benchmarking Guide

## Benchmark Categories

### 1. Perplexity (Language Modeling)

Measure how "surprised" the model is by test data.

```python
def compute_perplexity_dataset(model, tokenizer, dataset_path):
    """Compute perplexity across entire dataset."""
    
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        with open(dataset_path) as f:
            for line in f:
                tokens = tokenizer.encode(line, return_tensors='pt').to('cuda')
                
                if tokens.shape[1] > 2050:  # Skip very long sequences
                    continue
                
                outputs = model(tokens, labels=tokens)
                
                loss = outputs.loss.item()
                total_loss += loss * tokens.shape[1]
                total_tokens += tokens.shape[1]
    
    perplexity = np.exp(total_loss / total_tokens)
    return perplexity

# Compute on standard benchmarks
wikitext_ppl = compute_perplexity_dataset(model, tokenizer, 'wikitext-test.txt')
print(f"WikiText-103 Perplexity: {wikitext_ppl:.2f}")

# Expected (by model size):
# Nano (70M): 50-60
# Small (200M): 25-35
# Base (500M): 15-25
```

### 2. BLEU Score (Translation Quality)

```python
from nltk.translate.bleu_score import corpus_bleu

def eval_translation_bleu(model, tokenizer, source_lang, target_lang, test_pairs):
    """Evaluate translation quality."""
    
    hypotheses = []
    references = []
    
    for source_text, reference_text in test_pairs:
        # Generate translation
        input_ids = tokenizer.encode(
            f"[{source_lang}→{target_lang}]: {source_text}",
            return_tensors='pt'
        ).to('cuda')
        
        with torch.no_grad():
            output = model.generate(input_ids, max_new_tokens=100)
        
        hypothesis = tokenizer.decode(output[0], skip_special_tokens=True)
        hypotheses.append(hypothesis.split())
        references.append([reference_text.split()])
    
    # Compute corpus BLEU
    bleu = corpus_bleu(references, hypotheses)
    
    return bleu

# English → Banso Translation
en_banso_pairs = [
    ("Hello, how are you?", "Ayaba, u bè?"),
    ("I am fine.", "Mwə a bè kpele."),
    # ... more pairs
]

bleu_score = eval_translation_bleu(model, tokenizer, 'EN', 'BANSO', en_banso_pairs)
print(f"EN→BANSO BLEU: {bleu_score:.2f}")  # Expected: 20-35
```

---

## Task-Specific Benchmarks

### 3. Question Answering (SQuAD-style)

```python
def eval_qa_accuracy(model, tokenizer, qa_dataset):
    """Evaluate QA using exact match and F1."""
    
    exact_matches = 0
    f1_scores = []
    
    for example in qa_dataset:
        context = example['context']
        question = example['question']
        gold_answer = example['answer']
        
        # Create prompt
        prompt = f"""Context: {context}
Question: {question}
Answer:"""
        
        input_ids = tokenizer.encode(prompt, return_tensors='pt').to('cuda')
        
        with torch.no_grad():
            output = model.generate(
                input_ids,
                max_new_tokens=50,
                temperature=0.1
            )
        
        predicted_answer = tokenizer.decode(output[0], skip_special_tokens=True)
        
        # Exact match
        if predicted_answer.strip().lower() == gold_answer.strip().lower():
            exact_matches += 1
        
        # F1 score (token overlap)
        pred_tokens = set(predicted_answer.split())
        gold_tokens = set(gold_answer.split())
        
        common = pred_tokens & gold_tokens
        if len(pred_tokens) + len(gold_tokens) > 0:
            f1 = 2 * len(common) / (len(pred_tokens) + len(gold_tokens))
        else:
            f1 = 0
        
        f1_scores.append(f1)
    
    exact_match_pct = exact_matches / len(qa_dataset) * 100
    mean_f1 = np.mean(f1_scores) * 100
    
    return {
        'exact_match': exact_match_pct,
        'f1': mean_f1
    }

# Evaluate
qa_results = eval_qa_accuracy(model, tokenizer, squad_dev_set)
print(f"QA Exact Match: {qa_results['exact_match']:.1f}%")
print(f"QA F1: {qa_results['f1']:.1f}%")
```

### 4. Sentiment Classification

```python
def eval_sentiment_accuracy(model, tokenizer, sentiment_dataset):
    """Evaluate sentiment classification accuracy."""
    
    correct = 0
    
    for example in sentiment_dataset:
        text = example['text']
        gold_label = example['label']  # 0=negative, 1=positive
        
        prompt = f"""Classify sentiment: Negative or Positive

Text: "{text}"
Sentiment:"""
        
        input_ids = tokenizer.encode(prompt, return_tensors='pt').to('cuda')
        
        with torch.no_grad():
            output = model.generate(
                input_ids,
                max_new_tokens=1,
                temperature=0.1
            )
        
        prediction = tokenizer.decode(output[0], skip_special_tokens=True).lower()
        
        # Check if prediction matches
        if gold_label == 1 and 'positive' in prediction:
            correct += 1
        elif gold_label == 0 and 'negative' in prediction:
            correct += 1
    
    accuracy = correct / len(sentiment_dataset) * 100
    return accuracy

# Evaluate
sentiment_acc = eval_sentiment_accuracy(model, tokenizer, sst2_dataset)
print(f"Sentiment Accuracy: {sentiment_acc:.1f}%")  # Expected: 70-85%
```

---

## Multilingual Benchmarks (MarkGPT Specific)

### 5. Language Purity

Measure code-switching behavior:

```python
def eval_language_purity(model, tokenizer):
    """Ensure monolingual generation stays monolingual."""
    
    from langdetect import detect_langs
    
    en_prompts = [
        "The future of artificial intelligence",
        "Machine learning is"
    ]
    
    banso_prompts = [
        "Ulimi lwesibanso",
        "Ayaba, u bè"
    ]
    
    # English purity
    en_purity_scores = []
    for prompt in en_prompts:
        output = model.generate(prompt, max_tokens=30)
        
        # Detect language
        try:
            langs = detect_langs(output)
            en_ratio = next((l.prob for l in langs if l.lang == 'en'), 0)
        except:
            en_ratio = 0
        
        en_purity_scores.append(en_ratio)
    
    # Banso purity  
    banso_purity_scores = []
    for prompt in banso_prompts:
        output = model.generate(prompt, max_tokens=30)
        
        # Count Banso tokens
        tokens = tokenizer.encode(output)
        banso_token_count = sum(1 for t in tokens if 10000 <= t < 15000)  # Banso vocab range
        banso_ratio = banso_token_count / len(tokens)
        
        banso_purity_scores.append(banso_ratio)
    
    return {
        'english_purity': np.mean(en_purity_scores),
        'banso_purity': np.mean(banso_purity_scores),
        'overall_purity': (np.mean(en_purity_scores) + np.mean(banso_purity_scores)) / 2
    }

purity = eval_language_purity(model, tokenizer)
print(f"Language Purity: {purity['overall_purity']*100:.1f}%")  # Target: >85%
```

---

## Efficiency Benchmarks

### 6. Inference Speed

```python
def benchmark_inference_speed(model, tokenizer, num_iterations=100):
    """Measure tokens/second throughput."""
    
    prompt = "The future of AI is" * 5  # Longer prompt
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to('cuda')
    
    # Warmup
    with torch.no_grad():
        _ = model.generate(input_ids, max_new_tokens=10)
    
    torch.cuda.synchronize()
    start_time = time.time()
    
    total_tokens = 0
    with torch.no_grad():
        for _ in range(num_iterations):
            output = model.generate(input_ids, max_new_tokens=50)
            total_tokens += 50
    
    torch.cuda.synchronize()
    elapsed = time.time() - start_time
    
    throughput = total_tokens / elapsed
    
    return throughput  # tokens/second

# Benchmark each model
nano_speed = benchmark_inference_speed(nano_model, tokenizer)
small_speed = benchmark_inference_speed(small_model, tokenizer)
base_speed = benchmark_inference_speed(base_model, tokenizer)

print(f"Nano: {nano_speed:.0f} tok/s")    # Expected: 200-400
print(f"Small: {small_speed:.0f} tok/s")   # Expected: 100-200
print(f"Base: {base_speed:.0f} tok/s")     # Expected: 50-100
```

### 7. Memory Usage

```python
def benchmark_memory_usage(model, tokenizer):
    """Measure GPU/CPU memory."""
    
    import psutil
    import torch.cuda
    
    # CPU memory
    process = psutil.Process()
    cpu_mem_before = process.memory_info().rss / 1024**3  # GB
    
    # GPU memory
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    
    # Generate
    input_ids = tokenizer.encode("Warm up", return_tensors='pt').to('cuda')
    
    with torch.no_grad():
        _ = model.generate(input_ids, max_new_tokens=100)
    
    torch.cuda.synchronize()
    
    # Measurements
    cpu_mem_after = process.memory_info().rss / 1024**3
    gpu_mem = torch.cuda.max_memory_allocated() / 1024**3
    
    return {
        'cpu_memory_gb': cpu_mem_after - cpu_mem_before,
        'gpu_memory_gb': gpu_mem,
        'model_size_gb': sum(p.numel() * 4 / 1024**3 for p in model.parameters())
    }

memory = benchmark_memory_usage(model, tokenizer)
print(f"GPU Memory: {memory['gpu_memory_gb']:.1f} GB")
print(f"Model Size: {memory['model_size_gb']:.1f} GB")
```

---

## Automated Benchmark Suite

```python
class MarkGPTBenchmarkSuite:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.results = {}
    
    def run_all(self):
        """Execute complete benchmark suite."""
        
        print("🔄 Running MarkGPT Benchmark Suite...")
        
        # Language modeling
        print("  1/6 Perplexity...")
        self.results['perplexity'] = self.benchmark_perplexity()
        
        # Task performance
        print("  2/6 QA Accuracy...")
        self.results['qa'] = self.benchmark_qa()
        
        print("  3/6 Sentiment...")
        self.results['sentiment'] = self.benchmark_sentiment()
        
        # Multilingual
        print("  4/6 Language Purity...")
        self.results['language_purity'] = self.benchmark_language_purity()
        
        # Efficiency
        print("  5/6 Inference Speed...")
        self.results['speed'] = self.benchmark_speed()
        
        print("  6/6 Memory Usage...")
        self.results['memory'] = self.benchmark_memory()
        
        return self.results
    
    def save_results(self, output_path='benchmark_results.json'):
        """Save results."""
        
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"✅ Results saved to {output_path}")

# Run
suite = MarkGPTBenchmarkSuite(model, tokenizer)
results = suite.run_all()

# Print summary
print("\n📊 Benchmark Summary:")
print(json.dumps(results, indent=2))
```

---

**Benchmarking Guide v1.0**
**Last Updated**: 2024
