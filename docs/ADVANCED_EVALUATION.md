# Model Evaluation Framework

## Comprehensive Evaluation Strategy

This guide extends EVAL_FRAMEWORK.md with advanced evaluation techniques.

## Automated Benchmarking

### Continuous Evaluation Pipeline

```python
import json
from datetime import datetime
from pathlib import Path

class BenchmarkSuite:
    """Automated evaluation on multiple datasets."""
    
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        self.results = {}
    
    def run_all(self, datasets_config):
        """Run evaluation on all configured datasets."""
        
        for dataset_name, dataset_config in datasets_config.items():
            print(f"\nEvaluating on {dataset_name}...")
            
            dataset = load_dataset(dataset_config['path'])
            test_set = dataset['test']
            
            metrics = self.evaluate_dataset(test_set, dataset_config)
            self.results[dataset_name] = metrics
            
            # Save intermediate results
            self.save_results()
        
        return self.results
    
    def evaluate_dataset(self, dataset, config):
        """Evaluate on single dataset."""
        
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'model_name': self.model.config.model_id,
            'dataset': config['path'],
            'results': {}
        }
        
        # Evaluation loop
        all_losses = []
        all_preds = []
        all_labels = []
        
        loader = DataLoader(dataset, batch_size=config.get('batch_size', 32))
        
        self.model.eval()
        with torch.no_grad():
            for batch in tqdm(loader):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(**batch)
                loss = outputs['loss']
                logits = outputs['logits']
                
                all_losses.append(loss.item())
                all_preds.extend(logits.argmax(-1).cpu().numpy())
                all_labels.extend(batch['labels'].cpu().numpy())
        
        # Compute metrics
        avg_loss = np.mean(all_losses)
        ppl = np.exp(avg_loss)
        accuracy = (np.array(all_preds) == np.array(all_labels)).mean()
        
        metrics['results'] = {
            'loss': float(avg_loss),
            'perplexity': float(ppl),
            'accuracy': float(accuracy),
            'num_samples': len(dataset)
        }
        
        return metrics
    
    def save_results(self):
        """Save results to JSON."""
        output_path = Path('evaluation_results') / f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        output_path.parent.mkdir(exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"Results saved to {output_path}")

# Usage
bench = BenchmarkSuite(model)
datasets = {
    'wikitext': {'path': 'wikitext', 'batch_size': 32},
    'banso': {'path': 'banso_corpus', 'batch_size': 32},
    'arxiv': {'path': 'arxiv_abstracts', 'batch_size': 32},
}
results = bench.run_all(datasets)
```

## Human Evaluation Protocol

### Quality Assessment Framework

```python
class HumanEvaluator:
    """Structured human evaluation protocol."""
    
    RUBRIC = {
        'coherence': {
            '1': 'Incoherent, incomprehensible',
            '2': 'Mostly incoherent with occasional clarity',
            '3': 'Generally coherent with some confusing parts',
            '4': 'Coherent and easy to follow',
            '5': 'Perfectly coherent, professional quality'
        },
        'relevance': {
            '1': 'Completely off-topic',
            '2': 'Marginally relevant',
            '3': 'Mostly relevant',
            '4': 'Highly relevant',
            '5': 'Perfectly on-topic and contextual'
        },
        'factuality': {
            '1': 'Multiple significant errors',
            '2': 'Several errors',
            '3': 'Minor errors or unverifiable',
            '4': 'Mostly accurate',
            '5': 'Factually accurate'
        },
        'language_quality': {
            '1': 'Poor grammar, many errors',
            '2': 'Numerous errors',
            '3': 'Acceptable with minor errors',
            '4': 'Good grammar and style',
            '5': 'Excellent grammar and natural language'
        }
    }
    
    def generate_eval_interface(self, predictions, num_samples=100):
        """Create HTML interface for human evaluation."""
        
        html = """
        <html>
        <head>
            <style>
                body { font-family: Arial; margin: 20px; }
                .sample { border: 1px solid #ddd; padding: 15px; margin: 15px 0; }
                .prompt { font-weight: bold; color: #333; }
                .generation { background-color: #f5f5f5; padding: 10px; margin: 10px 0; }
                .rubric { margin: 10px 0; }
                .rubric label { display: block; margin: 5px 0; }
                textarea { width: 100%; height: 100px; }
            </style>
        </head>
        <body>
            <h1>MarkGPT Generation Evaluation</h1>
            <p>Instructions: Rate each generation on the criteria below. Be fair and specific.</p>
        """
        
        samples = random.sample(predictions, min(num_samples, len(predictions)))
        
        for idx, pred in enumerate(samples):
            html += f"""
            <div class="sample">
                <h3>Sample {idx + 1}</h3>
                <div class="prompt"><strong>Prompt:</strong> {html.escape(pred['prompt'])}</div>
                <div class="generation"><strong>Generation:</strong> {html.escape(pred['text'])}</div>
                
                <form>
            """
            
            for criterion in self.RUBRIC.keys():
                html += f"""
                <div class="rubric">
                    <label><strong>{criterion.title()}:</strong></label>
                """
                
                for score, description in self.RUBRIC[criterion].items():
                    html += f"""
                    <label>
                        <input type="radio" name="{idx}_{criterion}" value="{score}" required>
                        {score} - {description}
                    </label>
                    """
                
                html += "</div>"
            
            html += f"""
                    <textarea placeholder="Additional comments..." name="{idx}_comments"></textarea>
                </form>
            </div>
            """
        
        html += "</body></html>"
        
        return html
    
    def compute_inter_rater_agreement(self, ratings_list):
        """Calculate Cohen's kappa for multiple raters."""
        
        import statsmodels.api as sm
        
        # Flatten: [rater1_scores, rater2_scores, ...]
        table = np.array(ratings_list).T
        
        kappas = []
        for i in range(len(ratings_list)):
            for j in range(i + 1, len(ratings_list)):
                kappa = sm.stats.inter_rater_agreement(
                    np.column_stack([ratings_list[i], ratings_list[j]])
                )
                kappas.append(kappa)
        
        avg_kappa = np.mean(kappas)
        print(f"Average Inter-Rater Agreement (Cohen's κ): {avg_kappa:.3f}")
        
        # Interpretation
        if avg_kappa < 0.20:
            print("  → Poor agreement")
        elif avg_kappa < 0.40:
            print("  → Fair agreement")
        elif avg_kappa < 0.60:
            print("  → Moderate agreement")
        elif avg_kappa < 0.80:
            print("  → Good agreement")
        else:
            print("  → Very good agreement")
        
        return avg_kappa

# Usage
evaluator = HumanEvaluator()
html_interface = evaluator.generate_eval_interface(predictions, num_samples=50)

with open('evaluation_interface.html', 'w') as f:
    f.write(html_interface)

print("Evaluation interface saved. Share with evaluators and collect responses.")
```

## Error Analysis

### Failure Mode Classification

```python
class ErrorAnalyzer:
    """Categorize and analyze model failures."""
    
    FAILURE_MODES = {
        'hallucination': 'Model generates facts not in input',
        'repetition': 'Unnecessary repetition of words/phrases',
        'truncation': 'Output cuts off abruptly',
        'incoherence': 'Output doesn\'t make logical sense',
        'irrelevance': 'Output doesn\'t match prompt',
        'style_shift': 'Unexpected change in writing style',
        'language_mixing': 'Unexpected language switching',
    }
    
    def analyze_failures(self, predictions):
        """Categorize specific failure modes."""
        
        failures = {mode: [] for mode in self.FAILURE_MODES.keys()}
        
        for pred in predictions:
            text = pred['text']
            prompt = pred['prompt']
            gold = pred.get('reference')
            
            # Check for repetition
            words = text.split()
            if len(words) != len(set(words)):
                dup_ratio = 1 - (len(set(words)) / len(words))
                if dup_ratio > 0.2:
                    failures['repetition'].append({'text': text, 'dup_ratio': dup_ratio})
            
            # Check for truncation
            if len(text) < 50 or text.endswith(('...', '—', '_')):
                failures['truncation'].append({'text': text, 'length': len(text)})
            
            # Check for language mixing (if multilingual)
            if self._detect_language_mixing(text):
                failures['language_mixing'].append({'text': text})
            
            # Check for hallucination (if reference available)
            if gold and not self._is_factual(text, gold):
                failures['hallucination'].append({'text': text, 'reference': gold})
        
        # Print report
        print("\n" + "="*60)
        print("ERROR ANALYSIS REPORT")
        print("="*60)
        
        for mode, examples in failures.items():
            if examples:
                print(f"\n{mode.upper()}: {len(examples)} instances")
                for ex in examples[:3]:  # Show first 3
                    print(f"  → {ex['text'][:100]}...")
        
        return failures
    
    def _detect_language_mixing(self, text):
        """Simple heuristic for language mixing."""
        # Would use langdetect or similar in practice
        return False
    
    def _is_factual(self, text, reference):
        """Check if text aligns with reference."""
        # Would use semantic similarity or NER matching
        return True

# Usage
analyzer = ErrorAnalyzer()
failures = analyzer.analyze_failures(predictions)
```

---

**Framework Version**: 2.0
**Last Updated**: 2024
