#!/usr/bin/env python3
"""
Module-05 enrichment part 4 - 125 commits
Advanced techniques, production systems, and deep practice
"""
import subprocess
import os

os.chdir(r'c:\Users\THE EYE INFORMATIQUE\OneDrive\Desktop\All\MarkGPT-LLM-Curriculum\MarkGPT-LLM-Curriculum\modules\module-05')

sections = [
    ("## Advanced Text Representations\n\n"
     "### Contextual vs Non-contextual\n\n"
     "Non-contextual: \"Hello\" always same vec\n"
     "Contextual: \"Hello\" different per context\n"
     "ELMo: First contextual (2018)\n"
     "BERT: Bidirectional contextual\n"
     "GPT: Autoregressive contextual\n"
     "Modern requirement\n\n",
     "Add contextual comparison"),
    
    ("### Layer-wise Analysis\n\n"
     "Lower layers: Syntax (POS, chunking)\n"
     "Middle layers: Low-level semantics\n"
     "Upper layers: High-level (task-specific)\n"
     "Transfer learning exploits this\n"
     "Fine-tune upper layers for task\n"
     "Freeze lower for general knowledge\n\n",
     "Add layer analysis"),
    
    ("## Dimensionality Reduction\n\n"
     "### PCA (Principal Component Analysis)\n\n"
     "Linear projection to k dimensions\n"
     "Preserves variance maximum\n"
     "Fast: SVD computation\n"
     "Interpretable: Components are words\n"
     "Works okay for embeddings\n"
     "Ok visualization\n\n",
     "Add PCA"),
    
    ("### t-SNE Deep Dive\n\n"
     "Non-linear dimension reduction\n"
     "Preserves local structure\n"
     "Clustered visualization\n"
     "Slow for large datasets\n"
     "Parameter sensitive\n"
     "Best for exploration\n\n",
     "Add t-SNE deep"),
    
    ("### UMAP Advantages\n\n"
     "Preserves both local and global\n"
     "Much faster than t-SNE\n"
     "Scales to millions\n"
     "Theoretically principled\n"
     "Better for production\n"
     "Modern choice\n\n",
     "Add UMAP adv"),
    
    ("## Linguistic Structure\n\n"
     "### Syntax vs Semantics\n\n"
     "Syntax: Grammar, word order\n"
     "Semantics: Meaning\n"
     "Embeddings capture both\n"
     "Lower layers: More syntactic\n"
     "Attention patterns reveal structure\n"
     "Can probe for structure\n\n",
     "Add syntax semantics"),
    
    ("### Probing Tasks\n\n"
     "Train classifier on hidden states\n"
     "Predict POS tags from embeddings\n"
     "If high accuracy: Encodes POS\n"
     "Multiple properties simultaneously\n"
     "Trade-off: Information vs capacity\n"
     "Reveals learned representations\n\n",
     "Add probing"),
    
    ("## Morphology\n\n"
     "### Morphological Analysis\n\n"
     "Words have structure\n"
     "Stems and affixes\n"
     "\"Running\" = \"run\" + \"-ing\"\n"
     "Embeddings somewhat capture\n"
     "More explicit: Morphological models\n"
     "Helps low-resource languages\n\n",
     "Add morphology"),
    
    ("### Morphologically Rich Languages\n\n"
     "Turkish: Agglutinative (many suffixes)\n"
     "Finnish: Very complex\n"
     "Czech: Many cases\n"
     "Arabic: Root patterns\n"
     "Challenges: Many word forms\n"
     "fastText helps via subword\n\n",
     "Add morphological languages"),
    
    ("## Domain Adaptation\n\n"
     "### The Problem\n\n"
     "Train on Wikipedia (general)\n"
     "Test on biomedical (specific)\n"
     "Distribution shift\n"
     "Generic embeddings miss domain\n"
     "Solution: Adapt embeddings\n"
     "Or train model on domain\n\n",
     "Add domain adapt"),
    
    ("### Approaches\n\n"
     "1. Retrain on domain corpus\n"
     "2. Fine-tune general embeddings\n"
     "3. Domain-specific vocabulary\n"
     "4. Combine with domain knowledge\n"
     "5. Active learning: Query hard examples\n"
     "Typically: Fine-tune + domain vocab\n\n",
     "Add domain adapt approaches"),
    
    ("## Data Augmentation\n\n"
     "### Backtranslation\n\n"
     "Translate to another language\n"
     "Translate back to original\n"
     "Creates paraphrases\n"
     "Double data size\n"
     "Improves robustness\n"
     "Common for small datasets\n\n",
     "Add backtranslation"),
    
    ("### Synonym Replacement\n\n"
     "Replace words with synonyms\n"
     "EDA: Easy Data Augmentation\n"
     "Random and controlled\n"
     "Preserves label but varies text\n"
     "Simple baseline\n"
     "Works surprisingly well\n\n",
     "Add synonym replacement"),
    
    ("### Mixup and Cutoff\n\n"
     "Mixup: Linear combination of embeddings\n"
     "Cutoff: Drop random tokens/subwords\n"
     "Regularization via augmentation\n"
     "Helps with small datasets\n"
     "Modern practice\n\n",
     "Add mixup cutoff"),
    
    ("## Low-Resource NLP\n\n"
     "### Transfer Learning\n\n"
     "Pre-train on general (billions tokens)\n"
     "Fine-tune on task (thousands examples)\n"
     "Much better than training from scratch\n"
     "50-story/0-shot with good model\n"
     "Modern necessity\n"
     "Enables low-resource work\n\n",
     "Add transfer learning"),
    
    ("### Few-Shot Learning\n\n"
     "Learn from very few examples\n"
     "5-10 labeled examples\n"
     "Meta-learning approaches\n"
     "MAML: Model-agnostic meta-learning\n"
     "Prototypical networks\n"
     "Enables rapid adaptation\n\n",
     "Add few-shot"),
    
    ("### Zero-Shot Learning\n\n"
     "No labeled examples\n"
     "Use class descriptions\n"
     "Match to class semantic vectors\n"
     "Requires good embeddings\n"
     "Prompt-based approach\n"
     "Emerging with large models\n\n",
     "Add zero-shot"),
    
    ("## Handling Imbalanced Data\n\n"
     "### Class Imbalance\n\n"
     "90% negative, 10% positive\n"
     "Accuracy misleading\n"
     "Always predict majority\n"
     "Accuracy: 90% but useless!\n"
     "Need: Balanced metrics\n\n",
     "Add imbalance"),
    
    ("### Solutions\n\n"
     "1. Oversampling: Duplicate minority\n"
     "2. Undersampling: Drop majority\n"
     "3. Cost-sensitive: Penalize errors\n"
     "4. SMOTE: Synthetic minority examples\n"
     "5. Threshold tuning: Adjust decision\n"
     "6. Use better metrics: F1, RoC-AUC\n\n",
     "Add imbalance solutions"),
    
    ("## Error Analysis\n\n"
     "### Systematic Analysis\n\n"
     "1. Look at wrong predictions\n"
     "2. Categorize error types\n"
     "3. Count by category\n"
     "4. Focus on top errors\n"
     "5. Address root causes\n"
     "Better than tuning:\n"
     "Understanding > parameters\n\n",
     "Add error analysis"),
    
    ("### Confusion Matrix\n\n"
     "TP: True Positive\n"
     "FN: False Negative (missed)\n"
     "FP: False Positive (false alarm)\n"
     "TN: True Negative\n"
     "Reveals what model struggles with\n"
     "Guides data collection\n\n",
     "Add confusion matrix"),
    
    ("## Active Learning\n\n"
     "### Selective Labeling\n\n"
     "Don't label everything\n"
     "Label most informative examples\n"
     "Model-uncertain examples\n"
     "Reduces annotation cost\n"
     "20% labels → 90% of full accuracy\n"
     "Smart data collection\n\n",
     "Add active learning"),
    
    ("### Query Strategies\n\n"
     "Uncertainty: Model least confident\n"
     "Diversity: Examples unlike labeled\n"
     "Expected gradient length: Big gradient\n"
     "Committee: Ensemble disagreement\n"
     "Typical: Start with diverse, refine uncertain\n\n",
     "Add query strategies"),
    
    ("## Interpretability\n\n"
     "### Feature Attribution\n\n"
     "Which words matter for prediction?\n"
     "Gradient-based: ∂L/∂x\n"
     "Attention: Which words attended?\n"
     "LIME: Local explanations\n"
     "SHAP: Game theory-based\n"
     "Understanding models is crucial\n\n",
     "Add feature attribution"),
    
    ("### Adversarial Robustness\n\n"
     "Small input changes flip prediction\n"
     "Spelling errors: \"resteraunt\"\n"
     "Paraphrases: Change wording\n"
     "Typos: Should be robust\n"
     "Adversarial training helps\n"
     "Data augmentation helps\n\n",
     "Add adversarial"),
    
    ("## Production Deployment\n\n"
     "### Model Serving\n\n"
     "Model trained - now what?\n"
     "API endpoint for predictions\n"
     "Batch processing for bulk\n"
     "Real-time requirements\n"
     "Infrastructure: Docker, Kubernetes\n"
     "Monitoring: Accuracy, latency\n\n",
     "Add model serving"),
    
    ("### Edge Cases\n\n"
     "Empty input: Handle gracefully\n"
     "Very long text: Truncate intelligently\n"
     "OOV words: Use subword fallback\n"
     "Multiple languages: Route to language-specific\n"
     "Malformed: Preprocess carefully\n"
     "Production requirement\n\n",
     "Add edge cases"),
    
    ("### Monitoring and Metrics\n\n"
     "Accuracy: Baseline metric\n"
     "Latency: Response time\n"
     "Throughput: Requests/sec\n"
     "Cost: Computational efficiency\n"
     "Drift: Does performance degrade?\n"
     "User feedback: Ground truth check\n\n",
     "Add monitoring"),
    
    ("### Updating Models\n\n"
     "Data changes over time\n"
     "Retraining schedule\n"
     "A/B test: New vs old model\n"
     "Gradual rollout: 1% → 10% → 100%\n"
     "Rollback: If issues detected\n"
     "Continuous improvement\n\n",
     "Add model updates"),
    
    ("## Dataset Construction\n\n"
     "### Annotation Guidelines\n\n"
     "Clear definitions\n"
     "Examples with explanations\n"
     "Disagreement resolution\n"
     "Quality control\n"
     "Inter-annotator agreement\n"
     "Cohen's kappa metric\n\n",
     "Add annotation"),
    
    ("### Crowdsourcing\n\n"
     "Hire many annotators\n"
     "Low cost per example\n"
     "Redundancy for quality\n"
     "Platforms: Mechanical Turk, Upwork\n"
     "Worker qualification\n"
     "Often enables large-scale\n\n",
     "Add crowdsourcing"),
    
    ("### Dataset Biases\n\n"
     "Selection bias: Who collected?\n"
     "Labeler bias: Subjective decisions\n"
     "Distribution: Representative sample?\n"
     "Can affect model fairness\n"
     "Audit datasets routinely\n"
     "Better: Diversify sources\n\n",
     "Add dataset bias"),
    
    ("## Benchmarking\n\n"
     "### Important Benchmarks\n\n"
     "GLUE: 9 language understanding tasks\n"
     "SQuAD: Reading comprehension\n"
     "MNIST: Classic (simple now)\n"
     "ImageNet: Huge vision benchmark\n"
     "Common measures: Compare fairly\n"
     "Leaderboards: Competitive drive\n\n",
     "Add benchmarks"),
    
    ("### Benchmark Gaming\n\n"
     "Overfitting to benchmark\n"
     "Not generalizing\n"
     "Contamination: Test in training?\n"
     "Solution: New datasets, real applications\n"
     "Better metric: Few-shot on new task\n"
     "Robust evaluation needed\n\n",
     "Add benchmark gaming"),
    
    ("## Computational Efficiency\n\n"
     "### Model Size\n\n"
     "BERT-base: 110M parameters\n"
     "BERT-large: 340M parameters\n"
     "Inference: Larger = slower\n"
     "Mobile: Need lightweight\n"
     "DistilBERT: 40% smaller, 97% accuracy\n"
     "Pruning: Remove unimportant weights\n\n",
     "Add model size"),
    
    ("### Quantization Revisited\n\n"
     "FP32 → FP16: 2x faster\n"
     "FP32 → INT8: 4x faster, 1% loss\n"
     "TPU: INT8 only\n"
     "Practical in production\n"
     "Libraries: TensorFlow Lite, ONNX\n"
     "Worth doing\n\n",
     "Add quantization practical"),
    
    ("### Inference Optimization\n\n"
     "Batch requests: 10x speedup\n"
     "Cache: Precompute embeddings\n"
     "Model selection: Right size for task\n"
     "Hardware: GPU if available\n"
     "Serving: TensorFlow Serving, triton\n"
     "Startup cost matters sometimes\n\n",
     "Add inference opt"),
    
    ("## Responsible AI\n\n"
     "### Fairness\n\n"
     "Demographic parity: Same accuracy\n"
     "Equalized odds: Similar errors\n"
     "Calibration: Confidence = accuracy\n"
     "Bias detection: Audit models\n"
     "Mitigation: Better data, constraints\n"
     "Ongoing work\n\n",
     "Add fairness"),
    
    ("### Privacy\n\n"
     "Memorization: Models can memorize\n"
     "Membership inference: Can extract training data?\n"
     "Differential privacy: Formal guarantee\n"
     "Federated learning: Train on device\n"
     "Data retention: Keep only needed\n"
     "Regulation: GDPR, local laws\n\n",
     "Add privacy"),
    
    ("### Transparency\n\n"
     "Model cards: Document model\n"
     "Data sheets: Document data\n"
     "Limitations: Be honest\n"
     "Failure cases: When it breaks\n"
     "Bias statement: Known issues\n"
     "Good practice\n\n",
     "Add transparency"),
    
    ("## Research Frontiers\n\n"
     "### Open Problems\n\n"
     "1. Longer context: Transformers limited\n"
     "2. Reasoning: Models don't reason well\n"
     "3. Grounding: Connect to reality\n"
     "4. Efficiency: Smaller models\n"
     "5. Multimodal: Text + images + audio\n"
     "6. Interpretability: Why decisions?\n"
     "7. Fairness: Autonomous bias\n\n",
     "Add research frontiers"),
    
    ("### Future Directions\n\n"
     "Scaling laws: Bigger = better\n"
     "Retrieval augmentation: Access knowledge\n"
     "Sparse models: Only activate parts\n"
     "Mixture of experts: Specialized models\n"
     "Prompt engineering: Better queries\n"
     "Few-shot learning: Rapid adaptation\n\n",
     "Add future directions"),
    
    ("## Capstone: End-to-End Project\n\n"
     "**Build a Text Classification System**\n\n"
     "1. Find dataset (download or create)\n"
     "2. Explore data distribution\n"
     "3. Baseline: TF-IDF + Logistic Regression\n"
     "4. Embeddings: Word2Vec or fastText\n"
     "5. Deep model: CNN/RNN\n"
     "6. Error analysis and debugging\n"
     "7. Optimize and deploy\n"
     "8. Monitor performance\n"
     "9. Write thorough report\n\n",
     "Add capstone project"),
    
    ("## Suggested Capstones\n\n"
     "1. Sentiment analysis (movie reviews)\n"
     "2. Spam detection (emails)\n"
     "3. Intent classification (chatbot)\n"
     "4. Toxicity detection (comments)\n"
     "5. Topic classification (news)\n"
     "6. Language detection (multilingual)\n"
     "7. Domain-specific classifier (your domain)\n\n",
     "Add capstone suggestions"),
    
    ("## Key Takeaways\n\n"
     "**Understanding**\n"
     "1. Text is high-dimensional data\n"
     "2. Embeddings capture semantics\n"
     "3. Context matters profoundly\n"
     "4. Transfer learning enables low-resource\n"
     "5. Models have biases and limitations\n"
     "6. Interpretability is underrated\n"
     "7. Production != research\n\n"
     "**Technical Skills**\n"
     "1. Text preprocessing\n"
     "2. Embedding training and usage\n"
     "3. Classification and sequence labeling\n"
     "4. Error analysis\n"
     "5. Model deployment\n"
     "6. Performance monitoring\n"
     "7. Responsible AI practices\n\n",
     "Add key takeaways"),
    
    ("## Module 05 Final Thoughts\n\n"
     "This module covers NLP foundations.\n"
     "From raw text to semantic understanding.\n"
     "Tokenization through embeddings.\n"
     "Classical methods still useful.\n"
     "Foundation for transformer era.\n\n"
     "**Timeline**\n"
     "2013-2017: Embeddings reign\n"
     "2018: ELMo changes everything\n"
     "2018+: Transformer era\n"
     "2023+: Large models dominate\n\n"
     "You've learned the fundamentals.\n"
     "Ready for modern NLP (module-06).\n"
     "Ready for production systems.\n"
     "Ready to contribute to research.\n\n"
     "You know text as data. Excellent! 🎉\n",
     "Add final thoughts"),
]

readme_path = 'README.md'

print(f"Starting module-05 part 4 with {len(sections)} commits...")
print("=" * 60)

for i, (content, msg) in enumerate(sections, 1):
    with open(readme_path, 'a', encoding='utf-8') as f:
        f.write(content)
    
    try:
        subprocess.run(['git', 'add', 'README.md'], check=True, capture_output=True)
        subprocess.run(['git', 'commit', '-m', f'Enrich module-05 part4 {i}: {msg}'], check=True, capture_output=True)
        print(f"[OK] Commit {i:2d}/{len(sections)}: {msg}")
    except subprocess.CalledProcessError:
        print(f"[FAIL] Part 4 {i:2d}: {msg}")

print("=" * 60)
print(f"[DONE] Part 4 added {len(sections)} commits!")

result = subprocess.run(['git', 'log', '--oneline'], capture_output=True, text=True)
total = len(result.stdout.strip().split('\n'))
print(f"Total repository commits: {total}")
