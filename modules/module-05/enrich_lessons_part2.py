#!/usr/bin/env python3
"""
Module-05 lessons part 2 - Extended enrichment (30+ commits per lesson)
Expand lesson content for comprehensive coverage
"""
import subprocess
import os

os.chdir(r'c:\Users\THE EYE INFORMATIQUE\OneDrive\Desktop\All\MarkGPT-LLM-Curriculum\MarkGPT-LLM-Curriculum\modules\module-05\lessons')

lessons = {
    'L05.1_embeddings/README.md': [
        ("## Static Embeddings Analysis\n\n### Pre-trained vs Trained\n\n"
         "Pre-trained: Leverage general knowledge\n"
         "Task-specific: Better on target\n"
         "Fine-tune: Improve pre-trained\n"
         "Frozen: Use as-is\n"
         "Trade-off: Data size vs pre-training\n\n",
         "Static embeddings analysis"),
        
        ("### Embedding Drift\n\n"
         "Words change meaning over time\n"
         "\"gay\" historically = happy\n"
         "Embeddings capture era language\n"
         "Temporal analysis possible\n"
         "Historical text vs modern\n\n",
         "Add embedding drift"),
        
        ("### Sense Embeddings\n\n"
         "Multiple senses per word\n"
         "\"bank\": financial/river\n"
         "Mixture model approach\n"
         "Per-sense embeddings\n"
         "Disambiguate from context\n\n",
         "Add sense embeddings"),
        
        ("### Retrofitting\n\n"
         "Post-hoc refinement\n"
         "Adjust embeddings to resources\n"
         "Preserve similarity structure\n"
         "Add external knowledge\n"
         "Better semantic alignment\n\n",
         "Add retrofitting"),
        
        ("### Morphological Composition\n\n"
         "\"unhappy\" = \"un\" + \"happy\"\n"
         "Compositional model\n"
         "Predicts morphologically rich\n"
         "Helps low-resource\n"
         "Shares structure\n\n",
         "Add morphological"),
        
        ("### Frequency Band Analysis\n\n"
         "Common words: Stable embeddings\n"
         "Rare words: Noisy embeddings\n"
         "Frequency affects quality\n"
         "Regularization helps rare\n"
         "Inverse frequency weighting\n\n",
         "Add frequency analysis"),
    ],
    
    'L25_tokenization_deep_dive.md': [
        ("### Trie-Based Tokenization\n\n"
         "Prefix tree structure\n"
         "O(n) tokenization\n"
         "Efficient matching\n"
         "Works with variable vocab\n"
         "Used in BERT\n\n",
         "Add trie-based tokenization"),
        
        ("### Entropy-Based Analysis\n\n"
         "Tokenization ambiguity\n"
         "Multiple valid segmentations\n"
         "Entropy measures uncertainty\n"
         "High entropy: Fewer good options\n"
         "Evaluates tokenizer quality\n\n",
         "Add entropy analysis"),
        
        ("### Dynamic Programming\n\n"
         "Optimal substructure\n"
         "Best(0..n) = min over splits\n"
         "Viterbi-like algorithm\n"
         "Guaranteed optimal\n"
         "O(n^2) time\n\n",
         "Add DP tokenization"),
        
        ("### Morphologically-Aware\n\n"
         "Segment respecting morphology\n"
         "\"running\" = \"run\" + \"ing\"\n"
         "Preserves linguistic structure\n"
         "Better for low-resource\n"
         "Multilingual advantage\n\n",
         "Add morphological aware"),
        
        ("### Domain-Specific Tokenization\n\n"
         "Medical: Keep domain terms\n"
         "Programming: Split operators\n"
         "Social media: Hashtags, @mentions\n"
         "Customize for domain\n"
         "Improves downstream\n\n",
         "Add domain-specific token"),
        
        ("### Character n-gram Fallback\n\n"
         "No exact match?\n"
         "Use character n-grams\n"
         "Can represent anything\n"
         "Slow but complete\n"
         "Hybrid systems\n\n",
         "Add char n-gram fallback"),
    ],
    
    'L26.1_word-embeddings/README.md': [
        ("## Embedding Quality Evaluation\n\n### Intrinsic vs Extrinsic\n\n"
         "Intrinsic: Standalone tests\n"
         "Word similarity datasets\n"
         "Fast to compute\n"
         "Extrinsic: Downstream tasks\n"
         "Real application performance\n\n",
         "Intrinsic vs extrinsic"),
        
        ("### Analogy Evaluation\n\n"
         "\"king\" - \"man\" + \"woman\" = \"queen\"\n"
         "Semantic + syntactic\n"
         "Google word test\n"
         "Often not reliable\n"
         "Debate on validity\n\n",
         "Add analogy evaluation"),
        
        ("### Nearest Neighbor Analysis\n\n"
         "Find k-nearest neighbors\n"
         "Qualitative inspection\n"
         "Should be semantically similar\n"
         "Reveals embedding quality\n"
         "Most interpretable\n\n",
         "Add nearest neighbor"),
        
        ("### Alignment to Human Judgments\n\n"
         "Human rate word pairs\n"
         "1-10 similarity scale\n"
         "RareWord-353, SimLex\n"
         "Spearman correlation\n"
         "Standard benchmarks\n\n",
         "Add human alignment"),
        
        ("### Speed vs Quality Tradeoff\n\n"
         "Large embedding: Better quality\n"
         "Fast to compute\n"
         "Large embedding: Slow inference\n"
         "Need:Space tradeoff\n"
         "Application dependent\n\n",
         "Add speed quality tradeoff"),
        
        ("### Embedding Scaling Laws\n\n"
         "Quality improves with corpus size\n"
         "Larger embedding dimension\n"
         "More training examples\n"
         "Predictable improvements\n"
         "Foundational for modern LLMs\n\n",
         "Add embedding scaling"),
    ],
    
    'L27.1_text-classification/README.md': [
        ("## Classification Loss Functions\n\n### Cross-Entropy\n\n"
         "-Σ y_i * log(p_i)\n"
         "Single label setting\n"
         "Standard for classification\n"
         "Differentiable\n"
         "Numerically stable variants\n\n",
         "Add cross-entropy loss"),
        
        ("### Focal Loss\n\n"
         "Down-weight easy examples\n"
         "PT = probability of true label\n"
         "Loss = -alpha * (1-PT)^gamma * log(PT)\n"
         "Focus on hard negatives\n"
         "Helps imbalanced data\n\n",
         "Add focal loss"),
        
        ("### Contrastive Loss\n\n"
         "Triplet: anchor, positive, negative\n"
         "Margin-based ranking\n"
         "min(sim(a,p) - sim(a,n) + margin)\n"
         "Forces semantic structure\n"
         "Better embeddings\n\n",
         "Add contrastive loss"),
        
        ("### Metrics for Classification\n\n"
         "Accuracy: %correct\n"
         "Precision: TP/(TP+FP)\n"
         "Recall: TP/(TP+FN)\n"
         "F1: Harmonic mean\n"
         "ROC-AUC: Ranking metric\n\n",
         "Add classification metrics"),
        
        ("### Threshold Optimization\n\n"
         "Default: 0.5\n"
         "Move for cost-sensitive\n"
         "Optimize F1 or custom metric\n"
         "Search threshold\n"
         "Data-dependent\n\n",
         "Add threshold optimization"),
        
        ("### Multi-class vs Multi-label\n\n"
         "Mutually exclusive vs overlapping\n"
         "Softmax vs sigmoid\n"
         "Different losses\n"
         "Different evaluation\n"
         "Both common\n\n",
         "Add multi-class multilabel"),
    ],
    
    'L28.1_ner-tagging/README.md': [
        ("## NER System Architecture\n\n### Feature Engineering\n\n"
         "Lexical: word, case, digits\n"
         "Syntactic: POS tags\n"
         "Semantic: word embeddings\n"
         "External: gazetteers, wikipeda\n"
         "Combined into feature vector\n\n",
         "NER feature engineering"),
        
        ("### Gazetteer Resources\n\n"
         "Lists of known entities\n"
         "Person names, locations, orgs\n"
         "Wikipedia dumps, YAGO, DBpedia\n"
         "Signals but incomplete\n"
         "Noisy annotations\n\n",
         "Add gazetteers"),
        
        ("### Boundary Detection\n\n"
         "Where does entity start/end?\n"
         "\"New York City\" = where to split\n"
         "Token boundaries help\n"
         "Sentence context\n"
         "Hard in noisy text\n\n",
         "Add boundary detection"),
        
        ("### Type Disambiguation\n\n"
         "\"New York\" = location or org?\n"
         "Multiple types possible\n"
         "Hierarchical: Organization > Company\n"
         "Fine-grained often better\n"
         "Task-specific granularity\n\n",
         "Add type disambiguation"),
        
        ("### Cross-lingual NER\n\n"
         "Transfer to new language\n"
         "Low-resource languages\n"
         "Multilingual embeddings\n"
         "Shared structure\n"
         "Growing area\n\n",
         "Add cross-lingual NER"),
        
        ("### Slot-filling vs NER\n\n"
         "NER: Identify entities\n"
         "Slot-filling: Find specific values\n"
         "\"CEO of X\": Extract X\n"
         "More structured\n"
         "Relation-aware\n\n",
         "Add slot-filling"),
    ],
    
    'L29.2_gpt-pretraining/README.md': [
        ("## Fine-tuning Strategies\n\n### Task-Specific Fine-tuning\n\n"
         "Adapt general model to task\n"
         "Add task-specific head\n"
         "Fine-tune all or last layers\n"
         "Much faster than pre-training\n"
         "Works with limited data\n\n",
         "Task-specific finetuning"),
        
        ("### Domain Adaptation\n\n"
         "Further pre-train on domain\n"
         "Medical text on medical papers\n"
         "Code on code repositories\n"
         "Cheap pre-training phase\n"
         "Significant quality improvement\n\n",
         "Add domain adaptation"),
        
        ("### Few-shot Learning\n\n"
         "Show examples in context\n"
         "No parameter updates\n"
         "In-context learning\n"
         "Larger models better\n"
         "GPT-3: Few-shot phenomena\n\n",
         "Add few-shot learning"),
        
        ("### Prompt Engineering\n\n"
         "Design input prompts\n"
         "\"Translate English to French:\\n\"\n"
         "Huge impact on output\n"
         "Task description format\n"
         "Active research\n\n",
         "Add prompt engineering"),
        
        ("### Temperature and Sampling\n\n"
         "Temperature: Higher = more random\n"
         "Top-k: Sample from top k\n"
         "Top-p: Sample from top p prob\n"
         "Nucleus sampling: Balanced\n"
         "Controls output diversity\n\n",
         "Add temperature sampling"),
        
        ("### Tokenizer Impact\n\n"
         "Different tokenizers = different results\n"
         "BPE variants: GPT uses BPE\n"
         "Token count affects length\n"
         "Encoding efficiency\n"
         "Multilingual challenges\n\n",
         "Add tokenizer impact"),
    ],
}

total = 0
for lesson_path, sections in lessons.items():
    dirpath = lesson_path.rsplit('/', 1)[0] if '/' in lesson_path else '.'
    if dirpath and not os.path.exists(dirpath):
        os.makedirs(dirpath, exist_ok=True)
    
    if not os.path.exists(lesson_path):
        with open(lesson_path, 'w') as f:
            f.write(f"# {lesson_path}\n\n")
    
    print(f"\nEnriching: {lesson_path}")
    
    for i, (content, msg) in enumerate(sections, 1):
        with open(lesson_path, 'a', encoding='utf-8') as f:
            f.write(content)
        
        try:
            subprocess.run(['git', 'add', lesson_path], check=True, capture_output=True)
            subprocess.run(['git', 'commit', '-m', f'Extend {lesson_path}: {msg}'], 
                         check=True, capture_output=True)
            print(f"  [{i}/6] {msg}")
            total += 1
        except:
            pass

print(f"\n[DONE] Added {total} extended lesson commits!")
result = subprocess.run(['git', 'log', '--oneline'], capture_output=True, text=True)
final = len(result.stdout.strip().split('\n'))
print(f"Total repository commits: {final}")
