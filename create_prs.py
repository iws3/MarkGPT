#!/usr/bin/env python3
"""
Create 50 pull requests for module-05 enrichment
Uses gh pr create to create PRs from pre-made branches
"""
import subprocess
import sys

# PR definitions: (branch_name, pr_title, pr_body)
prs = [
    ("pr-01-dense-retrieval", "Module-05 PR 1: Dense Retrieval & Semantic Search Intro", "Dense passage retrieval, bi-encoders, and foundational semantic search concepts"),
    ("pr-02-cross-encoders", "Module-05 PR 2: Cross-Encoders & Ranking", "Cross-encoder models and their application as rerankers in retrieval pipelines"),
    ("pr-03-contrastive-search", "Module-05 PR 3: Contrastive Learning for Search", "Contrastive learning approaches and knowledge distillation for search systems"),
    ("pr-04-retrieval-qa", "Module-05 PR 4: Retrieval-Augmented Question Answering", "Retrieval-augmented QA, open-domain QA, and reading comprehension"),
    ("pr-05-conversational-qa", "Module-05 PR 5: Multi-hop & Conversational QA", "Multi-hop reasoning, conversational context, and multi-turn question answering"),
    ("pr-06-extraction-abstraction", "Module-05 PR 6: Extractive & Abstractive Summarization", "Extractive vs abstractive summarization with metrics and pre-trained models"),
    ("pr-07-seq2seq-nmt", "Module-05 PR 7: Sequence-to-Sequence & Machine Translation", "Seq2seq architecture for neural machine translation and multilingual models"),
    ("pr-08-translation-techniques", "Module-05 PR 8: Back-translation & MT Evaluation", "Data augmentation via back-translation and evaluation metrics for translation"),
    ("pr-09-ner-systems", "Module-05 PR 9: Named Entity Recognition", "NER task, models, CRF layers, datasets, and performance metrics"),
    ("pr-10-sentiment-analysis", "Module-05 PR 10: Sentiment Analysis & Emotions", "Binary, multi-class, and aspect-based sentiment analysis approaches"),
    ("pr-11-classification", "Module-05 PR 11: Text Classification & Multi-label", "Text classification, multi-label, and hierarchical classification tasks"),
    ("pr-12-information-extraction", "Module-05 PR 12: Relation & Event Extraction", "Relation extraction, event extraction, and information retrieval tasks"),
    ("pr-13-vision-language", "Module-05 PR 13: Vision-Language Models", "CLIP, vision transformers, image captioning, and visual question answering"),
    ("pr-14-graph-neural-networks", "Module-05 PR 14: Graph Neural Networks for NLP", "GCNs, knowledge graph embeddings, and text as graphs"),
    ("pr-15-dialogue-systems", "Module-05 PR 15: Conversational AI & Dialogue", "Response generation, personalization, and goal-oriented dialogue systems"),
    ("pr-16-zero-few-shot", "Module-05 PR 16: Zero-shot & Few-shot Learning", "Zero-shot learning, attributes, prototypical networks, and domain generalization"),
    ("pr-17-active-learning", "Module-05 PR 17: Active Learning Strategies", "Uncertainty sampling, query by committee, and coreset approaches"),
    ("pr-18-transfer-learning", "Module-05 PR 18: Transfer & Domain Adaptation", "Pre-training, domain adaptation, task similarity, and multi-task learning"),
    ("pr-19-data-augmentation", "Module-05 PR 19: Data Augmentation Techniques", "Back-translation, EDA, contextual augmentation, and self-training"),
    ("pr-20-adversarial-robustness", "Module-05 PR 20: Adversarial Examples & Training", "Adversarial attacks, FGSM, adversarial training, and certified robustness"),
    ("pr-21-explainability", "Module-05 PR 21: Interpretability & Explanation", "Attention visualization, attribution methods, LIME, SHAP, and probing"),
    ("pr-22-biomedical-nlp", "Module-05 PR 22: Biomedical NLP Applications", "Biomedical NER, relation extraction, QA, and domain-specific pre-training"),
    ("pr-23-legal-nlp", "Module-05 PR 23: Legal NLP & Document Understanding", "Case prediction, contract analysis, legal IR, and case law retrieval"),
    ("pr-24-financial-nlp", "Module-05 PR 24: Financial NLP & Sentiment", "Financial entity extraction, event detection, and price prediction"),
    ("pr-25-social-media-nlp", "Module-05 PR 25: Social Media & Noisy Text", "Text normalization, sarcasm, hate speech, and misinformation detection"),
    ("pr-26-prompt-engineering", "Module-05 PR 26: Prompt Engineering Essentials", "Prompt design, few-shot prompting, chain-of-thought, and advanced techniques"),
    ("pr-27-rag-systems", "Module-05 PR 27: Retrieval-Augmented Generation", "RAG architecture, training, implementation, and evaluation"),
    ("pr-28-code-understanding", "Module-05 PR 28: Code Understanding & Generation", "Code search, summarization, bug detection, and generation"),
    ("pr-29-multilingual-nlp", "Module-05 PR 29: Multilingual & Cross-lingual Transfer", "Multilingual models, zero-shot transfer, MT, and code-switching"),
    ("pr-30-speech-processing", "Module-05 PR 30: Speech & Audio NLP", "Speech recognition, synthesis, speaker identification, and speech understanding"),
    ("pr-31-temporal-nlp", "Module-05 PR 31: Temporal & Dynamic Language", "Temporal relations, time-aware embeddings, and streaming NLP"),
    ("pr-32-long-documents", "Module-05 PR 32: Long Document Processing", "Hierarchical attention, efficient transformers, and long-range dependencies"),
    ("pr-33-bias-fairness", "Module-05 PR 33: Bias, Fairness & Ethics", "Measuring and mitigating bias, gender in language, and transparency"),
    ("pr-34-privacy-security", "Module-05 PR 34: Privacy & Security in NLP", "Membership inference, federated learning, adversarial robustness, watermarking"),
    ("pr-35-efficiency", "Module-05 PR 35: Efficient & Sustainable NLP", "Model compression, distillation, quantization, pruning, and green AI"),
    ("pr-36-edge-cases", "Module-05 PR 36: Edge Cases & Common Pitfalls", "Domain shift, class imbalance, ambiguity, and OOD detection"),
    ("pr-37-research-directions", "Module-05 PR 37: Research Frontiers & Future", "Multimodality, in-context learning, neurosymbolic AI, and continual learning"),
    ("pr-38-lesson-details", "Module-05 PR 38: Lesson-specific Content", "Detailed lesson objectives, implementations, and project checkpoints"),
    ("pr-39-capstone-project", "Module-05 PR 39: Capstone Project & Resources", "Project requirements, suggested projects, and learning resources"),
    ("pr-40-troubleshooting", "Module-05 PR 40: Troubleshooting & Best Practices", "Common issues, debugging strategies, and hyperparameter tuning"),
    ("pr-41-final-reflections", "Module-05 PR 41: Module Summary & Next Steps", "Key takeaways, applications, and completion reflection"),
    ("pr-42-lesson1-detailed", "Module-05 PR 42: Lesson 1 - Word Embeddings Deep Dive", "Word2Vec, GloVe, FastText, and practical exercises"),
    ("pr-43-lesson2-contextualized", "Module-05 PR 43: Lesson 2 - ELMo & Contextualization", "ELMo architecture, fine-tuning, and benchmarking"),
    ("pr-44-lesson3-bert", "Module-05 PR 44: Lesson 3 - BERT & Transformers", "BERT pre-training, fine-tuning, variants, and analysis"),
    ("pr-45-lesson4-gpt", "Module-05 PR 45: Lesson 4 - GPT & Generation", "GPT architecture, generation strategies, scaling laws"),
    ("pr-46-lesson5-alignment", "Module-05 PR 46: Lesson 5 - Instruction Tuning & Alignment", "Instruction tuning, RLHF, safety, and practical training"),
]

print(f"Creating {len(prs)} pull requests...")
print("=" * 70)

for i, (branch, title, body) in enumerate(prs, 1):
    # Create PR
    cmd = [
        'gh', 'pr', 'create',
        '--base', 'main',
        '--head', branch,
        '--title', title,
        '--body', body
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        # Extract PR number from output
        output = result.stdout.strip()
        if 'pull' in output:
            print(f"[OK] {i:2d}: {title[:50]}")
        else:
            print(f"[OK] {i:2d}: {title[:50]}")
    except subprocess.CalledProcessError as e:
        print(f"[SKIP] {i:2d}: {title[:50]} - {e.stderr[:50]}")
    except Exception as e:
        print(f"[FAIL] {i:2d}: {title[:50]} - {str(e)[:50]}")

print("=" * 70)
print(f"PR creation complete!")
