#!/usr/bin/env python3
"""Create remaining module-05 PRs using gh pr create"""
import subprocess
import sys

prs = [
    (2, "feature/module05-pr02", "Cross-Encoders & Ranking"),
    (3, "feature/module05-pr03", "Contrastive Learning for Search"),
    (4, "feature/module05-pr04", "Retrieval-Augmented QA"),
    (5, "feature/module05-pr05", "Multi-hop & Conversational QA"),
    (6, "feature/module05-pr06", "Extractive & Abstractive Summarization"),
    (7, "feature/module05-pr07", "Sequence-to-Sequence & NMT"),
    (8, "feature/module05-pr08", "Back-translation & MT Evaluation"),
    (9, "feature/module05-pr09", "Named Entity Recognition"),
    (10, "feature/module05-pr10", "Sentiment Analysis & Emotions"),
    (11, "feature/module05-pr11", "Text Classification & Multi-label"),
    (12, "feature/module05-pr12", "Relation & Event Extraction"),
    (13, "feature/module05-pr13", "Vision-Language Models"),
    (14, "feature/module05-pr14", "Graph Neural Networks for NLP"),
    (15, "feature/module05-pr15", "Conversational AI & Dialogue"),
    (16, "feature/module05-pr16", "Zero-shot & Few-shot Learning"),
    (17, "feature/module05-pr17", "Active Learning Strategies"),
    (18, "feature/module05-pr18", "Transfer & Domain Adaptation"),
    (19, "feature/module05-pr19", "Data Augmentation Techniques"),
    (20, "feature/module05-pr20", "Adversarial Examples & Training"),
    (21, "feature/module05-pr21", "Interpretability & Explanation"),
    (22, "feature/module05-pr22", "Biomedical NLP Applications"),
    (23, "feature/module05-pr23", "Legal NLP & Document Understanding"),
    (24, "feature/module05-pr24", "Financial NLP & Sentiment"),
    (25, "feature/module05-pr25", "Social Media & Noisy Text"),
    (26, "feature/module05-pr26", "Prompt Engineering Essentials"),
    (27, "feature/module05-pr27", "Retrieval-Augmented Generation"),
    (28, "feature/module05-pr28", "Code Understanding & Generation"),
    (29, "feature/module05-pr29", "Multilingual & Cross-lingual Transfer"),
    (30, "feature/module05-pr30", "Speech & Audio NLP"),
    (31, "feature/module05-pr31", "Temporal & Dynamic Language"),
    (32, "feature/module05-pr32", "Long Document Processing"),
    (33, "feature/module05-pr33", "Bias, Fairness & Ethics"),
    (34, "feature/module05-pr34", "Privacy & Security in NLP"),
    (35, "feature/module05-pr35", "Efficient & Sustainable NLP"),
    (36, "feature/module05-pr36", "Edge Cases & Common Pitfalls"),
    (37, "feature/module05-pr37", "Research Frontiers & Future"),
    (38, "feature/module05-pr38", "Lesson-specific Content"),
    (39, "feature/module05-pr39", "Capstone Project & Resources"),
    (40, "feature/module05-pr40", "Troubleshooting & Best Practices"),
    (41, "feature/module05-pr41", "Module Summary & Next Steps"),
    (42, "feature/module05-pr42", "Lesson 1 - Word Embeddings"),
    (43, "feature/module05-pr43", "Lesson 2 - ELMo & Contextualization"),
]

print(f"Creating {len(prs)} pull requests...")
created = 0

for num, branch, title in prs:
    cmd = [
        'gh', 'pr', 'create',
        '--base', 'master',
        '--head', branch,
        '--title', f'Module-05 PR {num}: {title}',
        '--body', 'Comprehensive enrichment of module-05 NLP curriculum',
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print(f'✓ PR {num}: {title}')
            created += 1
        else:
            print(f'- PR {num}: {title} (already exists or error)')
    except Exception as e:
        print(f'- PR {num}: {title} (error: {str(e)[:30]})')

print(f'\n✓ Created/verified {created} pull requests!')
