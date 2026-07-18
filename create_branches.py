#!/usr/bin/env python3
"""
Create 50 branches for module-05 enrichment PRs
"""
import subprocess
import sys

pr_configs = [
    # (pr_number, branch_name, commits_start, commits_end, pr_title)
    (1, "pr-01-dense-retrieval", 1, 5, "Dense Retrieval & Semantic Search"),
    (2, "pr-02-cross-encoders", 6, 10, "Cross-Encoders & Ranking"),
    (3, "pr-03-contrastive-search", 11, 15, "Contrastive Learning for Search"),
    (4, "pr-04-retrieval-qa", 16, 20, "Retrieval-Augmented QA"),
    (5, "pr-05-conversational-qa", 21, 25, "Multi-hop & Conversational QA"),
    (6, "pr-06-extraction-abstraction", 26, 30, "Extractive & Abstractive Summarization"),
    (7, "pr-07-seq2seq-nmt", 31, 35, "Sequence-to-Sequence & Machine Translation"),
    (8, "pr-08-translation-techniques", 36, 40, "Back-translation & MT Evaluation"),
    (9, "pr-09-ner-systems", 41, 45, "Named Entity Recognition"),
    (10, "pr-10-sentiment-analysis", 46, 50, "Sentiment Analysis & Emotions"),
    (11, "pr-11-classification", 51, 55, "Text Classification & Multi-label"),
    (12, "pr-12-information-extraction", 56, 60, "Relation & Event Extraction"),
    (13, "pr-13-vision-language", 61, 65, "Vision-Language Models"),
    (14, "pr-14-graph-neural-networks", 66, 70, "Graph Neural Networks for NLP"),
    (15, "pr-15-dialogue-systems", 71, 75, "Conversational AI & Dialogue"),
    (16, "pr-16-zero-few-shot", 76, 80, "Zero-shot & Few-shot Learning"),
    (17, "pr-17-active-learning", 81, 85, "Active Learning Strategies"),
    (18, "pr-18-transfer-learning", 86, 90, "Transfer & Domain Adaptation"),
    (19, "pr-19-data-augmentation", 91, 95, "Data Augmentation Techniques"),
    (20, "pr-20-adversarial-robustness", 96, 100, "Adversarial Examples & Training"),
    (21, "pr-21-explainability", 101, 105, "Interpretability & Explanation"),
    (22, "pr-22-biomedical-nlp", 106, 110, "Biomedical NLP Applications"),
    (23, "pr-23-legal-nlp", 111, 115, "Legal NLP & Document Understanding"),
    (24, "pr-24-financial-nlp", 116, 120, "Financial NLP & Sentiment"),
    (25, "pr-25-social-media-nlp", 121, 125, "Social Media & Noisy Text"),
    (26, "pr-26-prompt-engineering", 126, 130, "Prompt Engineering Essentials"),
    (27, "pr-27-rag-systems", 131, 135, "Retrieval-Augmented Generation"),
    (28, "pr-28-code-understanding", 136, 140, "Code Understanding & Generation"),
    (29, "pr-29-multilingual-nlp", 141, 145, "Multilingual & Cross-lingual Transfer"),
    (30, "pr-30-speech-processing", 146, 150, "Speech & Audio NLP"),
    (31, "pr-31-temporal-nlp", 151, 155, "Temporal & Dynamic Language"),
    (32, "pr-32-long-documents", 156, 160, "Long Document Processing"),
    (33, "pr-33-bias-fairness", 161, 165, "Bias, Fairness & Ethics"),
    (34, "pr-34-privacy-security", 166, 170, "Privacy & Security in NLP"),
    (35, "pr-35-efficiency", 171, 175, "Efficient & Sustainable NLP"),
    (36, "pr-36-edge-cases", 176, 180, "Edge Cases & Common Pitfalls"),
    (37, "pr-37-research-directions", 181, 185, "Research Frontiers & Future"),
    (38, "pr-38-lesson-details", 186, 190, "Lesson-specific Content"),
    (39, "pr-39-capstone-project", 191, 195, "Capstone Project & Resources"),
    (40, "pr-40-troubleshooting", 196, 200, "Troubleshooting & Best Practices"),
    (41, "pr-41-final-reflections", 201, 205, "Module Summary & Next Steps"),
    (42, "pr-42-lesson1-detailed", 206, 210, "Lesson 1 - Word Embeddings"),
    (43, "pr-43-lesson2-contextualized", 211, 215, "Lesson 2 - ELMo & Contextualization"),
    (44, "pr-44-lesson3-bert", 216, 220, "Lesson 3 - BERT & Transformers"),
    (45, "pr-45-lesson4-gpt", 221, 225, "Lesson 4 - GPT & Generation"),
    (46, "pr-46-lesson5-alignment", 226, 230, "Lesson 5 - Instruction Tuning"),
    (47, "pr-47-additional-resources", 231, 235, "Additional Learning Resources"),
    (48, "pr-48-advanced-topics", 236, 240, "Advanced Topics & Specialized"),
    (49, "pr-49-final-summary", 241, 245, "Module Final Summary"),
    (50, "pr-50-completion", 246, 250, "Module Completion & Reflection"),
]

print(f"Creating {len(pr_configs)} PR branches...")
print("=" * 70)

for pr_num, branch, start, end, title in pr_configs:
    # Create branch from main
    try:
        # Check out main
        subprocess.run(['git', 'checkout', 'main'], capture_output=True, check=True)
        # Create new branch
        subprocess.run(['git', 'checkout', '-b', branch], capture_output=True, check=True)
        # Cherry-pick commits from module-05
        for commit_num in range(start, end + 1):
            subprocess.run(['git', 'cherry-pick', f'module-05~{251-commit_num}'], 
                         capture_output=True, check=True)
        # Push branch
        subprocess.run(['git', 'push', '-u', 'origin', branch], capture_output=True, check=True)
        print(f"[OK] {pr_num:2d}: {branch} ({start}-{end})")
    except subprocess.CalledProcessError as e:
        print(f"[FAIL] {pr_num:2d}: {branch} - {str(e)[:40]}")
    except Exception as e:
        print(f"[ERR] {pr_num:2d}: {branch} - {str(e)[:40]}")

print("=" * 70)
print("Branches created! Now creating PRs...")
