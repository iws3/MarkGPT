#!/usr/bin/env python3
"""
Create 43 GitHub pull requests using GitHub CLI
"""
import subprocess
import os
import sys

os.chdir(r'c:\Users\THE EYE INFORMATIQUE\OneDrive\Desktop\All\MarkGPT-LLM-Curriculum\MarkGPT-LLM-Curriculum\modules\module-05')

prs = [
    ("Dense Retrieval & Semantic Search", "Implement dense passage retrieval using FAISS and contrastive learning"),
    ("Bi-encoders & Cross-encoders", "Add dual-encoder and cross-encoder architectures for semantic matching"),
    ("Question Answering Systems", "Build retrieval-augmented and extractive QA pipelines"),
    ("Abstractive Summarization", "Implement abstractive summarization with BART and T5 models"),
    ("Machine Translation", "Create seq2seq and transformer-based translation systems"),
    ("Named Entity Recognition", "Add NER with BERT and CRF layers"),
    ("Sentiment Analysis", "Build sentiment classifiers and aspect-based analysis"),
    ("Text Classification", "Implement multi-class and hierarchical text classification"),
    ("Information Extraction", "Add relation, event extraction and coreference resolution"),
    ("Vision-Language Models", "Implement CLIP and vision transformer models"),
    ("Graph Neural Networks", "Add GCN and graph-based NLP methods"),
    ("Dialogue Systems", "Build conversational AI with persona and goal-oriented variants"),
    ("Zero-shot & Few-shot Learning", "Implement zero-shot classification and few-shot learning"),
    ("Active Learning", "Add uncertainty sampling and query-by-committee"),
    ("Transfer Learning", "Implement domain adaptation and multi-task learning"),
    ("Data Augmentation", "Add back-translation, EDA, and contextual augmentation"),
    ("Adversarial Robustness", "Implement adversarial training and certified robustness"),
    ("Explainability & Interpretability", "Add LIME, SHAP, and attention visualization"),
    ("Biomedical NLP", "Specialized NLP for medical texts and entity extraction"),
    ("Legal NLP", "Case outcome prediction and contract analysis"),
    ("Financial NLP", "Entity extraction and price prediction from financial texts"),
    ("Social Media NLP", "Text normalization, sarcasm detection, hate speech detection"),
    ("Prompt Engineering", "Advanced prompt design and in-context learning techniques"),
    ("Retrieval-Augmented Generation", "Implement RAG with vector databases"),
    ("Code Understanding", "Add code search, summarization and generation"),
    ("Multilingual NLP", "Cross-lingual transfer and code-switching models"),
    ("Speech Processing", "Speech recognition, synthesis and speaker identification"),
    ("Temporal NLP", "Temporal relation extraction and time-aware embeddings"),
    ("Long Document Processing", "Hierarchical attention for long documents"),
    ("Bias & Fairness", "Bias measurement and mitigation strategies"),
    ("Privacy & Security", "Privacy-preserving NLP and adversarial security"),
    ("Efficiency & Sustainability", "Model compression, distillation and quantization"),
    ("Edge Cases & Pitfalls", "Domain shift, class imbalance and out-of-distribution detection"),
    ("Research Directions", "Multimodality, continual learning and neurosymbolic AI"),
    ("Lesson-01 Foundations", "Word embeddings and foundational NLP concepts"),
    ("Lesson-02 ELMo Models", "Contextualized embeddings and transfer learning"),
    ("Lesson-03 BERT Transformers", "BERT architecture and fine-tuning techniques"),
    ("Lesson-04 GPT Models", "Autoregressive generation and scaling laws"),
    ("Lesson-05 Alignment", "Instruction tuning and RLHF training"),
    ("Capstone Projects", "End-to-end NLP projects and evaluation criteria"),
    ("Learning Resources", "Curated papers, courses and community resources"),
    ("Troubleshooting Guide", "Common issues and debugging strategies"),
    ("Final Reflections", "Module completion and next steps"),
]

print(f"Creating {len(prs)} GitHub pull requests...")
print("=" * 70)

# Check if gh CLI is available
try:
    subprocess.run(['gh', '--version'], capture_output=True, check=True)
except:
    print("ERROR: GitHub CLI (gh) not found. Install from https://cli.github.com/")
    print("\nAlternatively, create PRs manually at:")
    print("https://github.com/YourUsername/MarkGPT-LLM-Curriculum/pulls")
    sys.exit(1)

created_count = 0
failed_count = 0

for i, (pr_name, pr_body) in enumerate(prs, 1):
    branch = f"feature/module05-pr{i:02d}"
    
    try:
        # Create PR using GitHub CLI
        result = subprocess.run(
            [
                'gh', 'pr', 'create',
                '--base', 'main',
                '--head', branch,
                '--title', f'Module-05 PR{i:02d}: {pr_name}',
                '--body', pr_body
            ],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode == 0:
            # Extract PR URL from output
            pr_url = result.stdout.strip().split('\n')[-1] if result.stdout else f"Branch: {branch}"
            print(f"[{i:2d}/43] ✓ {pr_name}")
            print(f"         {pr_url}")
            created_count += 1
        else:
            print(f"[{i:2d}/43] ✗ {pr_name}")
            print(f"         Error: {result.stderr[:100]}")
            failed_count += 1
            
    except subprocess.TimeoutExpired:
        print(f"[{i:2d}/43] ✗ {pr_name} (timeout)")
        failed_count += 1
    except Exception as e:
        print(f"[{i:2d}/43] ✗ {pr_name} ({str(e)[:50]})")
        failed_count += 1

print("\n" + "=" * 70)
print(f"Results: {created_count} created, {failed_count} failed")
print("=" * 70)

if created_count > 0:
    print(f"\n✓ {created_count} pull requests created successfully!")
    print("\nView all PRs at: https://github.com/YourUsername/MarkGPT-LLM-Curriculum/pulls")
