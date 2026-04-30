#!/usr/bin/env python3
"""
Module-05 phase 4 - 70 commits
Specialized NLP domains, prompt engineering, and advanced techniques
"""
import subprocess
import os
import sys

os.chdir(r'c:\Users\THE EYE INFORMATIQUE\OneDrive\Desktop\All\MarkGPT-LLM-Curriculum\MarkGPT-LLM-Curriculum\modules\module-05')

sections = [
    # PR 24: Biomedical NLP (5 commits)
    ("## Biomedical NLP\n\n"
     "### Challenges\n\n"
     "Specialized vocabulary\n"
     "Domain-specific entities\n"
     "Complex syntax\n"
     "Limited labeled data\n"
     "High stakes\n\n",
     "Add biomed intro"),
    
    ("### Named Entity Recognition\n\n"
     "Disease, drug, gene entities\n"
     "Datasets: NCBI, BC5CDR\n"
     "Domain-specific embeddings\n"
     "BERT variants: SciBERT\n"
     "Much better than generic\n\n",
     "Add biomed NER"),
    
    ("### Relation Extraction\n\n"
     "Drug-disease interactions\n"
     "Protein-protein interactions\n"
     "Knowledge base construction\n"
     "Semi-supervised approaches\n"
     "Limited labeled data\n\n",
     "Add biomed relations"),
    
    ("### Question Answering\n\n"
     "Biomedical literature QA\n"
     "BioASQ challenge\n"
     "Multi-document reasoning\n"
     "Retrieved documents\n"
     "Evidence extraction\n\n",
     "Add biomed QA"),
    
    ("### Pre-training Domain\n\n"
     "SciBERT: Scientific text\n"
     "BioBERT: Biomedical focus\n"
     "Continue pre-training\n"
     "Better than generic\n"
     "Transfer to tasks\n\n",
     "Add biomed pretrain"),
    
    # PR 25: Legal NLP (5 commits)
    ("## Legal NLP\n\n"
     "### Document Understanding\n\n"
     "Long documents: 10K+ tokens\n"
     "Hierarchical structure\n"
     "Sections and subsections\n"
     "Cross-references\n"
     "Complex reasoning\n\n",
     "Add legal intro"),
    
    ("### Case Outcome Prediction\n\n"
     "Predict court decision\n"
     "Multiple factors\n"
     "Data: CASELAW database\n"
     "Feature extraction\n"
     "LSTM and transformers\n\n",
     "Add case outcome"),
    
    ("### Contract Analysis\n\n"
     "Clause extraction\n"
     "Risk identification\n"
     "Payment terms\n"
     "Liability clauses\n"
     "Automated review\n\n",
     "Add contracts"),
    
    ("### Legal Information Retrieval\n\n"
     "Find relevant case law\n"
     "Precedent search\n"
     "Dense retrieval helps\n"
     "Legal-specific models\n"
     "LegalBERT\n\n",
     "Add legal IR"),
    
    ("### Dataset and Challenges\n\n"
     "SCOTUS: Supreme Court\n"
     "CASELAW: Large corpus\n"
     "Data license restrictions\n"
     "Domain shift\n"
     "Emerging field\n\n",
     "Add legal datasets"),
    
    # PR 26: Financial NLP (5 commits)
    ("## Financial NLP\n\n"
     "### Sentiment Analysis\n\n"
     "Stock price prediction\n"
     "News sentiment\n"
     "Social media monitoring\n"
     "Earnings calls\n"
     "Financial-specific lexicon\n\n",
     "Add financial intro"),
    
    ("### Entity Extraction\n\n"
     "Company names\n"
     "Stock symbols\n"
     "Financial instruments\n"
     "Named entity recognition\n"
     "Domain-specific\n\n",
     "Add financial entities"),
    
    ("### Event Extraction\n\n"
     "Mergers and acquisitions\n"
     "Earnings announcements\n"
     "Regulatory filings\n"
     "Market events\n"
     "Impact prediction\n\n",
     "Add financial events"),
    
    ("### Price Movement Prediction\n\n"
     "Text features for regression\n"
     "Sentiment + technical indicators\n"
     "Ensemble approaches\n"
     "Challenge: Market efficiency\n"
     "Limited predictability\n\n",
     "Add price prediction"),
    
    ("### Domain Resources\n\n"
     "FinBERT: Financial text\n"
     "Financial corpora\n"
     "Specialized embeddings\n"
     "Lexicon-based approaches\n"
     "Hybrid methods\n\n",
     "Add financial resources"),
    
    # PR 27: Social Media NLP (5 commits)
    ("## Social Media and Noisy Text\n\n"
     "### Challenges\n\n"
     "Misspellings\n"
     "Abbreviations\n"
     "Slang and informal\n"
     "Code-switching\n"
     "Short length\n\n",
     "Add social intro"),
    
    ("### Text Normalization\n\n"
     "Correct spellings\n"
     "Expand abbreviations\n"
     "Neural approaches\n"
     "Rule-based methods\n"
     "Hybrid systems\n\n",
     "Add normalization"),
    
    ("### Sarcasm Detection\n\n"
     "Binary classification\n"
     "Context dependent\n"
     "Hard for humans\n"
     "Datasets: SemEval\n"
     "Contextual embeddings help\n\n",
     "Add sarcasm"),
    
    ("### Hate Speech Detection\n\n"
     "Toxic content flagging\n"
     "Content moderation\n"
     "False positives costly\n"
     "Interpretability crucial\n"
     "Explainable AI needed\n\n",
     "Add hate speech"),
    
    ("### Rumor and Misinformation\n\n"
     "Identify false claims\n"
     "Verification\n"
     "Evidence retrieval\n"
     "Fact-checking\n"
     "Emerging challenge\n\n",
     "Add misinformation"),
    
    # PR 28: Prompt Engineering (5 commits)
    ("## Prompt Engineering\n\n"
     "### Introduction\n\n"
     "Instructions to language model\n"
     "Quality impacts output\n"
     "Art and science\n"
     "Counter-intuitive patterns\n"
     "Rapidly evolving field\n\n",
     "Add prompt intro"),
    
    ("### Prompt Design Patterns\n\n"
     "Task description\n"
     "Few-shot examples\n"
     "Output format specification\n"
     "Role-playing\n"
     "System and user prompts\n\n",
     "Add prompt patterns"),
    
    ("### Few-shot Prompting\n\n"
     "In-context learning\n"
     "Example demonstrations\n"
     "Format consistency\n"
     "Example selection matters\n"
     "Small data regime\n\n",
     "Add few-shot prompt"),
    
    ("### Chain of Thought\n\n"
     "Intermediate reasoning steps\n"
     "Improve performance\n"
     "Explainability bonus\n"
     "Self-consistency\n"
     "Ensemble of prompts\n\n",
     "Add CoT"),
    
    ("### Advanced Techniques\n\n"
     "Temperature and top-k\n"
     "Decoding strategies\n"
     "System prompt tuning\n"
     "Instruction engineering\n"
     "Best practices evolving\n\n",
     "Add prompt advanced"),
    
    # PR 29: Retrieval-Augmented Generation (5 commits)
    ("## Retrieval-Augmented Generation\n\n"
     "### Motivation\n\n"
     "Language models hallucinate\n"
     "Lack current information\n"
     "RAG solves this\n"
     "Retrieve + generate\n"
     "Better factuality\n\n",
     "Add RAG intro"),
    
    ("### Architecture\n\n"
     "Retriever: Find documents\n"
     "Reader: Extract answer\n"
     "Or generator: Condition on docs\n"
     "Two-stage pipeline\n"
     "End-to-end training\n\n",
     "Add RAG arch"),
    
    ("### Training\n\n"
     "Maximize relevance of retrieved docs\n"
     "Answer quality improves\n"
     "Hard negative mining\n"
     "Dense passage retrieval\n"
     "Joint optimization\n\n",
     "Add RAG training"),
    
    ("### Practical Implementation\n\n"
     "Vector database\n"
     "FAISS or Pinecone\n"
     "Latency matters\n"
     "Batch retrieval\n"
     "Production challenges\n\n",
     "Add RAG impl"),
    
    ("### Evaluation\n\n"
     "Retrieval accuracy\n"
     "Answer correctness\n"
     "End-to-end metrics\n"
     "Human evaluation\n"
     "Trade-offs: Quality vs speed\n\n",
     "Add RAG eval"),
    
    # PR 30: Code Understanding (5 commits)
    ("## Code Understanding and Generation\n\n"
     "### Models\n\n"
     "CodeBERT: Code and natural language\n"
     "GraphCodeBERT: Graph structure\n"
     "CodeT5: Code-aware T5\n"
     "GPT for code\n"
     "Specialized architectures\n\n",
     "Add code intro"),
    
    ("### Code Search\n\n"
     "Find relevant code snippets\n"
     "Natural language query\n"
     "Dual encoders\n"
     "Github Copilot\n"
     "Developer productivity\n\n",
     "Add code search"),
    
    ("### Code Summarization\n\n"
     "Generate docstrings\n"
     "Seq2seq approaches\n"
     "Abstract syntax tree\n"
     "Comment generation\n"
     "Aids maintenance\n\n",
     "Add code summ"),
    
    ("### Bug Detection\n\n"
     "Identify vulnerabilities\n"
     "Security critical\n"
     "Type inference\n"
     "Data flow analysis\n"
     "Static analysis tools\n\n",
     "Add bug detection"),
    
    ("### Code Generation\n\n"
     "Generate from specification\n"
     "Copilot popularity\n"
     "Evaluation: Functional correctness\n"
     "Hallucinations\n"
     "Security and privacy concerns\n\n",
     "Add code gen"),
]

readme_path = 'README.md'

print(f"Starting module-05 phase 4 with {len(sections)} commits...")
print("=" * 70)

for i, (content, msg) in enumerate(sections, 1):
    with open(readme_path, 'a', encoding='utf-8') as f:
        f.write(content)
    
    try:
        subprocess.run(['git', 'add', 'README.md'], check=True, capture_output=True)
        subprocess.run(['git', 'commit', '-m', msg], check=True, capture_output=True)
        print(f"[OK] {i:3d}: {msg}")
        sys.stdout.flush()
    except Exception as e:
        print(f"[FAIL] {i:3d}: {msg}")

print("=" * 70)
print(f"[DONE] Phase 4 added {len(sections)} commits!")

result = subprocess.run(['git', 'log', '--oneline'], capture_output=True, text=True)
total = len(result.stdout.strip().split('\n'))
print(f"Total repository commits: {total}")
