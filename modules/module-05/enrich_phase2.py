#!/usr/bin/env python3
"""
Module-05 enrichment phase 2 - 70 commits
Semantic search, question answering, and advanced NLP
"""
import subprocess
import os
import sys

os.chdir(r'c:\Users\THE EYE INFORMATIQUE\OneDrive\Desktop\All\MarkGPT-LLM-Curriculum\MarkGPT-LLM-Curriculum\modules\module-05')

sections = [
    # PR 7: Semantic Search (5 commits)
    ("## Semantic Search and Dense Retrieval\n\n"
     "### Dense Passage Retrieval\n\n"
     "Encode queries and documents\n"
     "Retrieve by similarity in embedding space\n"
     "FAISS: Fast similarity search library\n"
     "100M documents: Milliseconds latency\n"
     "Much better than BM25 keyword matching\n\n",
     "Add dense retrieval"),
    
    ("### Bi-Encoders\n\n"
     "Query encoder and document encoder\n"
     "Independent networks (can batch separately)\n"
     "Fast inference at scale\n"
     "Similarity: Dot product or cosine\n"
     "Usually sufficient quality\n\n",
     "Add bi-encoders"),
    
    ("### Cross-Encoders\n\n"
     "Jointly encode query and document\n"
     "Higher quality than bi-encoder\n"
     "Slower: Must evaluate all pairs\n"
     "Use as re-ranker after bi-encoder\n"
     "Workflow: Bi-encoder then cross-encoder\n\n",
     "Add cross-encoders"),
    
    ("### Contrastive Learning\n\n"
     "Positive: Query with relevant doc\n"
     "Negative: Query with irrelevant doc\n"
     "Loss: Maximize positive, minimize negative\n"
     "SimCLR: Self-supervised version\n"
     "Data efficient approach\n\n",
     "Add contrastive learning"),
    
    ("### Knowledge Distillation for Search\n\n"
     "Large cross-encoder teaches small bi-encoder\n"
     "Student learns from teacher\n"
     "Fast inference with good quality\n"
     "Production approach\n"
     "Typical: 5x speedup, 90 percent quality\n\n",
     "Add distill search"),
    
    # PR 8: Question Answering (5 commits)
    ("## Question Answering: End-to-end Systems\n\n"
     "### Retrieval-Augmented QA\n\n"
     "1. Retrieve relevant passages\n"
     "2. Extract answer from passages\n"
     "3. Rank candidate answers\n"
     "Splits into modular components\n"
     "Each can be optimized separately\n"
     "Very effective approach\n\n",
     "Add retrieval QA"),
    
    ("### Open-domain QA\n\n"
     "Answer using entire Wikipedia\n"
     "Retrieve: BM25 or dense retrieval\n"
     "Extract: BERT span extraction\n"
     "Challenges: Scale and accuracy\n"
     "Modern: Dense plus BERT equals SOTA\n\n",
     "Add open-domain QA"),
    
    ("### Machine Reading Comprehension\n\n"
     "Given passage and question\n"
     "Extract answer span\n"
     "Datasets: SQuAD, MS MARCO\n"
     "BERT: 92.5 percent F1 (vs 91.5 human)\n"
     "Problem essentially solved\n\n",
     "Add reading comprehension"),
    
    ("### Multi-hop QA\n\n"
     "Question requires multiple reasoning steps\n"
     "Example: Who is parent of someone's child?\n"
     "Harder: Requires chained reasoning\n"
     "Datasets: HotpotQA\n"
     "Current performance: 65-70 percent F1\n"
     "Still open problem\n\n",
     "Add multi-hop"),
    
    ("### Conversational QA\n\n"
     "Keep context from previous turns\n"
     "Coreference resolution needed\n"
     "CoQA, QuAC datasets\n"
     "Harder than single-turn QA\n"
     "Context modeling is crucial\n\n",
     "Add conversational QA"),
    
    # PR 9: Summarization (5 commits)
    ("## Abstractive and Extractive Summarization\n\n"
     "### Extractive Summarization\n\n"
     "Select important sentences\n"
     "Combine into summary\n"
     "Preserves original wording\n"
     "Simple: Score each sentence\n"
     "Fast and stable\n\n",
     "Add extractive"),
    
    ("### Abstractive Summarization\n\n"
     "Generate new summary text\n"
     "Paraphrase and compress\n"
     "More flexible than extractive\n"
     "Seq2seq models: Encode-decode\n"
     "Transformers: Much better than RNNs\n\n",
     "Add abstractive"),
    
    ("### Evaluation Metrics\n\n"
     "ROUGE: Recall-oriented understudy\n"
     "BLEU: Machine translation metric\n"
     "METEOR: Alignment-based metric\n"
     "Human evaluation: Gold standard\n"
     "Automatic metrics imperfect\n\n",
     "Add summarization metrics"),
    
    ("### Pre-trained Models\n\n"
     "BART: Denoising autoencoder\n"
     "T5: Text-to-text transfer transformer\n"
     "PEGASUS: Pre-trained for summarization\n"
     "PGN: Pointer generator networks\n"
     "Copy mechanism important\n\n",
     "Add summarization models"),
    
    ("### Fine-tuning for Summarization\n\n"
     "Datasets: CNN/DailyMail, Gigaword\n"
     "Learning rate: Small like classification\n"
     "Beam search: K=4 typical\n"
     "Length penalties: Prevent too short\n"
     "Results: SOTA on standard benchmarks\n\n",
     "Add summarization finetuning"),
    
    # PR 10: Machine Translation (5 commits)
    ("## Neural Machine Translation\n\n"
     "### Sequence-to-Sequence Architecture\n\n"
     "Encoder: Read source sentence\n"
     "Decoder: Generate target sentence\n"
     "Attention: Focus on relevant parts\n"
     "Better than phrase-based SMT\n"
     "Transformers: Major breakthrough\n\n",
     "Add seq2seq NMT"),
    
    ("### Multilingual Translation\n\n"
     "Single model for many language pairs\n"
     "Language token: Mark source and target\n"
     "Shared vocabulary across languages\n"
     "Parameter sharing reduces model size\n"
     "Transfer between languages\n\n",
     "Add multilingual MT"),
    
    ("### Back-translation\n\n"
     "Synthetic data augmentation\n"
     "Translate target to source\n"
     "Use as additional training\n"
     "Doubles training data\n"
     "Improves performance significantly\n\n",
     "Add back-translation"),
    
    ("### Evaluation of Translation\n\n"
     "BLEU: Automatic metric\n"
     "Limitations: Not perfect\n"
     "Human evaluation: Best\n"
     "TER: Translation edit rate\n"
     "METEOR: Better alignment\n\n",
     "Add MT evaluation"),
    
    ("### Deployment\n\n"
     "Inference: Left-to-right generation\n"
     "Beam search: K=5 typical\n"
     "Length penalty: Adjust output length\n"
     "Latency: Critical for production\n"
     "Batching increases throughput\n\n",
     "Add MT deployment"),
    
    # PR 11: Named Entity Recognition (5 commits)
    ("## Named Entity Recognition\n\n"
     "### Task Definition\n\n"
     "Tag each token with entity type\n"
     "Categories: PERSON, LOCATION, ORG, O\n"
     "Sequence labeling task\n"
     "Dataset: CoNLL, ACE\n"
     "Challenging for nested entities\n\n",
     "Add NER task"),
    
    ("### Models\n\n"
     "BiLSTM-CRF: Previous SOTA\n"
     "BERT: Better with fine-tuning\n"
     "RoBERTa: Even better\n"
     "Character embeddings help\n"
     "Contextualization crucial\n\n",
     "Add NER models"),
    
    ("### CRF Layer\n\n"
     "Conditional random field\n"
     "Sequence-level modeling\n"
     "Enforce valid tag sequences\n"
     "Example: No O after B\n"
     "Improves accuracy\n\n",
     "Add CRF layer"),
    
    ("### Datasets\n\n"
     "CoNLL 2003: English, German\n"
     "ACE: Diverse text sources\n"
     "OntoNotes: Rich annotation\n"
     "WNUT: Noisy social media\n"
     "Cross-domain challenge\n\n",
     "Add NER datasets"),
    
    ("### Performance\n\n"
     "SOTA: 92-93 percent F1\n"
     "BERT: Much improvement\n"
     "Nested NER: Still harder\n"
     "Zero-shot: Transfer learning\n"
     "Language-specific variants\n\n",
     "Add NER performance"),
    
    # PR 12: Sentiment Analysis (5 commits)
    ("## Sentiment Analysis\n\n"
     "### Binary vs Multi-class\n\n"
     "Binary: Positive or negative\n"
     "Multi-class: 5-point scale\n"
     "Fine-grained: Aspect-based\n"
     "Multi-label: Multiple sentiments\n"
     "Task selection matters\n\n",
     "Add sentiment task"),
    
    ("### Challenges\n\n"
     "Sarcasm and irony\n"
     "Domain transfer: Product to movie\n"
     "Implicit sentiment\n"
     "Neutral/mixed reviews\n"
     "Context dependency\n\n",
     "Add sentiment challenges"),
    
    ("### Models\n\n"
     "BERT fine-tuning: Simple, effective\n"
     "Aspect-based: Target-specific\n"
     "Transfer learning: Domain adaptation\n"
     "Ensemble: Multiple models\n"
     "Results: 90+ percent accuracy\n\n",
     "Add sentiment models"),
    
    ("### Datasets\n\n"
     "Stanford Sentiment: Movie reviews\n"
     "SemEval: Shared tasks\n"
     "SST: Parse tree annotations\n"
     "Product reviews: Various domains\n"
     "Tweet sentiment: Noisy social media\n\n",
     "Add sentiment datasets"),
    
    ("### Aspect-based Sentiment\n\n"
     "Sentiment toward specific aspect\n"
     "Example: Restaurant decor negative\n"
     "But service positive\n"
     "Joint extraction and classification\n"
     "Fine-grained analysis\n\n",
     "Add aspect-based"),
    
    # PR 13: Text Classification (5 commits)
    ("## Text Classification\n\n"
     "### Categories\n\n"
     "Topic classification\n"
     "Spam detection\n"
     "Toxic comment detection\n"
     "Emotion classification\n"
     "Intent prediction\n\n",
     "Add classification types"),
    
    ("### Approaches\n\n"
     "Bag of words: Simple baseline\n"
     "TF-IDF: Weight important words\n"
     "RNN: Preserve order\n"
     "CNN: Local patterns\n"
     "Transformers: State of art\n\n",
     "Add classification approaches"),
    
    ("### BERT for Classification\n\n"
     "Add classification head\n"
     "CLS token: Document representation\n"
     "Fine-tune 2-5 epochs\n"
     "Learning rate: 2e-5\n"
     "Simple and effective\n\n",
     "Add BERT classification"),
    
    ("### Multi-label Classification\n\n"
     "Each document multiple labels\n"
     "Example: Movie has comedy and drama\n"
     "Loss: Binary cross-entropy per label\n"
     "Threshold per label\n"
     "More complex than single-label\n\n",
     "Add multi-label"),
    
    ("### Hierarchical Classification\n\n"
     "Labels form hierarchy\n"
     "Coarse and fine categories\n"
     "Leverage hierarchy\n"
     "Constrain predictions\n"
     "Better with structure\n\n",
     "Add hierarchical"),
    
    # PR 14: Information Extraction (5 commits)
    ("## Information Extraction\n\n"
     "### Relation Extraction\n\n"
     "Extract relationships between entities\n"
     "Example: Company founded by person\n"
     "Slot filling\n"
     "Knowledge base construction\n"
     "Can be supervised or distant\n\n",
     "Add relation extraction"),
    
    ("### Event Extraction\n\n"
     "Extract event mentions\n"
     "Participants (who, what, where)\n"
     "Temporal information (when)\n"
     "Event type and subtypes\n"
     "Complex task\n\n",
     "Add event extraction"),
    
    ("### Coreference Resolution\n\n"
     "Link mentions to same entity\n"
     "Pronouns: He, she, it\n"
     "Noun phrases: The company\n"
     "Challenge: Ambiguity\n"
     "Graph-based and span-based methods\n\n",
     "Add coreference"),
    
    ("### Dependency Parsing\n\n"
     "Extract grammatical structure\n"
     "Subject, object, modifiers\n"
     "Arc classification\n"
     "Dependency labels\n"
     "Useful for downstream tasks\n\n",
     "Add dependency parsing"),
    
    ("### Semantic Role Labeling\n\n"
     "Identify semantic roles\n"
     "Agent, patient, location\n"
     "Per predicate\n"
     "Proposition banks\n"
     "Improved by contextualized embeddings\n\n",
     "Add semantic roles"),
]

readme_path = 'README.md'

print(f"Starting module-05 phase 2 with {len(sections)} commits...")
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
print(f"[DONE] Phase 2 added {len(sections)} commits!")

result = subprocess.run(['git', 'log', '--oneline'], capture_output=True, text=True)
total = len(result.stdout.strip().split('\n'))
print(f"Total repository commits: {total}")
